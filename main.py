"""
TourGether FastAPI Backend
Complete API for Lovable frontend integration
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List
import os
import shutil
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
from io import BytesIO

# Import your existing modules
from vision import detect_attraction, load_model
from llm_rag import graph
from region_mapper import RegionMapper
from pdf_utils import PDFGenerator
from langchain_core.messages import HumanMessage

# ===============================
# INITIALIZATION
# ===============================
load_dotenv()

app = FastAPI(
    title="TourGether API",
    description="AI-powered travel itinerary planning with vision detection",
    version="1.0.0"
)

# CORS Configuration - Your Lovable Project
ALLOWED_ORIGINS = [
    "https://lovable.dev",  # Lovable development domain
    "https://*.lovable.app",  # Published apps
    "https://*.lovable.dev",  # All Lovable dev domains
    "http://localhost:5173",  # Local testing
    "http://localhost:3000"   # Alternative local port
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load vision model at startup
MODEL_PATH = os.getenv("MODEL_PATH", "models/best.pt")
vision_model = None

@app.on_event("startup")
async def startup_event():
    """Load ML models on startup"""
    global vision_model
    try:
        vision_model = load_model(MODEL_PATH)
        print(f"✅ Vision model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"⚠️ Warning: Could not load vision model: {e}")

# ===============================
# PYDANTIC MODELS
# ===============================
class RegionDetectionResponse(BaseModel):
    region: str
    confidence: float
    trip_type: str
    destinations: List[str]
    all_destinations: List[str]
    budget_modifier: float
    currency_hint: str
    seasons: List[str]

class ItineraryRequest(BaseModel):
    destination: str
    days: int
    budget_min: int
    budget_max: int
    currency: str
    trip_type: str
    pace: str = "Moderate"
    dining: str = "Mix of local & international"
    region: Optional[str] = None

class AttractionData(BaseModel):
    picture: Optional[str]
    name: Optional[str]

class ItineraryResponse(BaseModel):
    itinerary: str
    attractions: List[AttractionData]
    metadata: dict

# ===============================
# ENDPOINTS
# ===============================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "TourGether API",
        "version": "1.0.0",
        "endpoints": {
            "detect_region": "/api/detect-region (POST)",
            "generate_itinerary": "/api/generate-itinerary (POST)",
            "generate_pdf": "/api/generate-pdf (POST)"
        }
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "vision_model": vision_model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/detect-region", response_model=RegionDetectionResponse)
async def detect_region_endpoint(image: UploadFile = File(...)):
    """
    Detect region from uploaded image using YOLOv8
    
    Returns:
        - region: Detected region name
        - confidence: Detection confidence (0-1)
        - trip_type: Suggested trip focus
        - destinations: Top 8 suggested destinations
        - all_destinations: All destinations for the region
        - budget_modifier: Cost of living multiplier
        - currency_hint: Recommended currency
        - seasons: Best times to visit
    """
    if not vision_model:
        raise HTTPException(status_code=503, detail="Vision model not available")
    
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{datetime.now().timestamp()}_{image.filename}"
        
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # Detect region
        region, confidence = detect_attraction(temp_path, vision_model)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Get region information
        region_info = RegionMapper.get_region_info(region)
        trip_type = RegionMapper.get_trip_type(region, confidence)
        destinations = RegionMapper.get_destination_suggestions(region, limit=8)
        all_destinations = RegionMapper.get_all_destinations_for_region(region)
        
        return RegionDetectionResponse(
            region=region,
            confidence=confidence,
            trip_type=trip_type,
            destinations=destinations,
            all_destinations=all_destinations,
            budget_modifier=region_info['budget_info']['modifier'],
            currency_hint=region_info['budget_info']['currency'],
            seasons=region_info['season_info']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/api/generate-itinerary", response_model=ItineraryResponse)
async def generate_itinerary_endpoint(request: ItineraryRequest):
    """
    Generate AI-powered travel itinerary using RAG
    
    Input:
        - destination: City/country name
        - days: Number of days (1-30)
        - budget_min/max: Budget range
        - currency: Currency code (USD, EUR, etc.)
        - trip_type: Focus (landmarks, nature, etc.)
        - pace: Trip pace (Relaxed/Moderate/Fast-paced)
        - dining: Dining preference
        - region: Optional detected region
    
    Returns:
        - itinerary: Full markdown itinerary text
        - attractions: List of featured attractions with images
        - metadata: Trip metadata
    """
    try:
        # Build budget string
        budget_str = f"{request.currency} {request.budget_min:,} - {request.budget_max:,}"
        
        # Build query for LLM
        query = f"{request.days} day trip to {request.destination} focusing on {request.trip_type}"
        query += f" with budget {budget_str}. Pace: {request.pace}. Dining: {request.dining}."
        
        # Add region context if provided
        if request.region:
            region_name = request.region.replace('_', ' ').title()
            query += f" (Traveler interested in {region_name} destinations)"
        
        # Prepare graph input
        inputs = {
            "messages": [HumanMessage(content=query)], 
            "query": query,
            "region_filter": request.region
        }
        
        # Generate itinerary
        itinerary_text = ""
        attractions_data = []
        
        for output in graph.stream(inputs):
            for node, state in output.items():
                if node == "generate":
                    itinerary_text = state["messages"][-1].content
                    
                    # Extract attraction metadata
                    if "documents" in state:
                        attractions_data = [
                            AttractionData(
                                picture=d.metadata.get("PICTURE"),
                                name=d.metadata.get("NAME")
                            )
                            for d in state["documents"] 
                            if d.metadata.get("PICTURE")
                        ][:6]  # Limit to 6 attractions
        
        return ItineraryResponse(
            itinerary=itinerary_text,
            attractions=attractions_data,
            metadata={
                "destination": request.destination,
                "days": request.days,
                "budget": budget_str,
                "trip_type": request.trip_type,
                "region": request.region,
                "generated_at": datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Itinerary generation failed: {str(e)}")

@app.post("/api/generate-pdf")
async def generate_pdf_endpoint(
    itinerary: str = Form(...),
    destination: str = Form(...),
    days: int = Form(...),
    budget: str = Form(...),
    trip_type: str = Form(...),
    region: Optional[str] = Form(None),
    attractions: Optional[str] = Form(None)  # JSON string of attractions
):
    """
    Generate downloadable PDF from itinerary
    
    Input:
        - itinerary: Full itinerary markdown text
        - destination: City name
        - days: Number of days
        - budget: Budget string (e.g., "USD 2,000 - 6,000")
        - trip_type: Trip focus
        - region: Optional region name
        - attractions: Optional JSON array of {picture, name}
    
    Returns:
        PDF file download
    """
    try:
        # Parse attractions if provided
        attractions_df = None
        if attractions:
            import json
            attractions_list = json.loads(attractions)
            if attractions_list:
                attractions_df = pd.DataFrame(attractions_list)
                # Rename columns to match expected format
                if 'picture' in attractions_df.columns:
                    attractions_df.rename(columns={'picture': 'PICTURE', 'name': 'NAME'}, inplace=True)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"{destination.replace(' ', '_')}_{timestamp}.pdf"
        pdf_path = f"temp_{pdf_filename}"
        
        # Generate PDF
        generator = PDFGenerator()
        success = generator.generate_pdf(
            itinerary_text=itinerary,
            city=destination,
            days=days,
            budget=budget,
            trip_type=trip_type,
            attractions_df=attractions_df,
            output_path=pdf_path,
            region=region
        )
        
        if not success or not os.path.exists(pdf_path):
            raise Exception("PDF generation failed")
        
        # Return file and schedule cleanup
        response = FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=pdf_filename,
            background=None  # Cleanup handled by Railway
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")

@app.get("/api/regions")
async def get_all_regions():
    """Get all available regions and their metadata"""
    regions = {}
    for region in RegionMapper.REGION_TO_PREFERENCES.keys():
        regions[region] = RegionMapper.get_region_info(region)
    return regions

@app.get("/api/destinations/{region}")
async def get_destinations_by_region(region: str):
    """Get all destinations for a specific region"""
    destinations = RegionMapper.get_all_destinations_for_region(region)
    if not destinations:
        raise HTTPException(status_code=404, detail=f"Region '{region}' not found")
    return {"region": region, "destinations": destinations}

# ===============================
# ERROR HANDLERS
# ===============================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

# ===============================
# MAIN
# ===============================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True  # Set to False in production
    )