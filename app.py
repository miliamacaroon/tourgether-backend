
# ===============================
# IMPROVED APP.PY
# ===============================

import streamlit as st
import os
import pandas as pd
from datetime import date
from vision import detect_attraction, load_model
from llm_rag import graph
from langchain_core.messages import HumanMessage
from pdf_utils import generate_itinerary_pdf
from region_mapper import RegionMapper  # NEW IMPORT

# ===============================
# PAGE CONFIG & STATE
# ===============================
st.set_page_config(page_title="TourGether", page_icon="🌍", layout="wide")

# Initialize Session State - ALL KEYS MUST BE INITIALIZED
if "trip_data" not in st.session_state:
    st.session_state.trip_data = {
        "itinerary": "",
        "destination": "",
        "days": 0,
        "budget": "",
        "trip_type": "",
        "attractions_df": None,
        "detected_region": None,
        "vision_confidence": 0.0
    }

if "trip_type_key" not in st.session_state:
    st.session_state.trip_type_key = "landmarks"

if "destination_suggestions" not in st.session_state:
    st.session_state.destination_suggestions = []

if "show_vision_insights" not in st.session_state:
    st.session_state.show_vision_insights = False

# NEW: Initialize these separately for easier access
if "detected_region" not in st.session_state:
    st.session_state.detected_region = None

if "vision_confidence" not in st.session_state:
    st.session_state.vision_confidence = 0.0

if "auto_destination" not in st.session_state:
    st.session_state.auto_destination = ""

# ===============================
# LOAD VISION MODEL
# ===============================
PROJECT_ROOT = "/content/drive/MyDrive/Colab Notebooks/CAIE Project (TourGether)"
MODEL_PATH = os.path.join(PROJECT_ROOT, "runs/tourgether_region_final/weights/best.pt")

@st.cache_resource(show_spinner=False)
def load_vision_model():
    model = load_model(MODEL_PATH)
    model.to('cpu')  # Explicitly use CPU
    model.fuse()  # Optimize for inference
    return model

vision_model = load_vision_model()

# ===============================
# UI HEADER
# ===============================
st.title("🌍 TourGether")
st.markdown("Plan your trip using AI — Upload a photo or describe your dream destination")
st.divider()

# ===============================
# OPTIONAL IMAGE UPLOAD (BEFORE FORM)
# ===============================
st.subheader("📸 Smart Trip Planning (Optional)")
st.markdown("Upload a photo of a place that inspires you, and we'll suggest destinations and preferences!")

uploaded_image = st.file_uploader(
    "Upload Attraction Image", 
    type=["jpg","png","jpeg"],
    help="Optional: Upload a photo to auto-detect region and get personalized suggestions",
    key="image_uploader"
)

# Vision Processing (runs immediately on upload)
if uploaded_image and not st.session_state.show_vision_insights:
    with st.spinner("🔍 Analyzing your image..."):
        # Save temp image
        with open("temp.jpg", "wb") as f:
            f.write(uploaded_image.getvalue())
        
        # Detect region
        detected_region, confidence = detect_attraction("temp.jpg", vision_model)
        
        # Store in session state
        st.session_state.detected_region = detected_region
        st.session_state.vision_confidence = confidence
        
        # Get mapped trip type
        suggested_trip_type = RegionMapper.get_trip_type(detected_region, confidence)
        st.session_state.trip_type_key = suggested_trip_type
        
        # Get destination suggestions
        st.session_state.destination_suggestions = RegionMapper.get_destination_suggestions(detected_region)
        
        st.session_state.show_vision_insights = True
        st.rerun()

# Display Vision Insights
if st.session_state.show_vision_insights and st.session_state.detected_region:
    col_img, col_insights = st.columns([1, 2])
    
    with col_img:
        st.image(uploaded_image, caption="Your Uploaded Image", use_container_width=True)
    
    with col_insights:
        region = st.session_state.detected_region
        conf = st.session_state.vision_confidence
        
        st.success(f"✨ **Detected Region:** {region.replace('_', ' ').title()}")
        st.info(f"🎯 **Confidence:** {conf*100:.1f}%")
        
        # Get comprehensive region info
        region_info = RegionMapper.get_region_info(region)
        
        # Show suggestions based on confidence
        if conf >= 0.6:
            st.markdown(f"**💡 Recommended Focus:** {st.session_state.trip_type_key.replace('_', ' ').title()}")
            
            # Show season info
            if region_info['season_info']:
                st.caption(f"🌤️ Best time: {', '.join(region_info['season_info'])}")
            
            # Show ALL destinations in expandable section
            if st.session_state.destination_suggestions:
                st.markdown(f"**📍 Popular Destinations in {region.replace('_', ' ').title()}:**")
                
                # Get all destinations
                all_destinations = RegionMapper.get_all_destinations_for_region(region)
                
                # Show first 4 as buttons
                dest_cols = st.columns(4)
                for i, dest in enumerate(all_destinations[:4]):
                    if dest_cols[i % 4].button(dest, key=f"dest_{i}", use_container_width=True):
                        st.session_state.auto_destination = dest
                        st.rerun()
                
                # Show remaining as expander
                if len(all_destinations) > 4:
                    with st.expander(f"➕ See {len(all_destinations) - 4} more destinations"):
                        more_cols = st.columns(4)
                        for i, dest in enumerate(all_destinations[4:], start=4):
                            col_idx = (i - 4) % 4
                            if more_cols[col_idx].button(dest, key=f"dest_{i}", use_container_width=True):
                                st.session_state.auto_destination = dest
                                st.rerun()
        else:
            st.warning("⚠️ Low confidence detection. Manual selection recommended.")
            st.markdown("**Possible destinations:**")
            all_dests = RegionMapper.get_all_destinations_for_region(region)
            if all_dests:
                st.caption(f"{', '.join(all_dests[:6])}...")
        
        if st.button("🔄 Clear Image & Start Over", key="clear_image"):
            st.session_state.show_vision_insights = False
            st.session_state.detected_region = None
            st.session_state.destination_suggestions = []
            st.session_state.vision_confidence = 0.0
            st.session_state.auto_destination = ""
            st.rerun()

st.divider()

# ===============================
# TRIP FORM
# ===============================
st.subheader("🗺️ Trip Details")

with st.form("trip_form"):
    col1, col2, col3 = st.columns(3)
    
    # Auto-fill destination if suggested
    destination_input = col1.text_input(
        "📍 Destination", 
        value=st.session_state.auto_destination,
        placeholder="e.g. Tokyo, Paris, Bali"
    )
    
    start_date = col2.date_input("📅 Start Date", min_value=date.today())
    end_date = col3.date_input("📅 End Date", value=date.today())

    col4, col5, col6 = st.columns(3)
    
    # Comprehensive currency list organized by popularity
    currency_options = [
        "USD - US Dollar",
        "EUR - Euro", 
        "GBP - British Pound",
        "JPY - Japanese Yen",
        "AUD - Australian Dollar",
        "CAD - Canadian Dollar",
        "SGD - Singapore Dollar",
        "MYR - Malaysian Ringgit",
        "THB - Thai Baht",
        "CNY - Chinese Yuan",
        "KRW - South Korean Won",
        "HKD - Hong Kong Dollar",
        "NZD - New Zealand Dollar",
        "CHF - Swiss Franc",
        "AED - UAE Dirham",
        "ZAR - South African Rand",
        "INR - Indian Rupee",
        "BRL - Brazilian Real",
        "MXN - Mexican Peso",
        "IDR - Indonesian Rupiah",
        "PHP - Philippine Peso",
        "VND - Vietnamese Dong"
    ]
    
    # Smart default based on detected region
    default_currency = "USD - US Dollar"
    if st.session_state.detected_region:
        region_info = RegionMapper.get_region_info(st.session_state.detected_region)
        currency_hint = region_info['budget_info']['currency']
        
        # Map currency codes to full format
        currency_map = {
            'USD': 'USD - US Dollar',
            'EUR': 'EUR - Euro',
            'JPY': 'JPY - Japanese Yen',
            'AUD': 'AUD - Australian Dollar',
            'GBP': 'GBP - British Pound',
            'MYR': 'MYR - Malaysian Ringgit',
            'SGD': 'SGD - Singapore Dollar',
            'CNY': 'CNY - Chinese Yuan'
        }
        default_currency = currency_map.get(currency_hint, default_currency)
    
    # Find default index
    default_idx = 0
    if default_currency in currency_options:
        default_idx = currency_options.index(default_currency)
    
    selected_currency = col4.selectbox(
        "💱 Currency", 
        options=currency_options,
        index=default_idx,
        help="Budget will be displayed in this currency"
    )
    
    # Extract currency code (e.g., "USD - US Dollar" → "USD")
    currency = selected_currency.split(" - ")[0]
    
    # Show budget suggestion but DON'T auto-adjust
    base_budget = (2000, 6000)
    budget_range = col5.slider("💰 Budget", 100, 20000, base_budget)
    
    # Show suggestion AFTER user sees the slider
    if st.session_state.detected_region:
        suggested_budget = RegionMapper.adjust_budget(st.session_state.detected_region, base_budget)
        region_name = st.session_state.detected_region.replace('_', ' ').title()
        modifier = RegionMapper.REGION_TO_PREFERENCES[st.session_state.detected_region]['budget_modifier']
        
        if modifier > 1.1:
            st.info(f"💡 Trips in **{region_name}** typically cost more. Suggested: {currency} {suggested_budget[0]:,}–{suggested_budget[1]:,}")
        elif modifier < 0.95:
            st.success(f"💰 Great news! **{region_name}** is budget-friendly. Suggested: {currency} {suggested_budget[0]:,}–{suggested_budget[1]:,}")

    trip_type_options = ["landmarks", "historical_places", "nature", "entertainment"]
    
    # Auto-select based on vision
    default_idx = 0
    if st.session_state.trip_type_key in trip_type_options:
        default_idx = trip_type_options.index(st.session_state.trip_type_key)
    
    trip_type_input = col6.selectbox(
        "🎭 Trip Focus", 
        options=trip_type_options,
        index=default_idx,
        format_func=lambda x: x.replace('_', ' ').title()
    )

    # Additional preferences
    st.markdown("**🎨 Additional Preferences (Optional)**")
    col7, col8 = st.columns(2)
    
    pace = col7.selectbox("⏱️ Trip Pace", ["Relaxed", "Moderate", "Fast-paced"])
    dining_pref = col8.selectbox("🍽️ Dining Style", ["Local cuisine", "Mix of local & international", "Fine dining"])
    
    # SUBMIT BUTTON - MUST BE INSIDE THE FORM
    generate_btn = st.form_submit_button("🚀 Generate Itinerary", use_container_width=True, type="primary")

# ===============================
# GENERATION LOGIC
# ===============================
if generate_btn:
    if not destination_input:
        st.error("⚠️ Please enter a destination")
    elif end_date < start_date:
        st.error("⚠️ End date cannot be before start date")
    else:
        days_count = (end_date - start_date).days or 1
        budget_str = f"{currency} {budget_range[0]:,} - {budget_range[1]:,}"

        # Build query with user inputs (region only for enrichment, not constraint)
        query = f"{days_count} day trip to {destination_input} focusing on {trip_type_input}"
        query += f" with budget {budget_str}. Pace: {pace}. Dining: {dining_pref}."
        
        # Add region as soft context (for narrative tone only, not filtering)
        region_filter = None
        if st.session_state.detected_region:
            region_name = st.session_state.detected_region.replace('_', ' ').title()
            # Light touch - just for LLM tone/style
            query += f" (Traveler interested in {region_name} destinations)"
            # Store region for potential retrieval filtering
            region_filter = st.session_state.detected_region
        
        inputs = {
            "messages": [HumanMessage(content=query)], 
            "query": query,
            "region_filter": region_filter  # Pass to retrieval layer
        }

        with st.spinner("🤖 AI is crafting your perfect itinerary..."):
            for output in graph.stream(inputs):
                for node, state in output.items():
                    if node == "generate":
                        pics = []
                        if "documents" in state:
                            pics = [{"PICTURE": d.metadata.get("PICTURE"), 
                                   "NAME": d.metadata.get("NAME")} 
                                  for d in state["documents"] if d.metadata.get("PICTURE")]

                        attr_df = pd.DataFrame(pics) if pics else pd.DataFrame(columns=["PICTURE", "NAME"])

                        st.session_state.trip_data = {
                            "itinerary": state["messages"][-1].content,
                            "destination": destination_input,
                            "days": days_count,
                            "budget": budget_str,
                            "trip_type": trip_type_input,
                            "attractions_df": attr_df,
                            "detected_region": st.session_state.detected_region,
                            "vision_confidence": st.session_state.vision_confidence
                        }

# ===============================
# DISPLAY & PDF EXPORT
# ===============================
if st.session_state.trip_data["itinerary"]:
    data = st.session_state.trip_data

    st.divider()
    st.subheader(f"🗺️ Your Personalized Itinerary for {data['destination']}")
    
    # Show metadata
    meta_cols = st.columns(4)
    meta_cols[0].metric("Duration", f"{data['days']} days")
    meta_cols[1].metric("Budget", data['budget'])
    meta_cols[2].metric("Focus", data['trip_type'].replace('_', ' ').title())
    if data.get('detected_region'):
        meta_cols[3].metric("Region", data['detected_region'].replace('_', ' ').title())
    
    st.markdown("---")
    st.markdown(data["itinerary"])

    # Show attraction gallery
    if not data["attractions_df"].empty and "PICTURE" in data["attractions_df"].columns:
        st.divider()
        st.subheader("📸 Featured Attractions")
        
        # Display up to 6 attractions
        display_df = data["attractions_df"].head(6)
        cols = st.columns(min(3, len(display_df)))
        
        for idx, row in display_df.iterrows():
            col_idx = idx % 3
            with cols[col_idx]:
                if pd.notna(row.get("PICTURE")):
                    st.image(row["PICTURE"], caption=row.get("NAME", "Attraction"), use_container_width=True)

    # PDF Section
    st.divider()
    col_a, col_b = st.columns([4, 1])
    with col_b:
        if st.button("📄 Generate PDF", use_container_width=True):
            pdf_dir = os.path.join(PROJECT_ROOT, "PDF")
            os.makedirs(pdf_dir, exist_ok=True)
            pdf_filename = f"{data['destination'].replace(' ', '_')}_itinerary.pdf"
            pdf_path = os.path.join(pdf_dir, pdf_filename)

            try:
                generate_itinerary_pdf(
                    itinerary_text=data["itinerary"],
                    city=data["destination"],
                    days=data["days"],
                    budget=data["budget"],
                    trip_type=data["trip_type"],
                    attractions_df=data["attractions_df"],
                    output_path=pdf_path
                )
                st.success("✅ PDF Generated!")

                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download PDF",
                        data=f,
                        file_name=pdf_filename,
                        mime="application/pdf",
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"❌ Error generating PDF: {e}")

st.divider()
st.markdown("<p style='text-align:center; color:gray;'>© 2025 TourGether • Powered by AI Vision & RAG</p>", unsafe_allow_html=True)
