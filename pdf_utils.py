
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle, KeepTogether
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas
import requests
from io import BytesIO
import re
from datetime import datetime
from functools import lru_cache
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFGenerator:
    """Production-ready PDF generator with caching and error handling"""
    
    def __init__(self):
        self.styles = self._get_pdf_styles()
        self.image_cache = {}
        
    def _get_pdf_styles(self):
        """Enhanced styles with better typography"""
        styles = getSampleStyleSheet()
        
        # Custom color palette
        primary_color = colors.HexColor("#1A5490")      # Deep blue
        secondary_color = colors.HexColor("#2ECC71")    # Green
        accent_color = colors.HexColor("#E67E22")       # Orange
        dark_gray = colors.HexColor("#2C3E50")
        light_gray = colors.HexColor("#ECF0F1")
        
        # Title style
        styles.add(ParagraphStyle(
            name="TG_Title",
            fontSize=32,
            leading=38,
            spaceAfter=12,
            alignment=TA_CENTER,
            textColor=primary_color,
            fontName="Helvetica-Bold"
        ))
        
        # Subtitle
        styles.add(ParagraphStyle(
            name="TG_Subtitle",
            fontSize=18,
            leading=22,
            spaceAfter=8,
            alignment=TA_CENTER,
            textColor=dark_gray,
            fontName="Helvetica"
        ))
        
        # Section headers
        styles.add(ParagraphStyle(
            name="TG_SectionHeader",
            fontSize=16,
            leading=20,
            spaceBefore=12,
            spaceAfter=8,
            textColor=primary_color,
            fontName="Helvetica-Bold",
            borderWidth=0,
            borderPadding=0,
            borderColor=primary_color,
            borderRadius=None,
            leftIndent=0
        ))
        
        # Day headers with background
        styles.add(ParagraphStyle(
            name="TG_DayHeader",
            fontSize=14,
            leading=18,
            spaceBefore=16,
            spaceAfter=10,
            textColor=colors.white,
            backColor=primary_color,
            fontName="Helvetica-Bold",
            leftIndent=8,
            rightIndent=8,
            borderPadding=8,
            alignment=TA_LEFT
        ))
        
        # Activity headers
        styles.add(ParagraphStyle(
            name="TG_Activity",
            fontSize=11,
            leading=14,
            spaceBefore=6,
            spaceAfter=4,
            textColor=accent_color,
            fontName="Helvetica-Bold",
            leftIndent=20
        ))
        
        # Body text
        styles.add(ParagraphStyle(
            name="TG_Body",
            fontSize=10,
            leading=14,
            spaceAfter=6,
            leftIndent=25,
            alignment=TA_JUSTIFY,
            textColor=dark_gray
        ))
        
        # Time markers
        styles.add(ParagraphStyle(
            name="TG_Time",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#7F8C8D"),
            fontName="Helvetica-Oblique",
            leftIndent=25
        ))
        
        # Metadata
        styles.add(ParagraphStyle(
            name="TG_Meta",
            fontSize=10,
            leading=13,
            spaceAfter=4,
            textColor=dark_gray,
            alignment=TA_LEFT
        ))
        
        # Footer
        styles.add(ParagraphStyle(
            name="TG_Footer",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#95A5A6"),
            alignment=TA_CENTER
        ))
        
        return styles
    
    @lru_cache(maxsize=50)
    def _load_image_from_url(self, url, width=4*inch, max_height=3*inch):
        """
        Load and cache images with proper error handling
        Returns: Image object or None
        """
        if not url or url == "":
            return None
            
        try:
            # Check cache first
            if url in self.image_cache:
                logger.info(f"Using cached image: {url[:50]}...")
                return self.image_cache[url]
            
            logger.info(f"Downloading image: {url[:50]}...")
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'TourGether-PDF-Generator/1.0'
            })
            
            if response.status_code == 200:
                img_data = BytesIO(response.content)
                img = Image(img_data)
                
                # Smart resizing - maintain aspect ratio
                aspect = img.imageWidth / img.imageHeight
                
                if aspect > (width / max_height):
                    # Width is limiting factor
                    img.drawWidth = width
                    img.drawHeight = width / aspect
                else:
                    # Height is limiting factor
                    img.drawHeight = max_height
                    img.drawWidth = max_height * aspect
                
                # Cache the image
                self.image_cache[url] = img
                return img
            else:
                logger.warning(f"Failed to load image: {url}, status: {response.status_code}")
                return None
                
        except requests.Timeout:
            logger.error(f"Timeout loading image: {url}")
            return None
        except Exception as e:
            logger.error(f"Error loading image {url}: {str(e)}")
            return None
    
    def _clean_markdown(self, text):
        """Enhanced markdown cleaning with safety checks"""
        if not text:
            return ""
        
        # Convert markdown to ReportLab tags
        text = re.sub(r'\*\*\*(.*?)\*\*\*', r'<b><i>\1</i></b>', text)
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
        text = re.sub(r'__(.*?)__', r'<u>\1</u>', text)
        
        # Clean up special characters that might break PDF
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;').replace('>', '&gt;')
        
        # Restore our formatting tags
        text = text.replace('&lt;b&gt;', '<b>').replace('&lt;/b&gt;', '</b>')
        text = text.replace('&lt;i&gt;', '<i>').replace('&lt;/i&gt;', '</i>')
        text = text.replace('&lt;u&gt;', '<u>').replace('&lt;/u&gt;', '</u>')
        
        return text
    
    def _build_cover_page(self, city, days, budget, trip_type, region=None):
        """Create an attractive cover page"""
        story = []
        
        story.append(Spacer(1, 1.5*inch))
        
        # Main title
        story.append(Paragraph("🌍 TourGether", self.styles["TG_Title"]))
        story.append(Spacer(1, 0.2*inch))
        
        # Destination
        story.append(Paragraph(f"{city.upper()}", self.styles["TG_Subtitle"]))
        story.append(Paragraph(
            f"{days}-Day Travel Itinerary", 
            self.styles["TG_Meta"]
        ))
        
        story.append(Spacer(1, 0.5*inch))
        
        # Metadata box
        meta_data = [
            ["Trip Details", ""],
            ["📅 Duration", f"{days} days"],
            ["💰 Budget", budget],
            ["🎯 Focus", trip_type.replace('_', ' ').title()],
            ["📍 Region", region.replace('_', ' ').title() if region else "Global"],
            ["📆 Generated", datetime.now().strftime("%B %d, %Y")]
        ]
        
        meta_table = Table(meta_data, colWidths=[2*inch, 3.5*inch])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1A5490")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor("#BDC3C7")),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        story.append(meta_table)
        story.append(Spacer(1, 0.5*inch))
        
        # Disclaimer
        disclaimer = Paragraph(
            "<i>This itinerary was AI-generated based on your preferences. "
            "Please verify opening hours, prices, and availability before your trip.</i>",
            self.styles["TG_Footer"]
        )
        story.append(disclaimer)
        
        story.append(PageBreak())
        return story
    
    def _build_budget_summary(self, budget_text):
        """Enhanced budget breakdown with visual appeal"""
        data = [
            ["Expense Category", "Allocation", "Notes"],
            ["🎭 Activities & Attractions", "40%", "Entry fees, tours, experiences"],
            ["🍽️ Dining & Food", "35%", "Meals, snacks, beverages"],
            ["🚗 Transport & Misc", "25%", "Local transport, tips, souvenirs"],
            ["💰 Total Budget Range", budget_text, "Flexible based on choices"]
        ]
        
        table = Table(data, colWidths=[2.2*inch, 1.3*inch, 2*inch])
        table.setStyle(TableStyle([
            # Header row
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1A5490")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            
            # Data rows
            ('BACKGROUND', (0, 1), (-1, -2), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor("#2C3E50")),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ALIGN', (1, 1), (1, -1), 'CENTER'),
            
            # Total row
            ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor("#ECF0F1")),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, -1), (-1, -1), 10),
            
            # All rows
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ]))
        
        return table
    
    def _parse_itinerary_content(self, itinerary_text, attractions_df):
        """
        Smart parsing of itinerary with image placement
        Returns: story elements list
        """
        story = []
        lines = itinerary_text.split("\n")
        day_count = 0
        current_day_elements = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and decorative separators
            if not line or re.match(r'^[_\-=\s*·]+$', line):
                continue
            
            # Skip budget lines (we handle separately)
            if "budget:" in line.lower() and "summary" not in line.lower():
                continue
            
            formatted_line = self._clean_markdown(line)
            
            # DAY HEADERS
            if re.match(r'^#{1,3}\s*day\s+\d+', line.lower()) or line.lower().startswith("day"):
                # Add previous day's elements as KeepTogether
                if current_day_elements:
                    story.extend(current_day_elements)
                    current_day_elements = []
                
                # New day section
                story.append(Spacer(1, 15))
                day_title = formatted_line.replace("#", "").strip()
                story.append(Paragraph(day_title, self.styles["TG_DayHeader"]))
                
                # Add day image if available
                if attractions_df is not None and not attractions_df.empty:
                    img = self._load_image_from_url(
                        attractions_df.iloc[day_count % len(attractions_df)].get("PICTURE")
                    )
                    if img:
                        story.append(Spacer(1, 10))
                        story.append(img)
                        
                        # Add caption if name available
                        img_name = attractions_df.iloc[day_count % len(attractions_df)].get("NAME")
                        if img_name:
                            caption = Paragraph(
                                f"<i>{img_name}</i>", 
                                self.styles["TG_Footer"]
                            )
                            story.append(caption)
                
                day_count += 1
                story.append(Spacer(1, 10))
            
            # SECTION HEADERS (Morning, Afternoon, Evening)
            elif any(x in line.lower() for x in ["morning:", "afternoon:", "evening:", "night:"]):
                lower_line = line.lower()
                icon = ("🌅 " if "morning" in lower_line else 
                       "☀️ " if "afternoon" in lower_line else 
                       "🌆 " if "evening" in lower_line else "🌙 ")
                clean_text = formatted_line.replace("■", "").replace(":", "").strip()
                current_day_elements.append(
                    Paragraph(icon + clean_text, self.styles["TG_SectionHeader"])
                )
            
            # TIME MARKERS
            elif re.match(r'^\d{1,2}:\d{2}', line):
                current_day_elements.append(
                    Paragraph(f"⏰ {formatted_line}", self.styles["TG_Time"])
                )
            
            # ACTIVITIES
            elif line.startswith("•") or line.startswith("■") or line.startswith("-"):
                clean_act = formatted_line.lstrip("•■- ").strip()
                current_day_elements.append(
                    Paragraph(f"📍 {clean_act}", self.styles["TG_Activity"])
                )
            
            # NORMAL TEXT
            else:
                if "budget estimation summary" in line.lower():
                    break  # Stop processing text here
                current_day_elements.append(
                    Paragraph(formatted_line, self.styles["TG_Body"])
                )
        
        # Add remaining elements
        if current_day_elements:
            story.extend(current_day_elements)
        
        return story
    
    def generate_pdf(self, itinerary_text, city, days, budget, trip_type, 
                    attractions_df=None, output_path="itinerary.pdf", region=None):
        """
        Main PDF generation function with comprehensive error handling
        
        Args:
            itinerary_text: AI-generated itinerary content
            city: Destination city
            days: Number of days
            budget: Budget range string
            trip_type: Type of trip
            attractions_df: DataFrame with PICTURE and NAME columns
            output_path: Output file path
            region: Detected region (optional)
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Starting PDF generation for {city}")
            
            # Create document
            doc = SimpleDocTemplate(
                output_path,
                pagesize=A4,
                rightMargin=40,
                leftMargin=40,
                topMargin=40,
                bottomMargin=40,
                title=f"{city} Travel Itinerary - TourGether",
                author="TourGether AI"
            )
            
            story = []
            
            # 1. COVER PAGE
            story.extend(self._build_cover_page(city, days, budget, trip_type, region))
            
            # 2. ITINERARY CONTENT
            story.append(Paragraph("📋 Your Itinerary", self.styles["TG_SectionHeader"]))
            story.append(Spacer(1, 10))
            
            content_elements = self._parse_itinerary_content(itinerary_text, attractions_df)
            story.extend(content_elements)
            
            # 3. BUDGET SUMMARY
            story.append(Spacer(1, 30))
            story.append(Paragraph("💰 Budget Breakdown", self.styles["TG_SectionHeader"]))
            story.append(Spacer(1, 10))
            story.append(self._build_budget_summary(budget))
            
            # 4. FOOTER / TIPS
            story.append(Spacer(1, 20))
            tips = Paragraph(
                "<b>💡 Travel Tips:</b><br/>"
                "• Book accommodations and major attractions in advance<br/>"
                "• Keep digital and physical copies of important documents<br/>"
                "• Check visa requirements and travel advisories<br/>"
                "• Consider travel insurance for peace of mind<br/>"
                "• Download offline maps and translation apps",
                self.styles["TG_Body"]
            )
            story.append(tips)
            
            # Build PDF
            doc.build(story)
            logger.info(f"PDF successfully generated: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating PDF: {str(e)}")
            raise


# ===============================
# LEGACY FUNCTION (for backward compatibility)
# ===============================
def generate_itinerary_pdf(itinerary_text, city, days, budget, trip_type, 
                          attractions_df=None, output_path="itinerary.pdf"):
    """
    Legacy wrapper function for backward compatibility
    """
    generator = PDFGenerator()
    return generator.generate_pdf(
        itinerary_text=itinerary_text,
        city=city,
        days=days,
        budget=budget,
        trip_type=trip_type,
        attractions_df=attractions_df,
        output_path=output_path
    )
