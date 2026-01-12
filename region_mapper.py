
# ===============================
# REGION MAPPER (NEW FILE)
# Save as: region_mapper.py
# ===============================

class RegionMapper:
    """
    Maps vision-detected regions to trip preferences and destinations.
    
    ARCHITECTURE DECISION:
    - Region filtering happens at RETRIEVAL layer (FAISS/BM25)
    - NOT in LLM prompt (that would constrain creativity)
    - UI shows suggestions but lets user decide
    - Budget adjustment is suggested, not forced
    """
    
class RegionMapper:
    """
    Maps vision-detected regions to trip preferences and destinations.
    
    ARCHITECTURE DECISION:
    - Region filtering happens at RETRIEVAL layer (FAISS/BM25)
    - NOT in LLM prompt (that would constrain creativity)
    - UI shows suggestions but lets user decide
    - Budget adjustment is suggested, not forced
    """
    
    REGION_TO_PREFERENCES = {
        'north_america': {
            'primary_type': 'landmarks',
            'secondary_types': ['entertainment', 'nature'],
            'suggested_destinations': [
                'New York', 'Los Angeles', 'San Francisco', 'Chicago',
                'Las Vegas', 'Miami', 'Seattle', 'Boston',
                'Toronto', 'Vancouver', 'Montreal', 'Mexico City'
            ],
            'budget_modifier': 1.3,
            'currency_hint': 'USD',  # Also CAD, MXN depending on country
            'popular_seasons': ['Spring (Apr-Jun)', 'Fall (Sep-Nov)']
        },
        'europe': {
            'primary_type': 'historical_places',
            'secondary_types': ['landmarks', 'nature'],
            'suggested_destinations': [
                'Paris', 'Rome', 'London', 'Barcelona',
                'Amsterdam', 'Prague', 'Vienna', 'Athens',
                'Lisbon', 'Berlin', 'Venice', 'Dublin'
            ],
            'budget_modifier': 1.2,
            'currency_hint': 'EUR',
            'popular_seasons': ['Spring (Apr-Jun)', 'Summer (Jul-Aug)']
        },
        'east_asia': {
            'primary_type': 'landmarks',
            'secondary_types': ['historical_places', 'entertainment'],
            'suggested_destinations': [
                'Tokyo', 'Kyoto', 'Osaka', 'Seoul',
                'Beijing', 'Shanghai', 'Hong Kong', 'Taipei',
                'Busan', 'Nara', 'Yokohama', 'Jeju Island'
            ],
            'budget_modifier': 1.1,
            'currency_hint': 'JPY',
            'popular_seasons': ['Spring (Mar-May)', 'Fall (Sep-Nov)']
        },
        'south_southeast_asia': {
            'primary_type': 'nature',
            'secondary_types': ['historical_places', 'entertainment'],
            'suggested_destinations': [
                'Bangkok', 'Singapore', 'Bali', 'Kuala Lumpur',
                'Phuket', 'Hanoi', 'Ho Chi Minh City', 'Siem Reap',
                'Manila', 'Penang', 'Chiang Mai', 'Yangon',
                'Jakarta', 'Boracay', 'Luang Prabang', 'Ubud'
            ],
            'budget_modifier': 0.8,
            'currency_hint': 'MYR',  # Common for SEA travelers, also THB, SGD, IDR, PHP
            'popular_seasons': ['Nov-Feb (Dry)', 'Year-round (varies)']
        },
        'oceania': {
            'primary_type': 'nature',
            'secondary_types': ['entertainment', 'landmarks'],
            'suggested_destinations': [
                'Sydney', 'Melbourne', 'Auckland', 'Brisbane',
                'Gold Coast', 'Wellington', 'Perth', 'Queenstown',
                'Cairns', 'Christchurch', 'Hobart', 'Fiji'
            ],
            'budget_modifier': 1.4,
            'currency_hint': 'AUD',
            'popular_seasons': ['Summer (Dec-Feb)', 'Spring (Sep-Nov)']
        },
        'middle_east': {
            'primary_type': 'historical_places',
            'secondary_types': ['landmarks', 'entertainment'],
            'suggested_destinations': [
                'Dubai', 'Abu Dhabi', 'Istanbul', 'Jerusalem',
                'Petra', 'Doha', 'Muscat', 'Riyadh',
                'Cairo', 'Amman', 'Tel Aviv', 'Beirut'
            ],
            'budget_modifier': 1.2,
            'currency_hint': 'USD',
            'popular_seasons': ['Oct-Apr (Mild)', 'Avoid Jun-Aug']
        },
        'africa': {
            'primary_type': 'nature',
            'secondary_types': ['historical_places', 'landmarks'],
            'suggested_destinations': [
                'Cape Town', 'Marrakech', 'Cairo', 'Nairobi',
                'Johannesburg', 'Zanzibar', 'Victoria Falls', 'Serengeti',
                'Casablanca', 'Luxor', 'Durban', 'Mauritius',
                'Tunis', 'Addis Ababa', 'Kruger Park', 'Fez'
            ],
            'budget_modifier': 0.9,
            'currency_hint': 'USD',
            'popular_seasons': ['May-Oct (Dry)', 'Jun-Aug (Safari)']
        },
        'caribbean_central_america': {
            'primary_type': 'nature',
            'secondary_types': ['entertainment', 'historical_places'],
            'suggested_destinations': [
                'Cancun', 'Havana', 'San Juan', 'Panama City',
                'Costa Rica', 'Aruba', 'Bahamas', 'Jamaica',
                'Barbados', 'Antigua', 'Cartagena', 'Belize City',
                'Punta Cana', 'Turks & Caicos', 'Guatemala City', 'San Jose'
            ],
            'budget_modifier': 1.0,
            'currency_hint': 'USD',
            'popular_seasons': ['Dec-Apr (Dry)', 'Avoid Sep-Nov (Hurricane)']
        },
        'south_america': {
            'primary_type': 'nature',
            'secondary_types': ['historical_places', 'landmarks'],
            'suggested_destinations': [
                'Rio de Janeiro', 'Buenos Aires', 'Lima', 'Cusco',
                'Santiago', 'Bogota', 'Cartagena', 'Quito',
                'Sao Paulo', 'Machu Picchu', 'Iguazu Falls', 'Montevideo',
                'Galapagos', 'Patagonia', 'Medellin', 'La Paz'
            ],
            'budget_modifier': 0.9,
            'currency_hint': 'USD',
            'popular_seasons': ['Dec-Mar (Summer)', 'Jun-Aug (Winter)']
        }
    }
    
    @classmethod
    def get_trip_type(cls, region: str, confidence: float = 1.0) -> str:
        """Get primary trip type from detected region"""
        if region not in cls.REGION_TO_PREFERENCES:
            return 'landmarks'  # Default fallback
        
        mapping = cls.REGION_TO_PREFERENCES[region]
        
        # If confidence is low, suggest multiple options
        if confidence < 0.6:
            return mapping['primary_type']  # Still return primary but flag for user
        
        return mapping['primary_type']
    
    @classmethod
    def get_destination_suggestions(cls, region: str, limit: int = 8) -> list:
        """
        Get destination suggestions based on region
        
        Args:
            region: Detected region name
            limit: Maximum number of suggestions to return
        
        Returns:
            List of suggested destination names
        """
        if region not in cls.REGION_TO_PREFERENCES:
            return []
        return cls.REGION_TO_PREFERENCES[region]['suggested_destinations'][:limit]
    
    @classmethod
    def get_all_destinations_for_region(cls, region: str) -> list:
        """Get all available destinations for a region"""
        if region not in cls.REGION_TO_PREFERENCES:
            return []
        return cls.REGION_TO_PREFERENCES[region]['suggested_destinations']
    
    @classmethod
    def get_region_info(cls, region: str) -> dict:
        """
        Get comprehensive region information
        
        Returns dict with:
        - destinations: list of suggested cities
        - trip_types: primary and secondary
        - budget_info: modifier and currency hint
        - season_info: best times to visit
        """
        if region not in cls.REGION_TO_PREFERENCES:
            return {
                'destinations': [],
                'trip_types': {'primary': 'landmarks', 'secondary': []},
                'budget_info': {'modifier': 1.0, 'currency': 'USD'},
                'season_info': ['Year-round']
            }
        
        mapping = cls.REGION_TO_PREFERENCES[region]
        return {
            'destinations': mapping['suggested_destinations'],
            'trip_types': {
                'primary': mapping['primary_type'],
                'secondary': mapping['secondary_types']
            },
            'budget_info': {
                'modifier': mapping['budget_modifier'],
                'currency': mapping.get('currency_hint', 'USD')
            },
            'season_info': mapping.get('popular_seasons', ['Year-round'])
        }
    
    @classmethod
    def adjust_budget(cls, region: str, budget_range: tuple) -> tuple:
        """Adjust budget based on region cost"""
        if region not in cls.REGION_TO_PREFERENCES:
            return budget_range
        
        modifier = cls.REGION_TO_PREFERENCES[region]['budget_modifier']
        return (int(budget_range[0] * modifier), int(budget_range[1] * modifier))
    
    @classmethod
    def get_enriched_query_context(cls, region: str) -> str:
        """Get additional context for RAG query based on region"""
        if region not in cls.REGION_TO_PREFERENCES:
            return ""
        
        mapping = cls.REGION_TO_PREFERENCES[region]
        context = f"Region: {region.replace('_', ' ').title()}. "
        context += f"Focus on {mapping['primary_type'].replace('_', ' ')} "
        context += f"and consider {', '.join(mapping['secondary_types'])}."
        return context
