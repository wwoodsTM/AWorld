"""
Google Maps MCP Server

This module provides a microservice for interacting with the Google Maps API through MCP.
It offers tools for geocoding, distance matrix calculations, directions, and place details.

Key features:
- Geocode addresses to latitude and longitude
- Calculate distances and travel times between locations
- Retrieve directions and route information
- Access detailed place information

Main functions:
- geocode: Geocodes addresses using Google Maps API
- distance_matrix: Calculates distances and travel times
- directions: Retrieves directions and route information
- place_details: Fetches detailed information about places
"""

import os
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import googlemaps
from pydantic import BaseModel, Field

from aworld.logs.util import logger
from aworld.mcp_servers.utils import parse_port, run_mcp_server
from aworld.utils import import_package


# Define model classes for different Google Maps API responses
class GeocodingResult(BaseModel):
    """Model representing a geocoding result"""

    formatted_address: str
    place_id: str
    location: Dict[str, float]  # lat, lng
    location_type: str
    types: List[str]
    partial_match: Optional[bool] = None


class GeocodingResponse(BaseModel):
    """Model representing geocoding API response"""

    results: List[GeocodingResult]
    status: str


class DistanceMatrixElement(BaseModel):
    """Model representing a single element in distance matrix response"""

    status: str
    duration: Optional[Dict[str, Union[str, int]]] = None
    distance: Optional[Dict[str, Union[str, int]]] = None
    duration_in_traffic: Optional[Dict[str, Union[str, int]]] = None


class DistanceMatrixRow(BaseModel):
    """Model representing a row in distance matrix response"""

    elements: List[DistanceMatrixElement]


class DistanceMatrixResponse(BaseModel):
    """Model representing distance matrix API response"""

    origin_addresses: List[str]
    destination_addresses: List[str]
    rows: List[DistanceMatrixRow]
    status: str


class DirectionsLeg(BaseModel):
    """Model representing a leg in directions response"""

    start_address: str
    end_address: str
    distance: Dict[str, Union[str, int]]
    duration: Dict[str, Union[str, int]]
    steps: List[Dict[str, Any]]  # Steps are complex, using Dict for simplicity


class DirectionsRoute(BaseModel):
    """Model representing a route in directions response"""

    summary: str
    legs: List[DirectionsLeg]
    warnings: List[str]
    bounds: Dict[str, Dict[str, float]]
    copyrights: str
    overview_polyline: Dict[str, str]


class DirectionsResponse(BaseModel):
    """Model representing directions API response"""

    routes: List[DirectionsRoute]
    status: str
    geocoded_waypoints: List[Dict[str, Any]]


class PlaceDetails(BaseModel):
    """Model representing place details response"""

    place_id: str
    name: str
    formatted_address: Optional[str] = None
    formatted_phone_number: Optional[str] = None
    geometry: Optional[Dict[str, Any]] = None
    rating: Optional[float] = None
    types: Optional[List[str]] = None
    website: Optional[str] = None
    opening_hours: Optional[Dict[str, Any]] = None
    photos: Optional[List[Dict[str, Any]]] = None
    reviews: Optional[List[Dict[str, Any]]] = None


class PlaceSearchResult(BaseModel):
    """Model representing a place search result"""

    place_id: str
    name: str
    geometry: Dict[str, Any]
    vicinity: Optional[str] = None
    types: List[str]
    rating: Optional[float] = None
    user_ratings_total: Optional[int] = None
    photos: Optional[List[Dict[str, Any]]] = None


class PlaceSearchResponse(BaseModel):
    """Model representing place search API response"""

    results: List[PlaceSearchResult]
    status: str
    next_page_token: Optional[str] = None


class LatLngResult(BaseModel):
    """Model representing latitude and longitude result"""

    address: str
    latitude: float
    longitude: float
    status: str


class PostcodeResult(BaseModel):
    """Model representing postcode (zip code) result"""

    address: str
    postcode: str
    country: Optional[str] = None
    country_code: Optional[str] = None
    status: str


class GoogleMapsError(BaseModel):
    """Model representing an error in Google Maps API processing"""

    error: str
    status: str


class GoogleMapsServer:
    """
    Google Maps Server class for interacting with the Google Maps API.

    This class provides methods for geocoding, distance matrix calculations,
    directions, and place details using the Google Maps API.
    """

    _instance = None
    _gmaps = None
    _api_key = None

    def __new__(cls):
        """Implement singleton pattern"""
        if cls._instance is None:
            cls._instance = super(GoogleMapsServer, cls).__new__(cls)
            cls._instance._init_server()
        return cls._instance

    def _init_server(self):
        """Initialize the Google Maps server and client"""
        # Import googlemaps package, install if not available
        import_package("googlemaps")
        self._api_key = os.environ.get("GOOGLE_MAPS_SECRET")
        if not self._api_key:
            logger.warning("No Google Maps API key found. API calls will fail.")
        self._gmaps = googlemaps.Client(key=self._api_key)
        logger.info("GoogleMapsServer initialized")

    @classmethod
    def get_instance(cls):
        """Get the singleton instance of GoogleMapsServer"""
        if cls._instance is None:
            return cls()
        return cls._instance

    @staticmethod
    def handle_error(e: Exception, error_type: str) -> str:
        """Unified error handling and return standard format error message"""
        error_msg = f"{error_type} error: {str(e)}"
        logger.error(traceback.format_exc())

        error = GoogleMapsError(error=error_msg, status="ERROR")

        return error.model_dump_json()

    @classmethod
    def geocode(
        cls,
        address: str = Field(description="Address or place name to geocode"),
        language: str = Field(
            default="en", description="Language for results (default: en)"
        ),
        region: Optional[str] = Field(
            default=None, description="Region bias (optional)"
        ),
    ) -> str:
        """
        Convert an address or place name to geographic coordinates (latitude and longitude).

        Args:
            address: The address or place name to geocode
            language: Language code for results (e.g., 'en', 'zh-CN')
            region: Region bias for results (e.g., 'us', 'cn')

        Returns:
            JSON string containing geocoding results
        """
        try:
            # Handle Field objects if they're passed directly
            if hasattr(address, "default") and not isinstance(address, str):
                address = address.default

            if hasattr(language, "default") and not isinstance(language, str):
                language = language.default

            if (
                hasattr(region, "default")
                and not isinstance(region, str)
                and region is not None
            ):
                region = region.default

            # Get the singleton instance
            instance = cls.get_instance()

            # Call geocoding API
            geocode_result = instance._gmaps.geocode(
                address=address, language=language, region=region
            )

            # Process results
            results = []
            for result in geocode_result:
                # Extract location data
                location = result.get("geometry", {}).get("location", {})
                location_type = result.get("geometry", {}).get("location_type", "")

                # Create result object
                geocode_item = GeocodingResult(
                    formatted_address=result.get("formatted_address", ""),
                    place_id=result.get("place_id", ""),
                    location=location,
                    location_type=location_type,
                    types=result.get("types", []),
                    partial_match=result.get("partial_match", False),
                )
                results.append(geocode_item)

            # Create response
            response = GeocodingResponse(
                results=results,
                status="OK" if results else "ZERO_RESULTS",
            )

            return response.model_dump_json()

        except Exception as e:
            return cls.handle_error(e, "Geocoding")

    @classmethod
    def distance_matrix(
        cls,
        origins: List[str] = Field(description="List of origin addresses"),
        destinations: List[str] = Field(description="List of destination addresses"),
        mode: str = Field(
            default="driving",
            description="Travel mode (driving, walking, bicycling, transit)",
        ),
        language: str = Field(
            default="en", description="Language for results (default: en)"
        ),
        units: str = Field(default="metric", description="Units (metric or imperial)"),
        departure_time: Optional[str] = Field(
            default=None,
            description="Departure time (now or ISO datetime string)",
        ),
    ) -> str:
        """
        Calculate distance and travel time between multiple origins and destinations.

        Args:
            origins: List of origin addresses or coordinates
            destinations: List of destination addresses or coordinates
            mode: Travel mode (driving, walking, bicycling, transit)
            language: Language code for results
            units: Units system (metric or imperial)
            departure_time: Departure time (now or ISO datetime string)

        Returns:
            JSON string containing distance matrix results
        """
        try:
            # Handle Field objects if they're passed directly
            if hasattr(origins, "default") and not isinstance(origins, list):
                origins = origins.default

            if hasattr(destinations, "default") and not isinstance(destinations, list):
                destinations = destinations.default

            if hasattr(mode, "default") and not isinstance(mode, str):
                mode = mode.default

            if hasattr(language, "default") and not isinstance(language, str):
                language = language.default

            if hasattr(units, "default") and not isinstance(units, str):
                units = units.default

            if (
                hasattr(departure_time, "default")
                and not isinstance(departure_time, str)
                and departure_time is not None
            ):
                departure_time = departure_time.default

            # Get the singleton instance
            instance = cls.get_instance()

            # Process departure time
            departure_time_param = None
            if departure_time:
                if departure_time.lower() == "now":
                    departure_time_param = "now"
                else:
                    try:
                        dt = datetime.fromisoformat(departure_time)
                        departure_time_param = int(dt.timestamp())
                    except ValueError:
                        return cls.handle_error(
                            ValueError(
                                "Invalid departure_time format. Use 'now' or ISO datetime string."
                            ),
                            "Distance Matrix",
                        )

            # Call distance matrix API
            distance_result = instance._gmaps.distance_matrix(
                origins=origins,
                destinations=destinations,
                mode=mode,
                language=language,
                units=units,
                departure_time=departure_time_param,
            )

            # Process rows and elements
            rows = []
            for row_data in distance_result.get("rows", []):
                elements = []
                for element_data in row_data.get("elements", []):
                    element = DistanceMatrixElement(
                        status=element_data.get("status", ""),
                        duration=element_data.get("duration"),
                        distance=element_data.get("distance"),
                        duration_in_traffic=element_data.get("duration_in_traffic"),
                    )
                    elements.append(element)
                row = DistanceMatrixRow(elements=elements)
                rows.append(row)

            # Create response
            response = DistanceMatrixResponse(
                origin_addresses=distance_result.get("origin_addresses", []),
                destination_addresses=distance_result.get("destination_addresses", []),
                rows=rows,
                status=distance_result.get("status", ""),
            )

            return response.model_dump_json()

        except Exception as e:
            return cls.handle_error(e, "Distance Matrix")

    @classmethod
    def directions(
        cls,
        origin: str = Field(description="Origin address or coordinates"),
        destination: str = Field(description="Destination address or coordinates"),
        mode: str = Field(
            default="driving",
            description="Travel mode (driving, walking, bicycling, transit)",
        ),
        waypoints: List[str] = Field(
            default=[], description="List of waypoint addresses (optional)"
        ),
        alternatives: bool = Field(
            default=False, description="Request alternative routes"
        ),
        avoid: List[str] = Field(
            default=[],
            description="Features to avoid (tolls, highways, ferries, indoor)",
        ),
        language: str = Field(
            default="en", description="Language for results (default: en)"
        ),
        units: str = Field(default="metric", description="Units (metric or imperial)"),
        departure_time: Optional[str] = Field(
            default=None,
            description="Departure time (now or ISO datetime string)",
        ),
    ) -> str:
        """
        Get directions between an origin and destination.

        Args:
            origin: Origin address or coordinates
            destination: Destination address or coordinates
            mode: Travel mode (driving, walking, bicycling, transit)
            waypoints: List of waypoint addresses (optional)
            alternatives: Request alternative routes
            avoid: Features to avoid (tolls, highways, ferries, indoor)
            language: Language code for results
            units: Units system (metric or imperial)
            departure_time: Departure time (now or ISO datetime string)

        Returns:
            JSON string containing directions results
        """
        try:
            # Handle Field objects if they're passed directly
            if hasattr(origin, "default") and not isinstance(origin, str):
                origin = origin.default

            if hasattr(destination, "default") and not isinstance(destination, str):
                destination = destination.default

            if hasattr(mode, "default") and not isinstance(mode, str):
                mode = mode.default

            if hasattr(waypoints, "default") and not isinstance(waypoints, list):
                waypoints = waypoints.default

            if hasattr(alternatives, "default") and not isinstance(alternatives, bool):
                alternatives = alternatives.default

            if hasattr(avoid, "default") and not isinstance(avoid, list):
                avoid = avoid.default

            if hasattr(language, "default") and not isinstance(language, str):
                language = language.default

            if hasattr(units, "default") and not isinstance(units, str):
                units = units.default

            if (
                hasattr(departure_time, "default")
                and not isinstance(departure_time, str)
                and departure_time is not None
            ):
                departure_time = departure_time.default

            # Get the singleton instance
            instance = cls.get_instance()

            # Process departure time
            departure_time_param = None
            if departure_time:
                if departure_time.lower() == "now":
                    departure_time_param = "now"
                else:
                    try:
                        dt = datetime.fromisoformat(departure_time)
                        departure_time_param = int(dt.timestamp())
                    except ValueError:
                        return cls.handle_error(
                            ValueError(
                                "Invalid departure_time format. Use 'now' or ISO datetime string."
                            ),
                            "Directions",
                        )

            # Process avoid parameter
            avoid_param = "|".join(avoid) if avoid else None

            # Call directions API
            directions_result = instance._gmaps.directions(
                origin=origin,
                destination=destination,
                mode=mode,
                waypoints=waypoints if waypoints else None,
                alternatives=alternatives,
                avoid=avoid_param,
                language=language,
                units=units,
                departure_time=departure_time_param,
            )

            # Process routes
            routes = []
            for route_data in directions_result:
                # Process legs
                legs = []
                for leg_data in route_data.get("legs", []):
                    leg = DirectionsLeg(
                        start_address=leg_data.get("start_address", ""),
                        end_address=leg_data.get("end_address", ""),
                        distance=leg_data.get("distance", {}),
                        duration=leg_data.get("duration", {}),
                        steps=leg_data.get("steps", []),
                    )
                    legs.append(leg)

                # Create route object
                route = DirectionsRoute(
                    summary=route_data.get("summary", ""),
                    legs=legs,
                    warnings=route_data.get("warnings", []),
                    bounds=route_data.get("bounds", {}),
                    copyrights=route_data.get("copyrights", ""),
                    overview_polyline=route_data.get("overview_polyline", {}),
                )
                routes.append(route)

            # Create response
            response = DirectionsResponse(
                routes=routes,
                status="OK" if routes else "ZERO_RESULTS",
                geocoded_waypoints=(
                    directions_result[0].get("geocoded_waypoints", [])
                    if directions_result
                    else []
                ),
            )

            return response.model_dump_json()

        except Exception as e:
            return cls.handle_error(e, "Directions")

    @classmethod
    def place_details(
        cls,
        place_id: str = Field(description="Google Place ID"),
        language: str = Field(
            default="en", description="Language for results (default: en)"
        ),
        fields: List[str] = Field(
            default=[],
            description="Fields to include in the response (optional)",
        ),
    ) -> str:
        """
        Get detailed information about a place using its Place ID.

        Args:
            place_id: Google Place ID
            language: Language code for results
            fields: List of fields to include in the response

        Returns:
            JSON string containing place details
        """
        try:
            # Handle Field objects if they're passed directly
            if hasattr(place_id, "default") and not isinstance(place_id, str):
                place_id = place_id.default

            if hasattr(language, "default") and not isinstance(language, str):
                language = language.default

            if hasattr(fields, "default") and not isinstance(fields, list):
                fields = fields.default

            # Get the singleton instance
            instance = cls.get_instance()

            # Call place details API
            place_result = instance._gmaps.place(
                place_id=place_id,
                language=language,
                fields=fields if fields else None,
            )

            # Extract place data
            place_data = place_result.get("result", {})

            # Create place details object
            place_details = PlaceDetails(
                place_id=place_data.get("place_id", ""),
                name=place_data.get("name", ""),
                formatted_address=place_data.get("formatted_address"),
                formatted_phone_number=place_data.get("formatted_phone_number"),
                geometry=place_data.get("geometry"),
                rating=place_data.get("rating"),
                types=place_data.get("types"),
                website=place_data.get("website"),
                opening_hours=place_data.get("opening_hours"),
                photos=place_data.get("photos"),
                reviews=place_data.get("reviews"),
            )

            return place_details.model_dump_json()

        except Exception as e:
            return cls.handle_error(e, "Place Details")

    @classmethod
    def place_search(
        cls,
        query: str = Field(description="Search query"),
        location: Optional[str] = Field(
            default=None, description="Location (lat,lng) to search around"
        ),
        radius: int = Field(
            default=1000, description="Search radius in meters (max 50000)"
        ),
        language: str = Field(
            default="en", description="Language for results (default: en)"
        ),
        type: Optional[str] = Field(
            default=None, description="Place type (e.g., restaurant, cafe)"
        ),
        min_price: int = Field(default=0, description="Minimum price level (0-4)"),
        max_price: int = Field(default=4, description="Maximum price level (0-4)"),
        open_now: bool = Field(
            default=False, description="Only return places that are open now"
        ),
    ) -> str:
        """
        Search for places based on a text query and/or location.

        Args:
            query: Search query
            location: Location (lat,lng) to search around
            radius: Search radius in meters (max 50000)
            language: Language code for results
            type: Place type (e.g., restaurant, cafe)
            min_price: Minimum price level (0-4)
            max_price: Maximum price level (0-4)
            open_now: Only return places that are open now

        Returns:
            JSON string containing place search results
        """
        try:
            # Handle Field objects if they're passed directly
            if hasattr(query, "default") and not isinstance(query, str):
                query = query.default

            if (
                hasattr(location, "default")
                and not isinstance(location, str)
                and location is not None
            ):
                location = location.default

            if hasattr(radius, "default") and not isinstance(radius, int):
                radius = radius.default

            if hasattr(language, "default") and not isinstance(language, str):
                language = language.default

            if (
                hasattr(type, "default")
                and not isinstance(type, str)
                and type is not None
            ):
                type = type.default

            if hasattr(min_price, "default") and not isinstance(min_price, int):
                min_price = min_price.default

            if hasattr(max_price, "default") and not isinstance(max_price, int):
                max_price = max_price.default

            if hasattr(open_now, "default") and not isinstance(open_now, bool):
                open_now = open_now.default

            # Get the singleton instance
            instance = cls.get_instance()

            # Process location parameter
            location_param = None
            if location:
                try:
                    lat, lng = map(float, location.split(","))
                    location_param = (lat, lng)
                except ValueError:
                    return cls.handle_error(
                        ValueError("Invalid location format. Use 'lat,lng'."),
                        "Place Search",
                    )

            # Call places API
            if location_param:
                # Nearby search
                search_result = instance._gmaps.places_nearby(
                    location=location_param,
                    radius=radius,
                    keyword=query,
                    language=language,
                    type=type,
                    min_price=min_price,
                    max_price=max_price,
                    open_now=open_now,
                )
            else:
                # Text search
                search_result = instance._gmaps.places(
                    query=query,
                    language=language,
                    type=type,
                    min_price=min_price,
                    max_price=max_price,
                    open_now=open_now,
                )

            # Process results
            results = []
            for place_data in search_result.get("results", []):
                place = PlaceSearchResult(
                    place_id=place_data.get("place_id", ""),
                    name=place_data.get("name", ""),
                    geometry=place_data.get("geometry", {}),
                    vicinity=place_data.get("vicinity"),
                    types=place_data.get("types", []),
                    rating=place_data.get("rating"),
                    user_ratings_total=place_data.get("user_ratings_total"),
                    photos=place_data.get("photos"),
                )
                results.append(place)

            # Create response
            response = PlaceSearchResponse(
                results=results,
                status=search_result.get("status", ""),
                next_page_token=search_result.get("next_page_token"),
            )

            return response.model_dump_json()

        except Exception as e:
            return cls.handle_error(e, "Place Search")

    @classmethod
    def get_latlng(
        cls,
        address: str = Field(description="Address to convert to coordinates"),
    ) -> str:
        """
        Convert an address to latitude and longitude coordinates.

        Args:
            address: The address to convert

        Returns:
            JSON string containing latitude and longitude
        """
        try:
            # Handle Field objects if they're passed directly
            if hasattr(address, "default") and not isinstance(address, str):
                address = address.default

            # Get the singleton instance
            instance = cls.get_instance()

            # Call geocoding API
            geocode_result = instance._gmaps.geocode(address=address)

            # Extract coordinates
            if geocode_result:
                location = geocode_result[0].get("geometry", {}).get("location", {})
                lat = location.get("lat")
                lng = location.get("lng")

                if lat is not None and lng is not None:
                    result = LatLngResult(
                        address=address,
                        latitude=lat,
                        longitude=lng,
                        status="OK",
                    )
                    return result.model_dump_json()

            # No results found
            result = LatLngResult(
                address=address,
                latitude=0.0,
                longitude=0.0,
                status="ZERO_RESULTS",
            )
            return result.model_dump_json()

        except Exception as e:
            return cls.handle_error(e, "Get LatLng")

    @classmethod
    def get_postcode(
        cls,
        address: str = Field(description="Address to extract postcode from"),
    ) -> str:
        """
        Extract postcode (zip code) from an address.

        Args:
            address: The address to extract postcode from

        Returns:
            JSON string containing postcode information
        """
        try:
            # Handle Field objects if they're passed directly
            if hasattr(address, "default") and not isinstance(address, str):
                address = address.default

            # Get the singleton instance
            instance = cls.get_instance()

            # Call geocoding API
            geocode_result = instance._gmaps.geocode(address=address)

            # Extract postcode
            postcode = None
            country = None
            country_code = None

            if geocode_result:
                # Look for postal code in address components
                for component in geocode_result[0].get("address_components", []):
                    if "postal_code" in component.get("types", []):
                        postcode = component.get("long_name")
                    if "country" in component.get("types", []):
                        country = component.get("long_name")
                        country_code = component.get("short_name")

            # Create result
            if postcode:
                result = PostcodeResult(
                    address=address,
                    postcode=postcode,
                    country=country,
                    country_code=country_code,
                    status="OK",
                )
            else:
                result = PostcodeResult(
                    address=address,
                    postcode="",
                    country=country,
                    country_code=country_code,
                    status="NOT_FOUND",
                )

            return result.model_dump_json()

        except Exception as e:
            return cls.handle_error(e, "Get Postcode")


# Main function
if __name__ == "__main__":
    port = parse_port()

    google_maps_server = GoogleMapsServer.get_instance()
    logger.info("GoogleMapsServer initialized and ready to handle requests")

    run_mcp_server(
        "Google Maps Server",
        funcs=[
            google_maps_server.geocode,
            google_maps_server.distance_matrix,
            google_maps_server.directions,
            google_maps_server.place_details,
            google_maps_server.place_search,
            google_maps_server.get_latlng,
            google_maps_server.get_postcode,
        ],
        port=port,
    )
