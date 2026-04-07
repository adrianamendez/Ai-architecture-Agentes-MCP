"""
Weather tool functions used by both the MCP server and the orchestrator.
Data source: Open-Meteo API (no API key required).
"""

import httpx
from typing import Optional

OPEN_METEO_BASE_URL = "https://api.open-meteo.com/v1/forecast"

# WMO Weather interpretation codes → human-readable labels
WMO_CODES: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

# A small set of well-known cities with coordinates for convenience
KNOWN_CITIES: dict[str, dict] = {
    "bogota": {"lat": 4.71, "lon": -74.07, "label": "Bogotá, Colombia"},
    "new york": {"lat": 40.71, "lon": -74.01, "label": "New York, USA"},
    "london": {"lat": 51.51, "lon": -0.13, "label": "London, UK"},
    "paris": {"lat": 48.85, "lon": 2.35, "label": "Paris, France"},
    "tokyo": {"lat": 35.68, "lon": 139.69, "label": "Tokyo, Japan"},
    "sydney": {"lat": -33.87, "lon": 151.21, "label": "Sydney, Australia"},
    "berlin": {"lat": 52.52, "lon": 13.41, "label": "Berlin, Germany"},
    "madrid": {"lat": 40.42, "lon": -3.70, "label": "Madrid, Spain"},
    "sao paulo": {"lat": -23.55, "lon": -46.63, "label": "São Paulo, Brazil"},
    "dubai": {"lat": 25.20, "lon": 55.27, "label": "Dubai, UAE"},
    "mexico city": {"lat": 19.43, "lon": -99.13, "label": "Mexico City, Mexico"},
    "buenos aires": {"lat": -34.60, "lon": -58.38, "label": "Buenos Aires, Argentina"},
}


def get_current_weather(
    latitude: float,
    longitude: float,
    location_name: Optional[str] = None,
) -> dict:
    """
    Fetch current weather for given coordinates from Open-Meteo API.

    Returns a structured dict with temperature, wind speed, and condition.
    Raises httpx.HTTPError on network / API failure.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current_weather": "true",
        "hourly": "relative_humidity_2m,apparent_temperature",
        "timezone": "auto",
        "forecast_days": 1,
    }

    response = httpx.get(OPEN_METEO_BASE_URL, params=params, timeout=10.0)
    response.raise_for_status()
    data = response.json()

    current = data.get("current_weather", {})
    weather_code = int(current.get("weathercode", 0))

    # Pull first hourly apparent-temperature value as "feels like"
    hourly = data.get("hourly", {})
    feels_like_values = hourly.get("apparent_temperature", [])
    humidity_values = hourly.get("relative_humidity_2m", [])
    feels_like = feels_like_values[0] if feels_like_values else None
    humidity = humidity_values[0] if humidity_values else None

    is_day = current.get("is_day", 1)
    wind_direction = current.get("winddirection", None)

    return {
        "location": location_name or f"({latitude:.2f}, {longitude:.2f})",
        "latitude": latitude,
        "longitude": longitude,
        "temperature_celsius": current.get("temperature"),
        "feels_like_celsius": feels_like,
        "humidity_percent": humidity,
        "windspeed_kmh": current.get("windspeed"),
        "wind_direction_degrees": wind_direction,
        "weather_condition": WMO_CODES.get(weather_code, f"Code {weather_code}"),
        "weather_code": weather_code,
        "is_day": bool(is_day),
        "observation_time": current.get("time"),
        "timezone": data.get("timezone", "UTC"),
        "source": "Open-Meteo (https://open-meteo.com)",
    }


def resolve_location(city_name: str) -> Optional[dict]:
    """
    Try to resolve a city name to lat/lon from the known-cities lookup.
    Returns None if city is not in the table.
    """
    return KNOWN_CITIES.get(city_name.lower().strip())
