"""
Weather MCP Server — runnable as a standalone MCP service.

Usage (stdio transport, for use with MCP clients / Agent SDK):
    python -m mcp_servers.weather_server

Provides the tool: get_current_weather(latitude, longitude, location_name?)
"""

from mcp.server.fastmcp import FastMCP
from .weather_tools import get_current_weather as _fetch

mcp = FastMCP("weather-mcp-server")


@mcp.tool()
def get_current_weather(
    latitude: float,
    longitude: float,
    location_name: str = "",
) -> dict:
    """
    Get current weather conditions for a geographic location.

    Uses the Open-Meteo API (no API key required).

    Args:
        latitude: Geographic latitude in decimal degrees.
        longitude: Geographic longitude in decimal degrees.
        location_name: Human-readable name for the location (optional).

    Returns:
        Weather data including temperature (°C), wind speed (km/h),
        humidity (%), apparent temperature (°C), and weather condition.
    """
    return _fetch(latitude, longitude, location_name or None)


if __name__ == "__main__":
    mcp.run()
