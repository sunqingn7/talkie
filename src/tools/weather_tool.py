"""
Weather Tool - Get weather information.
Uses Open-Meteo (free, no API key) with OpenStreetMap geocoding and IP-based location detection.
"""

import asyncio
from typing import Any, Dict, Optional
import requests

from . import BaseTool


class WeatherTool(BaseTool):
    """Tool to get weather information using Open-Meteo API (free, no API key required).
    
    Features:
    - Weather data for any city worldwide
    - IP-based location auto-detection (ipinfo.io primary, ip-api.com fallback)
    - Support for both metric and imperial units
    - Temperature, humidity, wind speed, weather conditions
    """
    
    def _get_description(self) -> str:
        return (
            "Get current weather information for a specified location. "
            "Supports cities worldwide. Automatically detects user's location from IP if not specified. "
            "Returns temperature, conditions, humidity, wind, and forecast. "
            "Uses Open-Meteo API (free, no key required) and ipinfo.io for IP geolocation."
        )
    
    def _get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name or location (e.g., 'San Francisco', 'London', 'Tokyo')"
                },
                "units": {
                    "type": "string",
                    "description": "Temperature units: 'metric' (Celsius) or 'imperial' (Fahrenheit)",
                    "enum": ["metric", "imperial"],
                    "default": "metric"
                }
            }
        }
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.weather_config = config.get("weather", {})
        self.openweathermap_key = self.weather_config.get("api_key")  # Optional
        self.default_city = self.weather_config.get("default_city", "San Francisco")
        self.default_units = self.weather_config.get("units", "metric")
        self.auto_detect_location = self.weather_config.get("auto_detect_location", True)
        
        # Open-Meteo is always available (no key needed)
        self.use_open_meteo = True
        
        # Cache for IP-based location
        self._ip_location_cache = None
    
    async def _get_location_from_ip(self) -> Optional[Dict[str, str]]:
        """Get location from IP address using ipinfo.io (free, no API key needed)."""
        if self._ip_location_cache:
            return self._ip_location_cache
            
        try:
            # Try ipinfo.io (reliable, free, no key needed)
            response = requests.get("https://ipinfo.io/json", timeout=5)
            data = response.json()
            
            if data.get("city"):
                self._ip_location_cache = {
                    "city": data.get("city", ""),
                    "region": data.get("region", ""),
                    "country": data.get("country", ""),
                    "lat": float(data.get("loc", ",").split(",")[0]) if data.get("loc") else None,
                    "lon": float(data.get("loc", ",").split(",")[1]) if data.get("loc") else None
                }
                print(f"   📍 Detected location from IP: {self._ip_location_cache}")
                return self._ip_location_cache
            else:
                print(f"   ⚠️  No city found in IP response")
        except Exception as e:
            print(f"   ⚠️  Failed to detect location from IP (ipinfo.io): {e}")
        
        # Try ip-api.com as fallback
        try:
            response = requests.get("http://ip-api.com/json/?fields=status,country,city,regionName,lat,lon", timeout=5)
            data = response.json()
            
            if data.get("status") == "success" and data.get("city"):
                self._ip_location_cache = {
                    "city": data.get("city", ""),
                    "region": data.get("regionName", ""),
                    "country": data.get("country", ""),
                    "lat": data.get("lat"),
                    "lon": data.get("lon")
                }
                print(f"   📍 Detected location from IP-API: {self._ip_location_cache}")
                return self._ip_location_cache
        except Exception as e:
            print(f"   ⚠️  Failed to detect location from IP (ip-api.com): {e}")
        
        return None
    
    async def execute(self, location: str = None, units: str = None) -> Dict[str, Any]:
        """Get weather for location."""
        units = units or self.default_units
        
        # Auto-detect location from IP if not provided
        if not location or location.lower() in ["my location", "here", "current location", "me"]:
            if self.auto_detect_location:
                ip_location = await self._get_location_from_ip()
                if ip_location:
                    location = ip_location["city"]
                    print(f"   📍 Using IP-detected location: {location}")
                else:
                    location = self.default_city
            else:
                location = self.default_city
        else:
            location = location or self.default_city
        
        print(f"🌤️  Getting weather for: {location} ({units})")
        
        # Try Open-Meteo first (free, no key needed)
        if self.use_open_meteo:
            result = await self._get_weather_open_meteo(location, units)
            if result.get("success"):
                return result
        
        # Fallback to OpenWeatherMap if API key available
        if self.openweathermap_key:
            result = await self._get_weather_openweathermap(location, units)
            return result
        
        # Last resort: mock data
        return self._get_mock_weather(location, units)
    
    async def _get_weather_open_meteo(self, location: str, units: str) -> Dict[str, Any]:
        """Get weather using Open-Meteo API (free, no API key)."""
        try:
            # Step 1: Geocoding - convert city name to coordinates
            geocode_url = "https://geocoding-api.open-meteo.com/v1/search"
            geocode_params = {
                "name": location,
                "count": 1,
                "language": "en",
                "format": "json"
            }
            
            geo_response = requests.get(geocode_url, params=geocode_params, timeout=10)
            geo_data = geo_response.json()
            
            if not geo_data.get("results"):
                return {
                    "success": False,
                    "error": f"Location '{location}' not found",
                    "location": location
                }
            
            # Get coordinates
            lat = geo_data["results"][0]["latitude"]
            lon = geo_data["results"][0]["longitude"]
            city_name = geo_data["results"][0]["name"]
            country = geo_data["results"][0].get("country", "")
            
            # Step 2: Get weather data
            weather_url = "https://api.open-meteo.com/v1/forecast"
            weather_params = {
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m",
                "timezone": "auto"
            }
            
            weather_response = requests.get(weather_url, params=weather_params, timeout=10)
            weather_data = weather_response.json()
            
            if "current" not in weather_data:
                return {
                    "success": False,
                    "error": "Failed to fetch weather data",
                    "location": location
                }
            
            current = weather_data["current"]
            
            # Convert units
            temp = current["temperature_2m"]
            feels_like = current["apparent_temperature"]
            humidity = current["relative_humidity_2m"]
            wind_speed = current["wind_speed_10m"]
            
            if units == "imperial":
                temp = temp * 9/5 + 32
                feels_like = feels_like * 9/5 + 32
                wind_speed = wind_speed * 2.237  # m/s to mph
                temp_unit = "°F"
                wind_unit = "mph"
            else:
                temp_unit = "°C"
                wind_unit = "m/s"
            
            # Get weather condition from WMO code
            weather_code = current.get("weather_code", 0)
            conditions = self._get_weather_condition(weather_code)
            
            full_location = f"{city_name}, {country}" if country else city_name
            
            return {
                "success": True,
                "source": "Open-Meteo (free)",
                "location": full_location,
                "temperature": f"{temp:.1f}{temp_unit}",
                "feels_like": f"{feels_like:.1f}{temp_unit}",
                "conditions": conditions,
                "humidity": f"{humidity}%",
                "wind_speed": f"{wind_speed:.1f} {wind_unit}",
                "units": units,
                "note": "Powered by Open-Meteo (free, no API key needed)"
            }
            
        except requests.exceptions.RequestException as e:
            print(f"   ⚠️  Open-Meteo request failed: {e}")
            return {
                "success": False,
                "error": f"Failed to connect to weather service: {str(e)}",
                "location": location
            }
        except Exception as e:
            print(f"   ⚠️  Weather error: {e}")
            return {
                "success": False,
                "error": str(e),
                "location": location
            }
    
    def _get_weather_condition(self, code: int) -> str:
        """Convert WMO weather code to description."""
        weather_codes = {
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
            71: "Slight snow",
            73: "Moderate snow",
            75: "Heavy snow",
            77: "Snow grains",
            80: "Slight rain showers",
            81: "Moderate rain showers",
            82: "Violent rain showers",
            85: "Slight snow showers",
            86: "Heavy snow showers",
            95: "Thunderstorm",
            96: "Thunderstorm with slight hail",
            99: "Thunderstorm with heavy hail"
        }
        return weather_codes.get(code, "Unknown")
    
    async def _get_weather_openweathermap(self, location: str, units: str) -> Dict[str, Any]:
        """Get weather using OpenWeatherMap API (requires API key)."""
        if not self.openweathermap_key:
            return {
                "success": False,
                "error": "OpenWeatherMap API key not configured",
                "location": location
            }
        
        try:
            url = "http://api.openweathermap.org/data/2.5/weather"
            params = {
                "q": location,
                "appid": self.openweathermap_key,
                "units": "metric" if units == "metric" else "imperial"
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if response.status_code != 200:
                return {
                    "success": False,
                    "error": data.get("message", "Failed to fetch weather"),
                    "location": location
                }
            
            temp_unit = "°C" if units == "metric" else "°F"
            wind_unit = "m/s" if units == "metric" else "mph"
            
            return {
                "success": True,
                "source": "OpenWeatherMap",
                "location": f"{data['name']}, {data['sys']['country']}",
                "temperature": f"{data['main']['temp']:.1f}{temp_unit}",
                "feels_like": f"{data['main']['feels_like']:.1f}{temp_unit}",
                "conditions": data['weather'][0]['description'].title(),
                "humidity": f"{data['main']['humidity']}%",
                "wind_speed": f"{data['wind']['speed']} {wind_unit}",
                "units": units
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "location": location
            }
    
    def _get_mock_weather(self, location: str, units: str) -> Dict[str, Any]:
        """Return mock weather data when no API available."""
        temp_unit = "°C" if units == "metric" else "°F"
        temp = 22 if units == "metric" else 72
        
        return {
            "success": True,
            "note": "Using mock data (weather service unavailable)",
            "location": location,
            "temperature": f"{temp}{temp_unit}",
            "feels_like": f"{temp}{temp_unit}",
            "conditions": "Partly Cloudy",
            "humidity": "65%",
            "wind_speed": f"3.5 {'m/s' if units == 'metric' else 'mph'}",
            "units": units
        }
