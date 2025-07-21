#!/usr/bin/env python3
"""
Real-time data collection from various APIs for dam monitoring
"""

import os
import json
import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import psycopg2
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class RealtimeAPICollector:
    """Collect real-time data from various APIs"""
    
    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
        
        # API configurations
        self.apis = {
            'openweather': {
                'base_url': 'https://api.openweathermap.org/data/2.5',
                'api_key': os.getenv('OPENWEATHER_API_KEY', 'your_api_key_here'),
                'endpoints': {
                    'current': '/weather',
                    'forecast': '/forecast',
                    'onecall': '/onecall'
                }
            },
            'usgs_earthquake': {
                'base_url': 'https://earthquake.usgs.gov/fdsnws/event/1',
                'endpoints': {
                    'query': '/query'
                }
            },
            'nasa_gpm': {
                'base_url': 'https://gpm1.gesdisc.eosdis.nasa.gov/data',
                'info': 'Global Precipitation Measurement'
            }
        }
        
        # Dam locations for API queries
        self.dam_locations = {
            'Alta Dam': {'lat': 69.968, 'lon': 23.272, 'country': 'Norway'},
            'Tehri Dam': {'lat': 30.378, 'lon': 78.478, 'country': 'India'},
            'Kaptai Dam': {'lat': 22.5, 'lon': 92.3, 'country': 'Bangladesh'}
        }
    
    def collect_weather_data(self, dam_name: str) -> Optional[Dict]:
        """Collect current weather data for a dam location"""
        if dam_name not in self.dam_locations:
            logger.error(f"Dam {dam_name} not found in locations")
            return None
        
        location = self.dam_locations[dam_name]
        api_config = self.apis['openweather']
        
        params = {
            'lat': location['lat'],
            'lon': location['lon'],
            'appid': api_config['api_key'],
            'units': 'metric'
        }
        
        try:
            url = api_config['base_url'] + api_config['endpoints']['current']
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                weather_data = {
                    'dam_name': dam_name,
                    'timestamp': datetime.now(),
                    'temperature': data['main']['temp'],
                    'humidity': data['main']['humidity'],
                    'pressure': data['main']['pressure'],
                    'wind_speed': data['wind']['speed'],
                    'precipitation': data.get('rain', {}).get('1h', 0),
                    'weather_condition': data['weather'][0]['main'],
                    'clouds': data['clouds']['all']
                }
                
                logger.info(f"Weather data collected for {dam_name}")
                return weather_data
            else:
                logger.error(f"Weather API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error collecting weather data: {e}")
            return None
    
    def detect_extreme_weather(self, dam_name: str, hours: int = 24) -> List[Dict]:
        """Detect extreme weather events including cloudbursts"""
        location = self.dam_locations.get(dam_name)
        if not location:
            return []
        
        extreme_events = []
        
        # Get forecast data
        api_config = self.apis['openweather']
        params = {
            'lat': location['lat'],
            'lon': location['lon'],
            'appid': api_config['api_key'],
            'units': 'metric',
            'cnt': hours // 3  # 3-hour intervals
        }
        
        try:
            url = api_config['base_url'] + api_config['endpoints']['forecast']
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                for forecast in data['list']:
                    # Check for extreme precipitation (potential cloudburst)
                    rain_3h = forecast.get('rain', {}).get('3h', 0)
                    
                    if rain_3h > 50:  # mm in 3 hours
                        event = {
                            'dam_name': dam_name,
                            'event_type': 'cloudburst' if rain_3h > 100 else 'extreme_rainfall',
                            'timestamp': datetime.fromtimestamp(forecast['dt']),
                            'precipitation_mm': rain_3h,
                            'wind_speed': forecast['wind']['speed'],
                            'description': forecast['weather'][0]['description']
                        }
                        extreme_events.append(event)
                
        except Exception as e:
            logger.error(f"Error detecting extreme weather: {e}")
        
        return extreme_events
    
    def collect_seismic_data(self, dam_name: str, days: int = 7) -> List[Dict]:
        """Collect earthquake data near dam location"""
        location = self.dam_locations.get(dam_name)
        if not location:
            return []
        
        # USGS earthquake API parameters
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        params = {
            'format': 'geojson',
            'starttime': start_time.isoformat(),
            'endtime': end_time.isoformat(),
            'latitude': location['lat'],
            'longitude': location['lon'],
            'maxradiuskm': 200,  # 200km radius
            'minmagnitude': 2.5
        }
        
        try:
            url = self.apis['usgs_earthquake']['base_url'] + self.apis['usgs_earthquake']['endpoints']['query']
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                earthquakes = []
                
                for feature in data['features']:
                    props = feature['properties']
                    coords = feature['geometry']['coordinates']
                    
                    earthquake = {
                        'dam_name': dam_name,
                        'event_time': datetime.fromtimestamp(props['time'] / 1000),
                        'magnitude': props['mag'],
                        'depth_km': coords[2],
                        'distance_km': self.calculate_distance(
                            location['lat'], location['lon'],
                            coords[1], coords[0]
                        ),
                        'place': props['place']
                    }
                    earthquakes.append(earthquake)
                
                return earthquakes
            
        except Exception as e:
            logger.error(f"Error collecting seismic data: {e}")
        
        return []
    
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two points in km"""
        from math import sin, cos, sqrt, atan2, radians
        
        R = 6371  # Earth's radius in km
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
    
    def save_to_database(self, data_type: str, data: Dict):
        """Save collected data to database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            if data_type == 'weather':
                # Save weather data to climate_data table
                cursor.execute("""
                    INSERT INTO climate_data (
                        dam_id, measurement_date, temperature_avg_celsius,
                        humidity_percent, wind_speed_kmh, precipitation_mm
                    ) VALUES (
                        (SELECT dam_id FROM dams WHERE dam_name = %s),
                        %s, %s, %s, %s, %s
                    )
                """, (
                    data['dam_name'], data['timestamp'], data['temperature'],
                    data['humidity'], data['wind_speed'] * 3.6, data['precipitation']
                ))
            
            elif data_type == 'extreme_weather':
                # Save extreme weather events
                cursor.execute("""
                    INSERT INTO extreme_weather_events (
                        dam_id, event_date, event_type, intensity_value,
                        intensity_unit, max_precipitation_mm
                    ) VALUES (
                        (SELECT dam_id FROM dams WHERE dam_name = %s),
                        %s, %s, %s, %s, %s
                    )
                """, (
                    data['dam_name'], data['timestamp'], data['event_type'],
                    data['precipitation_mm'], 'mm/3h', data['precipitation_mm']
                ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
    
    def run_realtime_collection(self):
        """Run real-time data collection for all dams"""
        logger.info("Starting real-time data collection")
        
        for dam_name in self.dam_locations:
            # Collect weather data
            weather_data = self.collect_weather_data(dam_name)
            if weather_data:
                self.save_to_database('weather', weather_data)
            
            # Check for extreme weather
            extreme_events = self.detect_extreme_weather(dam_name)
            for event in extreme_events:
                self.save_to_database('extreme_weather', event)
            
            # Collect seismic data
            earthquakes = self.collect_seismic_data(dam_name)
            logger.info(f"Found {len(earthquakes)} earthquakes near {dam_name}")
        
        logger.info("Real-time data collection completed")


if __name__ == "__main__":
    load_dotenv('config/.env')
    
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'database': os.getenv('DB_DATABASE', 'dam_monitoring'),
        'user': os.getenv('DB_USER', 'dam_user'),
        'password': os.getenv('DB_PASSWORD'),
        'port': os.getenv('DB_PORT', '5432')
    }
    
    collector = RealtimeAPICollector(db_config)
    collector.run_realtime_collection() 