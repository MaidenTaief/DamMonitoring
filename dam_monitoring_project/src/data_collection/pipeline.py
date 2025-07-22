import os
import json
import time
import logging
import requests
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')

# Create logs directory if it doesn't exist
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/dam_data_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DamDataCollector:
    """
    Main class for collecting dam data from multiple sources
    """
    
    def __init__(self, db_config: Dict[str, str]):
        """
        Initialize the data collector
        
        Args:
            db_config: Database connection configuration
        """
        self.db_config = db_config
        self.engine = create_engine(
            f"postgresql://{db_config['user']}:{db_config['password']}@"
            f"{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
        
        # Priority countries for initial data collection
        self.priority_countries = ['Norway', 'India', 'Bangladesh']
        
        # Data source configurations
        self.data_sources = {
            'usgs': {
                'base_url': 'https://waterdata.usgs.gov/nwis',
                'api_url': 'https://waterservices.usgs.gov/nwis/site/',
                'rate_limit': 1.0  # seconds between requests
            },
            'global_dam_watch': {
                'base_url': 'https://www.globaldamwatch.org',
                'data_url': 'https://www.globaldamwatch.org/data',
                'rate_limit': 2.0
            }
        }
        
        logger.info("Dam Data Collector initialized successfully")
    
    def setup_database_connection(self):
        """
        Create database connection
        
        Returns:
            Database connection object
        """
        try:
            conn = psycopg2.connect(
                host=self.db_config['host'],
                database=self.db_config['database'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                port=self.db_config['port']
            )
            logger.info("Database connection established")
            return conn
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def collect_goodd_data(self, country_filter: List[str] = None) -> pd.DataFrame:
        """
        Collect data from GOODD (Global Online Observatory of Dams)
        
        Args:
            country_filter: List of countries to filter data
            
        Returns:
            DataFrame with GOODD dam data
        """
        logger.info("Collecting GOODD data")
        
        try:
            # Sample data structure based on GOODD format
            sample_data = []
            
            # Add sample data for priority countries
            if not country_filter:
                country_filter = self.priority_countries
            
            for country in country_filter:
                if country == 'Norway':
                    sample_data.extend([
                        {
                            'dam_name': 'Alta Dam',
                            'country': 'Norway',
                            'latitude': 69.968,
                            'longitude': 23.272,
                            'height_m': 108,
                            'capacity_mcm': 2650,
                            'construction_year': 1987,
                            'dam_type': 'arch',
                            'primary_purpose': 'hydropower'
                        },
                        {
                            'dam_name': 'Svartisen Dam',
                            'country': 'Norway',
                            'latitude': 66.666,
                            'longitude': 13.778,
                            'height_m': 75,
                            'capacity_mcm': 1200,
                            'construction_year': 1975,
                            'dam_type': 'embankment',
                            'primary_purpose': 'hydropower'
                        }
                    ])
                elif country == 'India':
                    sample_data.extend([
                        {
                            'dam_name': 'Tehri Dam',
                            'country': 'India',
                            'latitude': 30.378,
                            'longitude': 78.478,
                            'height_m': 260,
                            'capacity_mcm': 4000,
                            'construction_year': 2006,
                            'dam_type': 'embankment',
                            'primary_purpose': 'multipurpose'
                        },
                        {
                            'dam_name': 'Bhakra Dam',
                            'country': 'India',
                            'latitude': 31.409,
                            'longitude': 76.432,
                            'height_m': 226,
                            'capacity_mcm': 9870,
                            'construction_year': 1963,
                            'dam_type': 'gravity',
                            'primary_purpose': 'multipurpose'
                        }
                    ])
                elif country == 'Bangladesh':
                    sample_data.extend([
                        {
                            'dam_name': 'Kaptai Dam',
                            'country': 'Bangladesh',
                            'latitude': 22.5,
                            'longitude': 92.3,
                            'height_m': 54,
                            'capacity_mcm': 11000,
                            'construction_year': 1962,
                            'dam_type': 'embankment',
                            'primary_purpose': 'hydropower'
                        }
                    ])
            
            df = pd.DataFrame(sample_data)
            logger.info(f"Collected {len(df)} dam records from GOODD")
            return df
            
        except Exception as e:
            logger.error(f"Error collecting GOODD data: {e}")
            return pd.DataFrame()
    
    def check_dam_exists(self, cursor, dam_name, country_code):
        """Check if dam already exists in database"""
        cursor.execute("""
            SELECT dam_id FROM dams 
            WHERE dam_name = %s AND country_code = %s
        """, (dam_name, country_code))
        return cursor.fetchone() is not None

    def process_and_insert_dams(self, df: pd.DataFrame, source_name: str) -> int:
        """
        Process and insert dam data into the database
        
        Args:
            df: DataFrame with dam data
            source_name: Name of the data source
            
        Returns:
            Number of records inserted
        """
        if df.empty:
            logger.warning("No data to insert")
            return 0
        
        try:
            conn = self.setup_database_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get region and source IDs
            region_map = self.get_region_mapping(cursor)
            source_id = self.get_source_id(cursor, source_name)
            
            inserted_count = 0
            
            for _, row in df.iterrows():
                try:
                    # Get region ID
                    region_id = region_map.get(row.get('country'))
                    country_code = self.get_country_code(row.get('country'))
                    dam_name = row.get('dam_name', '')
                    # Check for duplicates
                    if self.check_dam_exists(cursor, dam_name, country_code):
                        logger.info(f"Skipping duplicate dam: {dam_name} ({country_code})")
                        continue
                    
                    # Prepare dam data
                    dam_data = {
                        'dam_name': row.get('dam_name', ''),
                        'region_id': region_id,
                        'country_code': country_code,
                        'coordinates': f"POINT({row.get('longitude', 0)} {row.get('latitude', 0)})",
                        'height_meters': row.get('height_m'),
                        'normal_capacity_mcm': row.get('capacity_mcm'),
                        'construction_year': row.get('construction_year'),
                        'primary_purpose': row.get('primary_purpose'),
                        'construction_status': 'operational',
                        'primary_data_source_id': source_id,
                        'data_quality_score': 0.8
                    }
                    
                    # Insert dam record
                    insert_query = """
                        INSERT INTO dams (
                            dam_name, region_id, country_code, coordinates,
                            height_meters, normal_capacity_mcm, construction_year,
                            primary_purpose, construction_status, primary_data_source_id,
                            data_quality_score
                        ) VALUES (
                            %(dam_name)s, %(region_id)s, %(country_code)s, 
                            ST_GeomFromText(%(coordinates)s, 4326),
                            %(height_meters)s, %(normal_capacity_mcm)s, %(construction_year)s,
                            %(primary_purpose)s, %(construction_status)s, %(primary_data_source_id)s,
                            %(data_quality_score)s
                        )
                    """
                    
                    cursor.execute(insert_query, dam_data)
                    inserted_count += 1
                    
                except Exception as e:
                    logger.error(f"Error inserting dam record: {e}")
                    continue
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info(f"Successfully inserted {inserted_count} dam records")
            return inserted_count
            
        except Exception as e:
            logger.error(f"Error processing dam data: {e}")
            return 0
    
    def get_region_mapping(self, cursor) -> Dict[str, str]:
        """
        Get region name to ID mapping
        
        Args:
            cursor: Database cursor
            
        Returns:
            Dictionary mapping region names to IDs
        """
        cursor.execute("SELECT region_name, region_id FROM regions")
        return {row['region_name']: row['region_id'] for row in cursor.fetchall()}
    
    def get_source_id(self, cursor, source_name: str) -> str:
        """
        Get data source ID by name
        
        Args:
            cursor: Database cursor
            source_name: Name of the data source
            
        Returns:
            Source ID
        """
        cursor.execute(
            "SELECT source_id FROM data_sources WHERE source_name = %s",
            (source_name,)
        )
        result = cursor.fetchone()
        return result['source_id'] if result else None
    
    def get_country_code(self, country_name: str) -> str:
        """
        Get ISO country code from country name
        
        Args:
            country_name: Full country name
            
        Returns:
            ISO country code
        """
        country_codes = {
            'Norway': 'NO',
            'India': 'IN',
            'Bangladesh': 'BD',
            'United States': 'US'
        }
        return country_codes.get(country_name, 'XX')
    
    def create_sample_monitoring_stations(self):
        """
        Create sample monitoring stations for testing
        """
        logger.info("Creating sample monitoring stations")
        
        try:
            conn = self.setup_database_connection()
            cursor = conn.cursor()
            
            # Get some dam IDs
            cursor.execute("SELECT dam_id, dam_name FROM dams LIMIT 10")
            dams = cursor.fetchall()
            
            for dam_id, dam_name in dams:
                # Create a monitoring station for each dam
                station_data = {
                    'dam_id': dam_id,
                    'station_name': f"{dam_name} Monitoring Station",
                    'station_code': f"MS_{dam_id[:8]}",
                    'station_type': 'automated',
                    'coordinates': f"POINT({-74.0 + len(dam_name) * 0.1} {40.7 + len(dam_name) * 0.05})",
                    'parameters_monitored': ['water_level', 'flow_rate', 'temperature'],
                    'sensor_types': ['acoustic', 'hydraulic', 'thermal'],
                    'measurement_frequency': 'real-time',
                    'installation_date': '2020-01-01',
                    'status': 'active'
                }
                
                insert_query = """
                    INSERT INTO monitoring_stations (
                        dam_id, station_name, station_code, station_type,
                        coordinates, parameters_monitored, sensor_types,
                        measurement_frequency, installation_date, status
                    ) VALUES (
                        %(dam_id)s, %(station_name)s, %(station_code)s, %(station_type)s,
                        ST_GeomFromText(%(coordinates)s, 4326),
                        %(parameters_monitored)s, %(sensor_types)s,
                        %(measurement_frequency)s, %(installation_date)s, %(status)s
                    )
                """
                
                cursor.execute(insert_query, station_data)
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info("Sample monitoring stations created successfully")
            
        except Exception as e:
            logger.error(f"Error creating monitoring stations: {e}")
    
    def generate_sample_sensor_data(self, days: int = 30):
        """
        Generate sample sensor data for testing ML models
        
        Args:
            days: Number of days of data to generate
        """
        logger.info(f"Generating {days} days of sample sensor data")
        
        try:
            conn = self.setup_database_connection()
            cursor = conn.cursor()
            
            # Get monitoring stations
            cursor.execute("SELECT station_id FROM monitoring_stations LIMIT 10")
            stations = [row[0] for row in cursor.fetchall()]
            
            import random
            
            for station_id in stations:
                # Generate data for each day
                for day in range(days):
                    date = datetime.now() - timedelta(days=day)
                    
                    # Generate hourly readings
                    for hour in range(24):
                        timestamp = date.replace(hour=hour, minute=0, second=0)
                        
                        # Generate realistic sensor readings
                        readings = [
                            {
                                'station_id': station_id,
                                'parameter_type': 'water_level',
                                'measurement_value': 10.0 + random.uniform(-2, 2),
                                'unit': 'meters',
                                'measurement_timestamp': timestamp,
                                'quality_code': 'good',
                                'sensor_modality': 'hydraulic'
                            },
                            {
                                'station_id': station_id,
                                'parameter_type': 'flow_rate',
                                'measurement_value': 150.0 + random.uniform(-30, 30),
                                'unit': 'cms',
                                'measurement_timestamp': timestamp,
                                'quality_code': 'good',
                                'sensor_modality': 'hydraulic'
                            },
                            {
                                'station_id': station_id,
                                'parameter_type': 'temperature',
                                'measurement_value': 15.0 + random.uniform(-5, 5),
                                'unit': 'celsius',
                                'measurement_timestamp': timestamp,
                                'quality_code': 'good',
                                'sensor_modality': 'thermal'
                            }
                        ]
                        
                        for reading in readings:
                            insert_query = """
                                INSERT INTO sensor_readings (
                                    station_id, parameter_type, measurement_value,
                                    unit, measurement_timestamp, quality_code,
                                    sensor_modality
                                ) VALUES (
                                    %(station_id)s, %(parameter_type)s, %(measurement_value)s,
                                    %(unit)s, %(measurement_timestamp)s, %(quality_code)s,
                                    %(sensor_modality)s
                                )
                            """
                            cursor.execute(insert_query, reading)
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info("Sample sensor data generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating sensor data: {e}")
    
    def create_shm_sensors(self):
        """Create structural health monitoring sensors"""
        logger.info("Creating SHM sensors for dams")
        
        shm_sensor_configs = [
            {
                'type': 'strain_gauge',
                'locations': ['crest_center', 'base_upstream', 'base_downstream'],
                'sampling_rate': 100,
                'range_min': -3000,
                'range_max': 3000,
                'unit': 'microstrain'
            },
            {
                'type': 'fiber_optic',
                'locations': ['spillway', 'gallery', 'foundation'],
                'sampling_rate': 1000,
                'range_min': -5000,
                'range_max': 5000,
                'unit': 'microstrain'
            },
            {
                'type': 'accelerometer',
                'locations': ['crest_center', 'abutment_left', 'abutment_right'],
                'sampling_rate': 200,
                'range_min': -2,
                'range_max': 2,
                'unit': 'g'
            },
            {
                'type': 'tiltmeter',
                'locations': ['crest_center', 'downstream_face'],
                'sampling_rate': 1,
                'range_min': -10,
                'range_max': 10,
                'unit': 'degrees'
            },
            {
                'type': 'piezometer',
                'locations': ['foundation', 'embankment_core'],
                'sampling_rate': 1,
                'range_min': 0,
                'range_max': 1000,
                'unit': 'kPa'
            }
        ]
        
        try:
            conn = self.setup_database_connection()
            cursor = conn.cursor()
            
            # Get monitoring stations
            cursor.execute("SELECT station_id FROM monitoring_stations")
            stations = cursor.fetchall()
            
            for station_id, in stations:
                for sensor_config in shm_sensor_configs:
                    for location in sensor_config['locations']:
                        sensor_data = {
                            'station_id': station_id,
                            'sensor_type': sensor_config['type'],
                            'sensor_model': f"{sensor_config['type'].upper()}_2024",
                            'installation_location': location,
                            'measurement_range_min': sensor_config['range_min'],
                            'measurement_range_max': sensor_config['range_max'],
                            'sampling_rate_hz': sensor_config['sampling_rate'],
                            'calibration_factor': 1.0,
                            'installation_date': '2024-01-01',
                            'status': 'active'
                        }
                        
                        insert_query = """
                            INSERT INTO shm_sensors (
                                station_id, sensor_type, sensor_model, installation_location,
                                measurement_range_min, measurement_range_max, sampling_rate_hz,
                                calibration_factor, installation_date, status
                            ) VALUES (
                                %(station_id)s, %(sensor_type)s, %(sensor_model)s, %(installation_location)s,
                                %(measurement_range_min)s, %(measurement_range_max)s, %(sampling_rate_hz)s,
                                %(calibration_factor)s, %(installation_date)s, %(status)s
                            )
                        """
                        cursor.execute(insert_query, sensor_data)
            
            conn.commit()
            cursor.close()
            conn.close()
            logger.info("SHM sensors created successfully")
            
        except Exception as e:
            logger.error(f"Error creating SHM sensors: {e}")

    def generate_shm_sensor_data(self, days: int = 30):
        """Generate structural health monitoring sensor data"""
        logger.info(f"Generating {days} days of SHM sensor data")
        
        try:
            conn = self.setup_database_connection()
            cursor = conn.cursor()
            
            # Get SHM sensors
            cursor.execute("""
                SELECT s.sensor_id, s.sensor_type, s.measurement_range_min, 
                       s.measurement_range_max, ms.station_id
                FROM shm_sensors s
                JOIN monitoring_stations ms ON s.station_id = ms.station_id
                WHERE s.status = 'active'
            """)
            sensors = cursor.fetchall()
            
            import random
            import numpy as np
            
            for sensor_id, sensor_type, range_min, range_max, station_id in sensors:
                # Convert to float to avoid decimal/float errors
                range_min = float(range_min)
                range_max = float(range_max)
                for day in range(days):
                    date = datetime.now() - timedelta(days=day)
                    
                    # Generate hourly readings with realistic patterns
                    for hour in range(24):
                        timestamp = date.replace(hour=hour, minute=0, second=0)
                        
                        # Base value depends on sensor type
                        if sensor_type == 'strain_gauge':
                            base_value = 50 + 10 * np.sin(hour * np.pi / 12)  # Daily thermal cycle
                            noise = random.gauss(0, 5)
                        elif sensor_type == 'accelerometer':
                            base_value = 0.001  # Near zero for normal conditions
                            noise = random.gauss(0, 0.0005)
                        elif sensor_type == 'tiltmeter':
                            base_value = 0.1 + 0.05 * np.sin(hour * np.pi / 12)
                            noise = random.gauss(0, 0.01)
                        elif sensor_type == 'piezometer':
                            base_value = 100 + 20 * np.sin(day * np.pi / 15)  # Seasonal variation
                            noise = random.gauss(0, 2)
                        else:
                            base_value = (range_max - range_min) / 2
                            noise = random.gauss(0, (range_max - range_min) * 0.02)
                        
                        measurement_value = base_value + noise
                        
                        # Ensure within sensor range
                        measurement_value = max(min(measurement_value, float(range_max)), float(range_min))
                        
                        reading_data = {
                            'station_id': station_id,
                            'parameter_type': sensor_type,
                            'measurement_value': measurement_value,
                            'unit': self.get_sensor_unit(sensor_type),
                            'measurement_timestamp': timestamp,
                            'quality_code': 'good' if random.random() > 0.05 else 'fair',
                            'sensor_modality': 'structural'
                        }
                        
                        insert_query = """
                            INSERT INTO sensor_readings (
                                station_id, parameter_type, measurement_value,
                                unit, measurement_timestamp, quality_code, sensor_modality
                            ) VALUES (
                                %(station_id)s, %(parameter_type)s, %(measurement_value)s,
                                %(unit)s, %(measurement_timestamp)s, %(quality_code)s,
                                %(sensor_modality)s
                            )
                        """
                        cursor.execute(insert_query, reading_data)
            
            conn.commit()
            cursor.close()
            conn.close()
            logger.info("SHM sensor data generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating SHM sensor data: {e}")

    def get_sensor_unit(self, sensor_type: str) -> str:
        """Get unit for sensor type"""
        units = {
            'strain_gauge': 'microstrain',
            'fiber_optic': 'microstrain',
            'accelerometer': 'g',
            'tiltmeter': 'degrees',
            'piezometer': 'kPa',
            'displacement': 'mm',
            'crack_meter': 'mm'
        }
        return units.get(sensor_type, 'unit')

    def generate_comprehensive_dam_data(self):
        """Generate realistic dam inventory for all three countries (stub for future expansion)"""
        dam_templates = {
            'Norway': {
                'count': 335,  # Norway has ~335 large dams
                'height_range': (15, 200),
                'capacity_range': (10, 5000),
                'year_range': (1920, 2020),
                'types': ['embankment', 'concrete', 'arch', 'gravity'],
                'purposes': ['hydropower', 'water_supply', 'flood_control']
            },
            'India': {
                'count': 5334,  # India has 5000+ large dams
                'height_range': (15, 261),
                'capacity_range': (1, 12000),
                'year_range': (1900, 2023),
                'types': ['embankment', 'gravity', 'arch', 'rockfill'],
                'purposes': ['irrigation', 'hydropower', 'multipurpose', 'flood_control']
            },
            'Bangladesh': {
                'count': 320,  # Bangladesh has ~320 dams/embankments
                'height_range': (5, 60),
                'capacity_range': (1, 11000),
                'year_range': (1960, 2020),
                'types': ['embankment', 'concrete'],
                'purposes': ['flood_control', 'irrigation', 'water_supply']
            }
        }
        # Implementation to be added

    def run_full_collection(self):
        """
        Run the complete data collection pipeline
        """
        logger.info("Starting full data collection pipeline")
        
        try:
            # Step 1: Collect dam data from multiple sources
            logger.info("=== Collecting Dam Data ===")
            
            # Collect from GOODD
            goodd_data = self.collect_goodd_data(self.priority_countries)
            self.process_and_insert_dams(goodd_data, 'GOODD')
            
            # Step 2: Create monitoring infrastructure
            logger.info("=== Setting Up Monitoring Infrastructure ===")
            self.create_sample_monitoring_stations()
            
            # NEW: Add SHM sensors
            logger.info("=== Creating SHM Sensors ===")
            self.create_shm_sensors()
            
            # Step 3: Generate sample sensor data
            logger.info("=== Generating Sample Sensor Data ===")
            self.generate_sample_sensor_data(days=30)
            
            # NEW: Generate SHM sensor data
            logger.info("=== Generating SHM Sensor Data ===")
            self.generate_shm_sensor_data(days=30)
            
            logger.info("Data collection pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Error in data collection pipeline: {e}")
            raise


def main():
    """
    Main function to run the data collection pipeline
    """
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv('config/.env')
    
    # Database configuration
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'database': os.getenv('DB_DATABASE', 'dam_monitoring'),
        'user': os.getenv('DB_USER', 'dam_user'),
        'password': os.getenv('DB_PASSWORD', 'your_secure_password'),
        'port': os.getenv('DB_PORT', '5432')
    }
    
    # Initialize collector
    collector = DamDataCollector(db_config)
    
    # Run collection pipeline
    collector.run_full_collection()


if __name__ == "__main__":
    main()
