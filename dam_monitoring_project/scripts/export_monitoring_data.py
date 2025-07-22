#!/usr/bin/env python3
"""
Export monitoring stations, SHM sensors, and sensor readings to CSV files.
"""
import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

load_dotenv('dam_monitoring_project/config/.env')
engine = create_engine(
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@"
    f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_DATABASE')}"
)

# Export monitoring stations
df_stations = pd.read_sql("SELECT * FROM monitoring_stations", engine)
df_stations.to_csv("monitoring_stations.csv", index=False)
print("Exported monitoring_stations.csv")

# Export SHM sensors
df_sensors = pd.read_sql("SELECT * FROM shm_sensors", engine)
df_sensors.to_csv("shm_sensors.csv", index=False)
print("Exported shm_sensors.csv")

# Export recent sensor readings (limit to 10,000 for size)
df_readings = pd.read_sql("SELECT * FROM sensor_readings ORDER BY measurement_timestamp DESC LIMIT 10000", engine)
df_readings.to_csv("sensor_readings_sample.csv", index=False)
print("Exported sensor_readings_sample.csv") 