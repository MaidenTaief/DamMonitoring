#!/usr/bin/env python3
"""
Dam Monitoring Data Analysis - Generate Research Insights
"""

import sys
import os
sys.path.append('.')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from python.data_collection_pipeline import DamDataCollector
from dotenv import load_dotenv
import numpy as np
from datetime import datetime

def create_research_outputs():
    """Generate meaningful research outputs and visualizations"""
    
    # Load environment and setup database connection
    load_dotenv('config/.env')
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'database': os.getenv('DB_DATABASE', 'dam_monitoring'),
        'user': os.getenv('DB_USER', 'dam_user'),
        'password': os.getenv('DB_PASSWORD', 'your_secure_password'),
        'port': os.getenv('DB_PORT', '5432')
    }
    
    collector = DamDataCollector(db_config)
    conn = collector.setup_database_connection()
    
    # 1. INFRASTRUCTURE ANALYSIS
    print("Generating Infrastructure Analysis...")
    
    dam_query = """
    SELECT 
        dam_name, 
        region_name, 
        height_meters, 
        normal_capacity_mcm, 
        construction_year,
        primary_dam_type,
        longitude,
        latitude
    FROM dam_summary 
    ORDER BY height_meters DESC
    """
    
    dams_df = pd.read_sql(dam_query, conn)
    
    # Save infrastructure summary
    dams_df.to_csv('dam_monitoring_project/analysis/dam_infrastructure_analysis.csv', index=False)
    print(f"Exported {len(dams_df)} dam records to dam_monitoring_project/analysis/dam_infrastructure_analysis.csv")
    
    # 2. SENSOR DATA ANALYSIS  
    print("Generating Sensor Data Analysis...")
    
    sensor_query = """
    SELECT 
        ds.dam_name,
        ds.region_name,
        sr.parameter_type,
        sr.sensor_modality,
        sr.measurement_value,
        sr.measurement_timestamp,
        sr.quality_code
    FROM dam_summary ds
    JOIN monitoring_stations ms ON ds.dam_id = ms.dam_id
    JOIN sensor_readings sr ON ms.station_id = sr.station_id
    ORDER BY ds.region_name, sr.measurement_timestamp
    """
    
    sensor_df = pd.read_sql(sensor_query, conn)
    sensor_df['measurement_timestamp'] = pd.to_datetime(sensor_df['measurement_timestamp'])
    
    # Save sensor data sample (first 1000 records for analysis)
    sensor_df.head(1000).to_csv('dam_monitoring_project/analysis/sensor_data_sample.csv', index=False)
    print(f"Exported sample sensor data: {len(sensor_df)} total readings")
    
    # 3. STATISTICAL ANALYSIS
    print("Generating Statistical Analysis...")
    
    stats_summary = sensor_df.groupby(['region_name', 'parameter_type']).agg({
        'measurement_value': ['count', 'mean', 'std', 'min', 'max'],
        'quality_code': lambda x: (x == 'good').sum() / len(x) * 100
    }).round(2)
    
    stats_summary.to_csv('dam_monitoring_project/analysis/statistical_summary.csv')
    print("Generated statistical summary by region and parameter")
    
    # 4. TIME SERIES ANALYSIS
    print("Generating Time Series Analysis...")
    
    # Daily aggregates for each dam
    daily_data = sensor_df.set_index('measurement_timestamp').groupby(['dam_name', 'parameter_type']).resample('D').agg({
        'measurement_value': ['mean', 'std', 'count']
    }).round(2)
    
    daily_data.to_csv('dam_monitoring_project/analysis/daily_time_series.csv')
    print("Generated daily time series aggregates")
    
    # 5. MULTI-MODAL CORRELATION ANALYSIS
    print("Generating Multi-Modal Correlation Analysis...")
    
    # Pivot data for correlation analysis
    correlation_data = sensor_df.pivot_table(
        index=['dam_name', 'measurement_timestamp'],
        columns='parameter_type',
        values='measurement_value'
    ).reset_index()
    
    # Calculate correlations between parameters
    correlations = correlation_data[['flow_rate', 'temperature', 'water_level']].corr()
    correlations.to_csv('dam_monitoring_project/analysis/parameter_correlations.csv')
    print("Generated multi-parameter correlation matrix")
    
    # 6. RESEARCH DATASET FOR ML
    print("Generating ML-Ready Dataset...")
    
    # Create sequences for transformer training
    ml_query = """
    SELECT 
        ds.dam_name,
        ds.region_name,
        sr.parameter_type,
        sr.sensor_modality,
        sr.measurement_value,
        sr.measurement_timestamp,
        EXTRACT(hour FROM sr.measurement_timestamp) as hour,
        EXTRACT(dow FROM sr.measurement_timestamp) as day_of_week
    FROM dam_summary ds
    JOIN monitoring_stations ms ON ds.dam_id = ms.dam_id
    JOIN sensor_readings sr ON ms.station_id = sr.station_id
    ORDER BY ds.dam_name, sr.measurement_timestamp, sr.parameter_type
    """
    
    ml_df = pd.read_sql(ml_query, conn)
    ml_df.to_csv('dam_monitoring_project/analysis/ml_training_dataset.csv', index=False)
    print(f"Generated ML training dataset: {len(ml_df)} records")
    
    conn.close()
    
    # 7. GENERATE RESEARCH SUMMARY
    print("Generating Research Summary...")
    
    summary_stats = {
        'total_dams': len(dams_df),
        'countries': dams_df['region_name'].nunique(),
        'tallest_dam': dams_df.loc[dams_df['height_meters'].idxmax(), 'dam_name'],
        'largest_capacity': dams_df.loc[dams_df['normal_capacity_mcm'].idxmax(), 'dam_name'],
        'total_sensor_readings': len(sensor_df),
        'date_range_days': (sensor_df['measurement_timestamp'].max() - sensor_df['measurement_timestamp'].min()).days,
        'parameters_monitored': sensor_df['parameter_type'].nunique(),
        'sensor_modalities': sensor_df['sensor_modality'].nunique()
    }
    
    with open('dam_monitoring_project/analysis/research_summary.txt', 'w') as f:
        f.write("DAM MONITORING RESEARCH DATASET SUMMARY\n")
        f.write("="*50 + "\n\n")
        for key, value in summary_stats.items():
            f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print("Generated research summary")
    print("\nAll analyses complete! Check the dam_monitoring_project/analysis/ directory for outputs.")
    
    return summary_stats

if __name__ == "__main__":
    stats = create_research_outputs()
    print(f"\nDataset Summary:")
    for key, value in stats.items():
        print(f"   {key.replace('_', ' ').title()}: {value}") 