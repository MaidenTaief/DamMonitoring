#!/usr/bin/env python3
"""
Check the status of your dam monitoring project
"""

import os
import sys

def check_project_structure():
    """Check if all required directories and files exist"""
    
    required_dirs = [
        'config',
        'python', 
        'sql',
        'logs',
        'analysis',
        'analysis/exports',
        'analysis/figures',
        'docs',
        'data'
    ]
    
    required_files = [
        'config/.env',
        'python/data_collection_pipeline.py',
        'sql/dam_schema.sql',
        'requirements.txt'
    ]
    
    print("🔍 Checking Dam Monitoring Project Structure...")
    print("=" * 50)
    
    # Check directories
    print("\n📁 Directory Structure:")
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"  ✅ {directory}/")
        else:
            print(f"  ❌ {directory}/ (MISSING)")
    
    # Check files  
    print("\n📄 Required Files:")
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"  ✅ {file_path} ({size} bytes)")
        else:
            print(f"  ❌ {file_path} (MISSING)")
    
    # Check database connection
    print("\n🔌 Database Connection:")
    try:
        from dotenv import load_dotenv
        load_dotenv('config/.env')
        
        db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'database': os.getenv('DB_DATABASE', 'dam_monitoring'),
            'user': os.getenv('DB_USER', 'dam_user'),
            'password': os.getenv('DB_PASSWORD'),
            'port': os.getenv('DB_PORT', '5432')
        }
        
        import psycopg2
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM dams")
        dam_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT count(*) FROM monitoring_stations")
        station_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT count(*) FROM sensor_readings")
        reading_count = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        print(f"  ✅ Database connected!")
        print(f"    📊 {dam_count} dams, {station_count} stations, {reading_count} readings")
        
    except Exception as e:
        print(f"  ❌ Database connection failed: {e}")
    
    print("\n" + "=" * 50)
    print("✨ Status check complete!")
    
    return True

if __name__ == "__main__":
    check_project_structure() 