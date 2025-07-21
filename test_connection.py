#!/usr/bin/env python3
"""
Test script to verify dam monitoring database setup
"""

import sys
import os
sys.path.append('.')
from python.data_collection_pipeline import DamDataCollector
from dotenv import load_dotenv

def test_database_connection():
    """Test database connection and basic operations"""
    
    # Load environment variables
    load_dotenv('config/.env')
    
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'database': os.getenv('DB_DATABASE', 'dam_monitoring'),
        'user': os.getenv('DB_USER', 'dam_user'),
        'password': os.getenv('DB_PASSWORD', 'your_secure_password'),
        'port': os.getenv('DB_PORT', '5432')
    }
    
    try:
        # Test connection
        collector = DamDataCollector(db_config)
        print("Database connection successful!")
        
        # Test basic database operations
        conn = collector.setup_database_connection()
        cursor = conn.cursor()
        
        # Check tables exist
        cursor.execute("SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public'")
        table_count = cursor.fetchone()[0]
        print(f"Found {table_count} tables in database")
        
        # Check regions data
        cursor.execute("SELECT count(*) FROM regions")
        region_count = cursor.fetchone()[0]
        print(f"Found {region_count} regions")
        
        # Check data sources
        cursor.execute("SELECT count(*) FROM data_sources")
        source_count = cursor.fetchone()[0]
        print(f"Found {source_count} data sources")
        
        cursor.close()
        conn.close()
        
        print("\nDatabase setup verification complete!")
        print("Your dam monitoring system is ready for use!")
        
        return True
        
    except Exception as e:
        print(f"Database test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_database_connection()
    exit(0 if success else 1)
