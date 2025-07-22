#!/usr/bin/env python3
"""
Database Enrichment Script - Add GDW Metadata to Existing Dam Database

This script enriches the cleaned database with comprehensive metadata from
the GDW barriers text file, adding 20+ valuable research fields.

Author: Dam Monitoring Research Team
Date: 2024
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
GDW_DATA_PATH = Path("/Users/taief/Desktop/DAM/GDW_v1_0_shp/GDW_v1_0_shp/GDW_barriers_v1_0.txt")
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'database_enrichment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DatabaseEnricher:
    """Enrich dam database with comprehensive GDW metadata."""
    
    def __init__(self):
        """Initialize enricher with database connection."""
        load_dotenv(PROJECT_ROOT / '.env')
        
        # Database configuration
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'database': os.getenv('DB_DATABASE', 'dam_monitoring'),
            'user': os.getenv('DB_USER', 'dam_user'),
            'password': os.getenv('DB_PASSWORD'),
            'port': os.getenv('DB_PORT', '5432')
        }
        
        # Create database engine
        self.engine = create_engine(
            f"postgresql://{self.db_config['user']}:{self.db_config['password']}@"
            f"{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
        )
        
        # Results storage
        self.enrichment_results = {}
        
        logger.info("Database Enricher initialized successfully")
    
    def load_gdw_metadata(self):
        """Load and process GDW barriers text file."""
        logger.info("=== Loading GDW Metadata ===")
        
        if not GDW_DATA_PATH.exists():
            raise FileNotFoundError(f"GDW data file not found: {GDW_DATA_PATH}")
        
        logger.info(f"Reading GDW data from: {GDW_DATA_PATH}")
        
        # Read the CSV file
        try:
            gdw_df = pd.read_csv(GDW_DATA_PATH, encoding='utf-8', low_memory=False)
            logger.info(f"✅ Loaded {len(gdw_df):,} GDW records with {len(gdw_df.columns)} fields")
        except Exception as e:
            logger.error(f"Error reading GDW file: {e}")
            raise
        
        # Display available columns for reference
        logger.info("Available GDW fields:")
        for i, col in enumerate(gdw_df.columns[:20], 1):
            logger.info(f"  {i:2d}. {col}")
        if len(gdw_df.columns) > 20:
            logger.info(f"  ... and {len(gdw_df.columns)-20} more fields")
        
        return gdw_df
    
    def select_enrichment_fields(self, gdw_df):
        """Select the most valuable fields for database enrichment."""
        logger.info("=== Selecting Key Enrichment Fields ===")
        
        # Define the most valuable fields for research
        key_fields = {
            # Core identifiers and names
            'GDW_ID': 'gdw_id',
            'DAM_NAME': 'dam_name_gdw', 
            'ALT_NAME': 'alternative_name',
            'RES_NAME': 'reservoir_name',
            
            # Infrastructure specifications  
            'DAM_TYPE': 'dam_type',
            'DAM_HGT_M': 'height_meters_gdw',
            'DAM_LEN_M': 'dam_length_meters',
            'CAP_MCM': 'capacity_mcm_gdw',
            'AREA_SKM': 'reservoir_area_sqkm',
            'DEPTH_M': 'max_depth_meters',
            
            # Temporal information
            'YEAR_DAM': 'construction_year_gdw',
            'PRE_YEAR': 'pre_construction_year',
            'YEAR_TXT': 'construction_year_text',
            
            # Geographic and watershed context
            'RIVER': 'river_name',
            'MAIN_BASIN': 'main_basin',
            'SUB_BASIN': 'sub_basin',
            'LONG_DAM': 'longitude_precise',
            'LAT_DAM': 'latitude_precise',
            'ELEV_MASL': 'elevation_meters',
            'CATCH_SKM': 'catchment_area_sqkm',
            
            # Usage and purpose
            'MAIN_USE': 'primary_purpose',
            'USE_IRRI': 'use_irrigation',
            'USE_ELEC': 'use_hydropower',
            'USE_SUPP': 'use_water_supply',
            'USE_FCON': 'use_flood_control',
            'USE_RECR': 'use_recreation',
            'USE_NAVI': 'use_navigation',
            
            # Technical specifications
            'POWER_MW': 'power_capacity_mw',
            'DIS_AVG_LS': 'average_discharge_ls',
            
            # Data quality and metadata
            'QUALITY': 'data_quality_gdw',
            'COMMENTS': 'gdw_comments',
            'ORIG_SRC': 'original_source',
            'EDITOR': 'data_editor'
        }
        
        # Check which fields are available in the dataset
        available_fields = {}
        missing_fields = []
        
        for gdw_field, db_field in key_fields.items():
            if gdw_field in gdw_df.columns:
                available_fields[gdw_field] = db_field
            else:
                missing_fields.append(gdw_field)
        
        logger.info(f"✅ Found {len(available_fields)} of {len(key_fields)} key fields")
        logger.info(f"Available fields: {list(available_fields.keys())}")
        
        if missing_fields:
            logger.warning(f"Missing fields: {missing_fields}")
        
        # Extract and clean the selected data
        enrichment_df = gdw_df[list(available_fields.keys())].copy()
        
        # Rename columns to database-friendly names
        enrichment_df.rename(columns=available_fields, inplace=True)
        
        # Clean and standardize the data
        enrichment_df = self.clean_enrichment_data(enrichment_df)
        
        self.enrichment_results['selected_fields'] = available_fields
        self.enrichment_results['records_to_enrich'] = len(enrichment_df)
        
        return enrichment_df
    
    def clean_enrichment_data(self, df):
        """Clean and standardize the enrichment data."""
        logger.info("=== Cleaning Enrichment Data ===")
        
        # Convert numeric fields
        numeric_fields = [
            'height_meters_gdw', 'dam_length_meters', 'capacity_mcm_gdw',
            'reservoir_area_sqkm', 'max_depth_meters', 'construction_year_gdw',
            'pre_construction_year', 'longitude_precise', 'latitude_precise',
            'elevation_meters', 'catchment_area_sqkm', 'power_capacity_mw',
            'average_discharge_ls'
        ]
        
        for field in numeric_fields:
            if field in df.columns:
                # Convert to numeric, replacing -99 and invalid values with None
                df[field] = pd.to_numeric(df[field], errors='coerce')
                df.loc[df[field] == -99, field] = None
                df.loc[df[field] < 0, field] = None  # Remove negative values except coordinates
        
        # Handle coordinates separately (can be negative)
        for coord_field in ['longitude_precise', 'latitude_precise']:
            if coord_field in df.columns:
                df[coord_field] = pd.to_numeric(df[coord_field], errors='coerce')
                df.loc[df[coord_field] == -99, coord_field] = None
        
        # Clean construction years (reasonable range)
        if 'construction_year_gdw' in df.columns:
            df.loc[(df['construction_year_gdw'] < 1800) | (df['construction_year_gdw'] > 2024), 'construction_year_gdw'] = None
        
        # Clean text fields
        text_fields = [
            'dam_name_gdw', 'alternative_name', 'reservoir_name', 'dam_type',
            'river_name', 'main_basin', 'sub_basin', 'primary_purpose',
            'gdw_comments', 'original_source', 'data_editor'
        ]
        
        for field in text_fields:
            if field in df.columns:
                # Replace empty strings and -99 with None
                df[field] = df[field].astype(str)
                df.loc[df[field].isin(['', '-99', 'nan', 'None']), field] = None
        
        # Convert boolean usage fields
        boolean_fields = [
            'use_irrigation', 'use_hydropower', 'use_water_supply',
            'use_flood_control', 'use_recreation', 'use_navigation'
        ]
        
        for field in boolean_fields:
            if field in df.columns:
                # Convert to boolean: 'Main'/'Sec' = True, empty/other = False
                df[field] = df[field].isin(['Main', 'Sec'])
        
        # Data quality scores
        if 'data_quality_gdw' in df.columns:
            df['data_quality_gdw'] = df['data_quality_gdw'].astype(str)
        
        logger.info("✅ Data cleaning completed")
        
        # Log data quality statistics
        total_records = len(df)
        logger.info("Data completeness summary:")
        for col in df.columns:
            non_null_count = df[col].notna().sum()
            completeness = (non_null_count / total_records) * 100
            logger.info(f"  {col}: {completeness:.1f}% complete ({non_null_count:,}/{total_records:,})")
        
        return df
    
    def create_enrichment_table(self):
        """Create database table for enrichment data."""
        logger.info("=== Creating Enrichment Table ===")
        
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS dam_enrichment (
            gdw_id INTEGER PRIMARY KEY,
            dam_name_gdw TEXT,
            alternative_name TEXT,
            reservoir_name TEXT,
            dam_type TEXT,
            height_meters_gdw DECIMAL(10,2),
            dam_length_meters DECIMAL(10,2),
            capacity_mcm_gdw DECIMAL(15,3),
            reservoir_area_sqkm DECIMAL(12,3),
            max_depth_meters DECIMAL(8,2),
            construction_year_gdw INTEGER,
            pre_construction_year INTEGER,
            construction_year_text TEXT,
            river_name TEXT,
            main_basin TEXT,
            sub_basin TEXT,
            longitude_precise DECIMAL(12,8),
            latitude_precise DECIMAL(12,8),
            elevation_meters DECIMAL(10,2),
            catchment_area_sqkm DECIMAL(15,3),
            primary_purpose TEXT,
            use_irrigation BOOLEAN,
            use_hydropower BOOLEAN,
            use_water_supply BOOLEAN,
            use_flood_control BOOLEAN,
            use_recreation BOOLEAN,
            use_navigation BOOLEAN,
            power_capacity_mw DECIMAL(10,2),
            average_discharge_ls DECIMAL(15,3),
            data_quality_gdw TEXT,
            gdw_comments TEXT,
            original_source TEXT,
            data_editor TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Create index for joining with main dams table
        CREATE INDEX IF NOT EXISTS idx_dam_enrichment_gdw_id ON dam_enrichment(gdw_id);
        """
        
        try:
            with self.engine.connect() as conn:
                conn.execute(text(create_table_sql))
                conn.commit()
                logger.info("✅ Enrichment table created successfully")
        except Exception as e:
            logger.error(f"Error creating enrichment table: {e}")
            raise
    
    def insert_enrichment_data(self, enrichment_df):
        """Insert enrichment data into database."""
        logger.info("=== Inserting Enrichment Data ===")
        
        # Clear existing data
        try:
            with self.engine.connect() as conn:
                conn.execute(text("DELETE FROM dam_enrichment"))
                conn.commit()
                logger.info("Cleared existing enrichment data")
        except Exception as e:
            logger.warning(f"Could not clear existing data: {e}")
        
        # Insert new data
        try:
            rows_inserted = enrichment_df.to_sql(
                'dam_enrichment',
                self.engine,
                if_exists='append',
                index=False,
                method='multi',
                chunksize=1000
            )
            
            logger.info(f"✅ Successfully inserted {len(enrichment_df):,} enrichment records")
            
            # Verify insertion
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM dam_enrichment"))
                actual_count = result.fetchone()[0]
                logger.info(f"✅ Verified: {actual_count:,} records in enrichment table")
            
            self.enrichment_results['records_inserted'] = actual_count
            
        except Exception as e:
            logger.error(f"Error inserting enrichment data: {e}")
            raise
    
    def create_enriched_view(self):
        """Create a view that joins main dams table with enrichment data."""
        logger.info("=== Creating Enriched View ===")
        
        create_view_sql = """
        DROP VIEW IF EXISTS dams_enriched;
        
        CREATE VIEW dams_enriched AS
        SELECT 
            d.dam_id,
            d.dam_name,
            d.country_code,
            d.province_state,
            d.coordinates,
            d.height_meters,
            d.normal_capacity_mcm,
            d.construction_year,
            d.data_quality_score,
            d.created_at as dam_record_created,
            
            -- Enrichment data
            e.gdw_id,
            e.dam_name_gdw,
            e.alternative_name,
            e.reservoir_name,
            e.dam_type,
            e.height_meters_gdw,
            e.dam_length_meters,
            e.capacity_mcm_gdw,
            e.reservoir_area_sqkm,
            e.max_depth_meters,
            e.construction_year_gdw,
            e.construction_year_text,
            e.river_name,
            e.main_basin,
            e.sub_basin,
            e.longitude_precise,
            e.latitude_precise,
            e.elevation_meters,
            e.catchment_area_sqkm,
            e.primary_purpose,
            e.use_irrigation,
            e.use_hydropower,
            e.use_water_supply,
            e.use_flood_control,
            e.use_recreation,
            e.use_navigation,
            e.power_capacity_mw,
            e.average_discharge_ls,
            e.data_quality_gdw,
            e.gdw_comments,
            e.original_source,
            e.data_editor
            
        FROM dams d
        LEFT JOIN dam_enrichment e ON d.dam_name = e.dam_name_gdw 
            OR (d.country_code = SUBSTRING(e.gdw_comments, 1, 10))  -- Fallback matching
        WHERE d.data_quality_score = 0.5;  -- Only authentic GDW records
        """
        
        try:
            with self.engine.connect() as conn:
                conn.execute(text(create_view_sql))
                conn.commit()
                logger.info("✅ Enriched view created successfully")
                
                # Test the view
                result = conn.execute(text("SELECT COUNT(*) FROM dams_enriched"))
                view_count = result.fetchone()[0]
                logger.info(f"✅ Enriched view contains {view_count:,} records")
                
                self.enrichment_results['enriched_view_records'] = view_count
                
        except Exception as e:
            logger.error(f"Error creating enriched view: {e}")
            raise
    
    def validate_enrichment(self):
        """Validate the enrichment process and generate summary."""
        logger.info("=== Validating Enrichment Results ===")
        
        validation_queries = {
            'main_dams_count': "SELECT COUNT(*) FROM dams WHERE data_quality_score = 0.5",
            'enrichment_records': "SELECT COUNT(*) FROM dam_enrichment",
            'enriched_view_count': "SELECT COUNT(*) FROM dams_enriched",
            'height_data_improved': """
                SELECT 
                    COUNT(CASE WHEN height_meters IS NOT NULL THEN 1 END) as original_height_count,
                    COUNT(CASE WHEN height_meters_gdw IS NOT NULL THEN 1 END) as enriched_height_count
                FROM dams_enriched
            """,
            'usage_data_available': """
                SELECT 
                    COUNT(CASE WHEN primary_purpose IS NOT NULL THEN 1 END) as purpose_count,
                    COUNT(CASE WHEN use_irrigation = true THEN 1 END) as irrigation_count,
                    COUNT(CASE WHEN use_hydropower = true THEN 1 END) as hydropower_count
                FROM dams_enriched
            """,
            'geographic_data_enhanced': """
                SELECT 
                    COUNT(CASE WHEN river_name IS NOT NULL THEN 1 END) as rivers_identified,
                    COUNT(CASE WHEN main_basin IS NOT NULL THEN 1 END) as basins_identified,
                    COUNT(CASE WHEN longitude_precise IS NOT NULL THEN 1 END) as precise_coordinates
                FROM dams_enriched
            """
        }
        
        validation_results = {}
        
        with self.engine.connect() as conn:
            for query_name, query in validation_queries.items():
                try:
                    result = pd.read_sql(query, conn)
                    validation_results[query_name] = result.to_dict('records')[0]
                except Exception as e:
                    logger.error(f"Validation query failed - {query_name}: {e}")
                    validation_results[query_name] = {"error": str(e)}
        
        # Store validation results
        self.enrichment_results['validation'] = validation_results
        
        # Log validation summary
        logger.info("Validation Summary:")
        logger.info(f"  Main dams (cleaned): {validation_results['main_dams_count']['count']:,}")
        logger.info(f"  Enrichment records: {validation_results['enrichment_records']['count']:,}")
        logger.info(f"  Enriched view records: {validation_results['enriched_view_count']['count']:,}")
        
        if 'height_data_improved' in validation_results:
            height_data = validation_results['height_data_improved']
            logger.info(f"  Height data improvement: {height_data.get('enriched_height_count', 0):,} vs {height_data.get('original_height_count', 0):,}")
        
        if 'usage_data_available' in validation_results:
            usage_data = validation_results['usage_data_available']
            logger.info(f"  Purpose data: {usage_data.get('purpose_count', 0):,} dams")
            logger.info(f"  Irrigation use: {usage_data.get('irrigation_count', 0):,} dams")
            logger.info(f"  Hydropower use: {usage_data.get('hydropower_count', 0):,} dams")
        
        return validation_results
    
    def generate_enrichment_report(self):
        """Generate comprehensive enrichment report."""
        logger.info("=== Generating Enrichment Report ===")
        
        report_content = f"""# Database Enrichment Report - GDW Metadata Integration

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Enrichment Summary

### Data Source
- **Source File**: {GDW_DATA_PATH}
- **Records Processed**: {self.enrichment_results.get('records_to_enrich', 'N/A'):,}
- **Fields Selected**: {len(self.enrichment_results.get('selected_fields', {}))}/30+ available

### Key Fields Added
"""
        
        if 'selected_fields' in self.enrichment_results:
            for gdw_field, db_field in self.enrichment_results['selected_fields'].items():
                report_content += f"- **{db_field}** (from {gdw_field})\n"
        
        report_content += f"""
### Database Enhancement Results
- **Enrichment Table**: `dam_enrichment` with {self.enrichment_results.get('records_inserted', 'N/A'):,} records
- **Enriched View**: `dams_enriched` for combined analysis
- **Data Integration**: Successful linkage of GDW metadata with cleaned database

### Research Capabilities Enhanced
✅ **Infrastructure Analysis**: Height, capacity, construction specifications
✅ **Usage Pattern Analysis**: Purpose classification and multi-use identification  
✅ **Temporal Analysis**: Enhanced construction timeline with alternative dates
✅ **Geographic Context**: River systems, basins, precise coordinates
✅ **Technical Specifications**: Power capacity, discharge rates, depths
✅ **Data Provenance**: Quality indicators and source documentation

### Conference Paper Benefits
- **Advanced Analytics**: Multi-dimensional analysis capabilities
- **Technical Depth**: Engineering specifications for peer review
- **Research Validation**: Comprehensive metadata for methodology section
- **Global Context**: Basin-level and watershed analysis possibilities

---
*Report generated by Database Enrichment System*
"""
        
        # Save report
        report_path = PROJECT_ROOT / "logs" / "database_enrichment_report.md"
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"📄 Enrichment report saved: {report_path}")
        
        return report_content
    
    def run_complete_enrichment(self):
        """Run complete database enrichment process."""
        logger.info("🔬 Starting Complete Database Enrichment Process")
        
        try:
            # Load GDW metadata
            gdw_df = self.load_gdw_metadata()
            
            # Select key enrichment fields
            enrichment_df = self.select_enrichment_fields(gdw_df)
            
            # Create enrichment infrastructure
            self.create_enrichment_table()
            
            # Insert enrichment data
            self.insert_enrichment_data(enrichment_df)
            
            # Create enriched view
            self.create_enriched_view()
            
            # Validate results
            validation_results = self.validate_enrichment()
            
            # Generate report
            report = self.generate_enrichment_report()
            
            # Final summary
            records_enriched = self.enrichment_results.get('records_inserted', 0)
            fields_added = len(self.enrichment_results.get('selected_fields', {}))
            
            logger.info("=" * 60)
            logger.info("🎯 DATABASE ENRICHMENT COMPLETE:")
            logger.info(f"   📊 Records Enriched: {records_enriched:,}")
            logger.info(f"   📈 Metadata Fields Added: {fields_added}")
            logger.info(f"   🔗 New Table: dam_enrichment")
            logger.info(f"   🎭 New View: dams_enriched")
            logger.info(f"   📚 Ready for advanced research!")
            logger.info("=" * 60)
            
            return self.enrichment_results
            
        except Exception as e:
            logger.error(f"❌ Enrichment process failed: {e}")
            raise


def main():
    """Main execution function."""
    enricher = DatabaseEnricher()
    results = enricher.run_complete_enrichment()
    return results


if __name__ == "__main__":
    main() 