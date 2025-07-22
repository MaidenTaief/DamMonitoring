#!/usr/bin/env python3
"""
Process Global Dam Watch (GDW) Shapefiles and Bulk Insert into PostgreSQL
Production-ready script for dam monitoring database integration
"""
import os
import logging
import geopandas as gpd
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from sqlalchemy import create_engine
from dotenv import load_dotenv
from pathlib import Path

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('../logs/process_gdw.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Utility functions ---
def map_quality_score(q):
    mapping = {1: 1.0, 2: 0.9, 3: 0.7, 4: 0.5, 5: 0.2}
    try:
        return mapping.get(int(q), 0.5)
    except Exception:
        return 0.5

def safe_float(val):
    try:
        f_val = float(val)
        # Convert GDW sentinel values to NULL
        if f_val == -99.0 or f_val < 0:
            return None
        return f_val
    except Exception:
        return None

def safe_int(val):
    try:
        i_val = int(val)
        # Convert GDW sentinel values to NULL
        if i_val == -99 or i_val < 0:
            return None
        # For construction years, filter out unreasonable values (before 1800 or after 2050)
        # This handles ancient dams with years like 284 that violate DB constraints
        if i_val < 1800 or i_val > 2050:
            return None
        return i_val
    except Exception:
        return None

# --- Main processing class ---
class GDWProcessor:
    def __init__(self, db_config, gdw_dir):
        self.db_config = db_config
        self.gdw_dir = Path(gdw_dir)
        self.engine = create_engine(
            f"postgresql://{db_config['user']}:{db_config['password']}@"
            f"{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )

    def process_barriers(self):
        barriers_shp = self.gdw_dir / "GDW_barriers_v1_0.shp"
        logger.info(f"Processing barriers shapefile: {barriers_shp}")
        
        # Check if file exists before reading
        if not barriers_shp.exists():
            logger.error(f"Barriers shapefile not found: {barriers_shp}")
            logger.info(f"Directory contents: {list(self.gdw_dir.glob('*'))}")
            raise FileNotFoundError(f"Barriers shapefile not found: {barriers_shp}")
        
        logger.info(f"File found, reading barriers shapefile: {barriers_shp}")
        gdf = gpd.read_file(barriers_shp)
        gdf = gdf.to_crs(epsg=4326)
        
        # Map GDW columns to database schema
        mapped_gdf = pd.DataFrame()
        
        # Essential columns for database - handle NULL dam names
        def get_dam_name(row):
            # Try dam name first
            if pd.notna(row['DAM_NAME']) and str(row['DAM_NAME']).strip():
                return str(row['DAM_NAME']).strip()
            # Fallback to reservoir name
            elif pd.notna(row['RES_NAME']) and str(row['RES_NAME']).strip():
                return str(row['RES_NAME']).strip()
            # Fallback to alternative name
            elif pd.notna(row['ALT_NAME']) and str(row['ALT_NAME']).strip():
                return str(row['ALT_NAME']).strip()
            # Last resort - use location-based name
            else:
                city = str(row['NEAR_CITY']) if pd.notna(row['NEAR_CITY']) else 'Unknown'
                return f"Dam near {city}"
        
        mapped_gdf['dam_name'] = gdf.apply(get_dam_name, axis=1)
        # Convert alternative names to PostgreSQL array format - skip arrays for now
        mapped_gdf['alternative_names'] = None  # Will add array support later
        # Truncate country names to fit database VARCHAR(10) limit
        mapped_gdf['country_code'] = gdf['COUNTRY'].apply(lambda x: str(x)[:10] if pd.notna(x) else None)
        mapped_gdf['province_state'] = gdf['ADMIN_UNIT']
        mapped_gdf['nearest_city'] = gdf['NEAR_CITY']
        
        # Coordinate geometry (keep as PostGIS geometry)
        mapped_gdf['coordinates'] = gdf['geometry']
        
        # Physical characteristics
        mapped_gdf['construction_year'] = gdf['YEAR_DAM'].apply(safe_int)
        mapped_gdf['height_meters'] = gdf['DAM_HGT_M'].apply(safe_float)
        mapped_gdf['length_meters'] = gdf['DAM_LEN_M'].apply(safe_float)
        mapped_gdf['elevation_masl'] = gdf['ELEV_MASL'].apply(safe_float)
        
        # Reservoir characteristics
        mapped_gdf['reservoir_name'] = gdf['RES_NAME']
        mapped_gdf['normal_capacity_mcm'] = gdf['CAP_MCM'].apply(safe_float)
        mapped_gdf['catchment_area_km2'] = gdf['CATCH_SKM'].apply(safe_float)
        mapped_gdf['reservoir_surface_area_km2'] = gdf['AREA_SKM'].apply(safe_float)
        
        # Operational information
        mapped_gdf['primary_purpose'] = gdf['MAIN_USE']
        mapped_gdf['installed_capacity_mw'] = gdf['POWER_MW'].apply(safe_float)
        
        # Data quality and source
        mapped_gdf['data_quality_score'] = gdf['QUALITY'].apply(map_quality_score)
        mapped_gdf['last_verified'] = pd.Timestamp.now()
        
        # Note: Additional GDW fields like dam_type, river, main_basin, sub_basin
        # are not part of the database schema so we skip them
        
        # Convert to GeoDataFrame to maintain geometry
        mapped_gdf = gpd.GeoDataFrame(mapped_gdf, geometry='coordinates', crs='EPSG:4326')
        
        logger.info(f"Mapped {len(mapped_gdf)} barrier records with {len(mapped_gdf.columns)} columns")
        return mapped_gdf

    def process_reservoirs(self):
        reservoirs_shp = self.gdw_dir / "GDW_reservoirs_v1_0.shp"
        logger.info(f"Processing reservoirs shapefile: {reservoirs_shp}")
        
        # Check if file exists before reading
        if not reservoirs_shp.exists():
            logger.error(f"Reservoirs shapefile not found: {reservoirs_shp}")
            logger.info(f"Directory contents: {list(self.gdw_dir.glob('*'))}")
            raise FileNotFoundError(f"Reservoirs shapefile not found: {reservoirs_shp}")
        
        logger.info(f"File found, reading reservoirs shapefile: {reservoirs_shp}")
        gdf = gpd.read_file(reservoirs_shp)
        gdf = gdf.to_crs(epsg=4326)
        
        # Map GDW reservoir columns to database schema
        mapped_gdf = pd.DataFrame()
        
        # Essential columns for database - handle NULL dam names  
        def get_dam_name(row):
            # Try dam name first
            if pd.notna(row['DAM_NAME']) and str(row['DAM_NAME']).strip():
                return str(row['DAM_NAME']).strip()
            # Fallback to reservoir name
            elif pd.notna(row['RES_NAME']) and str(row['RES_NAME']).strip():
                return str(row['RES_NAME']).strip()
            # Fallback to alternative name
            elif pd.notna(row['ALT_NAME']) and str(row['ALT_NAME']).strip():
                return str(row['ALT_NAME']).strip()
            # Last resort - use location-based name
            else:
                city = str(row['NEAR_CITY']) if pd.notna(row['NEAR_CITY']) else 'Unknown'
                return f"Dam near {city}"
        
        mapped_gdf['dam_name'] = gdf.apply(get_dam_name, axis=1)
        # Convert alternative names to PostgreSQL array format - skip arrays for now  
        mapped_gdf['alternative_names'] = None  # Will add array support later
        # Truncate country names to fit database VARCHAR(10) limit
        mapped_gdf['country_code'] = gdf['COUNTRY'].apply(lambda x: str(x)[:10] if pd.notna(x) else None)
        mapped_gdf['province_state'] = gdf['ADMIN_UNIT']
        mapped_gdf['nearest_city'] = gdf['NEAR_CITY']
        
        # Coordinate geometry (keep as PostGIS geometry)
        mapped_gdf['coordinates'] = gdf['geometry']
        
        # Physical characteristics
        mapped_gdf['construction_year'] = gdf['YEAR_DAM'].apply(safe_int)
        mapped_gdf['height_meters'] = gdf['DAM_HGT_M'].apply(safe_float)
        mapped_gdf['length_meters'] = gdf['DAM_LEN_M'].apply(safe_float)
        mapped_gdf['elevation_masl'] = gdf['ELEV_MASL'].apply(safe_float)
        
        # Reservoir characteristics
        mapped_gdf['reservoir_name'] = gdf['RES_NAME']
        mapped_gdf['normal_capacity_mcm'] = gdf['CAP_MCM'].apply(safe_float)
        mapped_gdf['catchment_area_km2'] = gdf['CATCH_SKM'].apply(safe_float)
        mapped_gdf['reservoir_surface_area_km2'] = gdf['AREA_SKM'].apply(safe_float)
        
        # Operational information
        mapped_gdf['primary_purpose'] = gdf['MAIN_USE']
        mapped_gdf['installed_capacity_mw'] = gdf['POWER_MW'].apply(safe_float)
        
        # Data quality and source
        mapped_gdf['data_quality_score'] = gdf['QUALITY'].apply(map_quality_score)
        mapped_gdf['last_verified'] = pd.Timestamp.now()
        
        # Note: Additional GDW fields like dam_type, river, main_basin, sub_basin
        # are not part of the database schema so we skip them
        
        # Convert to GeoDataFrame to maintain geometry
        mapped_gdf = gpd.GeoDataFrame(mapped_gdf, geometry='coordinates', crs='EPSG:4326')
        
        logger.info(f"Mapped {len(mapped_gdf)} reservoir records with {len(mapped_gdf.columns)} columns")
        return mapped_gdf

    def bulk_insert(self, gdf, table, unique_col='dam_name'):
        """
        Insert GeoDataFrame into PostGIS table using proper geometry handling
        """
        logger.info(f"Starting bulk insert of {len(gdf)} records to {table}")
        logger.info(f"Columns to insert: {list(gdf.columns)}")
        
        # Convert geometry to WKT format for PostgreSQL
        try:
            # Create a copy to avoid modifying original
            gdf_copy = gdf.copy()
            
            # Convert geometry column to WKT string that PostGIS can understand
            if 'coordinates' in gdf_copy.columns:
                gdf_copy['coordinates'] = gdf_copy['coordinates'].apply(lambda geom: geom.wkt if geom else None)
                logger.info("Converted geometry to WKT format")
            
            # Use basic to_sql method instead of geopandas since we converted geometry to text
            import pandas as pd
            regular_df = pd.DataFrame(gdf_copy)  # Convert to regular DataFrame
            
            # Insert records 
            regular_df.to_sql(
                name=table,
                con=self.engine,
                if_exists='append',
                index=False,
                chunksize=1000
            )
            
            logger.info(f"Successfully inserted {len(gdf)} records to {table}")
            return len(gdf), 0  # inserted, skipped
            
        except Exception as e:
            logger.error(f"Bulk insert failed: {e}")
            logger.error(f"Error type: {type(e)}")
            return 0, len(gdf)
    


    def validate_import(self, table):
        with self.engine.connect() as conn:
            try:
                logger.info("=== Import Validation Results ===")
                
                # Total count
                total_count = pd.read_sql(f"SELECT COUNT(*) as total_records FROM {table}", conn)
                logger.info(f"Total records in {table}: {total_count['total_records'].iloc[0]}")
                
                # Country distribution
                logger.info("\nCountry distribution:")
                country_dist = pd.read_sql(f"SELECT country_code, COUNT(*) as count FROM {table} GROUP BY country_code ORDER BY count DESC LIMIT 10", conn)
                logger.info(country_dist.to_string(index=False))
                
                # Quality score distribution
                logger.info("\nQuality score distribution:")
                quality_dist = pd.read_sql(f"SELECT data_quality_score, COUNT(*) as count FROM {table} GROUP BY data_quality_score ORDER BY data_quality_score", conn)
                logger.info(quality_dist.to_string(index=False))
                
                # Coordinate range using PostGIS functions
                logger.info("\nCoordinate range:")
                coord_range = pd.read_sql(f"""
                    SELECT 
                        MIN(ST_X(coordinates)) as min_longitude,
                        MAX(ST_X(coordinates)) as max_longitude,
                        MIN(ST_Y(coordinates)) as min_latitude,
                        MAX(ST_Y(coordinates)) as max_latitude
                    FROM {table}
                    WHERE coordinates IS NOT NULL
                """, conn)
                logger.info(coord_range.to_string(index=False))
                
                # Dam name examples
                logger.info("\nSample dam names:")
                sample_names = pd.read_sql(f"SELECT dam_name FROM {table} WHERE dam_name IS NOT NULL LIMIT 5", conn)
                logger.info(sample_names.to_string(index=False))
                
            except Exception as e:
                logger.error(f"Validation error: {e}")
                logger.info("Some validation queries failed, but data may still be inserted correctly")

    def run(self):
        logger.info("=== Starting GDW Data Processing ===")
        
        # Process barriers (point geometries)
        logger.info("Processing GDW barriers...")
        gdf_barriers = self.process_barriers()
        inserted_barriers, skipped_barriers = self.bulk_insert(gdf_barriers, "dams")
        logger.info(f"Barriers processed: {inserted_barriers} inserted, {skipped_barriers} skipped")
        
        # Process reservoirs (polygon geometries converted to points)
        logger.info("Processing GDW reservoirs...")
        gdf_reservoirs = self.process_reservoirs()
        
        # Convert polygon geometries to centroids for the dams table
        if not gdf_reservoirs.empty:
            logger.info("Converting reservoir polygons to centroid points...")
            gdf_reservoirs['coordinates'] = gdf_reservoirs['coordinates'].centroid
            logger.info(f"Converted {len(gdf_reservoirs)} reservoir polygons to point centroids")
        
        inserted_reservoirs, skipped_reservoirs = self.bulk_insert(gdf_reservoirs, "dams")
        logger.info(f"Reservoirs processed: {inserted_reservoirs} inserted, {skipped_reservoirs} skipped")
        
        # Final validation
        logger.info("=== Final Import Validation ===")
        self.validate_import("dams")
        
        total_inserted = inserted_barriers + inserted_reservoirs
        total_skipped = skipped_barriers + skipped_reservoirs
        logger.info(f"=== PROCESSING COMPLETE ===")
        logger.info(f"Total records processed: {total_inserted + total_skipped}")
        logger.info(f"Successfully inserted: {total_inserted}")
        logger.info(f"Skipped (duplicates/errors): {total_skipped}")
        
        if total_inserted > 0:
            logger.info("🎉 SUCCESS: GDW data processing completed successfully!")
        else:
            logger.warning("⚠️  WARNING: No records were inserted. Please check for issues.")

if __name__ == "__main__":
    load_dotenv('.env')
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'database': os.getenv('DB_DATABASE', 'dam_monitoring'),
        'user': os.getenv('DB_USER', 'dam_user'),
        'password': os.getenv('DB_PASSWORD'),
        'port': os.getenv('DB_PORT', '5432')
    }
    gdw_dir = '/Users/taief/Desktop/DAM/GDW_v1_0_shp/GDW_v1_0_shp'
    logger.info(f"Using GDW directory: {gdw_dir}")
    
    # Verify the directory exists
    if not os.path.exists(gdw_dir):
        logger.error(f"GDW directory not found: {gdw_dir}")
        raise FileNotFoundError(f"GDW directory not found: {gdw_dir}")
    
    processor = GDWProcessor(db_config, gdw_dir)
    processor.run() 