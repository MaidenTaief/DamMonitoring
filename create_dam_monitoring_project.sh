#!/bin/bash

# Create directory structure
echo "Creating directory structure..."
mkdir -p dam_monitoring_project/{sql,python,config,docs,logs,data}
cd dam_monitoring_project

# Create complete SQL schema
cat > sql/dam_schema.sql << 'EOF'
-- Dam Monitoring Database Schema
-- Designed for AI/ML applications with multi-modal transformer support
-- PostgreSQL with PostGIS extensions

-- Enable PostGIS extension
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ==========================================
-- CORE REFERENCE TABLES
-- ==========================================

-- Country/Region hierarchy for systematic expansion
CREATE TABLE regions (
    region_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    region_name VARCHAR(100) NOT NULL,
    region_type VARCHAR(50) NOT NULL, -- 'country', 'continent', 'subregion'
    parent_region_id UUID REFERENCES regions(region_id),
    iso_code VARCHAR(10),
    priority_order INTEGER DEFAULT 999,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert initial regions with your expansion plan
INSERT INTO regions (region_name, region_type, iso_code, priority_order) VALUES 
('Norway', 'country', 'NO', 1),
('India', 'country', 'IN', 2),
('Bangladesh', 'country', 'BD', 3),
('Asia', 'continent', 'AS', 4),
('Europe', 'continent', 'EU', 5),
('Africa', 'continent', 'AF', 6),
('South America', 'continent', 'SA', 7),
('North America', 'continent', 'NA', 8),
('Antarctica', 'continent', 'AN', 9);

-- Dam type classification for good organization
CREATE TABLE dam_types (
    type_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    type_name VARCHAR(100) NOT NULL,
    type_category VARCHAR(50) NOT NULL, -- 'structural', 'functional', 'material'
    description TEXT,
    typical_height_range VARCHAR(50),
    typical_applications TEXT[]
);

-- Insert dam type classifications
INSERT INTO dam_types (type_name, type_category, description, typical_applications) VALUES 
('Arch Dam', 'structural', 'Curved concrete/masonry dam transferring load to abutments', ARRAY['hydropower', 'water_supply']),
('Gravity Dam', 'structural', 'Straight dam resisting water pressure through weight', ARRAY['hydropower', 'flood_control']),
('Embankment Dam', 'structural', 'Earth/rock fill dam', ARRAY['flood_control', 'water_supply', 'hydropower']),
('Buttress Dam', 'structural', 'Reinforced concrete with supporting buttresses', ARRAY['hydropower', 'water_supply']),
('Hydropower Dam', 'functional', 'Dam with electricity generation facilities', ARRAY['hydropower']),
('Flood Control Dam', 'functional', 'Dam primarily for flood mitigation', ARRAY['flood_control']),
('Water Supply Dam', 'functional', 'Dam for municipal/industrial water supply', ARRAY['water_supply']),
('Multi-Purpose Dam', 'functional', 'Dam serving multiple functions', ARRAY['hydropower', 'flood_control', 'water_supply']);

-- Data sources for tracking provenance
CREATE TABLE data_sources (
    source_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_name VARCHAR(100) NOT NULL,
    source_type VARCHAR(50) NOT NULL, -- 'database', 'api', 'manual', 'satellite'
    url TEXT,
    api_endpoint TEXT,
    update_frequency VARCHAR(50),
    reliability_score DECIMAL(3,2) DEFAULT 0.95,
    last_accessed TIMESTAMP,
    is_active BOOLEAN DEFAULT true
);

-- Insert your primary data sources
INSERT INTO data_sources (source_name, source_type, url, update_frequency, reliability_score) VALUES 
('Global Dam Watch', 'database', 'https://www.globaldamwatch.org/', 'annual', 0.98),
('GOODD', 'database', 'https://www.goodd.org/', 'annual', 0.95),
('GRAND', 'database', 'https://www.mcgill.ca/gws/grand', 'annual', 0.96),
('FHReD', 'database', 'https://www.fhred.org/', 'annual', 0.94),
('USGS Water Data', 'api', 'https://waterdata.usgs.gov/', 'real-time', 0.99),
('National Water Dashboard', 'api', 'https://dashboard.waterdata.usgs.gov/', 'real-time', 0.97);

-- ==========================================
-- MAIN DAM INFORMATION TABLE
-- ==========================================

CREATE TABLE dams (
    dam_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Basic Information
    dam_name VARCHAR(255) NOT NULL,
    alternative_names TEXT[], -- array for multiple names
    
    -- Location & Geography
    region_id UUID REFERENCES regions(region_id),
    country_code VARCHAR(10),
    province_state VARCHAR(100),
    nearest_city VARCHAR(100),
    coordinates GEOMETRY(POINT, 4326) NOT NULL, -- WGS84
    elevation_masl DECIMAL(10,2), -- meters above sea level
    
    -- Physical Characteristics
    primary_dam_type_id UUID REFERENCES dam_types(type_id),
    secondary_dam_types UUID[], -- array of type_ids for multi-type dams
    construction_year INTEGER,
    completion_year INTEGER,
    height_meters DECIMAL(10,2),
    length_meters DECIMAL(10,2),
    width_base_meters DECIMAL(10,2),
    volume_concrete_mcm DECIMAL(12,3), -- million cubic meters
    
    -- Reservoir Characteristics
    reservoir_name VARCHAR(255),
    normal_capacity_mcm DECIMAL(15,2), -- million cubic meters
    flood_capacity_mcm DECIMAL(15,2),
    dead_storage_mcm DECIMAL(15,2),
    catchment_area_km2 DECIMAL(12,2),
    reservoir_surface_area_km2 DECIMAL(10,2),
    
    -- Operational Information
    primary_purpose VARCHAR(100), -- main function
    secondary_purposes TEXT[], -- array of additional purposes
    installed_capacity_mw DECIMAL(10,2), -- for hydropower dams
    annual_generation_gwh DECIMAL(12,2), -- gigawatt hours
    
    -- Status & Construction
    construction_status VARCHAR(50) DEFAULT 'operational', -- planned, under_construction, operational, decommissioned
    construction_start_date DATE,
    expected_completion_date DATE,
    decommission_date DATE,
    
    -- Risk & Safety
    hazard_potential VARCHAR(20), -- low, significant, high
    dam_safety_rating VARCHAR(20),
    last_safety_inspection DATE,
    
    -- Environmental & Social
    environmental_impact_score DECIMAL(3,2), -- 0-10 scale
    population_affected INTEGER,
    villages_submerged INTEGER,
    
    -- Data Provenance
    primary_data_source_id UUID REFERENCES data_sources(source_id),
    secondary_data_sources UUID[], -- array of source_ids
    data_quality_score DECIMAL(3,2) DEFAULT 0.8,
    last_verified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Spatial Index
    CONSTRAINT valid_coordinates CHECK (ST_IsValid(coordinates))
);

-- Create spatial index for efficient geographical queries
CREATE INDEX idx_dams_coordinates ON dams USING GIST(coordinates);
CREATE INDEX idx_dams_region ON dams(region_id);
CREATE INDEX idx_dams_construction_year ON dams(construction_year);
CREATE INDEX idx_dams_status ON dams(construction_status);

-- ==========================================
-- MONITORING INFRASTRUCTURE
-- ==========================================

-- Monitoring stations for real-time data
CREATE TABLE monitoring_stations (
    station_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dam_id UUID REFERENCES dams(dam_id),
    
    -- Station Information
    station_name VARCHAR(255) NOT NULL,
    station_code VARCHAR(50), -- official station identifier
    station_type VARCHAR(100), -- USGS, local_government, private, satellite
    operator VARCHAR(100),
    
    -- Location
    coordinates GEOMETRY(POINT, 4326) NOT NULL,
    elevation_masl DECIMAL(10,2),
    location_description TEXT,
    
    -- Monitoring Capabilities
    parameters_monitored TEXT[], -- water_level, flow_rate, temperature, etc.
    sensor_types TEXT[], -- acoustic, vibration, thermal, optical, etc.
    measurement_frequency VARCHAR(50), -- real-time, hourly, daily, weekly
    data_transmission_method VARCHAR(50), -- satellite, cellular, wifi, manual
    
    -- Operational Status
    installation_date DATE,
    last_calibration_date DATE,
    status VARCHAR(50) DEFAULT 'active', -- active, inactive, maintenance
    
    -- Data Source
    data_source_id UUID REFERENCES data_sources(source_id),
    api_endpoint TEXT,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_monitoring_stations_dam ON monitoring_stations(dam_id);
CREATE INDEX idx_monitoring_stations_coordinates ON monitoring_stations USING GIST(coordinates);

-- ==========================================
-- TIME-SERIES DATA FOR AI/ML
-- ==========================================

-- Sensor readings optimized for transformer models
CREATE TABLE sensor_readings (
    reading_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    station_id UUID REFERENCES monitoring_stations(station_id),
    
    -- Measurement Details
    parameter_type VARCHAR(100) NOT NULL, -- water_level, flow_rate, temperature, etc.
    measurement_value DECIMAL(15,4) NOT NULL,
    unit VARCHAR(20) NOT NULL,
    measurement_timestamp TIMESTAMP NOT NULL,
    
    -- Data Quality
    quality_code VARCHAR(10), -- excellent, good, fair, poor
    quality_flags TEXT[], -- array of quality indicators
    validation_status VARCHAR(20) DEFAULT 'unvalidated',
    
    -- Multi-modal support
    sensor_modality VARCHAR(50), -- acoustic, vibration, thermal, optical, etc.
    raw_data_blob BYTEA, -- for storing raw sensor data (images, audio, etc.)
    processed_features JSONB, -- for storing extracted features
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Optimized indexes for time-series queries
CREATE INDEX idx_sensor_readings_station_time ON sensor_readings(station_id, measurement_timestamp);
CREATE INDEX idx_sensor_readings_parameter ON sensor_readings(parameter_type);
CREATE INDEX idx_sensor_readings_timestamp ON sensor_readings(measurement_timestamp);

-- ==========================================
-- AI/ML SUPPORT TABLES
-- ==========================================

-- Training datasets for transformer models
CREATE TABLE ml_datasets (
    dataset_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dataset_name VARCHAR(255) NOT NULL,
    dataset_type VARCHAR(50), -- training, validation, test
    
    -- Dataset Characteristics
    dam_ids UUID[], -- array of dam_ids included
    parameter_types TEXT[], -- parameters included
    time_range_start TIMESTAMP,
    time_range_end TIMESTAMP,
    total_samples INTEGER,
    
    -- Multi-modal Information
    modalities TEXT[], -- acoustic, vibration, thermal, optical, etc.
    feature_dimensions INTEGER,
    sequence_length INTEGER, -- for transformer models
    
    -- Preprocessing
    preprocessing_steps JSONB,
    normalization_parameters JSONB,
    
    -- Metadata
    created_by VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    version INTEGER DEFAULT 1
);

-- Model performance tracking
CREATE TABLE model_performance (
    performance_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(255) NOT NULL,
    model_type VARCHAR(100), -- transformer, lstm, cnn, etc.
    
    -- Performance Metrics
    dataset_id UUID REFERENCES ml_datasets(dataset_id),
    accuracy DECIMAL(5,4),
    precision_score DECIMAL(5,4),
    recall DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    rmse DECIMAL(10,6),
    mae DECIMAL(10,6),
    
    -- Model Configuration
    model_parameters JSONB,
    training_duration_minutes INTEGER,
    
    -- Metadata
    evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT
);

-- ==========================================
-- CLIMATE DATA INTEGRATION
-- ==========================================

-- Climate data for climate-resilient monitoring
CREATE TABLE climate_data (
    climate_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dam_id UUID REFERENCES dams(dam_id),
    
    -- Climate Measurements
    measurement_date DATE NOT NULL,
    temperature_avg_celsius DECIMAL(5,2),
    temperature_max_celsius DECIMAL(5,2),
    temperature_min_celsius DECIMAL(5,2),
    precipitation_mm DECIMAL(8,2),
    humidity_percent DECIMAL(5,2),
    wind_speed_kmh DECIMAL(6,2),
    
    -- Climate Extremes
    extreme_weather_events TEXT[], -- flood, drought, storm, etc.
    climate_anomaly_score DECIMAL(5,2), -- deviation from normal
    
    -- Data Source
    data_source_id UUID REFERENCES data_sources(source_id),
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_climate_data_dam_date ON climate_data(dam_id, measurement_date);

-- ==========================================
-- UTILITY VIEWS FOR COMMON QUERIES
-- ==========================================

-- View for dam summary with region information
CREATE VIEW dam_summary AS
SELECT 
    d.dam_id,
    d.dam_name,
    r.region_name,
    d.country_code,
    dt.type_name as primary_dam_type,
    d.construction_year,
    d.height_meters,
    d.normal_capacity_mcm,
    d.construction_status,
    d.hazard_potential,
    ST_X(d.coordinates) as longitude,
    ST_Y(d.coordinates) as latitude
FROM dams d
JOIN regions r ON d.region_id = r.region_id
LEFT JOIN dam_types dt ON d.primary_dam_type_id = dt.type_id;

-- View for monitoring station coverage
CREATE VIEW monitoring_coverage AS
SELECT 
    d.dam_id,
    d.dam_name,
    COUNT(ms.station_id) as station_count,
    ARRAY_AGG(DISTINCT ms.station_type) as station_types,
    ARRAY_AGG(DISTINCT unnest(ms.parameters_monitored)) as all_parameters
FROM dams d
LEFT JOIN monitoring_stations ms ON d.dam_id = ms.dam_id
GROUP BY d.dam_id, d.dam_name;

-- ==========================================
-- FUNCTIONS FOR DATA MANAGEMENT
-- ==========================================

-- Function to update dam statistics
CREATE OR REPLACE FUNCTION update_dam_statistics()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-update timestamps
CREATE TRIGGER update_dam_timestamp
    BEFORE UPDATE ON dams
    FOR EACH ROW
    EXECUTE FUNCTION update_dam_statistics();

-- Function to calculate distance between dams
CREATE OR REPLACE FUNCTION calculate_dam_distance(dam1_id UUID, dam2_id UUID)
RETURNS DECIMAL AS $$
DECLARE
    distance_km DECIMAL;
BEGIN
    SELECT ST_Distance(
        ST_Transform(d1.coordinates, 3857),
        ST_Transform(d2.coordinates, 3857)
    ) / 1000 INTO distance_km
    FROM dams d1, dams d2
    WHERE d1.dam_id = dam1_id AND d2.dam_id = dam2_id;
    
    RETURN distance_km;
END;
$$ LANGUAGE plpgsql;

-- ==========================================
-- INITIAL DATA QUALITY CONSTRAINTS
-- ==========================================

-- Ensure data quality
ALTER TABLE dams ADD CONSTRAINT check_height_positive 
    CHECK (height_meters IS NULL OR height_meters > 0);

ALTER TABLE dams ADD CONSTRAINT check_capacity_positive 
    CHECK (normal_capacity_mcm IS NULL OR normal_capacity_mcm > 0);

ALTER TABLE dams ADD CONSTRAINT check_construction_year_reasonable 
    CHECK (construction_year IS NULL OR (construction_year >= 1800 AND construction_year <= 2050));

-- ==========================================
-- SAMPLE QUERIES FOR TESTING
-- ==========================================

-- Sample query: Get all dams in priority countries
-- SELECT * FROM dam_summary WHERE region_name IN ('Norway', 'India', 'Bangladesh');

-- Sample query: Find dams with monitoring stations
-- SELECT d.dam_name, mc.station_count, mc.all_parameters 
-- FROM dams d 
-- JOIN monitoring_coverage mc ON d.dam_id = mc.dam_id 
-- WHERE mc.station_count > 0;

-- Sample query: Get recent sensor readings for a specific dam
-- SELECT sr.parameter_type, sr.measurement_value, sr.measurement_timestamp
-- FROM sensor_readings sr
-- JOIN monitoring_stations ms ON sr.station_id = ms.station_id
-- WHERE ms.dam_id = 'your-dam-id-here'
-- ORDER BY sr.measurement_timestamp DESC
-- LIMIT 100;
EOF

# Create complete Python pipeline
cat > python/data_collection_pipeline.py << 'EOF'
[PASTE FULL PYTHON CONTENT HERE]
EOF

# Create complete configuration file
cat > config/database.conf << 'EOF'
[PASTE FULL CONFIGURATION CONTENT HERE]
EOF

# Create complete environment template
cat > config/.env.example << 'EOF'
[PASTE FULL ENV EXAMPLE CONTENT HERE]
EOF

# Create requirements.txt
cat > requirements.txt << 'EOF'
# Core dependencies
psycopg2-binary==2.9.7
pandas==2.0.3
geopandas==0.13.2
sqlalchemy==2.0.19
requests==2.31.0

# Data processing
numpy==1.24.3
shapely==2.0.1
fiona==1.9.4

# Database tools
python-dotenv==1.0.0
configparser==6.0.0

# Optional: For advanced features
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.15.0
streamlit==1.25.0
EOF

# Create setup.sh
cat > setup.sh << 'EOF'
#!/bin/bash

echo "Setting up Dam Monitoring Database Project..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create environment file
cp config/.env.example config/.env
echo "Please edit config/.env with your database credentials"

# Create database (requires PostgreSQL to be running)
echo "Creating database..."
sudo -u postgres psql -c "CREATE DATABASE dam_monitoring;"
sudo -u postgres psql -c "CREATE USER dam_user WITH PASSWORD 'your_secure_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE dam_monitoring TO dam_user;"

# Run database schema
echo "Setting up database schema..."
psql -U dam_user -d dam_monitoring -f sql/dam_schema.sql

echo "Setup complete!"
echo "Next steps:"
echo "1. Edit config/.env with your database credentials"
echo "2. Run: python python/data_collection_pipeline.py"
echo "3. Check logs in logs/ directory"
EOF

chmod +x setup.sh

# Create README.md
cat > README.md << 'EOF'
# Dam Monitoring Database Project

## Overview
This project creates a comprehensive database system for dam monitoring with AI/ML integration capabilities. It's designed to support research in climate-resilient infrastructure monitoring using transformer-based multi-modal approaches.

## Features
- **PostgreSQL with PostGIS** for spatial data handling
- **Multi-modal sensor data** support (acoustic, vibration, thermal, optical)
- **Real-time monitoring** integration with USGS and other sources
- **AI/ML ready** tables for transformer model training
- **Climate data integration** for resilience analysis
- **Scalable architecture** supporting global expansion

## Quick Start

### 1. Prerequisites
```bash
# Install PostgreSQL with PostGIS
sudo apt update
sudo apt install postgresql postgresql-contrib postgis

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Database Setup
```bash
# Run the automated setup
./setup.sh

# Or manually:
sudo -u postgres psql -c "CREATE DATABASE dam_monitoring;"
psql -U dam_user -d dam_monitoring -f sql/dam_schema.sql
```

### 3. Data Collection
```bash
# Edit configuration
nano config/.env

# Run data collection
python python/data_collection_pipeline.py
```

## Directory Structure
```
dam_monitoring_project/
├── sql/                    # Database schemas
├── python/                 # Data collection scripts
├── config/                 # Configuration files
├── docs/                   # Documentation
├── logs/                   # Log files
├── data/                   # Data files
├── requirements.txt        # Python dependencies
├── setup.sh               # Setup script
└── README.md              # This file
```

## Database Schema
The database includes:
- **dams**: Main dam information with spatial data
- **monitoring_stations**: Sensor station locations
- **sensor_readings**: Time-series sensor data
- **climate_data**: Climate measurements
- **ml_datasets**: AI/ML training datasets
- **model_performance**: Model evaluation metrics

## Data Sources
- **USGS Water Data**: Real-time monitoring stations
- **Global Dam Watch**: Comprehensive dam database
- **GOODD**: Global dam observations
- **GRAND**: Large dam and reservoir data
- **FHReD**: Future hydropower dam data

## Usage Examples

### Query dams by country
```sql
SELECT * FROM dam_summary 
WHERE region_name IN ('Norway', 'India', 'Bangladesh');
```

### Get monitoring station data
```sql
SELECT sr.parameter_type, AVG(sr.measurement_value)
FROM sensor_readings sr
GROUP BY sr.parameter_type;
```

### Find dams with monitoring coverage
```sql
SELECT d.dam_name, mc.station_count 
FROM dams d 
JOIN monitoring_coverage mc ON d.dam_id = mc.dam_id;
```

## API Development (Future)
The database is designed to support REST APIs and GraphQL interfaces for:
- Dam information retrieval
- Real-time sensor data access
- ML model training data preparation
- Climate impact analysis

## Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License
This project is licensed under the MIT License.

## Support
For questions or issues:
1. Check the documentation in docs/
2. Review logs in logs/
3. Open an issue on GitHub

## Acknowledgments
- USGS for providing real-time water data
- Global Dam Watch for comprehensive dam databases
- PostgreSQL and PostGIS communities
EOF

echo "All files created successfully!"
echo ""
echo "Project structure:"
echo "dam_monitoring_project/"
echo "├── sql/"
echo "│   └── dam_schema.sql"
echo "├── python/"
echo "│   └── data_collection_pipeline.py"
echo "├── config/"
echo "│   ├── database.conf"
echo "│   └── .env.example"
echo "├── docs/"
echo "├── logs/"
echo "├── data/"
echo "├── requirements.txt"
echo "├── setup.sh"
echo "└── README.md"
echo ""
echo "Next steps:"
echo "1. cd dam_monitoring_project"
echo "2. Edit config/.env with your database credentials"
echo "3. Run: ./setup.sh"
echo "4. Start collecting data: python python/data_collection_pipeline.py" 