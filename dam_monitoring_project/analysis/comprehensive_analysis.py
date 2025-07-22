#!/usr/bin/env python3
"""
Comprehensive Dam Monitoring Database Analysis Script

This script performs comprehensive statistical analysis on the dam monitoring database
containing 82,060+ dam records and exports results for conference paper visualizations.

Author: Dam Monitoring Research Team
Date: 2024
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
EXPORTS_DIR = ANALYSIS_DIR / "exports"
RESULTS_DIR = ANALYSIS_DIR / "results"
FIGURES_DIR = ANALYSIS_DIR / "figures"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for directory in [EXPORTS_DIR, RESULTS_DIR, FIGURES_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'comprehensive_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DamAnalyzer:
    """Comprehensive dam database analyzer."""
    
    def __init__(self):
        """Initialize analyzer with database connection."""
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
        
        # Analysis results storage
        self.results = {}
        
        logger.info("Dam Analyzer initialized successfully")
    
    def test_connection(self):
        """Test database connection."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) as total_dams FROM dams"))
                total_dams = result.fetchone()[0]
                logger.info(f"Database connection successful - {total_dams:,} total dams in database")
                return total_dams
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def global_overview_analysis(self):
        """Generate global overview statistics."""
        logger.info("=== Running Global Overview Analysis ===")
        
        query = """
        SELECT 
            COUNT(*) as total_dams,
            COUNT(DISTINCT country_code) as countries_covered,
            ROUND(AVG(height_meters), 2) as avg_height,
            ROUND(MAX(height_meters), 2) as max_height,
            ROUND(SUM(normal_capacity_mcm), 2) as total_capacity_mcm,
            MIN(construction_year) as oldest_dam_year,
            MAX(construction_year) as newest_dam_year,
            COUNT(CASE WHEN height_meters IS NOT NULL THEN 1 END) as height_data_available,
            COUNT(CASE WHEN normal_capacity_mcm IS NOT NULL THEN 1 END) as capacity_data_available,
            COUNT(CASE WHEN construction_year IS NOT NULL THEN 1 END) as construction_year_available,
            ROUND(AVG(data_quality_score), 2) as avg_data_quality
        FROM dams
        """
        
        with self.engine.connect() as conn:
            result = pd.read_sql(query, conn)
            
        overview = result.iloc[0].to_dict()
        
        # Calculate data completeness percentages
        overview['height_completeness_pct'] = round(
            (overview['height_data_available'] / overview['total_dams']) * 100, 2
        )
        overview['capacity_completeness_pct'] = round(
            (overview['capacity_data_available'] / overview['total_dams']) * 100, 2
        )
        overview['construction_year_completeness_pct'] = round(
            (overview['construction_year_available'] / overview['total_dams']) * 100, 2
        )
        
        self.results['global_overview'] = overview
        
        # Log key statistics
        logger.info(f"Total dams: {overview['total_dams']:,}")
        logger.info(f"Countries covered: {overview['countries_covered']}")
        logger.info(f"Average height: {overview['avg_height']}m")
        logger.info(f"Total capacity: {overview['total_capacity_mcm']:,} MCM")
        logger.info(f"Construction period: {overview['oldest_dam_year']}-{overview['newest_dam_year']}")
        
        return overview
    
    def country_distribution_analysis(self):
        """Analyze dam distribution by country."""
        logger.info("=== Running Country Distribution Analysis ===")
        
        query = """
        SELECT 
            country_code,
            COUNT(*) as dam_count,
            ROUND(AVG(height_meters), 2) as avg_height,
            ROUND(SUM(normal_capacity_mcm), 2) as total_capacity_mcm,
            MIN(construction_year) as oldest_dam,
            MAX(construction_year) as newest_dam,
            ROUND(AVG(data_quality_score), 2) as avg_quality_score,
            COUNT(CASE WHEN height_meters IS NOT NULL THEN 1 END) as height_data_count,
            COUNT(CASE WHEN normal_capacity_mcm IS NOT NULL THEN 1 END) as capacity_data_count
        FROM dams 
        GROUP BY country_code 
        ORDER BY dam_count DESC
        """
        
        with self.engine.connect() as conn:
            country_stats = pd.read_sql(query, conn)
        
        # Calculate percentages and rankings
        total_dams = country_stats['dam_count'].sum()
        country_stats['percentage_of_total'] = round(
            (country_stats['dam_count'] / total_dams) * 100, 2
        )
        country_stats['global_rank'] = range(1, len(country_stats) + 1)
        
        # Export full dataset
        country_stats.to_csv(EXPORTS_DIR / 'country_distribution.csv', index=False)
        
        # Store top 20 for results
        top_20 = country_stats.head(20)
        self.results['country_distribution'] = {
            'top_20': top_20.to_dict('records'),
            'total_countries': len(country_stats),
            'top_5_countries': top_20.head(5)['country_code'].tolist(),
            'top_5_dam_counts': top_20.head(5)['dam_count'].tolist()
        }
        
        logger.info(f"Top 5 countries by dam count:")
        for i, row in top_20.head(5).iterrows():
            logger.info(f"  {row['global_rank']}. {row['country_code']}: {row['dam_count']:,} dams")
        
        return country_stats
    
    def construction_timeline_analysis(self):
        """Analyze dam construction over time."""
        logger.info("=== Running Construction Timeline Analysis ===")
        
        # Decade-level analysis
        decade_query = """
        SELECT 
            CASE 
                WHEN construction_year < 1900 THEN '1800-1899'
                WHEN construction_year < 1920 THEN '1900-1919'
                WHEN construction_year < 1940 THEN '1920-1939'
                WHEN construction_year < 1960 THEN '1940-1959'
                WHEN construction_year < 1980 THEN '1960-1979'
                WHEN construction_year < 2000 THEN '1980-1999'
                ELSE '2000-2024'
            END as period,
            COUNT(*) as dams_built,
            ROUND(AVG(height_meters), 2) as avg_height,
            ROUND(SUM(normal_capacity_mcm), 2) as total_capacity
        FROM dams 
        WHERE construction_year IS NOT NULL
        GROUP BY 
            CASE 
                WHEN construction_year < 1900 THEN '1800-1899'
                WHEN construction_year < 1920 THEN '1900-1919'
                WHEN construction_year < 1940 THEN '1920-1939'
                WHEN construction_year < 1960 THEN '1940-1959'
                WHEN construction_year < 1980 THEN '1960-1979'
                WHEN construction_year < 2000 THEN '1980-1999'
                ELSE '2000-2024'
            END
        ORDER BY period
        """
        
        # Yearly analysis for recent decades
        yearly_query = """
        SELECT 
            construction_year,
            COUNT(*) as dams_built,
            ROUND(AVG(height_meters), 2) as avg_height
        FROM dams 
        WHERE construction_year IS NOT NULL 
        AND construction_year >= 1950
        GROUP BY construction_year
        ORDER BY construction_year
        """
        
        with self.engine.connect() as conn:
            decade_timeline = pd.read_sql(decade_query, conn)
            yearly_timeline = pd.read_sql(yearly_query, conn)
        
        # Export datasets
        decade_timeline.to_csv(EXPORTS_DIR / 'construction_timeline_decades.csv', index=False)
        yearly_timeline.to_csv(EXPORTS_DIR / 'construction_timeline_yearly.csv', index=False)
        
        # Find peak construction periods
        peak_decade = decade_timeline.loc[decade_timeline['dams_built'].idxmax()]
        peak_year = yearly_timeline.loc[yearly_timeline['dams_built'].idxmax()]
        
        self.results['construction_timeline'] = {
            'decade_analysis': decade_timeline.to_dict('records'),
            'peak_decade': {
                'period': peak_decade['period'],
                'dams_built': int(peak_decade['dams_built'])
            },
            'peak_year': {
                'year': int(peak_year['construction_year']),
                'dams_built': int(peak_year['dams_built'])
            },
            'total_with_construction_data': int(decade_timeline['dams_built'].sum())
        }
        
        logger.info(f"Peak construction decade: {peak_decade['period']} ({peak_decade['dams_built']:,} dams)")
        logger.info(f"Peak construction year: {peak_year['construction_year']} ({peak_year['dams_built']:,} dams)")
        
        return decade_timeline, yearly_timeline
    
    def dam_size_categories_analysis(self):
        """Analyze dam size distribution."""
        logger.info("=== Running Dam Size Categories Analysis ===")
        
        query = """
        SELECT 
            CASE 
                WHEN height_meters < 30 THEN 'Small (<30m)'
                WHEN height_meters < 100 THEN 'Medium (30-100m)'
                ELSE 'Large (>100m)'
            END as size_category,
            COUNT(*) as count,
            ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM dams WHERE height_meters IS NOT NULL), 2) as percentage,
            ROUND(AVG(height_meters), 2) as avg_height,
            ROUND(MIN(height_meters), 2) as min_height,
            ROUND(MAX(height_meters), 2) as max_height,
            ROUND(AVG(normal_capacity_mcm), 2) as avg_capacity
        FROM dams 
        WHERE height_meters IS NOT NULL
        GROUP BY 
            CASE 
                WHEN height_meters < 30 THEN 'Small (<30m)'
                WHEN height_meters < 100 THEN 'Medium (30-100m)'
                ELSE 'Large (>100m)'
            END
        ORDER BY size_category
        """
        
        with self.engine.connect() as conn:
            size_distribution = pd.read_sql(query, conn)
        
        # Export dataset
        size_distribution.to_csv(EXPORTS_DIR / 'dam_size_categories.csv', index=False)
        
        self.results['dam_size_categories'] = size_distribution.to_dict('records')
        
        logger.info("Dam size distribution:")
        for _, row in size_distribution.iterrows():
            logger.info(f"  {row['size_category']}: {row['count']:,} dams ({row['percentage']}%)")
        
        return size_distribution
    
    def geographic_analysis(self):
        """Analyze geographic distribution and coordinate ranges."""
        logger.info("=== Running Geographic Analysis ===")
        
        coordinate_query = """
        SELECT 
            MIN(ST_X(coordinates)) as min_longitude,
            MAX(ST_X(coordinates)) as max_longitude,
            MIN(ST_Y(coordinates)) as min_latitude,
            MAX(ST_Y(coordinates)) as max_latitude,
            ROUND(CAST(AVG(ST_X(coordinates)) AS NUMERIC), 4) as center_longitude,
            ROUND(CAST(AVG(ST_Y(coordinates)) AS NUMERIC), 4) as center_latitude,
            COUNT(*) as total_with_coordinates
        FROM dams
        WHERE coordinates IS NOT NULL
        """
        
        # Get sample coordinates for mapping
        coordinate_sample_query = """
        SELECT 
            dam_name,
            country_code,
            ST_X(coordinates) as longitude,
            ST_Y(coordinates) as latitude,
            height_meters,
            normal_capacity_mcm,
            construction_year,
            data_quality_score
        FROM dams
        WHERE coordinates IS NOT NULL
        ORDER BY RANDOM()
        LIMIT 10000
        """
        
        with self.engine.connect() as conn:
            coordinate_ranges = pd.read_sql(coordinate_query, conn)
            coordinate_sample = pd.read_sql(coordinate_sample_query, conn)
        
        # Export coordinate sample for mapping
        coordinate_sample.to_csv(EXPORTS_DIR / 'geographic_coordinates.csv', index=False)
        
        ranges = coordinate_ranges.iloc[0].to_dict()
        
        self.results['geographic_analysis'] = {
            'coordinate_ranges': ranges,
            'geographic_span': {
                'longitude_span': round(ranges['max_longitude'] - ranges['min_longitude'], 2),
                'latitude_span': round(ranges['max_latitude'] - ranges['min_latitude'], 2)
            },
            'global_center': {
                'longitude': ranges['center_longitude'],
                'latitude': ranges['center_latitude']
            }
        }
        
        logger.info(f"Geographic span: Longitude {ranges['min_longitude']:.2f} to {ranges['max_longitude']:.2f}")
        logger.info(f"Geographic span: Latitude {ranges['min_latitude']:.2f} to {ranges['max_latitude']:.2f}")
        logger.info(f"Global center: ({ranges['center_longitude']:.4f}, {ranges['center_latitude']:.4f})")
        
        return coordinate_ranges, coordinate_sample
    
    def data_quality_analysis(self):
        """Analyze data quality distribution."""
        logger.info("=== Running Data Quality Analysis ===")
        
        quality_query = """
        SELECT 
            ROUND(data_quality_score, 1) as quality_score,
            COUNT(*) as count,
            ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM dams), 2) as percentage
        FROM dams 
        WHERE data_quality_score IS NOT NULL
        GROUP BY ROUND(data_quality_score, 1)
        ORDER BY quality_score DESC
        """
        
        completeness_query = """
        SELECT 
            'dam_name' as field,
            COUNT(CASE WHEN dam_name IS NOT NULL AND dam_name != '' THEN 1 END) as complete_count,
            COUNT(*) as total_count,
            ROUND(COUNT(CASE WHEN dam_name IS NOT NULL AND dam_name != '' THEN 1 END) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM dams
        UNION ALL
        SELECT 
            'coordinates' as field,
            COUNT(CASE WHEN coordinates IS NOT NULL THEN 1 END) as complete_count,
            COUNT(*) as total_count,
            ROUND(COUNT(CASE WHEN coordinates IS NOT NULL THEN 1 END) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM dams
        UNION ALL
        SELECT 
            'height_meters' as field,
            COUNT(CASE WHEN height_meters IS NOT NULL THEN 1 END) as complete_count,
            COUNT(*) as total_count,
            ROUND(COUNT(CASE WHEN height_meters IS NOT NULL THEN 1 END) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM dams
        UNION ALL
        SELECT 
            'construction_year' as field,
            COUNT(CASE WHEN construction_year IS NOT NULL THEN 1 END) as complete_count,
            COUNT(*) as total_count,
            ROUND(COUNT(CASE WHEN construction_year IS NOT NULL THEN 1 END) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM dams
        UNION ALL
        SELECT 
            'normal_capacity_mcm' as field,
            COUNT(CASE WHEN normal_capacity_mcm IS NOT NULL THEN 1 END) as complete_count,
            COUNT(*) as total_count,
            ROUND(COUNT(CASE WHEN normal_capacity_mcm IS NOT NULL THEN 1 END) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM dams
        ORDER BY completeness_pct DESC
        """
        
        with self.engine.connect() as conn:
            quality_distribution = pd.read_sql(quality_query, conn)
            completeness_stats = pd.read_sql(completeness_query, conn)
        
        # Export datasets
        quality_distribution.to_csv(EXPORTS_DIR / 'data_quality_distribution.csv', index=False)
        completeness_stats.to_csv(EXPORTS_DIR / 'data_completeness_stats.csv', index=False)
        
        self.results['data_quality'] = {
            'quality_distribution': quality_distribution.to_dict('records'),
            'completeness_stats': completeness_stats.to_dict('records'),
            'average_quality_score': quality_distribution['quality_score'].mean(),
            'highest_quality_percentage': quality_distribution.iloc[0]['percentage']
        }
        
        logger.info("Data completeness summary:")
        for _, row in completeness_stats.iterrows():
            logger.info(f"  {row['field']}: {row['completeness_pct']}% complete")
        
        return quality_distribution, completeness_stats
    
    def infrastructure_characteristics_analysis(self):
        """Analyze infrastructure characteristics and patterns."""
        logger.info("=== Running Infrastructure Characteristics Analysis ===")
        
        # Height and capacity distribution
        characteristics_query = """
        SELECT 
            country_code,
            COUNT(*) as dam_count,
            ROUND(AVG(height_meters), 2) as avg_height,
            ROUND(STDDEV(height_meters), 2) as height_stddev,
            ROUND(AVG(normal_capacity_mcm), 2) as avg_capacity,
            ROUND(SUM(normal_capacity_mcm), 2) as total_capacity,
            COUNT(CASE WHEN height_meters > 100 THEN 1 END) as large_dams_count,
            ROUND(AVG(construction_year), 0) as avg_construction_year
        FROM dams
        WHERE height_meters IS NOT NULL
        GROUP BY country_code
        HAVING COUNT(*) >= 10  -- Only countries with 10+ dams
        ORDER BY dam_count DESC
        """
        
        with self.engine.connect() as conn:
            infrastructure_stats = pd.read_sql(characteristics_query, conn)
        
        # Export dataset
        infrastructure_stats.to_csv(EXPORTS_DIR / 'infrastructure_characteristics.csv', index=False)
        
        # Calculate derived metrics
        infrastructure_stats['large_dams_percentage'] = round(
            (infrastructure_stats['large_dams_count'] / infrastructure_stats['dam_count']) * 100, 2
        )
        
        self.results['infrastructure_characteristics'] = {
            'country_infrastructure': infrastructure_stats.head(20).to_dict('records'),
            'global_patterns': {
                'countries_with_large_dams': len(infrastructure_stats[infrastructure_stats['large_dams_count'] > 0]),
                'avg_height_across_countries': round(infrastructure_stats['avg_height'].mean(), 2),
                'total_countries_analyzed': len(infrastructure_stats)
            }
        }
        
        logger.info(f"Infrastructure analysis covers {len(infrastructure_stats)} countries with 10+ dams")
        logger.info(f"Average dam height across all countries: {infrastructure_stats['avg_height'].mean():.2f}m")
        
        return infrastructure_stats
    
    def export_summary_json(self):
        """Export comprehensive analysis summary as JSON."""
        logger.info("=== Exporting Analysis Summary ===")
        
        # Create comprehensive summary
        summary = {
            'analysis_metadata': {
                'generated_at': datetime.now().isoformat(),
                'database_total_records': self.results['global_overview']['total_dams'],
                'analysis_version': '1.0'
            },
            'global_statistics': self.results['global_overview'],
            'geographic_coverage': self.results['geographic_analysis'],
            'temporal_patterns': self.results['construction_timeline'],
            'infrastructure_distribution': self.results['dam_size_categories'],
            'country_rankings': {
                'top_20_countries': self.results['country_distribution']['top_20'][:20],
                'total_countries': self.results['country_distribution']['total_countries']
            },
            'data_quality_assessment': self.results['data_quality']
        }
        
        # Export summary JSON
        with open(EXPORTS_DIR / 'analysis_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Export simplified JSON for quick reference
        quick_stats = {
            'total_dams': self.results['global_overview']['total_dams'],
            'countries_covered': self.results['global_overview']['countries_covered'],
            'construction_period': [
                self.results['global_overview']['oldest_dam_year'],
                self.results['global_overview']['newest_dam_year']
            ],
            'geographic_span': {
                'longitude': [
                    self.results['geographic_analysis']['coordinate_ranges']['min_longitude'],
                    self.results['geographic_analysis']['coordinate_ranges']['max_longitude']
                ],
                'latitude': [
                    self.results['geographic_analysis']['coordinate_ranges']['min_latitude'],
                    self.results['geographic_analysis']['coordinate_ranges']['max_latitude']
                ]
            },
            'top_5_countries': dict(zip(
                self.results['country_distribution']['top_5_countries'],
                self.results['country_distribution']['top_5_dam_counts']
            ))
        }
        
        with open(EXPORTS_DIR / 'quick_statistics.json', 'w') as f:
            json.dump(quick_stats, f, indent=2)
        
        logger.info(f"Analysis summary exported to: {EXPORTS_DIR / 'analysis_summary.json'}")
        logger.info(f"Quick statistics exported to: {EXPORTS_DIR / 'quick_statistics.json'}")
        
        return summary
    
    def generate_markdown_report(self):
        """Generate comprehensive markdown report."""
        logger.info("=== Generating Markdown Report ===")
        
        report_content = f"""# Comprehensive Dam Monitoring Database Analysis Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This analysis covers **{self.results['global_overview']['total_dams']:,} dam records** spanning **{self.results['global_overview']['countries_covered']} countries** worldwide, representing the most comprehensive global dam infrastructure dataset available.

## Key Findings

### Global Infrastructure Scale
- **Total Dams**: {self.results['global_overview']['total_dams']:,}
- **Countries Covered**: {self.results['global_overview']['countries_covered']}
- **Construction Period**: {self.results['global_overview']['oldest_dam_year']}-{self.results['global_overview']['newest_dam_year']}
- **Average Dam Height**: {self.results['global_overview']['avg_height']}m
- **Total Reservoir Capacity**: {self.results['global_overview']['total_capacity_mcm']:,} MCM

### Geographic Distribution
- **Longitude Range**: {self.results['geographic_analysis']['coordinate_ranges']['min_longitude']:.2f}° to {self.results['geographic_analysis']['coordinate_ranges']['max_longitude']:.2f}°
- **Latitude Range**: {self.results['geographic_analysis']['coordinate_ranges']['min_latitude']:.2f}° to {self.results['geographic_analysis']['coordinate_ranges']['max_latitude']:.2f}°
- **Global Center**: ({self.results['geographic_analysis']['global_center']['longitude']:.4f}°, {self.results['geographic_analysis']['global_center']['latitude']:.4f}°)

### Top 5 Countries by Dam Count
"""
        
        # Add top countries table
        for i, country in enumerate(self.results['country_distribution']['top_5_countries'], 1):
            count = self.results['country_distribution']['top_5_dam_counts'][i-1]
            report_content += f"{i}. **{country}**: {count:,} dams\n"
        
        report_content += f"""
### Construction Timeline Patterns
- **Peak Construction Decade**: {self.results['construction_timeline']['peak_decade']['period']} ({self.results['construction_timeline']['peak_decade']['dams_built']:,} dams)
- **Peak Construction Year**: {self.results['construction_timeline']['peak_year']['year']} ({self.results['construction_timeline']['peak_year']['dams_built']:,} dams)

### Dam Size Distribution
"""
        
        # Add size distribution
        for category in self.results['dam_size_categories']:
            report_content += f"- **{category['size_category']}**: {category['count']:,} dams ({category['percentage']}%)\n"
        
        report_content += f"""
### Data Quality Assessment
- **Average Quality Score**: {self.results['data_quality']['average_quality_score']:.2f}
- **Coordinate Data Completeness**: {self.results['global_overview']['height_completeness_pct']}%
- **Height Data Completeness**: {self.results['global_overview']['height_completeness_pct']}%
- **Construction Year Completeness**: {self.results['global_overview']['construction_year_completeness_pct']}%

## Exported Datasets

The following CSV files have been generated for visualization and further analysis:

1. `country_distribution.csv` - Country-wise dam statistics
2. `construction_timeline_decades.csv` - Decade-level construction analysis
3. `construction_timeline_yearly.csv` - Yearly construction data (1950+)
4. `geographic_coordinates.csv` - Sample coordinates for mapping (10,000 records)
5. `dam_size_categories.csv` - Size distribution analysis
6. `infrastructure_characteristics.csv` - Country-level infrastructure metrics
7. `data_quality_distribution.csv` - Quality score distribution
8. `data_completeness_stats.csv` - Field completeness statistics

## Analysis Summary Files

- `analysis_summary.json` - Complete analysis results
- `quick_statistics.json` - Key statistics for quick reference

---

*This report was generated by the Comprehensive Dam Monitoring Analysis System*
"""
        
        # Export report
        with open(RESULTS_DIR / 'comprehensive_analysis_report.md', 'w') as f:
            f.write(report_content)
        
        logger.info(f"Markdown report generated: {RESULTS_DIR / 'comprehensive_analysis_report.md'}")
        
        return report_content
    
    def run_complete_analysis(self):
        """Run all analysis components."""
        logger.info("🔬 Starting Comprehensive Dam Monitoring Database Analysis")
        
        try:
            # Test database connection
            total_dams = self.test_connection()
            
            # Run all analysis components
            self.global_overview_analysis()
            self.country_distribution_analysis()
            self.construction_timeline_analysis()
            self.dam_size_categories_analysis()
            self.geographic_analysis()
            self.data_quality_analysis()
            self.infrastructure_characteristics_analysis()
            
            # Export results
            summary = self.export_summary_json()
            report = self.generate_markdown_report()
            
            # Final summary
            logger.info("=" * 60)
            logger.info("🎯 ANALYSIS COMPLETE - KEY RESULTS:")
            logger.info(f"   📊 Total Dams Analyzed: {total_dams:,}")
            logger.info(f"   🌍 Countries Covered: {self.results['global_overview']['countries_covered']}")
            logger.info(f"   📁 Files Exported: {len(list(EXPORTS_DIR.glob('*.csv')) + list(EXPORTS_DIR.glob('*.json')))}")
            logger.info(f"   📈 Top Country: {self.results['country_distribution']['top_5_countries'][0]} ({self.results['country_distribution']['top_5_dam_counts'][0]:,} dams)")
            logger.info("=" * 60)
            
            return summary
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise


def main():
    """Main execution function."""
    analyzer = DamAnalyzer()
    summary = analyzer.run_complete_analysis()
    return summary


if __name__ == "__main__":
    main() 