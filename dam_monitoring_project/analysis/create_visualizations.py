#!/usr/bin/env python3
"""
Create visualizations for dam monitoring research
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import os

def create_research_visualizations():
    """Generate publication-ready visualizations"""
    
    # Set style for research papers
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Load data
    dams_df = pd.read_csv('dam_monitoring_project/analysis/dam_infrastructure_analysis.csv')
    sensor_df = pd.read_csv('dam_monitoring_project/analysis/sensor_data_sample.csv')
    sensor_df['measurement_timestamp'] = pd.to_datetime(sensor_df['measurement_timestamp'])
    
    # Create output directory
    os.makedirs('dam_monitoring_project/analysis/figures', exist_ok=True)
    
    # 1. Dam Infrastructure Overview
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Height by country
    dams_df.groupby('region_name')['height_meters'].mean().plot(kind='bar', ax=ax1, color='skyblue')
    ax1.set_title('Average Dam Height by Country')
    ax1.set_ylabel('Height (meters)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Capacity distribution
    ax2.hist(dams_df['normal_capacity_mcm'], bins=10, alpha=0.7, color='lightgreen')
    ax2.set_title('Dam Capacity Distribution')
    ax2.set_xlabel('Capacity (MCM)')
    ax2.set_ylabel('Frequency')
    
    # Construction timeline
    dams_df['construction_year'].hist(bins=10, ax=ax3, alpha=0.7, color='orange')
    ax3.set_title('Dam Construction Timeline')
    ax3.set_xlabel('Construction Year')
    ax3.set_ylabel('Number of Dams')
    
    # Purpose distribution
    if 'primary_purpose' in dams_df.columns:
        dams_df['primary_purpose'].value_counts().plot(kind='pie', ax=ax4, autopct='%1.1f%%')
        ax4.set_title('Dam Purpose Distribution')
    else:
        ax4.axis('off')
        ax4.set_title('Dam Purpose Distribution (N/A)')
    
    plt.tight_layout()
    plt.savefig('dam_monitoring_project/analysis/figures/dam_infrastructure_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Sensor Data Time Series
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    for i, param in enumerate(['water_level', 'flow_rate', 'temperature']):
        param_data = sensor_df[sensor_df['parameter_type'] == param]
        
        for region in param_data['region_name'].unique():
            region_data = param_data[param_data['region_name'] == region]
            daily_avg = region_data.set_index('measurement_timestamp').resample('D')['measurement_value'].mean()
            axes[i].plot(daily_avg.index, daily_avg.values, label=region, linewidth=2)
        
        axes[i].set_title(f'{param.replace("_", " ").title()} Time Series by Region')
        axes[i].set_ylabel(f'{param.replace("_", " ").title()}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dam_monitoring_project/analysis/figures/sensor_time_series.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Multi-modal Sensor Correlation Heatmap
    correlation_df = pd.read_csv('dam_monitoring_project/analysis/parameter_correlations.csv', index_col=0)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_df, annot=True, cmap='coolwarm', center=0, 
                square=True, cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('Multi-Parameter Sensor Correlation Matrix')
    plt.tight_layout()
    plt.savefig('dam_monitoring_project/analysis/figures/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Regional Comparison Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Sensor reading distribution by region
    sns.boxplot(data=sensor_df, x='region_name', y='measurement_value', 
                hue='parameter_type', ax=axes[0,0])
    axes[0,0].set_title('Sensor Reading Distribution by Region')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Data quality by region
    quality_data = sensor_df.groupby(['region_name', 'quality_code']).size().unstack(fill_value=0)
    quality_data.plot(kind='bar', stacked=True, ax=axes[0,1])
    axes[0,1].set_title('Data Quality Distribution by Region')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Hourly patterns
    sensor_df['hour'] = sensor_df['measurement_timestamp'].dt.hour
    hourly_avg = sensor_df.groupby(['hour', 'parameter_type'])['measurement_value'].mean().unstack()
    hourly_avg.plot(ax=axes[1,0])
    axes[1,0].set_title('Average Hourly Sensor Patterns')
    axes[1,0].set_xlabel('Hour of Day')
    
    # Sensor modality distribution
    sensor_df['sensor_modality'].value_counts().plot(kind='pie', ax=axes[1,1], autopct='%1.1f%%')
    axes[1,1].set_title('Sensor Modality Distribution')
    
    plt.tight_layout()
    plt.savefig('dam_monitoring_project/analysis/figures/regional_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Generated publication-ready visualizations:")
    print("   analysis/figures/dam_infrastructure_overview.png")
    print("   analysis/figures/sensor_time_series.png") 
    print("   analysis/figures/correlation_heatmap.png")
    print("   analysis/figures/regional_comparison.png")

if __name__ == "__main__":
    create_research_visualizations() 