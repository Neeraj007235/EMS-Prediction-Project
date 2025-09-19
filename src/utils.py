# utils.py - Reusable helper functions for EMS Demand Prediction project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from datetime import datetime, timedelta
import os

def print_separator():
    """Print a separator line for console output"""
    print("="*40)

def calculate_response_time(row):
    """Simulate response time calculation based on location and time factors"""
    # This is a simplified model - in a real system, this would use actual distance calculations
    # and historical traffic data
    base_time = 5  # Base response time in minutes
    
    # Add time based on hour of day (rush hour penalty)
    hour = row['hour']
    if 7 <= hour <= 9 or 16 <= hour <= 18:  # Rush hours
        time_factor = 1.5
    else:
        time_factor = 1.0
    
    # Random variation (simulates traffic conditions, weather, etc.)
    random_factor = np.random.uniform(0.8, 1.2)
    
    return base_time * time_factor * random_factor

def generate_weekly_report(df, week_start_date=None):
    """Generate a summary report for a specific week"""
    if week_start_date is None:
        # Default to the most recent complete week in the dataset
        max_date = pd.to_datetime(df['timeStamp']).max()
        week_start_date = max_date - timedelta(days=max_date.weekday() + 7)
    
    week_end_date = week_start_date + timedelta(days=6)
    
    # Filter data for the specified week
    mask = (pd.to_datetime(df['timeStamp']) >= week_start_date) & \
           (pd.to_datetime(df['timeStamp']) <= week_end_date)
    week_df = df[mask].copy()
    
    if len(week_df) == 0:
        return "No data available for the specified week."
    
    # Calculate statistics
    total_calls = len(week_df)
    calls_by_day = week_df.groupby(pd.to_datetime(week_df['timeStamp']).dt.day_name()).size()
    calls_by_type = week_df['title'].apply(lambda x: x.split(':')[0]).value_counts().head(5)
    calls_by_township = week_df['twp'].value_counts().head(5)
    
    # Format the report
    report = f"Weekly Emergency Services Report: {week_start_date.strftime('%Y-%m-%d')} to {week_end_date.strftime('%Y-%m-%d')}\n\n"
    report += f"Total Calls: {total_calls}\n\n"
    
    report += "Calls by Day:\n"
    for day, count in calls_by_day.items():
        report += f"  {day}: {count} calls ({count/total_calls*100:.1f}%)\n"
    
    report += "\nTop 5 Emergency Types:\n"
    for etype, count in calls_by_type.items():
        report += f"  {etype}: {count} calls ({count/total_calls*100:.1f}%)\n"
    
    report += "\nTop 5 Townships:\n"
    for township, count in calls_by_township.items():
        report += f"  {township}: {count} calls ({count/total_calls*100:.1f}%)\n"
    
    return report

def optimize_resource_allocation(df, num_resources=10):
    """Optimize placement of emergency resources based on historical call data"""
    # This is a simplified model that places resources in areas with highest call density
    # A real system would use more sophisticated algorithms considering time, road networks, etc.
    
    # Group data by location (using lat/lng rounded to 2 decimal places for simplicity)
    df['lat_rounded'] = np.round(df['lat'], 2)
    df['lng_rounded'] = np.round(df['lng'], 2)
    
    location_counts = df.groupby(['lat_rounded', 'lng_rounded']).size().reset_index(name='call_count')
    
    # Sort by call count and get top locations
    top_locations = location_counts.sort_values('call_count', ascending=False).head(num_resources)
    
    # Create a map with optimal resource locations
    m = folium.Map(location=[df['lat'].mean(), df['lng'].mean()], zoom_start=10)
    
    # Add markers for each optimal location
    for idx, row in top_locations.iterrows():
        folium.Marker(
            location=[row['lat_rounded'], row['lng_rounded']],
            popup=f"Recommended resource location<br>Call volume: {row['call_count']}",
            icon=folium.Icon(color='green', icon='ambulance', prefix='fa')
        ).add_to(m)
    
    # Save the map
    output_path = 'resource_optimization.html'
    m.save(output_path)
    
    return top_locations, output_path

def simulate_emergency_call(df):
    """Simulate a new emergency call based on historical patterns"""
    # Sample a random row from the dataset as a template
    random_call = df.sample(1).iloc[0]
    
    # Create a simulated call with current timestamp
    current_time = datetime.now()
    
    # Add some random variation to location (within ~1km)
    lat_variation = np.random.uniform(-0.01, 0.01)
    lng_variation = np.random.uniform(-0.01, 0.01)
    
    simulated_call = {
        'type': random_call['title'],
        'lat': random_call['lat'] + lat_variation,
        'lng': random_call['lng'] + lng_variation,
        'location': f"{random_call['twp']}, {random_call['addr']}",
        'time': current_time.strftime("%Y-%m-%d %H:%M:%S"),
        'priority': np.random.choice([1, 2, 3], p=[0.2, 0.5, 0.3])  # 1=highest priority
    }
    
    return simulated_call

def get_nearest_resources(call_location, resource_locations, num_nearest=3):
    """Find the nearest emergency resources to a call location"""
    # Calculate distances from call to each resource
    distances = []
    for idx, resource in resource_locations.iterrows():
        # Simple Euclidean distance - a real system would use road networks
        dist = np.sqrt((call_location['lat'] - resource['lat_rounded'])**2 + 
                      (call_location['lng'] - resource['lng_rounded'])**2)
        distances.append((idx, dist))
    
    # Sort by distance and get the nearest resources
    nearest_resources = sorted(distances, key=lambda x: x[1])[:num_nearest]
    
    return [resource_locations.iloc[idx] for idx, _ in nearest_resources]
