import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import time

from src.data_preprocessing import load_data
from src.visualize import plot_calls_over_time, plot_heatmap
from src.train_classification import train_classifier
from src.train_forecasting import train_forecast

st.set_page_config(layout="wide", page_title="EMS Demand Prediction")

st.title("Emergency Medical Service (EMS) Demand Prediction Dashboard")

# Add sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page",
    ["Overview", "Historical Analysis", "Heatmap", "Predictions", "Real-time Simulation"]
)

@st.cache_data
def get_data():
    # Load and preprocess data
    df = load_data("data/911.csv")
    return df

# Load the data
df = get_data()

# Overview page
if page == "Overview":
    st.header("Project Overview")
    st.write("""
    This dashboard analyzes emergency service call data to help optimize resource allocation 
    and response times. By understanding patterns in emergency calls, we can better predict 
    future demand and place resources where they're most needed.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Information")
        st.write(f"Total number of emergency calls: {len(df)}")
        st.write(f"Date range: {df['timeStamp'].min()} to {df['timeStamp'].max()}")
        st.write(f"Number of townships: {df['twp'].nunique()}")
        
    with col2:
        st.subheader("Emergency Types")
        emergency_counts = df['title'].apply(lambda x: x.split(':')[0]).value_counts().head(10)
        fig, ax = plt.subplots(figsize=(10, 6))
        emergency_counts.plot(kind='bar', ax=ax)
        plt.title('Distribution of All Emergency Types in the Dataset')
        plt.xlabel('Emergency Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        st.pyplot(fig)

# Historical Analysis page
elif page == "Historical Analysis":
    st.header("Historical Call Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Emergency Calls by Hour of Day")
        fig = plot_calls_over_time(df)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Emergency Calls by Day of Week")
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        calls_by_day = df.groupby('weekday').size()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=calls_by_day.index, y=calls_by_day.values, ax=ax)
        plt.title('Number of Calls by Day of Week')
        plt.xlabel('Day of Week')
        plt.ylabel('Number of Calls')
        plt.xticks(ticks=range(7), labels=days, rotation=45)
        st.pyplot(fig)
    
    st.subheader("Emergency Calls by Township")
    top_townships = df['twp'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(12, 6))
    top_townships.plot(kind='bar', ax=ax)
    plt.title('Top 10 Townships by Number of Calls')
    plt.xlabel('Township')
    plt.ylabel('Number of Calls')
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Heatmap page
elif page == "Heatmap":
    st.header("Emergency Call Locations")
    st.write("This map shows the geographic distribution of emergency calls.")
    
    # Create a sample of the data for faster rendering
    if len(df) > 1000:
        map_data = df.sample(n=1000, random_state=42)
    else:
        map_data = df
    
    # Create a folium map centered on the mean coordinates
    m = folium.Map(location=[map_data['lat'].mean(), map_data['lng'].mean()], zoom_start=10)
    
    # Add markers for each emergency call
    for idx, row in map_data.iterrows():
        emergency_type = row['title'].split(':')[0]
        folium.CircleMarker(
            location=[row['lat'], row['lng']],
            radius=5,
            popup=f"Type: {row['title']}<br>Location: {row['twp']}, {row['addr']}",
            fill=True,
            color='red',
            fill_opacity=0.7
        ).add_to(m)
    
    # Display the map
    folium_static(m)

# Predictions page
elif page == "Predictions":
    st.header("EMS Demand Predictions")
    
    tab1, tab2 = st.tabs(["Emergency Type Prediction", "Call Volume Forecasting"])
    
    with tab1:
        st.subheader("Predict Emergency Type Based on Location and Time")
        st.write("This model predicts the most likely type of emergency based on location and time factors.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            hour = st.slider("Hour of Day", 0, 23, 12)
            day = st.slider("Day of Month", 1, 31, 15)
            month = st.slider("Month", 1, 12, 6)
            weekday = st.slider("Day of Week (0=Monday, 6=Sunday)", 0, 6, 2)
        
        with col2:
            # Get unique zip codes from the dataset
            zip_codes = df['zip'].dropna().astype(str).unique()
            zip_code = st.selectbox("ZIP Code", options=zip_codes)
            
            if st.button("Predict Emergency Type"):
                with st.spinner("Training model and making prediction..."):
                    # Train a small model for demo purposes
                    sample_df = df.sample(frac=0.1, random_state=42)
                    model = train_classifier(sample_df)
                    
                    # Make prediction
                    try:
                        prediction_input = pd.DataFrame({
                            'hour': [hour],
                            'day': [day],
                            'month': [month],
                            'weekday': [weekday],
                            'zip': [int(zip_code) if zip_code.isdigit() else 0]
                        })
                        
                        prediction = model.predict(prediction_input)[0]
                        confidence = np.max(model.predict_proba(prediction_input)[0]) * 100
                        
                        st.success(f"Predicted Emergency Type: **{prediction}**")
                        st.write(f"Confidence: {confidence:.2f}%")
                    except Exception as e:
                        st.error(f"Error making prediction: {e}")
    
    with tab2:
        st.subheader("Call Volume Forecast")
        st.write("This model forecasts the expected number of emergency calls in the next 24 hours.")
        
        if st.button("Generate Forecast"):
            with st.spinner("Training forecasting model..."):
                try:
                    # Use a sample for faster demo
                    sample_df = df.sample(frac=0.2, random_state=42)
                    model = train_forecast(sample_df)
                    
                    # Generate forecast for next 24 hours
                    future = model.make_future_dataframe(periods=24, freq='H')
                    forecast = model.predict(future)
                    
                    # Plot the forecast
                    fig = plt.figure(figsize=(12, 6))
                    plt.plot(forecast['ds'][-48:], forecast['yhat'][-48:], label='Forecast')
                    plt.fill_between(forecast['ds'][-48:], 
                                    forecast['yhat_lower'][-48:], 
                                    forecast['yhat_upper'][-48:], 
                                    alpha=0.3, color='blue')
                    plt.title('Emergency Call Volume Forecast (Next 24 Hours)')
                    plt.xlabel('Time')
                    plt.ylabel('Number of Calls')
                    plt.legend()
                    plt.grid(True)
                    st.pyplot(fig)
                    
                    # Show forecast table
                    st.subheader("Hourly Forecast Values")
                    forecast_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(24)
                    forecast_table.columns = ['Time', 'Predicted Calls', 'Lower Bound', 'Upper Bound']
                    st.dataframe(forecast_table)
                except Exception as e:
                    st.error(f"Error generating forecast: {e}")

# Real-time Simulation page
elif page == "Real-time Simulation":
    st.header("Real-time Emergency Call Simulation")
    st.write("""
    This simulation shows how emergency calls might come in real-time. 
    The system updates every few seconds with new simulated emergency calls.
    """)
    
    # Parameters for simulation
    simulation_speed = st.slider("Simulation Speed (seconds between calls)", 1, 10, 3)
    max_calls = st.slider("Maximum Calls to Simulate", 5, 50, 20)
    
    # Button to start/stop simulation
    start_button = st.button("Start Simulation")
    
    if start_button:
        # Create a placeholder for the map
        map_placeholder = st.empty()
        stats_placeholder = st.empty()
        calls_placeholder = st.empty()
        
        # Create a base map
        m = folium.Map(location=[df['lat'].mean(), df['lng'].mean()], zoom_start=10)
        
        # Initialize simulation data
        simulated_calls = []
        emergency_types = df['title'].apply(lambda x: x.split(':')[0]).unique()
        townships = df['twp'].unique()
        
        # Run simulation
        for i in range(max_calls):
            # Sample a random row from the dataset
            random_call = df.sample(1).iloc[0]
            
            # Create a simulated call with current timestamp
            current_time = datetime.now()
            simulated_call = {
                'id': i + 1,
                'type': random_call['title'],
                'lat': random_call['lat'],
                'lng': random_call['lng'],
                'location': f"{random_call['twp']}, {random_call['addr']}",
                'time': current_time.strftime("%H:%M:%S")
            }
            
            simulated_calls.append(simulated_call)
            
            # Add marker to map
            folium.CircleMarker(
                location=[simulated_call['lat'], simulated_call['lng']],
                radius=8,
                popup=f"ID: {simulated_call['id']}<br>Type: {simulated_call['type']}<br>Location: {simulated_call['location']}",
                fill=True,
                color='red',
                fill_opacity=0.7
            ).add_to(m)
            
            # Update the map
            with map_placeholder:
                folium_static(m)
            
            # Update statistics
            with stats_placeholder:
                col1, col2, col3 = st.columns(3)
                col1.metric("Active Calls", len(simulated_calls))
                col2.metric("Average Lat", f"{sum(call['lat'] for call in simulated_calls) / len(simulated_calls):.4f}")
                col3.metric("Average Lng", f"{sum(call['lng'] for call in simulated_calls) / len(simulated_calls):.4f}")
            
            # Show calls table
            with calls_placeholder:
                st.subheader("Recent Emergency Calls")
                calls_df = pd.DataFrame(simulated_calls)
                st.dataframe(calls_df[['id', 'type', 'location', 'time']].sort_values(by='id', ascending=False))
            
            # Wait for next call
            time.sleep(simulation_speed)
