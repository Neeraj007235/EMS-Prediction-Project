# Emergency Medical Service (EMS) Demand Prediction and Optimization

This project analyzes historical 911 emergency call data to detect patterns, predict high-risk time zones, and optimize ambulance/staff deployment in real-time or per day/week.

## 🎯 Project Overview

Emergency services (ambulance, fire, etc.) often suffer from uneven demand, delays, and resource misallocation. This project aims to solve these problems by:

- Analyzing historical emergency call data to identify patterns
- Predicting high-risk zones and times for emergency incidents
- Optimizing resource allocation to improve response times
- Providing interactive visualizations and simulations

## 📊 Features

1. **Historical Data Analysis**
   - Analyze emergency calls by time, location, and type
   - Identify peak times and high-risk zones

2. **Interactive Heatmap**
   - Visualize emergency call hotspots on an interactive map

3. **Predictive Models**
   - Predict emergency types based on location and time
   - Forecast call volumes for future planning

4. **Real-time Simulation**
   - Simulate incoming emergency calls
   - Visualize real-time resource allocation

5. **Resource Optimization**
   - Recommend optimal placement of emergency resources
   - Generate weekly summary reports

## 🧰 Tech Stack

- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, prophet, xgboost
- **Geo Analysis**: folium, geopandas
- **Visualization**: matplotlib, seaborn
- **Dashboard**: streamlit, streamlit-folium

## 📦 Dataset

This project uses the Montgomery County, PA 911 Emergency Calls dataset, which contains information about emergency calls including:

- Timestamps
- Geographic coordinates
- Emergency types
- Locations (townships, addresses)

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. Clone the repository

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit dashboard:
   ```
   streamlit run app.py
   ```

## 📋 Project Structure

```
EMS-Demand-Prediction/
│
├── data/
│   └── 911.csv               # Emergency calls dataset
│
├── src/
│   ├── data_preprocessing.py  # Data loading and preprocessing
│   ├── train_classification.py # Emergency type prediction model
│   ├── train_forecasting.py    # Call volume forecasting model
│   ├── visualize.py            # Visualization functions
│   └── utils.py                # Utility functions
│
├── app.py                     # Streamlit dashboard
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## 📈 Dashboard Pages

1. **Overview**: General project information and dataset statistics
2. **Historical Analysis**: Visualizations of historical emergency call patterns
3. **Heatmap**: Geographic distribution of emergency calls
4. **Predictions**: Emergency type prediction and call volume forecasting
5. **Real-time Simulation**: Simulation of incoming emergency calls

## 🔮 Future Enhancements

- Integration with real-time traffic data for more accurate response time estimation
- Advanced machine learning models for more precise predictions
- Mobile app for field responders
- Integration with hospital capacity data for optimal patient routing

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgements

- Montgomery County, PA for providing the emergency call dataset
- The open-source community for the amazing tools and libraries used in this project