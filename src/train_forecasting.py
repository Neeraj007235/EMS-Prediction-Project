import pandas as pd
from prophet import Prophet
from src.data_preprocessing import load_data

def train_forecast(df):
    df_grouped = df.groupby(pd.Grouper(key='timeStamp', freq='H')).size().reset_index(name='calls')
    df_prophet = df_grouped.rename(columns={'timeStamp': 'ds', 'calls': 'y'})
    
    model = Prophet()
    model.fit(df_prophet)
    
    future = model.make_future_dataframe(periods=24, freq='H')
    forecast = model.predict(future)
    
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(24))
    
    return model

if __name__ == "__main__":
    df = load_data("./data/911.csv")
    model = train_forecast(df)
    