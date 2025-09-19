import pandas as pd

def load_data(filepath):
    df = pd.read_csv(filepath)
    df['timeStamp'] = pd.to_datetime(df['timeStamp'])
    
    # Extract datetime features
    df['hour'] = df['timeStamp'].dt.hour
    df['day'] = df['timeStamp'].dt.day
    df['month'] = df['timeStamp'].dt.month
    df['weekday'] = df['timeStamp'].dt.weekday
    
    # Extract emergency type from title
    df['emergency_type'] = df['title'].apply(lambda x: x.split(':')[0])
    
    return df

if __name__ == "__main__":
    df = load_data("./data/911.csv")
    print(df.head())
