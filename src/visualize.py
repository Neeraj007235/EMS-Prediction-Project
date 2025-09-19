import folium
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_heatmap(df):
    # If df has fewer than 5000 rows, use all rows, otherwise sample 5000
    if len(df) < 5000:
        sample_df = df
    else:
        sample_df = df.sample(n=5000, random_state=42)  # sample 5000 rows for faster plotting
    
    m = folium.Map(location=[sample_df['lat'].mean(), sample_df['lng'].mean()], zoom_start=10)
    
    for _, row in sample_df.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lng']],
            radius=3,
            fill=True,
            color='red',
            fill_opacity=0.6
        ).add_to(m)
    
    m.save('heatmap.html')
    print("Heatmap saved as heatmap.html")

def plot_calls_over_time(df):
    if 'hour' not in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df['timeStamp']):
            df['hour'] = df['timeStamp'].dt.hour
        else:
            df['hour'] = pd.to_datetime(df['timeStamp']).dt.hour

    calls_per_hour = df.groupby('hour').size()
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=calls_per_hour.index, y=calls_per_hour.values)
    plt.title('Number of Calls by Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Calls')
    plt.grid(True)
    
    fig = plt.gcf()
    return fig

if __name__ == "__main__":
    df = pd.read_csv("./data/911.csv", nrows=10000)  # load 10k rows for testing
    print(df.head())

    plot_heatmap(df)
    plot_calls_over_time(df)
