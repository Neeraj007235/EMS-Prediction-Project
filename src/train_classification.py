import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from src.data_preprocessing import load_data

def train_classifier(df):
    # Extract emergency type from title if not already done
    if 'emergency_type' not in df.columns:
        df['emergency_type'] = df['title'].apply(lambda x: x.split(':')[0])
    
    features = ['hour', 'day', 'month', 'weekday', 'zip']
    X = df[features].copy()
    y = df['emergency_type']
    
    # Handle missing or non-numeric zip codes
    X['zip'] = pd.to_numeric(X['zip'], errors='coerce').fillna(0).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=1)  # smaller + no parallel
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    return clf

if __name__ == "__main__":
    df = load_data("./data/911.csv")
    print(f"Dataset shape: {df.shape}")
    df_sample = df.sample(frac=0.1, random_state=42)  # sample 10% for quick run
    model = train_classifier(df_sample)
