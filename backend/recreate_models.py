import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import joblib
import os
from textblob import TextBlob 

print("--- Starting Model and Scaler/Encoder Recreation ---")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')
MODEL_PATH = os.path.join(MODELS_DIR, 'virality_model.pkl')
OHE_PATH = os.path.join(MODELS_DIR, 'one_hot_encoder.pkl')

TRAINING_DATA_CSV_PATH = os.path.join(DATA_DIR, 'mock_social_virality_dataset_extended.csv')


EXPECTED_PLATFORMS = ['YouTube', 'Facebook', 'Instagram', 'X'] 

def parse_count_for_training_data(value):
    if pd.isna(value): return 0
    value = str(value).strip().upper()
    try:
        if 'K' in value: val = float(value.replace('K', '')) * 1_000
        elif 'M' in value: val = float(value.replace('M', '')) * 1_000_000
        elif 'B' in value: val = float(value.replace('B', '')) * 1_000_000_000
        else: val = float(value)
        return int(val)
    except ValueError: return 0


try:
    if not os.path.exists(TRAINING_DATA_CSV_PATH):
        raise FileNotFoundError(f"Training data CSV not found at: {TRAINING_DATA_CSV_PATH}")

    df_train = pd.read_csv(TRAINING_DATA_CSV_PATH)
    print(f"Loaded training data from: {TRAINING_DATA_CSV_PATH} with shape: {df_train.shape}")

  
    df_train['platform'] = df_train['platform'].replace('X (Twitter)', 'X')

    df_train['likes_parsed'] = df_train['likes'].apply(parse_count_for_training_data)
    df_train['views_parsed'] = df_train['views'].apply(parse_count_for_training_data)
    df_train['views_for_calc'] = df_train['views_parsed'].apply(lambda x: max(1, x))
    df_train['caption_length_derived'] = df_train['caption'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
    df_train['engagement_rate_derived'] = df_train.apply(
        lambda row: round(row['likes_parsed'] / (row['likes_parsed'] + row['views_for_calc']), 4)
        if (row['likes_parsed'] + row['views_for_calc']) > 0 else 0, axis=1
    )
    df_train['sentiment_score_derived'] = df_train['caption'].apply(
        lambda x: round(TextBlob(str(x)).sentiment.polarity, 3) if pd.notna(x) else 0
    )
    df_train['total_chars_derived'] = df_train['caption'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
    df_train['hashtag_count_derived'] = df_train['hashtags'].apply(
        lambda x: len([tag for tag in str(x).split() if tag.startswith('#')]) if pd.notna(x) else 0
    )

    df_train['subscribers'] = df_train.get('subscribers', 0) 
    df_train['channel_views'] = df_train.get('channel_views', 0) 

    numerical_features = [
        'caption_length_derived', 'engagement_rate_derived', 'sentiment_score_derived',
        'total_chars_derived', 'likes_parsed', 'views_parsed', 'hashtag_count_derived',
        'subscribers',        
        'channel_views'       
    ]
    categorical_features = ['platform']

    X_train_numerical_raw = df_train[numerical_features].values
    X_train_categorical_raw = df_train[categorical_features].values

    
    scaler = StandardScaler()
    scaler.fit(X_train_numerical_raw)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\n✅ Scaler (scaler.pkl) created and saved successfully at: {SCALER_PATH}")

    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False, categories=[EXPECTED_PLATFORMS])
    ohe.fit(X_train_categorical_raw)
    joblib.dump(ohe, OHE_PATH)
    print(f"✅ OneHotEncoder (one_hot_encoder.pkl) created and saved successfully at: {OHE_PATH}")

   
    X_train_numerical_scaled = scaler.transform(X_train_numerical_raw)
    X_train_categorical_encoded = ohe.transform(X_train_categorical_raw)
    
    X_train_processed = np.hstack((X_train_numerical_scaled, X_train_categorical_encoded))

    y_train = df_train['virality_score'].values 

    print(f"\nProcessed Features (X_train_processed) shape: {X_train_processed.shape}")
    print("First 5 rows of PROCESSED features (numerical scaled + categorical one-hot encoded):")
    print(X_train_processed[:5]) 
    print(f"Target (y_train) shape: {y_train.shape}")

    try:
        model = joblib.load(MODEL_PATH)
        print(f"\nLoaded existing model for retraining: {type(model).__name__}")
    except FileNotFoundError:
        print(f"\nWarning: Model file not found at {MODEL_PATH}. Initializing a new RandomForestRegressor.")
        model = RandomForestRegressor(random_state=42)

    print("Retraining the model on processed features...")
    model.fit(X_train_processed, y_train)
    print("Model retraining complete.")

    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model (virality_model.pkl) re-saved successfully at: {MODEL_PATH}")

    print("\n--- Model and Scaler/Encoder Recreation Complete ---")
    print("Next: Restart your Flask server and clear your browser cache.")

except FileNotFoundError as fnfe:
    print(f"Error: {fnfe}. Please ensure your training data CSV is correctly placed at {TRAINING_DATA_CSV_PATH}.")
except KeyError as ke:
    print(f"Error: Missing expected column in training data: {ke}. Ensure 'caption', 'likes', 'views', 'hashtags', 'platform' (and target 'virality_score') columns exist or are properly generated.")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    import traceback
    traceback.print_exc()