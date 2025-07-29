import joblib
import re
import os
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
from uuid import uuid4
import pandas as pd
import random
import zipfile
import gdown
import nltk


# --- Base Directories Configuration ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'virality_model.pkl')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')
OHE_PATH = os.path.join(MODELS_DIR, 'one_hot_encoder.pkl')

# --- Google Drive Model Download Configuration ---
GOOGLE_DRIVE_MODEL_ZIP_ID = "1k2J7h4xGBdbN3DF9c01ylt8l_WJwvOOw"
MODEL_ZIP_PATH = os.path.join(BASE_DIR, "models.zip")

# --- NLTK Data Configuration (if needed) ---
NLTK_DATA_DIR = os.path.join('/tmp', 'nltk_data')


# --- Global Model Variables (will be loaded by the init function) ---
model = None
scaler = None
one_hot_encoder = None

# --- Constants from your original analyzer.py ---
EXPECTED_PLATFORMS = ['YouTube', 'Facebook', 'Instagram', 'X']
TRENDING_HASHTAGS = [
    "#trending", "#viral", "#reels", "#explore", "#instagood",
    "#socialmedia", "#influencer", "#contentcreator", "#marketing", "#growth",
    "#fyp", "#foryou", "#discover", "#instadaily", "#photooftheday"
]


# --- Function to Download and Extract Models ---
def download_and_extract_models():
    """
    Downloads the model zip from Google Drive and extracts it.
    Checks if model files already exist to avoid redundant downloads.
    """
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(OHE_PATH):
        print("INFO: Model files already exist. Skipping download and extraction.")
        return True

    os.makedirs(MODELS_DIR, exist_ok=True)

    print("🔽 Downloading model zip from Google Drive using gdown...")
    try:
        gdown.download(id=GOOGLE_DRIVE_MODEL_ZIP_ID, output=MODEL_ZIP_PATH, quiet=False)
        print(f"✅ Download complete: {MODEL_ZIP_PATH}")
    except Exception as e:
        print(f"❌ ERROR: Failed to download model zip: {e}")
        return False

    print(f"📦 Extracting models.zip to {MODELS_DIR}...")
    try:
        with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(MODELS_DIR)
        print("✅ Model files extracted.")
    except Exception as e:
        print(f"❌ ERROR: Failed to extract models.zip: {e}")
        return False

    try:
        os.remove(MODEL_ZIP_PATH)
        print(f"🧹 Cleaned up models.zip from {MODEL_ZIP_PATH}.")
    except Exception as e:
        print(f"⚠️ WARNING: Could not remove models.zip: {e}")
    
    return True

# --- NLTK Data Download Function ---
def download_nltk_data():
    """
    Downloads NLTK data required by TextBlob if not already present.
    """
    try:
        os.makedirs(NLTK_DATA_DIR, exist_ok=True)
        nltk.data.path.append(NLTK_DATA_DIR)

        try:
            nltk.data.find('tokenizers/punkt', paths=[NLTK_DATA_DIR])
            print("INFO: NLTK 'punkt' data already exists.")
        except LookupError:
            print("DEBUG: Downloading NLTK 'punkt' data...")
            nltk.download('punkt', download_dir=NLTK_DATA_DIR, quiet=False)
            print("DEBUG: NLTK 'punkt' data downloaded.")

    except ImportError:
        print("WARNING: NLTK not installed or not used. Skipping NLTK data download.")
    except Exception as e:
        print(f"❌ ERROR: Failed to download NLTK data: {e}")

# --- Function to Initialize/Load All Models and Data ---
def initialize_models_and_data():
    """
    This function will be called explicitly by app.py after startup.
    It ensures models are downloaded/extracted before attempting to load them.
    """
    global model, scaler, one_hot_encoder # Declare global to modify them

    # Attempt to download and extract models
    if not download_and_extract_models():
        print("❌ FATAL ERROR: Model download/extraction failed. Application cannot start.")
        raise RuntimeError("Model download/extraction failed.")

    # Now, attempt to load the models
    try:
        model = joblib.load(MODEL_PATH)
        print(f"✅ Loaded main model from {MODEL_PATH}")
        print(f"DEBUG: Type of loaded model: {type(model)}") # NEW DEBUG

        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            print(f"✅ Loaded scaler from {SCALER_PATH}")
            print(f"DEBUG: Type of loaded scaler: {type(scaler)}") # NEW DEBUG
        else:
            print(f"⚠️ Scaler not found at {SCALER_PATH}. Prediction might fail without it.")

        if os.path.exists(OHE_PATH):
            one_hot_encoder = joblib.load(OHE_PATH)
            print(f"✅ Loaded OneHotEncoder from {OHE_PATH}")
            print(f"DEBUG: Type of loaded OHE: {type(one_hot_encoder)}") # NEW DEBUG
        else:
            print(f"⚠️ OneHotEncoder not found at {OHE_PATH}. Prediction might fail without it.")

        # Download NLTK data after models are ready
        download_nltk_data()

    except Exception as e:
        print(f"❌ FATAL ERROR: Unable to load all required model components after extraction: {e}")
        raise

# --- Utility Functions ---

def parse_count(value):
    value = str(value).strip().upper()
    try:
        if 'K' in value:
            val = float(value.replace('K', '')) * 1_000
        elif 'M' in value:
            val = float(value.replace('M', '')) * 1_000_000
        elif 'B' in value:
            val = float(value.replace('B', '')) * 1_000_000_000
        else:
            val = float(value)
        return int(val)
    except ValueError:
        return 0

def scale_virality_score(raw_score):
    min_raw = -1.24
    max_raw = 713.93
    if abs(max_raw - min_raw) < 1e-6:
        return 0
    scaled = (raw_score - min_raw) / (max_raw - min_raw)
    return round(np.clip(scaled, 0, 1) * 100, 2)

def adjust_score_heuristically(score, caption, hashtags):
    sentiment = TextBlob(caption).sentiment.polarity
    if score < 10 and sentiment > 0.1:
        score = max(score, 10)

    if 20 < len(caption.split()) < 100:
        score += 2
    elif len(caption.split()) > 100:
        score -= 5

    if hashtags and hashtags.count('#') > 3:
        score += 1

    if any(word in caption.lower() for word in ['amazing', 'viral', 'trending', 'challenge', 'must-see']):
        score += 2

    return int(np.clip(score, 0, 100))


def predict_virality(caption, likes, views, hashtags, platform, subscribers, channel_views):
    print(f"DEBUG: predict_virality input: likes={likes}, views={views}, subscribers={subscribers}, channel_views={channel_views}")
    
    # Check if models are loaded. This check is crucial now.
    if not all([model, scaler, one_hot_encoder]):
        print("❌ Missing model/scaler/encoder. Cannot predict virality. Attempting to re-initialize...")
        try:
            initialize_models_and_data() # Try to load them if not already loaded
            if not all([model, scaler, one_hot_encoder]): # Check again after attempt
                 raise RuntimeError("Models still not loaded after re-initialization attempt.")
        except Exception as e:
            print(f"❌ CRITICAL: Failed to load models for prediction: {e}")
            return 0

    try:
        # --- NEW DEBUGGING PRINTS FOR INPUTS AND TRANSFORMATIONS ---
        print(f"DEBUG: Predict - Caption length: {len(caption.split())}")
        print(f"DEBUG: Predict - Engagement rate: {round(likes / max(likes + max(1, views), 1), 4)}")
        print(f"DEBUG: Predict - Sentiment: {round(TextBlob(caption).sentiment.polarity, 3)}")
        print(f"DEBUG: Predict - Hashtag count: {len([tag for tag in hashtags.split() if tag.startswith('#')])}")
        print(f"DEBUG: Predict - Platform raw: '{platform}'")

        engagement_rate = round(likes / max(likes + max(1, views), 1), 4)
        sentiment = round(TextBlob(caption).sentiment.polarity, 3)
        hashtag_count = len([tag for tag in hashtags.split() if tag.startswith('#')])

        numerical = np.array([[
            len(caption.split()), engagement_rate, sentiment, len(caption),
            likes, views, hashtag_count, subscribers, channel_views
        ]])
        print(f"DEBUG: Predict - Numerical input to scaler: {numerical}") # NEW DEBUG

        scaled = scaler.transform(numerical)
        print(f"DEBUG: Predict - Scaled numerical features: {scaled}") # NEW DEBUG

        platform_clean = platform.strip().replace(' (Twitter)', '').strip()
        platform_encoded = one_hot_encoder.transform([[platform_clean]])
        if hasattr(platform_encoded, 'toarray'):
            platform_encoded = platform_encoded.toarray()
        print(f"DEBUG: Predict - Encoded platform features: {platform_encoded}") # NEW DEBUG

        input_features = np.hstack((scaled, platform_encoded))
        print(f"DEBUG: Predict - Final input features to model: {input_features}") # NEW DEBUG

        raw_score = model.predict(input_features)[0]
        print(f"DEBUG: Predict - Raw model prediction: {raw_score}") # NEW DEBUG

        scaled_score = scale_virality_score(raw_score)
        print(f"DEBUG: Predict - Scaled score (0-100): {scaled_score}") # NEW DEBUG

        final_score = adjust_score_heuristically(scaled_score, caption, hashtags)
        print(f"DEBUG: Predict - Final adjusted score: {final_score}") # NEW DEBUG
        
        return final_score

    except Exception as e:
        print(f"❌ Error in predict_virality: {e}")
        import traceback
        traceback.print_exc()
        return 0


def analyze_sentiment_distribution(text):
    try:
        nltk.data.find('tokenizers/punkt', paths=[NLTK_DATA_DIR])
    except LookupError:
        print("DEBUG: NLTK data not found for TextBlob. Attempting to download...")
        download_nltk_data()

    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > 0.2:
        overall = "Positive"
    elif polarity < -0.2:
        overall = "Negative"
    else:
        overall = "Neutral"

    pos, neg, neu = 0, 0, 0
    for word in text.split():
        pol = TextBlob(word).sentiment.polarity
        if pol > 0.2:
            pos += 1
        elif pol < -0.2:
            neg += 1
        else:
            neu += 1

    chart_dir = os.path.join(BASE_DIR, 'frontend', 'static', 'charts')
    os.makedirs(chart_dir, exist_ok=True)
    graph_id = uuid4().hex[:6]
    graph_path = os.path.join(chart_dir, f"sentiment_{graph_id}.png")

    plt.figure(figsize=(5, 3))
    plt.bar(['Positive', 'Negative', 'Neutral'], [pos, neg, neu], color=['#4CAF50', '#F44336', '#9E9E9E'])
    plt.title('Sentiment Analysis')
    plt.ylabel('Word Count')
    plt.tight_layout()
    plt.savefig(graph_path)
    plt.close()

    return {
        "overall_sentiment": overall,
        "graph_data": {"labels": ['Positive', 'Negative', 'Neutral'], "scores": [pos, neg, neu]}
    }


def get_caption_suggestions(original):
    if "handsome" in original.lower():
        return [
            "Feeling confident and handsome today! #confidence #style",
            "Handsome vibes only. ✨ #mensfashion #goodlooks",
            "Just a handsome guy doing handsome things. 😉 #instastyle"
        ]
    return [
        f"A great moment captured!",
        "Express yourself with confidence!",
        "Share your story!"
    ]


def analyze_hashtag_effectiveness(hashtags_str):
    tips = []
    if not hashtags_str:
        return [
            "Consider adding relevant hashtags for better reach.",
            "Research trending hashtags in your niche."
        ]

    tags = [tag.strip() for tag in hashtags_str.split() if tag.startswith('#')]
    plain = [tag[1:].lower() for tag in tags]

    if len(tags) < 3:
        tips.append("Use at least 3–5 relevant hashtags.")
    elif len(tags) > 15:
        tips.append("Too many hashtags (over 15) may look spammy.")

    if len(set(plain)) < len(plain):
        tips.append("Avoid repeating hashtags.")

    for tag in tags:
        text = tag[1:]
        if len(text) <= 2 or not text.isalpha():
            continue
        if text.isupper() and len(text) > 5:
            tips.append(f"Consider #CamelCase like #{text.title()}")
            break
        elif not (text.islower() or text.isupper()):
            tips.append(f"Consider consistent casing in {tag}")
            break

    if len(tags) < 15:
        trending = random.sample(TRENDING_HASHTAGS, k=min(3, len(TRENDING_HASHTAGS)))
        tips.append("Try mixing niche & trending hashtags: " + ", ".join(trending))

    return tips if tips else ["Hashtag usage looks good!"]
