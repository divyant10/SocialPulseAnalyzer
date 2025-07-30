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

# <<< CRITICAL CHANGE HERE >>>
# The zip extracts to BASE_DIR/models/ which then contains another 'models' folder.
# So, the actual models are in BASE_DIR/models/models/
MODELS_DIR_EXTRACTED_ROOT = os.path.join(BASE_DIR, 'models') # This is /opt/render/project/src/models
MODELS_DIR = os.path.join(MODELS_DIR_EXTRACTED_ROOT, 'models') # <<< THIS IS THE CORRECT PATH TO YOUR PKL FILES

MODEL_PATH = os.path.join(MODELS_DIR, 'virality_model.pkl')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')
OHE_PATH = os.path.join(MODELS_DIR, 'one_hot_encoder.pkl')

# --- Google Drive Model Download Configuration ---
GOOGLE_DRIVE_MODEL_ZIP_ID = "1k2J7h4xGBdbN3DF9c01ylt8l_WJwvOOw"
MODEL_ZIP_PATH = os.path.join(BASE_DIR, "models.zip") # This is /opt/render/project/src/models.zip

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
    # Check if the main model file already exists in the FINAL expected location
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(OHE_PATH):
        print("INFO: Model files already exist. Skipping download and extraction.")
        return True

    # Ensure the root extraction directory exists (e.g., /opt/render/project/src/models)
    os.makedirs(MODELS_DIR_EXTRACTED_ROOT, exist_ok=True)

    print("🔽 Downloading model zip from Google Drive using gdown...")
    try:
        gdown.download(id=GOOGLE_DRIVE_MODEL_ZIP_ID, output=MODEL_ZIP_PATH, quiet=False)
        print(f"✅ Download complete: {MODEL_ZIP_PATH}")
    except Exception as e:
        print(f"❌ ERROR: Failed to download model zip: {e}")
        return False

    print(f"📦 Extracting models.zip to {MODELS_DIR_EXTRACTED_ROOT}...")
    try:
        # Extract all contents of the zip file into MODELS_DIR_EXTRACTED_ROOT
        with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(MODELS_DIR_EXTRACTED_ROOT)
        print("✅ Model files extracted.")
    except Exception as e:
        print(f"❌ ERROR: Failed to extract models.zip: {e}")
        return False

    # --- NEW DEBUGGING: Verify contents after extraction ---
    # Now check the contents of the *expected final directory*
    print(f"DEBUG: Verifying contents of {MODELS_DIR} (final model location) after extraction...")
    if os.path.exists(MODELS_DIR):
        extracted_files = os.listdir(MODELS_DIR)
        print(f"DEBUG: Files in {MODELS_DIR}: {extracted_files}")
        if 'virality_model.pkl' in extracted_files and \
           'scaler.pkl' in extracted_files and \
           'one_hot_encoder.pkl' in extracted_files:
            print("DEBUG: All expected model files found in FINAL MODELS_DIR.")
        else:
            print("❌ ERROR: Expected model files NOT found in FINAL MODELS_DIR after extraction.")
            print(f"DEBUG: Contents: {extracted_files}")
            return False
    else:
        print(f"❌ ERROR: FINAL MODELS_DIR ({MODELS_DIR}) does not exist after extraction attempt.")
        return False
    # --- END NEW DEBUGGING ---

    try:
        os.remove(MODEL_ZIP_PATH)
        print(f"🧹 Cleaned up models.zip from {MODEL_ZIP_PATH}.")
    except Exception as e:
        print(f"⚠️ WARNING: Could not remove models.zip: {e}")
    
    return True

# --- NLTK Data Download Function (unchanged) ---
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

# --- Function to Initialize/Load All Models and Data (unchanged) ---
def initialize_models_and_data():
    global model, scaler, one_hot_encoder

    if not download_and_extract_models():
        print("❌ FATAL ERROR: Model download/extraction failed. Application cannot start.")
        raise RuntimeError("Model download/extraction failed.")

    try:
        model = joblib.load(MODEL_PATH)
        print(f"✅ Loaded main model from {MODEL_PATH}")
        print(f"DEBUG: Type of loaded model: {type(model)}")

        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            print(f"✅ Loaded scaler from {SCALER_PATH}")
            print(f"DEBUG: Type of loaded scaler: {type(scaler)}")
        else:
            print(f"⚠️ Scaler not found at {SCALER_PATH}. Prediction might fail without it.")

        if os.path.exists(OHE_PATH):
            one_hot_encoder = joblib.load(OHE_PATH)
            print(f"✅ Loaded OneHotEncoder from {OHE_PATH}")
            print(f"DEBUG: Type of loaded OHE: {type(one_hot_encoder)}")
        else:
            print(f"⚠️ OneHotEncoder not found at {OHE_PATH}. Prediction might fail without it.")

        download_nltk_data()

    except Exception as e:
        print(f"❌ FATAL ERROR: Unable to load all required model components after extraction: {e}")
        raise


# --- Utility Functions (unchanged) ---

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
    
    if not all([model, scaler, one_hot_encoder]):
        print("❌ Missing model/scaler/encoder. Cannot predict virality. Attempting to re-initialize...")
        try:
            initialize_models_and_data()
            if not all([model, scaler, one_hot_encoder]):
                 raise RuntimeError("Models still not loaded after re-initialization attempt.")
        except Exception as e:
            print(f"❌ CRITICAL: Failed to load models for prediction: {e}")
            return 0

    try:
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
        scaled = scaler.transform(numerical)
        print(f"DEBUG: Predict - Numerical input to scaler: {numerical}")

        scaled = scaler.transform(numerical)
        print(f"DEBUG: Predict - Scaled numerical features: {scaled}")

        platform_clean = platform.strip().replace(' (Twitter)', '').strip()
        platform_encoded = one_hot_encoder.transform([[platform_clean]])
        if hasattr(platform_encoded, 'toarray'):
            platform_encoded = platform_encoded.toarray()
        print(f"DEBUG: Predict - Encoded platform features: {platform_encoded}")

        input_features = np.hstack((scaled, platform_encoded))
        print(f"DEBUG: Predict - Final input features to model: {input_features}")

        raw_score = model.predict(input_features)[0]
        print(f"DEBUG: Predict - Raw model prediction: {raw_score}")

        scaled_score = scale_virality_score(raw_score)
        print(f"DEBUG: Predict - Scaled score (0-100): {scaled_score}")

        final_score = adjust_score_heuristically(scaled_score, caption, hashtags)
        print(f"DEBUG: Predict - Final adjusted score: {final_score}")
        
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
