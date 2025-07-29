import joblib
import re
import os
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt # Keep if you are generating and serving charts
from uuid import uuid4
import pandas as pd
import random
import zipfile
# import requests # <<< REMOVED: Replaced by gdown for Google Drive downloads
import gdown # <<< ADDED: For reliable Google Drive downloads
import nltk # Assuming you use NLTK for something, otherwise can remove related parts


# --- Base Directories Configuration ---
# BASE_DIR should point to the root of your SocialPulseAnalyzer project
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # This correctly points to SocialPulseAnalyzer/

# Directory where models are expected to be extracted (e.g., SocialPulseAnalyzer/models/)
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Paths for individual model components
MODEL_PATH = os.path.join(MODELS_DIR, 'virality_model.pkl')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')
OHE_PATH = os.path.join(MODELS_DIR, 'one_hot_encoder.pkl')

# --- Google Drive Model Download Configuration ---
# !! IMPORTANT !! Use the correct Google Drive File ID for your models.zip
# This ID (1k2J7h4xGBdbN3DF9c01ylt8l_WJwvOOw) is from your analyzer.py
GOOGLE_DRIVE_MODEL_ZIP_ID = "1k2J7h4xGBdbN3DF9c01ylt8l_WJwvOOw"
# Temporary path for the downloaded zip file before extraction
MODEL_ZIP_PATH = os.path.join(BASE_DIR, "models.zip") # Will download to SocialPulseAnalyzer/models.zip

# --- NLTK Data Configuration (if needed) ---
NLTK_DATA_DIR = os.path.join('/tmp', 'nltk_data') # NLTK data also goes into ephemeral storage


# --- Global Model Variables (will be loaded at startup) ---
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


# --- Function to Download and Extract Models (Core of Free-Tier Strategy) ---
def download_and_extract_models():
    """
    Downloads the model zip from Google Drive and extracts it.
    Checks if model files already exist to avoid redundant downloads.
    """
    # Check if the main model file already exists. If yes, assume other components are also there.
    # This prevents re-downloading on container restarts that don't clear /tmp.
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(OHE_PATH):
        print("INFO: Model files already exist. Skipping download and extraction.")
        return

    # Ensure the directory for extracted models exists (e.g., SocialPulseAnalyzer/models/)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("🔽 Downloading model zip from Google Drive using gdown...")
    try:
        # Use gdown for more reliable Google Drive downloads for large files
        gdown.download(id=GOOGLE_DRIVE_MODEL_ZIP_ID, output=MODEL_ZIP_PATH, quiet=False)
        print(f"✅ Download complete: {MODEL_ZIP_PATH}")
    except Exception as e:
        print(f"❌ ERROR: Failed to download model zip: {e}")
        # Re-raise the exception to prevent the application from starting without the model
        raise

    print(f"📦 Extracting models.zip to {MODELS_DIR}...")
    try:
        with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
            # Extract all contents of the zip file into MODELS_DIR
            zip_ref.extractall(MODELS_DIR)
        print("✅ Model files extracted.")
    except Exception as e:
        print(f"❌ ERROR: Failed to extract models.zip: {e}")
        # Re-raise the exception if extraction fails
        raise

    # Clean up the downloaded zip file to save ephemeral disk space
    try:
        os.remove(MODEL_ZIP_PATH)
        print(f"🧹 Cleaned up models.zip from {MODEL_ZIP_PATH}.")
    except Exception as e:
        # This is a warning, not critical if cleanup fails
        print(f"⚠️ WARNING: Could not remove models.zip: {e}")

# --- NLTK Data Download Function ---
def download_nltk_data():
    """
    Downloads NLTK data required by TextBlob if not already present.
    """
    try:
        # Create the NLTK data directory if it doesn't exist
        os.makedirs(NLTK_DATA_DIR, exist_ok=True)
        # Add this path to NLTK's data search paths
        nltk.data.path.append(NLTK_DATA_DIR)

        # Check if 'punkt' tokenizer data is needed for TextBlob (common dependency)
        # You might need to add other NLTK datasets here if your TextBlob usage requires them
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
        # Decide if this is critical for your app's functionality

# --- Initial Model Loading Block ---
# This block runs once when analyzer.py is imported (i.e., when your Flask app starts)
try:
    # First, download and extract the models
    download_and_extract_models()

    # Then, load the scikit-learn model components
    model = joblib.load(MODEL_PATH)
    print(f"✅ Loaded main model from {MODEL_PATH}")

    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        print(f"✅ Loaded scaler from {SCALER_PATH}")
    else:
        print(f"⚠️ Scaler not found at {SCALER_PATH}. Prediction might fail without it.")

    if os.path.exists(OHE_PATH):
        one_hot_encoder = joblib.load(OHE_PATH)
        print(f"✅ Loaded OneHotEncoder from {OHE_PATH}")
    else:
        print(f"⚠️ OneHotEncoder not found at {OHE_PATH}. Prediction might fail without it.")

    # Download NLTK data
    download_nltk_data()

except Exception as e:
    # This fatal error will cause Render deployment to fail, which is desired if models are missing
    print(f"❌ FATAL ERROR: Unable to initialize application due to missing/corrupt model components: {e}")
    raise # Re-raise the exception to signal a critical failure

# --- Your Existing Utility Functions ---

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
        print("❌ Missing model/scaler/encoder. Cannot predict virality.")
        return 0 # Or raise an error if this state indicates a critical failure

    try:
        engagement_rate = round(likes / max(likes + max(1, views), 1), 4)
        sentiment = round(TextBlob(caption).sentiment.polarity, 3)
        hashtag_count = len([tag for tag in hashtags.split() if tag.startswith('#')])

        numerical = np.array([[
            len(caption.split()), engagement_rate, sentiment, len(caption),
            likes, views, hashtag_count, subscribers, channel_views
        ]])
        scaled = scaler.transform(numerical)

        platform_clean = platform.strip().replace(' (Twitter)', '').strip()
        platform_encoded = one_hot_encoder.transform([[platform_clean]])
        if hasattr(platform_encoded, 'toarray'): # For sparse matrices from older OHE versions
            platform_encoded = platform_encoded.toarray()

        input_features = np.hstack((scaled, platform_encoded))

        raw_score = model.predict(input_features)[0]
        scaled_score = scale_virality_score(raw_score)
        return adjust_score_heuristically(scaled_score, caption, hashtags)

    except Exception as e:
        print(f"❌ Error in predict_virality: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return 0


def analyze_sentiment_distribution(text):
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