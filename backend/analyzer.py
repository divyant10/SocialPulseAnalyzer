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
import requests

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

EXPECTED_PLATFORMS = ['YouTube', 'Facebook', 'Instagram', 'X']

TRENDING_HASHTAGS = [
    "#trending", "#viral", "#reels", "#explore", "#instagood",
    "#socialmedia", "#influencer", "#contentcreator", "#marketing", "#growth",
    "#fyp", "#foryou", "#discover", "#instadaily", "#photooftheday"
]

MODELS_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'virality_model.pkl')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')
OHE_PATH = os.path.join(MODELS_DIR, 'one_hot_encoder.pkl')

model = None
scaler = None
one_hot_encoder = None


def download_and_extract_models():
    drive_url = "https://drive.google.com/uc?export=download&id=1k2J7h4xGBdbN3DF9c01ylt8l_WJwvOOw"
    zip_path = os.path.join(BASE_DIR, "models.zip")

    print("🔽 Downloading model zip from Google Drive...")
    response = requests.get(drive_url)
    with open(zip_path, 'wb') as f:
        f.write(response.content)

    print("📦 Extracting models.zip...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(MODELS_DIR)

    os.remove(zip_path)
    print("✅ Model files extracted.")


# Try loading model and dependencies
try:
    if not os.path.exists(MODEL_PATH):
        download_and_extract_models()

    model = joblib.load(MODEL_PATH)
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
    else:
        print(f"⚠️ Scaler not found at {SCALER_PATH}")

    if os.path.exists(OHE_PATH):
        one_hot_encoder = joblib.load(OHE_PATH)
    else:
        print(f"⚠️ OneHotEncoder not found at {OHE_PATH}")

except Exception as e:
    print(f"❌ Error loading model or dependencies: {e}")


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
        print("❌ Missing model/scaler/encoder.")
        return 0

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
        if hasattr(platform_encoded, 'toarray'):
            platform_encoded = platform_encoded.toarray()

        input_features = np.hstack((scaled, platform_encoded))

        raw_score = model.predict(input_features)[0]
        scaled_score = scale_virality_score(raw_score)
        return adjust_score_heuristically(scaled_score, caption, hashtags)

    except Exception as e:
        print(f"❌ Error in predict_virality: {e}")
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
