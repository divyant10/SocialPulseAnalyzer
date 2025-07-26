import joblib
import re
import os
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt 
from uuid import uuid4
import pandas as pd
import random 


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))



EXPECTED_PLATFORMS = ['YouTube', 'Facebook', 'Instagram', 'X']



TRENDING_HASHTAGS = [
    "#trending", "#viral", "#reels", "#explore", "#instagood",
    "#socialmedia", "#influencer", "#contentcreator", "#marketing", "#growth",
    "#fyp", "#foryou", "#discover", "#instadaily", "#photooftheday"
]



MODEL_PATH = os.path.join(BASE_DIR, 'models', 'virality_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
OHE_PATH = os.path.join(BASE_DIR, 'models', 'one_hot_encoder.pkl')

model = None
scaler = None
one_hot_encoder = None

try:
    model = joblib.load(MODEL_PATH)
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
    else:
        print(f"Warning: Scaler not found at {SCALER_PATH}. Ensure your model doesn't require scaled input, or train and save a scaler.")
    
    if os.path.exists(OHE_PATH):
        one_hot_encoder = joblib.load(OHE_PATH)
    else:
        print(f"Warning: OneHotEncoder not found at {OHE_PATH}. Ensure it's saved if platform is a feature.")

except Exception as e:
    print(f"Error loading model, scaler, or OneHotEncoder: {e}. Please ensure all .pkl files exist.")
    model = None
    scaler = None
    one_hot_encoder = None



def parse_count(value):
    """Convert likes/views like 1.2K or 3M into integer."""
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
    """
    Convert raw model score (which ranges from ~-1.24 to ~713.93 based on training data)
    to a 0-100 scale.
    """
    min_observed_raw_prediction = -1.24
    max_observed_raw_prediction = 713.93

    if abs(max_observed_raw_prediction - min_observed_raw_prediction) < 1e-6:
        return 0

    scaled_val = (raw_score - min_observed_raw_prediction) / (max_observed_raw_prediction - min_observed_raw_prediction)
    scaled_val = np.clip(scaled_val, 0, 1)
    
    return round(scaled_val * 100, 2)


def adjust_score_heuristically(score, caption, hashtags):
    """Apply heuristic boosts based on useful caption or hashtag traits."""
    current_score = score
    sentiment_polarity = TextBlob(caption).sentiment.polarity

    if current_score < 10 and sentiment_polarity > 0.1:
        current_score = max(current_score, 10)

    if 20 < len(caption.split()) < 100:
        current_score += 2
    elif len(caption.split()) > 100:
        current_score -= 5

    if hashtags and hashtags.count('#') > 3:
        current_score += 1

    if any(word in caption.lower() for word in ['amazing', 'viral', 'trending', 'challenge', 'must-see']):
        current_score += 2

    return int(np.clip(current_score, 0, 100))


def predict_virality(caption, likes, views, hashtags, platform, subscribers, channel_views):
    print(f"DEBUG: predict_virality RECEIVED: likes={likes} (type: {type(likes)}), views={views} (type: {type(views)}), subscribers={subscribers} (type: {type(subscribers)}), channel_views={channel_views} (type: {type(channel_views)})")

    if model is None:
        print("[predict_virality Error]: ML model not loaded. Returning 0.")
        return 0
    if scaler is None:
        print("[predict_virality Error]: Scaler not loaded. Cannot process numerical features. Returning 0.")
        return 0
    if one_hot_encoder is None:
        print("[predict_virality Error]: OneHotEncoder not loaded. Cannot process platform feature. Returning 0.")
        return 0

    try:
        views_for_calc = max(1, views)

        caption_length = len(caption.split())
        
        if (likes + views_for_calc) > 0:
            engagement_rate = round(likes / (likes + views_for_calc), 4)
        else:
            engagement_rate = 0

        sentiment_score = round(TextBlob(caption).sentiment.polarity, 3)
        total_chars = len(caption)
        hashtag_count = len([tag for tag in hashtags.split() if tag.startswith('#')])

        numerical_features_array = np.array([[
            caption_length,
            engagement_rate,
            sentiment_score,
            total_chars,
            likes,
            views,
            hashtag_count,
            subscribers,
            channel_views
        ]])

        numerical_features_scaled = scaler.transform(numerical_features_array)

        processed_platform = platform.strip().replace(' (Twitter)', '').strip()
        
        platform_one_hot_array = one_hot_encoder.transform(np.array([[processed_platform]]))
        if hasattr(platform_one_hot_array, 'toarray'):
            platform_one_hot_array = platform_one_hot_array.toarray()
        
        input_features = np.hstack((numerical_features_scaled, platform_one_hot_array))

        print(f"DEBUG: Numerical Features (pre-scaled): {numerical_features_array}")
        print(f"DEBUG: Platform (processed): '{processed_platform}' -> One-Hot: {platform_one_hot_array}")
        print(f"DEBUG: Final Input Features shape: {input_features.shape}")
        print(f"DEBUG: Final Input Features (scaled + encoded): {input_features}")


        raw_prediction = model.predict(input_features)[0]
        print(f"DEBUG: Raw prediction from model: {raw_prediction}")

        scaled_score = scale_virality_score(raw_prediction)
        final_score = adjust_score_heuristically(scaled_score, caption, hashtags)

        print(f"DEBUG: Scaled score before heuristics: {scaled_score}")
        print(f"DEBUG: Final Virality Score: {final_score}")
        return final_score

    except Exception as e:
        print(f"[Error in predict_virality]: {e}")
        import traceback
        traceback.print_exc()
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

    words = text.split()
    positive, negative, neutral = 0, 0, 0
    for word in words:
        word_polarity = TextBlob(word).sentiment.polarity
        if word_polarity > 0.2:
            positive += 1
        elif word_polarity < -0.2:
            negative += 1
        else:
            neutral += 1

    charts_dir = os.path.join(BASE_DIR, 'frontend', 'static', 'charts')
    os.makedirs(charts_dir, exist_ok=True)

    graph_id = uuid4().hex[:6]
    graph_filename = f"sentiment_{graph_id}.png"
    graph_path_full = os.path.join(charts_dir, graph_filename)

    labels = ['Positive', 'Negative', 'Neutral'] 
    values = [positive, negative, neutral]       
    
    plt.figure(figsize=(5, 3))
    plt.bar(labels, values, color=['#4CAF50', '#F44336', '#9E9E9E'])
    plt.title('Sentiment Analysis')
    plt.ylabel('Word Count')
    plt.tight_layout()
    plt.savefig(graph_path_full)
    plt.close()

    return {
        "overall_sentiment": overall,
        "graph_data": {"labels": labels, "scores": values}
    }



def get_caption_suggestions(original_caption):
    if "handsome" in original_caption.lower():
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
        tips.append("Consider adding relevant hashtags for better reach.")
        tips.append("Research trending hashtags in your niche.")
        return tips

    raw_hashtags = [tag.strip() for tag in hashtags_str.split() if tag.startswith('#')]
    processed_hashtags_text = [tag[1:].lower() for tag in raw_hashtags]

    if len(raw_hashtags) < 3:
        tips.append("Consider adding 3–5 relevant hashtags for better visibility.")
    elif len(raw_hashtags) > 15:
        tips.append("Using too many hashtags (over 15) can sometimes look spammy. Consider focusing on the most relevant ones.")

    if len(set(processed_hashtags_text)) < len(processed_hashtags_text):
        tips.append("Avoid repeating hashtags. Each unique hashtag gives a new chance to be discovered.")
    
    casing_suggestion_made = False
    for original_tag_with_hash in raw_hashtags:
        tag_text = original_tag_with_hash[1:]

        if len(tag_text) <= 2 or not tag_text.isalpha():
            continue

        is_all_lower = tag_text.islower()
        is_all_upper = tag_text.isupper()
        is_camel_case = bool(re.match(r'^[a-zA-Z][a-zA-Z0-9]*(?:[A-Z][a-zA-Z0-9]*)*$', tag_text))

        if is_all_upper and len(tag_text) > 5:
            tips.append(f"For readability, consider **#CamelCase** (e.g., #{tag_text.title().replace(' ', '')}) or **#lowercase** (e.g., #{tag_text.lower()}) instead of all caps like **{original_tag_with_hash}**.")
            casing_suggestion_made = True
            break
        
        elif not is_all_lower and not is_all_upper and not is_camel_case:
            tips.append(f"Consider consistent casing for **{original_tag_with_hash}**: try **#CamelCase** (e.g., #{tag_text.title().replace(' ', '')}) or **#lowercase** (e.g., #{tag_text.lower()}).")
            casing_suggestion_made = True
            break

    if not casing_suggestion_made and len(raw_hashtags) > 0:
        all_relevant_tags_are_lower = True
        for tag in raw_hashtags:
            if len(tag[1:]) > 2 and tag[1:].isalpha() and not tag[1:].islower():
                all_relevant_tags_are_lower = False
                break
        
        if all_relevant_tags_are_lower:
             tips.append("For multi-word hashtags (e.g., #myawesometag), consider **#CamelCase** (e.g., #MyAwesomeTag) for better readability and discoverability.")

    if len(raw_hashtags) < 15 and not any("mix of niche and broader/trending tags" in tip for tip in tips):
        try:
            suggested_tags = [tag for tag in random.sample(TRENDING_HASHTAGS, k=min(3, len(TRENDING_HASHTAGS))) if tag.lower() not in processed_hashtags_text]
            if suggested_tags:
                tips.append("Consider using a mix of niche and broader/trending tags. Examples: " + ", ".join(suggested_tags))
        except ValueError:
            pass
    
    if not tips:
        tips.append("Hashtag usage looks good!")

    return tips