

import pandas as pd
import numpy as np
import os
import random
from textblob import TextBlob
import re 

print("--- Starting New Data Generation (Lower Scores for Low Engagement) ---")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

OUTPUT_CSV_PATH = os.path.join(DATA_DIR, 'new_mock_social_virality_dataset_extended.csv')
NUM_ENTRIES = 10000

PLATFORMS = ['YouTube', 'Facebook', 'Instagram', 'X']


POSITIVE_CAPTIONS = [
    "Absolutely loving this! So much joy and inspiration. ✨",
    "Feeling incredibly happy and grateful today. Best day ever! 😊",
    "What an amazing experience! Definitely a must-see. 🔥",
    "So proud of this accomplishment! Hard work pays off. 💪",
    "Good vibes only. Spread happiness! ❤️"
]
NEGATIVE_CAPTIONS = [
    "Absolutely disgusted by this terrible news. A complete disaster. 😠",
    "Can't believe this happened. Feeling really frustrated and disappointed. 😞",
    "This is unacceptable. Major issues need to be addressed immediately. 😡",
    "Worst experience ever. So frustrating and a waste of time. 👎",
    "Deeply concerned about the future. #crisis #worry"
]
NEUTRAL_CAPTIONS = [
    "Just sharing an update on the project. More details soon.",
    "A quiet afternoon walk in the park. #nature #relax",
    "Reviewing the latest industry report. Interesting findings.",
    "Testing out a new feature. Feedback welcome.",
    "Captured this moment. #dailyphoto"
]

POSITIVE_HASHTAGS = ["#positivevibes", "#inspiration", "#happiness", "#success", "#joy", "#blessed"]
NEGATIVE_HASHTAGS = ["#disaster", "#fail", "#frustration", "#unacceptable", "#problem", "#struggle"]
NEUTRAL_HASHTAGS = ["#update", "#news", "#info", "#daily", "#project", "#facts"]


def calculate_virality_score(likes, views, sentiment_polarity, platform):
    """
    Simulates a virality score calculation, aiming for very low scores
    for low engagement, and higher for high, with randomness.
    """
    base_score = 0
    
    log_likes = np.log1p(likes)
    log_views = np.log1p(views)
    
    likes_coeff = 15 + np.random.normal(0, 0.5) 
    views_coeff = 8 + np.random.normal(0, 0.25) 
    sentiment_coeff = 15 + np.random.normal(0, 1.5)

    base_score = (log_likes * likes_coeff) + (log_views * views_coeff)
    base_score += sentiment_polarity * sentiment_coeff
   
    
    if platform == 'YouTube':
        base_score *= 1.2
    elif platform == 'Instagram':
        base_score *= 1.0
    elif platform == 'X':
        base_score *= 0.9
    elif platform == 'Facebook':
        base_score *= 1.1

    random_noise = np.random.normal(0, 20) 
    base_score += random_noise

    min_target_score = -100.0 
    max_target_score = 800.0

    final_score = np.clip(base_score, min_target_score, max_target_score)
    
    return float(f"{final_score:.2f}")



data = []
for i in range(NUM_ENTRIES):
    platform = random.choice(PLATFORMS)

    
    if random.random() < 0.2: 
        likes = int(np.exp(np.random.normal(loc=11, scale=1.5)))
        views = int(np.exp(np.random.normal(loc=13, scale=1.5)))
    elif random.random() < 0.6: 
        likes = int(np.exp(np.random.normal(loc=8, scale=1.0)))
        views = int(np.exp(np.random.normal(loc=10, scale=1.0)))
    else:
        likes = int(np.exp(np.random.normal(loc=5, scale=0.8)))
        views = int(np.exp(np.random.normal(loc=7, scale=0.8)))

    likes = max(100, likes)
    views = max(1000, views)

    if likes > views:
        views = likes + random.randint(100, 1000)

    
    sentiment_type = random.choice(['positive', 'negative', 'neutral'])
    if sentiment_type == 'positive':
        caption_template = random.choice(POSITIVE_CAPTIONS)
        hashtags_list = random.sample(POSITIVE_HASHTAGS, k=random.randint(2,4))
    elif sentiment_type == 'negative':
        caption_template = random.choice(NEGATIVE_CAPTIONS)
        hashtags_list = random.sample(NEGATIVE_HASHTAGS, k=random.randint(2,4))
    else: 
        caption_template = random.choice(NEUTRAL_CAPTIONS)
        hashtags_list = random.sample(NEUTRAL_HASHTAGS, k=random.randint(2,4))
    
    caption = caption_template
    hashtags_str = " ".join(hashtags_list)

    
    caption_length = len(caption.split())
    sentiment_polarity = TextBlob(caption).sentiment.polarity
    total_chars = len(caption)
    hashtag_count = len(re.findall(r'#\w+', hashtags_str))

    engagement_rate = round(likes / (likes + views), 4) if (likes + views) > 0 else 0

    virality_score = calculate_virality_score(likes, views, sentiment_polarity, platform)

    data.append({
        'platform': platform,
        'caption': caption,
        'likes': likes,
        'views': views,
        'hashtags': hashtags_str,
        'caption_length': caption_length,
        'sentiment': sentiment_polarity,
        'virality_score': virality_score
    })

df_new = pd.DataFrame(data)


print(f"\nGenerated {NUM_ENTRIES} entries.")
print("\n--- New Dataset Head ---")
print(df_new.head())
print("\n--- New Dataset Describe ---")
print(df_new.describe())
print("\n--- New Dataset Platform Counts ---")
print(df_new['platform'].value_counts())
print("\n--- New Dataset Virality Score Distribution (first 10 unique values) ---")
print(df_new['virality_score'].value_counts().sort_index().head(10))
print(df_new['virality_score'].value_counts().sort_index(ascending=False).head(10))


df_new.to_csv(OUTPUT_CSV_PATH, index=False)
print(f"\nNew CSV file created successfully at: {OUTPUT_CSV_PATH}")
print("\n--- Data Generation Complete ---")