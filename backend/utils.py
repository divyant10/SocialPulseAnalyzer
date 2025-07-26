import re
import joblib
import os
import json
import matplotlib.pyplot as plt
import uuid

def clean_caption(text):
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def extract_hashtags(text):
    return re.findall(r"#(\w+)", text)

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    return joblib.load(model_path)

def normalize_score(score, min_val=0, max_val=1):
    if max_val - min_val == 0:
        return 0
    return round(100 * (score - min_val) / (max_val - min_val), 2)

def save_analysis_to_json(data, output_path="data/analysis_log.json"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    existing = []
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []
    existing.append(data)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=4)
    return True

def map_sentiment_to_score(sentiment):
    sentiment = sentiment.lower()
    if sentiment == "positive":
        return 80
    elif sentiment == "neutral":
        return 50
    elif sentiment == "negative":
        return 20
    return 40  

def analyze_hashtags(hashtags_string):
    hashtags = hashtags_string.strip().split()
    effectiveness = {}
    for tag in hashtags:
        clean_tag = tag.lstrip('#')
        score = min(100, len(clean_tag) * 8 + hash(tag) % 20)
        effectiveness[clean_tag] = score
    return effectiveness

def generate_sentiment_chart(breakdown, output_folder="frontend/static/charts"):
    os.makedirs(output_folder, exist_ok=True)
    sentiments = list(breakdown.keys())
    scores = []

    for sentiment in sentiments:
        score = breakdown[sentiment]
        if isinstance(score, dict):
            score = score.get('score', 0)
        scores.append(score)

    colors = [
        "green" if s > 60 else "yellow" if s > 20 else "red"
        for s in scores
    ]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(sentiments, scores, color=colors)
    ax.set_title("Sentiment Breakdown")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 100)

    filename = f"sentiment_{uuid.uuid4().hex[:6]}.png"
    filepath = os.path.join(output_folder, filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=120)
    plt.close()

    return f"/static/charts/{filename}"

def generate_summary_points(platform, caption, score, sentiment, hashtags):
    summary = [
        f"Platform analyzed: {platform}",
        f"Virality score is {score} indicating {'low' if score < 20 else 'moderate' if score < 60 else 'high'} reach potential.",
        f"Sentiment detected as {sentiment}.",
        f"Total hashtags used: {len(hashtags.split())}.",
        f"Caption length: {len(caption.split())} words.",
        "Consider enhancing visual elements of your post.",
        "Posting time can affect reach — experiment with timing.",
        "Hashtag relevance matters — include trending or niche ones.",
        "Engagement rate (likes/views) impacts virality.",
        "Try A/B testing different captions or formats."
    ]
    return summary[:10]
