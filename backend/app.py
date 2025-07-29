from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import os
import time
import re
# Removed gdown and zipfile imports as model download is now handled solely by analyzer.py
# import gdown
# import zipfile

# Import your analysis functions using RELATIVE IMPORTS
# This is crucial because app.py is part of the 'backend' package
from .analyzer import predict_virality, analyze_sentiment_distribution
from .caption_suggester import get_caption_suggestions
from .hashtag_effectiveness import analyze_hashtag_effectiveness
from .tts_summary import generate_tts_summary


# Flask App Configuration
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # Points to SocialPulseAnalyzer/

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'frontend', 'templates'),
    static_folder=os.path.join(BASE_DIR, 'frontend', 'static')
)

# --- SECURED: Use Environment Variable for Secret Key ---
# In production, set a strong SECRET_KEY environment variable on Render.
# For local development, it will fall back to a default.
app.secret_key = os.environ.get('SECRET_KEY', 'a_very_secret_and_random_key_for_dev')

# Ensure debug mode is off in production
# It's implicitly off when run with Waitress, but explicit control is good
app.debug = os.environ.get('FLASK_DEBUG') == '1' # Set FLASK_DEBUG=1 for local debug

UPLOAD_FOLDER = os.path.join(app.static_folder, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Simple in-memory user storage (for demonstration, not for production)
users = {}

def _parse_input_string(input_str_val):
    val = 0
    if input_str_val:
        s = str(input_str_val).strip().upper()
        s = s.replace(',', '')
        s = re.sub(r'\(.*\)', '', s)
        s = s.strip()
        try:
            if s.endswith('K'): val = int(float(s[:-1]) * 1_000)
            elif s.endswith('M'): val = int(float(s[:-1]) * 1_000_000)
            elif s.endswith('B'): val = int(float(s[:-1]) * 1_000_000_000)
            else: val = int(float(s))
        except ValueError:
            val = 0
            print(f"DEBUG: Error parsing value '{input_str_val}'. Setting to 0.")
    return val


# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/welcome')
def welcome():
    return render_template('welcome.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        if email in users and users[email] == password:
            session['user'] = email
            session.permanent = False
            session['history'] = []
            return redirect(url_for('main'))
        else:
            return render_template('login.html', error="Invalid credentials.")
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        if email in users:
            return render_template('signup.html', error="User already exists.")
        users[email] = password
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/main')
def main():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('main.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'user' not in session:
        return redirect(url_for('login'))

    caption = request.form.get('caption', '')
    likes_input_str = request.form.get('likes', '')
    views_input_str = request.form.get('views', '')
    hashtags = request.form.get('hashtags', '')
    platform = request.form.get('platform', '')
    subscribers_input_str = request.form.get('subscribers', '')
    channel_views_input_str = request.form.get('channel_views', '')

    likes = _parse_input_string(likes_input_str)
    views = _parse_input_string(views_input_str)
    subscribers = _parse_input_string(subscribers_input_str)
    channel_views = _parse_input_string(channel_views_input_str)

    print(f"DEBUG: app.py parsed values: likes={likes}, views={views}, subs={subscribers}, ch_views={channel_views}")

    virality_score = int(predict_virality(caption, likes, views, hashtags, platform, subscribers, channel_views))
    sentiment_data = analyze_sentiment_distribution(caption)
    caption_results = get_caption_suggestions(caption)
    hashtag_tips = analyze_hashtag_effectiveness(hashtags)
    top_hashtags = [tag.strip() for tag in hashtags.split() if tag.strip().startswith('#')]

    history_item_minimal = {
        "timestamp": int(time.time()),
        "datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "caption": caption,
        "likes": likes,
        "views": views,
        "virality_score": virality_score,
        "platform": platform,
        "subscribers": subscribers,
        "channel_views": channel_views,
        "hashtags_raw": hashtags
    }

    full_analysis_data = {
        **history_item_minimal,
        "sentiment": sentiment_data.get('overall_sentiment', 'Neutral'),
        "top_hashtags": top_hashtags,
        "hashtag_tips": hashtag_tips,
        "sentiment_graph_data": sentiment_data.get("graph_data", {}),
        "caption_suggestions_data": caption_results
    }

    if 'history' not in session:
        session['history'] = []
    session['history'].insert(0, history_item_minimal)
    session['history'] = session['history'][:20]

    tts_summary_output = generate_tts_summary(full_analysis_data)
    session['summary_audio'] = tts_summary_output['audio_path']
    session['summary_points'] = tts_summary_output['summary_points']
    session['current_analysis'] = full_analysis_data

    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))

    current_analysis = session.get('current_analysis', None)
    history = session.get('history', [])

    if not current_analysis and history:
        temp_analysis_data = history[0]
        sentiment_data = analyze_sentiment_distribution(temp_analysis_data['caption'])
        hashtag_tips = analyze_hashtag_effectiveness(temp_analysis_data.get('hashtags_raw', ''))
        top_hashtags = [tag.strip() for tag in temp_analysis_data.get('hashtags_raw', '').split() if tag.strip().startswith('#')]
        caption_results = get_caption_suggestions(temp_analysis_data['caption'])

        full_temp_analysis = {
            **temp_analysis_data,
            "sentiment": sentiment_data.get('overall_sentiment', 'Neutral'),
            "top_hashtags": top_hashtags,
            "hashtag_tips": hashtag_tips,
            "sentiment_graph_data": sentiment_data.get("graph_data", {}),
            "caption_suggestions_data": caption_results
        }
        session['current_analysis'] = full_temp_analysis
        current_analysis = full_temp_analysis

    if not current_analysis:
        return redirect(url_for('main'))

    return render_template('dashboard.html',
                           analysis=current_analysis,
                           history=history)

@app.route('/load_history_item/<int:timestamp>')
def load_history_item(timestamp):
    if 'user' not in session or 'history' not in session:
        return redirect(url_for('login'))

    found_item_minimal = next((item for item in session['history'] if item['timestamp'] == timestamp), None)

    if found_item_minimal:
        sentiment_data = analyze_sentiment_distribution(found_item_minimal['caption'])
        caption_results = get_caption_suggestions(found_item_minimal['caption'])
        hashtag_tips = analyze_hashtag_effectiveness(found_item_minimal.get('hashtags_raw', ''))
        top_hashtags = [tag.strip() for tag in found_item_minimal.get('hashtags_raw', '').split() if tag.strip().startswith('#')]

        full_analysis_for_display = {
            **found_item_minimal,
            "sentiment": sentiment_data.get('overall_sentiment', 'Neutral'),
            "top_hashtags": top_hashtags,
            "hashtag_tips": hashtag_tips,
            "sentiment_graph_data": sentiment_data.get("graph_data", {}),
            "caption_suggestions_data": caption_results
        }
        session['current_analysis'] = full_analysis_for_display
        return redirect(url_for('dashboard'))

    return redirect(url_for('dashboard'))

@app.route('/delete_history_item', methods=['POST'])
def delete_history_item():
    if 'user' not in session or 'history' not in session:
        return jsonify({"success": False, "message": "Not logged in or no history."})

    data = request.get_json()
    timestamp_to_delete = data.get('timestamp')

    if timestamp_to_delete is None:
        return jsonify({"success": False, "message": "No timestamp provided."})

    original_length = len(session['history'])
    session['history'] = [item for item in session['history'] if item['timestamp'] != timestamp_to_delete]

    if len(session['history']) < original_length:
        if session.get('current_analysis', {}).get('timestamp') == timestamp_to_delete:
            if session['history']:
                new_top_item_minimal = session['history'][0]
                sentiment_data = analyze_sentiment_distribution(new_top_item_minimal['caption'])
                caption_results = get_caption_suggestions(new_top_item_minimal['caption'])
                hashtag_tips = analyze_hashtag_effectiveness(new_top_item_minimal.get('hashtags_raw', ''))
                top_hashtags = [tag.strip() for tag in new_top_item_minimal.get('hashtags_raw', '').split() if tag.strip().startswith('#')]

                full_new_top_analysis = {
                    **new_top_item_minimal,
                    "sentiment": sentiment_data.get('overall_sentiment', 'Neutral'),
                    "top_hashtags": top_hashtags,
                    "hashtag_tips": hashtag_tips,
                    "sentiment_graph_data": sentiment_data.get("graph_data", {}),
                    "caption_suggestions_data": caption_results
                }
                session['current_analysis'] = full_new_top_analysis
            else:
                session['current_analysis'] = None
        return jsonify({"success": True, "message": "History item deleted."})
    else:
        return jsonify({"success": False, "message": "History item not found."})

@app.route('/summary')
def summary():
    if 'user' not in session:
        return redirect(url_for('login'))

    current_analysis = session.get('current_analysis', None)
    summary_audio = session.get('summary_audio', None)
    summary_points = session.get('summary_points', [])

    if not current_analysis or not summary_audio or not summary_points:
        return redirect(url_for('main'))

    return render_template('summary.html',
                           audio=summary_audio,
                           analysis=current_analysis,
                           summary_points=summary_points)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# --- REMOVED: This block is for local development only. Waitress handles starting the app on Render. ---
# if __name__ == '__main__':
#     app.run(debug=True, port=8001)
