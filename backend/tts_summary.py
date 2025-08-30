import os
import datetime
from gtts import gTTS 
import uuid 


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
STATIC_AUDIO_DIR = os.path.join(BASE_DIR, 'frontend', 'static', 'audio')
os.makedirs(STATIC_AUDIO_DIR, exist_ok=True) 

def generate_tts_summary(analysis_data):
    """
    Generates a 10-15 point summary and converts it to speech.
    Returns a dictionary with 'audio_path' and 'summary_points'.
    """
    summary_points = []
    
    summary_points.append(f"Analyzing your post submitted on {analysis_data.get('datetime', 'a recent date')}.")
    summary_points.append(f"You selected the {analysis_data.get('platform', 'N/A')} platform.")
    summary_points.append(f"Your caption was: '{analysis_data.get('caption', 'N/A')}.'")

    
    likes = analysis_data.get('likes', 0)
    views = analysis_data.get('views', 0)
    summary_points.append(f"The post received {likes:,} likes and {views:,} views.") # Add comma formatting

    if analysis_data.get('platform') == 'YouTube':
        subscribers = analysis_data.get('subscribers', 0)
        channel_views = analysis_data.get('channel_views', 0)
        summary_points.append(f"Your YouTube channel has {subscribers:,} subscribers and {channel_views:,} total views.")

    virality_score = analysis_data.get('virality_score', 0)
    summary_points.append(f"The predicted virality score is {virality_score} out of 100.")
    if virality_score >= 80:
        summary_points.append("This indicates very high potential for viral reach! Well done!")
    elif virality_score >= 50:
        summary_points.append("This suggests a good potential for strong engagement and reach.")
    else:
        summary_points.append("This score indicates moderate to lower virality potential. There's room for optimization.")

    overall_sentiment = analysis_data.get('sentiment', 'Neutral')
    summary_points.append(f"The overall sentiment of your caption is detected as {overall_sentiment}.")
    if overall_sentiment == 'Positive':
        summary_points.append("Positive sentiment often encourages sharing and engagement.")
    elif overall_sentiment == 'Negative':
        summary_points.append("Negative sentiment, while sometimes viral, can also limit broad appeal.")
    else:
        summary_points.append("Neutral sentiment may require stronger calls to action or unique content to stand out.")

    hashtag_tips = analysis_data.get('hashtag_tips', [])
    if hashtag_tips:
        summary_points.append("Regarding your hashtags, here are some key tips:")
        for i, tip in enumerate(hashtag_tips[:3]): 
            clean_tip = tip.replace('**', '').replace('#', 'hashtag ') 
            summary_points.append(f"- {clean_tip}")
    else:
        summary_points.append("Your hashtag usage looks good!")

    caption_suggestions = analysis_data.get('caption_suggestions_data', [])
    if caption_suggestions:
        summary_points.append(f"For caption improvements, consider: '{caption_suggestions[0]}'")
        summary_points.append("Optimizing captions can significantly boost engagement.")


    full_summary_text = "Here is a summary of your social media post analysis. " + " ".join(summary_points)

    try:
        tts = gTTS(text=full_summary_text, lang='en', slow=False)
        audio_filename = f"summary_{analysis_data['timestamp']}.mp3"
        audio_filepath = os.path.join(STATIC_AUDIO_DIR, audio_filename)
        tts.save(audio_filepath)
        audio_url = f"audio/{audio_filename}" 
        print(f"DEBUG: TTS audio saved at: {audio_filepath}")
    except Exception as e:
        print(f"ERROR: Failed to generate TTS audio: {e}")
        audio_url = None
        summary_points.append("Warning: Failed to generate audio summary.")


    return {"audio_path": audio_url, "summary_points": summary_points}