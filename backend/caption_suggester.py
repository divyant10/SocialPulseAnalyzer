import random
from textblob import TextBlob


EMOJIS = {
    "positive": ["🔥", "💯", "🚀", "🎉", "🙌", "😍", "✨"],
    "call_to_action": ["👉", "📢", "🗣️", "📸", "🔔"],
    "fun": ["😎", "🤣", "🎊", "🌟", "🥳"],
    "love": ["❤️", "💖", "💕", "💘", "😘"]
}

def get_caption_suggestions(original_caption):
    """Generate 2–3 improved versions of the given caption."""

    suggestions = []

    
    cta = random.choice([
        "Tag a friend!", "What do you think?", "Comment below 👇", 
        "Share your thoughts!", "Double tap if you agree!"
    ])
    suggestions.append(f"{original_caption} {random.choice(EMOJIS['call_to_action'])} {cta}")

   
    short_version = ' '.join(original_caption.split()[:7])
    short_caption = f"{short_version}... {random.choice(EMOJIS['positive'])} #Trending"
    suggestions.append(short_caption)

    
    blob = TextBlob(original_caption)
    polarity = blob.sentiment.polarity
    if polarity > 0.2:
        emoji_pack = EMOJIS['positive'] + EMOJIS['fun']
    elif polarity < -0.2:
        emoji_pack = ["😢", "😔", "💔"]
    else:
        emoji_pack = EMOJIS['love']

    enhanced = f"{original_caption} {' '.join(random.sample(emoji_pack, 2))}"
    suggestions.append(enhanced)

    return suggestions
