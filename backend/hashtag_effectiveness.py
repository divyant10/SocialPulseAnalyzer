import random
import re

TRENDING_HASHTAGS = [
    "#trending", "#viral", "#reels", "#explore", "#instagood",
    "#socialmedia", "#influencer", "#contentcreator", "#marketing", "#growth",
    "#fyp", "#foryou", "#discover", "#instadaily", "#photooftheday"
]

def analyze_hashtag_effectiveness(hashtags_str):
    """
    Analyzes hashtag effectiveness and returns a LIST of improvement tips,
    including formatting suggestions.
    """
    improvement_tips = []

    if not hashtags_str:
        improvement_tips.append("You haven't added any hashtags. Try using 3–5 relevant hashtags to boost reach.")
        improvement_tips.append("Research trending and niche-specific tags for your content.")
        return improvement_tips

    raw_hashtags = [tag.strip() for tag in hashtags_str.split() if tag.startswith('#')]
    processed_hashtags_text = [tag[1:].lower() for tag in raw_hashtags]

    
    if len(raw_hashtags) < 3:
        improvement_tips.append("Consider adding 3–5 relevant hashtags for better visibility.")
    elif len(raw_hashtags) > 15:
        improvement_tips.append("Using too many hashtags (over 15) can sometimes look spammy. Consider focusing on the most relevant ones.")
    
    
    if len(set(processed_hashtags_text)) < len(processed_hashtags_text):
        improvement_tips.append("Avoid repeating hashtags. Each unique hashtag gives a new chance to be discovered.")

    
    long_tags_found = [tag for tag in raw_hashtags if len(tag) > 25]
    if long_tags_found:
        improvement_tips.append(f"Consider shortening very long hashtags for readability (e.g., '{', '.join(long_tags_found)}').")

   
    casing_issue_detected_and_suggested = False
    for original_tag_with_hash in raw_hashtags:
        tag_text = original_tag_with_hash[1:] 

        if len(tag_text) <= 2 or not tag_text.isalpha():
            continue

        is_all_lower = tag_text.islower()
        is_all_upper = tag_text.isupper()
        
        is_camel_case = bool(re.match(r'^[a-zA-Z][a-zA-Z0-9]*(?:[A-Z][a-zA-Z0-9]*)*$', tag_text))

        if is_all_upper and len(tag_text) > 5:
            improvement_tips.append(f"For readability, consider **#CamelCase** (e.g., #{tag_text.title().replace(' ', '')}) or **#lowercase** (e.g., #{tag_text.lower()}) instead of all caps like **{original_tag_with_hash}**.")
            casing_issue_detected_and_suggested = True
            break 
        
        elif not is_all_lower and not is_all_upper and not is_camel_case:
            improvement_tips.append(f"Consider consistent casing for **{original_tag_with_hash}**: try **#CamelCase** (e.g., #{tag_text.title().replace(' ', '')}) or **#lowercase** (e.g., #{tag_text.lower()}).")
            casing_issue_detected_and_suggested = True
            break 
    if not casing_issue_detected_and_suggested and len(raw_hashtags) > 0:
        all_relevant_tags_are_lower = True
        for tag in raw_hashtags:
            if len(tag[1:]) > 2 and tag[1:].isalpha() and not tag[1:].islower():
                all_relevant_tags_are_lower = False
                break
        
        if all_relevant_tags_are_lower: 
             improvement_tips.append("For multi-word hashtags (e.g., #myawesometag), consider **#CamelCase** (e.g., #MyAwesomeTag) for better readability and discoverability.")

    if len(raw_hashtags) < 15 and not any("mix of niche and broader/trending tags" in tip for tip in improvement_tips):
        try:
            suggested_tags = [tag for tag in random.sample(TRENDING_HASHTAGS, k=min(3, len(TRENDING_HASHTAGS))) if tag.lower() not in processed_hashtags_text]
            if suggested_tags:
                improvement_tips.append("Consider using a mix of niche and broader/trending tags. Examples: " + ", ".join(suggested_tags))
        except ValueError:
            pass 
    
    if not improvement_tips:
        improvement_tips.append("Hashtag usage looks good!")

    return improvement_tips