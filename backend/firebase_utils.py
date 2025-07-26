from firebase_init import db
from google.cloud import firestore
from datetime import datetime

# ✅ Save or update user info in Firestore
def save_user(uid, email, name):
    user_ref = db.collection("users").document(uid)
    user_ref.set({
        "email": email,
        "username": name,
        "created_at": datetime.utcnow()
    }, merge=True)

# ✅ Save an analysis entry for a user
def save_analysis(uid, caption, virality_score, sentiment, hashtags):
    analysis_ref = db.collection("users").document(uid).collection("analysis")
    analysis_ref.add({
        "caption": caption,
        "virality_score": virality_score,
        "sentiment": sentiment,
        "hashtags": hashtags,
        "created_at": datetime.utcnow()
    })

# ✅ Get analysis history for a user (most recent first)
def get_user_history(uid):
    history_ref = db.collection("users").document(uid).collection("analysis")\
                    .order_by("created_at", direction=firestore.Query.DESCENDING)
    docs = history_ref.stream()
    return [doc.to_dict() for doc in docs]

# ✅ Optionally: Fetch a single analysis by its document ID
def get_analysis_by_id(uid, analysis_id):
    doc_ref = db.collection("users").document(uid).collection("analysis").document(analysis_id)
    doc = doc_ref.get()
    if doc.exists:
        return doc.to_dict()
    return None
