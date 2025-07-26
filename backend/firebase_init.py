import firebase_admin
from firebase_admin import credentials, firestore

# Path to your downloaded Firebase private key (adjusted to run from within backend/)
cred = credentials.Certificate("firebase_admin_config.json")

# Initialize Firebase app (only once)
firebase_admin.initialize_app(cred)

# Initialize Firestore client
db = firestore.client()
