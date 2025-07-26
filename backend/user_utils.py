# user_utils.py

from firebase_utils import save_user, get_user_history

# Store sessions in memory (temporary, better to use Flask session or JWT later)
user_sessions = {}  # Format: {session_id: user_id}

def login_user(username, email):
    """
    Logs in the user by adding to the session and Firebase if needed.
    Returns a session_id and user_id.
    """
    user_id = save_user(username, email)
    session_id = f"{username}_{user_id}"
    user_sessions[session_id] = user_id
    return session_id, user_id

def logout_user(session_id):
    """
    Logs out a user by removing their session.
    """
    if session_id in user_sessions:
        del user_sessions[session_id]
        return True
    return False

def is_logged_in(session_id):
    """
    Checks if a session_id is currently active.
    """
    return session_id in user_sessions

def get_user_id(session_id):
    """
    Returns the user_id associated with the session.
    """
    return user_sessions.get(session_id, None)

def get_user_analysis_history(session_id):
    """
    Gets past analysis history for the logged-in user.
    """
    if not is_logged_in(session_id):
        return None
    user_id = get_user_id(session_id)
    return get_user_history(user_id)
