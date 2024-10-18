import bcrypt
import streamlit as st
from database import Database
import logging

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def login(db, username, password):
    with db.get_db_cursor() as cur:
        cur.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cur.fetchone()
    if user and check_password(password, user['password_hash']):
        st.session_state.user_id = user['id']
        st.success("Login successful!")
        st.session_state.page = 'word_list'
    else:
        st.error("Invalid username or password.")

def register(db, username, password):
    hashed_password = hash_password(password)
    try:
        with db.get_db_cursor() as cur:
            cur.execute("""
                INSERT INTO users (username, password_hash)
                VALUES (%s, %s)
            """, (username, hashed_password))
        return True
    except Exception as e:
        logging.error(f"Error registering user {username}: {e}")
        return False
