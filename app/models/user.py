from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import os

class User(UserMixin):
    def __init__(self, id, username, email, password_hash, is_admin=False):
        self.id = id
        self.username = username
        self.email = email
        self.password_hash = password_hash
        self.is_admin = is_admin
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    @staticmethod
    def get(user_id):
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'users.db')
        print("[user.py] DB path:", db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
        user_data = cursor.fetchone()
        
        conn.close()
        
        if user_data:
            return User(
                id=user_data[0],
                username=user_data[1],
                email=user_data[2],
                password_hash=user_data[3],
                is_admin=bool(user_data[4])
            )
        return None
    
    @staticmethod
    def get_by_username(username):
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'users.db')
        print("[user.py] DB path:", db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        user_data = cursor.fetchone()
        
        conn.close()
        
        if user_data:
            return User(
                id=user_data[0],
                username=user_data[1],
                email=user_data[2],
                password_hash=user_data[3],
                is_admin=bool(user_data[4])
            )
        return None
    
    @staticmethod
    def create(username, email, password, is_admin=False):
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'users.db')
        print("[user.py] DB path:", db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create users table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0
        )
        ''')
        
        # Check if username or email already exists
        cursor.execute('SELECT * FROM users WHERE username = ? OR email = ?', (username, email))
        if cursor.fetchone():
            conn.close()
            return None
        
        # Create new user
        password_hash = generate_password_hash(password)
        cursor.execute(
            'INSERT INTO users (username, email, password_hash, is_admin) VALUES (?, ?, ?, ?)',
            (username, email, password_hash, 1 if is_admin else 0)
        )
        
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return User(user_id, username, email, password_hash, is_admin) 