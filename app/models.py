from flask_login import UserMixin
import sqlite3
import os

class User(UserMixin):
    def __init__(self, id, username=None, email=None):
        self.id = id
        self.username = username
        self.email = email

    @staticmethod
    def get(user_id):
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'users.db')
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE id = ?', (user_id,))
        user = c.fetchone()
        conn.close()
        
        if user:
            return User(user[0], user[1], user[2])
        return None

    @staticmethod
    def get_by_username(username):
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'users.db')
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = c.fetchone()
        conn.close()
        
        if user:
            return User(user[0], user[1], user[2])
        return None

    @staticmethod
    def create(username, email, password):
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'users.db')
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        # Create users table if it doesn't exist
        c.execute('''CREATE TABLE IF NOT EXISTS users
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     username TEXT UNIQUE NOT NULL,
                     email TEXT UNIQUE NOT NULL,
                     password TEXT NOT NULL)''')
        
        try:
            c.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                     (username, email, password))
            conn.commit()
            user_id = c.lastrowid
            conn.close()
            return User(user_id, username, email)
        except sqlite3.IntegrityError:
            conn.close()
            return None 