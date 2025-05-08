import sqlite3
import os

db_path = os.path.join(os.path.dirname(__file__), 'app', 'users.db')
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    is_admin INTEGER DEFAULT 0
)
''')
conn.commit()
conn.close()
print("Database initialized.") 