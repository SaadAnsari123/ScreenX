import sqlite3
import threading
import json
from pathlib import Path
import time
from datetime import datetime

class SharedResumeDB:
    def __init__(self, db_path="shared_resume.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create resumes table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS resumes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT,
                    phone TEXT,
                    skills TEXT,
                    experience TEXT,
                    education TEXT,
                    raw_text TEXT,
                    pdf_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create updates table for tracking changes
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS updates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    resume_id INTEGER,
                    action TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (resume_id) REFERENCES resumes (id)
                )
            ''')
            
            conn.commit()
            conn.close()
    
    def add_or_update_resume(self, resume_data):
        """Add or update a resume in the database"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if resume already exists
            cursor.execute('SELECT id FROM resumes WHERE name = ?', (resume_data['name'],))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing resume
                resume_id = existing[0]
                cursor.execute('''
                    UPDATE resumes SET 
                    email = ?, phone = ?, skills = ?, experience = ?, education = ?,
                    raw_text = ?, pdf_path = ?, updated_at = ?
                    WHERE id = ?
                ''', (
                    resume_data.get('email', ''),
                    resume_data.get('phone', ''),
                    resume_data.get('skills', ''),
                    resume_data.get('experience', ''),
                    resume_data.get('education', ''),
                    resume_data.get('raw_text', ''),
                    resume_data.get('pdf_path', ''),
                    datetime.now(),
                    resume_id
                ))
                action = 'update'
            else:
                # Insert new resume
                cursor.execute('''
                    INSERT INTO resumes 
                    (name, email, phone, skills, experience, education, raw_text, pdf_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    resume_data['name'],
                    resume_data.get('email', ''),
                    resume_data.get('phone', ''),
                    resume_data.get('skills', ''),
                    resume_data.get('experience', ''),
                    resume_data.get('education', ''),
                    resume_data.get('raw_text', ''),
                    resume_data.get('pdf_path', '')
                ))
                resume_id = cursor.lastrowid
                action = 'add'
            
            # Log the update
            cursor.execute('''
                INSERT INTO updates (resume_id, action, timestamp)
                VALUES (?, ?, ?)
            ''', (resume_id, action, datetime.now()))
            
            conn.commit()
            conn.close()
            
            return resume_id
    
    def get_all_resumes(self):
        """Get all resumes from the database"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, name, email, phone, skills, experience, education, 
                       pdf_path, created_at, updated_at
                FROM resumes 
                ORDER BY updated_at DESC
            ''')
            
            resumes = []
            for row in cursor.fetchall():
                resume = {
                    'id': row[0],
                    'name': row[1],
                    'email': row[2],
                    'phone': row[3],
                    'skills': row[4],
                    'experience': row[5],
                    'education': row[6],
                    'pdf_path': row[7],
                    'created_at': row[8],
                    'updated_at': row[9]
                }
                resumes.append(resume)
            
            conn.close()
            return resumes
    
    def get_recent_updates(self, limit=10):
        """Get recent updates for real-time tracking"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT u.action, u.timestamp, r.name 
                FROM updates u
                JOIN resumes r ON u.resume_id = r.id
                ORDER BY u.timestamp DESC
                LIMIT ?
            ''', (limit,))
            
            updates = []
            for row in cursor.fetchall():
                update = {
                    'action': row[0],
                    'timestamp': row[1],
                    'candidate_name': row[2]
                }
                updates.append(update)
            
            conn.close()
            return updates
    
    def delete_resume(self, resume_id):
        """Delete a resume from the database"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM resumes WHERE id = ?', (resume_id,))
            cursor.execute('DELETE FROM updates WHERE resume_id = ?', (resume_id,))
            
            conn.commit()
            conn.close()
    
    def search_resumes(self, query):
        """Search resumes by name, skills, or experience"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            search_term = f"%{query}%"
            cursor.execute('''
                SELECT id, name, email, phone, skills, experience, education, pdf_path
                FROM resumes 
                WHERE name LIKE ? OR skills LIKE ? OR experience LIKE ? OR education LIKE ?
                ORDER BY updated_at DESC
            ''', (search_term, search_term, search_term, search_term))
            
            resumes = []
            for row in cursor.fetchall():
                resume = {
                    'id': row[0],
                    'name': row[1],
                    'email': row[2],
                    'phone': row[3],
                    'skills': row[4],
                    'experience': row[5],
                    'education': row[6],
                    'pdf_path': row[7]
                }
                resumes.append(resume)
            
            conn.close()
            return resumes
    
    def get_stats(self):
        """Get database statistics"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM resumes')
            total_resumes = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM updates WHERE action = "add"')
            total_adds = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM updates WHERE action = "update"')
            total_updates = cursor.fetchone()[0]
            
            cursor.execute('SELECT MAX(updated_at) FROM resumes')
            last_update = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'total_resumes': total_resumes,
                'total_adds': total_adds,
                'total_updates': total_updates,
                'last_update': last_update
            }

# Global database instance
shared_db = SharedResumeDB()