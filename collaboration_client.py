import socket
import threading
import json
import time
from pathlib import Path

class CollaborationClient:
    def __init__(self, host='localhost', port=8888):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        self.callback = None
        self.receive_thread = None
        
    def set_callback(self, callback):
        """Set callback function for receiving updates"""
        self.callback = callback
    
    def connect(self):
        """Connect to collaboration server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.connected = True
            
            # Start receiving thread
            self.receive_thread = threading.Thread(target=self._receive_messages, daemon=True)
            self.receive_thread.start()
            
            # Request sync with current server data
            self.request_sync()
            
            return True
        except Exception as e:
            print(f"Failed to connect to server: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from server"""
        self.connected = False
        if self.socket:
            self.socket.close()
            self.socket = None
    
    def send_update(self, resume_data):
        """Send resume data update to server"""
        if not self.connected:
            return False
            
        try:
            message = json.dumps({
                'type': 'update_resumes',
                'data': resume_data,
                'timestamp': time.time()
            })
            self.socket.send((message + '\n').encode('utf-8'))
            return True
        except Exception as e:
            print(f"Failed to send update: {e}")
            self.connected = False
            return False
    
    def request_sync(self):
        """Request synchronization with server data"""
        if not self.connected:
            return False
            
        try:
            message = json.dumps({'type': 'request_sync'})
            self.socket.send((message + '\n').encode('utf-8'))
            return True
        except Exception as e:
            print(f"Failed to request sync: {e}")
            return False
    
    def _receive_messages(self):
        """Receive messages from server in background thread"""
        buffer = ""
        while self.connected:
            try:
                data = self.socket.recv(1024).decode('utf-8')
                if not data:
                    break
                
                buffer += data
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        self._handle_message(line.strip())
                        
            except Exception as e:
                if self.connected:  # Only print if we're supposed to be connected
                    print(f"Error receiving message: {e}")
                break
        
        self.connected = False
    
    def _handle_message(self, message):
        """Handle incoming message from server"""
        try:
            data = json.loads(message)
            
            if data.get('type') == 'sync_data' and self.callback:
                self.callback('sync', data.get('data', {}))
            elif data.get('type') == 'update_resumes' and self.callback:
                self.callback('update', data.get('data', {}))
                
        except json.JSONDecodeError:
            print(f"Invalid JSON message: {message}")