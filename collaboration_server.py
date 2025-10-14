import socket
import threading
import json
import time
from pathlib import Path

class CollaborationServer:
    def __init__(self, host='localhost', port=8888):
        self.host = host
        self.port = port
        self.clients = []
        self.resume_data = {}
        self.lock = threading.Lock()
        
    def broadcast(self, message, sender_client=None):
        """Broadcast message to all connected clients except sender"""
        with self.lock:
            disconnected_clients = []
            for client in self.clients:
                if client != sender_client:
                    try:
                        client.send((message + '\n').encode('utf-8'))
                    except:
                        disconnected_clients.append(client)
            
            # Remove disconnected clients
            for client in disconnected_clients:
                self.clients.remove(client)
    
    def handle_client(self, client_socket, address):
        """Handle individual client connection"""
        print(f"New connection from {address}")
        
        with self.lock:
            self.clients.append(client_socket)
        
        try:
            while True:
                message = client_socket.recv(1024).decode('utf-8').strip()
                if not message:
                    break
                
                try:
                    data = json.loads(message)
                    if data.get('type') == 'update_resumes':
                        self.resume_data = data.get('data', {})
                        # Broadcast update to all other clients
                        self.broadcast(message, client_socket)
                        print(f"Resume data updated by {address}")
                    
                    elif data.get('type') == 'request_sync':
                        # Send current resume data to requesting client
                        sync_message = json.dumps({
                            'type': 'sync_data',
                            'data': self.resume_data
                        })
                        client_socket.send((sync_message + '\n').encode('utf-8'))
                        
                except json.JSONDecodeError:
                    print(f"Invalid JSON from {address}")
                    
        except Exception as e:
            print(f"Client {address} disconnected: {e}")
        finally:
            with self.lock:
                if client_socket in self.clients:
                    self.clients.remove(client_socket)
            client_socket.close()
            print(f"Connection with {address} closed")
    
    def start_server(self):
        """Start the collaboration server"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            server_socket.bind((self.host, self.port))
            server_socket.listen(5)
            print(f"Collaboration server running on {self.host}:{self.port}")
            
            while True:
                client_socket, address = server_socket.accept()
                client_thread = threading.Thread(
                    target=self.handle_client, 
                    args=(client_socket, address),
                    daemon=True
                )
                client_thread.start()
                
        except Exception as e:
            print(f"Server error: {e}")
        finally:
            server_socket.close()

def start_collaboration_server():
    """Start the collaboration server in a separate thread"""
    server = CollaborationServer()
    server_thread = threading.Thread(target=server.start_server, daemon=True)
    server_thread.start()
    return server