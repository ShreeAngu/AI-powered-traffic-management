"""
Simple HTTP server to serve the UI files
This ensures proper CORS and file serving for the web interface
"""
import http.server
import socketserver
import webbrowser
import os
import threading
import time

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory="ui", **kwargs)
    
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()

def start_http_server(port=8080):
    """Start HTTP server for UI files"""
    try:
        with socketserver.TCPServer(("", port), CustomHTTPRequestHandler) as httpd:
            print(f"🌐 HTTP Server running on http://localhost:{port}")
            print(f"📁 Serving files from: {os.path.abspath('ui')}")
            print("🚀 Opening web browser...")
            
            # Open browser after a short delay
            def open_browser():
                time.sleep(1)
                webbrowser.open(f"http://localhost:{port}")
            
            threading.Thread(target=open_browser, daemon=True).start()
            
            print("Press Ctrl+C to stop the server")
            httpd.serve_forever()
            
    except OSError as e:
        if e.errno == 10048:  # Port already in use
            print(f"❌ Port {port} is already in use")
            print(f"🔄 Trying port {port + 1}...")
            start_http_server(port + 1)
        else:
            print(f"❌ Error starting HTTP server: {e}")

if __name__ == "__main__":
    print("🚦 TrafficAI Manager - UI Server")
    print("=" * 50)
    
    # Check if UI files exist
    if not os.path.exists("ui/index.html"):
        print("❌ UI files not found. Please ensure ui/index.html exists.")
        exit(1)
    
    start_http_server()