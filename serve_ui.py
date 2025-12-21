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

def start_ui_server(port=8080):
    """Start the UI server"""
    try:
        with socketserver.TCPServer(("", port), CustomHTTPRequestHandler) as httpd:
            print(f"🌐 UI Server running at http://localhost:{port}")
            print(f"📁 Serving files from: {os.path.abspath('ui')}")
            print("🚀 Open http://localhost:8080 in your browser")
            print("Press Ctrl+C to stop the server")
            
            # Auto-open browser after a short delay
            def open_browser():
                time.sleep(1)
                webbrowser.open(f'http://localhost:{port}')
            
            browser_thread = threading.Thread(target=open_browser)
            browser_thread.daemon = True
            browser_thread.start()
            
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n🛑 UI Server stopped")
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ Port {port} is already in use. Try a different port:")
            print(f"   python serve_ui.py --port 8081")
        else:
            print(f"❌ Error starting server: {e}")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Serve the AI Traffic Light Control UI')
    parser.add_argument('--port', type=int, default=8080, help='Port to serve on (default: 8080)')
    
    args = parser.parse_args()
    
    print("🚦 AI Traffic Light Control System - UI Server")
    print("=" * 50)
    
    # Check if ui directory exists
    if not os.path.exists('ui'):
        print("❌ UI directory not found. Make sure you're in the project root directory.")
        return
    
    # Check if index.html exists
    if not os.path.exists('ui/index.html'):
        print("❌ UI files not found. Make sure ui/index.html exists.")
        return
    
    start_ui_server(args.port)

if __name__ == "__main__":
    main()