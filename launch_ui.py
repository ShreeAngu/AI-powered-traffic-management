"""
UI Launcher for AI Traffic Light Control System
Starts the backend server and opens the web interface
"""
import os
import sys
import time
import webbrowser
import subprocess
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import websockets
        return True
    except ImportError:
        print("Installing websockets...")
        subprocess.run([sys.executable, "-m", "pip", "install", "websockets"], check=True)
        return True


def start_server():
    """Start both WebSocket and HTTP servers"""
    print("Starting AI Traffic Light Control System...")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        print("Failed to install dependencies")
        return False
    
    # Start WebSocket server in background
    print("🔌 Starting WebSocket server...")
    ws_server_process = subprocess.Popen([
        sys.executable, "ui_server.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait a moment for WebSocket server to start
    time.sleep(2)
    
    # Check if WebSocket server is running
    if ws_server_process.poll() is None:
        print("✓ WebSocket server started successfully")
        
        # Start HTTP server for UI files
        print("🌐 Starting HTTP server for UI...")
        http_server_process = subprocess.Popen([
            sys.executable, "serve_ui.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for HTTP server to start
        time.sleep(2)
        
        if http_server_process.poll() is None:
            print("✓ HTTP server started successfully")
            
            print("\n" + "=" * 60)
            print("🚦 AI Traffic Light Control System is running!")
            print("=" * 60)
            print("🌐 Web Interface: http://localhost:8080")
            print("🔌 WebSocket Server: ws://localhost:8765")
            print("🛑 To stop: Press Ctrl+C")
            print("=" * 60)
            
            try:
                # Wait for servers to finish
                while True:
                    if ws_server_process.poll() is not None:
                        print("WebSocket server stopped")
                        break
                    if http_server_process.poll() is not None:
                        print("HTTP server stopped")
                        break
                    time.sleep(1)
                    
            except KeyboardInterrupt:
                print("\n\nShutting down...")
                ws_server_process.terminate()
                http_server_process.terminate()
                ws_server_process.wait()
                http_server_process.wait()
                print("✓ Servers stopped")
            
            return True
        else:
            print("✗ Failed to start HTTP server")
            ws_server_process.terminate()
            return False
        
    else:
        print("✗ Failed to start WebSocket server")
        stdout, stderr = ws_server_process.communicate()
        if stderr:
            print(f"Error: {stderr.decode()}")
        return False


def main():
    """Main entry point"""
    print("🚦 AI Traffic Light Control System Launcher")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("src/traffic_env.py").exists():
        print("✗ Please run this script from the project root directory")
        print("  Current directory should contain 'src/traffic_env.py'")
        return
    
    # Check if SUMO is available
    try:
        import traci
        print("✓ SUMO Python tools available")
    except ImportError:
        print("✗ SUMO Python tools not found")
        print("  Please install SUMO and add it to your PATH")
        return
    
    # Start the system
    if start_server():
        print("System launched successfully!")
    else:
        print("Failed to launch system")


if __name__ == "__main__":
    main()