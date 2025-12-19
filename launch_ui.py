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
    """Start the WebSocket server"""
    print("Starting AI Traffic Light Control System...")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        print("Failed to install dependencies")
        return False
    
    # Start server in background
    server_process = subprocess.Popen([
        sys.executable, "ui_server.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait a moment for server to start
    time.sleep(2)
    
    # Check if server is running
    if server_process.poll() is None:
        print("✓ Backend server started successfully")
        
        # Open web browser
        ui_path = Path("ui/index.html").absolute()
        if ui_path.exists():
            print(f"✓ Opening web interface: {ui_path}")
            webbrowser.open(f"file://{ui_path}")
        else:
            print("✗ UI files not found")
            return False
        
        print("\n" + "=" * 60)
        print("AI Traffic Light Control System is running!")
        print("=" * 60)
        print("📊 Web Interface: Open ui/index.html in your browser")
        print("🔌 WebSocket Server: ws://localhost:8765")
        print("🛑 To stop: Press Ctrl+C")
        print("=" * 60)
        
        try:
            # Wait for server to finish
            server_process.wait()
        except KeyboardInterrupt:
            print("\n\nShutting down...")
            server_process.terminate()
            server_process.wait()
            print("✓ Server stopped")
        
        return True
    else:
        print("✗ Failed to start backend server")
        stdout, stderr = server_process.communicate()
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