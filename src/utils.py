"""
Utility functions for the traffic light control system
"""
import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import xml.etree.ElementTree as ET


def setup_directories():
    """Create necessary directories for the project"""
    directories = [
        "results",
        "results/detailed", 
        "results/eval",
        "results/tensorboard",
        "models",
        "models/checkpoints",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")


def check_sumo_installation():
    """Check if SUMO is properly installed"""
    try:
        import traci
        import sumolib
        print("✓ SUMO Python tools found")
        
        # Try to find SUMO binary
        sumo_binary = "sumo"
        if os.system(f"{sumo_binary} --version > /dev/null 2>&1") == 0:
            print("✓ SUMO binary found")
            return True
        else:
            print("✗ SUMO binary not found in PATH")
            return False
            
    except ImportError as e:
        print(f"✗ SUMO Python tools not found: {e}")
        print("Install with: pip install sumo-tools")
        return False


def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        "gymnasium",
        "stable_baselines3", 
        "torch",
        "matplotlib",
        "pandas",
        "numpy",
        "seaborn"
    ]
    
    optional_packages = [
        "ultralytics",  # For YOLOv8
        "cv2"          # OpenCV
    ]
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing_required.append(package)
            print(f"✗ {package}")
    
    for package in optional_packages:
        try:
            if package == "cv2":
                import cv2
            else:
                __import__(package)
            print(f"✓ {package} (optional)")
        except ImportError:
            missing_optional.append(package)
            print(f"- {package} (optional)")
    
    if missing_required:
        print(f"\nMissing required packages: {missing_required}")
        print("Install with: pip install " + " ".join(missing_required))
        return False
    
    if missing_optional:
        print(f"\nMissing optional packages: {missing_optional}")
        print("For camera perception, install: pip install ultralytics opencv-python")
    
    return True


def parse_sumo_tripinfo(tripinfo_file: str) -> pd.DataFrame:
    """
    Parse SUMO tripinfo XML file to extract vehicle statistics
    
    Args:
        tripinfo_file: Path to tripinfo.xml file
        
    Returns:
        DataFrame with trip information
    """
    if not os.path.exists(tripinfo_file):
        print(f"Tripinfo file not found: {tripinfo_file}")
        return pd.DataFrame()
    
    try:
        tree = ET.parse(tripinfo_file)
        root = tree.getroot()
        
        trips = []
        for tripinfo in root.findall('tripinfo'):
            trip_data = {
                'id': tripinfo.get('id'),
                'depart': float(tripinfo.get('depart', 0)),
                'arrival': float(tripinfo.get('arrival', 0)),
                'duration': float(tripinfo.get('duration', 0)),
                'routeLength': float(tripinfo.get('routeLength', 0)),
                'waitingTime': float(tripinfo.get('waitingTime', 0)),
                'waitingCount': int(tripinfo.get('waitingCount', 0)),
                'stopTime': float(tripinfo.get('stopTime', 0)),
                'timeLoss': float(tripinfo.get('timeLoss', 0)),
                'rerouteNo': int(tripinfo.get('rerouteNo', 0)),
                'vType': tripinfo.get('vType', 'unknown')
            }
            trips.append(trip_data)
        
        return pd.DataFrame(trips)
        
    except Exception as e:
        print(f"Error parsing tripinfo file: {e}")
        return pd.DataFrame()


def analyze_traffic_performance(tripinfo_df: pd.DataFrame) -> Dict:
    """
    Analyze traffic performance from tripinfo data
    
    Args:
        tripinfo_df: DataFrame with trip information
        
    Returns:
        Dictionary with performance metrics
    """
    if tripinfo_df.empty:
        return {}
    
    metrics = {
        'total_vehicles': len(tripinfo_df),
        'avg_duration': tripinfo_df['duration'].mean(),
        'avg_waiting_time': tripinfo_df['waitingTime'].mean(),
        'avg_time_loss': tripinfo_df['timeLoss'].mean(),
        'avg_speed': (tripinfo_df['routeLength'] / tripinfo_df['duration']).mean(),
        'total_waiting_time': tripinfo_df['waitingTime'].sum(),
        'max_waiting_time': tripinfo_df['waitingTime'].max(),
        'vehicles_with_waiting': (tripinfo_df['waitingTime'] > 0).sum(),
        'avg_stops': tripinfo_df['waitingCount'].mean()
    }
    
    return metrics


def plot_traffic_metrics(tripinfo_file: str, save_path: str = None):
    """
    Create plots from SUMO tripinfo data
    
    Args:
        tripinfo_file: Path to tripinfo.xml file
        save_path: Optional path to save plots
    """
    df = parse_sumo_tripinfo(tripinfo_file)
    
    if df.empty:
        print("No data to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Traffic Performance Analysis', fontsize=16)
    
    # Plot 1: Waiting time distribution
    axes[0, 0].hist(df['waitingTime'], bins=30, alpha=0.7, color='blue')
    axes[0, 0].set_title('Waiting Time Distribution')
    axes[0, 0].set_xlabel('Waiting Time (s)')
    axes[0, 0].set_ylabel('Number of Vehicles')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Duration vs Route Length
    axes[0, 1].scatter(df['routeLength'], df['duration'], alpha=0.6)
    axes[0, 1].set_title('Trip Duration vs Route Length')
    axes[0, 1].set_xlabel('Route Length (m)')
    axes[0, 1].set_ylabel('Duration (s)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Time loss distribution
    axes[1, 0].hist(df['timeLoss'], bins=30, alpha=0.7, color='red')
    axes[1, 0].set_title('Time Loss Distribution')
    axes[1, 0].set_xlabel('Time Loss (s)')
    axes[1, 0].set_ylabel('Number of Vehicles')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Average speed by vehicle type
    if 'vType' in df.columns:
        speed_by_type = df.groupby('vType').apply(
            lambda x: (x['routeLength'] / x['duration']).mean()
        )
        axes[1, 1].bar(speed_by_type.index, speed_by_type.values)
        axes[1, 1].set_title('Average Speed by Vehicle Type')
        axes[1, 1].set_xlabel('Vehicle Type')
        axes[1, 1].set_ylabel('Speed (m/s)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def create_config_file(config_path: str = "config.json"):
    """Create a configuration file with default settings"""
    config = {
        "simulation": {
            "max_steps": 3600,
            "step_length": 1.0,
            "use_gui": False
        },
        "traffic_light": {
            "yellow_time": 4,
            "min_green_time": 10,
            "junction_id": "J0"
        },
        "training": {
            "total_timesteps": 100000,
            "eval_freq": 5000,
            "save_freq": 10000,
            "n_eval_episodes": 10
        },
        "ppo": {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99
        },
        "dqn": {
            "learning_rate": 1e-3,
            "buffer_size": 50000,
            "batch_size": 32,
            "gamma": 0.99,
            "exploration_fraction": 0.1
        },
        "camera": {
            "model_path": "yolov8n.pt",
            "confidence_threshold": 0.5,
            "image_width": 800,
            "image_height": 600
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration file created: {config_path}")
    return config


def load_config(config_path: str = "config.json") -> Dict:
    """Load configuration from file"""
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        print("Creating default configuration...")
        return create_config_file(config_path)
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Configuration loaded from: {config_path}")
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return create_config_file(config_path)


def print_system_info():
    """Print system information and status"""
    print("="*60)
    print("AI-POWERED ADAPTIVE TRAFFIC LIGHT SYSTEM")
    print("System Information")
    print("="*60)
    
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    print("\nDependency Check:")
    deps_ok = check_dependencies()
    
    print("\nSUMO Installation:")
    sumo_ok = check_sumo_installation()
    
    print("\nProject Structure:")
    for root, dirs, files in os.walk("."):
        # Skip hidden directories and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        level = root.replace(".", "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 2 * (level + 1)
        for file in files:
            if not file.startswith('.') and not file.endswith('.pyc'):
                print(f"{subindent}{file}")
    
    print("\nModel Status:")
    model_files = [
        "models/ppo_traffic_final.zip",
        "models/dqn_traffic_final.zip", 
        "models/best_model.zip"
    ]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            size = os.path.getsize(model_file) / (1024*1024)  # MB
            print(f"✓ {model_file} ({size:.1f} MB)")
        else:
            print(f"- {model_file} (not found)")
    
    print(f"\nSystem Status: {'✓ Ready' if deps_ok and sumo_ok else '✗ Issues found'}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "setup":
            setup_directories()
        elif sys.argv[1] == "check":
            print_system_info()
        elif sys.argv[1] == "config":
            create_config_file()
        else:
            print("Usage: python utils.py [setup|check|config]")
    else:
        print_system_info()