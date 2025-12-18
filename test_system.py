"""
System test script to verify the AI traffic light system works correctly
"""
import os
import sys
import traceback

# Add src to path
sys.path.append('src')

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import numpy as np
        print("✓ numpy")
    except ImportError as e:
        print(f"✗ numpy: {e}")
        return False
    
    try:
        import gymnasium as gym
        print("✓ gymnasium")
    except ImportError as e:
        print(f"✗ gymnasium: {e}")
        return False
    
    try:
        import stable_baselines3
        print("✓ stable_baselines3")
    except ImportError as e:
        print(f"✗ stable_baselines3: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ matplotlib")
    except ImportError as e:
        print(f"✗ matplotlib: {e}")
        return False
    
    try:
        import pandas as pd
        print("✓ pandas")
    except ImportError as e:
        print(f"✗ pandas: {e}")
        return False
    
    try:
        import traci
        import sumolib
        print("✓ SUMO tools")
    except ImportError as e:
        print(f"✗ SUMO tools: {e}")
        print("Install with: pip install sumo-tools")
        return False
    
    # Check for SUMO binary
    sumo_found = check_sumo_binary()
    if sumo_found:
        print("✓ SUMO binary")
    else:
        print("⚠️  SUMO binary not in PATH")
        print("Run: python setup_windows.py (Windows) or python setup.py")
    
    return True


def check_sumo_binary():
    """Check if SUMO binary is available"""
    import subprocess
    
    # Try common SUMO paths on Windows
    if sys.platform == "win32":
        sumo_paths = [
            "sumo.exe",  # In PATH
            "D:\\sumo\\bin\\sumo.exe",
            "C:\\Program Files (x86)\\Eclipse\\Sumo\\bin\\sumo.exe",
            "C:\\Program Files\\Eclipse\\Sumo\\bin\\sumo.exe"
        ]
        
        for sumo_path in sumo_paths:
            try:
                result = subprocess.run([sumo_path, "--version"], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    # Add to PATH if not already there
                    if sumo_path != "sumo.exe":
                        sumo_dir = os.path.dirname(sumo_path)
                        if sumo_dir not in os.environ.get("PATH", ""):
                            os.environ["PATH"] = sumo_dir + ";" + os.environ.get("PATH", "")
                            print(f"Added {sumo_dir} to PATH for this session")
                    return True
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                continue
    else:
        # Linux/macOS
        try:
            result = subprocess.run(["sumo", "--version"], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass
    
    return False


def test_sumo_files():
    """Test that SUMO files exist and are valid"""
    print("\nTesting SUMO files...")
    
    required_files = [
        "sumo/intersection.net.xml",
        "sumo/intersection.rou.xml", 
        "sumo/intersection.sumocfg"
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} not found")
            return False
    
    return True


def test_environment():
    """Test the traffic environment"""
    print("\nTesting traffic environment...")
    
    try:
        from traffic_env import TrafficLightEnv, FixedTimeController
        
        # Test environment creation
        env = TrafficLightEnv(use_gui=False, max_steps=10)
        print("✓ Environment created")
        
        # Test reset
        obs, info = env.reset()
        print(f"✓ Environment reset, observation shape: {obs.shape}")
        
        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✓ Environment step, reward: {reward:.2f}")
        
        # Test fixed controller
        controller = FixedTimeController()
        action = controller.get_action(0)
        print(f"✓ Fixed controller, action: {action}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        traceback.print_exc()
        return False


def test_training_setup():
    """Test training components"""
    print("\nTesting training setup...")
    
    try:
        from stable_baselines3 import PPO, DQN
        from stable_baselines3.common.env_util import make_vec_env
        from traffic_env import TrafficLightEnv
        
        # Test PPO creation
        def make_env():
            return TrafficLightEnv(use_gui=False, max_steps=10)
        
        env = make_vec_env(make_env, n_envs=1)
        model = PPO("MlpPolicy", env, verbose=0)
        print("✓ PPO model created")
        
        # Test DQN creation
        model = DQN("MlpPolicy", env, verbose=0)
        print("✓ DQN model created")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Training setup test failed: {e}")
        traceback.print_exc()
        return False


def test_optional_features():
    """Test optional features like camera perception"""
    print("\nTesting optional features...")
    
    # Test YOLOv8 (optional)
    try:
        from ultralytics import YOLO
        import cv2
        print("✓ YOLOv8 and OpenCV available")
        
        from camera_perception import CameraPerceptionSystem
        camera_system = CameraPerceptionSystem()
        print("✓ Camera perception system created")
        
    except ImportError:
        print("- YOLOv8/OpenCV not available (optional)")
    except Exception as e:
        print(f"- Camera perception test failed: {e}")
    
    return True


def run_full_test():
    """Run complete system test"""
    print("="*60)
    print("AI-POWERED ADAPTIVE TRAFFIC LIGHT SYSTEM")
    print("System Test")
    print("="*60)
    
    tests = [
        ("Import Test", test_imports),
        ("SUMO Files Test", test_sumo_files),
        ("Environment Test", test_environment),
        ("Training Setup Test", test_training_setup),
        ("Optional Features Test", test_optional_features)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * len(test_name))
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Run training: python src/train.py")
        print("2. Run evaluation: python src/evaluate.py")
        print("3. Run demo: python src/demo.py")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        print("\nTroubleshooting:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Install SUMO: https://eclipse.dev/sumo/")
        print("3. Run setup: python setup.py")
    
    return passed == total


if __name__ == "__main__":
    success = run_full_test()
    sys.exit(0 if success else 1)