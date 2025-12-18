"""
Demonstration script for the AI-powered traffic light system
"""
import os
import sys
import time
import numpy as np
from stable_baselines3 import PPO, DQN
from traffic_env import TrafficLightEnv, FixedTimeController
from camera_perception import CameraBasedTrafficEnv, CameraPerceptionSystem


def demo_fixed_time_controller(duration: int = 300):
    """
    Demonstrate fixed-time traffic light controller
    
    Args:
        duration: Demo duration in simulation steps
    """
    print("="*60)
    print("DEMO: Fixed-Time Traffic Light Controller")
    print("="*60)
    
    env = TrafficLightEnv(use_gui=True, max_steps=duration)
    controller = FixedTimeController(green_time=30, yellow_time=4)
    
    try:
        obs, info = env.reset()
        total_reward = 0
        step = 0
        
        print("Running fixed-time controller...")
        print("Green phases: 30 seconds NS, 30 seconds EW, repeating")
        
        while step < duration:
            action = controller.get_action(step)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            
            # Print status every 30 steps
            if step % 30 == 0:
                phase_name = "North-South" if action == 0 else "East-West"
                print(f"Step {step}: {phase_name} Green, "
                      f"Vehicles: {info['vehicle_count']}, "
                      f"Avg Speed: {info['avg_speed']:.2f} m/s, "
                      f"Total Waiting: {info['total_waiting_time']:.1f}s")
            
            if terminated or truncated:
                break
                
        print(f"\nFixed-Time Controller Results:")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Final Waiting Time: {info['total_waiting_time']:.2f} seconds")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    finally:
        env.close()


def demo_rl_agent(model_path: str, model_type: str = "PPO", duration: int = 300):
    """
    Demonstrate trained RL agent
    
    Args:
        model_path: Path to trained model
        model_type: Type of model (PPO or DQN)
        duration: Demo duration in simulation steps
    """
    print("="*60)
    print(f"DEMO: {model_type} RL Agent")
    print("="*60)
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Please train a model first using train.py")
        return
    
    env = TrafficLightEnv(use_gui=True, max_steps=duration)
    
    try:
        # Load model
        if model_type.upper() == "PPO":
            model = PPO.load(model_path)
        elif model_type.upper() == "DQN":
            model = DQN.load(model_path)
        else:
            print(f"Unsupported model type: {model_type}")
            return
        
        obs, info = env.reset()
        total_reward = 0
        step = 0
        
        print(f"Running {model_type} agent...")
        print("Agent makes decisions based on real-time traffic conditions")
        
        while step < duration:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            
            # Print status every 30 steps
            if step % 30 == 0:
                phase_name = "North-South" if action == 0 else "East-West"
                print(f"Step {step}: {phase_name} Green, "
                      f"Queue: {obs}, "
                      f"Vehicles: {info['vehicle_count']}, "
                      f"Avg Speed: {info['avg_speed']:.2f} m/s, "
                      f"Total Waiting: {info['total_waiting_time']:.1f}s")
            
            if terminated or truncated:
                break
                
        print(f"\n{model_type} Agent Results:")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Final Waiting Time: {info['total_waiting_time']:.2f} seconds")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Error running {model_type} demo: {e}")
    finally:
        env.close()


def demo_camera_perception(duration: int = 200):
    """
    Demonstrate camera-based perception system
    
    Args:
        duration: Demo duration in simulation steps
    """
    print("="*60)
    print("DEMO: Camera-Based Vehicle Detection")
    print("="*60)
    
    try:
        # Create environment with camera perception
        base_env = TrafficLightEnv(use_gui=True, max_steps=duration)
        camera_env = CameraBasedTrafficEnv(base_env, use_camera=True)
        
        obs, info = env.reset()
        step = 0
        
        print("Running camera-based perception...")
        print("YOLOv8 detects vehicles from SUMO screenshots")
        
        while step < duration:
            action = base_env.action_space.sample()  # Random actions for demo
            obs, reward, terminated, truncated, info = camera_env.step(action)
            step += 1
            
            # Print comparison every 20 steps
            if step % 20 == 0 and 'traci_obs' in info:
                print(f"Step {step}:")
                print(f"  Camera detection: {obs}")
                print(f"  TraCI ground truth: {info['traci_obs']}")
                print(f"  Difference: {info['obs_difference']:.2f}")
            
            if terminated or truncated:
                break
                
        print("\nCamera perception demo completed!")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Error running camera demo: {e}")
        print("Make sure YOLOv8 is installed: pip install ultralytics")
    finally:
        if 'base_env' in locals():
            base_env.close()


def interactive_demo():
    """Interactive demo menu"""
    print("="*60)
    print("AI-POWERED ADAPTIVE TRAFFIC LIGHT SYSTEM")
    print("Interactive Demo")
    print("="*60)
    
    while True:
        print("\nAvailable Demos:")
        print("1. Fixed-Time Controller (Baseline)")
        print("2. PPO RL Agent")
        print("3. DQN RL Agent")
        print("4. Camera-Based Perception")
        print("5. Compare All Methods")
        print("0. Exit")
        
        try:
            choice = input("\nSelect demo (0-5): ").strip()
            
            if choice == "0":
                print("Goodbye!")
                break
            elif choice == "1":
                demo_fixed_time_controller()
            elif choice == "2":
                demo_rl_agent("models/ppo_traffic_final.zip", "PPO")
            elif choice == "3":
                demo_rl_agent("models/dqn_traffic_final.zip", "DQN")
            elif choice == "4":
                demo_camera_perception()
            elif choice == "5":
                compare_all_methods()
            else:
                print("Invalid choice. Please select 0-5.")
                
        except KeyboardInterrupt:
            print("\nDemo interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def compare_all_methods(duration: int = 180):
    """
    Run all methods sequentially for comparison
    
    Args:
        duration: Duration for each method demo
    """
    print("="*60)
    print("COMPARISON: All Traffic Control Methods")
    print("="*60)
    
    methods = [
        ("Fixed-Time Controller", lambda: demo_fixed_time_controller(duration)),
    ]
    
    # Add RL methods if models exist
    if os.path.exists("models/ppo_traffic_final.zip"):
        methods.append(("PPO RL Agent", 
                       lambda: demo_rl_agent("models/ppo_traffic_final.zip", "PPO", duration)))
    
    if os.path.exists("models/dqn_traffic_final.zip"):
        methods.append(("DQN RL Agent", 
                       lambda: demo_rl_agent("models/dqn_traffic_final.zip", "DQN", duration)))
    
    print(f"Running {len(methods)} methods for {duration} steps each...")
    
    for i, (name, demo_func) in enumerate(methods):
        print(f"\n[{i+1}/{len(methods)}] Running {name}...")
        input("Press Enter to start (or Ctrl+C to skip)...")
        
        try:
            demo_func()
        except KeyboardInterrupt:
            print(f"Skipped {name}")
            continue
        
        if i < len(methods) - 1:
            input("\nPress Enter to continue to next method...")
    
    print("\nComparison completed!")
    print("Check results/performance_comparison.csv for detailed metrics")


def quick_test():
    """Quick test to verify everything works"""
    print("Running quick system test...")
    
    try:
        # Test environment creation
        env = TrafficLightEnv(use_gui=False, max_steps=10)
        obs, info = env.reset()
        
        # Test a few steps
        for i in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
        
        env.close()
        print("✓ Environment test passed")
        
        # Test fixed controller
        controller = FixedTimeController()
        action = controller.get_action(0)
        print("✓ Fixed controller test passed")
        
        print("✓ All tests passed! System is ready.")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        print("Please check your SUMO installation and Python dependencies")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            quick_test()
        elif sys.argv[1] == "fixed":
            demo_fixed_time_controller()
        elif sys.argv[1] == "ppo":
            demo_rl_agent("models/ppo_traffic_final.zip", "PPO")
        elif sys.argv[1] == "dqn":
            demo_rl_agent("models/dqn_traffic_final.zip", "DQN")
        elif sys.argv[1] == "camera":
            demo_camera_perception()
        elif sys.argv[1] == "compare":
            compare_all_methods()
        else:
            print("Usage: python demo.py [test|fixed|ppo|dqn|camera|compare]")
    else:
        interactive_demo()