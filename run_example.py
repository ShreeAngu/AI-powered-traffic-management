"""
Quick example runner for the AI-powered traffic light system
This script demonstrates the complete workflow from training to evaluation
"""
import os
import sys
import time

# Add src directory to path
sys.path.append('src')

from utils import setup_directories, print_system_info
from traffic_env import TrafficLightEnv, FixedTimeController


def quick_demo():
    """Run a quick demonstration of the system"""
    print("="*60)
    print("AI-POWERED ADAPTIVE TRAFFIC LIGHT SYSTEM")
    print("Quick Demo")
    print("="*60)
    
    # Setup directories
    setup_directories()
    
    # Print system info
    print_system_info()
    
    print("\n" + "="*60)
    print("Running Quick Traffic Simulation Demo...")
    print("="*60)
    
    try:
        # Create environment
        print("Creating traffic environment...")
        env = TrafficLightEnv(use_gui=False, max_steps=100)
        
        # Test environment
        print("Testing environment...")
        obs, info = env.reset()
        print(f"Initial observation (vehicle queues): {obs}")
        
        # Run simulation with random actions
        print("Running simulation with random actions...")
        total_reward = 0
        
        for step in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if step % 10 == 0:
                phase_name = "North-South" if action == 0 else "East-West"
                print(f"Step {step}: {phase_name} Green, "
                      f"Queues: {obs}, "
                      f"Vehicles: {info['vehicle_count']}, "
                      f"Reward: {reward:.2f}")
            
            if terminated or truncated:
                break
        
        print(f"\nSimulation completed!")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Final waiting time: {info['total_waiting_time']:.2f} seconds")
        
        env.close()
        
        # Test fixed-time controller
        print("\n" + "-"*40)
        print("Testing Fixed-Time Controller...")
        
        controller = FixedTimeController(green_time=20, yellow_time=3)
        
        for step in range(10):
            action = controller.get_action(step)
            phase_name = "North-South" if action == 0 else "East-West"
            cycle_pos = step % controller.cycle_time
            print(f"Step {step}: {phase_name} (cycle position: {cycle_pos})")
        
        print("\n✓ Quick demo completed successfully!")
        print("\nTo run the full system:")
        print("1. Train models: python src/train.py")
        print("2. Evaluate performance: python src/evaluate.py")
        print("3. Interactive demo: python src/demo.py")
        
    except Exception as e:
        print(f"✗ Demo failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure SUMO is installed and in PATH")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Run setup: python setup.py")


def run_training_example():
    """Run a minimal training example"""
    print("="*60)
    print("MINIMAL TRAINING EXAMPLE")
    print("="*60)
    
    try:
        from train import train_ppo_agent, evaluate_fixed_time_baseline
        
        print("Running baseline evaluation...")
        baseline_results = evaluate_fixed_time_baseline(episodes=3)
        
        print("\nRunning minimal PPO training...")
        model = train_ppo_agent(total_timesteps=5000, eval_freq=1000)
        
        print("✓ Training example completed!")
        
    except Exception as e:
        print(f"✗ Training example failed: {e}")


def main():
    """Main function"""
    if len(sys.argv) > 1:
        if sys.argv[1] == "demo":
            quick_demo()
        elif sys.argv[1] == "train":
            run_training_example()
        elif sys.argv[1] == "info":
            print_system_info()
        else:
            print("Usage: python run_example.py [demo|train|info]")
    else:
        print("AI-Powered Adaptive Traffic Light System")
        print("Available commands:")
        print("  python run_example.py demo   - Quick demonstration")
        print("  python run_example.py train  - Minimal training example")
        print("  python run_example.py info   - System information")
        print("\nFor full functionality:")
        print("  python setup.py              - Complete setup")
        print("  python src/train.py          - Full training")
        print("  python src/evaluate.py       - Model evaluation")
        print("  python src/demo.py           - Interactive demo")


if __name__ == "__main__":
    main()