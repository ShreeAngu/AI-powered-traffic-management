#!/usr/bin/env python3
"""
Standalone training script for AI Traffic Light Control System
This script trains models using real SUMO simulation backend
"""
import sys
import os
sys.path.append('src')

from traffic_env import TrafficLightEnv
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class TrainingCallback(BaseCallback):
    """Custom callback for training progress"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # Log progress every 1000 steps
        if self.num_timesteps % 1000 == 0:
            mean_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
            print(f"Step {self.num_timesteps}: Mean reward (last 10 episodes): {mean_reward:.2f}")
        
        return True
    
    def _on_rollout_end(self) -> None:
        # Called at the end of each episode
        if len(self.locals.get('episode_rewards', [])) > len(self.episode_rewards):
            new_rewards = self.locals['episode_rewards'][len(self.episode_rewards):]
            self.episode_rewards.extend(new_rewards)
            
            if new_rewards:
                print(f"Episode {len(self.episode_rewards)}: Reward = {new_rewards[-1]:.2f}")

def train_ppo(total_steps=25000):
    """Train PPO model with real SUMO backend"""
    print("🚦 Training PPO Model with Real SUMO Backend")
    print("=" * 50)
    
    # Create environment
    env = TrafficLightEnv(use_gui=False, max_steps=1000, sumo_port=8814)
    
    # Create PPO model with optimized hyperparameters
    model = PPO(
        'MlpPolicy', 
        env, 
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01
    )
    
    # Create callback
    callback = TrainingCallback(verbose=1)
    
    print(f"Starting training for {total_steps} steps...")
    
    # Train the model
    model.learn(
        total_timesteps=total_steps,
        callback=callback,
        progress_bar=True
    )
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/ppo_traffic_final.zip'
    model.save(model_path)
    print(f"✅ PPO model saved to: {model_path}")
    
    # Evaluate the model
    print("\n🔍 Evaluating trained model...")
    eval_rewards = []
    for episode in range(5):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        eval_rewards.append(episode_reward)
        print(f"Evaluation Episode {episode + 1}: {episode_reward:.2f}")
    
    mean_eval_reward = np.mean(eval_rewards)
    print(f"\n📊 Mean evaluation reward: {mean_eval_reward:.2f}")
    
    env.close()
    return model, mean_eval_reward

def train_dqn(total_steps=25000):
    """Train DQN model with real SUMO backend"""
    print("🚦 Training DQN Model with Real SUMO Backend")
    print("=" * 50)
    
    # Create environment
    env = TrafficLightEnv(use_gui=False, max_steps=1000, sumo_port=8815)
    
    # Create DQN model with optimized hyperparameters
    model = DQN(
        'MlpPolicy', 
        env, 
        verbose=1,
        learning_rate=1e-4,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05
    )
    
    # Create callback
    callback = TrainingCallback(verbose=1)
    
    print(f"Starting training for {total_steps} steps...")
    
    # Train the model
    model.learn(
        total_timesteps=total_steps,
        callback=callback,
        progress_bar=True
    )
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/dqn_traffic_final.zip'
    model.save(model_path)
    print(f"✅ DQN model saved to: {model_path}")
    
    # Evaluate the model
    print("\n🔍 Evaluating trained model...")
    eval_rewards = []
    for episode in range(5):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        eval_rewards.append(episode_reward)
        print(f"Evaluation Episode {episode + 1}: {episode_reward:.2f}")
    
    mean_eval_reward = np.mean(eval_rewards)
    print(f"\n📊 Mean evaluation reward: {mean_eval_reward:.2f}")
    
    env.close()
    return model, mean_eval_reward

def main():
    """Main training function"""
    print("🚦 AI Traffic Light Control System - Standalone Training")
    print("=" * 60)
    
    # Check SUMO availability
    try:
        import traci
        print("✅ SUMO TraCI available")
    except ImportError:
        print("❌ SUMO TraCI not available. Please install SUMO.")
        return
    
    # Check if SUMO files exist
    if not os.path.exists("sumo/intersection.sumocfg"):
        print("❌ SUMO configuration files not found.")
        return
    
    print("✅ SUMO configuration files found")
    
    # Ask user which model to train
    print("\nSelect model to train:")
    print("1. PPO (Proximal Policy Optimization)")
    print("2. DQN (Deep Q-Network)")
    print("3. Both")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        train_ppo()
    elif choice == "2":
        train_dqn()
    elif choice == "3":
        print("\n🔄 Training both models...")
        train_ppo()
        print("\n" + "="*60)
        train_dqn()
    else:
        print("Invalid choice. Exiting.")
        return
    
    print("\n🎉 Training completed!")
    print("You can now use the trained models in the web interface.")

if __name__ == "__main__":
    main()