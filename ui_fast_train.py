#!/usr/bin/env python3
"""
UI Fast Training - Ultra-Quick Training for Web Interface
Optimized for 1-2 minute training sessions with immediate results
"""
import sys
import os
sys.path.append('src')

from traffic_env import TrafficLightEnv
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import torch

class UICallback(BaseCallback):
    """Ultra-fast callback for UI training"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        
    def _on_step(self) -> bool:
        # Update every 100 steps for ultra-fast feedback
        if self.num_timesteps % 100 == 0:
            progress = self.num_timesteps / 3000  # 3000 total steps
            print(f"UI Training: {progress*100:.1f}% complete ({self.num_timesteps}/3000 steps)")
        return True

def ui_train_ppo():
    """Ultra-fast PPO training for UI (1-2 minutes)"""
    print("🚀 UI Fast PPO Training - 1-2 Minutes")
    print("=" * 40)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔥 Training on: {device.upper()}")
    
    # Create environment with very short episodes
    env = TrafficLightEnv(
        use_gui=False, 
        max_steps=200,  # Very short episodes (3.3 minutes)
        sumo_port=8850,
        yellow_time=2,
        min_green_time=4
    )
    
    # Ultra-fast PPO configuration
    model = PPO(
        'MlpPolicy', 
        env, 
        verbose=1,
        learning_rate=1e-3,      # Very high learning rate
        n_steps=256,             # Very small buffer
        batch_size=64,           # Small batch for speed
        n_epochs=3,              # Minimal epochs
        gamma=0.95,              # Lower gamma for faster convergence
        gae_lambda=0.9,          
        clip_range=0.3,          # Higher clip range
        ent_coef=0.05,           # High entropy
        device=device,
        policy_kwargs=dict(
            net_arch=[64, 64],    # Very small network
            activation_fn=torch.nn.ReLU
        )
    )
    
    callback = UICallback(verbose=1)
    
    print("🎯 Starting ULTRA-FAST UI training...")
    print("📊 Target: 3000 steps (~15 episodes)")
    print("⏱️  Time: ~1-2 minutes")
    
    # Train with minimal steps for UI demo
    model.learn(
        total_timesteps=3000,
        callback=callback
    )
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/ppo_traffic_ui_fast.zip'
    model.save(model_path)
    print(f"✅ UI Fast model saved: {model_path}")
    
    env.close()
    return model

def ui_train_dqn():
    """Ultra-fast DQN training for UI (1-2 minutes)"""
    print("🚀 UI Fast DQN Training - 1-2 Minutes")
    print("=" * 40)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔥 Training on: {device.upper()}")
    
    # Create environment with very short episodes
    env = TrafficLightEnv(
        use_gui=False, 
        max_steps=200,
        sumo_port=8851,
        yellow_time=2,
        min_green_time=4
    )
    
    # Ultra-fast DQN configuration
    model = DQN(
        'MlpPolicy', 
        env, 
        verbose=1,
        learning_rate=5e-4,      # Very high learning rate
        buffer_size=5000,        # Very small buffer
        learning_starts=100,     # Start learning very early
        batch_size=32,           # Small batch
        tau=0.05,               # Very fast target updates
        gamma=0.95,             # Lower gamma
        train_freq=1,           # Train every step
        gradient_steps=1,       
        target_update_interval=100,  # Very frequent updates
        exploration_fraction=0.02,   # Minimal exploration
        exploration_initial_eps=0.8,
        exploration_final_eps=0.05,
        device=device,
        policy_kwargs=dict(
            net_arch=[64, 64],    # Very small network
            activation_fn=torch.nn.ReLU
        )
    )
    
    callback = UICallback(verbose=1)
    
    print("🎯 Starting ULTRA-FAST UI DQN training...")
    print("📊 Target: 2500 steps (~12 episodes)")
    print("⏱️  Time: ~1-2 minutes")
    
    # Train with minimal steps
    model.learn(
        total_timesteps=2500,
        callback=callback
    )
    
    # Save the model
    model_path = 'models/dqn_traffic_ui_fast.zip'
    model.save(model_path)
    print(f"✅ UI Fast model saved: {model_path}")
    
    env.close()
    return model

def main():
    """Main UI training function"""
    print("🚀 UI Fast Training - Instant Results")
    print("=" * 50)
    
    print("Select UI training mode:")
    print("1. ⚡ UI Fast PPO (3K steps, ~1-2 min)")
    print("2. ⚡ UI Fast DQN (2.5K steps, ~1-2 min)")
    print("3. 🔥 Both UI Models (~2-4 min total)")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        ui_train_ppo()
    elif choice == "2":
        ui_train_dqn()
    elif choice == "3":
        print("Training both UI models...")
        ui_train_ppo()
        print("\n" + "="*50)
        ui_train_dqn()
    else:
        print("Invalid choice.")
        return
    
    print("\n🎉 UI Fast Training completed!")
    print("🚀 Models ready for immediate use in web interface!")

if __name__ == "__main__":
    main()