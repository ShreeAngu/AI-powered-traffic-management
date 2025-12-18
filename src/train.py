"""
Training script for RL agents on traffic light control
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import pandas as pd
from traffic_env import TrafficLightEnv, FixedTimeController


def create_results_dir():
    """Create results directory if it doesn't exist"""
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)


def train_ppo_agent(total_timesteps: int = 100000, 
                   eval_freq: int = 5000,
                   save_freq: int = 10000):
    """
    Train PPO agent for traffic light control
    
    Args:
        total_timesteps: Total training timesteps
        eval_freq: Evaluation frequency
        save_freq: Model save frequency
    """
    print("Training PPO Agent...")
    
    # Create training environment
    def make_env():
        env = TrafficLightEnv(use_gui=False, max_steps=1800)  # 30 minutes
        return Monitor(env, "results/")
    
    # Create vectorized environment
    env = make_vec_env(make_env, n_envs=1)
    
    # Create evaluation environment
    eval_env = Monitor(TrafficLightEnv(use_gui=False, max_steps=1800), 
                      "results/eval/")
    
    # Initialize PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log="results/tensorboard/"
    )
    
    # Setup callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/",
        log_path="results/eval/",
        eval_freq=eval_freq,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path="models/checkpoints/",
        name_prefix="ppo_traffic"
    )
    
    # Train the agent
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Save final model
    model.save("models/ppo_traffic_final")
    
    # Close environments
    env.close()
    eval_env.close()
    
    return model


def train_dqn_agent(total_timesteps: int = 100000,
                   eval_freq: int = 5000,
                   save_freq: int = 10000):
    """
    Train DQN agent for traffic light control
    
    Args:
        total_timesteps: Total training timesteps
        eval_freq: Evaluation frequency
        save_freq: Model save frequency
    """
    print("Training DQN Agent...")
    
    # Create training environment
    def make_env():
        env = TrafficLightEnv(use_gui=False, max_steps=1800)
        return Monitor(env, "results/")
    
    env = make_vec_env(make_env, n_envs=1)
    
    # Create evaluation environment
    eval_env = Monitor(TrafficLightEnv(use_gui=False, max_steps=1800), 
                      "results/eval/")
    
    # Initialize DQN agent
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-3,
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
        exploration_final_eps=0.05,
        tensorboard_log="results/tensorboard/"
    )
    
    # Setup callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/",
        log_path="results/eval/",
        eval_freq=eval_freq,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path="models/checkpoints/",
        name_prefix="dqn_traffic"
    )
    
    # Train the agent
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Save final model
    model.save("models/dqn_traffic_final")
    
    # Close environments
    env.close()
    eval_env.close()
    
    return model


def evaluate_fixed_time_baseline(episodes: int = 10):
    """
    Evaluate fixed-time controller baseline
    
    Args:
        episodes: Number of evaluation episodes
    """
    print("Evaluating Fixed-Time Baseline...")
    
    env = TrafficLightEnv(use_gui=False, max_steps=1800)
    controller = FixedTimeController(green_time=30, yellow_time=4)
    
    episode_rewards = []
    episode_waiting_times = []
    
    for episode in range(episodes):
        obs, info = env.reset()
        episode_reward = 0
        step = 0
        
        while True:
            action = controller.get_action(step)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_waiting_times.append(info['total_waiting_time'])
        
        print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, "
              f"Total Waiting Time={info['total_waiting_time']:.2f}")
    
    env.close()
    
    # Save results
    baseline_results = {
        'episode_rewards': episode_rewards,
        'episode_waiting_times': episode_waiting_times,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_waiting_time': np.mean(episode_waiting_times),
        'std_waiting_time': np.std(episode_waiting_times)
    }
    
    # Save to file
    pd.DataFrame({
        'episode': range(1, episodes + 1),
        'reward': episode_rewards,
        'waiting_time': episode_waiting_times
    }).to_csv('results/baseline_results.csv', index=False)
    
    print(f"Baseline Results:")
    print(f"Mean Reward: {baseline_results['mean_reward']:.2f} ± {baseline_results['std_reward']:.2f}")
    print(f"Mean Waiting Time: {baseline_results['mean_waiting_time']:.2f} ± {baseline_results['std_waiting_time']:.2f}")
    
    return baseline_results


def plot_training_progress():
    """Plot training progress from monitor logs"""
    try:
        # Read monitor logs
        monitor_files = [f for f in os.listdir("results/") if f.endswith(".monitor.csv")]
        
        if not monitor_files:
            print("No monitor files found for plotting")
            return
        
        # Combine all monitor files
        all_data = []
        for file in monitor_files:
            df = pd.read_csv(f"results/{file}", skiprows=1)
            all_data.append(df)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Plot episode rewards
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(combined_df['r'])
            plt.title('Episode Rewards During Training')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            # Rolling average
            window = min(100, len(combined_df) // 10)
            if window > 1:
                rolling_mean = combined_df['r'].rolling(window=window).mean()
                plt.plot(rolling_mean)
                plt.title(f'Rolling Average Reward (window={window})')
                plt.xlabel('Episode')
                plt.ylabel('Average Reward')
                plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('results/training_progress.png', dpi=150, bbox_inches='tight')
            plt.show()
            
    except Exception as e:
        print(f"Error plotting training progress: {e}")


def main():
    """Main training function"""
    print("Starting Traffic Light RL Training...")
    
    # Create directories
    create_results_dir()
    
    # Evaluate baseline first
    print("\n" + "="*50)
    baseline_results = evaluate_fixed_time_baseline(episodes=5)
    
    # Train PPO agent
    print("\n" + "="*50)
    ppo_model = train_ppo_agent(total_timesteps=50000, eval_freq=2500)
    
    # Train DQN agent
    print("\n" + "="*50)
    dqn_model = train_dqn_agent(total_timesteps=50000, eval_freq=2500)
    
    # Plot training progress
    print("\n" + "="*50)
    plot_training_progress()
    
    print("\nTraining completed! Check the 'results/' and 'models/' directories.")
    print("Run 'python evaluate.py' to compare all trained models.")


if __name__ == "__main__":
    main()