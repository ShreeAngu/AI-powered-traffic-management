"""
Evaluation script for comparing RL agents with baseline controller
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO, DQN
from traffic_env import TrafficLightEnv, FixedTimeController


def evaluate_model(model, env, episodes=10, model_name="Model"):
    """
    Evaluate a trained model
    
    Args:
        model: Trained RL model
        env: Environment instance
        episodes: Number of evaluation episodes
        model_name: Name for logging
        
    Returns:
        Dictionary with evaluation results
    """
    print(f"Evaluating {model_name}...")
    
    episode_rewards = []
    episode_waiting_times = []
    episode_throughput = []
    episode_avg_speeds = []
    
    for episode in range(episodes):
        obs, info = env.reset()
        episode_reward = 0
        vehicles_completed = 0
        total_speed = 0
        speed_measurements = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            # Track additional metrics
            if info['avg_speed'] > 0:
                total_speed += info['avg_speed']
                speed_measurements += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_waiting_times.append(info['total_waiting_time'])
        
        # Calculate average speed for episode
        avg_speed = total_speed / speed_measurements if speed_measurements > 0 else 0
        episode_avg_speeds.append(avg_speed)
        
        print(f"  Episode {episode + 1}: Reward={episode_reward:.2f}, "
              f"Waiting Time={info['total_waiting_time']:.2f}, "
              f"Avg Speed={avg_speed:.2f}")
    
    results = {
        'model_name': model_name,
        'episode_rewards': episode_rewards,
        'episode_waiting_times': episode_waiting_times,
        'episode_avg_speeds': episode_avg_speeds,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_waiting_time': np.mean(episode_waiting_times),
        'std_waiting_time': np.std(episode_waiting_times),
        'mean_avg_speed': np.mean(episode_avg_speeds),
        'std_avg_speed': np.std(episode_avg_speeds)
    }
    
    return results


def evaluate_baseline(episodes=10):
    """Evaluate fixed-time baseline controller"""
    print("Evaluating Fixed-Time Baseline...")
    
    env = TrafficLightEnv(use_gui=False, max_steps=1800)
    controller = FixedTimeController(green_time=30, yellow_time=4)
    
    episode_rewards = []
    episode_waiting_times = []
    episode_avg_speeds = []
    
    for episode in range(episodes):
        obs, info = env.reset()
        episode_reward = 0
        step = 0
        total_speed = 0
        speed_measurements = 0
        
        while True:
            action = controller.get_action(step)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
            
            # Track additional metrics
            if info['avg_speed'] > 0:
                total_speed += info['avg_speed']
                speed_measurements += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_waiting_times.append(info['total_waiting_time'])
        
        # Calculate average speed for episode
        avg_speed = total_speed / speed_measurements if speed_measurements > 0 else 0
        episode_avg_speeds.append(avg_speed)
        
        print(f"  Episode {episode + 1}: Reward={episode_reward:.2f}, "
              f"Waiting Time={info['total_waiting_time']:.2f}, "
              f"Avg Speed={avg_speed:.2f}")
    
    env.close()
    
    results = {
        'model_name': 'Fixed-Time Baseline',
        'episode_rewards': episode_rewards,
        'episode_waiting_times': episode_waiting_times,
        'episode_avg_speeds': episode_avg_speeds,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_waiting_time': np.mean(episode_waiting_times),
        'std_waiting_time': np.std(episode_waiting_times),
        'mean_avg_speed': np.mean(episode_avg_speeds),
        'std_avg_speed': np.std(episode_avg_speeds)
    }
    
    return results


def load_and_evaluate_models(episodes=10):
    """Load and evaluate all trained models"""
    results = []
    
    # Evaluate baseline
    baseline_results = evaluate_baseline(episodes)
    results.append(baseline_results)
    
    # Create environment for RL models
    env = TrafficLightEnv(use_gui=False, max_steps=1800)
    
    # Evaluate PPO model
    try:
        if os.path.exists("models/ppo_traffic_final.zip"):
            ppo_model = PPO.load("models/ppo_traffic_final")
            ppo_results = evaluate_model(ppo_model, env, episodes, "PPO")
            results.append(ppo_results)
        else:
            print("PPO model not found. Skipping PPO evaluation.")
    except Exception as e:
        print(f"Error loading PPO model: {e}")
    
    # Evaluate DQN model
    try:
        if os.path.exists("models/dqn_traffic_final.zip"):
            dqn_model = DQN.load("models/dqn_traffic_final")
            dqn_results = evaluate_model(dqn_model, env, episodes, "DQN")
            results.append(dqn_results)
        else:
            print("DQN model not found. Skipping DQN evaluation.")
    except Exception as e:
        print(f"Error loading DQN model: {e}")
    
    # Evaluate best model if available
    try:
        if os.path.exists("models/best_model.zip"):
            best_model = PPO.load("models/best_model")  # Assuming PPO, adjust if needed
            best_results = evaluate_model(best_model, env, episodes, "Best Model")
            results.append(best_results)
    except Exception as e:
        print(f"Error loading best model: {e}")
    
    env.close()
    return results


def create_comparison_plots(results):
    """Create comparison plots for all models"""
    if not results:
        print("No results to plot")
        return
    
    # Prepare data for plotting
    model_names = [r['model_name'] for r in results]
    mean_rewards = [r['mean_reward'] for r in results]
    std_rewards = [r['std_reward'] for r in results]
    mean_waiting_times = [r['mean_waiting_time'] for r in results]
    std_waiting_times = [r['std_waiting_time'] for r in results]
    mean_speeds = [r['mean_avg_speed'] for r in results]
    std_speeds = [r['std_avg_speed'] for r in results]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Traffic Light Control: Model Comparison', fontsize=16)
    
    # Plot 1: Mean Rewards
    axes[0, 0].bar(model_names, mean_rewards, yerr=std_rewards, capsize=5, 
                   color=['red', 'blue', 'green', 'orange'][:len(model_names)])
    axes[0, 0].set_title('Average Episode Reward')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Mean Waiting Times
    axes[0, 1].bar(model_names, mean_waiting_times, yerr=std_waiting_times, capsize=5,
                   color=['red', 'blue', 'green', 'orange'][:len(model_names)])
    axes[0, 1].set_title('Average Total Waiting Time')
    axes[0, 1].set_ylabel('Waiting Time (seconds)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Mean Average Speeds
    axes[1, 0].bar(model_names, mean_speeds, yerr=std_speeds, capsize=5,
                   color=['red', 'blue', 'green', 'orange'][:len(model_names)])
    axes[1, 0].set_title('Average Vehicle Speed')
    axes[1, 0].set_ylabel('Speed (m/s)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Box plot of rewards
    reward_data = []
    labels = []
    for result in results:
        reward_data.extend(result['episode_rewards'])
        labels.extend([result['model_name']] * len(result['episode_rewards']))
    
    df_rewards = pd.DataFrame({'Model': labels, 'Reward': reward_data})
    sns.boxplot(data=df_rewards, x='Model', y='Reward', ax=axes[1, 1])
    axes[1, 1].set_title('Reward Distribution')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def create_performance_table(results):
    """Create a performance comparison table"""
    if not results:
        print("No results to create table")
        return
    
    # Create DataFrame
    df_data = []
    for result in results:
        df_data.append({
            'Model': result['model_name'],
            'Mean Reward': f"{result['mean_reward']:.2f} ± {result['std_reward']:.2f}",
            'Mean Waiting Time': f"{result['mean_waiting_time']:.2f} ± {result['std_waiting_time']:.2f}",
            'Mean Speed': f"{result['mean_avg_speed']:.2f} ± {result['std_avg_speed']:.2f}",
            'Improvement vs Baseline (Reward)': 'Baseline' if result['model_name'] == 'Fixed-Time Baseline' 
                                              else f"{((result['mean_reward'] - results[0]['mean_reward']) / abs(results[0]['mean_reward']) * 100):+.1f}%",
            'Improvement vs Baseline (Waiting Time)': 'Baseline' if result['model_name'] == 'Fixed-Time Baseline'
                                                    else f"{((results[0]['mean_waiting_time'] - result['mean_waiting_time']) / results[0]['mean_waiting_time'] * 100):+.1f}%"
        })
    
    df = pd.DataFrame(df_data)
    
    # Save to CSV
    df.to_csv('results/performance_comparison.csv', index=False)
    
    # Print table
    print("\n" + "="*100)
    print("PERFORMANCE COMPARISON TABLE")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100)
    
    return df


def save_detailed_results(results):
    """Save detailed results to files"""
    os.makedirs('results/detailed', exist_ok=True)
    
    for result in results:
        model_name = result['model_name'].replace(' ', '_').replace('-', '_').lower()
        
        # Save episode-by-episode results
        episode_df = pd.DataFrame({
            'episode': range(1, len(result['episode_rewards']) + 1),
            'reward': result['episode_rewards'],
            'waiting_time': result['episode_waiting_times'],
            'avg_speed': result['episode_avg_speeds']
        })
        
        episode_df.to_csv(f'results/detailed/{model_name}_episodes.csv', index=False)
        
        # Save summary statistics
        summary_df = pd.DataFrame({
            'metric': ['mean_reward', 'std_reward', 'mean_waiting_time', 'std_waiting_time', 
                      'mean_avg_speed', 'std_avg_speed'],
            'value': [result['mean_reward'], result['std_reward'], result['mean_waiting_time'],
                     result['std_waiting_time'], result['mean_avg_speed'], result['std_avg_speed']]
        })
        
        summary_df.to_csv(f'results/detailed/{model_name}_summary.csv', index=False)


def main():
    """Main evaluation function"""
    print("Starting Traffic Light Model Evaluation...")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Load and evaluate all models
    print("\n" + "="*60)
    results = load_and_evaluate_models(episodes=10)
    
    if not results:
        print("No models found to evaluate!")
        return
    
    # Create comparison plots
    print("\n" + "="*60)
    print("Creating comparison plots...")
    create_comparison_plots(results)
    
    # Create performance table
    print("\n" + "="*60)
    performance_df = create_performance_table(results)
    
    # Save detailed results
    print("\n" + "="*60)
    print("Saving detailed results...")
    save_detailed_results(results)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    if len(results) > 1:
        baseline = results[0]  # Assuming first is baseline
        best_model = max(results[1:], key=lambda x: x['mean_reward']) if len(results) > 1 else None
        
        if best_model:
            reward_improvement = ((best_model['mean_reward'] - baseline['mean_reward']) / 
                                abs(baseline['mean_reward']) * 100)
            waiting_time_improvement = ((baseline['mean_waiting_time'] - best_model['mean_waiting_time']) / 
                                      baseline['mean_waiting_time'] * 100)
            
            print(f"Best performing model: {best_model['model_name']}")
            print(f"Reward improvement over baseline: {reward_improvement:+.1f}%")
            print(f"Waiting time reduction: {waiting_time_improvement:+.1f}%")
    
    print("\nEvaluation completed! Check the 'results/' directory for detailed outputs.")
    print("Files generated:")
    print("- results/model_comparison.png")
    print("- results/performance_comparison.csv")
    print("- results/detailed/ (individual model results)")


if __name__ == "__main__":
    main()