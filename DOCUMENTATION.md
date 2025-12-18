# AI-Powered Adaptive Traffic Light System - Technical Documentation

## Overview

This project implements a complete reinforcement learning system for adaptive traffic light control using SUMO (Simulation of Urban MObility). The system demonstrates how AI can outperform traditional fixed-time traffic signals by adapting to real-time traffic conditions.

## Architecture

### Core Components

1. **SUMO Simulation Environment** (`sumo/`)
   - Network definition with 4-way intersection
   - Traffic flow patterns and vehicle routes
   - Configuration files for simulation parameters

2. **Gymnasium Environment** (`src/traffic_env.py`)
   - Custom RL environment implementing Gymnasium interface
   - TraCI integration for SUMO communication
   - Observation: Vehicle queue lengths per direction
   - Action: Traffic light phase selection (NS/EW green)
   - Reward: Negative total waiting time (minimize congestion)

3. **RL Agents** (`src/train.py`)
   - PPO (Proximal Policy Optimization) implementation
   - DQN (Deep Q-Network) implementation
   - Stable-Baselines3 integration
   - Hyperparameter optimization

4. **Baseline Controller** (`src/traffic_env.py`)
   - Fixed-time traffic light controller
   - Configurable green/yellow phase durations
   - Performance comparison baseline

5. **Camera Perception** (`src/camera_perception.py`)
   - YOLOv8-based vehicle detection
   - SUMO screenshot processing
   - Real-world sensor simulation

6. **Evaluation System** (`src/evaluate.py`)
   - Performance metrics calculation
   - Statistical analysis and visualization
   - Model comparison framework

## Technical Specifications

### Environment Details

**Observation Space:**
- Type: `Box(4,)` - Continuous values [0, 100]
- Content: `[west_queue, north_queue, east_queue, south_queue]`
- Units: Number of halting vehicles per direction

**Action Space:**
- Type: `Discrete(2)`
- Actions: `0` = North-South green, `1` = East-West green
- Constraints: Minimum green time, yellow phase transitions

**Reward Function:**
```python
reward = -sum(vehicle_waiting_times) + flow_bonus
```

### Network Architecture

**SUMO Network:**
- 4-way intersection with traffic lights
- 2 lanes per direction (8 total incoming lanes)
- Speed limit: 50 km/h (13.89 m/s)
- Lane length: 200m per segment

**Traffic Patterns:**
- Morning rush hour simulation
- Asymmetric flow patterns (higher EW traffic)
- Mixed vehicle types (cars, trucks)
- Stochastic arrival patterns

### RL Algorithm Configuration

**PPO Parameters:**
```python
learning_rate = 3e-4
n_steps = 2048
batch_size = 64
n_epochs = 10
gamma = 0.99
gae_lambda = 0.95
clip_range = 0.2
```

**DQN Parameters:**
```python
learning_rate = 1e-3
buffer_size = 50000
batch_size = 32
gamma = 0.99
exploration_fraction = 0.1
target_update_interval = 1000
```

## Performance Metrics

### Primary Metrics
1. **Total Waiting Time**: Sum of all vehicle waiting times
2. **Average Speed**: Mean vehicle speed across simulation
3. **Throughput**: Vehicles completed per time unit
4. **Queue Length**: Average queue size per direction

### Secondary Metrics
1. **Episode Reward**: Cumulative reward per episode
2. **Phase Switching Frequency**: Traffic light change rate
3. **Fairness**: Waiting time distribution across directions
4. **Fuel Consumption**: Estimated based on speed profiles

## Installation and Setup

### Prerequisites
- Python 3.10+
- SUMO 1.15.0+
- CUDA (optional, for GPU acceleration)

### Quick Setup
```bash
# Clone repository
git clone <repository-url>
cd ai-traffic-light-system

# Install dependencies
pip install -r requirements.txt

# Run setup
python setup.py

# Quick test
python run_example.py demo
```

### Manual Setup
```bash
# Install SUMO
# Windows: Download from https://eclipse.dev/sumo/
# Linux: sudo apt-get install sumo sumo-tools
# macOS: brew install sumo

# Install Python packages
pip install sumo-tools gymnasium stable-baselines3 torch matplotlib pandas numpy

# Optional: Camera perception
pip install ultralytics opencv-python

# Create directories
python src/utils.py setup
```

## Usage Guide

### 1. Basic Training
```bash
# Train both PPO and DQN agents
python src/train.py

# Monitor training progress
tensorboard --logdir results/tensorboard/
```

### 2. Model Evaluation
```bash
# Evaluate all trained models
python src/evaluate.py

# View results
cat results/performance_comparison.csv
```

### 3. Interactive Demo
```bash
# Run interactive demonstration
python src/demo.py

# Specific demos
python src/demo.py fixed    # Fixed-time controller
python src/demo.py ppo      # PPO agent
python src/demo.py camera   # Camera perception
```

### 4. Camera Perception
```bash
# Test YOLOv8 vehicle detection
python src/camera_perception.py

# Run with camera-based environment
python src/demo.py camera
```

## File Structure

```
ai-traffic-light-system/
├── README.md                 # Project overview
├── DOCUMENTATION.md          # This file
├── requirements.txt          # Python dependencies
├── setup.py                 # Setup script
├── run_example.py           # Quick examples
├── .gitignore              # Git ignore rules
├── config.json             # Configuration file
│
├── sumo/                   # SUMO simulation files
│   ├── intersection.net.xml    # Network definition
│   ├── intersection.rou.xml    # Traffic routes
│   ├── intersection.sumocfg    # SUMO configuration
│   └── gui-settings.cfg        # GUI settings
│
├── src/                    # Python source code
│   ├── traffic_env.py          # Gymnasium environment
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation script
│   ├── demo.py                # Demonstration script
│   ├── camera_perception.py   # YOLOv8 integration
│   └── utils.py               # Utility functions
│
├── results/                # Generated results
│   ├── model_comparison.png    # Performance plots
│   ├── performance_comparison.csv
│   ├── detailed/              # Detailed results
│   └── tensorboard/           # Training logs
│
└── models/                 # Trained models
    ├── ppo_traffic_final.zip
    ├── dqn_traffic_final.zip
    ├── best_model.zip
    └── checkpoints/           # Training checkpoints
```

## Customization Guide

### Modifying Traffic Patterns
Edit `sumo/intersection.rou.xml`:
```xml
<flow id="custom_flow" route="WE" begin="0" end="3600" vehsPerHour="1200"/>
```

### Adjusting Reward Function
Modify `traffic_env.py`:
```python
def _calculate_reward(self):
    waiting_penalty = -sum(waiting_times)
    speed_bonus = avg_speed * 0.1
    fairness_penalty = -std(lane_queues) * 0.5
    return waiting_penalty + speed_bonus + fairness_penalty
```

### Adding New Metrics
Extend `evaluate.py`:
```python
def custom_metric(self, info):
    return info['custom_value'] * weight
```

### Hyperparameter Tuning
Modify `config.json`:
```json
{
  "ppo": {
    "learning_rate": 1e-4,
    "n_steps": 4096,
    "batch_size": 128
  }
}
```

## Troubleshooting

### Common Issues

1. **SUMO Not Found**
   ```bash
   # Check SUMO installation
   sumo --version
   
   # Add to PATH (Linux/macOS)
   export PATH=$PATH:/usr/share/sumo/bin
   ```

2. **TraCI Connection Error**
   ```python
   # Ensure SUMO process is closed
   if traci.isLoaded():
       traci.close()
   ```

3. **Memory Issues**
   ```python
   # Reduce buffer size for DQN
   buffer_size = 10000
   
   # Use smaller batch sizes
   batch_size = 16
   ```

4. **Slow Training**
   ```bash
   # Use GPU acceleration
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # Reduce simulation complexity
   max_steps = 1800  # 30 minutes instead of 60
   ```

### Performance Optimization

1. **Vectorized Environments**
   ```python
   env = make_vec_env(make_env, n_envs=4)  # Parallel training
   ```

2. **Faster Simulation**
   ```xml
   <step-length value="0.5"/>  <!-- Larger time steps -->
   ```

3. **Reduced Logging**
   ```python
   verbose=0  # Disable training output
   ```

## Research Extensions

### Potential Improvements

1. **Multi-Agent Systems**
   - Coordinate multiple intersections
   - Distributed traffic control
   - Communication protocols

2. **Advanced Perception**
   - LiDAR simulation
   - Multi-modal sensor fusion
   - Uncertainty quantification

3. **Real-World Deployment**
   - Hardware integration
   - Safety constraints
   - Regulatory compliance

4. **Advanced Algorithms**
   - Multi-objective optimization
   - Hierarchical RL
   - Meta-learning approaches

### Experimental Variations

1. **Network Topologies**
   - T-intersections
   - Roundabouts
   - Highway on-ramps

2. **Traffic Scenarios**
   - Emergency vehicles
   - Pedestrian crossings
   - Construction zones

3. **Evaluation Metrics**
   - Environmental impact
   - Economic costs
   - User satisfaction

## References

1. SUMO Documentation: https://eclipse.dev/sumo/
2. Stable-Baselines3: https://stable-baselines3.readthedocs.io/
3. Gymnasium: https://gymnasium.farama.org/
4. YOLOv8: https://github.com/ultralytics/ultralytics

## License

This project is provided as an educational prototype. Please ensure compliance with local traffic regulations and safety standards before any real-world deployment.