"""
Traffic Light Control Environment using SUMO and Gymnasium
"""
import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import traci
import sumolib
from typing import Dict, List, Tuple, Optional
import xml.etree.ElementTree as ET


class TrafficLightEnv(gym.Env):
    """
    Custom Gymnasium environment for traffic light control using SUMO simulation.
    
    Observation Space: Number of halting vehicles on each of 4 incoming lanes
    Action Space: Discrete(2) - 0: North-South green, 1: East-West green
    Reward: Negative sum of waiting times (minimize congestion)
    """
    
    def __init__(self, 
                 sumo_cfg_file: str = "sumo/intersection.sumocfg",
                 use_gui: bool = False,
                 max_steps: int = 3600,
                 yellow_time: int = 4,
                 min_green_time: int = 10):
        """
        Initialize the traffic light environment.
        
        Args:
            sumo_cfg_file: Path to SUMO configuration file
            use_gui: Whether to use SUMO GUI
            max_steps: Maximum simulation steps
            yellow_time: Duration of yellow phase
            min_green_time: Minimum green phase duration
        """
        super().__init__()
        
        # Environment parameters
        self.sumo_cfg_file = sumo_cfg_file
        self.use_gui = use_gui
        self.max_steps = max_steps
        self.yellow_time = yellow_time
        self.min_green_time = min_green_time
        
        # Traffic light junction ID (main intersection)
        self.junction_id = "J0"
        
        # Incoming lanes to monitor (4 directions)
        self.incoming_lanes = ["W0_0", "W0_1", "N0_0", "N0_1", 
                              "E0_0", "E0_1", "S0_0", "S0_1"]
        
        # Define observation and action spaces
        # Observation: halting vehicles count for each lane (4 lanes aggregated by direction)
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(4,), dtype=np.float32
        )
        
        # Action: 0 = NS green, 1 = EW green
        self.action_space = spaces.Discrete(2)
        
        # Traffic light phases
        self.phases = {
            0: "GGrr",  # North-South green, East-West red
            1: "rrGG"   # North-South red, East-West green
        }
        
        # State variables
        self.current_phase = 0
        self.time_since_last_phase_change = 0
        self.step_count = 0
        self.total_waiting_time = 0
        self.is_yellow = False
        self.yellow_timer = 0
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_waiting_times = []
        
    def _start_sumo(self):
        """Start SUMO simulation"""
        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        
        # Build SUMO command
        sumo_cmd = [
            sumo_binary,
            "-c", self.sumo_cfg_file,
            "--tripinfo-output", "results/tripinfo.xml",
            "--summary-output", "results/summary.xml",
            "--no-step-log",
            "--no-warnings",
            "--random"
        ]
        
        # Start TraCI connection
        traci.start(sumo_cmd)
        
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        # Close existing connection if any
        if traci.isLoaded():
            traci.close()
            
        # Start new simulation
        self._start_sumo()
        
        # Reset state variables
        self.current_phase = 0
        self.time_since_last_phase_change = 0
        self.step_count = 0
        self.total_waiting_time = 0
        self.is_yellow = False
        self.yellow_timer = 0
        
        # Set initial traffic light phase
        traci.trafficlight.setPhase(self.junction_id, self.current_phase)
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """Execute one environment step"""
        # Handle yellow phase transition
        if self.is_yellow:
            self.yellow_timer += 1
            if self.yellow_timer >= self.yellow_time:
                # End yellow phase, switch to new phase
                self.current_phase = action
                traci.trafficlight.setPhase(self.junction_id, self.current_phase)
                self.is_yellow = False
                self.yellow_timer = 0
                self.time_since_last_phase_change = 0
        else:
            # Check if we need to change phase
            if (action != self.current_phase and 
                self.time_since_last_phase_change >= self.min_green_time):
                # Start yellow phase
                yellow_phase = "yyyy"
                traci.trafficlight.setRedYellowGreenState(self.junction_id, yellow_phase)
                self.is_yellow = True
                self.yellow_timer = 0
            
        # Advance simulation
        traci.simulationStep()
        self.step_count += 1
        self.time_since_last_phase_change += 1
        
        # Get new observation and reward
        observation = self._get_observation()
        reward = self._calculate_reward()
        
        # Check if episode is done
        terminated = (self.step_count >= self.max_steps or 
                     traci.simulation.getMinExpectedNumber() <= 0)
        truncated = False
        
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation: halting vehicles per direction
        Returns array of shape (4,) representing [West, North, East, South]
        """
        try:
            # Get halting vehicles for each direction
            west_halting = (traci.lane.getLastStepHaltingNumber("W0_0") + 
                           traci.lane.getLastStepHaltingNumber("W0_1"))
            
            north_halting = (traci.lane.getLastStepHaltingNumber("N0_0") + 
                            traci.lane.getLastStepHaltingNumber("N0_1"))
            
            east_halting = (traci.lane.getLastStepHaltingNumber("E0_0") + 
                           traci.lane.getLastStepHaltingNumber("E0_1"))
            
            south_halting = (traci.lane.getLastStepHaltingNumber("S0_0") + 
                            traci.lane.getLastStepHaltingNumber("S0_1"))
            
            observation = np.array([west_halting, north_halting, east_halting, south_halting], 
                                 dtype=np.float32)
            
        except traci.exceptions.TraCIException:
            # Return zeros if lanes not found
            observation = np.zeros(4, dtype=np.float32)
            
        return observation
    
    def _calculate_reward(self) -> float:
        """
        Calculate reward based on traffic performance.
        Reward = -sum(waiting_times) to minimize congestion
        """
        try:
            # Get waiting time for all vehicles
            vehicle_ids = traci.vehicle.getIDList()
            current_waiting_time = 0
            
            for veh_id in vehicle_ids:
                waiting_time = traci.vehicle.getWaitingTime(veh_id)
                current_waiting_time += waiting_time
            
            # Reward is negative waiting time (minimize congestion)
            reward = -current_waiting_time
            
            # Add small bonus for keeping traffic flowing
            if current_waiting_time == 0:
                reward += 10
                
            self.total_waiting_time += current_waiting_time
            
        except traci.exceptions.TraCIException:
            reward = 0
            
        return reward
    
    def _get_info(self) -> Dict:
        """Get additional information about the environment state"""
        try:
            vehicle_count = traci.vehicle.getIDCount()
            avg_speed = 0
            if vehicle_count > 0:
                speeds = [traci.vehicle.getSpeed(veh_id) 
                         for veh_id in traci.vehicle.getIDList()]
                avg_speed = np.mean(speeds) if speeds else 0
                
        except traci.exceptions.TraCIException:
            vehicle_count = 0
            avg_speed = 0
            
        return {
            "step": self.step_count,
            "current_phase": self.current_phase,
            "vehicle_count": vehicle_count,
            "avg_speed": avg_speed,
            "total_waiting_time": self.total_waiting_time,
            "is_yellow": self.is_yellow
        }
    
    def close(self):
        """Close the environment"""
        if traci.isLoaded():
            traci.close()
    
    def render(self, mode="human"):
        """Render the environment (SUMO GUI handles this)"""
        pass


class FixedTimeController:
    """
    Baseline fixed-time traffic light controller for comparison
    """
    
    def __init__(self, green_time: int = 30, yellow_time: int = 4):
        """
        Initialize fixed-time controller
        
        Args:
            green_time: Duration of green phase
            yellow_time: Duration of yellow phase
        """
        self.green_time = green_time
        self.yellow_time = yellow_time
        self.cycle_time = 2 * (green_time + yellow_time)
        
    def get_action(self, step: int) -> int:
        """
        Get action based on fixed timing
        
        Args:
            step: Current simulation step
            
        Returns:
            Action (0 or 1)
        """
        cycle_position = step % self.cycle_time
        
        if cycle_position < self.green_time:
            return 0  # NS green
        elif cycle_position < self.green_time + self.yellow_time:
            return 0  # NS yellow (keep same action)
        elif cycle_position < 2 * self.green_time + self.yellow_time:
            return 1  # EW green
        else:
            return 1  # EW yellow (keep same action)


def test_environment():
    """Test the traffic environment"""
    print("Testing Traffic Light Environment...")
    
    # Create environment
    env = TrafficLightEnv(use_gui=False, max_steps=100)
    
    try:
        # Test reset
        obs, info = env.reset()
        print(f"Initial observation: {obs}")
        print(f"Initial info: {info}")
        
        # Test random actions
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step {i}: Action={action}, Reward={reward:.2f}, "
                  f"Vehicles={info['vehicle_count']}")
            
            if terminated or truncated:
                break
                
        print("Environment test completed successfully!")
        
    except Exception as e:
        print(f"Error testing environment: {e}")
    finally:
        env.close()


if __name__ == "__main__":
    test_environment()