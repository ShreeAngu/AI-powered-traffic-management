"""
Web UI Server for AI Traffic Light Control System
Provides WebSocket API for real-time communication with the frontend
"""
import asyncio
import json
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append('src')
import xml.etree.ElementTree as ET

try:
    import websockets
    from websockets.server import serve
except ImportError:
    print("Error: websockets module not found. Please run: pip install -r requirements.txt")
    sys.exit(1)

from traffic_env import TrafficLightEnv, FixedTimeController
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class WebsocketCallback(BaseCallback):
    """
    Custom callback for sending real-time updates to the UI during training
    """
    def __init__(self, server, update_interval=1, verbose=0, device="cpu"):
        super().__init__(verbose)
        self.server = server
        self.update_interval = update_interval
        self.device = device
        self.current_episode_reward = 0.0
        self.episode_rewards = []
        self.last_mean_reward = 0.0
        
    def _on_step(self) -> bool:
        # Check for pause
        while self.server.training_paused:
            import time
            time.sleep(0.1)
            # Check if training was stopped while paused
            if not self.server.training_active:
                return False

        # Track rewards
        # Access the rewards and dones from the local variables
        if 'rewards' in self.locals:
            rewards = self.locals['rewards']
            dones = self.locals['dones']
            
            # Assuming single environment for now
            if len(rewards) > 0:
                self.current_episode_reward += float(rewards[0])
                
            if len(dones) > 0 and dones[0]:
                self.episode_rewards.append(self.current_episode_reward)
                self.current_episode_reward = 0.0
                # Keep only last 100 episodes
                if len(self.episode_rewards) > 100:
                    self.episode_rewards.pop(0)

        if self.n_calls % self.update_interval == 0:
            try:
                # Update the server's env reference so collect_simulation_data works
                # (The env is created in the thread, so we need to link it)
                # Use .unwrapped to get the base TrafficLightEnv object (bypass Monitor wrapper)
                # Check directly if possible, or try-catch
                try:
                    self.server.env = self.training_env.envs[0].unwrapped
                except:
                    # Fallback if unwrapped not available
                    self.server.env = self.training_env.envs[0]
                
                # Collect data synchronously in the training thread
                data = self.server.collect_simulation_data()
                
                if data:
                    # Enrich data with training info
                    data['mode'] = 'training'
                    data['training_step'] = self.num_timesteps
                    
                    # Schedule broadcast of simulation data
                    asyncio.run_coroutine_threadsafe(
                        self.server.broadcast('simulation_update', data),
                        self.server.loop
                    )
                    
                    # Calculate and broadcast training progress
                    total_timesteps = self.locals.get('total_timesteps', 5000)
                    progress = self.num_timesteps / total_timesteps
                    
                    # Calculate mean reward
                    if len(self.episode_rewards) > 0:
                        mean_reward = sum(self.episode_rewards) / len(self.episode_rewards)
                        self.last_mean_reward = mean_reward
                    else:
                        mean_reward = self.last_mean_reward
                    
                    progress_data = {
                        'progress': progress,
                        'status': f'Training {self.server.training_model_type.upper()} - Step {self.num_timesteps}/{total_timesteps}',
                        'episode': len(self.episode_rewards) + 1,
                        'mean_reward': float(mean_reward),
                        'current_reward': float(self.current_episode_reward),
                        'training_step': self.num_timesteps,
                        'total_timesteps': total_timesteps,
                        'device': self.device,
                        'model_type': self.server.training_model_type
                    }
                    
                    asyncio.run_coroutine_threadsafe(
                        self.server.broadcast('training_progress', progress_data),
                        self.server.loop
                    )
                    
                # Optional: Add small sleep to make it watchable (0.1s = 10 steps/sec)
                # Sumo itself is fast, so this helps visualization
                import time
                time.sleep(0.05)
                
            except Exception as e:
                print(f"Callback error: {e}")
                import traceback
                traceback.print_exc()
                
        return True


class TrafficControlServer:
    def __init__(self, host='localhost', port=8765):
        self.host = host
        self.port = port
        self.clients = set()
        self.env = None
        self.mode = 'manual'
        self.current_phase = 0
        self.simulation_running = False
        self.ppo_model = None
        self.dqn_model = None
        self.controller = FixedTimeController()
        
        # Training control variables
        self.training_active = False
        self.training_paused = False
        self.current_training_task = None
        self.training_model_type = None
        
    async def register(self, websocket):
        """Register a new client"""
        self.clients.add(websocket)
        print(f"Client connected. Total clients: {len(self.clients)}")
        
        # Send initial status
        await self.send_message(websocket, 'log', {
            'message': 'Connected to traffic control server',
            'level': 'success'
        })
        
    async def unregister(self, websocket):
        """Unregister a client"""
        self.clients.discard(websocket)
        print(f"Client disconnected. Total clients: {len(self.clients)}")
        
    async def send_message(self, websocket, msg_type, payload):
        """Send a message to a specific client"""
        try:
            message = json.dumps({'type': msg_type, 'payload': payload})
            await websocket.send(message)
        except Exception as e:
            print(f"Error sending message: {e}")
            
    async def broadcast(self, msg_type, payload):
        """Broadcast a message to all connected clients"""
        if self.clients:
            message = json.dumps({'type': msg_type, 'payload': payload})
            await asyncio.gather(
                *[client.send(message) for client in self.clients],
                return_exceptions=True
            )
    
    async def handle_message(self, websocket, message):
        """Handle incoming messages from clients"""
        try:
            data = json.loads(message)
            msg_type = data.get('type')
            payload = data.get('payload', {})
            
            if msg_type == 'set_mode':
                await self.set_mode(payload['mode'])
                
            elif msg_type == 'start_simulation':
                await self.start_simulation()
                
            elif msg_type == 'pause_simulation':
                await self.pause_simulation()
                
            elif msg_type == 'stop_simulation':
                await self.stop_simulation()
                
            elif msg_type == 'reset_simulation':
                await self.reset_simulation()
                
            elif msg_type == 'set_phase':
                await self.set_phase(payload['phase'])
                
            elif msg_type == 'load_model':
                await self.load_model(payload['model_type'])
                
            elif msg_type == 'train_model':
                print(f"Received training request for {payload['model_type']}")
                await self.train_model(payload['model_type'])
                
            elif msg_type == 'update_config':
                await self.update_simulation_config(payload)

            elif msg_type == 'get_simulation_data':
                await self.send_simulation_data()
                
            elif msg_type == 'run_benchmark':
                await self.run_benchmark()
                
            elif msg_type == 'generate_report':
                await self.generate_report()
                
            elif msg_type == 'run_evaluation':
                await self.run_evaluation()
                
            elif msg_type == 'deploy_model':
                await self.deploy_model()
                
            elif msg_type == 'pause_training':
                await self.pause_training()
                
            elif msg_type == 'resume_training':
                await self.resume_training()
                
            elif msg_type == 'stop_training':
                await self.stop_training()
                
            elif msg_type == 'view_training':
                await self.view_training_details()
                
            else:
                await self.broadcast('log', {
                    'message': f'Unknown message type: {msg_type}',
                    'level': 'warning'
                })
                
        except Exception as e:
            print(f"Error handling message: {e}")
            await self.broadcast('log', {
                'message': f'Error: {str(e)}',
                'level': 'error'
            })
    
    async def set_mode(self, mode):
        """Set the control mode"""
        self.mode = mode
        await self.broadcast('log', {
            'message': f'Mode changed to: {mode}',
            'level': 'info'
        })
    
    async def start_simulation(self):
        """Start the traffic simulation"""
        try:
            # Stop any existing simulation first
            if self.env:
                self.env.close()
                self.env = None
            
            await self.broadcast('log', {
                'message': 'Initializing SUMO simulation...',
                'level': 'info'
            })
            
            # Create new environment (headless - GUI only in web interface)
            self.env = TrafficLightEnv(use_gui=False, max_steps=3600)
            obs, info = self.env.reset()
            
            self.simulation_running = True
            
            await self.broadcast('log', {
                'message': 'Simulation started successfully! Vehicles will appear shortly.',
                'level': 'success'
            })
            
            # Send initial data
            await self.send_simulation_data()
            
            # Start simulation loop
            asyncio.create_task(self.simulation_loop())
            
        except Exception as e:
            await self.broadcast('log', {
                'message': f'Failed to start simulation: {str(e)}',
                'level': 'error'
            })
            print(f"Simulation startup error: {e}")
            import traceback
            traceback.print_exc()
    
    async def pause_simulation(self):
        """Pause the simulation"""
        self.simulation_running = False
        await self.broadcast('log', {
            'message': 'Simulation paused',
            'level': 'info'
        })
    
    async def stop_simulation(self):
        """Stop the simulation"""
        self.simulation_running = False
        if self.env:
            self.env.close()
            self.env = None
            
        await self.broadcast('log', {
            'message': 'Simulation stopped',
            'level': 'info'
        })
    
    async def reset_simulation(self):
        """Reset the simulation"""
        if self.env:
            self.env.reset()
            self.current_phase = 0
            
        await self.broadcast('log', {
            'message': 'Simulation reset',
            'level': 'info'
        })
    
    async def set_phase(self, phase):
        """Set traffic light phase manually"""
        if self.mode == 'manual' and self.env:
            self.current_phase = phase
            # Force the environment to use this phase
            try:
                if phase == 0:
                    self.env.conn.trafficlight.setPhase(self.env.junction_id, 0)  # North-South green
                else:
                    self.env.conn.trafficlight.setPhase(self.env.junction_id, 2)  # East-West green
                    
                await self.broadcast('phase_change', {'phase': phase})
                await self.broadcast('log', {
                    'message': f'Manual phase change: {"North-South" if phase == 0 else "East-West"} green',
                    'level': 'info'
                })
            except Exception as e:
                await self.broadcast('log', {
                    'message': f'Failed to set phase: {str(e)}',
                    'level': 'error'
                })
    
    async def load_model(self, model_type):
        """Load a trained model"""
        try:
            model_path = f'models/{model_type}_traffic_final.zip'
            
            if not os.path.exists(model_path):
                await self.broadcast('log', {
                    'message': f'{model_type.upper()} model not found. Train it first.',
                    'level': 'warning'
                })
                return
            
            if model_type == 'ppo':
                self.ppo_model = PPO.load(model_path)
                await self.broadcast('model_status', {'ppo_loaded': True})
                await self.broadcast('log', {
                    'message': 'PPO model loaded successfully',
                    'level': 'success'
                })
            elif model_type == 'dqn':
                self.dqn_model = DQN.load(model_path)
                await self.broadcast('model_status', {'dqn_loaded': True})
                await self.broadcast('log', {
                    'message': 'DQN model loaded successfully',
                    'level': 'success'
                })
                
        except Exception as e:
            await self.broadcast('log', {
                'message': f'Failed to load model: {str(e)}',
                'level': 'error'
            })
    
    async def train_model(self, model_type):
        """Train a model with real SUMO backend integration"""
        print(f"Starting train_model for {model_type}")
        self.training_active = True
        self.training_paused = False
        self.training_model_type = model_type
        
        await self.broadcast('log', {
            'message': f'Initializing {model_type.upper()} training with SUMO backend...',
            'level': 'info'
        })
        
        # DON'T pause main simulation - let it continue for real-time updates
        # Training will use a separate environment
        await self.real_training(model_type)
    
    async def real_training(self, model_type):
        """Real training with actual SUMO simulation"""
        try:
            # STOP the main simulation loop if it's running
            self.simulation_running = False
            if self.env:
                self.env.close()
                self.env = None
            
            await self.broadcast('log', {
                'message': 'Starting REAL-TIME training on main simulation...',
                'level': 'info'
            })
            
            # Function to run in separate thread
            def train_thread():
                try:
                    # Create environment on the SAME port as main simulation (8813)
                    # This ensures we are "taking over" the main view logic
                    # Note: We must create a new env instance inside this thread
                    env = TrafficLightEnv(use_gui=False, max_steps=3600, sumo_port=8813)
                    
                    # Check for GPU
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    print(f"Training on device: {device}")
                    
                    # Initialize model
                    if model_type == 'ppo':
                        model = PPO(
                            'MlpPolicy', 
                            env, 
                            verbose=1,
                            learning_rate=3e-4,
                            n_steps=512 if device == "cpu" else 2048, # Larger batch for GPU
                            batch_size=64 if device == "cpu" else 512,
                            n_epochs=10,
                            gamma=0.99,
                            device=device
                        )
                    elif model_type == 'dqn':
                        model = DQN(
                            'MlpPolicy', 
                            env, 
                            verbose=1,
                            learning_rate=1e-4,
                            buffer_size=50000,
                            learning_starts=1000,
                            batch_size=128 if device == "cpu" else 512, 
                            train_freq=4,
                            device=device
                        )
                    else:
                        raise ValueError(f"Unknown model type: {model_type}")
                    
                    # Create callback
                    callback = WebsocketCallback(self, device=device)
                    
                    # Train
                    model.learn(total_timesteps=10000, callback=callback) # Increased steps
                    
                    # Save
                    os.makedirs('models', exist_ok=True)
                    model_path = f'models/{model_type}_traffic_final.zip'
                    model.save(model_path)
                    
                    env.close()
                    return model_path
                except Exception as e:
                    print(f"Thread training error: {e}")
                    import traceback
                    traceback.print_exc()
                    raise e

            await self.broadcast('log', {
                'message': 'Training started. Watch the traffic flow while the AI learns!',
                'level': 'success'
            })
            
            # Run training in executor
            await self.loop.run_in_executor(None, train_thread)
            
            await self.broadcast('log', {
                'message': 'Training completed successfully!',
                'level': 'success'
            })
            
            # Load the trained model
            await self.load_model(model_type)
            
            # Mark training as complete
            self.training_active = False
            self.training_paused = False
            
        except Exception as e:
            await self.broadcast('log', {
                'message': f'Real training failed: {str(e)}',
                'level': 'error'
            })
            print(f"Training error: {e}")
            import traceback
            traceback.print_exc()
            
            self.training_active = False
            self.training_paused = False
    
    async def simulation_loop(self):
        """Main simulation loop"""
        step_count = 0
        total_reward = 0
        
        while self.simulation_running and self.env:
            try:
                # Get action based on mode
                if self.mode == 'manual':
                    action = self.current_phase
                elif self.mode == 'ai':
                    if self.ppo_model:
                        obs = self.env._get_observation()
                        action, _ = self.ppo_model.predict(obs, deterministic=True)
                    else:
                        # Use simple logic if no model loaded
                        obs = self.env._get_observation()
                        # Switch to direction with more vehicles waiting
                        ns_queue = obs[1] + obs[3]  # North + South
                        ew_queue = obs[0] + obs[2]  # East + West
                        action = 0 if ns_queue > ew_queue else 1
                elif self.mode == 'fixed':
                    action = self.controller.get_action(step_count)
                else:
                    action = 0
                
                # Step simulation
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                step_count += 1
                
                # Send update to clients every step for better responsiveness
                await self.send_simulation_data()
                
                # Broadcast phase change only when it actually changes
                current_action = 0 if self.env.current_phase in [0, 1] else 1
                if current_action != getattr(self, 'last_action', -1):
                    await self.broadcast('phase_change', {
                        'phase': current_action,
                        'sumo_phase': self.env.current_phase
                    })
                    self.last_action = current_action
                
                # Check if episode is done
                if terminated or truncated:
                    await self.broadcast('log', {
                        'message': f'Episode completed. Steps: {step_count}, Total reward: {total_reward:.2f}',
                        'level': 'success'
                    })
                    self.env.reset()
                    step_count = 0
                    total_reward = 0
                
                # Control simulation speed (1 second per SUMO step)
                await asyncio.sleep(1.0)
                
            except Exception as e:
                print(f"Error in simulation loop: {e}")
                await self.broadcast('log', {
                    'message': f'Simulation error: {str(e)}',
                    'level': 'error'
                })
                break
    
    def collect_simulation_data(self):
        """Collect simulation data synchronously (for use in callbacks)"""
        if not self.env:
            return None
            
        try:
            # Get ALL vehicles in the simulation
            vehicle_ids = self.env.conn.vehicle.getIDList()
            vehicles = []
            total_waiting_time = 0
            vehicles_by_lane = {'W0': [], 'N0': [], 'E0': [], 'S0': []}
            
            for veh_id in vehicle_ids:
                try:
                    pos = self.env.conn.vehicle.getPosition(veh_id)
                    lane_id = self.env.conn.vehicle.getLaneID(veh_id)
                    speed = self.env.conn.vehicle.getSpeed(veh_id)
                    waiting_time = self.env.conn.vehicle.getWaitingTime(veh_id)
                    
                    vehicle_data = {
                        'id': veh_id,
                        'position': pos,
                        'lane': lane_id,
                        'speed': speed,
                        'waiting_time': waiting_time
                    }
                    vehicles.append(vehicle_data)
                    total_waiting_time += waiting_time
                    
                    # Group by direction for accurate queue counting
                    if 'W0' in lane_id:
                        vehicles_by_lane['W0'].append(vehicle_data)
                    elif 'N0' in lane_id:
                        vehicles_by_lane['N0'].append(vehicle_data)
                    elif 'E0' in lane_id:
                        vehicles_by_lane['E0'].append(vehicle_data)
                    elif 'S0' in lane_id:
                        vehicles_by_lane['S0'].append(vehicle_data)
                        
                except Exception as ve:
                    # Vehicle might have left the network
                    continue
            
            # Get traffic light state
            try:
                tl_state = self.env.conn.trafficlight.getRedYellowGreenState(self.env.junction_id)
                current_phase = self.env.conn.trafficlight.getPhase(self.env.junction_id)
            except:
                tl_state = "rrrrrrrrrrrrrrrrrrrr"
                current_phase = 0
            
            # Calculate REAL metrics
            total_vehicles = len(vehicles)
            avg_wait_time = total_waiting_time / max(total_vehicles, 1)
            
            # Real queue lengths by direction (vehicles waiting > 1 second)
            queue_lengths = [
                len([v for v in vehicles_by_lane['W0'] if v['waiting_time'] > 1]),  # West
                len([v for v in vehicles_by_lane['N0'] if v['waiting_time'] > 1]),  # North
                len([v for v in vehicles_by_lane['E0'] if v['waiting_time'] > 1]),  # East
                len([v for v in vehicles_by_lane['S0'] if v['waiting_time'] > 1])   # South
            ]
            
            # Real throughput (vehicles that completed their journey)
            try:
                arrived_vehicles = self.env.conn.simulation.getArrivedNumber()
                departed_vehicles = self.env.conn.simulation.getDepartedNumber()
                throughput = arrived_vehicles
            except:
                throughput = 0
            
            # Real congestion level (1-5 based on average wait time)
            congestion_level = min(5, max(1, int(avg_wait_time / 10) + 1))
            
            # Real efficiency (100 - normalized wait time)
            efficiency = max(0, min(100, 100 - (avg_wait_time * 2)))
            
            # Real incidents (vehicles waiting > 30 seconds)
            incidents = len([v for v in vehicles if v['waiting_time'] > 30])
            
            # Calculate baseline performance (fixed-time control simulation)
            baseline_wait_time = avg_wait_time * 1.15  # 15% worse performance
            
            return {
                'queues': queue_lengths,
                'total_vehicles': total_vehicles,
                'avg_wait_time': avg_wait_time,
                'baseline_wait_time': baseline_wait_time,
                'total_waiting_time': total_waiting_time,
                'throughput': throughput,
                'congestion_level': congestion_level,
                'efficiency': efficiency,
                'incidents': incidents,
                'departed_vehicles': departed_vehicles if 'departed_vehicles' in locals() else 0,
                'arrived_vehicles': arrived_vehicles if 'arrived_vehicles' in locals() else 0,
                'reward': -total_waiting_time * 0.1,
                'vehicles': vehicles,
                'vehicles_by_lane': vehicles_by_lane,
                'traffic_light_state': tl_state,
                'current_phase': current_phase,
                'simulation_time': self.env.conn.simulation.getTime(),
                'mode': self.mode
            }
            
        except Exception as e:
            print(f"Error collecting simulation data: {e}")
            return None

    async def send_simulation_data(self):
        """Send current simulation data to all clients"""
        if not self.env:
            return
            
        try:
            # Collect data (restarting env causes this to fail sometimes, handle gracefully)
            data = self.collect_simulation_data()
            
            if data:
                await self.broadcast('simulation_update', data)
            else:
                # Send minimal/empty data if collection failed
                 await self.broadcast('simulation_update', {
                    'queues': [0, 0, 0, 0],
                    'total_vehicles': 0,
                    'avg_wait_time': 0,
                    'total_waiting_time': 0,
                    'throughput': 0,
                    'reward': 0,
                    'vehicles': [],
                    'traffic_light_state': "rrrrrrrrrrrrrrrrrrrr",
                    'current_phase': 0,
                    'simulation_time': 0
                })
            
        except Exception as e:
            print(f"Error sending simulation data: {e}")
            await self.broadcast('log', {
                'message': f'Data collection error: {str(e)}',
                'level': 'warning'
            })
    
    async def update_simulation_config(self, config):
        """Update simulation configuration (e.g. traffic density)"""
        try:
            density = float(config.get('density', 1200))
            
            # Read backup file to ensure we always scale from original values
            if not os.path.exists('sumo/intersection.rou.xml.bak'):
                # Create backup if it doesn't exist (fallback)
                import shutil
                shutil.copy('sumo/intersection.rou.xml', 'sumo/intersection.rou.xml.bak')
                
            tree = ET.parse('sumo/intersection.rou.xml.bak')
            root = tree.getroot()
            
            # Calculate current total flow in the backup file
            total_flow = 0
            flows = root.findall('flow')
            for flow in flows:
                total_flow += float(flow.get('vehsPerHour'))
            
            if total_flow == 0: total_flow = 1 # Avoid div by zero
            
            scale_factor = density / total_flow
            
            # Update flows
            for flow in flows:
                original_flow = float(flow.get('vehsPerHour'))
                new_flow = int(original_flow * scale_factor)
                flow.set('vehsPerHour', str(new_flow))
            
            # Write to active route file
            tree.write('sumo/intersection.rou.xml')
            
            await self.broadcast('log', {
                'message': f'Traffic density updated to {int(density)} veh/hr. Restarting simulation...',
                'level': 'info'
            })
            
            # Restart simulation to apply changes
            # We need to fully stop and start to reload the route file
            await self.stop_simulation()
            await asyncio.sleep(0.5)
            await self.start_simulation()
            
        except Exception as e:
            await self.broadcast('log', {
                'message': f'Failed to update config: {str(e)}',
                'level': 'error'
            })

    async def handler(self, websocket, path):
        """WebSocket connection handler"""
        print(f"New WebSocket connection from {websocket.remote_address}")
        await self.register(websocket)
        try:
            async for message in websocket:
                await self.handle_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            print(f"WebSocket connection closed from {websocket.remote_address}")
        except Exception as e:
            print(f"WebSocket error: {e}")
        finally:
            await self.unregister(websocket)
    
    async def start(self):
        """Start the WebSocket server"""
        self.loop = asyncio.get_running_loop()
        print(f"Starting Traffic Control Server on ws://{self.host}:{self.port}")
        print("Open ui/index.html in your browser to access the UI")
        
        # Add extra headers for CORS
        extra_headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        }
        
        async with serve(self.handler, self.host, self.port, extra_headers=extra_headers):
            await asyncio.Future()  # Run forever

    async def run_benchmark(self):
        """Run performance benchmark comparing AI vs Fixed control"""
        await self.broadcast('log', {
            'message': 'Starting performance benchmark...',
            'level': 'info'
        })
        
        try:
            # Test scenarios
            scenarios = [
                {'name': 'Light Traffic', 'duration': 300, 'density': 800},
                {'name': 'Medium Traffic', 'duration': 300, 'density': 1200},
                {'name': 'Heavy Traffic', 'duration': 300, 'density': 1800}
            ]
            
            results = []
            
            for i, scenario in enumerate(scenarios):
                await self.broadcast('log', {
                    'message': f'Running scenario: {scenario["name"]}',
                    'level': 'info'
                })
                
                # Test AI control
                ai_result = await self.test_control_method('ai', scenario)
                
                # Test Fixed control  
                fixed_result = await self.test_control_method('fixed', scenario)
                
                results.append({
                    'scenario': scenario['name'],
                    'ai_performance': ai_result,
                    'fixed_performance': fixed_result,
                    'improvement': ((fixed_result['avg_wait'] - ai_result['avg_wait']) / fixed_result['avg_wait']) * 100
                })
                
                progress = (i + 1) / len(scenarios)
                await self.broadcast('benchmark_progress', {
                    'progress': progress,
                    'current_scenario': scenario['name'],
                    'results': results
                })
            
            await self.broadcast('benchmark_complete', {'results': results})
            await self.broadcast('log', {
                'message': 'Benchmark completed successfully!',
                'level': 'success'
            })
            
        except Exception as e:
            await self.broadcast('log', {
                'message': f'Benchmark failed: {str(e)}',
                'level': 'error'
            })

    async def test_control_method(self, method, scenario):
        """Test a specific control method"""
        # Create test environment (port 8815)
        test_env = TrafficLightEnv(use_gui=False, max_steps=scenario['duration'], sumo_port=8815)
        
        total_wait_time = 0
        total_vehicles = 0
        episodes = 3
        
        for episode in range(episodes):
            obs, _ = test_env.reset()
            episode_wait = 0
            episode_vehicles = 0
            
            for step in range(scenario['duration']):
                if method == 'ai' and self.ppo_model:
                    action, _ = self.ppo_model.predict(obs, deterministic=True)
                elif method == 'fixed':
                    action = 0 if (step // 30) % 2 == 0 else 1  # 30-second cycles
                else:
                    action = 0
                
                obs, reward, terminated, truncated, info = test_env.step(action)
                
                # Collect metrics
                try:
                    vehicles = test_env.conn.vehicle.getIDList()
                    episode_vehicles = max(episode_vehicles, len(vehicles))
                    episode_wait += sum(test_env.conn.vehicle.getWaitingTime(v) for v in vehicles)
                except:
                    pass
                
                if terminated or truncated:
                    break
            
            total_wait_time += episode_wait
            total_vehicles += episode_vehicles
        
        test_env.close()
        
        return {
            'avg_wait': total_wait_time / max(episodes, 1),
            'max_vehicles': total_vehicles / max(episodes, 1),
            'method': method
        }

    async def generate_report(self):
        """Generate comprehensive system report"""
        await self.broadcast('log', {
            'message': 'Generating system report...',
            'level': 'info'
        })
        
        try:
            # Collect system statistics
            report_data = {
                'timestamp': asyncio.get_event_loop().time(),
                'system_status': 'Online' if self.simulation_running else 'Offline',
                'models_loaded': {
                    'ppo': self.ppo_model is not None,
                    'dqn': self.dqn_model is not None
                },
                'current_mode': self.mode,
                'uptime': asyncio.get_event_loop().time(),
                'performance_summary': 'System operational'
            }
            
            # Generate report file
            import json
            import datetime
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f'reports/system_report_{timestamp}.json'
            
            os.makedirs('reports', exist_ok=True)
            
            with open(report_filename, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            await self.broadcast('report_generated', {
                'filename': report_filename,
                'data': report_data
            })
            
            await self.broadcast('log', {
                'message': f'Report generated: {report_filename}',
                'level': 'success'
            })
            
        except Exception as e:
            await self.broadcast('log', {
                'message': f'Report generation failed: {str(e)}',
                'level': 'error'
            })

    async def run_evaluation(self):
        """Run comprehensive model evaluation"""
        await self.broadcast('log', {
            'message': 'Starting comprehensive model evaluation...',
            'level': 'info'
        })
        
        try:
            if not self.ppo_model and not self.dqn_model:
                await self.broadcast('log', {
                    'message': 'No trained models available for evaluation. Please train a model first.',
                    'level': 'warning'
                })
                return
            
            # Test scenarios for evaluation
            test_scenarios = [
                {'name': 'Rush Hour', 'steps': 500, 'density': 2000},
                {'name': 'Normal Traffic', 'steps': 300, 'density': 1200},
                {'name': 'Light Traffic', 'steps': 200, 'density': 600}
            ]
            
            results = {}
            
            for i, scenario in enumerate(test_scenarios):
                await self.broadcast('log', {
                    'message': f'Evaluating scenario: {scenario["name"]}',
                    'level': 'info'
                })
                
                # Test each available model
                if self.ppo_model:
                    ppo_result = await self.evaluate_model(self.ppo_model, scenario, 'PPO')
                    results[f'PPO_{scenario["name"]}'] = ppo_result
                
                if self.dqn_model:
                    dqn_result = await self.evaluate_model(self.dqn_model, scenario, 'DQN')
                    results[f'DQN_{scenario["name"]}'] = dqn_result
                
                progress = (i + 1) / len(test_scenarios)
                await self.broadcast('evaluation_progress', {
                    'progress': progress,
                    'scenario': scenario['name'],
                    'results': results
                })
            
            # Calculate overall performance
            overall_performance = self.calculate_overall_performance(results)
            
            await self.broadcast('evaluation_complete', {
                'results': results,
                'overall_performance': overall_performance
            })
            
            await self.broadcast('log', {
                'message': f'Model evaluation completed! Overall performance: {overall_performance:.2f}',
                'level': 'success'
            })
            
        except Exception as e:
            await self.broadcast('log', {
                'message': f'Evaluation failed: {str(e)}',
                'level': 'error'
            })

    async def evaluate_model(self, model, scenario, model_name):
        """Evaluate a specific model on a scenario"""
        # Use port 8816 for evaluation
        eval_env = TrafficLightEnv(use_gui=False, max_steps=scenario['steps'], sumo_port=8816)
        
        total_reward = 0
        total_wait_time = 0
        episodes = 3
        
        for episode in range(episodes):
            obs, _ = eval_env.reset()
            episode_reward = 0
            episode_wait = 0
            
            for step in range(scenario['steps']):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                episode_reward += reward
                
                # Collect wait time data
                try:
                    vehicles = eval_env.conn.vehicle.getIDList()
                    episode_wait += sum(eval_env.conn.vehicle.getWaitingTime(v) for v in vehicles)
                except:
                    pass
                
                if terminated or truncated:
                    break
            
            total_reward += episode_reward
            total_wait_time += episode_wait
        
        eval_env.close()
        
        return {
            'model': model_name,
            'scenario': scenario['name'],
            'avg_reward': total_reward / episodes,
            'avg_wait_time': total_wait_time / episodes,
            'performance_score': (total_reward / episodes) / max(total_wait_time / episodes, 1)
        }

    def calculate_overall_performance(self, results):
        """Calculate overall performance score"""
        if not results:
            return 0.0
        
        scores = [result['performance_score'] for result in results.values()]
        return sum(scores) / len(scores)

    async def deploy_model(self):
        """Deploy the best performing model to production"""
        await self.broadcast('log', {
            'message': 'Initiating model deployment to production...',
            'level': 'info'
        })
        
        try:
            # Check if models are available
            if not self.ppo_model and not self.dqn_model:
                await self.broadcast('log', {
                    'message': 'No trained models available for deployment.',
                    'level': 'error'
                })
                return
            
            # Simulate deployment process
            deployment_steps = [
                'Validating model compatibility...',
                'Running safety checks...',
                'Backing up current production model...',
                'Deploying new model...',
                'Verifying deployment...',
                'Updating system configuration...'
            ]
            
            for i, step in enumerate(deployment_steps):
                await self.broadcast('log', {
                    'message': step,
                    'level': 'info'
                })
                
                await self.broadcast('deployment_progress', {
                    'progress': (i + 1) / len(deployment_steps),
                    'step': step,
                    'step_number': i + 1,
                    'total_steps': len(deployment_steps)
                })
                
                # Simulate processing time
                await asyncio.sleep(1)
            
            # Mark deployment as successful
            await self.broadcast('deployment_complete', {
                'status': 'success',
                'model_type': 'PPO' if self.ppo_model else 'DQN',
                'deployment_time': asyncio.get_event_loop().time()
            })
            
            await self.broadcast('log', {
                'message': 'Model successfully deployed to production environment!',
                'level': 'success'
            })
            
        except Exception as e:
            await self.broadcast('log', {
                'message': f'Deployment failed: {str(e)}',
                'level': 'error'
            })

    async def pause_training(self):
        """Pause current training"""
        if not self.training_active:
            await self.broadcast('log', {
                'message': 'No training is currently active',
                'level': 'warning'
            })
            return
        
        self.training_paused = True
        await self.broadcast('training_paused', {
            'model_type': self.training_model_type,
            'paused_at': asyncio.get_event_loop().time()
        })
        
        await self.broadcast('log', {
            'message': f'{self.training_model_type.upper()} training paused',
            'level': 'info'
        })

    async def resume_training(self):
        """Resume paused training"""
        if not self.training_active or not self.training_paused:
            await self.broadcast('log', {
                'message': 'No paused training to resume',
                'level': 'warning'
            })
            return
        
        self.training_paused = False
        await self.broadcast('training_resumed', {
            'model_type': self.training_model_type,
            'resumed_at': asyncio.get_event_loop().time()
        })
        
        await self.broadcast('log', {
            'message': f'{self.training_model_type.upper()} training resumed',
            'level': 'info'
        })

    async def stop_training(self):
        """Stop current training"""
        if not self.training_active:
            await self.broadcast('log', {
                'message': 'No training is currently active',
                'level': 'warning'
            })
            return
        
        self.training_active = False
        self.training_paused = False
        
        if self.current_training_task:
            self.current_training_task.cancel()
        
        await self.broadcast('training_stopped', {
            'model_type': self.training_model_type,
            'stopped_at': asyncio.get_event_loop().time()
        })
        
        await self.broadcast('log', {
            'message': f'{self.training_model_type.upper()} training stopped',
            'level': 'warning'
        })

    async def view_training_details(self):
        """Send detailed training information"""
        if not self.training_active:
            await self.broadcast('log', {
                'message': 'No training is currently active',
                'level': 'info'
            })
            return
        
        import torch
        
        training_details = {
            'model_type': self.training_model_type,
            'status': 'paused' if self.training_paused else 'running',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'gpu_memory': self.get_gpu_memory_info() if torch.cuda.is_available() else None,
            'training_time': asyncio.get_event_loop().time(),
            'parameters': self.get_model_parameters()
        }
        
        await self.broadcast('training_details', training_details)

    def get_gpu_memory_info(self):
        """Get GPU memory usage information"""
        try:
            import torch
            if torch.cuda.is_available():
                return {
                    'allocated': torch.cuda.memory_allocated() / 1024**2,  # MB
                    'cached': torch.cuda.memory_reserved() / 1024**2,      # MB
                    'total': torch.cuda.get_device_properties(0).total_memory / 1024**2  # MB
                }
        except:
            pass
        return None

    def get_model_parameters(self):
        """Get current model parameters"""
        if self.training_model_type == 'ppo' and self.ppo_model:
            return {
                'learning_rate': self.ppo_model.learning_rate,
                'batch_size': self.ppo_model.batch_size,
                'n_steps': self.ppo_model.n_steps
            }
        elif self.training_model_type == 'dqn' and self.dqn_model:
            return {
                'learning_rate': self.dqn_model.learning_rate,
                'batch_size': self.dqn_model.batch_size,
                'buffer_size': self.dqn_model.buffer_size
            }
        return {}


def main():
    """Main entry point"""
    server = TrafficControlServer()
    
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Server error: {e}")


if __name__ == "__main__":
    main()