
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
    print("Installing websockets...")
    os.system("pip install websockets")
    import websockets
    from websockets.server import serve

from traffic_env import TrafficLightEnv, FixedTimeController
from stable_baselines3 import PPO, DQN
import numpy as np


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
                await self.train_model(payload['model_type'])
                
            elif msg_type == 'update_config':
                await self.update_simulation_config(payload)

            elif msg_type == 'get_simulation_data':
                await self.send_simulation_data()
                
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
            if not self.env:
                self.env = TrafficLightEnv(use_gui=False, max_steps=1000)
                self.env.reset()
                
            self.simulation_running = True
            
            await self.broadcast('log', {
                'message': 'Simulation started',
                'level': 'success'
            })
            
            # Start simulation loop
            asyncio.create_task(self.simulation_loop())
            
        except Exception as e:
            await self.broadcast('log', {
                'message': f'Failed to start simulation: {str(e)}',
                'level': 'error'
            })
    
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
                import traci
                if phase == 0:
                    traci.trafficlight.setPhase(self.env.junction_id, 0)  # North-South green
                else:
                    traci.trafficlight.setPhase(self.env.junction_id, 2)  # East-West green
                    
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
        """Train a model (simplified version)"""
        await self.broadcast('log', {
            'message': f'Training {model_type.upper()} model... This may take a while.',
            'level': 'info'
        })
        
        # This is a placeholder - actual training should be done separately
        # For demo purposes, we'll simulate training progress
        for i in range(11):
            await asyncio.sleep(1)
            progress = i / 10
            await self.broadcast('training_progress', {
                'progress': progress,
                'status': f'Training step {i * 10}%'
            })
        
        await self.broadcast('log', {
            'message': f'{model_type.upper()} training completed (demo)',
            'level': 'success'
        })
    
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
                
                # Send update to clients every few steps to avoid overwhelming
                if step_count % 2 == 0:  # Send every 2 steps
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
    
    async def send_simulation_data(self):
        """Send current simulation data to all clients"""
        if not self.env:
            return
            
        try:
            import traci
            
            # Get real SUMO data
            obs = self.env._get_observation()
            info = self.env._get_info()
            
            # Get vehicle positions and data
            vehicle_ids = traci.vehicle.getIDList()
            vehicles = []
            
            for veh_id in vehicle_ids:
                try:
                    pos = traci.vehicle.getPosition(veh_id)
                    lane_id = traci.vehicle.getLaneID(veh_id)
                    speed = traci.vehicle.getSpeed(veh_id)
                    waiting_time = traci.vehicle.getWaitingTime(veh_id)
                    
                    vehicles.append({
                        'id': veh_id,
                        'position': pos,
                        'lane': lane_id,
                        'speed': speed,
                        'waiting_time': waiting_time
                    })
                except:
                    continue
            
            # Get traffic light state
            try:
                tl_state = traci.trafficlight.getRedYellowGreenState(self.env.junction_id)
                current_phase = traci.trafficlight.getPhase(self.env.junction_id)
            except:
                tl_state = "rrrrrrrrrrrrrrrrrrrr"
                current_phase = 0
            
            # Calculate real metrics
            total_waiting_time = sum(v['waiting_time'] for v in vehicles)
            avg_wait_time = total_waiting_time / max(len(vehicles), 1)
            
            # Get queue lengths per direction
            queue_lengths = obs.tolist() if len(obs) >= 4 else [0, 0, 0, 0]
            
            data = {
                'queues': queue_lengths,
                'total_vehicles': len(vehicles),
                'avg_wait_time': avg_wait_time,
                'total_waiting_time': total_waiting_time,
                'throughput': info.get('vehicle_count', len(vehicles)),
                'reward': getattr(self.env, 'total_waiting_time', 0) * -0.1,
                'vehicles': vehicles,
                'traffic_light_state': tl_state,
                'current_phase': current_phase,
                'simulation_time': traci.simulation.getTime() if 'traci' in globals() else 0
            }
            
            await self.broadcast('simulation_update', data)
            
        except Exception as e:
            print(f"Error sending simulation data: {e}")
            # Send minimal data if SUMO connection fails
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
        await self.register(websocket)
        try:
            async for message in websocket:
                await self.handle_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister(websocket)
    
    async def start(self):
        """Start the WebSocket server"""
        print(f"Starting Traffic Control Server on ws://{self.host}:{self.port}")
        print("Open ui/index.html in your browser to access the UI")
        
        async with serve(self.handler, self.host, self.port):
            await asyncio.Future()  # Run forever


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