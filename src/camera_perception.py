"""
Camera-based vehicle detection using YOLOv8 for traffic light control
This module demonstrates how real-world camera input could replace TraCI's ground truth
"""
import cv2
import numpy as np
from ultralytics import YOLO
import traci
from typing import List, Tuple, Dict
import os


class CameraPerceptionSystem:
    """
    Camera-based perception system using YOLOv8 for vehicle detection
    Simulates real-world camera input by capturing SUMO screenshots
    """
    
    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.5):
        """
        Initialize camera perception system
        
        Args:
            model_path: Path to YOLOv8 model (downloads automatically if not found)
            confidence_threshold: Minimum confidence for vehicle detection
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Vehicle class IDs in COCO dataset (YOLOv8 default)
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        
        # Define detection zones for each lane (normalized coordinates)
        self.detection_zones = {
            'west': {'x1': 0.1, 'y1': 0.4, 'x2': 0.45, 'y2': 0.6},    # West approach
            'north': {'x1': 0.4, 'y1': 0.1, 'x2': 0.6, 'y2': 0.45},   # North approach  
            'east': {'x1': 0.55, 'y1': 0.4, 'x2': 0.9, 'y2': 0.6},    # East approach
            'south': {'x1': 0.4, 'y1': 0.55, 'x2': 0.6, 'y2': 0.9}    # South approach
        }
        
    def capture_sumo_screenshot(self, width: int = 800, height: int = 600) -> np.ndarray:
        """
        Capture screenshot from SUMO GUI
        
        Args:
            width: Screenshot width
            height: Screenshot height
            
        Returns:
            Screenshot as numpy array
        """
        try:
            # Take screenshot using TraCI
            traci.gui.screenshot("View #0", "temp_screenshot.png", width, height)
            
            # Read the screenshot
            image = cv2.imread("temp_screenshot.png")
            
            # Clean up temporary file
            if os.path.exists("temp_screenshot.png"):
                os.remove("temp_screenshot.png")
                
            return image
            
        except Exception as e:
            print(f"Error capturing screenshot: {e}")
            # Return dummy image if screenshot fails
            return np.zeros((height, width, 3), dtype=np.uint8)
    
    def detect_vehicles(self, image: np.ndarray) -> List[Dict]:
        """
        Detect vehicles in the image using YOLOv8
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detection dictionaries with bbox and confidence
        """
        if image is None or image.size == 0:
            return []
        
        try:
            # Run YOLOv8 inference
            results = self.model(image, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get class ID and confidence
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        # Filter for vehicles with sufficient confidence
                        if class_id in self.vehicle_classes and confidence >= self.confidence_threshold:
                            # Get bounding box coordinates (normalized)
                            x1, y1, x2, y2 = box.xyxyn[0].tolist()
                            
                            detections.append({
                                'bbox': [x1, y1, x2, y2],
                                'confidence': confidence,
                                'class_id': class_id,
                                'center': [(x1 + x2) / 2, (y1 + y2) / 2]
                            })
            
            return detections
            
        except Exception as e:
            print(f"Error in vehicle detection: {e}")
            return []
    
    def count_vehicles_by_lane(self, detections: List[Dict]) -> Dict[str, int]:
        """
        Count vehicles in each lane based on detection zones
        
        Args:
            detections: List of vehicle detections
            
        Returns:
            Dictionary with vehicle counts per lane
        """
        lane_counts = {'west': 0, 'north': 0, 'east': 0, 'south': 0}
        
        for detection in detections:
            center_x, center_y = detection['center']
            
            # Check which detection zone the vehicle center falls into
            for lane, zone in self.detection_zones.items():
                if (zone['x1'] <= center_x <= zone['x2'] and 
                    zone['y1'] <= center_y <= zone['y2']):
                    lane_counts[lane] += 1
                    break  # Vehicle can only be in one lane
        
        return lane_counts
    
    def get_camera_observation(self) -> np.ndarray:
        """
        Get traffic observation using camera perception
        
        Returns:
            Observation array [west_count, north_count, east_count, south_count]
        """
        try:
            # Capture screenshot from SUMO
            image = self.capture_sumo_screenshot()
            
            # Detect vehicles
            detections = self.detect_vehicles(image)
            
            # Count vehicles by lane
            lane_counts = self.count_vehicles_by_lane(detections)
            
            # Convert to observation format
            observation = np.array([
                lane_counts['west'],
                lane_counts['north'], 
                lane_counts['east'],
                lane_counts['south']
            ], dtype=np.float32)
            
            return observation
            
        except Exception as e:
            print(f"Error getting camera observation: {e}")
            # Return zeros if camera perception fails
            return np.zeros(4, dtype=np.float32)
    
    def visualize_detections(self, image: np.ndarray, detections: List[Dict], 
                           save_path: str = None) -> np.ndarray:
        """
        Visualize vehicle detections on image
        
        Args:
            image: Input image
            detections: Vehicle detections
            save_path: Optional path to save visualization
            
        Returns:
            Image with detection visualizations
        """
        if image is None or image.size == 0:
            return image
        
        vis_image = image.copy()
        height, width = vis_image.shape[:2]
        
        # Draw detection zones
        colors = {'west': (255, 0, 0), 'north': (0, 255, 0), 
                 'east': (0, 0, 255), 'south': (255, 255, 0)}
        
        for lane, zone in self.detection_zones.items():
            x1 = int(zone['x1'] * width)
            y1 = int(zone['y1'] * height)
            x2 = int(zone['x2'] * width)
            y2 = int(zone['y2'] * height)
            
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), colors[lane], 2)
            cv2.putText(vis_image, lane.upper(), (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[lane], 2)
        
        # Draw vehicle detections
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            x1, y1, x2, y2 = int(x1*width), int(y1*height), int(x2*width), int(y2*height)
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
            # Draw confidence
            conf_text = f"{detection['confidence']:.2f}"
            cv2.putText(vis_image, conf_text, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Count vehicles by lane
        lane_counts = self.count_vehicles_by_lane(detections)
        
        # Display counts
        y_offset = 30
        for lane, count in lane_counts.items():
            text = f"{lane.upper()}: {count} vehicles"
            cv2.putText(vis_image, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30
        
        if save_path:
            cv2.imwrite(save_path, vis_image)
        
        return vis_image


class CameraBasedTrafficEnv:
    """
    Traffic environment using camera-based perception instead of TraCI ground truth
    """
    
    def __init__(self, base_env, use_camera: bool = True):
        """
        Initialize camera-based environment wrapper
        
        Args:
            base_env: Base TrafficLightEnv instance
            use_camera: Whether to use camera perception or fall back to TraCI
        """
        self.base_env = base_env
        self.use_camera = use_camera
        
        if use_camera:
            self.camera_system = CameraPerceptionSystem()
        else:
            self.camera_system = None
    
    def get_observation(self):
        """Get observation using camera perception or TraCI fallback"""
        if self.use_camera and self.camera_system:
            try:
                return self.camera_system.get_camera_observation()
            except Exception as e:
                print(f"Camera perception failed, falling back to TraCI: {e}")
                return self.base_env._get_observation()
        else:
            return self.base_env._get_observation()
    
    def step(self, action):
        """Environment step with camera-based observation"""
        # Use base environment for action execution
        obs, reward, terminated, truncated, info = self.base_env.step(action)
        
        # Replace observation with camera-based one
        camera_obs = self.get_observation()
        
        # Add comparison info
        if self.use_camera:
            info['traci_obs'] = obs
            info['camera_obs'] = camera_obs
            info['obs_difference'] = np.abs(obs - camera_obs).sum()
        
        return camera_obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        """Reset environment"""
        obs, info = self.base_env.reset(**kwargs)
        
        # Replace with camera observation
        camera_obs = self.get_observation()
        
        if self.use_camera:
            info['traci_obs'] = obs
            info['camera_obs'] = camera_obs
        
        return camera_obs, info


def test_camera_perception():
    """Test camera perception system"""
    print("Testing Camera Perception System...")
    
    try:
        from traffic_env import TrafficLightEnv
        
        # Create environment with GUI for screenshot capture
        env = TrafficLightEnv(use_gui=True, max_steps=100)
        camera_env = CameraBasedTrafficEnv(env, use_camera=True)
        
        # Reset environment
        obs, info = camera_env.reset()
        print(f"Initial camera observation: {obs}")
        
        # Run a few steps
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = camera_env.step(action)
            
            print(f"Step {i+1}:")
            print(f"  Camera obs: {obs}")
            if 'traci_obs' in info:
                print(f"  TraCI obs: {info['traci_obs']}")
                print(f"  Difference: {info['obs_difference']:.2f}")
            
            if terminated or truncated:
                break
        
        env.close()
        print("Camera perception test completed!")
        
    except Exception as e:
        print(f"Error testing camera perception: {e}")
        print("Make sure SUMO GUI is available and YOLOv8 is installed")


if __name__ == "__main__":
    test_camera_perception()