# AI Traffic Light Control System - Web UI

A modern, responsive web interface for monitoring and controlling the AI-powered adaptive traffic light system.

## Features

### 🎛️ Control Panel
- **System Modes**: Switch between Manual, AI Control, and Fixed Time modes
- **Simulation Control**: Start, pause, stop, and reset traffic simulations
- **Manual Override**: Direct control of traffic light phases

### 🚦 Real-time Visualization
- **Interactive Intersection**: Visual representation of the traffic intersection
- **Traffic Lights**: Real-time traffic light status with color-coded indicators
- **Vehicle Queues**: Live display of waiting vehicles in each direction

### 📊 Performance Monitoring
- **Live Statistics**: Total vehicles, average wait time, throughput, and AI reward scores
- **Performance Charts**: Real-time graphs showing system performance over time
- **Historical Data**: Track performance trends and optimization results

### 🤖 AI Model Management
- **Model Loading**: Load pre-trained PPO and DQN models
- **Training Control**: Start training sessions with progress monitoring
- **Model Status**: Real-time status of AI models and their performance

### 📝 System Logs
- **Real-time Logging**: Live system events and status updates
- **Log Levels**: Info, warning, error, and success message categorization
- **Export Functionality**: Download logs for analysis and debugging

## Quick Start

### Option 1: Using the Launcher (Recommended)
```bash
python launch_ui.py
```

### Option 2: Manual Setup
1. Start the backend server:
   ```bash
   python ui_server.py
   ```

2. Open `ui/index.html` in your web browser

## System Requirements

- **Python 3.7+** with required packages
- **Modern Web Browser** (Chrome, Firefox, Safari, Edge)
- **SUMO Traffic Simulator** installed and configured
- **WebSocket Support** for real-time communication

## Architecture

```
┌─────────────────┐    WebSocket    ┌──────────────────┐
│   Web Browser   │ ←──────────────→ │  Python Backend  │
│   (Frontend)    │   ws://8765     │   (ui_server.py) │
└─────────────────┘                 └──────────────────┘
                                             │
                                             ▼
                                    ┌──────────────────┐
                                    │  SUMO Simulator  │
                                    │ (Traffic System) │
                                    └──────────────────┘
```

## API Reference

### WebSocket Messages

#### Client → Server
- `set_mode`: Change control mode (manual/ai/fixed)
- `start_simulation`: Start traffic simulation
- `pause_simulation`: Pause current simulation
- `stop_simulation`: Stop and cleanup simulation
- `reset_simulation`: Reset simulation state
- `set_phase`: Set traffic light phase (manual mode)
- `load_model`: Load AI model (ppo/dqn)
- `train_model`: Start model training
- `get_simulation_data`: Request current simulation data

#### Server → Client
- `simulation_update`: Real-time simulation data
- `phase_change`: Traffic light phase changes
- `training_progress`: Model training progress
- `model_status`: AI model loading status
- `log`: System log messages

## Customization

### Styling
Edit `ui/styles.css` to customize the appearance:
- Color schemes and themes
- Layout and spacing
- Animation effects
- Responsive breakpoints

### Functionality
Modify `ui/script.js` to add features:
- New control modes
- Additional statistics
- Custom visualizations
- Enhanced interactions

### Backend Integration
Extend `ui_server.py` for advanced features:
- Database integration
- Advanced AI models
- External API connections
- Performance analytics

## Troubleshooting

### Common Issues

1. **WebSocket Connection Failed**
   - Ensure `ui_server.py` is running
   - Check firewall settings
   - Verify port 8765 is available

2. **Simulation Not Starting**
   - Confirm SUMO is installed and in PATH
   - Check SUMO configuration files
   - Verify Python dependencies

3. **UI Not Loading**
   - Use a modern web browser
   - Check browser console for errors
   - Ensure all UI files are present

4. **Performance Issues**
   - Reduce simulation speed
   - Close unnecessary browser tabs
   - Check system resources

### Debug Mode
Enable debug logging by modifying the server:
```python
# In ui_server.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Browser Compatibility

| Browser | Version | Status |
|---------|---------|--------|
| Chrome  | 80+     | ✅ Full Support |
| Firefox | 75+     | ✅ Full Support |
| Safari  | 13+     | ✅ Full Support |
| Edge    | 80+     | ✅ Full Support |

## Performance Tips

1. **Optimize Update Frequency**: Adjust simulation update intervals
2. **Limit Chart Data**: Keep chart history reasonable (20-50 points)
3. **Use Efficient Rendering**: Minimize DOM updates
4. **Close Unused Tabs**: Free up browser resources

## Contributing

To contribute to the UI:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is part of the AI Traffic Light Control System and follows the same license terms.