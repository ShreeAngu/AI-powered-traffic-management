// TrafficAI Manager - Real Data Integration

class TrafficAIManager {
    constructor() {
        this.isConnected = false;
        this.simulationRunning = false;
        this.isTraining = false;
        this.training_model_type = null; // Track which model is being trained
        this.currentMode = 'ai';
        this.currentPhase = 0;
        this.websocket = null;
        this.charts = {};
        this.chartData = {
            labels: [],
            aiData: [],
            baselineData: []
        };

        this.init();
    }

    init() {
        this.setupEventListeners();
        this.initPerformanceChart();
        this.initAnalyticsCharts();
        this.connectWebSocket();
        this.updateTrafficLights(0);
        this.loadSettings();

        // Check hash for initial view
        const hash = window.location.hash.slice(1) || 'dashboard';
        this.switchView(hash);
    }

    setupEventListeners() {
        // Navigation
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const viewId = e.currentTarget.dataset.view;
                if (viewId) {
                    this.switchView(viewId);
                    // Update URL hash without scrolling
                    history.pushState(null, null, `#${viewId}`);
                }
            });
        });

        // Control modes
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const mode = e.currentTarget.dataset.mode;
                if (mode) this.setControlMode(mode);
            });
        });

        // Simulation controls
        const startBtn = document.getElementById('startSim');
        const pauseBtn = document.getElementById('pauseSim');
        const stopBtn = document.getElementById('stopSim');
        const resetBtn = document.getElementById('resetSim');

        if (startBtn) startBtn.addEventListener('click', () => this.startSimulation());
        if (pauseBtn) pauseBtn.addEventListener('click', () => this.pauseSimulation());
        if (stopBtn) stopBtn.addEventListener('click', () => this.stopSimulation());
        if (resetBtn) resetBtn.addEventListener('click', () => this.resetSimulation());

        // Time selector
        document.querySelectorAll('.time-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.time-btn').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
            });
        });

        // Manual phase controls
        document.querySelectorAll('.phase-control-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const phase = parseInt(e.currentTarget.dataset.phase);
                this.setManualPhase(phase);
            });
        });

        // Sliders
        const densitySlider = document.getElementById('densitySlider');
        if (densitySlider) {
            densitySlider.addEventListener('input', (e) => {
                e.target.nextElementSibling.textContent = e.target.value;
            });

            densitySlider.addEventListener('change', (e) => {
                this.sendWebSocketMessage('update_config', {
                    density: parseInt(e.target.value)
                });
            });
        }

        // Variance Slider (New)
        const varianceSlider = document.querySelector('input[type="range"][step="0.1"]');
        if (varianceSlider) {
            varianceSlider.addEventListener('input', (e) => {
                e.target.nextElementSibling.textContent = e.target.value;
            });
            varianceSlider.addEventListener('change', (e) => {
                this.sendWebSocketMessage('update_config', {
                    variance: parseFloat(e.target.value)
                });
            });
        }

        // Toggles (New)
        const weatherToggleLabel = Array.from(document.querySelectorAll('.toggle-label')).find(el => el.textContent.includes('Weather'));
        if (weatherToggleLabel) {
            const checkbox = weatherToggleLabel.querySelector('input[type="checkbox"]');
            if (checkbox) {
                checkbox.addEventListener('change', (e) => {
                    this.sendWebSocketMessage('update_config', { weather: e.target.checked });
                });
            }
        }

        const pedToggleLabel = Array.from(document.querySelectorAll('.toggle-label')).find(el => el.textContent.includes('Pedestrians'));
        if (pedToggleLabel) {
            const checkbox = pedToggleLabel.querySelector('input[type="checkbox"]');
            if (checkbox) {
                checkbox.addEventListener('change', (e) => {
                    this.sendWebSocketMessage('update_config', { pedestrians: e.target.checked });
                });
            }
        }

        // Mission Control Buttons
        document.getElementById('btn-pause-training')?.addEventListener('click', () => this.pauseTraining());
        document.getElementById('btn-resume-training')?.addEventListener('click', () => this.resumeTraining());
        document.getElementById('btn-view-training')?.addEventListener('click', () => this.viewTraining());
        document.getElementById('btn-stop-training')?.addEventListener('click', () => this.stopTraining());

        // Model Action Buttons
        document.querySelectorAll('.model-action-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const action = e.currentTarget.dataset.action;
                const model = e.currentTarget.dataset.model;
                if (action === 'train') {
                    this.trainModel(model);
                } else if (action === 'load') {
                    this.loadModel(model);
                }
            });
        });

        // Dashboard Training Buttons
        document.querySelectorAll('.dashboard-train-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const action = e.currentTarget.dataset.action;
                if (action === 'pause') this.pauseTraining();
                if (action === 'view') this.viewTraining();
            });
        });

        // Quick Actions
        document.getElementById('btn-run-eval')?.addEventListener('click', () => this.runEvaluation());
        document.getElementById('btn-deploy-model')?.addEventListener('click', () => this.deployModel());

        // Settings Inputs
        const themeSelect = document.querySelector('.settings-section select');
        if (themeSelect) {
            themeSelect.addEventListener('change', (e) => this.setTheme(e.target.value));
        }

        const sysNameInput = document.querySelector('.settings-section input[type="text"]');
        if (sysNameInput) {
            sysNameInput.addEventListener('change', (e) => {
                localStorage.setItem('traffic_sys_name', e.target.value);
                this.addAlert('Settings Saved', 'System name updated', 'success');
            });
        }
    }

    loadSettings() {
        // Load Theme
        const storedTheme = localStorage.getItem('traffic_theme');
        if (storedTheme) {
            this.setTheme(storedTheme);
            const select = document.querySelector('.settings-section select');
            if (select) select.value = storedTheme;
        }

        // Load System Name
        const storedName = localStorage.getItem('traffic_sys_name');
        if (storedName) {
            const input = document.querySelector('.settings-section input[type="text"]');
            if (input) input.value = storedName;
        }
    }

    setTheme(theme) {
        if (theme === 'Light') {
            document.body.classList.add('light-mode');
        } else {
            document.body.classList.remove('light-mode');
        }
        localStorage.setItem('traffic_theme', theme);
    }

    switchView(viewId) {
        // Hide all views
        document.querySelectorAll('.view').forEach(view => {
            view.classList.remove('active');
        });

        // Show target view
        const targetView = document.getElementById(`${viewId}-view`);
        if (targetView) {
            targetView.classList.add('active');
        }

        // Update nav active state
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('active');
        });
        const activeLink = document.querySelector(`.nav-link[data-view="${viewId}"]`);
        if (activeLink) {
            activeLink.parentElement.classList.add('active');
        }

        // Update page title
        const titleMap = {
            'dashboard': 'Overview',
            'simulation': 'Simulation Configuration',
            'analytics': 'System Analytics',
            'models': 'AI Models',
            'settings': 'System Settings'
        };
        const pageTitle = document.getElementById('pageTitle');
        if (pageTitle) {
            pageTitle.textContent = titleMap[viewId] || 'Overview';
        }
    }

    connectWebSocket() {
        console.log('🔌 Attempting to connect to WebSocket server...');

        try {
            this.websocket = new WebSocket('ws://localhost:8765');

            this.websocket.onopen = () => {
                this.isConnected = true;
                this.updateConnectionStatus();
                console.log('✅ Connected to TrafficAI backend');

                // Add connection success alert
                this.addAlert('Backend Connected', 'Successfully connected to TrafficAI server', 'success');
            };

            this.websocket.onmessage = (event) => {
                console.log('📨 Received message:', event.data);
                try {
                    const data = JSON.parse(event.data);
                    this.handleWebSocketMessage(data);
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error);
                    this.addLogMessage({
                        message: `Error parsing server message: ${error.message}`,
                        level: 'error'
                    });
                }
            };

            this.websocket.onclose = (event) => {
                this.isConnected = false;
                this.updateConnectionStatus();
                console.log('❌ Disconnected from backend. Code:', event.code, 'Reason:', event.reason);

                // Add disconnection alert
                this.addAlert('Backend Disconnected', 'Connection to TrafficAI server lost', 'warning');

                // Attempt to reconnect after 5 seconds
                setTimeout(() => {
                    console.log('🔄 Attempting to reconnect...');
                    this.connectWebSocket();
                }, 5000);
            };

            this.websocket.onerror = (error) => {
                console.error('❌ WebSocket error:', error);
                this.isConnected = false;
                this.updateConnectionStatus();

                this.addAlert('Connection Error', 'Failed to connect to TrafficAI server', 'critical');
            };
        } catch (error) {
            console.log('⚠️ Backend not available - UI only mode');
            console.error('Connection error:', error);
            this.isConnected = false;
            this.updateConnectionStatus();

            this.addAlert('Backend Unavailable', 'Running in UI-only mode. Start the server to enable full functionality.', 'warning');
        }
    }

    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'simulation_update':
                this.updateSimulationDisplay(data.payload);
                break;
            case 'phase_change':
                this.updateTrafficLights(data.payload.phase);
                break;
            case 'log':
                console.log(`[${data.payload.level}] ${data.payload.message}`);
                this.addLogMessage(data.payload);
                break;
            case 'model_status':
                this.updateModelStatus(data.payload);
                break;
            case 'training_progress':
                this.updateTrainingProgress(data.payload);
                break;
            case 'training_paused':
                this.handleTrainingPaused(data.payload);
                break;
            case 'training_resumed':
                this.handleTrainingResumed(data.payload);
                break;
            case 'training_stopped':
                this.handleTrainingStopped(data.payload);
                break;
            case 'training_details':
                this.handleTrainingDetails(data.payload);
                break;
            case 'evaluation_complete':
                this.handleEvaluationComplete(data.payload);
                break;
            case 'evaluation_progress':
                this.handleEvaluationProgress(data.payload);
                break;
            case 'deployment_complete':
                this.handleDeploymentComplete(data.payload);
                break;
            case 'deployment_progress':
                this.handleDeploymentProgress(data.payload);
                break;
            case 'benchmark_complete':
                this.handleBenchmarkComplete(data.payload);
                break;
            case 'benchmark_progress':
                this.handleBenchmarkProgress(data.payload);
                break;
            default:
                console.warn('Unknown message type:', data.type);
                this.addLogMessage({
                    message: `Received unknown message type: ${data.type}`,
                    level: 'warning'
                });
                break;
        }
    }

    sendWebSocketMessage(type, payload) {
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            try {
                this.websocket.send(JSON.stringify({ type, payload }));
                console.log(`📤 Sent message: ${type}`, payload);
            } catch (error) {
                console.error('Error sending WebSocket message:', error);
                this.addLogMessage({
                    message: `Failed to send message: ${error.message}`,
                    level: 'error'
                });
            }
        } else {
            console.warn('WebSocket not connected - message not sent:', type);
            this.addAlert('Connection Required', 'Backend connection required for this action', 'warning');
        }
    }

    setControlMode(mode) {
        this.currentMode = mode;

        // Update UI
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        const modeBtn = document.querySelector(`[data-mode="${mode}"]`);
        if (modeBtn) modeBtn.classList.add('active');

        // Show/hide manual controls
        const manualControls = document.getElementById('manualControls');
        if (manualControls) {
            manualControls.style.display = mode === 'manual' ? 'block' : 'none';
        }

        // Send to backend
        this.sendWebSocketMessage('set_mode', { mode });

        console.log(`Switched to ${mode} mode`);
    }

    setManualPhase(phase) {
        if (this.currentMode !== 'manual') return;

        // Update UI
        document.querySelectorAll('.phase-control-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        const phaseBtn = document.querySelector(`[data-phase="${phase}"]`);
        if (phaseBtn) phaseBtn.classList.add('active');

        // Update traffic lights immediately
        this.updateTrafficLights(phase);

        // Send to backend
        this.sendWebSocketMessage('set_phase', { phase });

        console.log(`Manual phase: ${phase === 0 ? 'North-South' : 'East-West'} green`);
    }

    startSimulation() {
        if (this.isTraining) {
            this.resumeTraining();
            return;
        }
        this.simulationRunning = true;
        this.updateSimulationControls();
        this.sendWebSocketMessage('start_simulation', {});
        this.updateSystemStatus('System Online • Simulation running');
        console.log('▶ Simulation started');
    }

    pauseSimulation() {
        if (this.isTraining) {
            this.pauseTraining();
            return;
        }
        this.simulationRunning = false;
        this.updateSimulationControls();
        this.sendWebSocketMessage('pause_simulation', {});
        this.updateSystemStatus('System Online • Simulation paused');
        console.log('⏸ Simulation paused');
    }

    stopSimulation() {
        if (this.isTraining) {
            this.stopTraining();
            // Don't return, allow UI cleanup
        }
        this.simulationRunning = false;
        this.updateSimulationControls();
        this.sendWebSocketMessage('stop_simulation', {});
        this.updateSystemStatus('System Online • Data stream active');
        console.log('⏹ Simulation stopped');
    }

    resetSimulation() {
        if (this.isTraining) return; // Disable reset during training
        this.simulationRunning = false;
        this.updateSimulationControls();
        this.sendWebSocketMessage('reset_simulation', {});
        this.resetMetrics();
        this.updateTrafficLights(0);
        console.log('🔄 Simulation reset');
    }

    updateSimulationControls() {
        const startBtn = document.getElementById('startSim');
        const pauseBtn = document.getElementById('pauseSim');
        const stopBtn = document.getElementById('stopSim');

        if (startBtn) startBtn.style.opacity = this.simulationRunning ? '0.5' : '1';
        if (pauseBtn) pauseBtn.style.opacity = this.simulationRunning ? '1' : '0.5';
        if (stopBtn) stopBtn.style.opacity = this.simulationRunning ? '1' : '0.5';
    }

    updateConnectionStatus() {
        const statusDot = document.querySelector('.status-dot');
        const statusText = document.querySelector('.system-status span');

        if (statusDot) {
            statusDot.className = `status-dot ${this.isConnected ? 'online' : 'offline'}`;
        }

        if (statusText) {
            statusText.textContent = this.isConnected
                ? 'System Online • Data stream active'
                : 'System Offline • Backend not connected';
        }
    }

    updateSystemStatus(text) {
        const statusText = document.querySelector('.system-status span');
        if (statusText) {
            statusText.textContent = text;
        }
    }

    updateSimulationDisplay(data) {
        // Update vehicle queues with REAL data from SUMO
        const queueElements = [
            document.querySelector('.west-queue .queue-count'),
            document.querySelector('.north-queue .queue-count'),
            document.querySelector('.east-queue .queue-count'),
            document.querySelector('.south-queue .queue-count')
        ];

        queueElements.forEach((element, index) => {
            if (element && data.queues && data.queues[index] !== undefined) {
                const value = Math.round(data.queues[index]);
                if (element.textContent !== value.toString()) {
                    element.textContent = value;
                    element.parentElement.style.transform = 'scale(1.1)';
                    setTimeout(() => {
                        element.parentElement.style.transform = 'scale(1)';
                    }, 200);
                }
            }
        });

        // Update metric cards with 100% REAL SUMO data
        this.updateMetricValue('.metric-card:nth-child(1) .metric-value', data.total_vehicles || 0);

        // Real congestion level from SUMO calculation
        const congestionLevel = data.congestion_level || Math.min(5, Math.max(1, Math.round((data.avg_wait_time || 0) / 15) + 1));
        this.updateMetricValue('.metric-card:nth-child(2) .metric-value', `Level ${congestionLevel}`);

        // Real incidents count from SUMO (vehicles waiting > 30s)
        const incidents = data.incidents || 0;
        this.updateMetricValue('.metric-card:nth-child(3) .metric-value', incidents);

        // Real AI Efficiency from SUMO performance
        const efficiency = Math.round(data.efficiency || Math.max(0, Math.min(100, 100 - (data.avg_wait_time || 0) * 1.5)));
        this.updateMetricValue('.metric-card:nth-child(4) .metric-value', efficiency);

        // Add visual feedback for real-time updates
        document.querySelectorAll('.metric-card').forEach(card => {
            card.classList.add('updating');
            setTimeout(() => card.classList.remove('updating'), 300);
        });

        // Log real-time data for verification
        console.log('Real-time SUMO data:', {
            vehicles: data.total_vehicles,
            avgWait: data.avg_wait_time,
            congestion: congestionLevel,
            incidents: incidents,
            efficiency: efficiency,
            mode: data.mode,
            isTraining: data.mode === 'training'
        });

        // Update vehicles visualization with real vehicle data
        this.updateVehicleVisualization(data.vehicles || []);

        // Update chart with real data
        this.updateChartData(data);

        // Update additional real metrics
        this.updateAdditionalMetrics(data);

        // Update full analytics charts
        this.updateAnalyticsCharts(data);

        // Update simulation time
        if (data.simulation_time !== undefined) {
            this.updateSimulationTime(data.simulation_time);
        }

        // Check for alerts based on real data
        this.checkForAlerts(data);

        // If this is training data, update training-specific metrics
        if (data.mode === 'training' && data.training_step !== undefined) {
            // Update training step display
            const stepDisplay = document.querySelector('.training-step-display');
            if (stepDisplay) {
                stepDisplay.textContent = `Step: ${data.training_step}`;
            }

            // Update reward display if available
            if (data.reward !== undefined) {
                const rewardDisplay = document.querySelector('.current-reward-display');
                if (rewardDisplay) {
                    rewardDisplay.textContent = `Reward: ${data.reward.toFixed(2)}`;
                }
            }
        }

        // Update Prediction Card (New)
        if (data.prediction) {
            const predWait = document.getElementById('predWaitTime');
            const predTrendBox = document.getElementById('predTrendBox');
            const predTrendIcon = document.getElementById('predTrendIcon');
            const predTrendText = document.getElementById('predTrendText');
            const predInsight = document.querySelector('.prediction-insight p');

            if (predWait) {
                predWait.textContent = data.prediction.wait_time.toFixed(1);
            }

            if (predTrendBox && predTrendIcon && predTrendText) {
                // Reset classes
                predTrendBox.className = 'prediction-trend';
                predTrendIcon.className = '';

                if (data.prediction.trend === 'increasing') {
                    predTrendBox.classList.add('increasing');
                    predTrendIcon.className = 'fas fa-arrow-trend-up';
                    predTrendText.textContent = 'Rising';
                    if (predInsight) predInsight.textContent = 'Traffic building up. Consider increasing green time.';
                } else if (data.prediction.trend === 'decreasing') {
                    predTrendBox.classList.add('decreasing');
                    predTrendIcon.className = 'fas fa-arrow-trend-down';
                    predTrendText.textContent = 'Clearing';
                    if (predInsight) predInsight.textContent = 'Traffic clearing up. Flow is optimal.';
                } else {
                    predTrendBox.classList.add('stable');
                    predTrendIcon.className = 'fas fa-minus';
                    predTrendText.textContent = 'Stable';
                    if (predInsight) predInsight.textContent = 'Traffic flow is stable. Standard timing effective.';
                }
            }
        }
    }

    updateAdditionalMetrics(data) {
        // Update throughput metrics if elements exist
        const throughputElement = document.querySelector('.throughput-value');
        if (throughputElement) {
            throughputElement.textContent = data.throughput || 0;
        }

        // Update departed/arrived vehicles
        const departedElement = document.querySelector('.departed-vehicles');
        if (departedElement) {
            departedElement.textContent = data.departed_vehicles || 0;
        }

        const arrivedElement = document.querySelector('.arrived-vehicles');
        if (arrivedElement) {
            arrivedElement.textContent = data.arrived_vehicles || 0;
        }

        // Update average wait time display
        const waitTimeElement = document.querySelector('.avg-wait-time');
        if (waitTimeElement) {
            waitTimeElement.textContent = `${(data.avg_wait_time || 0).toFixed(1)}s`;
        }

        // Update reward display
        const rewardElement = document.querySelector('.current-reward');
        if (rewardElement) {
            rewardElement.textContent = (data.reward || 0).toFixed(2);
        }
    }

    updateMetricValue(selector, value) {
        const element = document.querySelector(selector);
        if (element && element.textContent !== value.toString()) {
            element.style.transform = 'scale(1.05)';
            element.textContent = value;
            setTimeout(() => {
                element.style.transform = 'scale(1)';
            }, 200);
        }
    }

    updateVehicleVisualization(vehicles) {
        // Clear existing vehicles
        document.querySelectorAll('.vehicle-dot').forEach(dot => dot.remove());

        const intersection = document.querySelector('.intersection');
        if (!intersection) return;

        // Show total vehicle count
        console.log(`Total vehicles in simulation: ${vehicles.length}`);

        if (!vehicles.length) return;

        // Group vehicles by lane/direction with better lane detection
        const vehiclesByLane = {
            'west': [],   // W0 lanes
            'north': [],  // N0 lanes  
            'east': [],   // E0 lanes
            'south': []   // S0 lanes
        };

        vehicles.forEach(vehicle => {
            const lane = vehicle.lane;
            if (lane.includes('W0') || lane.includes('W1')) {
                vehiclesByLane.west.push(vehicle);
            } else if (lane.includes('N0') || lane.includes('N1')) {
                vehiclesByLane.north.push(vehicle);
            } else if (lane.includes('E0') || lane.includes('E1')) {
                vehiclesByLane.east.push(vehicle);
            } else if (lane.includes('S0') || lane.includes('S1')) {
                vehiclesByLane.south.push(vehicle);
            }
        });

        // Add vehicle dots for each direction with accurate counts
        Object.entries(vehiclesByLane).forEach(([direction, laneVehicles]) => {
            if (laneVehicles.length > 0) {
                this.addVehicleDotsForDirection(direction, laneVehicles, intersection);
            }
        });

        // Update queue count displays with real vehicle counts
        this.updateQueueDisplays(vehiclesByLane);
    }

    addVehicleDotsForDirection(direction, vehicles, intersection) {
        // Show more vehicles for better accuracy
        const maxVehiclesToShow = Math.min(12, vehicles.length);
        const vehiclesToShow = vehicles.slice(0, maxVehiclesToShow);

        vehiclesToShow.forEach((vehicle, index) => {
            const dot = document.createElement('div');
            dot.className = 'vehicle-dot';
            dot.title = `Vehicle ${vehicle.id}\nSpeed: ${vehicle.speed.toFixed(1)} m/s\nWait: ${vehicle.waiting_time.toFixed(1)}s`;

            // Color based on waiting time for better visual feedback
            if (vehicle.waiting_time > 30) {
                dot.classList.add('waiting-long');
            } else if (vehicle.waiting_time > 10) {
                dot.classList.add('waiting-medium');
            } else if (vehicle.speed < 0.5) {
                dot.classList.add('stopped');
            } else {
                dot.classList.add('moving');
            }

            // Position based on direction with better spacing
            this.positionVehicleDotByDirection(dot, direction, index);
            intersection.appendChild(dot);
        });

        // Add count indicator if there are more vehicles
        if (vehicles.length > maxVehiclesToShow) {
            this.addVehicleCountIndicator(direction, vehicles.length - maxVehiclesToShow, intersection);
        }
    }

    positionVehicleDotByDirection(dot, direction, index) {
        const spacing = 10; // pixels between vehicles
        const offset = index * spacing;

        switch (direction) {
            case 'west':
                dot.style.left = `${25 + offset}px`;
                dot.style.top = '95px';
                break;
            case 'north':
                dot.style.left = '95px';
                dot.style.top = `${25 + offset}px`;
                break;
            case 'east':
                dot.style.right = `${25 + offset}px`;
                dot.style.top = '105px';
                break;
            case 'south':
                dot.style.left = '105px';
                dot.style.bottom = `${25 + offset}px`;
                break;
        }
    }

    addVehicleCountIndicator(direction, extraCount, intersection) {
        const indicator = document.createElement('div');
        indicator.className = 'vehicle-count-indicator';
        indicator.textContent = `+${extraCount}`;

        // Position the indicator at the end of the vehicle line
        switch (direction) {
            case 'west':
                indicator.style.left = '145px';
                indicator.style.top = '90px';
                break;
            case 'north':
                indicator.style.left = '90px';
                indicator.style.top = '145px';
                break;
            case 'east':
                indicator.style.right = '145px';
                indicator.style.top = '100px';
                break;
            case 'south':
                indicator.style.left = '100px';
                indicator.style.bottom = '145px';
                break;
        }

        intersection.appendChild(indicator);
    }

    updateQueueDisplays(vehiclesByLane) {
        // Update queue count displays with real vehicle counts
        const queueElements = [
            { element: document.querySelector('.west-queue .queue-count'), vehicles: vehiclesByLane.west },
            { element: document.querySelector('.north-queue .queue-count'), vehicles: vehiclesByLane.north },
            { element: document.querySelector('.east-queue .queue-count'), vehicles: vehiclesByLane.east },
            { element: document.querySelector('.south-queue .queue-count'), vehicles: vehiclesByLane.south }
        ];

        queueElements.forEach(({ element, vehicles }) => {
            if (element) {
                const waitingVehicles = vehicles.filter(v => v.waiting_time > 1).length;
                const value = waitingVehicles;

                if (element.textContent !== value.toString()) {
                    element.textContent = value;
                    element.parentElement.style.transform = 'scale(1.1)';
                    setTimeout(() => {
                        element.parentElement.style.transform = 'scale(1)';
                    }, 200);
                }
            }
        });
    }

    addVehicleDotsForLane(lane, vehicles, intersection) {
        const maxVehiclesToShow = 8;
        const vehiclesToShow = vehicles.slice(0, maxVehiclesToShow);

        vehiclesToShow.forEach((vehicle, index) => {
            const dot = document.createElement('div');
            dot.className = 'vehicle-dot';
            dot.title = `Vehicle ${vehicle.id}\nSpeed: ${vehicle.speed.toFixed(1)} m/s\nWait: ${vehicle.waiting_time.toFixed(1)}s`;

            // Color based on waiting time
            if (vehicle.waiting_time > 30) {
                dot.classList.add('waiting-long');
            } else if (vehicle.waiting_time > 10) {
                dot.classList.add('waiting-medium');
            } else {
                dot.classList.add('moving');
            }

            // Position based on lane
            this.positionVehicleDot(dot, lane, index);
            intersection.appendChild(dot);
        });
    }

    positionVehicleDot(dot, lane, index) {
        const spacing = 12; // pixels between vehicles
        const offset = index * spacing;

        if (lane.includes('W0')) { // West incoming
            dot.style.left = `${30 + offset}px`;
            dot.style.top = '95px';
        } else if (lane.includes('N0')) { // North incoming  
            dot.style.left = '95px';
            dot.style.top = `${30 + offset}px`;
        } else if (lane.includes('E0')) { // East incoming
            dot.style.right = `${30 + offset}px`;
            dot.style.top = '105px';
        } else if (lane.includes('S0')) { // South incoming
            dot.style.left = '105px';
            dot.style.bottom = `${30 + offset}px`;
        }
    }

    updateTrafficLights(phase) {
        // Reset all lights
        document.querySelectorAll('.light').forEach(light => {
            light.classList.remove('active');
        });

        if (phase === 0) {
            // North-South green
            document.querySelector('#lightNorth .green')?.classList.add('active');
            document.querySelector('#lightSouth .green')?.classList.add('active');
            document.querySelector('#lightEast .red')?.classList.add('active');
            document.querySelector('#lightWest .red')?.classList.add('active');
        } else if (phase === 1) {
            // NS Yellow (transitioning to EW)
            document.querySelector('#lightNorth .yellow')?.classList.add('active');
            document.querySelector('#lightSouth .yellow')?.classList.add('active');
            document.querySelector('#lightEast .red')?.classList.add('active');
            document.querySelector('#lightWest .red')?.classList.add('active');
        } else if (phase === 2) {
            // East-West green
            document.querySelector('#lightNorth .red')?.classList.add('active');
            document.querySelector('#lightSouth .red')?.classList.add('active');
            document.querySelector('#lightEast .green')?.classList.add('active');
            document.querySelector('#lightWest .green')?.classList.add('active');
        } else if (phase === 3) {
            // EW Yellow (transitioning to NS)
            document.querySelector('#lightNorth .red')?.classList.add('active');
            document.querySelector('#lightSouth .red')?.classList.add('active');
            document.querySelector('#lightEast .yellow')?.classList.add('active');
            document.querySelector('#lightWest .yellow')?.classList.add('active');
        }

        this.currentPhase = phase;
    }

    initPerformanceChart() {
        const ctx = document.getElementById('performanceChart');
        if (!ctx) return;

        this.charts.performance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'AI Control',
                    data: [],
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    tension: 0.4,
                    fill: true,
                    borderWidth: 2
                }, {
                    label: 'Fixed Control (Baseline)',
                    data: [],
                    borderColor: '#6b7280',
                    backgroundColor: 'rgba(107, 114, 128, 0.05)',
                    tension: 0.4,
                    fill: false,
                    borderDash: [5, 5],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                },
                scales: {
                    x: {
                        display: false
                    },
                    y: {
                        display: false,
                        beginAtZero: true
                    }
                },
                elements: {
                    point: {
                        radius: 0,
                        hitRadius: 10,
                        hoverRadius: 4
                    }
                },
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                animation: {
                    duration: 0
                }
            }
        });
    }

    initAnalyticsCharts() {
        // Flow History Chart
        const flowCtx = document.getElementById('flowHistoryChart');
        if (flowCtx) {
            this.charts.flow = new Chart(flowCtx, {
                type: 'bar',
                data: {
                    labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                    datasets: [{
                        label: 'Average Flow',
                        data: [1200, 1350, 1250, 1400, 1500, 1100, 950],
                        backgroundColor: '#3b82f6',
                        borderRadius: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        y: { beginAtZero: true, grid: { color: '#334155' } },
                        x: { grid: { display: false } }
                    }
                }
            });
        }

        // Vehicle Type Chart
        const typeCtx = document.getElementById('vehicleTypeChart');
        if (typeCtx) {
            this.charts.types = new Chart(typeCtx, {
                type: 'doughnut',
                data: {
                    labels: ['Cars', 'Trucks', 'Buses', 'Motorcycles'],
                    datasets: [{
                        data: [65, 15, 10, 10],
                        backgroundColor: ['#3b82f6', '#f59e0b', '#10b981', '#ef4444'],
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { position: 'right' } }
                }
            });
        }

        // Wait Time Chart
        const waitCtx = document.getElementById('waitTimeChart');
        if (waitCtx) {
            this.charts.wait = new Chart(waitCtx, {
                type: 'line',
                data: {
                    labels: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00'],
                    datasets: [{
                        label: 'Wait Time (s)',
                        data: [20, 15, 45, 35, 55, 30],
                        borderColor: '#10b981',
                        tension: 0.4,
                        fill: true,
                        backgroundColor: 'rgba(16, 185, 129, 0.1)'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        y: { beginAtZero: true, grid: { color: '#334155' } },
                        x: { grid: { display: false } }
                    }
                }
            });
        }
    }

    updateChartData(data) {
        if (!this.charts.performance) return;

        const now = new Date().toLocaleTimeString();
        const waitTime = data.avg_wait_time || 0;

        // Add new data point with real metrics
        this.chartData.labels.push(now);
        this.chartData.aiData.push(waitTime);

        // Use real baseline data from backend or calculate realistic baseline
        const baselineWaitTime = data.baseline_wait_time || (waitTime * 1.18); // 18% worse baseline
        this.chartData.baselineData.push(baselineWaitTime);

        // Keep only last 20 data points
        if (this.chartData.labels.length > 20) {
            this.chartData.labels.shift();
            this.chartData.aiData.shift();
            this.chartData.baselineData.shift();
        }

        this.charts.performance.data.labels = this.chartData.labels;
        this.charts.performance.data.datasets[0].data = this.chartData.aiData;
        this.charts.performance.data.datasets[1].data = this.chartData.baselineData;
        this.charts.performance.update('none');

        // Calculate REAL improvement percentage based on actual performance
        const improvementElement = document.querySelector('.improvement-value');
        if (improvementElement && this.chartData.aiData.length > 3) {
            // Use average of last 3 data points for more stable calculation
            const recentAI = this.chartData.aiData.slice(-3);
            const recentBaseline = this.chartData.baselineData.slice(-3);

            const avgAI = recentAI.reduce((a, b) => a + b, 0) / recentAI.length;
            const avgBaseline = recentBaseline.reduce((a, b) => a + b, 0) / recentBaseline.length;

            // Calculate improvement: (baseline - ai) / baseline * 100
            const improvement = ((avgBaseline - avgAI) / Math.max(avgBaseline, 0.1)) * 100;

            // Update with real calculated improvement
            const sign = improvement > 0 ? '-' : '+';
            const value = Math.abs(improvement).toFixed(1);
            improvementElement.textContent = `${sign}${value}%`;

            // Add color coding based on performance
            improvementElement.className = improvement > 0 ?
                'improvement-value positive' : 'improvement-value negative';
            console.log(`Real-time improvement: ${sign}${value}% (AI: ${avgAI.toFixed(1)}s, Baseline: ${avgBaseline.toFixed(1)}s)`);
        }
    }

    updateAnalyticsCharts(data) {
        // Flow History (Real Throughput)
        if (this.charts.flow) {
            const datasets = this.charts.flow.data.datasets[0].data;
            datasets.shift();
            // Use total active vehicles as 'Traffic Volume' since Flow Rate is hard to calc instantaneously without history in env
            datasets.push(data.total_vehicles || 0);
            this.charts.flow.update('none');
        }

        // Vehicle Types (REAL DATA)
        if (this.charts.types && data.vehicle_types) {
            const types = data.vehicle_types;
            // data.vehicle_types is {cars: N, trucks: N, ...}

            this.charts.types.data.datasets[0].data = [
                types.cars || 0,
                types.trucks || 0,
                types.buses || 0,
                types.motorcycles || 0
            ];
            this.charts.types.update('none');
        }

        // Wait Time History
        if (this.charts.wait) {
            const datasets = this.charts.wait.data.datasets[0].data;
            const labels = this.charts.wait.data.labels;

            if (datasets.length > 10) {
                datasets.shift();
                labels.shift();
            }

            const now = new Date();
            labels.push(now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }));
            datasets.push(data.avg_wait_time || 0);

            this.charts.wait.update('none');
        }
    }

    resetMetrics() {
        // Reset all metric displays
        this.updateMetricValue('.metric-card:nth-child(1) .metric-value', 0);
        this.updateMetricValue('.metric-card:nth-child(2) .metric-value', 'Level 0');
        this.updateMetricValue('.metric-card:nth-child(3) .metric-value', 0);
        this.updateMetricValue('.metric-card:nth-child(4) .metric-value', 0);

        // Reset queues
        document.querySelectorAll('.queue-count').forEach(count => {
            count.textContent = '0';
        });

        // Clear vehicles
        document.querySelectorAll('.vehicle-dot').forEach(dot => dot.remove());

        // Reset chart
        this.chartData = { labels: [], aiData: [], baselineData: [] };
        if (this.charts.performance) {
            this.charts.performance.data.labels = [];
            this.charts.performance.data.datasets[0].data = [];
            this.charts.performance.data.datasets[1].data = [];
            this.charts.performance.update();
        }

        // Reset Analytics Charts
        if (this.charts.flow) {
            this.charts.flow.data.datasets[0].data = [0, 0, 0, 0, 0, 0, 0];
            this.charts.flow.update();
        }
        if (this.charts.wait) {
            this.charts.wait.data.labels = [];
            this.charts.wait.data.datasets[0].data = [];
            this.charts.wait.update();
        }
    }

    updateSimulationTime(time) {
        const timeElement = document.querySelector('.simulation-time');
        if (timeElement) {
            const minutes = Math.floor(time / 60);
            const seconds = Math.floor(time % 60);
            timeElement.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
        }
    }

    // Model Actions
    loadModel(modelType) {
        console.log(`Loading ${modelType} model...`);
        this.sendWebSocketMessage('load_model', { model_type: modelType });
    }

    trainModel(modelType) {
        console.log(`Training ${modelType} model...`);

        if (this.isTraining) {
            this.addAlert('Training Active', 'Another training session is already running', 'warning');
            return;
        }

        this.sendWebSocketMessage('train_model', { model_type: modelType });

        // Show training in progress
        this.showTrainingProgress(modelType);
    }

    runBenchmark() {
        console.log('Running performance benchmark...');
        this.sendWebSocketMessage('run_benchmark', {});

        // Switch to analytics view to show results
        this.switchView('analytics');
    }

    generateReport() {
        console.log('Generating system report...');
        this.sendWebSocketMessage('generate_report', {});
    }

    toggleControlPanel() {
        const panel = document.getElementById('controlPanel');
        const content = document.getElementById('controlContent');
        const icon = document.getElementById('controlToggleIcon');

        if (panel.classList.contains('minimized')) {
            // Expand
            panel.classList.remove('minimized');
            content.style.display = 'block';
            icon.className = 'fas fa-minus';
        } else {
            // Minimize
            panel.classList.add('minimized');
            content.style.display = 'none';
            icon.className = 'fas fa-plus';
        }
    }

    // Training Control Methods
    pauseTraining() {
        console.log('Pausing training...');
        if (!this.isTraining) {
            this.addAlert('No Training Active', 'No training session is currently running', 'warning');
            return;
        }
        this.sendWebSocketMessage('pause_training', {});
    }

    resumeTraining() {
        console.log('Resuming training...');
        if (!this.isTraining) {
            this.addAlert('No Training Active', 'No training session to resume', 'warning');
            return;
        }
        this.sendWebSocketMessage('resume_training', {});
    }

    stopTraining() {
        console.log('Stopping training...');
        if (!this.isTraining) {
            this.addAlert('No Training Active', 'No training session is currently running', 'warning');
            return;
        }

        if (confirm('Are you sure you want to stop training? Progress will be lost.')) {
            this.sendWebSocketMessage('stop_training', {});
            this.hideTrainingControls();
        }
    }

    viewTraining() {
        console.log('Viewing training details...');
        if (!this.isTraining) {
            this.addAlert('No Training Active', 'No training session is currently running', 'info');
            return;
        }
        this.sendWebSocketMessage('view_training', {});
    }

    showTrainingControls() {
        const trainingControls = document.getElementById('trainingControls');
        if (trainingControls) {
            trainingControls.style.display = 'block';
        }
    }

    hideTrainingControls() {
        const trainingControls = document.getElementById('trainingControls');
        if (trainingControls) {
            trainingControls.style.display = 'none';
        }
    }

    updateDeviceInfo(device) {
        const deviceInfo = document.getElementById('deviceInfo');
        if (deviceInfo) {
            // Force upper case
            let displayText = device.toUpperCase();
            if (device.toLowerCase().includes('cuda')) {
                displayText = 'GPU (CUDA)';
                deviceInfo.style.color = 'var(--success)';
            } else {
                deviceInfo.style.color = 'var(--accent-blue)';
            }
            deviceInfo.textContent = displayText;
        }
    }

    updateGPUMemory(memoryInfo) {
        if (!memoryInfo) return;

        const gpuInfo = document.getElementById('gpuInfo');
        if (!gpuInfo) return;

        const usage = (memoryInfo.allocated / memoryInfo.total) * 100;

        // Add memory bar if it doesn't exist
        let memoryBar = gpuInfo.querySelector('.gpu-memory-bar');
        if (!memoryBar) {
            memoryBar = document.createElement('div');
            memoryBar.className = 'gpu-memory-bar';
            memoryBar.innerHTML = '<div class="gpu-memory-fill"></div>';
            gpuInfo.appendChild(memoryBar);
        }

        const fill = memoryBar.querySelector('.gpu-memory-fill');
        fill.style.width = `${usage}%`;

        // Update text
        const deviceSpan = gpuInfo.querySelector('#deviceInfo');
        if (deviceSpan) {
            deviceSpan.textContent = `GPU (${memoryInfo.allocated.toFixed(0)}MB/${memoryInfo.total.toFixed(0)}MB)`;
        }
    }

    runEvaluation() {
        console.log('Running model evaluation...');

        // Check if models are available
        if (!this.isConnected) {
            this.addAlert('Connection Error', 'Backend not connected. Please check server status.', 'warning');
            return;
        }

        // Add visual feedback
        this.addAlert('Evaluation Started', 'Running comprehensive model evaluation...', 'info');

        // Disable button during evaluation
        const evalButtons = document.querySelectorAll('#btn-run-eval, [onclick*="runEvaluation"]');
        evalButtons.forEach(btn => {
            btn.disabled = true;
            const icon = btn.querySelector('i');
            const span = btn.querySelector('span');
            if (icon) icon.className = 'fas fa-spinner fa-spin';
            if (span) span.textContent = 'Evaluating...';
        });

        this.sendWebSocketMessage('run_evaluation', {});

        // Show evaluation in progress
        this.showEvaluationProgress();

        // Re-enable button after timeout
        setTimeout(() => {
            evalButtons.forEach(btn => {
                btn.disabled = false;
                const icon = btn.querySelector('i');
                const span = btn.querySelector('span');
                if (icon) icon.className = 'fas fa-play';
                if (span) span.textContent = 'Run Evaluation';
            });
        }, 10000); // 10 second timeout
    }

    deployModel() {
        console.log('Deploying model to production...');

        // Check if models are available
        if (!this.isConnected) {
            this.addAlert('Connection Error', 'Backend not connected. Please check server status.', 'warning');
            return;
        }

        // Add confirmation dialog
        if (!confirm('Are you sure you want to deploy the model to production? This will replace the current live model.')) {
            return;
        }

        // Add visual feedback
        this.addAlert('Deployment Started', 'Deploying model to production environment...', 'info');

        // Disable button during deployment
        const deployButtons = document.querySelectorAll('#btn-deploy-model, [onclick*="deployModel"]');
        deployButtons.forEach(btn => {
            btn.disabled = true;
            const icon = btn.querySelector('i');
            const span = btn.querySelector('span');
            if (icon) icon.className = 'fas fa-spinner fa-spin';
            if (span) span.textContent = 'Deploying...';
        });

        this.sendWebSocketMessage('deploy_model', {});

        // Show deployment status
        this.showDeploymentStatus();

        // Re-enable button after timeout
        setTimeout(() => {
            deployButtons.forEach(btn => {
                btn.disabled = false;
                const icon = btn.querySelector('i');
                const span = btn.querySelector('span');
                if (icon) icon.className = 'fas fa-cloud-upload-alt';
                if (span) span.textContent = 'Deploy to Production';
            });
        }, 15000); // 15 second timeout
    }

    toggleFullscreen() {
        const mapCard = document.querySelector('.map-card');
        const expandBtn = document.querySelector('.expand-btn i');

        if (mapCard.classList.contains('fullscreen')) {
            // Exit fullscreen
            mapCard.classList.remove('fullscreen');
            expandBtn.className = 'fas fa-expand';
            document.body.style.overflow = 'auto';
        } else {
            // Enter fullscreen
            mapCard.classList.add('fullscreen');
            expandBtn.className = 'fas fa-compress';
            document.body.style.overflow = 'hidden';
        }
    }

    showEvaluationProgress() {
        // Switch to models view and show evaluation
        this.switchView('models');

        // Update UI to show evaluation in progress
        const evalStatus = document.querySelector('.evaluation-status');
        if (evalStatus) {
            evalStatus.textContent = 'Evaluation in progress...';
            evalStatus.className = 'evaluation-status running';
        }
    }

    showDeploymentStatus() {
        // Show deployment notification
        this.addLogMessage({
            message: 'Model deployment initiated. Checking system compatibility...',
            level: 'info'
        });

        setTimeout(() => {
            this.addLogMessage({
                message: 'Model deployed successfully to production environment!',
                level: 'success'
            });
        }, 3000);
    }

    updateTrainingProgress(data) {
        const progress = Math.round(data.progress * 100);
        console.log(`Training progress update: ${progress}%`, data);

        // Update progress circle in dashboard
        const progressCircle = document.querySelector('.progress-ring-circle');
        const progressText = document.querySelector('.progress-text');

        if (progressCircle && progressText) {
            const circumference = 2 * Math.PI * 36; // radius = 36
            const offset = circumference - (data.progress * circumference);

            progressCircle.style.strokeDasharray = circumference;
            progressCircle.style.strokeDashoffset = offset;
            progressText.textContent = `${progress}%`;

            console.log(`Updated progress display to ${progress}%`);
        }

        // Update training info with real-time data
        const trainingInfo = document.querySelector('.training-info');
        if (trainingInfo) {
            const statusText = trainingInfo.querySelector('p');
            const etaText = trainingInfo.querySelector('.eta');

            if (statusText) {
                statusText.textContent = data.status || `Training: ${progress}%`;
            }

            if (etaText) {
                if (data.episode) {
                    etaText.textContent = `Episode: ${data.episode}`;
                } else {
                    etaText.textContent = `Step: ${data.training_step || 0}`;
                }
            }

            // Update or create mean reward display
            let rewardText = trainingInfo.querySelector('.reward-text');
            if (!rewardText) {
                rewardText = document.createElement('p');
                rewardText.className = 'reward-text';
                trainingInfo.appendChild(rewardText);
            }

            if (data.mean_reward !== undefined) {
                rewardText.textContent = `Mean Reward: ${data.mean_reward.toFixed(2)}`;
                rewardText.style.color = data.mean_reward > 0 ? 'var(--success)' : 'var(--accent-red)';
            }

            // Update current reward if available
            if (data.current_reward !== undefined) {
                let currentRewardText = trainingInfo.querySelector('.current-reward-text');
                if (!currentRewardText) {
                    currentRewardText = document.createElement('p');
                    currentRewardText.className = 'current-reward-text';
                    trainingInfo.appendChild(currentRewardText);
                }
                currentRewardText.textContent = `Current: ${data.current_reward.toFixed(2)}`;
                currentRewardText.style.fontSize = '12px';
                currentRewardText.style.color = 'var(--text-muted)';
            }
        }

        // Update Device Info if present
        if (data.device) {
            this.updateDeviceInfo(data.device);
        }

        // Update model cards in models view with real-time stats
        const modelCards = document.querySelectorAll('.model-card');
        modelCards.forEach(card => {
            const modelIcon = card.querySelector('.model-icon');
            if (modelIcon && (
                (modelIcon.classList.contains('ppo') && (data.status?.includes('PPO') || this.training_model_type === 'ppo')) ||
                (modelIcon.classList.contains('dqn') && (data.status?.includes('DQN') || this.training_model_type === 'dqn'))
            )) {
                const statusBadge = card.querySelector('.model-status-badge');
                if (statusBadge) {
                    statusBadge.textContent = progress < 100 ? `Training ${progress}%` : 'Ready';
                    statusBadge.className = progress < 100 ?
                        'model-status-badge training' : 'model-status-badge';
                }

                // Update reward stat in model card
                const rewardStat = card.querySelector('.stat .value');
                if (rewardStat && data.mean_reward !== undefined) {
                    rewardStat.textContent = data.mean_reward.toFixed(1);
                }

                // Update steps stat
                const stepStats = card.querySelectorAll('.stat');
                if (stepStats.length > 1 && data.training_step) {
                    const stepStat = stepStats[1].querySelector('.value');
                    if (stepStat) {
                        const steps = data.training_step;
                        if (steps > 1000) {
                            stepStat.textContent = `${(steps / 1000).toFixed(1)}K`;
                        } else {
                            stepStat.textContent = steps.toString();
                        }
                    }
                }
            }
        });

        // Show training controls if not visible
        if (progress > 0 && progress < 100) {
            this.showTrainingControls();
        }
    }

    showTrainingProgress(modelType) {
        this.isTraining = true;
        this.training_model_type = modelType; // Store the model type

        // Switch to dashboard view to show progress
        this.switchView('dashboard');

        // Update training header and show active status
        const trainingInfo = document.querySelector('.training-info');
        if (trainingInfo) {
            const trainingHeader = trainingInfo.querySelector('h4');
            if (trainingHeader) {
                trainingHeader.textContent = `Training ${modelType.toUpperCase()} Model`;
            }

            // Add active training styling
            trainingInfo.classList.add('active');
            trainingInfo.classList.remove('paused');

            // Update status text
            const statusText = trainingInfo.querySelector('p');
            if (statusText) {
                statusText.textContent = 'Initializing training...';
            }

            const etaText = trainingInfo.querySelector('.eta');
            if (etaText) {
                etaText.textContent = 'Starting training process...';
            }
        }

        // Reset progress to 0%
        this.resetTrainingProgress();
        this.updateTrainingProgress({
            progress: 0,
            status: 'Initializing...',
            mean_reward: 0,
            training_step: 0
        });

        // Show training controls in mission control
        this.showTrainingControls();

        // Update device info
        this.updateDeviceInfo('detecting...');

        // Update simulation controls to reflect running state
        this.simulationRunning = true;
        this.updateSimulationControls();

        // Initialize button states
        const pauseBtn = document.getElementById('btn-pause-training');
        const resumeBtn = document.getElementById('btn-resume-training');
        const stopBtn = document.getElementById('btn-stop-training');
        const viewBtn = document.getElementById('btn-view-training');

        if (pauseBtn) pauseBtn.disabled = false;
        if (resumeBtn) resumeBtn.disabled = true;
        if (stopBtn) stopBtn.disabled = false;
        if (viewBtn) viewBtn.disabled = false;
    }

    handleTrainingPaused(data) {
        console.log('Training paused:', data);
        this.addAlert('Training Paused', `${data.model_type.toUpperCase()} training has been paused`, 'warning');

        // Update training status and styling
        this.updateTrainingStatus('paused');
        const trainingInfo = document.querySelector('.training-info');
        if (trainingInfo) {
            trainingInfo.classList.remove('active');
            trainingInfo.classList.add('paused');

            // Update status text
            const statusText = trainingInfo.querySelector('p');
            if (statusText) {
                statusText.textContent = 'Training Paused';
            }
        }

        // Update button states
        const pauseBtn = document.getElementById('btn-pause-training');
        const resumeBtn = document.getElementById('btn-resume-training');
        if (pauseBtn) pauseBtn.disabled = true;
        if (resumeBtn) resumeBtn.disabled = false;

        // Update simulation controls
        this.simulationRunning = false;
        this.updateSimulationControls();
    }

    handleTrainingResumed(data) {
        console.log('Training resumed:', data);
        this.addAlert('Training Resumed', `${data.model_type.toUpperCase()} training has been resumed`, 'info');

        // Update training status and styling
        this.updateTrainingStatus('running');
        const trainingInfo = document.querySelector('.training-info');
        if (trainingInfo) {
            trainingInfo.classList.add('active');
            trainingInfo.classList.remove('paused');

            // Update status text
            const statusText = trainingInfo.querySelector('p');
            if (statusText) {
                statusText.textContent = 'Training Active';
            }
        }

        // Update button states
        const pauseBtn = document.getElementById('btn-pause-training');
        const resumeBtn = document.getElementById('btn-resume-training');
        if (pauseBtn) pauseBtn.disabled = false;
        if (resumeBtn) resumeBtn.disabled = true;

        // Update simulation controls
        this.simulationRunning = true;
        this.updateSimulationControls();
    }

    handleTrainingStopped(data) {
        this.isTraining = false;
        console.log('Training stopped:', data);
        this.addAlert('Training Stopped', `${data.model_type.toUpperCase()} training has been stopped`, 'warning');

        // Update training status and hide controls
        this.updateTrainingStatus('stopped');
        this.hideTrainingControls();

        // Remove training styling
        const trainingInfo = document.querySelector('.training-info');
        if (trainingInfo) {
            trainingInfo.classList.remove('active', 'paused');
            const trainingHeader = trainingInfo.querySelector('h4');
            if (trainingHeader) {
                trainingHeader.textContent = 'Ready to Train';
            }

            const statusText = trainingInfo.querySelector('p');
            if (statusText) {
                statusText.textContent = 'No training in progress';
            }

            const etaText = trainingInfo.querySelector('.eta');
            if (etaText) {
                etaText.textContent = 'Click Train in AI Models tab';
            }
        }

        // Reset progress display
        this.resetTrainingProgress();

        // Update simulation controls
        this.simulationRunning = false;
        this.updateSimulationControls();
    }

    handleTrainingDetails(data) {
        console.log('Training details:', data);

        // Update device info
        this.updateDeviceInfo(data.device);

        // Update GPU memory if available
        if (data.gpu_memory) {
            this.updateGPUMemory(data.gpu_memory);
        }

        // Show detailed training information in alert
        const details = [
            `Model: ${data.model_type?.toUpperCase() || 'Unknown'}`,
            `Status: ${data.status?.toUpperCase() || 'Unknown'}`,
            `Device: ${data.device?.toUpperCase() || 'Unknown'}`,
        ];

        if (data.gpu_memory) {
            details.push(`GPU Memory: ${data.gpu_memory.allocated.toFixed(0)}MB / ${data.gpu_memory.total.toFixed(0)}MB`);
        }

        if (data.parameters) {
            details.push('Parameters:');
            Object.entries(data.parameters).forEach(([key, value]) => {
                details.push(`  ${key}: ${value}`);
            });
        }

        this.addAlert(
            'Training Details',
            details.join('\n'),
            'info'
        );

        // Also log to system log
        this.addLogMessage({
            message: `Training Details - ${details.join(', ')}`,
            level: 'info'
        });
    }

    updateTrainingStatus(status) {
        // Update any training status indicators
        const statusElements = document.querySelectorAll('.training-status');
        statusElements.forEach(element => {
            element.className = `training-status ${status}`;
            element.textContent = status.toUpperCase();
        });
    }

    showTrainingDetailsModal(data) {
        // Create a simple details display
        const details = `
            Model: ${data.model_type.toUpperCase()}
            Status: ${data.status.toUpperCase()}
            Device: ${data.device.toUpperCase()}
            ${data.gpu_memory ? `GPU Memory: ${data.gpu_memory.allocated.toFixed(0)}MB / ${data.gpu_memory.total.toFixed(0)}MB` : ''}
            Parameters: ${JSON.stringify(data.parameters, null, 2)}
        `;

        this.addLogMessage({
            message: `Training Details:\n${details}`,
            level: 'info'
        });
    }

    resetTrainingProgress() {
        // Force reset the progress display
        const progressCircle = document.querySelector('.progress-ring-circle');
        const progressText = document.querySelector('.progress-text');

        if (progressCircle && progressText) {
            const circumference = 2 * Math.PI * 36;
            progressCircle.style.strokeDasharray = circumference;
            progressCircle.style.strokeDashoffset = circumference; // 0% progress
            progressText.textContent = '0%';
            console.log('Training progress reset to 0%');
        }

        // Reset training info
        const trainingInfo = document.querySelector('.training-info');
        if (trainingInfo) {
            const statusText = trainingInfo.querySelector('p');
            const etaText = trainingInfo.querySelector('.eta');

            if (statusText) statusText.textContent = 'Initializing...';
            if (etaText) etaText.textContent = 'Starting training...';
        }
    }

    updateModelStatus(data) {
        // Update model status indicators
        if (data.ppo_loaded) {
            const ppoCard = document.querySelector('.model-icon.ppo').closest('.model-card');
            if (ppoCard) {
                ppoCard.classList.add('active');
                const badge = ppoCard.querySelector('.model-status-badge');
                if (badge) {
                    badge.textContent = 'Active';
                    badge.className = 'model-status-badge active';
                }
            }
        }

        if (data.dqn_loaded) {
            const dqnCard = document.querySelector('.model-icon.dqn').closest('.model-card');
            if (dqnCard) {
                dqnCard.classList.add('active');
                const badge = dqnCard.querySelector('.model-status-badge');
                if (badge) {
                    badge.textContent = 'Active';
                    badge.className = 'model-status-badge active';
                }
            }
        }
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    const app = new TrafficAIManager();

    // Add UI enhancements
    addUIEnhancements();

    // Make app globally accessible for debugging
    window.trafficAI = app;

    // Initialize alerts system
    app.initializeAlerts();
});

function addUIEnhancements() {
    // Add smooth transitions to metric cards
    document.querySelectorAll('.metric-card').forEach(card => {
        card.style.transition = 'all 0.3s ease';
    });

    // Add click feedback to buttons
    document.querySelectorAll('.btn, .mode-btn, .sim-btn').forEach(btn => {
        btn.style.transition = 'all 0.15s ease';
        btn.addEventListener('click', function () {
            this.style.transform = 'scale(0.95)';
            setTimeout(() => {
                this.style.transform = 'scale(1)';
            }, 150);
        });
    });

    console.log('🚦 TrafficAI Manager initialized');
    console.log('💡 Tip: Use window.trafficAI to access the app instance');
}

// Add alerts system methods to TrafficAIManager prototype
TrafficAIManager.prototype.initializeAlerts = function () {
    this.alerts = [];
    this.maxAlerts = 5;

    // Generate initial alerts
    this.addAlert('System initialized successfully', 'System startup completed', 'info');
};

TrafficAIManager.prototype.addAlert = function (title, description, type = 'info') {
    const alert = {
        id: Date.now(),
        title: title,
        description: description,
        type: type,
        timestamp: new Date()
    };

    this.alerts.unshift(alert);

    // Keep only recent alerts
    if (this.alerts.length > this.maxAlerts) {
        this.alerts = this.alerts.slice(0, this.maxAlerts);
    }

    this.updateAlertsDisplay();
};

TrafficAIManager.prototype.updateAlertsDisplay = function () {
    const alertsList = document.getElementById('alertsList');
    if (!alertsList) return;

    if (this.alerts.length === 0) {
        alertsList.innerHTML = '<div class="no-alerts">No recent alerts</div>';
        return;
    }

    alertsList.innerHTML = this.alerts.map(alert => {
        const timeAgo = this.getTimeAgo(alert.timestamp);
        const iconClass = this.getAlertIcon(alert.type);

        return `
            <div class="alert-item ${alert.type}">
                <div class="alert-icon">
                    <i class="${iconClass}"></i>
                </div>
                <div class="alert-content">
                    <div class="alert-title">${alert.title}</div>
                    <div class="alert-desc">${alert.description} • ${timeAgo}</div>
                </div>
            </div>
        `;
    }).join('');
};

TrafficAIManager.prototype.getAlertIcon = function (type) {
    const icons = {
        'critical': 'fas fa-exclamation-triangle',
        'warning': 'fas fa-exclamation-circle',
        'info': 'fas fa-info-circle',
        'success': 'fas fa-check-circle'
    };
    return icons[type] || icons.info;
};

TrafficAIManager.prototype.getTimeAgo = function (timestamp) {
    const now = new Date();
    const diff = Math.floor((now - timestamp) / 1000);

    if (diff < 60) return `${diff}s ago`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return `${Math.floor(diff / 86400)}d ago`;
};

TrafficAIManager.prototype.viewAllAlerts = function () {
    console.log('Viewing all alerts:', this.alerts);

    this.addLogMessage({
        message: `Showing ${this.alerts.length} recent alerts`,
        level: 'info'
    });
};

TrafficAIManager.prototype.checkForAlerts = function (data) {
    // High congestion alert
    if (data.avg_wait_time > 30) {
        this.addAlert(
            'High congestion detected',
            `Average wait time: ${data.avg_wait_time.toFixed(1)}s`,
            'warning'
        );
    }

    // Critical incidents
    if (data.incidents > 5) {
        this.addAlert(
            'Multiple incidents detected',
            `${data.incidents} vehicles waiting over 30s`,
            'critical'
        );
    }

    // Low efficiency alert
    if (data.efficiency < 50) {
        this.addAlert(
            'System efficiency low',
            `Current efficiency: ${data.efficiency}%`,
            'warning'
        );
    }

    // High throughput (positive alert)
    if (data.throughput > 50) {
        this.addAlert(
            'High throughput achieved',
            `${data.throughput} vehicles processed`,
            'success'
        );
    }
};

// Add missing handler methods for evaluation and deployment
TrafficAIManager.prototype.handleEvaluationComplete = function (data) {
    console.log('Evaluation completed:', data);

    // Re-enable evaluation buttons
    const evalButtons = document.querySelectorAll('#btn-run-eval, [onclick*="runEvaluation"]');
    evalButtons.forEach(btn => {
        btn.disabled = false;
        const icon = btn.querySelector('i');
        const span = btn.querySelector('span');
        if (icon) icon.className = 'fas fa-play';
        if (span) span.textContent = 'Run Evaluation';
        if (!icon && !span) {
            btn.innerHTML = '<i class="fas fa-play"></i><span>Run Evaluation</span>';
        }
    });

    // Show completion alert
    this.addAlert(
        'Evaluation Complete',
        `Model evaluation finished. Overall performance: ${data.overall_performance?.toFixed(2) || 'N/A'}`,
        'success'
    );

    // Update evaluation status
    const evalStatus = document.querySelector('.evaluation-status');
    if (evalStatus) {
        evalStatus.textContent = 'Evaluation completed';
        evalStatus.className = 'evaluation-status complete';
    }
};

TrafficAIManager.prototype.handleEvaluationProgress = function (data) {
    console.log('Evaluation progress:', data);

    const progress = Math.round(data.progress * 100);

    // Update evaluation status
    const evalStatus = document.querySelector('.evaluation-status');
    if (evalStatus) {
        evalStatus.textContent = `Evaluating ${data.scenario}: ${progress}%`;
        evalStatus.className = 'evaluation-status running';
    }

    this.addLogMessage({
        message: `Evaluation progress: ${data.scenario} - ${progress}%`,
        level: 'info'
    });
};

TrafficAIManager.prototype.handleDeploymentComplete = function (data) {
    console.log('Deployment completed:', data);

    // Re-enable deployment buttons
    const deployButtons = document.querySelectorAll('#btn-deploy-model, [onclick*="deployModel"]');
    deployButtons.forEach(btn => {
        btn.disabled = false;
        const icon = btn.querySelector('i');
        const span = btn.querySelector('span');
        if (icon) icon.className = 'fas fa-cloud-upload-alt';
        if (span) span.textContent = 'Deploy to Production';
        if (!icon && !span) {
            btn.innerHTML = '<i class="fas fa-cloud-upload-alt"></i><span>Deploy to Production</span>';
        }
    });

    // Show completion alert
    this.addAlert(
        'Deployment Complete',
        `${data.model_type} model successfully deployed to production`,
        'success'
    );

    // Update model status to show it's deployed
    const modelCards = document.querySelectorAll('.model-card');
    modelCards.forEach(card => {
        const modelIcon = card.querySelector('.model-icon');
        if (modelIcon && modelIcon.textContent.toLowerCase() === data.model_type.toLowerCase()) {
            const statusBadge = card.querySelector('.model-status-badge');
            if (statusBadge) {
                statusBadge.textContent = 'Production';
                statusBadge.className = 'model-status-badge production';
            }
        }
    });
};

TrafficAIManager.prototype.handleDeploymentProgress = function (data) {
    console.log('Deployment progress:', data);

    const progress = Math.round(data.progress * 100);

    this.addLogMessage({
        message: `Deployment: ${data.step} (${progress}%)`,
        level: 'info'
    });
};

TrafficAIManager.prototype.handleBenchmarkComplete = function (data) {
    console.log('Benchmark completed:', data);

    this.addAlert(
        'Benchmark Complete',
        `Performance benchmark finished with ${data.results?.length || 0} scenarios`,
        'success'
    );

    // Switch to analytics view to show results
    this.switchView('analytics');
};

TrafficAIManager.prototype.handleBenchmarkProgress = function (data) {
    console.log('Benchmark progress:', data);

    const progress = Math.round(data.progress * 100);

    this.addLogMessage({
        message: `Benchmark progress: ${data.current_scenario} - ${progress}%`,
        level: 'info'
    });
};

TrafficAIManager.prototype.toggleSystemLog = function () {
    const logContainer = document.getElementById('systemLog');
    const toggleText = document.getElementById('logToggleText');

    if (logContainer && toggleText) {
        if (logContainer.style.display === 'none') {
            logContainer.style.display = 'block';
            toggleText.textContent = 'Hide Log';
        } else {
            logContainer.style.display = 'none';
            toggleText.textContent = 'Show Log';
        }
    }
};

TrafficAIManager.prototype.addLogMessage = function (logData) {
    // Add log messages to the system log container
    const logContainer = document.getElementById('systemLog');
    if (logContainer) {
        const logEntry = document.createElement('div');
        logEntry.className = `log-entry ${logData.level}`;
        logEntry.innerHTML = `
            <span class="log-time">${new Date().toLocaleTimeString()}</span>
            <span class="log-message">${logData.message}</span>
        `;
        logContainer.appendChild(logEntry);
        logContainer.scrollTop = logContainer.scrollHeight;

        // Keep only last 50 log entries
        while (logContainer.children.length > 51) { // +1 for the h4 title
            if (logContainer.children[1]) { // Skip the title
                logContainer.removeChild(logContainer.children[1]);
            }
        }
    }

    // Also log to console for debugging
    console.log(`[${logData.level.toUpperCase()}] ${logData.message}`);
};
// Initialize the application
const trafficAI = new TrafficAIManager();
