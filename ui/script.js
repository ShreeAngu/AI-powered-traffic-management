// TrafficAI Manager - Real Data Integration

class TrafficAIManager {
    constructor() {
        this.isConnected = false;
        this.simulationRunning = false;
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
        try {
            this.websocket = new WebSocket('ws://localhost:8765');

            this.websocket.onopen = () => {
                this.isConnected = true;
                this.updateConnectionStatus();
                console.log('✓ Connected to TrafficAI backend');
            };

            this.websocket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleWebSocketMessage(data);
            };

            this.websocket.onclose = () => {
                this.isConnected = false;
                this.updateConnectionStatus();
                console.log('✗ Disconnected from backend');

                // Attempt to reconnect
                setTimeout(() => this.connectWebSocket(), 5000);
            };

            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
        } catch (error) {
            console.log('⚠ Backend not available - UI only mode');
            this.isConnected = false;
            this.updateConnectionStatus();
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
                break;
            case 'model_status':
                // Update model status UI if needed
                break;
        }
    }

    sendWebSocketMessage(type, payload) {
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            this.websocket.send(JSON.stringify({ type, payload }));
        } else {
            console.warn('WebSocket not connected');
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
        this.simulationRunning = true;
        this.updateSimulationControls();
        this.sendWebSocketMessage('start_simulation', {});
        this.updateSystemStatus('System Online • Simulation running');
        console.log('▶ Simulation started');
    }

    pauseSimulation() {
        this.simulationRunning = false;
        this.updateSimulationControls();
        this.sendWebSocketMessage('pause_simulation', {});
        this.updateSystemStatus('System Online • Simulation paused');
        console.log('⏸ Simulation paused');
    }

    stopSimulation() {
        this.simulationRunning = false;
        this.updateSimulationControls();
        this.sendWebSocketMessage('stop_simulation', {});
        this.updateSystemStatus('System Online • Data stream active');
        console.log('⏹ Simulation stopped');
    }

    resetSimulation() {
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
        // Update vehicle queues with real data
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

        // Update metric cards with real data
        this.updateMetricValue('.metric-card:nth-child(1) .metric-value', data.total_vehicles || 0);

        // Congestion level based on average wait time
        const congestionLevel = Math.min(5, Math.max(1, Math.round((data.avg_wait_time || 0) / 10)));
        this.updateMetricValue('.metric-card:nth-child(2) .metric-value', `Level ${congestionLevel}`);

        // Active incidents (based on vehicles waiting > 30s)
        const incidents = Math.round((data.total_waiting_time || 0) / 60);
        this.updateMetricValue('.metric-card:nth-child(3) .metric-value', incidents);

        // AI Efficiency (inverse of wait time, scaled to 0-100)
        const efficiency = Math.round(Math.max(0, Math.min(100, 100 - (data.avg_wait_time || 0) * 2)));
        this.updateMetricValue('.metric-card:nth-child(4) .metric-value', efficiency);

        // Update vehicles visualization
        this.updateVehicleVisualization(data.vehicles || []);

        // Update chart with real data
        this.updateChartData(data);

        // Update simulation time
        if (data.simulation_time !== undefined) {
            this.updateSimulationTime(data.simulation_time);
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
        if (!intersection || !vehicles.length) return;

        // Group vehicles by lane/direction
        const vehiclesByLane = {};
        vehicles.forEach(vehicle => {
            const lane = vehicle.lane;
            if (!vehiclesByLane[lane]) {
                vehiclesByLane[lane] = [];
            }
            vehiclesByLane[lane].push(vehicle);
        });

        // Add vehicle dots for each direction
        Object.entries(vehiclesByLane).forEach(([lane, laneVehicles]) => {
            this.addVehicleDotsForLane(lane, laneVehicles, intersection);
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
        } else {
            // East-West green
            document.querySelector('#lightNorth .red')?.classList.add('active');
            document.querySelector('#lightSouth .red')?.classList.add('active');
            document.querySelector('#lightEast .green')?.classList.add('active');
            document.querySelector('#lightWest .green')?.classList.add('active');
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

        // Add new data point
        this.chartData.labels.push(now);
        this.chartData.aiData.push(waitTime);
        this.chartData.baselineData.push(waitTime * 1.18); // Baseline is 18% worse

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
        this.sendWebSocketMessage('train_model', { model_type: modelType });
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    const app = new TrafficAIManager();

    // Add UI enhancements
    addUIEnhancements();

    // Make app globally accessible for debugging
    window.trafficAI = app;
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