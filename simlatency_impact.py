import time
import random
import statistics
from dataclasses import dataclass
from typing import Callable, Any
import matplotlib.pyplot as plt

@dataclass
class LatencyConfig:
    """Configuration for latency simulation"""
    base_delay_ms: float = 100
    jitter_ms: float = 20
    distribution: str = 'normal'  # 'normal', 'uniform', 'spike'
    packet_loss_pct: float = 0
    correlation: float = 0  # 0-1, for bursty latency

class LatencySimulator:
    """Simulate network/processing latency"""
    
    def __init__(self, config: LatencyConfig):
        self.config = config
        self.last_delay = config.base_delay_ms
        self.latency_history = []
        
    def get_delay(self) -> float:
        """Generate realistic latency with jitter"""
        base = self.config.base_delay_ms
        
        if self.config.distribution == 'normal':
            jitter = random.gauss(0, self.config.jitter_ms)
        elif self.config.distribution == 'uniform':
            jitter = random.uniform(-self.config.jitter_ms, self.config.jitter_ms)
        else:  # spike - occasional large delays
            if random.random() < 0.05:  # 5% spike probability
                jitter = self.config.jitter_ms * 5
            else:
                jitter = random.gauss(0, self.config.jitter_ms/2)
        
        # Add correlation (bursty latency)
        if self.config.correlation > 0:
            jitter = (self.config.correlation * self.last_delay + 
                     (1 - self.config.correlation) * (base + jitter)) - base
            self.last_delay = base + jitter
        
        # Packet loss simulation
        if random.random() < self.config.packet_loss_pct / 100:
            return float('inf')  # Lost packet
            
        delay = max(0, base + jitter)  # No negative delays
        self.latency_history.append(delay)
        return delay
    
    def apply_latency(self, func: Callable, *args, **kwargs) -> Any:
        """Apply latency to a function call"""
        delay_ms = self.get_delay()
        
        if delay_ms == float('inf'):
            return None  # Packet lost
            
        time.sleep(delay_ms / 1000.0)
        return func(*args, **kwargs)
    
import numpy as np
from collections import deque

class DelayedControlSystem:
    """Simulate a control system with latency in feedback loop"""
    
    def __init__(self, sensor_delay_ms=50, actuator_delay_ms=50, process_delay_ms=10):
        self.sensor_delay_steps = max(1, int(sensor_delay_ms / 10))  # 10ms per step
        self.actuator_delay_steps = max(1, int(actuator_delay_ms / 10))
        self.process_delay_steps = max(1, int(process_delay_ms / 10))
        
        # Delay buffers
        self.sensor_buffer = deque([0.0] * self.sensor_delay_steps, maxlen=self.sensor_delay_steps)
        self.actuator_buffer = deque([0.0] * self.actuator_delay_steps, maxlen=self.actuator_delay_steps)
        
        # System state
        self.position = 0.0
        self.velocity = 0.0
        self.setpoint = 1.0
        
        # PID gains
        self.Kp = 1.0
        self.Ki = 0.1
        self.Kd = 0.05
        
        # PID state
        self.integral = 0.0
        self.prev_error = 0.0
        self.time = 0.0
        
    def process_dynamics(self, control_input, dt=0.01):
        """Simple mass-spring-damper system"""
        # Process delay
        for _ in range(self.process_delay_steps):
            mass = 1.0
            damping = 0.5
            spring = 0.1
            
            acceleration = (control_input - damping * self.velocity - spring * self.position) / mass
            self.velocity += acceleration * dt
            self.position += self.velocity * dt
            
        return self.position
    
    def update(self, dt=0.01):
        """Run one control cycle with latency"""
        # 1. Measure current position (with sensor delay)
        delayed_measurement = self.sensor_buffer[0]  # Get oldest measurement
        self.sensor_buffer.append(self.position)  # Store new measurement
        
        # 2. Compute control (using delayed measurement)
        error = self.setpoint - delayed_measurement
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        
        control = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        control = np.clip(control, -10, 10)  # Saturation
        
        # 3. Apply actuator delay
        self.actuator_buffer.append(control)
        delayed_control = self.actuator_buffer[0]
        
        # 4. Apply to process
        self.process_dynamics(delayed_control, dt)
        
        self.time += dt
        return self.position, delayed_measurement

# Run simulation with different latencies
def analyze_latency_impact():
    latencies = [0, 20, 50, 100, 200, 500]  # ms
    results = {}
    
    for latency in latencies:
        system = DelayedControlSystem(
            sensor_delay_ms=latency,
            actuator_delay_ms=latency
        )
        
        positions = []
        times = []
        
        for _ in range(1000):  # 10 seconds at 100Hz
            pos, _ = system.update()
            positions.append(pos)
            times.append(system.time)
            
        # Calculate settling time and overshoot
        overshoot = max(positions) - system.setpoint
        settling_idx = next((i for i, p in enumerate(positions[500:]) 
                           if abs(p - system.setpoint) < 0.05), len(positions))
        settling_time = times[settling_idx] if settling_idx < len(times) else float('inf')
        
        results[latency] = {
            'overshoot': overshoot,
            'settling_time': settling_time,
            'positions': positions
        }
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    for latency, data in results.items():
        if latency in [0, 50, 200]:  # Plot only selected latencies
            ax1.plot(times[:500], data['positions'][:500], 
                    label=f'{latency}ms latency')
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position')
    ax1.set_title('System Response with Latency')
    ax1.legend()
    ax1.grid(True)
    
    # Plot stability metrics
    latencies_list = list(results.keys())
    overshoots = [results[l]['overshoot'] for l in latencies_list]
    settling_times = [results[l]['settling_time'] for l in latencies_list]
    
    ax2.plot(latencies_list, overshoots, 'o-', label='Overshoot')
    ax2.plot(latencies_list, settling_times, 's-', label='Settling time')
    ax2.set_xlabel('Latency (ms)')
    ax2.set_ylabel('Metric')
    ax2.set_title('Latency Impact on Performance')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return results    

import asyncio
import time
from enum import Enum

class PacketType(Enum):
    DATA = 1
    ACK = 2
    NACK = 3

class NetworkNode:
    def __init__(self, node_id, latency_sim):
        self.node_id = node_id
        self.latency = latency_sim
        self.packets_sent = 0
        self.packets_received = 0
        self.packets_lost = 0
        
    async def send_packet(self, dest_node, packet_type, data=None):
        """Send packet with simulated network conditions"""
        self.packets_sent += 1
        
        # Apply latency and possible loss
        delay_ms = self.latency.get_delay()
        
        if delay_ms == float('inf'):
            self.packets_lost += 1
            print(f"Packet from Node {self.node_id} to Node {dest_node.node_id} LOST!")
            return False
        
        await asyncio.sleep(delay_ms / 1000.0)
        
        # Deliver packet
        await dest_node.receive_packet(self.node_id, packet_type, data)
        return True
    
    async def receive_packet(self, src_id, packet_type, data):
        """Handle received packet"""
        self.packets_received += 1
        print(f"Node {self.node_id} received {packet_type.name} from Node {src_id}")
        
        # Auto-respond to DATA packets with ACK
        if packet_type == PacketType.DATA:
            await self.send_packet(
                self,  # Hack: using self as dest for simplicity
                PacketType.ACK,
                f"ACK for {data}"
            )

class ReliableProtocol:
    """Simple reliable protocol with retransmission"""
    
    def __init__(self, node, timeout_ms=500, max_retries=3):
        self.node = node
        self.timeout = timeout_ms / 1000.0
        self.max_retries = max_retries
        
    async def send_reliable(self, dest_node, data):
        """Send data with retransmission on timeout/loss"""
        for attempt in range(self.max_retries):
            print(f"Attempt {attempt + 1} to send: {data}")
            
            # Send packet
            success = await self.node.send_packet(dest_node, PacketType.DATA, data)
            
            if not success:
                print(f"Packet lost, retrying...")
                continue
            
            # Wait for ACK (simulated)
            await asyncio.sleep(self.timeout)
            
            # In real implementation, would check if ACK received
            # Here we'll simulate ACK success with probability
            
            if random.random() < 0.7:  # 70% success rate
                print(f"✓ Data delivered successfully!")
                return True
            else:
                print(f"✗ No ACK received, retrying...")
                
        print(f"✗✗ Failed to deliver after {self.max_retries} attempts")
        return False

async def run_network_simulation():
    # Create nodes with different latency profiles
    config1 = LatencyConfig(base_delay_ms=50, jitter_ms=10, packet_loss_pct=5)
    config2 = LatencyConfig(base_delay_ms=200, jitter_ms=50, packet_loss_pct=10)
    
    node1 = NetworkNode(1, LatencySimulator(config1))
    node2 = NetworkNode(2, LatencySimulator(config2))
    
    protocol = ReliableProtocol(node1)
    
    # Send some data
    await protocol.send_reliable(node2, "Hello, Node 2!")
    
    # Statistics
    print(f"\n=== Network Statistics ===")
    print(f"Node 1 - Sent: {node1.packets_sent}, Lost: {node1.packets_lost}")
    print(f"Node 2 - Received: {node2.packets_received}")
import time
import threading
from collections import deque
import psutil  # pip install psutil

class RealTimeLatencyMonitor:
    """Monitor actual execution latency in real-time"""
    
    def __init__(self, target_rate_hz=100):
        self.target_interval = 1.0 / target_rate_hz
        self.latencies = deque(maxlen=1000)
        self.overruns = 0
        self.running = False
        self.thread = None
        
    def start(self):
        """Start monitoring in background thread"""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
            
    def _monitor_loop(self):
        """Monitor loop execution timing"""
        next_time = time.perf_counter()
        
        while self.running:
            current_time = time.perf_counter()
            actual_interval = current_time - next_time + self.target_interval
            
            self.latencies.append(actual_interval * 1000)  # Convert to ms
            
            if actual_interval > self.target_interval * 1.1:  # 10% overrun
                self.overruns += 1
                print(f"⚠ Overrun: {actual_interval*1000:.2f}ms (target: {self.target_interval*1000:.2f}ms)")
            
            # Do some work (simulated)
            self._do_work()
            
            # Schedule next iteration
            next_time += self.target_interval
            sleep_time = max(0, next_time - time.perf_counter())
            time.sleep(sleep_time)
    
    def _do_work(self):
        """Simulate varying workload"""
        # Random work duration between 0.5ms and 15ms
        work_duration = random.uniform(0.0005, 0.015)
        start = time.perf_counter()
        while time.perf_counter() - start < work_duration:
            # Busy wait (simulate CPU work)
            _ = [i**2 for i in range(100)]
    
    def get_stats(self):
        """Get latency statistics"""
        if not self.latencies:
            return {}
        
        return {
            'min_ms': min(self.latencies),
            'max_ms': max(self.latencies),
            'mean_ms': statistics.mean(self.latencies),
            'median_ms': statistics.median(self.latencies),
            'p95_ms': np.percentile(list(self.latencies), 95),
            'p99_ms': np.percentile(list(self.latencies), 99),
            'overruns': self.overruns,
            'cpu_percent': psutil.cpu_percent()
        }

# Run monitor
monitor = RealTimeLatencyMonitor(target_rate_hz=100)  # 10ms target
monitor.start()

try:
    time.sleep(10)  # Run for 10 seconds
finally:
    monitor.stop()
    stats = monitor.get_stats()
    print("\n=== Latency Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value:.3f}" if isinstance(value, float) else f"{key}: {value}")    
def monte_carlo_latency_analysis(num_runs=100):
    """Run multiple simulations with random latency profiles"""
    
    results = {
        'latency_ms': [],
        'success_rate': [],
        'avg_response_time': []
    }
    
    for run in range(num_runs):
        # Random latency profile
        config = LatencyConfig(
            base_delay_ms=random.uniform(10, 500),
            jitter_ms=random.uniform(0, 100),
            packet_loss_pct=random.uniform(0, 20),
            correlation=random.uniform(0, 0.8)
        )
        
        simulator = LatencySimulator(config)
        
        # Run test transactions
        successes = 0
        total_time = 0
        num_transactions = 50
        
        for _ in range(num_transactions):
            start = time.perf_counter()
            
            # Simulate a transaction
            result = simulator.apply_latency(lambda x: x**2, 5)
            
            if result is not None:
                successes += 1
                total_time += (time.perf_counter() - start) * 1000
        
        results['latency_ms'].append(config.base_delay_ms)
        results['success_rate'].append(successes / num_transactions)
        results['avg_response_time'].append(total_time / max(1, successes))
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.scatter(results['latency_ms'], results['success_rate'], alpha=0.5)
    ax1.set_xlabel('Base Latency (ms)')
    ax1.set_ylabel('Success Rate')
    ax1.set_title('Latency vs Success Rate')
    ax1.grid(True)
    
    ax2.scatter(results['latency_ms'], results['avg_response_time'], alpha=0.5, color='orange')
    ax2.set_xlabel('Base Latency (ms)')
    ax2.set_ylabel('Avg Response Time (ms)')
    ax2.set_title('Latency vs Response Time')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return results        