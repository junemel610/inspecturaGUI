#!/usr/bin/env python3
"""
Performance Monitoring Module for Wood Sorting Application

This module provides real-time performance monitoring capabilities
including FPS, memory usage, CPU usage, and processing time tracking.
"""

import time
import psutil
import threading
from typing import Dict, List, Optional, Callable
from collections import deque
from dataclasses import dataclass
import logging

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    timestamp: float
    fps: float
    memory_usage_mb: float
    cpu_usage_percent: float
    processing_time_ms: float
    detection_time_ms: float = 0.0
    arduino_time_ms: float = 0.0
    gui_update_time_ms: float = 0.0

class PerformanceMonitor:
    """Real-time performance monitoring system"""
    
    def __init__(self, history_size: int = 100, update_interval: float = 1.0):
        """
        Initialize performance monitor
        
        Args:
            history_size: Number of metrics to keep in history
            update_interval: Update interval in seconds
        """
        self.history_size = history_size
        self.update_interval = update_interval
        
        # Performance metrics storage
        self.metrics_history: deque = deque(maxlen=history_size)
        self.current_metrics = PerformanceMetrics(
            timestamp=time.time(),
            fps=0.0,
            memory_usage_mb=0.0,
            cpu_usage_percent=0.0,
            processing_time_ms=0.0
        )
        
        # Frame rate tracking
        self.frame_times: deque = deque(maxlen=30)  # Last 30 frames for FPS calculation
        self.last_frame_time = time.time()
        
        # Processing time tracking
        self.processing_start_time = None
        self.processing_times: deque = deque(maxlen=10)  # Last 10 processing times
        
        # Component timing
        self.component_times: Dict[str, deque] = {
            'detection': deque(maxlen=10),
            'arduino': deque(maxlen=10),
            'gui_update': deque(maxlen=10)
        }
        
        # Monitoring state
        self.monitoring_enabled = True
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        
        # Callbacks for real-time updates
        self.update_callbacks: List[Callable] = []
        
        # Process object for system metrics
        self.process = psutil.Process()
        
    def start_monitoring(self):
        """Start background performance monitoring"""
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            self.stop_monitoring.clear()
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            logging.info("Performance monitoring started")
            
    def stop_monitoring_thread(self):
        """Stop background performance monitoring"""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.stop_monitoring.set()
            self.monitoring_thread.join(timeout=2.0)
            logging.info("Performance monitoring stopped")
            
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while not self.stop_monitoring.wait(self.update_interval):
            if self.monitoring_enabled:
                self._update_system_metrics()
                self._notify_callbacks()
                
    def _update_system_metrics(self):
        """Update system-level performance metrics"""
        try:
            # Memory usage
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            # CPU usage
            cpu_percent = self.process.cpu_percent()
            
            # Calculate average processing time
            avg_processing_time = (
                sum(self.processing_times) / len(self.processing_times)
                if self.processing_times else 0.0
            )
            
            # Calculate average component times
            avg_detection_time = (
                sum(self.component_times['detection']) / len(self.component_times['detection'])
                if self.component_times['detection'] else 0.0
            )
            
            avg_arduino_time = (
                sum(self.component_times['arduino']) / len(self.component_times['arduino'])
                if self.component_times['arduino'] else 0.0
            )
            
            avg_gui_time = (
                sum(self.component_times['gui_update']) / len(self.component_times['gui_update'])
                if self.component_times['gui_update'] else 0.0
            )
            
            # Update current metrics
            self.current_metrics = PerformanceMetrics(
                timestamp=time.time(),
                fps=self.get_current_fps(),
                memory_usage_mb=memory_mb,
                cpu_usage_percent=cpu_percent,
                processing_time_ms=avg_processing_time,
                detection_time_ms=avg_detection_time,
                arduino_time_ms=avg_arduino_time,
                gui_update_time_ms=avg_gui_time
            )
            
            # Add to history
            self.metrics_history.append(self.current_metrics)
            
        except Exception as e:
            logging.error(f"Error updating system metrics: {e}")
            
    def update_frame_rate(self):
        """Update frame rate calculation (call on each frame)"""
        current_time = time.time()
        frame_time = current_time - self.last_frame_time
        self.frame_times.append(frame_time)
        self.last_frame_time = current_time
        
    def get_current_fps(self) -> float:
        """Get current FPS based on recent frame times"""
        if len(self.frame_times) < 2:
            return 0.0
            
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
        
    def start_processing_timer(self):
        """Start timing a processing operation"""
        self.processing_start_time = time.time()
        
    def end_processing_timer(self):
        """End timing a processing operation"""
        if self.processing_start_time is not None:
            processing_time = (time.time() - self.processing_start_time) * 1000  # Convert to ms
            self.processing_times.append(processing_time)
            self.processing_start_time = None
            return processing_time
        return 0.0
        
    def start_component_timer(self, component: str) -> float:
        """Start timing a specific component"""
        return time.time()
        
    def end_component_timer(self, component: str, start_time: float):
        """End timing a specific component"""
        if component in self.component_times:
            elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
            self.component_times[component].append(elapsed_time)
            
    def get_performance_summary(self) -> Dict[str, float]:
        """Get current performance summary"""
        return {
            'fps': self.current_metrics.fps,
            'memory_mb': self.current_metrics.memory_usage_mb,
            'cpu_percent': self.current_metrics.cpu_usage_percent,
            'processing_time_ms': self.current_metrics.processing_time_ms,
            'detection_time_ms': self.current_metrics.detection_time_ms,
            'arduino_time_ms': self.current_metrics.arduino_time_ms,
            'gui_update_time_ms': self.current_metrics.gui_update_time_ms
        }
        
    def get_performance_history(self, minutes: int = 5) -> List[PerformanceMetrics]:
        """Get performance history for the last N minutes"""
        cutoff_time = time.time() - (minutes * 60)
        return [
            metric for metric in self.metrics_history
            if metric.timestamp >= cutoff_time
        ]
        
    def add_update_callback(self, callback: Callable):
        """Add callback for performance updates"""
        self.update_callbacks.append(callback)
        
    def remove_update_callback(self, callback: Callable):
        """Remove callback for performance updates"""
        if callback in self.update_callbacks:
            self.update_callbacks.remove(callback)
            
    def _notify_callbacks(self):
        """Notify all registered callbacks of performance updates"""
        for callback in self.update_callbacks:
            try:
                callback(self.current_metrics)
            except Exception as e:
                logging.error(f"Error in performance callback: {e}")
                
    def reset_metrics(self):
        """Reset all performance metrics"""
        self.metrics_history.clear()
        self.frame_times.clear()
        self.processing_times.clear()
        for component_times in self.component_times.values():
            component_times.clear()
            
    def enable_monitoring(self):
        """Enable performance monitoring"""
        self.monitoring_enabled = True
        
    def disable_monitoring(self):
        """Disable performance monitoring"""
        self.monitoring_enabled = False
        
    def get_performance_report(self) -> str:
        """Generate detailed performance report"""
        if not self.metrics_history:
            return "No performance data available"
            
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 measurements
        
        # Calculate averages
        avg_fps = sum(m.fps for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics)
        avg_cpu = sum(m.cpu_usage_percent for m in recent_metrics) / len(recent_metrics)
        avg_processing = sum(m.processing_time_ms for m in recent_metrics) / len(recent_metrics)
        
        # Calculate max values
        max_memory = max(m.memory_usage_mb for m in recent_metrics)
        max_cpu = max(m.cpu_usage_percent for m in recent_metrics)
        max_processing = max(m.processing_time_ms for m in recent_metrics)
        
        return f"""
Wood Sorting Application Performance Report
==========================================

Current Performance:
- FPS: {self.current_metrics.fps:.1f}
- Memory Usage: {self.current_metrics.memory_usage_mb:.1f} MB
- CPU Usage: {self.current_metrics.cpu_usage_percent:.1f}%
- Processing Time: {self.current_metrics.processing_time_ms:.1f} ms

Recent Averages (last 10 measurements):
- Average FPS: {avg_fps:.1f}
- Average Memory: {avg_memory:.1f} MB
- Average CPU: {avg_cpu:.1f}%
- Average Processing Time: {avg_processing:.1f} ms

Peak Values:
- Peak Memory: {max_memory:.1f} MB
- Peak CPU: {max_cpu:.1f}%
- Peak Processing Time: {max_processing:.1f} ms

Component Timing:
- Detection: {self.current_metrics.detection_time_ms:.1f} ms
- Arduino Communication: {self.current_metrics.arduino_time_ms:.1f} ms
- GUI Updates: {self.current_metrics.gui_update_time_ms:.1f} ms

History: {len(self.metrics_history)} measurements stored
"""

# Global performance monitor instance
performance_monitor = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance"""
    global performance_monitor
    if performance_monitor is None:
        performance_monitor = PerformanceMonitor()
    return performance_monitor

def start_performance_monitoring():
    """Start global performance monitoring"""
    monitor = get_performance_monitor()
    monitor.start_monitoring()
    
def stop_performance_monitoring():
    """Stop global performance monitoring"""
    global performance_monitor
    if performance_monitor:
        performance_monitor.stop_monitoring_thread()
