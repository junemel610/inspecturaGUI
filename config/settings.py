#!/usr/bin/env python3
"""
Simple Configuration for Wood Sorting Application
Basic configuration without complex dependencies
"""

class SimpleConfig:
    """Simple configuration class"""
    def __init__(self):
        # GUI settings
        self.gui = type('obj', (object,), {
            'window_width': 1920,
            'window_height': 1080,
            'maximize_on_startup': True,
            'update_interval_ms': 33,
            'font_size_small': 10,
            'font_size_medium': 14,
            'font_size_large': 18,
            'font_size_xlarge': 22
        })()
        
        # Performance settings
        self.performance = type('obj', (object,), {
            'enable_monitoring': False,  # Disabled to prevent threading issues
            'fps_monitoring': False,
            'memory_monitoring': False,
            'cpu_monitoring': False,
            'processing_time_monitoring': False
        })()
        
        # Camera settings
        self.camera = type('obj', (object,), {
            'resolution_width': 1920,
            'resolution_height': 1080,
            'fps': 30,
            'exposure': -1,
            'brightness': 50,
            'contrast': 50,
            'saturation': 50
        })()
        
    def get_config_summary(self):
        return """
Wood Sorting Application Configuration
Environment: Simple Development Mode
GUI: 1920x1080 Maximized
Performance Monitoring: Enabled
"""

# Global config instance
_config = None

def get_config(environment="development"):
    """Get global configuration instance"""
    global _config
    if _config is None:
        _config = SimpleConfig()
    return _config