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

        # Alignment settings
        self.alignment = type('obj', (object,), {
            'top_roi_margin_percent': 0.15,  # 15% of frame height for top ROI
            'bottom_roi_margin_percent': 0.15,  # 15% of frame height for bottom ROI
            'min_overlap_threshold': 0.6,  # 60% overlap required for alignment
            'alignment_tolerance_percent': 0.1,  # 10% tolerance for alignment checks
            'enable_alignment_visualization': True,
            'roi_display_color': (255, 255, 0),  # Yellow for ROI display
            'aligned_color': (0, 255, 0),  # Green for aligned wood
            'misaligned_color': (0, 0, 255)  # Red for misaligned wood
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