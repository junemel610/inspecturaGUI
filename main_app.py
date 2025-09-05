#!/usr/bin/env python3
"""
Main Wood Sorting Application

This is the main entry point for the wood sorting application.
Run this file to start the application.

Usage:
    python main_app.py [--dev-mode]

Arguments:
    --dev-mode: Run in development mode with simulated hardware
"""

import sys
import os
import argparse
from PyQt5.QtWidgets import QApplication

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description='Wood Sorting Application')
    parser.add_argument('--dev-mode', action='store_true', 
                       help='Run in development mode with simulated hardware')
    
    args = parser.parse_args()
    
    # Create QApplication
    app = QApplication(sys.argv)
    app.setApplicationName("Wood Sorting System")
    app.setApplicationVersion("1.0")
    
    try:
        # Import the working GUI module
        from modules.gui_module import WoodSortingApp
        
        # Create and show main application window
        main_window = WoodSortingApp(dev_mode=args.dev_mode)
        main_window.show()
        
        print(f"Wood Sorting Application started successfully (dev_mode={args.dev_mode})")
        
        # Run the application
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f"Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
