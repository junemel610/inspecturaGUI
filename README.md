# Wood Sorting Application

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Usage](#usage)
6. [Modules](#modules)
7. [Performance Monitoring](#performance-monitoring)
8. [Troubleshooting](#troubleshooting)
9. [API Reference](#api-reference)
10. [Contributing](#contributing)

## Overview


The Wood Sorting Application is a modular PyQt5-based system for automated wood quality assessment and sorting, compliant with the SS-EN 1611-1 European standard. It provides real-time defect detection, grading, and sorting with integrated camera systems and Arduino-based hardware control. The application is designed for both development (with simulated hardware) and production environments.


### Key Features

- **Two-Stage Detection**: Wood detection first, then defect analysis for improved accuracy
- **Real-time Wood Detection**: Computer vision algorithms for defect identification
- **SS-EN 1611-1 Compliance**: Automated grading logic
- **Multi-Camera Support**: Dual camera system for comprehensive analysis
- **Arduino Integration**: Hardware control for sorting and sensors
- **Performance Monitoring**: Real-time system metrics
- **Configuration Management**: SimpleConfig-based environment selection
- **Reporting**: TXT and PDF report generation, detection frame saving
- **Robust Error Handling**: Centralized logging and enhanced error handler


### System Requirements

- **Operating System**: Windows 10/11, Linux, macOS
- **Python**: 3.8 or higher
- **Hardware**:
   - Dual USB cameras
   - Arduino-compatible microcontroller
   - IR sensors
   - Sorting mechanism hardware
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 2GB+ free space

## Architecture


### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    PyQt5 GUI Layer                         │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐│
│  │ Camera View │ │ Defect Panel│ │ Control & Statistics    ││
│  └─────────────┘ └─────────────┘ └─────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                    Core Modules                            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐│
│  │Camera Module│ │Detection Mod│ │ Grading Module          ││
│  └─────────────┘ └─────────────┘ └─────────────────────────┘│
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐│
│  │Arduino Mod  │ │Utils Module │ │ Reporting Module        ││
│  └─────────────┘ └─────────────┘ └─────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                Infrastructure Layer                        │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐│
│  │Config System│ │Perf Monitor │ │ Error Handler           ││
│  └─────────────┘ └─────────────┘ └─────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```


## Modules

### Detection Workflow

The detection system uses a two-stage approach for improved accuracy:

1. **Stage 1 - Wood Detection**: Uses the `Wood_Plank--640x640_quant_hailort_hailo8_2` model to detect and locate wood planks in the frame
2. **Stage 2 - Defect Detection**: Uses the `Defect_Detection--640x640_quant_hailort_hailo8_1` model to analyze the identified wood region for defects

This approach reduces false positives by only running defect detection on confirmed wood regions.

### Module Descriptions

- **main_app.py**: Main entry point. Launches the GUI and sets up the application.
- **modules/gui_module.py** and **modules/gui_module_clean.py**: Main PyQt5 GUI application (use `gui_module.py` by default).
- **modules/camera_module.py**: Handles multi-camera management and frame capture.
- **modules/detection_module.py**: Two-stage computer vision (wood detection → defect detection) using DeGirum models.
- **modules/grading_module.py**: Implements SS-EN 1611-1 grading logic.
- **modules/arduino_module.py**: Handles Arduino communication and hardware control.
- **modules/reporting_module.py**: Generates TXT/PDF reports and saves detection frames.
- **modules/performance_monitor.py**: Real-time performance monitoring (FPS, memory, CPU, timing).
- **modules/error_handler.py** and **modules/enhanced_error_handler.py**: Centralized and advanced error handling.
- **modules/utils_module.py**: Utility functions, calibration, and grading thresholds.
- **config/settings.py**: SimpleConfig class for environment and settings management.

## Installation


### Prerequisites

1. Install Python 3.8 or higher
2. Install required packages:

```bash
pip install -r requirements.txt
```

### Application Setup

1. Clone or download the application to your desired directory
2. Navigate to the application directory:

```bash
cd wood_sorting_app
```

3. Create necessary directories (if not present):

```bash
mkdir logs reports
```

### Hardware Setup

1. Connect two USB cameras and ensure they are recognized by your OS.
2. Connect Arduino via USB and upload your sketch.
3. Connect IR sensors and sorting mechanism to Arduino.

### Model Files

The application uses two DeGirum AI models stored in the `models/` directory:

- **Wood_Plank--640x640_quant_hailort_hailo8_2**: For wood detection and localization
- **Defect_Detection--640x640_quant_hailort_hailo8_1**: For defect detection within wood regions

Each model directory contains:
- `.hef` file: The compiled model for Hailo hardware
- `.json` file: Model metadata and configuration
- `labels_*.json` file: Class labels for the model outputs

## Configuration


The application uses a simple configuration system in `config/settings.py` (`SimpleConfig` class). You can select the environment (development/production) and adjust GUI, performance, and camera settings in Python code.

**To modify configuration:**
1. Edit `config/settings.py` as needed.
2. Restart the application to apply changes.

## Usage


### Starting the Application

#### Development Mode (simulated hardware)
```bash
python main_app.py --dev-mode
```

#### Production Mode
```bash
python main_app.py
```

### User Interface

#### Main Window Layout

1. **Camera View Section** (Left, 70% width):
   - Top Camera View: Primary wood piece inspection
   - Bottom Camera View: Secondary angle inspection

2. **Live Defect Analysis Panel** (Right, 30% width):
   - Real-time defect detection results
   - Defect size measurements
   - Visual progress indicators

3. **Control Buttons** (Bottom):
   - IDLE: System standby mode
   - TRIGGER: Single piece analysis mode  
   - CONTINUOUS: Automated continuous processing

4. **Statistics Tabs** (Bottom Right):
   - System Health: Component status monitoring
   - Error Monitoring: System error tracking
   - Defect Details: Detailed defect analysis
   - Performance: Real-time performance metrics
   - Recent Activity: System activity log

#### Operating Modes

**IDLE Mode**:
- System in standby
- Cameras active for preview
- Arduino in idle state
- Checkboxes enabled but unchecked

**TRIGGER Mode**:
- Single piece processing
- Manual trigger activation
- Detailed analysis and reporting

**CONTINUOUS Mode**:
- Automated processing
- IR sensor triggering
- Real-time wood piece analysis
- Automatic sorting decisions

### Checkbox Features

- **Live Mode**: Real-time processing display
- **Auto Grading**: Automatic SS-EN 1611-1 grading
- **Auto Mode**: Automated checkbox behavior based on operating mode


## Testing

*No dedicated test scripts or test directory found in this version. Add your own tests as needed.*

## Performance Monitoring

The application includes real-time performance monitoring capabilities.

### Performance Metrics

- **Frame Rate**: Real-time FPS monitoring
- **Memory Usage**: System memory consumption tracking
- **CPU Usage**: Processor utilization monitoring  
- **Processing Time**: Component-specific timing analysis

### Performance Display

Real-time performance metrics are displayed in the Performance tab:

```
Real-Time Performance Metrics
=================================

Frame Rate: 29.8 FPS
Memory Usage: 245.6 MB
CPU Usage: 15.2%
Processing Time: 23.4 ms

Component Timing:
• Detection: 18.2 ms
• Arduino: 1.1 ms
• GUI Updates: 4.1 ms

System Status: OPTIMAL
```

### Performance Optimization

- Monitor frame rates to ensure > 25 FPS for smooth operation
- Keep memory usage under 500MB for optimal performance
- CPU usage should remain under 80% for system stability
- Processing time should be < 50ms per cycle

## Troubleshooting

### Common Issues

#### Camera Issues

**Problem**: No cameras detected
**Solution**: 
1. Check USB connections
2. Verify camera drivers are installed
3. Test cameras with other applications
4. Check configuration file camera settings

**Problem**: Poor image quality
**Solution**:
1. Adjust camera parameters in configuration
2. Check lighting conditions
3. Clean camera lenses
4. Verify camera positioning

#### Arduino Issues

**Problem**: Arduino not connecting
**Solution**:
1. Check USB cable and connection
2. Verify correct COM port in configuration
3. Ensure Arduino sketch is uploaded
4. Check Arduino IDE serial monitor

**Problem**: Sensor readings incorrect
**Solution**:
1. Check sensor wiring
2. Verify sensor power supply
3. Test sensors individually
4. Check Arduino code logic

#### Performance Issues

**Problem**: Low frame rate
**Solution**:
1. Check system resources
2. Close unnecessary applications
3. Reduce camera resolution if needed
4. Optimize detection algorithms

**Problem**: High memory usage
**Solution**:
1. Monitor memory leaks in Performance tab
2. Restart application periodically
3. Check for memory-intensive operations
4. Review detection module memory usage

### Error Logs


Application logs are stored in the `logs/` and `wood_sorting_app/logs/` directories.

### Debug Mode

Enable debug mode for detailed logging:
1. Set log level to "DEBUG" in configuration
2. Monitor error monitoring tab for real-time issues
3. Check console output for immediate feedback

## API Reference

detection_config = config.detection
arduino_config = config.arduino

### Configuration API

#### SimpleConfig

```python
from config.settings import get_config

config = get_config(environment="development")
camera_config = config.camera
gui_config = config.gui
```

### Performance Monitoring API

def my_callback(metrics):

#### PerformanceMonitor

```python
from modules.performance_monitor import get_performance_monitor

monitor = get_performance_monitor()
monitor.start_monitoring()
metrics = monitor.get_performance_summary()
```

### Detection API

detector = DetectionModule()
detector.set_roi(100, 100, 500, 400)
defects = detector.detect_defects(image)

#### DetectionModule

```python
from modules.detection_module import DetectionModule

detector = DetectionModule()

# Two-stage detection
wood_detected, confidence, bbox = detector.detect_wood_presence(frame)
if wood_detected:
    annotated_frame, defects, measurements = detector.detect_defects_in_wood_region(frame, bbox)

# Or use the combined analysis
annotated_frame, defects, measurements = detector.analyze_frame(frame)
```

### Grading API

grader = GradingModule()
grade_result = grader.grade_wood_piece(defects)
grade = grade_result['grade']

#### GradingModule

```python
from modules.grading_module import determine_surface_grade

grade = determine_surface_grade(defect_measurements)
```

## Raspberry Pi Migration

To migrate your application to a Raspberry Pi:

1. **Transfer Files**: Copy your entire project directory (including `models/`) to the Raspberry Pi:
   ```bash
   scp -r wood_sorting_app/ pi@your-pi-ip:~/
   ```

2. **Setup Python Environment**: On the Raspberry Pi:
   ```bash
   cd wood_sorting_app
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
   **Note**: DeGirum libraries may need to be installed separately based on your Hailo hardware setup.

4. **Hardware Considerations**:
   - Ensure USB cameras are compatible with the Pi
   - Check Arduino USB permissions: `sudo usermod -a -G dialout $USER`
   - Install camera drivers if needed: `sudo apt-get install v4l-utils`

5. **Test the Installation**:
   ```bash
   python main_app.py --dev-mode  # Test without hardware first
   python main_app.py             # Test with hardware
   ```

## Contributing

### Development Setup


1. Fork the repository
2. Create a development branch
3. Set up your environment and run:

```bash
python main_app.py --dev-mode
```

### Code Standards

- Follow PEP 8 Python style guidelines
- Add comprehensive docstrings to all functions and classes
- Include unit tests for new functionality
- Update documentation for any API changes

### Adding New Features

1. Create feature branch from main
2. Implement feature with tests
3. Update configuration if needed
4. Add documentation
5. Submit pull request

### Testing Requirements


- Add tests for new code where possible

### Documentation Updates

- Update this README for major changes
- Add inline code documentation
- Update configuration documentation
- Include usage examples

---


For additional support or questions, refer to the project documentation or contact the development team.
