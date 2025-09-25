# Wood Detection Program Documentation

## Overview

The `detection.py` program is a comprehensive wood sorting application that utilizes computer vision and artificial intelligence to automatically detect and grade defects in wood pieces according to the SS-EN 1611-1 European wood grading standard. The application provides a real-time GUI interface for monitoring and controlling the wood sorting process, integrating with hardware components including cameras, Arduino microcontrollers, and conveyor systems.

### Key Features

- **Dual Camera System**: Top and bottom camera views for comprehensive wood inspection
- **AI-Powered Defect Detection**: Uses DeGirum AI models to identify wood defects
- **SS-EN 1611-1 Compliance**: Implements European wood grading standards for accurate quality assessment
- **Real-time Processing**: Live video feed analysis with immediate grading results
- **Hardware Integration**: Arduino-based conveyor control and sorting mechanisms
- **Multiple Operation Modes**: IDLE, TRIGGER, and CONTINUOUS modes for different operational needs
- **Comprehensive Reporting**: Generates detailed PDF and text reports with defect analysis
- **Calibration System**: Pixel-to-millimeter conversion for accurate defect size measurement

### System Architecture

The application is built using Python with the following main components:

- **Tkinter GUI**: User interface for real-time monitoring and control
- **OpenCV**: Camera handling and image processing
- **DeGirum AI Framework**: Neural network inference for defect detection
- **Serial Communication**: Arduino integration for hardware control
- **ReportLab**: PDF report generation
- **Threading**: Background processing for smooth UI operation

### Operation Modes

1. **IDLE Mode**: System disabled, no operations
2. **TRIGGER Mode**: Waits for IR beam trigger to start detection
3. **CONTINUOUS Mode**: Live detection with automatic grading

## Configuration Constants

### Camera Settings
- `CAMERA_WIDTH = 1280` - Camera capture width in pixels
- `CAMERA_HEIGHT = 720` - Camera capture height in pixels
- `CAMERA_FPS = 30` - Camera frame rate

### Calibration Settings
- `TOP_CAMERA_DISTANCE_CM = 37` - Distance of top camera from wood surface
- `BOTTOM_CAMERA_DISTANCE_CM = 29` - Distance of bottom camera from wood surface
- `TOP_CAMERA_PIXEL_TO_MM = 0.4` - mm per pixel for top camera
- `BOTTOM_CAMERA_PIXEL_TO_MM = 0.3` - mm per pixel for bottom camera
- `WOOD_PALLET_WIDTH_MM = 115` - Actual width of wood pallet in mm

### Grading Standards (SS-EN 1611-1)
- `GRADE_G2_0 = "G2-0"` - Perfect grade
- `GRADE_G2_1 = "G2-1"` - Good grade
- `GRADE_G2_2 = "G2-2"` - Fair grade
- `GRADE_G2_3 = "G2-3"` - Lower fair grade
- `GRADE_G2_4 = "G2-4"` - Poor grade

## Function Documentation

### Core Application Class: App

#### `__init__(self)`
**Purpose**: Initializes the wood sorting application GUI and sets up all necessary components.

**Parameters**: None

**Return Value**: None

**Details**: Creates the main Tkinter window, initializes cameras, loads the DeGirum AI model, sets up Arduino communication, and configures the user interface with camera feeds, controls, and statistics panels.

---

### Calibration Functions

#### `calibrate_pixel_to_mm(self, reference_object_width_px, reference_object_width_mm, camera_name="top")`
**Purpose**: Calibrates the pixel-to-millimeter conversion factor for accurate defect size measurement.

**Parameters**:
- `reference_object_width_px` (float): Width of reference object in pixels
- `reference_object_width_mm` (float): Actual width of reference object in millimeters
- `camera_name` (str): Camera to calibrate ("top" or "bottom")

**Return Value**: float - The calculated conversion factor (mm per pixel)

**Details**: Updates the global pixel-to-millimeter conversion constants based on known reference measurements.

#### `calibrate_with_wood_pallet(self, wood_pallet_width_px_top, wood_pallet_width_px_bottom)`
**Purpose**: Auto-calibrates both cameras using the known wood pallet width.

**Parameters**:
- `wood_pallet_width_px_top` (float): Wood pallet width in pixels (top camera)
- `wood_pallet_width_px_bottom` (float): Wood pallet width in pixels (bottom camera)

**Return Value**: tuple - (top_factor, bottom_factor) conversion factors

**Details**: Uses the standard wood pallet width to calibrate both cameras simultaneously.

---

### AI Model and Detection Functions

#### `map_model_output_to_standard(self, model_label)`
**Purpose**: Maps the AI model's defect labels to standardized defect categories.

**Parameters**:
- `model_label` (str): Raw label output from the AI model

**Return Value**: str - Standardized defect type ("Sound_Knot" or "Unsound_Knot")

**Details**: Normalizes various model output formats to consistent defect categories for grading.

#### `calculate_defect_size(self, detection_box, camera_name="top")`
**Purpose**: Calculates defect size in millimeters and percentage from bounding box coordinates.

**Parameters**:
- `detection_box` (dict): Detection bounding box with 'bbox' key containing [x1, y1, x2, y2]
- `camera_name` (str): Camera name for appropriate calibration factor

**Return Value**: tuple - (size_mm, percentage) where size_mm is defect size in mm and percentage is defect size as percentage of wood width

**Details**: Uses camera-specific pixel-to-mm conversion and wood pallet width for accurate measurements.

#### `analyze_frame(self, frame, camera_name="top", run_defect_model=True)`
**Purpose**: Analyzes a camera frame using the DeGirum AI model to detect defects.

**Parameters**:
- `frame` (numpy.ndarray): Input camera frame
- `camera_name` (str): Name of camera ("top" or "bottom")
- `run_defect_model` (bool): Whether to run AI inference

**Return Value**: tuple - (annotated_frame, defect_dict, defect_measurements) where:
- `annotated_frame`: Frame with detection overlays
- `defect_dict`: Dictionary of defect counts by type
- `defect_measurements`: List of (defect_type, size_mm, percentage) tuples

**Details**: Runs AI inference, processes detections, and calculates defect sizes and measurements.

---

### Grading Functions

#### `grade_individual_defect(self, defect_type, size_mm, percentage)`
**Purpose**: Grades an individual defect based on SS-EN 1611-1 standards.

**Parameters**:
- `defect_type` (str): Type of defect ("Sound_Knot" or "Unsound_Knot")
- `size_mm` (float): Defect size in millimeters
- `percentage` (float): Defect size as percentage of wood width

**Return Value**: str - Grade string (e.g., "G2-0", "G2-1", etc.)

**Details**: Applies SS-EN 1611-1 threshold rules where defect is assigned the worst grade it exceeds.

#### `determine_surface_grade(self, defect_measurements)`
**Purpose**: Determines overall grade for a single surface based on all detected defects.

**Parameters**:
- `defect_measurements` (list): List of (defect_type, size_mm, percentage) tuples

**Return Value**: str - Surface grade according to SS-EN 1611-1

**Details**: Applies defect count limitations and individual defect grade hierarchy to determine surface quality.

#### `determine_final_grade(self, top_grade, bottom_grade)`
**Purpose**: Determines final wood piece grade based on worst surface grade.

**Parameters**:
- `top_grade` (str): Grade of top surface
- `bottom_grade` (str): Grade of bottom surface

**Return Value**: str - Final grade (worst of top/bottom surfaces)

**Details**: Returns the worse grade between top and bottom surfaces per SS-EN 1611-1 standard.

#### `convert_grade_to_arduino_command(self, standard_grade)`
**Purpose**: Converts SS-EN 1611-1 grade to Arduino sorting command.

**Parameters**:
- `standard_grade` (str): Standard grade (G2-0 through G2-4)

**Return Value**: int - Arduino command (1=Good gate, 2=Fair gate, 3=Poor gate)

**Details**: Maps 5 standard grades to 3 sorting gates: Good (G2-0), Fair (G2-1,G2-2,G2-3), Poor (G2-4).

#### `get_grade_color(self, grade)`
**Purpose**: Gets color coding for displaying grades in the UI.

**Parameters**:
- `grade` (str): Grade string

**Return Value**: str - Color name for UI display

**Details**: Returns appropriate colors for different grade levels.

---

### UI Creation Functions

#### `create_section(self, parent, title, col)`
**Purpose**: Creates a camera section in the main UI layout.

**Parameters**:
- `parent` (tk.Widget): Parent widget
- `title` (str): Section title
- `col` (int): Column position

**Return Value**: tuple - (live_feed_label, details_text_widget, details_text)

**Details**: Creates the basic structure for camera feed display areas.

#### `create_detection_details_section(self, parent, title, camera_name)`
**Purpose**: Creates a scrollable detection details section.

**Parameters**:
- `parent` (tk.Widget): Parent widget
- `title` (str): Section title
- `camera_name` (str): Camera identifier

**Return Value**: tuple - (frame, details_widgets)

**Details**: Creates scrollable widget containers for detection information.

#### `create_detection_widgets(self, parent, camera_name)`
**Purpose**: Creates the widget structure for detection details display.

**Parameters**:
- `parent` (tk.Widget): Parent widget
- `camera_name` (str): Camera name

**Return Value**: dict - Dictionary of created widgets

**Details**: Builds the UI components for showing detection status, calibration info, and defect details.

#### `create_simple_detection_tracker(self, camera_name)`
**Purpose**: Creates a simplified detection tracking object.

**Parameters**:
- `camera_name` (str): Camera identifier

**Return Value**: dict - Tracker object with detection state

**Details**: Maintains detection logic without complex UI components.

---

### Logging and Data Management Functions

#### `log_detection_details(self, camera_name, defect_dict, measurements, surface_grade)`
**Purpose**: Logs detailed defect information for documentation and analysis.

**Parameters**:
- `camera_name` (str): Camera name
- `defect_dict` (dict): Defect counts by type
- `measurements` (list): Detailed defect measurements
- `surface_grade` (str): Calculated surface grade

**Return Value**: None

**Details**: Creates comprehensive detection entries with camera info, defect details, and grading reasoning.

#### `save_detection_log(self, detection_entry)`
**Purpose**: Saves detection log entry to JSON file.

**Parameters**:
- `detection_entry` (dict): Detection data to save

**Return Value**: None

**Details**: Appends detection data to daily log files for long-term record keeping.

#### `start_test_case(self, test_case_number)`
**Purpose**: Starts a new test case for documentation purposes.

**Parameters**:
- `test_case_number` (int): Test case identifier

**Return Value**: None

**Details**: Resets counters and sets up tracking for specific test scenarios.

#### `export_test_case_summary(self, test_case_number)`
**Purpose**: Exports summary of a specific test case.

**Parameters**:
- `test_case_number` (int): Test case to export

**Return Value**: None

**Details**: Creates JSON summary files with test case statistics and defect analysis.

---

### Video Feed and Processing Functions

#### `update_feeds(self)`
**Purpose**: Updates both camera feeds and processes detection at configured intervals.

**Parameters**: None

**Return Value**: None

**Details**: Main video processing loop that handles frame capture, detection, and UI updates.

#### `update_single_feed(self, cap, label, camera_name)`
**Purpose**: Processes a single camera feed with detection and display.

**Parameters**:
- `cap` (cv2.VideoCapture): Camera capture object
- `label` (tk.Label): UI label for display
- `camera_name` (str): Camera identifier

**Return Value**: None

**Details**: Handles individual camera processing including ROI application, detection, and UI updates.

#### `apply_roi(self, frame, camera_name)`
**Purpose**: Applies Region of Interest cropping to focus detection.

**Parameters**:
- `frame` (numpy.ndarray): Input frame
- `camera_name` (str): Camera name

**Return Value**: tuple - (roi_frame, roi_info) where roi_info contains coordinates

**Details**: Crops frame to configured ROI coordinates for focused defect detection.

#### `draw_roi_overlay(self, frame, camera_name)`
**Purpose**: Draws ROI rectangle overlay on frame for visualization.

**Parameters**:
- `frame` (numpy.ndarray): Input frame
- `camera_name` (str): Camera name

**Return Value**: numpy.ndarray - Frame with ROI overlay

**Details**: Adds visual rectangle showing the active detection region.

---

### Arduino Communication Functions

#### `setup_arduino(self)`
**Purpose**: Initializes Arduino serial communication.

**Parameters**: None

**Return Value**: None

**Details**: Attempts connection to Arduino on multiple possible serial ports with automatic retry.

#### `listen_for_arduino(self)`
**Purpose**: Background thread for listening to Arduino messages.

**Parameters**: None

**Return Value**: None

**Details**: Continuously monitors serial connection for Arduino messages with reconnection logic.

#### `send_arduino_command(self, command)`
**Purpose**: Sends command to Arduino with error handling.

**Parameters**:
- `command` (str): Command string to send

**Return Value**: None

**Details**: Handles serial communication with rate limiting and reconnection on failure.

#### `process_message_queue(self)`
**Purpose**: Processes Arduino messages in the main thread.

**Parameters**: None

**Return Value**: None

**Details**: Safely handles Arduino messages including IR beam triggers and length measurements.

---

### Operation Mode Functions

#### `set_continuous_mode(self)`
**Purpose**: Sets system to fully automatic continuous mode.

**Parameters**: None

**Return Value**: None

**Details**: Enables live detection and auto-grading for continuous operation.

#### `set_trigger_mode(self)`
**Purpose**: Sets system to wait for IR beam trigger.

**Parameters**: None

**Return Value**: None

**Details**: Configures system for triggered operation with IR beam detection.

#### `set_idle_mode(self)`
**Purpose**: Disables all operations and stops conveyor.

**Parameters**: None

**Return Value**: None

**Details**: Puts system in idle state with all operations disabled.

---

### Grading and Control Functions

#### `finalize_grading(self, final_grade, all_measurements)`
**Purpose**: Central function for logging, statistics update, and Arduino command sending.

**Parameters**:
- `final_grade` (str): Final SS-EN 1611-1 grade
- `all_measurements` (list): All defect measurements from both surfaces

**Return Value**: None

**Details**: Updates statistics, logs the piece, and sends sorting command to Arduino.

#### `start_automatic_detection(self)`
**Purpose**: Starts automatic detection when IR beam detects object.

**Parameters**: None

**Return Value**: None

**Details**: Initializes detection session data and prepares for automatic processing.

#### `stop_automatic_detection_and_grade(self)`
**Purpose**: Stops detection and processes final grading results.

**Parameters**: None

**Return Value**: None

**Details**: Analyzes accumulated detection data and determines final grade.

#### `determine_final_grade_from_session(self, camera_name, detections_list)`
**Purpose**: Determines final grade from all detections in a session.

**Parameters**:
- `camera_name` (str): Camera identifier
- `detections_list` (list): List of detection data from session

**Return Value**: str - Final grade for the session

**Details**: Combines multiple detection frames into comprehensive grading decision.

---

### UI Update Functions

#### `update_live_grading_display(self)`
**Purpose**: Updates the live grading display with current detection results.

**Parameters**: None

**Return Value**: None

**Details**: Shows real-time grading results for both cameras and combined final grade.

#### `update_live_stats_display(self)`
**Purpose**: Updates live statistics display with thread safety.

**Parameters**: None

**Return Value**: None

**Details**: Safely updates grade count displays in the main thread.

#### `update_detection_status_display(self)`
**Purpose**: Updates status display based on current operation mode.

**Parameters**: None

**Return Value**: None

**Details**: Shows current system status (IDLE, TRIGGER, CONTINUOUS, etc.).

#### `update_dashboard_display(self, camera_name, defect_dict, measurements=None)`
**Purpose**: Updates simplified dashboard display and logs defect data.

**Parameters**:
- `camera_name` (str): Camera name
- `defect_dict` (dict): Defect counts
- `measurements` (list): Detailed measurements

**Return Value**: None

**Details**: Maintains detection tracking and triggers detailed logging.

---

### Statistics and Reporting Functions

#### `update_defect_details_tab(self)`
**Purpose**: Updates the Defect Details statistics tab.

**Parameters**: None

**Return Value**: None

**Details**: Shows current defect information and grading thresholds reference.

#### `update_performance_tab(self)`
**Purpose**: Updates the Performance Metrics tab.

**Parameters**: None

**Return Value**: None

**Details**: Displays system calibration info and processing statistics.

#### `update_recent_activity_tab(self)`
**Purpose**: Updates the Recent Activity tab with processing log.

**Parameters**: None

**Return Value**: None

**Details**: Shows session summary and scrollable processing log with defect details.

#### `generate_report(self)`
**Purpose**: Generates comprehensive PDF and text reports.

**Parameters**: None

**Return Value**: None

**Details**: Creates detailed reports with session statistics and individual piece logs.

#### `manual_generate_report(self)`
**Purpose**: Manually triggers report generation.

**Parameters**: None

**Return Value**: None

**Details**: User-initiated report creation with status updates.

---

### Utility Functions

#### `calculate_grade(self, defect_dict)`
**Purpose**: Calculates grade based on simple defect counting.

**Parameters**:
- `defect_dict` (dict): Dictionary of defect counts by type

**Return Value**: dict - Grade information with text, color, and counts

**Details**: Simple grading based on total defect counts (fallback method).

#### `log_action(self, message)`
**Purpose**: Logs system actions to file with timestamps.

**Parameters**:
- `message` (str): Action message to log

**Return Value**: None

**Details**: Maintains activity log for system monitoring and debugging.

#### `calculate_and_display_length(self, duration_ms)`
**Purpose**: Calculates wood piece length from conveyor timing.

**Parameters**:
- `duration_ms` (int): Time in milliseconds wood blocked IR beam

**Return Value**: None

**Details**: Uses conveyor speed to estimate wood piece dimensions.

#### `reset_inactivity_timer(self)`
**Purpose**: Resets the inactivity timer for auto-reporting.

**Parameters**: None

**Return Value**: None

**Details**: Prevents auto-report generation during active use.

#### `check_inactivity(self)`
**Purpose**: Checks for inactivity and generates reports automatically.

**Parameters**: None

**Return Value**: None

**Details**: Auto-generates reports after 30 seconds of inactivity if pieces were processed.

---

### System Management Functions

#### `on_closing(self)`
**Purpose**: Handles application shutdown gracefully.

**Parameters**: None

**Return Value**: None

**Details**: Closes serial connections, releases cameras, and destroys the application.

#### `toggle_fullscreen(self, event=None)`
**Purpose**: Toggles fullscreen mode (F11 key).

**Parameters**:
- `event` (tk.Event): Key event (optional)

**Return Value**: str - "break" to prevent event propagation

**Details**: Switches between windowed and fullscreen display modes.

#### `exit_fullscreen(self, event=None)`
**Purpose**: Exits fullscreen mode (Escape key).

**Parameters**:
- `event` (tk.Event): Key event (optional)

**Return Value**: str - "break" to prevent event propagation

**Details**: Returns to windowed mode from fullscreen.

#### `auto_fullscreen_rpi(self)`
**Purpose**: Automatically enables fullscreen for Raspberry Pi deployment.

**Parameters**: None

**Return Value**: None

**Details**: Detects Raspberry Pi environment and enables fullscreen mode.

---

### Legacy/Compatibility Functions

#### `update_detection_details(self, camera_name, defect_dict, measurements=None)`
**Purpose**: Legacy function for updating detection details display.

**Parameters**:
- `camera_name` (str): Camera name
- `defect_dict` (dict): Defect dictionary
- `measurements` (list): Defect measurements

**Return Value**: None

**Details**: Maintains backward compatibility with older UI update methods.

#### `update_detection_details_widgets(self, camera_name, defect_dict, measurements=None)`
**Purpose**: Updates detection details using widget objects.

**Parameters**:
- `camera_name` (str): Camera name
- `defect_dict` (dict): Defect counts
- `measurements` (list): Detailed measurements

**Return Value**: None

**Details**: Updates UI widgets with current detection information.

#### `create_tabbed_detection_details(self, parent, camera_name)`
**Purpose**: Creates tabbed interface for detection details.

**Parameters**:
- `parent` (tk.Widget): Parent widget
- `camera_name` (str): Camera name

**Return Value**: dict - Dictionary of tab widgets

**Details**: Creates tabbed UI for current detection, statistics, and history.

#### `create_current_detection_widgets(self, parent, camera_name)`
**Purpose**: Creates widgets for current detection display.

**Parameters**:
- `parent` (tk.Widget): Parent widget
- `camera_name` (str): Camera name

**Return Value**: dict - Created widgets

**Details**: Builds UI components for real-time detection status.

#### `create_grid_detection_display(self, parent, camera_name)`
**Purpose**: Creates fixed grid layout for detection display.

**Parameters**:
- `parent` (tk.Widget): Parent widget
- `camera_name` (str): Camera name

**Return Value**: dict - Created widgets

**Details**: Creates non-scrolling grid layout for detection information.

#### `create_dashboard_detection_display(self, parent, camera_name)`
**Purpose**: Compatibility method for dashboard creation.

**Parameters**:
- `parent` (tk.Widget): Parent widget
- `camera_name` (str): Camera name

**Return Value**: dict - Tracker object

**Details**: Returns simple detection tracker for compatibility.

#### `save_detection_session(self)`
**Purpose**: Saves complete detection session data.

**Parameters**: None

**Return Value**: None

**Details**: Archives session data including best frames and detection results.

#### `save_detection_frame(self, camera_name, frame)`
**Purpose**: Saves a detection frame as image file.

**Parameters**:
- `camera_name` (str): Camera name
- `frame` (numpy.ndarray): Frame to save

**Return Value**: None

**Details**: Saves annotated frames for documentation and analysis.

#### `toggle_roi(self)`
**Purpose**: Toggles ROI for top camera.

**Parameters**: None

**Return Value**: None

**Details**: Enables/disables region of interest for focused detection.

#### `toggle_live_detection_mode(self)`
**Purpose**: Handles toggling between IR trigger and live detection modes.

**Parameters**: None

**Return Value**: None

**Details**: Updates status display when detection mode changes.

#### `_execute_manual_grade(self)`
**Purpose**: Executes manual grading based on current detections.

**Parameters**: None

**Return Value**: None

**Details**: Manually triggers grading for current wood piece.

#### `_safe_update_label(self, grade_key, count)`
**Purpose**: Safely updates a statistics label with error handling.

**Parameters**:
- `grade_key` (str): Label identifier
- `count` (int): Count value

**Return Value**: None

**Details**: Thread-safe label updates with existence checks.

#### `update_detailed_statistics(self)`
**Purpose**: Legacy method that redirects to recent activity updates.

**Parameters**: None

**Return Value**: None

**Details**: Maintains compatibility with older statistics update calls.

#### `_generate_stats_content(self)`
**Purpose**: Generates string representation of current statistics.

**Parameters**: None

**Return Value**: str - Statistics summary string

**Details**: Creates content hash for change detection in UI updates.

---

## Dependencies

The application requires the following Python packages:
- tkinter (built-in)
- opencv-python (cv2)
- pillow (PIL)
- degirum
- degirum-tools
- json (built-in)
- os (built-in)
- datetime (built-in)
- reportlab
- numpy
- psutil
- serial
- threading (built-in)
- time (built-in)
- queue (built-in)

## Hardware Requirements

- **Cameras**: Two USB cameras (tested with indices 0 and 2)
- **Arduino**: Microcontroller with serial communication for conveyor control
- **IR Beam Sensor**: For trigger-based detection mode
- **Conveyor System**: Automated wood transport system

## Usage Instructions

1. **Setup**: Ensure cameras and Arduino are connected
2. **Calibration**: Run calibration with known reference objects
3. **Configuration**: Adjust settings in the configuration section
4. **Operation**: Select operation mode (IDLE/TRIGGER/CONTINUOUS)
5. **Monitoring**: Use GUI to monitor real-time detection and grading
6. **Reporting**: Generate reports for session analysis

## File Outputs

- **detection_log_YYYY-MM-DD.json**: Daily detection logs
- **report_YYYY-MM-DD_HH-MM-SS.txt**: Text reports
- **report_YYYY-MM-DD_HH-MM-SS.pdf**: PDF reports
- **wood_sorting_activity_log.txt**: Activity log
- **detection_frames/**: Saved detection frame images

This documentation provides comprehensive coverage of the wood detection program's functionality, from basic camera operations to advanced AI-powered grading according to European standards.