import serial
import threading
import time
import queue
from modules.error_handler import log_arduino_error, log_info, log_warning, SystemComponent
from modules.grading_module import convert_grade_to_arduino_command

class ArduinoModule:
    def __init__(self, message_queue):
        self.ser = None
        self.message_queue = message_queue
        self._shutting_down = False
        self.arduino_thread = None
        
        # Enhanced connection status tracking
        self.connection_status = {
            "connected": False,
            "port": None,
            "last_successful_command": None,
            "last_error": None,
            "error_count": 0,
            "reconnection_attempts": 0
        }
        
        # Command queue for reliable delivery
        self.command_queue = queue.Queue()
        self.command_thread = None
        
        log_info(SystemComponent.ARDUINO, "ArduinoModule initialized")

    def setup_arduino(self):
        """Enhanced Arduino setup with comprehensive error handling."""
        if self._shutting_down:
            return False
            
        log_info(SystemComponent.ARDUINO, "Attempting to setup Arduino connection")
        
        try:
            # Close existing connection if any
            if self.ser and self.ser.is_open:
                try:
                    self.ser.close()
                except:
                    pass
            
            # List of ports to try (Windows and Linux/Raspberry Pi)
            ports_to_try = [
                '/dev/ttyACM0', '/dev/ttyACM1', '/dev/ttyACM2', '/dev/ttyACM3',
                '/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyUSB2', '/dev/ttyUSB3',
                '/dev/ttyUSB01', '/dev/ttyACM01',
                '/dev/ttyAMA0', '/dev/ttyAMA1', '/dev/ttyAMA10',
                'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'COM10', 'COM11', 'COM12'
            ]
            
            for port in ports_to_try:
                try:
                    log_info(SystemComponent.ARDUINO, f"Trying to connect on port {port}")
                    
                    self.ser = serial.Serial(
                        port=port, 
                        baudrate=9600, 
                        timeout=2,           
                        write_timeout=2,     
                        bytesize=serial.EIGHTBITS,
                        parity=serial.PARITY_NONE,
                        stopbits=serial.STOPBITS_ONE,
                        xonxoff=False,       
                        rtscts=False,        
                        dsrdtr=False         
                    )
                    
                    # Wait for Arduino to initialize
                    time.sleep(3)  
                    
                    # Clear buffers
                    self.ser.reset_input_buffer()
                    self.ser.reset_output_buffer()
                    
                    # Send test command
                    self.ser.write(b'X')  
                    self.ser.flush()
                    time.sleep(0.5)
                    
                    # Update status
                    self.connection_status["connected"] = True
                    self.connection_status["port"] = port
                    self.connection_status["error_count"] = 0
                    self.connection_status["reconnection_attempts"] = 0
                    
                    log_info(SystemComponent.ARDUINO, f"Successfully connected on port {port}")
                    break
                    
                except (serial.SerialException, OSError) as e:
                    log_arduino_error(f"Failed to connect on port {port}: {str(e)}")
                    continue
            else:
                # No port worked
                raise serial.SerialException("No Arduino found on any available port")
            
            # Start listener and command processor threads
            if not self._shutting_down:
                self._start_threads()
                
            return True
            
        except Exception as e:
            self.ser = None
            self.connection_status["connected"] = False
            self.connection_status["last_error"] = str(e)
            self.connection_status["error_count"] += 1
            
            log_arduino_error(f"Arduino setup failed: {str(e)}", e)
            if self.message_queue:
                self.message_queue.put(("status_update", "Arduino not found. Running in manual mode."))
            return False

    def _start_threads(self):
        """Start listener and command processor threads"""
        # Start listener thread
        if not self.arduino_thread or not self.arduino_thread.is_alive():
            self.arduino_thread = threading.Thread(target=self.listen_for_arduino, daemon=True)
            self.arduino_thread.start()
            log_info(SystemComponent.ARDUINO, "Started Arduino listener thread")
        
        # Start command processor thread
        if not self.command_thread or not self.command_thread.is_alive():
            self.command_thread = threading.Thread(target=self._process_command_queue, daemon=True)
            self.command_thread.start()
            log_info(SystemComponent.ARDUINO, "Started Arduino command processor thread")

    def _process_command_queue(self):
        """Process commands from queue for reliable delivery"""
        while not self._shutting_down:
            try:
                # Wait for command with timeout
                command = self.command_queue.get(timeout=1.0)
                
                if command is None:  # Shutdown signal
                    break
                    
                # Attempt to send command with retries
                max_retries = 3
                for attempt in range(max_retries):
                    if self._send_command_direct(command):
                        break
                    else:
                        log_warning(SystemComponent.ARDUINO, f"Command '{command}' failed (attempt {attempt + 1}/{max_retries})")
                        if attempt < max_retries - 1:
                            time.sleep(0.5)  # Wait before retry
                else:
                    log_arduino_error(f"Failed to send command '{command}' after {max_retries} attempts")
                    
                self.command_queue.task_done()
                
            except queue.Empty:
                continue  # Normal timeout, continue loop
            except Exception as e:
                log_arduino_error(f"Error in command processor: {str(e)}", e)

    def listen_for_arduino(self):
        """Robust Arduino listener with automatic reconnection."""
        reconnect_attempts = 0
        max_reconnect_attempts = 5
        
        while True:
            try:
                if self._shutting_down:
                    print("Arduino listener: Shutdown detected, exiting thread")
                    break
                
                if self.ser and self.ser.is_open:
                    if self.ser.in_waiting > 0:
                        try:
                            message = self.ser.readline().decode('utf-8').strip()
                            if not message:
                                continue

                            print(f"📨 Arduino Message: '{message}' (Port: {self.ser.port})")
                            self.message_queue.put(("arduino_message", message))
                            reconnect_attempts = 0  
                            
                        except UnicodeDecodeError as e:
                            print(f"⚠️ Arduino message decode error: {e}")
                            try:
                                self.ser.reset_input_buffer()
                            except:
                                pass
                            continue
                            
                elif not self.ser or (hasattr(self.ser, 'is_open') and not self.ser.is_open):
                    if reconnect_attempts < max_reconnect_attempts:
                        reconnect_attempts += 1
                        print(f"🔄 Arduino disconnected, attempting reconnection {reconnect_attempts}/{max_reconnect_attempts}...")
                        time.sleep(2)  
                        
                        try:
                            self.setup_arduino()
                            if self.ser and self.ser.is_open:
                                print(f"✅ Arduino reconnected successfully on {self.ser.port}")
                                reconnect_attempts = 0
                            else:
                                print(f"❌ Reconnection attempt {reconnect_attempts} failed")
                        except Exception as e:
                            print(f"❌ Reconnection attempt {reconnect_attempts} failed: {e}")
                    else:
                        print(f"❌ Max reconnection attempts ({max_reconnect_attempts}) reached, exiting listener thread")
                        break
                    
                time.sleep(0.1)  
                
            except (serial.SerialException, OSError, TypeError) as e:
                print(f"🔥 Arduino communication error: {e}")
                
                if self._shutting_down:
                    print("Arduino listener: Application shutting down, exiting thread")
                    break
                    
                if reconnect_attempts < max_reconnect_attempts:
                    reconnect_attempts += 1
                    print(f"🔄 Communication error, attempting reconnection {reconnect_attempts}/{max_reconnect_attempts}...")
                    time.sleep(2)
                    try:
                        self.setup_arduino()
                        if self.ser and self.ser.is_open:
                            print(f"✅ Arduino reconnected after error on {self.ser.port}")
                            reconnect_attempts = 0
                    except Exception as reconnect_error:
                        print(f"❌ Reconnection failed: {reconnect_error}")
                else:
                    print(f"❌ Max reconnection attempts reached after error, exiting thread")
                    break
                    
            except Exception as e:
                print(f"❌ Unexpected error in Arduino listener: {e}")
                time.sleep(1)
                self.message_queue.put(("status_update", "Arduino connection lost"))
                break

    def send_arduino_command(self, command):
        """Queue a command for reliable delivery to Arduino."""
        if self._shutting_down:
            return False
            
        try:
            # Add command to queue for processing
            self.command_queue.put(command, timeout=1.0)
            log_info(SystemComponent.ARDUINO, f"Queued command: '{command}'")
            return True
            
        except queue.Full:
            log_arduino_error(f"Command queue full, dropping command: '{command}'")
            return False
        except Exception as e:
            log_arduino_error(f"Error queueing command '{command}': {str(e)}", e)
            return False

    def _send_command_direct(self, command):
        """Send a command directly to Arduino (used by command processor)."""
        if self._shutting_down:
            return False
            
        current_time = time.time()
        if hasattr(self, '_last_command_time'):
            time_since_last = current_time - self._last_command_time
            if time_since_last < 0.1:  
                time.sleep(0.1 - time_since_last)
        
        try:
            if self.ser and self.ser.is_open:
                # Clear buffers before sending
                try:
                    self.ser.reset_input_buffer()
                    self.ser.reset_output_buffer()
                except:
                    pass
                
                # Send command
                command_bytes = command.encode('utf-8')
                self.ser.write(command_bytes)
                self.ser.flush()
                
                self._last_command_time = time.time()
                self.connection_status["last_successful_command"] = current_time
                
                log_info(SystemComponent.ARDUINO, f"Sent command: '{command}' (Port: {self.ser.port})")
                return True
                
            else:
                log_arduino_error("Cannot send command: Arduino not connected")
                # Attempt reconnection
                if self.connection_status["reconnection_attempts"] < 3:
                    self.connection_status["reconnection_attempts"] += 1
                    log_info(SystemComponent.ARDUINO, "Attempting reconnection for command delivery")
                    if self.setup_arduino():
                        return self._send_command_direct(command)  # Retry once after reconnection
                return False
                    
        except (serial.SerialException, OSError, TypeError) as e:
            self.connection_status["connected"] = False
            self.connection_status["last_error"] = str(e)
            self.connection_status["error_count"] += 1
            
            log_arduino_error(f"Error sending command '{command}': {str(e)}", e)
            
            if self.message_queue:
                self.message_queue.put(("status_update", "Arduino communication error"))
            
            # Attempt reconnection for next command
            if not self._shutting_down and self.connection_status["reconnection_attempts"] < 3:
                time.sleep(1)
                self.setup_arduino()
                
            return False

    def send_grade_command(self, grade):
        """Convert grade to Arduino command and send it."""
        try:
            arduino_command = convert_grade_to_arduino_command(grade)
            success = self.send_arduino_command(str(arduino_command))
            
            if success:
                log_info(SystemComponent.ARDUINO, f"Sent grade command: Grade {grade} -> Command {arduino_command}")
            else:
                log_arduino_error(f"Failed to send grade command: Grade {grade} -> Command {arduino_command}")
                
            return success
            
        except Exception as e:
            log_arduino_error(f"Error converting/sending grade command for grade '{grade}': {str(e)}", e)
            return False

    def get_connection_status(self):
        """Get current Arduino connection status"""
        return self.connection_status.copy()

    def is_connected(self):
        """Check if Arduino is currently connected"""
        return self.connection_status.get("connected", False)

    def convert_grade_to_arduino_command(self, standard_grade):
        """Convert SS-EN 1611-1 grade to Arduino sorting command (fallback method)"""
        return convert_grade_to_arduino_command(standard_grade)

    def close_connection(self):
        """Close the Arduino serial connection with proper cleanup."""
        log_info(SystemComponent.ARDUINO, "Closing Arduino connection")
        
        self._shutting_down = True
        
        # Signal command processor to stop
        try:
            self.command_queue.put(None, timeout=1.0)  # Shutdown signal
        except:
            pass
        
        # Wait for threads to finish
        if self.command_thread and self.command_thread.is_alive():
            self.command_thread.join(timeout=2.0)
            
        if self.arduino_thread and self.arduino_thread.is_alive():
            self.arduino_thread.join(timeout=2.0)
        
        # Close serial connection
        if self.ser:
            try:
                self.ser.close()
                self.ser = None
                log_info(SystemComponent.ARDUINO, "Arduino serial connection closed")
            except Exception as e:
                log_arduino_error(f"Error closing Arduino connection: {str(e)}", e)
        
        # Reset connection status
        self.connection_status["connected"] = False
        self.connection_status["port"] = None
        
        log_info(SystemComponent.ARDUINO, "Arduino module shutdown complete")