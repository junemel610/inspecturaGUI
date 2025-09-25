"""
Wood Detection App
Standalone application for wood detection using color recognition (HSV) and Canny edge detection.
Captures from two USB cameras (video0 and video2), applies masking and edge detection, and draws bounding boxes around detected wood objects.
"""
import cv2
import numpy as np

class WoodDetectionApp:
    def __init__(self):
        self.top_camera_index = 0  # Cam0
        self.bottom_camera_index = 2  # Cam2
        self.top_camera_settings = {
            'brightness': 0,
            'contrast': 32,
            'saturation': 64,
            'hue': 0,
            'exposure': -6,
            'white_balance': 4520,
            'gain': 0
        }
        self.bottom_camera_settings = {
            'brightness': 135,
            'contrast': 75,
            'saturation': 155,
            'hue': 0,
            'exposure': -6,
            'white_balance': 5400,
            'gain': 0
        }
        self.top_camera = self.init_camera(self.top_camera_index, self.top_camera_settings)
        self.bottom_camera = self.init_camera(self.bottom_camera_index, self.bottom_camera_settings)

    def init_camera(self, index, settings):
        cam = cv2.VideoCapture(index)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cam.set(cv2.CAP_PROP_BRIGHTNESS, settings['brightness'])
        cam.set(cv2.CAP_PROP_CONTRAST, settings['contrast'])
        cam.set(cv2.CAP_PROP_SATURATION, settings['saturation'])
        cam.set(cv2.CAP_PROP_HUE, settings['hue'])
        cam.set(cv2.CAP_PROP_EXPOSURE, settings['exposure'])
        cam.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, settings['white_balance'])
        cam.set(cv2.CAP_PROP_GAIN, settings['gain'])
        return cam

    def process_frame(self, frame, hsv_lower, hsv_upper):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
        masked = cv2.bitwise_and(frame, frame, mask=mask)
        edges = cv2.Canny(masked, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:  # Filter small objects
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return frame, mask, edges

    def run(self):
        # Example HSV range for wood color (tune as needed)
        hsv_lower = np.array([10, 30, 100])
        hsv_upper = np.array([30, 255, 255])
        while True:
            ret_top, frame_top = self.top_camera.read()
            ret_bottom, frame_bottom = self.bottom_camera.read()
            if not ret_top or not ret_bottom:
                print("Camera read error.")
                break
            processed_top, _, _ = self.process_frame(frame_top, hsv_lower, hsv_upper)
            processed_bottom, _, _ = self.process_frame(frame_bottom, hsv_lower, hsv_upper)
            # Resize to 360p for display
            display_top = cv2.resize(processed_top, (640, 360))
            display_bottom = cv2.resize(processed_bottom, (640, 360))
            orig_top = cv2.resize(frame_top, (640, 360))
            orig_bottom = cv2.resize(frame_bottom, (640, 360))
            # Stack original and processed side by side
            top_combined = np.hstack((orig_top, display_top))
            bottom_combined = np.hstack((orig_bottom, display_bottom))
            cv2.imshow('Top Camera: Original | Processed', top_combined)
            cv2.imshow('Bottom Camera: Original | Processed', bottom_combined)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.top_camera.release()
        self.bottom_camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = WoodDetectionApp()
    app.run()
