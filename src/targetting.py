#!/usr/bin/env python3

import cv2
import numpy as np
import serial
import struct
import time
import math
import threading
from datetime import datetime
import queue
from collections import deque

CAMERA_FOV_DEGREES = 65.5 
CAMERA_WIDTH = 320 
CAMERA_HEIGHT = 240 

# Calculate focal length, output in pixels for 
FOCAL_LENGTH_PIXELS = (CAMERA_WIDTH / 2) / math.tan(math.radians(CAMERA_FOV_DEGREES / 2))

BUCKET_HEIGHT_INCHES = 14.25
BUCKET_DIAMETER_INCHES = 11.25
BUCKET_ASPECT_RATIO = BUCKET_HEIGHT_INCHES / BUCKET_DIAMETER_INCHES

HSV_RANGES = {
    # Lower saturation/value for compression
    'red_lower1': np.array([0, 50, 30]),     
    # Wider hue range
    'red_upper1': np.array([15, 255, 255]), 
    # Wrap-around red
    'red_lower2': np.array([165, 50, 30]),   
    'red_upper2': np.array([180, 255, 255]),
    # Orange-red range (Harbor Freight buckets can appear orange)
    'orange_lower': np.array([10, 50, 50]),
    'orange_upper': np.array([25, 255, 255]),
}


class ESP32StreamReceiver:

    def __init__(self, serial_conn):
        self.ser = serial_conn
        self.buffer = bytearray()
        self.frame_queue = queue.Queue(maxsize=3)
        self.running = True
        self.thread = threading.Thread(target=self._receive_loop)
        self.thread.daemon = True
        self.stats = {
            'frames_received': 0,
            'frames_dropped': 0,
            'bad_frames': 0
        }
        
    def start(self):
        """Start receiving thread"""
        self.thread.start()
        print("[Stream] Receiver started")
        
    def stop(self):
        """Stop receiving thread"""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2)
            
    def _receive_loop(self):
        """Main receiving loop"""
        while self.running:
            try:
                if self.ser.in_waiting > 0:
                    data = self.ser.read(self.ser.in_waiting)
                    self.buffer.extend(data)
                    
                self._process_buffer()
                time.sleep(0.001)
                
            except Exception as e:
                print(f"[Stream] Error: {e}")
                
    def _process_buffer(self):
        """Extract frames from buffer"""
        while len(self.buffer) >= 12:
            mjpg_idx = self.buffer.find(b'MJPG')
            
            if mjpg_idx < 0:
                if len(self.buffer) > 10000:
                    self.buffer = self.buffer[-5000:]
                return
                
            if mjpg_idx > 0:
                self.buffer = self.buffer[mjpg_idx:]
                
            if len(self.buffer) < 12:
                return
                
            try:
                magic = self.buffer[0:4]
                if magic != b'MJPG':
                    self.buffer = self.buffer[1:]
                    continue
                    
                length = struct.unpack('<I', bytes(self.buffer[4:8]))[0]
                width = struct.unpack('<H', bytes(self.buffer[8:10]))[0]
                height = struct.unpack('<H', bytes(self.buffer[10:12]))[0]
                
                if length > 100000 or width != CAMERA_WIDTH or height != CAMERA_HEIGHT:
                    self.stats['bad_frames'] += 1
                    self.buffer = self.buffer[1:]
                    continue
                    
                if len(self.buffer) < 12 + length:
                    return
                    
                jpeg_data = bytes(self.buffer[12:12+length])
                self.buffer = self.buffer[12+length:]
                
                arr = np.frombuffer(jpeg_data, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    self.stats['frames_received'] += 1
                    
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                            self.stats['frames_dropped'] += 1
                        except queue.Empty:
                            pass
                            
                    self.frame_queue.put(frame)
                    
            except Exception:
                self.buffer = self.buffer[1:]
                self.stats['bad_frames'] += 1
                
    def get_frame(self):
        """Get latest frame from queue"""
        try:
            return self.frame_queue.get(timeout=0.1)
        except queue.Empty:
            return None
            
    def get_stats(self):
        return self.stats.copy()


class SmoothTracker:
    
    def __init__(self, history_size=10):
        self.distance_history = deque(maxlen=history_size)
        self.angle_history = deque(maxlen=history_size)
        self.position_history = deque(maxlen=history_size)
        self.confidence_threshold = 0.3
        self.last_valid_bbox = None
        self.lost_frames = 0
        self.max_lost_frames = 10
        
    def update(self, bbox, distance, angle, center):
        """Update tracking with new measurement"""
        if bbox is not None:
            self.last_valid_bbox = bbox
            self.lost_frames = 0
            
            if distance is not None:
                self.distance_history.append(distance)
                
            if angle is not None:
                self.angle_history.append(angle)
                
            if center is not None:
                self.position_history.append(center)
                
            return True
        else:
            self.lost_frames += 1
            return False
            
    def get_smooth_distance(self):
        """Get averaged distance with outlier rejection"""
        if len(self.distance_history) < 3:
            return None if not self.distance_history else self.distance_history[-1]
            
        # Remove outliers using median absolute deviation
        distances = list(self.distance_history)
        median = np.median(distances)
        mad = np.median([abs(d - median) for d in distances])
        
        if mad == 0:
            return median
            
        # Keep values within 2 MAD of median
        filtered = [d for d in distances if abs(d - median) <= 2 * mad]
        
        if filtered:
            return np.mean(filtered)
        return median
        
    def get_smooth_angle(self):
        """Get averaged angle"""
        if not self.angle_history:
            return None
            
        # Simple moving average for angle (usually stable)
        return np.mean(list(self.angle_history))
        
    def get_smooth_center(self):
        """Get averaged center position"""
        if not self.position_history:
            return None
            
        centers = list(self.position_history)
        avg_x = int(np.mean([c[0] for c in centers]))
        avg_y = int(np.mean([c[1] for c in centers]))
        return (avg_x, avg_y)
        
    def is_tracking(self):
        """Check if we're actively tracking a target"""
        return self.lost_frames < self.max_lost_frames and len(self.distance_history) > 0


class ESP32CatapultTargeting:
    """Enhanced catapult targeting system with auto-tracking"""
    
    def __init__(self, port, baud=2000000):
        # Serial connection
        self.ser = serial.Serial(port, baud, timeout=0.1)
        print(f"[System] Connected to {port}")
        
        # Stream receiver
        self.receiver = ESP32StreamReceiver(self.ser)
        
        # Tracking
        self.tracker = SmoothTracker(history_size=15)
        
        # Auto-centering
        self.auto_center = True
        self.center_threshold = 10
        self.last_center_time = 0
        self.center_interval = 0.05  # seconds between centering commands
        
        # System state
        self.armed = False
        self.fire_solution = None
        self.debug_mode = False
        
        # Detection parameters (less aggressive)
        self.min_area = 200  # Smaller minimum
        self.max_area = 60000  # Larger maximum
        
        # Morphological kernels
        self.kernel_small = np.ones((3, 3), np.uint8)
        self.kernel_medium = np.ones((5, 5), np.uint8)
        
        # Calibration
        self.firing_table = [
            (12, 25, 30),   # distance, angle, power
            (24, 35, 45),
            (36, 40, 60),
            (48, 45, 75),
            (60, 48, 85),
            (72, 50, 95),
        ]
        
        # Initialize
        time.sleep(2)
        self.send_command("RESET")
        
    def send_command(self, cmd):
        """Send command to ESP32"""
        was_streaming = False
        if hasattr(self, 'receiver') and self.receiver.running:
            self.send_raw("Q")
            time.sleep(0.1)
            was_streaming = True
            
        if not cmd.endswith('\n'):
            cmd += '\n'
        self.ser.write(cmd.encode())
        time.sleep(0.05)
        
        if was_streaming:
            time.sleep(0.1)
            self.send_raw("S")
            
    def send_raw(self, char):
        """Send single character command"""
        self.ser.write(char.encode())
        
    def detect_bucket(self, frame):
        """Enhanced bucket detection for low-quality streams"""
        # Apply slight blur to reduce noise
        blurred = cv2.GaussianBlur(frame, (5, 5), 1)
        
        # Convert to HSV
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # Create masks for multiple color ranges
        mask1 = cv2.inRange(hsv, HSV_RANGES['red_lower1'], HSV_RANGES['red_upper1'])
        mask2 = cv2.inRange(hsv, HSV_RANGES['red_lower2'], HSV_RANGES['red_upper2'])
        mask3 = cv2.inRange(hsv, HSV_RANGES['orange_lower'], HSV_RANGES['orange_upper'])
        
        # Combine all masks
        mask = cv2.bitwise_or(mask1, mask2)
        mask = cv2.bitwise_or(mask, mask3)
        
        # Less aggressive morphology for low-quality video
        mask = cv2.erode(mask, self.kernel_small, iterations=1)
        mask = cv2.dilate(mask, self.kernel_medium, iterations=2)
        mask = cv2.erode(mask, self.kernel_small, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, mask
            
        # Find best contour with relaxed criteria
        best_contour = None
        best_score = 0
        best_bbox = None
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue
                
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Very relaxed aspect ratio check for compressed video
            aspect_ratio = h / w if w > 0 else 0
            
            # Accept wider range of aspect ratios
            if aspect_ratio < 0.5 or aspect_ratio > 2.5:
                continue
                
            # Score calculation (less strict)
            aspect_diff = abs(aspect_ratio - BUCKET_ASPECT_RATIO)
            aspect_score = max(0, 1.0 - aspect_diff)
            
            # Fill ratio (how much of bbox is filled)
            fill_ratio = area / (w * h) if w > 0 and h > 0 else 0
            
            # Size score (prefer larger objects)
            size_score = min(area / 2000, 1.0)
            
            # Position score (prefer centered objects)
            center_x = x + w // 2
            center_dist = abs(center_x - CAMERA_WIDTH // 2)
            position_score = max(0, 1.0 - center_dist / (CAMERA_WIDTH // 2))
            
            # Combined score with relaxed weighting
            score = (aspect_score * 0.2 + 
                    fill_ratio * 0.3 + 
                    size_score * 0.3 +
                    position_score * 0.2)
            
            if score > best_score:
                best_score = score
                best_contour = contour
                best_bbox = (x, y, w, h, score)
                
        if best_contour is not None and best_score > 0.25:  # Lower threshold
            return best_bbox, mask
            
        return None, mask
        
    def calculate_distance(self, height_pixels):
        """Calculate distance to bucket"""
        if height_pixels <= 0:
            return None
        distance = (BUCKET_HEIGHT_INCHES * FOCAL_LENGTH_PIXELS) / height_pixels
        return distance
        
    def calculate_angle(self, center_x):
        """Calculate horizontal angle to bucket"""
        offset_pixels = center_x - (CAMERA_WIDTH / 2)
        angle_radians = math.atan(offset_pixels / FOCAL_LENGTH_PIXELS)
        return math.degrees(angle_radians)
        
    def auto_center_turret(self, center_x, force=False):
        """Automatically center turret on target"""
        if not self.auto_center:
            return
            
        now = time.time()
        if not force and (now - self.last_center_time) < self.center_interval:
            return
            
        # Calculate offset from center
        offset = center_x - (CAMERA_WIDTH // 2)
        
        # Only adjust if significantly off-center
        if abs(offset) > self.center_threshold:
            angle = self.calculate_angle(center_x)
            steps = int(angle * 10000 / 360)  # 200 steps per rev
            
            # Limit step size for smooth tracking
            steps = max(-100, min(100, steps))
            
            if abs(steps) >= 2:
                # Send small adjustment
                steps *= -1
                cmd = f"A {steps}\n"
                self.ser.write(cmd.encode())
                self.last_center_time = now
                
                if self.debug_mode:
                    print(f"[Auto-center] Adjusting {steps} steps (angle: {angle:.1f}°)")
                    
    def calculate_firing_solution(self, distance, angle):
        """Calculate firing parameters"""
        launch_angle = 45
        power = 50
        
        for i in range(len(self.firing_table) - 1):
            if self.firing_table[i][0] <= distance <= self.firing_table[i+1][0]:
                d0, a0, p0 = self.firing_table[i]
                d1, a1, p1 = self.firing_table[i+1]
                ratio = (distance - d0) / (d1 - d0)
                launch_angle = a0 + ratio * (a1 - a0)
                power = p0 + ratio * (p1 - p0)
                break
                
        servo_us = int(np.interp(launch_angle, [20, 60], [1900, 1200]))
        steps = int(angle * 200 / 360)
        
        return {
            'distance': distance,
            'angle': angle,
            'launch_angle': launch_angle,
            'power': power,
            'servo_us': servo_us,
            'base_steps': steps
        }
        
    def execute_targeting(self):
        """Execute targeting sequence"""
        if not self.fire_solution:
            print("[!] No firing solution")
            return
            
        sol = self.fire_solution
        
        print("\n=== TARGETING SEQUENCE ===")
        print(f"Distance: {sol['distance']:.1f} inches ({sol['distance']/12:.1f} ft)")
        print(f"Angle: {sol['angle']:.1f}°")
        print(f"Launch: {sol['launch_angle']:.1f}°")
        print(f"Power: {sol['power']:.0f}%")
        
        # Turn off auto-centering during targeting
        self.auto_center = False
        
        # Final alignment
        if abs(sol['base_steps']) > 2:
            self.send_command(f"A {sol['base_steps']}")
            time.sleep(1)
            
        # Set launch angle
        self.send_command(f"US {sol['servo_us']}")
        time.sleep(0.5)
        
        # Load
        self.send_command("COORDLOAD")
        time.sleep(2)
        
        self.armed = True
        print("[ARMED] Ready to fire!")
        
    def run(self):
        """Main targeting loop"""
        print("\n=== ESP32-CAM CATAPULT TARGETING (ENHANCED) ===")
        print("\nControls:")
        print("  SPACE - Lock target & calculate solution")
        print("  t - Execute targeting sequence")
        print("  f - FIRE!")
        print("  r - Reset")
        print("  a - Toggle auto-centering")
        print("  d - Debug mode")
        print("  +/- - Adjust detection sensitivity")
        print("  q - Quit")
        print("\nStarting video stream...")
        
        # Start video stream
        self.send_raw("S")
        time.sleep(0.5)
        self.receiver.start()
        
        # Create windows
        cv2.namedWindow('ESP32 Targeting', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('ESP32 Targeting', 640, 480)
        
        last_frame = None
        frame_count = 0
        fps_time = time.time()
        fps = 0
        
        try:
            while True:
                # Get frame
                frame = self.receiver.get_frame()
                if frame is not None:
                    last_frame = frame.copy()
                    frame_count += 1
                    
                    if frame_count % 10 == 0:
                        fps = 10 / (time.time() - fps_time)
                        fps_time = time.time()
                        
                if last_frame is not None:
                    display = last_frame.copy()
                    
                    # Detect bucket
                    bbox_data, mask = self.detect_bucket(last_frame)
                    
                    # Process detection
                    if bbox_data is not None:
                        x, y, w, h, confidence = bbox_data
                        bbox = (x, y, w, h)
                        
                        # Calculate measurements
                        distance = self.calculate_distance(h)
                        center_x = x + w // 2
                        center_y = y + h // 2
                        angle = self.calculate_angle(center_x)
                        
                        # Update tracker
                        self.tracker.update(bbox, distance, angle, (center_x, center_y))
                        
                        # Auto-center if tracking
                        if self.tracker.is_tracking() and not self.armed:
                            self.auto_center_turret(center_x)
                            
                    else:
                        self.tracker.update(None, None, None, None)
                        
                    # Draw visualization
                    if self.tracker.is_tracking():
                        # Get smoothed values
                        smooth_distance = self.tracker.get_smooth_distance()
                        smooth_angle = self.tracker.get_smooth_angle()
                        smooth_center = self.tracker.get_smooth_center()
                        
                        # Draw detection box
                        if self.tracker.last_valid_bbox:
                            x, y, w, h = self.tracker.last_valid_bbox
                            color = (0, 255, 0) if self.tracker.lost_frames == 0 else (0, 165, 255)
                            cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
                            
                            # Draw tracking point
                            if smooth_center:
                                cv2.circle(display, smooth_center, 5, (255, 255, 0), -1)
                                
                        # Draw measurements
                        if smooth_distance and smooth_angle is not None:
                            text = f"{smooth_distance:.1f}in @ {smooth_angle:.1f}deg"
                            cv2.putText(display, text, (10, 90),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            
                            # Distance in feet
                            text_ft = f"{smooth_distance/12:.1f} ft"
                            cv2.putText(display, text_ft, (10, 110),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                            
                    # Draw HUD
                    h, w = display.shape[:2]
                    
                    # Center crosshair
                    cv2.line(display, (w//2-20, h//2), (w//2+20, h//2), (0, 255, 0), 1)
                    cv2.line(display, (w//2, h//2-20), (w//2, h//2+20), (0, 255, 0), 1)
                    cv2.circle(display, (w//2, h//2), 40, (0, 255, 0), 1)
                    
                    # Center zone indicator
                    if self.auto_center:
                        cv2.rectangle(display, 
                                    (w//2 - self.center_threshold, 0),
                                    (w//2 + self.center_threshold, h),
                                    (100, 100, 0), 1)
                        
                    # Status
                    if self.armed:
                        status = "ARMED"
                        color = (0, 0, 255)
                    elif self.tracker.is_tracking():
                        status = "TRACKING"
                        color = (0, 255, 0)
                    else:
                        status = "SEARCHING"
                        color = (0, 165, 255)
                        
                    cv2.putText(display, status, (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Auto-center indicator
                    if self.auto_center:
                        cv2.putText(display, "AUTO", (w-60, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        
                    # FPS
                    cv2.putText(display, f"FPS: {fps:.1f}", (10, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                    
                    # Stats
                    stats = self.receiver.get_stats()
                    cv2.putText(display, f"Frames: {stats['frames_received']}", (10, h-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    
                    # Show frames
                    cv2.imshow('ESP32 Targeting', display)
                    
                    if self.debug_mode and mask is not None:
                        # Resize mask for better visibility
                        mask_display = cv2.resize(mask, (640, 480))
                        cv2.imshow('Detection Mask', mask_display)
                        
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    # Lock target
                    if self.tracker.is_tracking():
                        distance = self.tracker.get_smooth_distance()
                        angle = self.tracker.get_smooth_angle()
                        if distance and angle is not None:
                            self.fire_solution = self.calculate_firing_solution(distance, angle)
                            print(f"\n[TARGET LOCKED] {distance:.1f}in @ {angle:.1f}°")
                            self.auto_center = False  # Stop auto-centering
                elif key == ord('t'):
                    # Execute targeting
                    self.execute_targeting()
                elif key == ord('f'):
                    # Fire
                    if self.armed:
                        self.send_command("FIRE")
                        self.armed = False
                        self.auto_center = True  # Resume auto-centering
                        print("[FIRE!]")
                elif key == ord('r'):
                    # Reset
                    self.send_command("RESET")
                    self.armed = False
                    self.auto_center = True
                    self.tracker = SmoothTracker(history_size=15)
                    print("[Reset]")
                elif key == ord('a'):
                    # Toggle auto-centering
                    self.auto_center = not self.auto_center
                    print(f"[Auto-center] {'ON' if self.auto_center else 'OFF'}")
                elif key == ord('d'):
                    # Debug mode
                    self.debug_mode = not self.debug_mode
                    if not self.debug_mode:
                        cv2.destroyWindow('Detection Mask')
                elif key == ord('+') or key == ord('='):
                    # Increase detection area
                    self.min_area = max(100, self.min_area - 50)
                    print(f"[Detection] Min area: {self.min_area}")
                elif key == ord('-'):
                    # Decrease detection area
                    self.min_area = min(1000, self.min_area + 50)
                    print(f"[Detection] Min area: {self.min_area}")
                elif key == ord('w'):
                    # Test Servo 1 going to ?
                    self.send_command("W 0")
                elif key == ord('0'):
                    # Test Servo 1 going to ?
                    self.send_command("W 190")
                elif key == ord('1'):
                    # Disconnect Servo 1
                    self.send_command("E")
                elif key == ord('2'):
                    # Reconnect Servo 1
                    self.send_command("R1")
                elif key == ord('3'):
                    # Disconnect Servo 2
                    self.send_command("Z")
                elif key == ord('4'):
                    # Reconnect Servo 2
                    self.send_command("R2")
                elif key == ord('5'):
                    # Test Servo 2 going to ?
                    self.send_command("L 170")
                elif key == ord('6'):
                    # Test Servo 3 going to ?
                    self.send_command("L 190")

                    
        except KeyboardInterrupt:
            print("\n[!] Interrupted")
        finally:
            # Cleanup
            self.send_raw("Q")
            self.receiver.stop()
            cv2.destroyAllWindows()
            self.send_command("RESET")
            self.ser.close()
            print("[System] Shutdown complete")


def main():
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='ESP32-CAM Catapult Targeting (Enhanced)')
    
    default_port = "/dev/ttyACM0" if os.name == 'posix' else "COM4"
    parser.add_argument('--port', default=default_port, help='Serial port')
    parser.add_argument('--baud', type=int, default=2000000, help='Baud rate')
    
    args = parser.parse_args()
    
    # Run targeting system
    system = ESP32CatapultTargeting(args.port, args.baud)
    system.run()


if __name__ == "__main__":
    main()
