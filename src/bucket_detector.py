#!/usr/bin/env python3
"""
Harbor Freight Red Bucket Detection and Distance Measurement
Uses color detection and known bucket dimensions for distance calculation
Camera: OV3660 with 66.5° FOV
Bucket: 14.25" tall, 11.25" diameter
"""

import cv2
import numpy as np
import math
import time
from datetime import datetime

# ---- CAMERA CALIBRATION ----
CAMERA_FOV_DEGREES = 66.5  # OV3660 FOV
CAMERA_WIDTH = 320  # QVGA width
CAMERA_HEIGHT = 240  # QVGA height

# Calculate focal length in pixels (assuming no distortion)
FOCAL_LENGTH_PIXELS = (CAMERA_WIDTH / 2) / math.tan(math.radians(CAMERA_FOV_DEGREES / 2))

# ---- BUCKET SPECIFICATIONS ----
BUCKET_HEIGHT_INCHES = 14.25
BUCKET_DIAMETER_INCHES = 11.25
BUCKET_ASPECT_RATIO = BUCKET_HEIGHT_INCHES / BUCKET_DIAMETER_INCHES  # ~1.27

# ---- COLOR RANGES FOR RED BUCKET ----
# Harbor Freight bucket is typically bright red/orange
# These ranges work well for the typical HF red bucket
HSV_RANGES = {
    'red_lower1': np.array([0, 120, 70]),     # Lower red range
    'red_upper1': np.array([10, 255, 255]),   # Upper red range
    'red_lower2': np.array([170, 120, 70]),   # Wrap-around red
    'red_upper2': np.array([180, 255, 255]),  # Wrap-around red
}

# Alternative ranges for different lighting
HSV_RANGES_BRIGHT = {
    'red_lower1': np.array([0, 100, 100]),
    'red_upper1': np.array([10, 255, 255]),
    'red_lower2': np.array([160, 100, 100]),
    'red_upper2': np.array([180, 255, 255]),
}

HSV_RANGES_DIM = {
    'red_lower1': np.array([0, 50, 50]),
    'red_upper1': np.array([15, 255, 255]),
    'red_lower2': np.array([165, 50, 50]),
    'red_upper2': np.array([180, 255, 255]),
}


class BucketDetector:
    def __init__(self, debug=False, lighting='normal'):
        """
        Initialize bucket detector
        
        Args:
            debug: Show debug windows and info
            lighting: 'normal', 'bright', or 'dim'
        """
        self.debug = debug
        self.focal_length = FOCAL_LENGTH_PIXELS
        
        # Select HSV ranges based on lighting
        if lighting == 'bright':
            self.hsv_ranges = HSV_RANGES_BRIGHT
        elif lighting == 'dim':
            self.hsv_ranges = HSV_RANGES_DIM
        else:
            self.hsv_ranges = HSV_RANGES
            
        # Detection parameters
        self.min_area = 500  # Minimum contour area
        self.max_area = 50000  # Maximum contour area
        
        # Smoothing parameters
        self.distance_history = []
        self.history_size = 5
        
        # Morphological operation kernels
        self.kernel_erode = np.ones((3, 3), np.uint8)
        self.kernel_dilate = np.ones((5, 5), np.uint8)
        
        print(f"[BucketDetector] Initialized")
        print(f"  Camera FOV: {CAMERA_FOV_DEGREES}°")
        print(f"  Focal length: {self.focal_length:.1f} pixels")
        print(f"  Bucket: {BUCKET_HEIGHT_INCHES}\" x {BUCKET_DIAMETER_INCHES}\"")
        print(f"  Lighting mode: {lighting}")
        
    def detect_red_regions(self, frame):
        """
        Detect red regions in the frame using HSV color space
        
        Returns:
            Binary mask of red regions
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create masks for both red ranges (red wraps around in HSV)
        mask1 = cv2.inRange(hsv, self.hsv_ranges['red_lower1'], self.hsv_ranges['red_upper1'])
        mask2 = cv2.inRange(hsv, self.hsv_ranges['red_lower2'], self.hsv_ranges['red_upper2'])
        
        # Combine masks
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Clean up the mask
        mask = cv2.erode(mask, self.kernel_erode, iterations=1)
        mask = cv2.dilate(mask, self.kernel_dilate, iterations=2)
        mask = cv2.erode(mask, self.kernel_erode, iterations=1)
        
        # Apply Gaussian blur to smooth edges
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
        
        return mask
        
    def find_bucket_contour(self, mask):
        """
        Find the contour most likely to be the bucket
        
        Returns:
            Best contour and its properties (x, y, w, h, area)
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None
            
        best_score = 0
        best_contour = None
        best_props = None
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < self.min_area or area > self.max_area:
                continue
                
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate aspect ratio
            aspect_ratio = h / w if w > 0 else 0
            
            # Score based on how close to expected aspect ratio
            aspect_score = 1.0 - abs(aspect_ratio - BUCKET_ASPECT_RATIO) / BUCKET_ASPECT_RATIO
            if aspect_score < 0:
                aspect_score = 0
                
            # Score based on fill ratio (how much of bounding box is filled)
            fill_ratio = area / (w * h) if w > 0 and h > 0 else 0
            
            # Score based on size (larger is better, but not too large)
            size_score = min(area / 5000, 1.0)
            
            # Combined score
            score = aspect_score * 0.5 + fill_ratio * 0.3 + size_score * 0.2
            
            if score > best_score:
                best_score = score
                best_contour = contour
                best_props = (x, y, w, h, area, aspect_ratio, fill_ratio, score)
                
        return best_contour, best_props
        
    def calculate_distance(self, height_pixels):
        """
        Calculate distance to bucket using pinhole camera model
        
        Distance = (Real Height * Focal Length) / Pixel Height
        
        Args:
            height_pixels: Height of bucket in pixels
            
        Returns:
            Distance in inches
        """
        if height_pixels <= 0:
            return None
            
        distance = (BUCKET_HEIGHT_INCHES * self.focal_length) / height_pixels
        
        # Add to history for smoothing
        self.distance_history.append(distance)
        if len(self.distance_history) > self.history_size:
            self.distance_history.pop(0)
            
        # Return average of recent measurements
        return sum(self.distance_history) / len(self.distance_history)
        
    def calculate_angle(self, center_x):
        """
        Calculate horizontal angle to bucket from camera center
        
        Args:
            center_x: X coordinate of bucket center
            
        Returns:
            Angle in degrees (negative = left, positive = right)
        """
        offset_pixels = center_x - (CAMERA_WIDTH / 2)
        angle_radians = math.atan(offset_pixels / self.focal_length)
        return math.degrees(angle_radians)
        
    def process_frame(self, frame):
        """
        Process a single frame to detect bucket and calculate distance
        
        Args:
            frame: BGR image from camera
            
        Returns:
            Dictionary with detection results
        """
        result = {
            'detected': False,
            'distance': None,
            'angle': None,
            'center': None,
            'bbox': None,
            'confidence': 0
        }
        
        # Detect red regions
        mask = self.detect_red_regions(frame)
        
        # Find bucket contour
        contour, props = self.find_bucket_contour(mask)
        
        if contour is not None and props is not None:
            x, y, w, h, area, aspect_ratio, fill_ratio, score = props
            
            # Calculate distance
            distance = self.calculate_distance(h)
            
            # Calculate angle
            center_x = x + w // 2
            center_y = y + h // 2
            angle = self.calculate_angle(center_x)
            
            # Update result
            result['detected'] = True
            result['distance'] = distance
            result['angle'] = angle
            result['center'] = (center_x, center_y)
            result['bbox'] = (x, y, w, h)
            result['confidence'] = score
            
            # Draw on frame if debug
            if self.debug:
                # Draw contour
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                # Draw center point
                cv2.circle(frame, (center_x, center_y), 5, (0, 255, 255), -1)
                
                # Add text annotations
                if distance:
                    text_distance = f"{distance:.1f} in ({distance/12:.1f} ft)"
                    cv2.putText(frame, text_distance, (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
                text_angle = f"{angle:.1f} deg"
                cv2.putText(frame, text_angle, (x, y + h + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                text_conf = f"Conf: {score:.2f}"
                cv2.putText(frame, text_conf, (x, y + h + 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                           
        # Show debug windows if enabled
        if self.debug:
            cv2.imshow('Original', frame)
            cv2.imshow('Red Mask', mask)
            
        return result, frame
        
    def calculate_firing_solution(self, distance_inches, angle_degrees):
        """
        Calculate firing parameters for the catapult
        
        Args:
            distance_inches: Distance to target
            angle_degrees: Horizontal angle to target
            
        Returns:
            Dictionary with firing parameters
        """
        # These values need calibration for your specific catapult
        # This is a starting point based on typical projectile physics
        
        # Convert to useful units
        distance_feet = distance_inches / 12.0
        
        # Estimate required launch angle (45° is optimal for max range in vacuum)
        # Adjust for air resistance and catapult characteristics
        if distance_feet < 3:
            launch_angle = 30  # Shallow for close targets
        elif distance_feet < 6:
            launch_angle = 45  # Optimal for medium range
        else:
            launch_angle = 50  # Higher for long range
            
        # Estimate required power (0-100%)
        # This is highly dependent on your catapult design
        power = min(100, distance_feet * 15)  # Simple linear model
        
        # Calculate servo positions
        # Map launch angle to servo microseconds
        arc_servo_us = int(np.interp(launch_angle, [20, 60], [1200, 2000]))
        
        # Map horizontal angle to stepper steps
        # Assuming 200 steps per revolution (1.8° per step)
        steps_per_degree = 200 / 360.0
        base_steps = int(angle_degrees * steps_per_degree)
        
        return {
            'distance_feet': distance_feet,
            'horizontal_angle': angle_degrees,
            'launch_angle': launch_angle,
            'power': power,
            'arc_servo_us': arc_servo_us,
            'base_steps': base_steps,
            'ready': True
        }


def main():
    """
    Main function for standalone testing
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Bucket Detection for Catapult Targeting')
    parser.add_argument('--video', type=int, default=0, help='Video device number')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--lighting', choices=['normal', 'bright', 'dim'], 
                       default='normal', help='Lighting conditions')
    parser.add_argument('--calibrate', action='store_true', 
                       help='Show HSV calibration trackbars')
    args = parser.parse_args()
    
    # Initialize detector
    detector = BucketDetector(debug=args.debug, lighting=args.lighting)
    
    # Open video capture
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    
    print("\n[Controls]")
    print("  q - Quit")
    print("  d - Toggle debug")
    print("  s - Save screenshot")
    print("  SPACE - Calculate firing solution")
    
    # HSV Calibration trackbars
    if args.calibrate:
        cv2.namedWindow('HSV Calibration')
        cv2.createTrackbar('H Min', 'HSV Calibration', 0, 180, lambda x: None)
        cv2.createTrackbar('S Min', 'HSV Calibration', 120, 255, lambda x: None)
        cv2.createTrackbar('V Min', 'HSV Calibration', 70, 255, lambda x: None)
        cv2.createTrackbar('H Max', 'HSV Calibration', 10, 180, lambda x: None)
        cv2.createTrackbar('S Max', 'HSV Calibration', 255, 255, lambda x: None)
        cv2.createTrackbar('V Max', 'HSV Calibration', 255, 255, lambda x: None)
    
    frame_count = 0
    fps_start = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
                
            # Update HSV ranges if calibrating
            if args.calibrate:
                h_min = cv2.getTrackbarPos('H Min', 'HSV Calibration')
                s_min = cv2.getTrackbarPos('S Min', 'HSV Calibration')
                v_min = cv2.getTrackbarPos('V Min', 'HSV Calibration')
                h_max = cv2.getTrackbarPos('H Max', 'HSV Calibration')
                s_max = cv2.getTrackbarPos('S Max', 'HSV Calibration')
                v_max = cv2.getTrackbarPos('V Max', 'HSV Calibration')
                
                detector.hsv_ranges['red_lower1'] = np.array([h_min, s_min, v_min])
                detector.hsv_ranges['red_upper1'] = np.array([h_max, s_max, v_max])
                
            # Process frame
            result, annotated_frame = detector.process_frame(frame)
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_start)
                fps_start = time.time()
                
                # Add FPS to frame
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show status
            status = "TRACKING" if result['detected'] else "SEARCHING"
            color = (0, 255, 0) if result['detected'] else (0, 0, 255)
            cv2.putText(annotated_frame, status, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Display results
            cv2.imshow('Bucket Detection', annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('d'):
                detector.debug = not detector.debug
                print(f"Debug mode: {detector.debug}")
            elif key == ord('s'):
                filename = f"bucket_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(filename, annotated_frame)
                print(f"Saved: {filename}")
            elif key == ord(' ') and result['detected']:
                # Calculate and display firing solution
                solution = detector.calculate_firing_solution(
                    result['distance'], 
                    result['angle']
                )
                
                print("\n=== FIRING SOLUTION ===")
                print(f"Target Distance: {solution['distance_feet']:.1f} ft")
                print(f"Horizontal Angle: {solution['horizontal_angle']:.1f}°")
                print(f"Launch Angle: {solution['launch_angle']:.0f}°")
                print(f"Power: {solution['power']:.0f}%")
                print(f"Arc Servo: {solution['arc_servo_us']} us")
                print(f"Base Steps: {solution['base_steps']}")
                print("=" * 23)
                
    except KeyboardInterrupt:
        print("\n[!] Interrupted")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
