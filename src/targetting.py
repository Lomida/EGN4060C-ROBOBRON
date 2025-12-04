#!/usr/bin/env python3

import cv2
import numpy as np
import serial
import struct
import time
import math
import threading
import queue
from collections import deque


CAMERA_FOV_DEGREES = 65.5
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
FOCAL_LENGTH_PIXELS = (CAMERA_WIDTH / 2) / math.tan(math.radians(CAMERA_FOV_DEGREES / 2))

BUCKET_HEIGHT_INCHES = 14.25
BUCKET_DIAMETER_INCHES = 11.25
BUCKET_ASPECT_RATIO = BUCKET_HEIGHT_INCHES / BUCKET_DIAMETER_INCHES

HSV_RANGES = {
    'red_lower1': np.array([0, 50, 30]),
    'red_upper1': np.array([15, 255, 255]),
    'red_lower2': np.array([165, 50, 30]),
    'red_upper2': np.array([180, 255, 255]),
    'orange_lower': np.array([10, 50, 50]),
    'orange_upper': np.array([25, 255, 255]),
}

# Distance lookup table: (distance_inches, servo1_restriction_angle)
# Servo1 at 0 degrees stops catapult at ~45 degrees backward
# Higher servo1 angles = lower stopping angles (more restriction)
DISTANCE_TABLE = [
    (24, 0),      # 2 ft  -> servo1 at 0 (stops at ~45 deg)
    (36, 20),     # 3 ft  -> servo1 at 20 (stops at ~35 deg)
    (48, 40),     # 4 ft  -> servo1 at 40 (stops at ~25 deg)
    (60, 60),     # 5 ft  -> servo1 at 60 (stops at ~15 deg)
    (72, 80),     # 6 ft  -> servo1 at 80 (stops at ~5 deg)
    (84, 100),    # 7 ft  -> servo1 at 100
    (96, 120),    # 8 ft  -> servo1 at 120
    (120, 150),   # 10 ft -> servo1 at 150
    (144, 180),   # 12 ft -> servo1 at 180
]


def interpolate_servo_angle(distance_inches):
    """Get servo1 restriction angle for a given distance using linear interpolation."""
    if distance_inches <= DISTANCE_TABLE[0][0]:
        return DISTANCE_TABLE[0][1]
    if distance_inches >= DISTANCE_TABLE[-1][0]:
        return DISTANCE_TABLE[-1][1]
    
    for i in range(len(DISTANCE_TABLE) - 1):
        d0, a0 = DISTANCE_TABLE[i]
        d1, a1 = DISTANCE_TABLE[i + 1]
        if d0 <= distance_inches <= d1:
            ratio = (distance_inches - d0) / (d1 - d0)
            return int(a0 + ratio * (a1 - a0))
    
    return DISTANCE_TABLE[-1][1]


class ESP32StreamReceiver:
    def __init__(self, serial_conn):
        self.ser = serial_conn
        self.buffer = bytearray()
        self.frame_queue = queue.Queue(maxsize=3)
        self.running = True
        self.thread = threading.Thread(target=self._receive_loop)
        self.thread.daemon = True
        self.stats = {'frames_received': 0, 'frames_dropped': 0, 'bad_frames': 0}

    def start(self):
        self.thread.start()
        print("[Stream] Receiver started")

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2)

    def _receive_loop(self):
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

                jpeg_data = bytes(self.buffer[12:12 + length])
                self.buffer = self.buffer[12 + length:]

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
        self.last_valid_bbox = None
        self.lost_frames = 0
        self.max_lost_frames = 10

    def update(self, bbox, distance, angle, center):
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
        if len(self.distance_history) < 3:
            return None if not self.distance_history else self.distance_history[-1]
        distances = list(self.distance_history)
        median = np.median(distances)
        mad = np.median([abs(d - median) for d in distances])
        if mad == 0:
            return median
        filtered = [d for d in distances if abs(d - median) <= 2 * mad]
        return np.mean(filtered) if filtered else median

    def get_smooth_angle(self):
        if not self.angle_history:
            return None
        return np.mean(list(self.angle_history))

    def get_smooth_center(self):
        if not self.position_history:
            return None
        centers = list(self.position_history)
        avg_x = int(np.mean([c[0] for c in centers]))
        avg_y = int(np.mean([c[1] for c in centers]))
        return (avg_x, avg_y)

    def is_tracking(self):
        return self.lost_frames < self.max_lost_frames and len(self.distance_history) > 0


class ESP32CatapultTargeting:
    def __init__(self, port, baud=2000000):
        self.ser = serial.Serial(port, baud, timeout=0.1)
        print(f"[System] Connected to {port}")

        self.receiver = ESP32StreamReceiver(self.ser)
        self.tracker = SmoothTracker(history_size=15)

        self.auto_center = True
        self.center_threshold = 10
        self.last_center_time = 0
        self.center_interval = 0.05

        self.armed = False
        self.fire_solution = None
        self.debug_mode = False
        self.loading = False

        self.min_area = 200
        self.max_area = 60000
        self.kernel_small = np.ones((3, 3), np.uint8)
        self.kernel_medium = np.ones((5, 5), np.uint8)

        time.sleep(2)
        self.send_command("RESET")

    def send_command(self, cmd):
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
        self.ser.write(char.encode())

    def detect_bucket(self, frame):
        blurred = cv2.GaussianBlur(frame, (5, 5), 1)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv, HSV_RANGES['red_lower1'], HSV_RANGES['red_upper1'])
        mask2 = cv2.inRange(hsv, HSV_RANGES['red_lower2'], HSV_RANGES['red_upper2'])
        mask3 = cv2.inRange(hsv, HSV_RANGES['orange_lower'], HSV_RANGES['orange_upper'])

        mask = cv2.bitwise_or(mask1, mask2)
        mask = cv2.bitwise_or(mask, mask3)

        mask = cv2.erode(mask, self.kernel_small, iterations=1)
        mask = cv2.dilate(mask, self.kernel_medium, iterations=2)
        mask = cv2.erode(mask, self.kernel_small, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, mask

        best_score = 0
        best_bbox = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = h / w if w > 0 else 0

            if aspect_ratio < 0.5 or aspect_ratio > 2.5:
                continue

            aspect_diff = abs(aspect_ratio - BUCKET_ASPECT_RATIO)
            aspect_score = max(0, 1.0 - aspect_diff)
            fill_ratio = area / (w * h) if w > 0 and h > 0 else 0
            size_score = min(area / 2000, 1.0)
            center_x = x + w // 2
            center_dist = abs(center_x - CAMERA_WIDTH // 2)
            position_score = max(0, 1.0 - center_dist / (CAMERA_WIDTH // 2))

            score = (aspect_score * 0.2 + fill_ratio * 0.3 + size_score * 0.3 + position_score * 0.2)

            if score > best_score:
                best_score = score
                best_bbox = (x, y, w, h, score)

        if best_bbox is not None and best_score > 0.25:
            return best_bbox, mask
        return None, mask

    def calculate_distance(self, height_pixels):
        if height_pixels <= 0:
            return None
        return (BUCKET_HEIGHT_INCHES * FOCAL_LENGTH_PIXELS) / height_pixels

    def calculate_angle(self, center_x):
        offset_pixels = center_x - (CAMERA_WIDTH / 2)
        angle_radians = math.atan(offset_pixels / FOCAL_LENGTH_PIXELS)
        return math.degrees(angle_radians)

    def auto_center_turret(self, center_x, force=False):
        if not self.auto_center:
            return

        now = time.time()
        if not force and (now - self.last_center_time) < self.center_interval:
            return

        offset = center_x - (CAMERA_WIDTH // 2)
        if abs(offset) > self.center_threshold:
            angle = self.calculate_angle(center_x)
            steps = int(angle * 10000 / 360)
            steps = max(-100, min(100, steps))

            if abs(steps) >= 2:
                steps *= -1
                cmd = f"A {steps}\n"
                self.ser.write(cmd.encode())
                self.last_center_time = now

    def calculate_firing_solution(self, distance, angle):
        restriction_angle = interpolate_servo_angle(distance)
        steps = int(angle * 200 / 360)

        return {
            'distance': distance,
            'angle': angle,
            'restriction_angle': restriction_angle,
            'base_steps': steps
        }

    def auto_load_and_fire(self):
        """
        Automatic load and fire sequence:
        1. Servo1 pulls back catapult to 190 degrees (load position)
        2. Servo2 latches at 190 degrees
        3. Servo1 moves to restriction angle (based on distance)
        4. Wait for fire command
        5. Servo2 unlatches at 170 degrees to release
        """
        if not self.fire_solution:
            print("[!] No firing solution - lock target first")
            return

        sol = self.fire_solution
        restriction_angle = sol['restriction_angle']

        print("\n=== AUTO LOAD SEQUENCE ===")
        print(f"Distance: {sol['distance']:.1f} inches ({sol['distance']/12:.1f} ft)")
        print(f"Restriction angle: {restriction_angle} degrees")

        self.loading = True
        self.auto_center = False

        # Final alignment
        if abs(sol['base_steps']) > 2:
            print(f"[1/5] Aligning turret ({sol['base_steps']} steps)...")
            self.send_command(f"A {sol['base_steps']}")
            time.sleep(1)

        # Servo1 pulls back to 190 (full pullback)
        print("[2/5] Pulling back catapult (Servo1 -> 190)...")
        self.send_command("W 190")
        time.sleep(2)

        # Servo2 latches at 190
        print("[3/5] Latching (Servo2 -> 190)...")
        self.send_command("L 190")
        time.sleep(1)

        # Servo1 moves to restriction position
        print(f"[4/5] Setting restriction (Servo1 -> {restriction_angle})...")
        self.send_command(f"W {restriction_angle}")
        time.sleep(1.5)

        self.loading = False
        self.armed = True
        print("[5/5] ARMED - Press 'f' to fire!")

    def fire(self):
        """Release the latch to fire."""
        if not self.armed:
            print("[!] Not armed - run load sequence first")
            return

        print("\n[FIRE!] Releasing latch (Servo2 -> 170)...")
        self.send_command("L 170")
        time.sleep(0.5)

        self.armed = False
        self.fire_solution = None
        self.auto_center = True
        print("[Complete] Ready for next target")

    def run(self):
        print("\n=== ESP32-CAM CATAPULT TARGETING ===")
        print("\nControls:")
        print("  SPACE - Lock target")
        print("  l     - Auto load sequence")
        print("  f     - FIRE")
        print("  r     - Reset")
        print("  a     - Toggle auto-centering")
        print("  d     - Debug mode")
        print("  +/-   - Adjust detection sensitivity")
        print("  q     - Quit")
        print("\nStarting video stream...")

        self.send_raw("S")
        time.sleep(0.5)
        self.receiver.start()

        cv2.namedWindow('ESP32 Targeting', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('ESP32 Targeting', 640, 480)

        last_frame = None
        frame_count = 0
        fps_time = time.time()
        fps = 0

        try:
            while True:
                frame = self.receiver.get_frame()
                if frame is not None:
                    last_frame = frame.copy()
                    frame_count += 1
                    if frame_count % 10 == 0:
                        fps = 10 / (time.time() - fps_time)
                        fps_time = time.time()

                if last_frame is not None:
                    display = last_frame.copy()
                    bbox_data, mask = self.detect_bucket(last_frame)

                    if bbox_data is not None:
                        x, y, w, h, confidence = bbox_data
                        bbox = (x, y, w, h)
                        distance = self.calculate_distance(h)
                        center_x = x + w // 2
                        center_y = y + h // 2
                        angle = self.calculate_angle(center_x)
                        self.tracker.update(bbox, distance, angle, (center_x, center_y))

                        if self.tracker.is_tracking() and not self.armed and not self.loading:
                            self.auto_center_turret(center_x)
                    else:
                        self.tracker.update(None, None, None, None)

                    if self.tracker.is_tracking():
                        smooth_distance = self.tracker.get_smooth_distance()
                        smooth_angle = self.tracker.get_smooth_angle()
                        smooth_center = self.tracker.get_smooth_center()

                        if self.tracker.last_valid_bbox:
                            x, y, w, h = self.tracker.last_valid_bbox
                            color = (0, 255, 0) if self.tracker.lost_frames == 0 else (0, 165, 255)
                            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
                            if smooth_center:
                                cv2.circle(display, smooth_center, 5, (255, 255, 0), -1)

                        if smooth_distance and smooth_angle is not None:
                            restriction = interpolate_servo_angle(smooth_distance)
                            text = f"{smooth_distance:.1f}in @ {smooth_angle:.1f}deg"
                            cv2.putText(display, text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            text_ft = f"{smooth_distance / 12:.1f} ft | S1:{restriction}deg"
                            cv2.putText(display, text_ft, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                    h_disp, w_disp = display.shape[:2]
                    cv2.line(display, (w_disp // 2 - 20, h_disp // 2), (w_disp // 2 + 20, h_disp // 2), (0, 255, 0), 1)
                    cv2.line(display, (w_disp // 2, h_disp // 2 - 20), (w_disp // 2, h_disp // 2 + 20), (0, 255, 0), 1)
                    cv2.circle(display, (w_disp // 2, h_disp // 2), 40, (0, 255, 0), 1)

                    if self.auto_center:
                        cv2.rectangle(display,
                                      (w_disp // 2 - self.center_threshold, 0),
                                      (w_disp // 2 + self.center_threshold, h_disp),
                                      (100, 100, 0), 1)

                    if self.loading:
                        status, color = "LOADING", (255, 165, 0)
                    elif self.armed:
                        status, color = "ARMED", (0, 0, 255)
                    elif self.tracker.is_tracking():
                        status, color = "TRACKING", (0, 255, 0)
                    else:
                        status, color = "SEARCHING", (0, 165, 255)

                    cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    if self.auto_center:
                        cv2.putText(display, "AUTO", (w_disp - 60, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    cv2.putText(display, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

                    stats = self.receiver.get_stats()
                    cv2.putText(display, f"Frames: {stats['frames_received']}", (10, h_disp - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                    cv2.imshow('ESP32 Targeting', display)

                    if self.debug_mode and mask is not None:
                        mask_display = cv2.resize(mask, (640, 480))
                        cv2.imshow('Detection Mask', mask_display)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord(' '):
                    if self.tracker.is_tracking():
                        distance = self.tracker.get_smooth_distance()
                        angle = self.tracker.get_smooth_angle()
                        if distance and angle is not None:
                            self.fire_solution = self.calculate_firing_solution(distance, angle)
                            restriction = self.fire_solution['restriction_angle']
                            print(f"\n[TARGET LOCKED] {distance:.1f}in @ {angle:.1f}deg | Restriction: {restriction}deg")
                            self.auto_center = False
                elif key == ord('l'):
                    self.auto_load_and_fire()
                elif key == ord('f'):
                    self.fire()
                elif key == ord('r'):
                    self.send_command("RESET")
                    self.armed = False
                    self.loading = False
                    self.fire_solution = None
                    self.auto_center = True
                    self.tracker = SmoothTracker(history_size=15)
                    print("[Reset]")
                elif key == ord('a'):
                    self.auto_center = not self.auto_center
                    print(f"[Auto-center] {'ON' if self.auto_center else 'OFF'}")
                elif key == ord('d'):
                    self.debug_mode = not self.debug_mode
                    if not self.debug_mode:
                        cv2.destroyWindow('Detection Mask')
                elif key == ord('+') or key == ord('='):
                    self.min_area = max(100, self.min_area - 50)
                    print(f"[Detection] Min area: {self.min_area}")
                elif key == ord('-'):
                    self.min_area = min(1000, self.min_area + 50)
                    print(f"[Detection] Min area: {self.min_area}")
                elif key == ord('w'):
                    self.send_command("W 0")
                elif key == ord('0'):
                    self.send_command("W 190")
                elif key == ord('1'):
                    self.send_command("E")
                elif key == ord('2'):
                    self.send_command("R1")
                elif key == ord('3'):
                    self.send_command("Z")
                elif key == ord('4'):
                    self.send_command("R2")
                elif key == ord('5'):
                    self.send_command("L 170")
                elif key == ord('6'):
                    self.send_command("L 190")

        except KeyboardInterrupt:
            print("\n[!] Interrupted")
        finally:
            self.send_raw("Q")
            self.receiver.stop()
            cv2.destroyAllWindows()
            self.send_command("RESET")
            self.ser.close()
            print("[System] Shutdown complete")


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description='ESP32-CAM Catapult Targeting')
    default_port = "/dev/ttyACM0" if os.name == 'posix' else "COM4"
    parser.add_argument('--port', default=default_port, help='Serial port')
    parser.add_argument('--baud', type=int, default=2000000, help='Baud rate')
    args = parser.parse_args()

    system = ESP32CatapultTargeting(args.port, args.baud)
    system.run()


if __name__ == "__main__":
    main()