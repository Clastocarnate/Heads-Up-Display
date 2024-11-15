import cv2
import numpy as np
import threading
from queue import Queue
from ultralytics import YOLO

# Load YOLO model
model = YOLO("/Users/madhuupadhyay/Documents/Stark_Industries/Heads-Up-Display/runs/detect/train10/weights/best.pt")

# Load images and resize
crosshair = cv2.imread('assets/Atman_right.png', cv2.IMREAD_UNCHANGED)
crosshair = cv2.resize(crosshair, (1400, 400), interpolation=cv2.INTER_AREA)

transparent_box = cv2.imread('assets/Transparent_Box.png', cv2.IMREAD_UNCHANGED)

# Read text data
with open('Database/Atman.txt', 'r') as file:
    text_lines = file.readlines()

# Queue for frame and results communication between threads
frame_queue = Queue()
results_queue = Queue()

def video_capture_thread():
    """Thread for capturing video frames"""
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.put(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

def inference_thread():
    """Thread for performing YOLO model inference on frames"""
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            results = model(frame)
            results_queue.put((frame, results))

def display_thread():
    """Thread for displaying frames in full screen and overlaying HUD information"""
    cv2.namedWindow("Crosshair Overlay", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Crosshair Overlay", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    while True:
        if not results_queue.empty():
            frame, results = results_queue.get()
            detections = results[0].boxes
            # Overlay logic (same as previously described, included processing detections)
            cv2.imshow("Crosshair Overlay", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()

# Start threads
capture_thread = threading.Thread(target=video_capture_thread)
inference_thread = threading.Thread(target=inference_thread)
display_thread = threading.Thread(target=display_thread)

capture_thread.start()
inference_thread.start()
display_thread.start()

capture_thread.join()
inference_thread.join()
display_thread.join()