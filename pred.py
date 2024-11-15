import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("/Users/madhuupadhyay/Documents/Stark_Industries/Heads-Up-Display/runs/detect/train10/weights/best.pt")

# Load the crosshair image and resize it to 500x500
crosshair = cv2.imread('assets/crosshair.png', cv2.IMREAD_UNCHANGED)  # Load with alpha channel
crosshair = cv2.resize(crosshair, (500, 500), interpolation=cv2.INTER_AREA)
crosshair_height, crosshair_width, _ = crosshair.shape

# Start capturing video from the webcam (source=0)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Make predictions on the frame
    results = model(frame)

    # Get predictions from the results
    detections = results[0].boxes

    # Loop through each detection and overlay the crosshair
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection.xyxy[0].cpu().numpy())  # Bounding box coordinates

        # Calculate the center of the bounding box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Calculate where to place the crosshair
        top_left_x = center_x - crosshair_width // 2
        top_left_y = center_y - crosshair_height // 2

        # Ensure the crosshair is not placed outside of the frame boundaries
        if top_left_x < 0 or top_left_y < 0 or (top_left_x + crosshair_width) > frame.shape[1] or (top_left_y + crosshair_height) > frame.shape[0]:
            continue

        # Extract region of interest (ROI) where crosshair will be placed
        roi = frame[top_left_y:top_left_y + crosshair_height, top_left_x:top_left_x + crosshair_width]

        # Split the crosshair into its color and alpha channel
        crosshair_rgb = crosshair[:, :, :3]
        crosshair_alpha = crosshair[:, :, 3] / 255.0

        # Blend the crosshair with the ROI based on alpha channel
        for c in range(0, 3):  # Loop over RGB channels
            roi[:, :, c] = (crosshair_alpha * crosshair_rgb[:, :, c] +
                            (1 - crosshair_alpha) * roi[:, :, c])

        # Replace the ROI in the frame with the blended image
        frame[top_left_y:top_left_y + crosshair_height, top_left_x:top_left_x + crosshair_width] = roi

    # Display the frame with crosshair overlays
    cv2.imshow("Crosshair Overlay", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
