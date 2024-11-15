import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("/Users/madhuupadhyay/Documents/Stark_Industries/Heads-Up-Display/runs/detect/train10/weights/best.pt")

# Load the crosshair image (Atman_right.png) and resize it to use as a bounding box overlay
crosshair = cv2.imread('assets/Atman_right.png', cv2.IMREAD_UNCHANGED)  # Load with alpha channel

# Load the transparent box image for displaying text
transparent_box = cv2.imread('assets/Transparent_Box.png', cv2.IMREAD_UNCHANGED)

# Load the text data from Atman.txt
with open('Database/Atman.txt', 'r') as file:
    text_lines = file.readlines()

# Start capturing video from the webcam (source=0)
cap = cv2.VideoCapture(0)

# Define the target window size for smartphone viewing
target_width = 1920  # Adjust this width for VR headset use
target_height = 1080  # Adjust this height for VR headset use

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Get frame dimensions
    frame_height, frame_width, _ = frame.shape

    # Draw a blue circle at the center of the screen
    center_x, center_y = frame_width // 2, frame_height // 2
    cv2.circle(frame, (center_x, center_y), radius=10, color=(255, 0, 0), thickness=-1)  # Blue circle

    # Make predictions on the frame
    results = model(frame)

    # Get predictions from the results
    detections = results[0].boxes

    # Flag to check if center circle is inside any detection
    center_in_detection = False

    # Loop through each detection and overlay the crosshair image (bounding box)
    for detection in detections:
        if detection.conf.cpu().item() < 0.85:
            continue

        x1, y1, x2, y2 = map(int, detection.xyxy[0].cpu().numpy())  # Bounding box coordinates

        # Check if the center circle is inside the detection box
        if x1 <= center_x <= x2 and y1 <= center_y <= y2:
            center_in_detection = True

        # Calculate where to place the crosshair image (as bounding box overlay)
        top_left_x = x1
        top_left_y = y1
        width = x2 - x1
        height = y2 - y1

        # Resize the crosshair to match the size of the bounding box
        crosshair_resized = cv2.resize(crosshair, (width, height), interpolation=cv2.INTER_AREA)

        # Ensure the crosshair is not placed outside of the frame boundaries
        if top_left_x < 0 or top_left_y < 0 or (top_left_x + width) > frame.shape[1] or (top_left_y + height) > frame.shape[0]:
            continue

        # Extract region of interest (ROI) where crosshair will be placed
        roi = frame[top_left_y:top_left_y + height, top_left_x:top_left_x + width]

        # Split the crosshair into its color and alpha channel
        crosshair_rgb = crosshair_resized[:, :, :3]
        crosshair_alpha = crosshair_resized[:, :, 3] / 255.0

        # Blend the crosshair with the ROI based on alpha channel
        for c in range(0, 3):  # Loop over RGB channels
            roi[:, :, c] = (crosshair_alpha * crosshair_rgb[:, :, c] +
                            (1 - crosshair_alpha) * roi[:, :, c])

        # Replace the ROI in the frame with the blended image
        frame[top_left_y:top_left_y + height, top_left_x:top_left_x + width] = roi

    # If the center circle is inside Atman_right.png (detected bounding box), overlay the text
    if center_in_detection:
        # Resize the transparent box to accommodate the text
        box_height, box_width, _ = transparent_box.shape
        box_width_resized = 400
        scale = box_width_resized / box_width
        box_height_resized = int(box_height * scale)
        transparent_box_resized = cv2.resize(transparent_box, (box_width_resized, box_height_resized), interpolation=cv2.INTER_AREA)

        # Set text properties
        font = cv2.FONT_HERSHEY_COMPLEX
        font_scale = 0.7
        font_thickness = 1
        text_color = (255, 255, 255)  # White

        # Position to overlay the transparent box
        box_pos_x = 20
        box_pos_y = frame_height - box_height_resized

        # Overlay the transparent box on the frame
        box_roi = frame[box_pos_y:box_pos_y + box_height_resized, box_pos_x:box_pos_x + box_width_resized]

        # Split the transparent box into its color and alpha channel
        box_rgb = transparent_box_resized[:, :, :3]
        box_alpha = transparent_box_resized[:, :, 3] / 255.0

        # Blend the transparent box with the ROI based on alpha channel
        for c in range(0, 3):
            box_roi[:, :, c] = (box_alpha * box_rgb[:, :, c] +
                                (1 - box_alpha) * box_roi[:, :, c])

        frame[box_pos_y:box_pos_y + box_height_resized, box_pos_x:box_pos_x + box_width_resized] = box_roi

        # Overlay the text on the transparent box
        y_offset = box_pos_y + 80
        for line in text_lines:
            cv2.putText(frame, line.strip(), (box_pos_x + 30, y_offset), font, font_scale, text_color, font_thickness)
            y_offset += 30

    # Create a side-by-side view for VR
    sbs_frame = np.concatenate((frame, frame), axis=1)

    # Resize the side-by-side frame to the target window size
    resized_frame = cv2.resize(sbs_frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

    # Display the frame with crosshair overlays and text if applicable in a resized window
    cv2.imshow("Crosshair Overlay", resized_frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
