import cv2
import mediapipe as mp
import numpy as np
import time

# MediaPipe imports for the new task-based API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Path of landmark model
MODEL_PATH = '/Users/rylandalpez/Documents/repos/models/hand_landmarker.task'

# Global variable to store the latest results
latest_result = None
latest_output_image = None


# Callback function to handle results from the live stream mode
def result_callback(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result, latest_output_image
    latest_result = result
    latest_output_image = output_image


# Create hand landmarker instance with live stream mode
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=result_callback
)

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Get video properties for better performance
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Webcam FPS: {fps}")

print("Press 'q' to quit")
print("Make sure you have downloaded 'hand_landmarker.task' model file")

with HandLandmarker.create_from_options(options) as landmarker:
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Calculate timestamp in milliseconds
        timestamp_ms = int(time.time() * 1000)

        # Send frame to landmarker asynchronously
        landmarker.detect_async(mp_image, timestamp_ms)

        # Draw results if available
        if latest_result is not None and latest_result.hand_landmarks:
            for hand_landmarks in latest_result.hand_landmarks:
                # Convert normalized coordinates to pixel coordinates
                h, w, _ = frame.shape

                # Draw landmarks
                for landmark in hand_landmarks:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                # Draw connections between landmarks
                connections = [
                    # Thumb
                    (0, 1), (1, 2), (2, 3), (3, 4),
                    # Index finger
                    (0, 5), (5, 6), (6, 7), (7, 8),
                    # Middle finger
                    (0, 9), (9, 10), (10, 11), (11, 12),
                    # Ring finger
                    (0, 13), (13, 14), (14, 15), (15, 16),
                    # Pinky
                    (0, 17), (17, 18), (18, 19), (19, 20)
                ]

                for connection in connections:
                    start_idx, end_idx = connection
                    if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                        start_point = hand_landmarks[start_idx]
                        end_point = hand_landmarks[end_idx]

                        start_x = int(start_point.x * w)
                        start_y = int(start_point.y * h)
                        end_x = int(end_point.x * w)
                        end_y = int(end_point.y * h)

                        cv2.line(frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)

            # Display handedness
            if latest_result.handedness:
                for i, handedness in enumerate(latest_result.handedness):
                    if handedness:
                        hand_label = handedness[0].category_name
                        confidence = handedness[0].score
                        cv2.putText(frame, f'{hand_label}: {confidence:.2f}',
                                    (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 255, 255), 2)

        # Display frame count for performance monitoring
        frame_count += 1
        cv2.putText(frame, f'Frame: {frame_count}', (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Display the frame
        cv2.imshow('Hand Detection - Live Stream', frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Clean up
cap.release()
cv2.destroyAllWindows()