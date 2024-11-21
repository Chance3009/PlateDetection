from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import math
import pyttsx3
import time

# Load self-trained YOLO model
model = YOLO(
    r"C:\Users\Chance\Desktop\PlateDetection\runs\detect\train5\weights\best.pt")

# Open the video file
# video_path = r"C:\Users\Chance\Desktop\ObjectDetection\video1.mp4"
# cap = cv2.VideoCapture(video_path)

# Live camera from laptop webcam
cap = cv2.VideoCapture(0)

# Stream from IP Webcam
# cap = cv2.VideoCapture("http://192.168.43.14:8080/video")

cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)

height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

print(height)
print(width)

engine = pyttsx3.init()

'''
# Divide grid regions
regions = {
    "FL": (0, 0, width//3, height//2),
    "FC": (width//3, 0, 2*width//3, height//2),
    "FR": (2*width//3, 0, width, height//2),
    "CL": (0, height//2, width//3, height),
    "CC": (width//3, height//2, 2*width//3, height),
    "CR": (2*width//3, height//2, width, height)
}

def get_region(x, y):
    for region, (x1, y1, x2, y2) in regions.items():
        if x1 <= x < x2 and y1 <= y < y2:
            return region
    return None
'''

# Determine the focal point
cx = width // 2
cy = height

# Define regions based on angles
regions = {
    "9 o'clock": (165, 180),
    "10 o'clock": (135, 165),
    "11 o'clock": (105, 135),
    "12 o'clock": (75, 105),
    "1 o'clock": (45, 75),
    "2 o'clock": (15, 45),
    "3 o'clock": (0, 15)
}


def get_region(center_x, center_y):
    dx = center_x - cx
    dy = cy - center_y
    angle = math.degrees(math.atan2(dy, dx))

    # Angle Normalization
    if angle < 0:
        angle += 360

    # Determine proximity based on distance
    distance = cy - center_y
    proximity = "near" if distance < 400 else "far"

    # Match angle to a region
    for region, (start_angle, end_angle) in regions.items():
        if start_angle <= angle < end_angle:
            return region, proximity

    return None, None


cooldown_time = 3
last_voice_time = 0

while True:
    # Read a frame from the video
    success, frame = cap.read()
    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame, conf=0.5)
        boxes = results[0].boxes
        num_plates = 0  # Counter for plates
        filtered_boxes = []  # To store only plate boxes

        # Filter detections for plates
        for box in boxes:
            class_id = int(box.cls[0])
            num_plates += 1
            x1, y1, x2, y2 = box.xyxy[0]
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            region = get_region(center_x, center_y)

            current_time = time.time()
            if region and current_time - last_voice_time >= cooldown_time:
                text = f"plate in {region}"

                cv2.putText(frame, text, (center_x, center_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_4)

                engine.say(text)
                engine.runAndWait()

                last_voice_time = current_time

            num_plates += 1

            # Draw circle at the center of the plate
            radius = 1

            cv2.circle(frame, (center_x, center_y), radius, (0, 0, 255), 2)
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (255, 0, 0), 2)        # Bounding box

            annotated_frame = results[0].plot()
            print(annotated_frame)
            # Add this box to the filtered_boxes to visualize later
            filtered_boxes.append(box)

        # Display the number of plates detected
        if num_plates == 0:  # No plates
            cv2.putText(frame, 'No plates detected', (20, 350),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_4)
        else:  # Show number of plates
            cv2.putText(frame, f'plates detected: {num_plates}', (
                20, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_4)

        print(f"Number of plates: {num_plates}")

        # Display the annotated frame
        cv2.imshow("YOLOv8 plate Detection", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
