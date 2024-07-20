import torch
import cv2
import numpy as np

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Function to detect objects, excluding humans
def detect_objects(image):
    # Perform detection
    results = model(image)
    # Extract bounding box coordinates and class IDs
    boxes = results.xyxy[0].numpy()
    # Filter out humans (class ID for person is 0)
    filtered_boxes = [box for box in boxes if int(box[5]) != 0]
    return filtered_boxes

# Function to measure objects
def measure_objects(image, boxes, pixels_per_cm):
    measurements = []

    for box in boxes:
        x1, y1, x2, y2, confidence, class_id = box
        width = (x2 - x1) / pixels_per_cm
        height = (y2 - y1) / pixels_per_cm
        measurements.append((width, height))

        # Draw bounding box and dimensions on image
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = f'{model.names[int(class_id)]} W: {width:.2f} cm, H: {height:.2f} cm'
        cv2.putText(image, label, (int(x1), int(y2) + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Resize the image for better display
    scale_percent = 150  # percentage of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    cv2.imshow('Object Measurements', resized_image)
    cv2.waitKey(1)

    return measurements

# Function to calculate pixels per centimeter using a known reference object
def calculate_pixels_per_cm(image, reference_width_cm, reference_box):
    x1, y1, x2, y2 = reference_box
    reference_width_pixels = x2 - x1
    pixels_per_cm = reference_width_pixels / reference_width_cm
    return pixels_per_cm



# Function for real-time detection using webcam
def real_time_detection(reference_width_cm, reference_box):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    pixels_per_cm = calculate_pixels_per_cm(frame, reference_width_cm, reference_box)

    while True:
        ret, frame = cap.read() 
        frame = cv2.flip(frame,1)
        if not ret:
            break

        boxes = detect_objects(frame)
        measure_objects(frame, boxes, pixels_per_cm)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Known reference object dimensions (in cm) and its bounding box (x1, y1, x2, y2)
    reference_width_cm = 10.0  # Replace with the actual width of the reference object
    reference_box = (50, 50, 150, 150)  # Replace with the actual bounding box of the reference object in the image

    # For real-time detection using webcam:
    real_time_detection(reference_width_cm, reference_box)
