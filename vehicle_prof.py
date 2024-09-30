import cv2
import numpy as np
import tensorflow as tf

# Load pre-trained MobileNet SSD model and the label map for object detection
MODEL_PATH = 'ssd_mobilenet_v2_fpnlite/saved_model'  # Update with actual model path
LABELS_PATH = 'mscoco_label_map.pbtxt'  # Update with label map path (e.g., for COCO dataset)

# Load the pre-trained model for vehicle detection
model = tf.saved_model.load(MODEL_PATH)

# Load COCO label names
def load_labels(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    labels = {}
    for line in lines:
        line = line.strip()
        if 'id' in line:
            key = int(line.split(': ')[1])
        if 'name' in line:
            value = line.split(': ')[1].replace("'", "")
            labels[key] = value
    return labels

labels = load_labels(LABELS_PATH)

# Function to perform object detection
def detect_vehicles(image):
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = model(input_tensor)

    # Extract bounding boxes, class labels, and confidence scores
    detection_boxes = detections['detection_boxes'][0].numpy()
    detection_classes = detections['detection_classes'][0].numpy().astype(np.int32)
    detection_scores = detections['detection_scores'][0].numpy()

    return detection_boxes, detection_classes, detection_scores

# Load the input video or image
video_path = 'input_video.mp4'  # Or use 'input_image.jpg' for an image
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame for faster processing (optional)
    image = cv2.resize(frame, (640, 640))

    # Detect vehicles
    boxes, classes, scores = detect_vehicles(image)

    # Loop through detections and draw bounding boxes
    for i in range(len(scores)):
        if scores[i] > 0.5:  # Confidence threshold
            class_id = classes[i]
            label = labels[class_id]

            if label in ['car', 'truck', 'motorcycle', 'bus']:  # Vehicle categories
                # Get bounding box coordinates
                box = boxes[i]
                (startY, startX, endY, endX) = (box[0] * image.shape[0], box[1] * image.shape[1],
                                                box[2] * image.shape[0], box[3] * image.shape[1])

                # Draw the bounding box and label
                cv2.rectangle(image, (int(startX), int(startY)), (int(endX), int(endY)), (0, 255, 0), 2)
                cv2.putText(image, label, (int(startX), int(startY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the image with detected vehicles
    cv2.imshow('Vehicle Profiling System', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import pytesseract

# Function to perform license plate recognition
def recognize_license_plate(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plate_text = pytesseract.image_to_string(gray_image)
    return plate_text

# Example usage after detecting a vehicle (crop license plate area)
plate_crop = image[int(startY):int(endY), int(startX):int(endX)]
plate_text = recognize_license_plate(plate_crop)
print("License Plate:", plate_text)
