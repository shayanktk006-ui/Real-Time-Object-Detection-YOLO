# Import required libraries
from pathlib import Path
import cv2
from collections import Counter
from ultralytics import YOLO

# Define model path
MODEL_PATH = Path("yolo11n.pt")

# Validate model file
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"YOLO model not found: {MODEL_PATH}"
    )

print("YOLO model file found!")

# Load YOLO model
model = YOLO(str(MODEL_PATH))

print("YOLO model loaded successfully!")

# Run YOLO detection on a frame
def detect_objects(frame, confidence=0.60):
    results = model(frame, conf=confidence)
    return results[0]

    # Draw detection results on the frame
def draw_detections(frame, result):
    return result.plot()

# Save the processed detection result
def save_detection_result(image, filename):
    success = cv2.imwrite(filename, image)

    if not success:
        raise RuntimeError(
            f"Failed to save image: {filename}"
        )

    print(f"Detection result saved: {filename}")

    # Process a single frame
def process_frame(frame, confidence=0.60):
    result = detect_objects(
        frame,
        confidence
    )

    annotated_frame = draw_detections(
        frame,
        result
    )

    return annotated_frame

    # Run real-time object detection
def run_webcam():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam.")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                raise RuntimeError("Unable to read webcam frame.")

            annotated_frame = process_frame(
                frame,
                confidence=0.60
            )

            cv2.imshow(
                "YOLO Real-Time Detection",
                annotated_frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

        # Start the application
if __name__ == "__main__":
    run_webcam()