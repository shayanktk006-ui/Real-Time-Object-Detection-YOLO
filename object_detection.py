# Import required libraries
from pathlib import Path
from collections import Counter
import time

import cv2
from ultralytics import YOLO


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = Path("yolo11n.pt")
CONFIDENCE_THRESHOLD = 0.60
CAMERA_INDEX = 0
WINDOW_NAME = "YOLO Real-Time Object Detection"


# ============================================================
# Model Setup
# ============================================================

def load_model(model_path):
    """Load and validate the YOLO model."""

    if not model_path.exists():
        raise FileNotFoundError(
            f"YOLO model not found: {model_path}"
        )

    print(f"Loading YOLO model: {model_path}")

    model = YOLO(str(model_path))

    print("YOLO model loaded successfully!")

    return model


# ============================================================
# Object Detection
# ============================================================

def detect_objects(frame, model, confidence=CONFIDENCE_THRESHOLD):
    """Run YOLO object detection on a frame."""

    results = model(
        frame,
        conf=confidence,
        verbose=False
    )

    return results[0]


# ============================================================
# Draw Detection Results
# ============================================================

def draw_detections(frame, result):
    """Draw YOLO bounding boxes and labels."""

    return result.plot()


# ============================================================
# Count Detected Objects
# ============================================================

def count_objects(result, model):
    """Count detected objects by class."""

    class_ids = result.boxes.cls

    return Counter(
        model.names[int(class_id)]
        for class_id in class_ids
    )


# ============================================================
# Display Object Counter
# ============================================================

def draw_object_counter(frame, class_counts):
    """Display detected object counts on the frame."""

    y_position = 40

    for class_name, count in class_counts.items():

        cv2.putText(
            frame,
            f"{class_name}: {count}",
            (20, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        y_position += 30

    return y_position


# ============================================================
# Display FPS
# ============================================================

def draw_fps(frame, fps, y_position):
    """Display FPS on the frame."""

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, y_position + 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


# ============================================================
# Webcam Detection
# ============================================================

def run_webcam(model):
    """Run real-time YOLO object detection using webcam."""

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError(
            "Unable to open webcam."
        )

    previous_time = time.time()

    print("Webcam started successfully.")
    print("Press Q or ESC to exit.")

    try:

        while True:

            # Capture webcam frame
            ret, frame = cap.read()

            if not ret:
                raise RuntimeError(
                    "Unable to read webcam frame."
                )

            # Run YOLO detection
            result = detect_objects(
                frame,
                model
            )

            # Count detected objects
            class_counts = count_objects(
                result,
                model
            )

            # Draw YOLO detections
            annotated_frame = draw_detections(
                frame,
                result
            )

            # Calculate FPS
            current_time = time.time()

            time_difference = (
                current_time - previous_time
            )

            fps = (
                1 / time_difference
                if time_difference > 0
                else 0
            )

            previous_time = current_time

            # Display object counter
            y_position = draw_object_counter(
                annotated_frame,
                class_counts
            )

            # Display FPS
            draw_fps(
                annotated_frame,
                fps,
                y_position
            )

            # Show final frame
            cv2.imshow(
                WINDOW_NAME,
                annotated_frame
            )

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

    finally:

        # Release webcam resources
        cap.release()

        # Close OpenCV windows
        cv2.destroyAllWindows()

        print("Webcam stopped successfully.")


# ============================================================
# Main Application
# ============================================================

def main():
    """Start the YOLO real-time detection system."""

    model = load_model(
        MODEL_PATH
    )

    run_webcam(
        model
    )


# ============================================================
# Application Entry Point
# ============================================================

if __name__ == "__main__":
    main()
