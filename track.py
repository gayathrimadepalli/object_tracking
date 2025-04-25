import os
import cv2
import numpy as np
import time
import yaml
import argparse
from pathlib import Path
from collections import deque
from ultralytics import YOLO
from dotenv import load_dotenv

from tracker.deepsort import DeepSORT
from tracker.utils import extract_features_from_yolo
from utils.visualization import draw_boxes

# Initialize track history for visualization
track_history = {}
# Update the extract_features function definition
def extract_features(frame, boxes, model):
    return extract_features_from_yolo(model, frame, boxes)


def process_frame(frame, model, tracker, classes, conf_threshold=0.5, show=True):
    # Run YOLO model
    results = model(frame, conf=conf_threshold)

    # Extract detections
    detections = results[0].boxes

    # Check if there are any detections
    if len(detections) == 0:
        if show:
            return frame, []
        return frame, []

    # Get boxes, scores, class IDs
    boxes = detections.xyxy.cpu().numpy()
    scores = detections.conf.cpu().numpy()
    class_ids = detections.cls.cpu().numpy().astype(int)

    # Extract features
    features = extract_features(frame, boxes, model)

    # Update tracker
    outputs = tracker.update(boxes, scores, class_ids, features)

    # Process tracker outputs
    if len(outputs) > 0:
        # Convert list of dictionaries to arrays for drawing
        bbox_xyxy = np.array([output["bbox"] for output in outputs])
        track_ids = np.array([output["track_id"] for output in outputs])
        class_ids = np.array([output["class_id"] for output in outputs])

        # Extract confidence scores if available in tracker outputs
        if "score" in outputs[0]:
            track_scores = np.array([output["score"] for output in outputs])
        else:
            track_scores = None

        # Draw boxes and labels
        if show:
            frame = draw_boxes(frame, bbox_xyxy, track_ids, class_ids, classes, track_scores)
    else:
        # Return empty arrays if no tracks
        if show:
            return frame, []
        return frame, []

    return frame, outputs

def main():
    parser = argparse.ArgumentParser(description='Run YOLOv8 with DeepSORT tracking')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--source', type=str, help='Path to input video or webcam index')
    parser.add_argument('--output', type=str, default='output.mp4', help='Path to output video')
    parser.add_argument('--show', action='store_true', help='Display output while processing')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override source if provided
    if args.source:
        config['input']['source'] = args.source

    # Initialize model
    model_path = os.path.join('models', config['yolo']['model'])
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}, downloading...")
        from utils.download import download_yolo_model
        model_path = download_yolo_model(config['yolo']['model'])

    model = YOLO(model_path)

    # Get class names
    class_names = model.names

    # Initialize tracker
    # Initialize tracker
    tracker = DeepSORT(
        max_age=config['deepsort']['max_age'],
        n_init=config['deepsort']['min_hits'],  # Replace min_hits with n_init
        max_dist=config['deepsort']['max_cosine_distance'],  # Replace max_cosine_distance with max_dist
        max_iou_distance=0.7,  # Add default value or use from config
        nn_budget=100  # Add default value or use from config
    )
    # Open video source
    source = config['input']['source']
    if source.isdigit():
        source = int(source)  # Convert to int for webcam

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video source: {source}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Setup output video
    output_path = args.output
    if config['input']['save_results']:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )

    # Process video
    frame_count = 0
    processing_times = []

    print(f"Processing video: {source}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        # Process frame
        tracked_frame, outputs = process_frame(
            frame,
            model,
            tracker,
            class_names,
            show=True,
            conf_threshold=config['yolo']['confidence'],

        )

        # Calculate processing time
        processing_time = time.time() - start_time
        processing_times.append(processing_time)

        # Display FPS on frame
        fps_text = f"FPS: {1.0 / processing_time:.2f}"
        cv2.putText(tracked_frame, fps_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Save frame to output video
        if config['input']['save_results']:
            out.write(tracked_frame)

        # Display frame
        if args.show:
            cv2.imshow("YOLOv8 + DeepSORT Tracking", tracked_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames. Avg FPS: {1.0 / np.mean(processing_times[-100:]):.2f}")

    # Release resources
    cap.release()
    if config['input']['save_results']:
        out.release()
    if args.show:
        cv2.destroyAllWindows()

    # Print statistics
    avg_fps = 1.0 / np.mean(processing_times)
    print(f"\nProcessing completed:")
    print(f"Total frames: {frame_count}")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()