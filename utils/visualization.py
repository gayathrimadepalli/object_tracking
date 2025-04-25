import cv2
import numpy as np
import random

# Create a color map for visualization
color_map = {}


def get_color(idx):
    """Get a color for visualization based on the track ID."""
    if idx not in color_map:
        color_map[idx] = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
    return color_map[idx]


def draw_boxes(frame, bbox_xyxy, identities=None, class_ids=None, class_names=None, confidences=None):
    """
    Draw bounding boxes and labels on frame

    Args:
        frame: Image
        bbox_xyxy: Bounding boxes in xyxy format
        identities: Track IDs
        class_ids: Class IDs
        class_names: Class names dictionary
        confidences: Detection confidences (optional)

    Returns:
        Frame with drawn boxes and labels
    """
    frame_height, frame_width = frame.shape[:2]

    # Generate random colors for each identity
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)

    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]

        # Get track ID and class
        track_id = int(identities[i]) if identities is not None else 0
        class_id = int(class_ids[i]) if class_ids is not None else -1
        class_name = class_names[class_id] if class_names is not None and class_id >= 0 else "unknown"

        # Get color for this track
        color = [int(c) for c in COLORS[track_id % len(COLORS)]]

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        label_text = f"{class_name}-{track_id}"
        if confidences is not None and i < len(confidences):
            # Only add confidence if available
            try:
                conf_value = float(confidences[i])
                label_text = f"{label_text} {conf_value:.2f}"
            except (ValueError, TypeError):
                # If confidence conversion fails, just use the label without confidence
                pass

        # Get text size
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

        # Draw label background
        cv2.rectangle(frame, (x1, y1 - text_size[1] - 4), (x1 + text_size[0] + 2, y1), color, -1)

        # Draw label text
        cv2.putText(frame, label_text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame