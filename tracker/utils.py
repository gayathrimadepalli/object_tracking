import numpy as np
import cv2


def xyxy2xywh(x):
    """
    Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h] where x, y are the center coordinates
    """
    # Convert to numpy array if it's a list or tuple
    if isinstance(x, (list, tuple)):
        x = np.array(x, dtype=np.float32)

    # Handle 1D array (single box)
    if len(x.shape) == 1:
        # For a single box [x1, y1, x2, y2], convert to [center_x, center_y, width, height]
        return np.array([
            (x[0] + x[2]) / 2,  # center x
            (x[1] + x[3]) / 2,  # center y
            x[2] - x[0],  # width
            x[3] - x[1]  # height
        ])

    # Handle multiple boxes (2D array)
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def extract_features_from_yolo(model, frame, bbox_xyxy):
    """
    Extract features from YOLOv8 model for DeepSORT tracking

    Args:
        model: YOLOv8 model
        frame: Input image
        bbox_xyxy: Bounding boxes in xyxy format

    Returns:
        List of extracted features
    """
    features = []

    for box in bbox_xyxy:
        # Convert bounding box to center format
        center_x, center_y, w, h = xyxy2xywh(box)

        # Crop the detection area
        x1 = max(0, int(center_x - w / 2))
        y1 = max(0, int(center_y - h / 2))
        x2 = min(frame.shape[1], int(center_x + w / 2))
        y2 = min(frame.shape[0], int(center_y + h / 2))

        # Skip if invalid crop dimensions
        if x1 >= x2 or y1 >= y2:
            continue

        crop = frame[y1:y2, x1:x2]

        # Skip empty crops
        if crop.size == 0:
            continue

        # Simple feature extraction - resize and flatten
        # In a real implementation, you might use a dedicated feature extractor
        try:
            crop_resized = cv2.resize(crop, (64, 128))
            feature = crop_resized.flatten() / 255.0  # Normalize
            features.append(feature)
        except Exception as e:
            print(f"Error extracting feature: {e}")
            continue

    return features