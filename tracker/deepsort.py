import numpy as np
from .deep_sort.detection import Detection
from .deep_sort.nn_matching import NearestNeighborDistanceMetric
from .deep_sort.tracker import Tracker as DeepSortTracker


class DeepSORT:
    def __init__(self, max_dist=0.2, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100, min_hits=3):
        """
        Initialize DeepSORT tracker

        Parameters:
        -----------
        max_dist : float
            Maximum cosine distance threshold for feature similarity
        max_iou_distance : float
            Maximum IOU distance threshold for track association
        max_age : int
            Maximum number of missed misses before a track is deleted
        n_init : int
            Number of consecutive detections before the track is confirmed
        nn_budget : int
            Maximum size of the appearance descriptors gallery
        min_hits : int
            Minimum number of hits for track confirmation (passed to tracker if supported)
        """
        self.max_dist = max_dist
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.nn_budget = nn_budget
        self.min_hits = min_hits  # Store min_hits even if not used directly

        # Initialize the metric for feature distance calculation
        self.metric = NearestNeighborDistanceMetric("cosine", max_dist, nn_budget)

        # Initialize tracker
        # Note: We'll use n_init as min_hits if the DeepSortTracker doesn't support min_hits
        try:
            self.tracker = DeepSortTracker(
                self.metric,
                max_iou_distance=max_iou_distance,
                max_age=max_age,
                n_init=n_init,

            )
        except TypeError:
            # If min_hits is not supported, fall back to using only supported parameters
            print("Warning: DeepSortTracker does not support min_hits parameter. Using n_init instead.")
            self.tracker = DeepSortTracker(
                self.metric,
                max_iou_distance=max_iou_distance,
                max_age=max_age,
                n_init=n_init
            )

    def update(self, bboxes, scores, class_ids, features):
        """
        Update the tracker with new detections

        Parameters:
        -----------
        bboxes : array
            Bounding boxes in format [x1, y1, x2, y2]
        scores : array
            Detection confidence scores
        class_ids : array
            Class IDs for each detection
        features : array
            Feature vectors for each detection

        Returns:
        --------
        outputs : list
            List of tracks with information (bbox, track_id, class_id)
        """
        # Initialize detections
        detections = []

        # Ensure inputs have the same length
        if len(bboxes) != len(features) or len(bboxes) != len(scores) or len(bboxes) != len(class_ids):
            print("Error: Mismatched input lengths in DeepSORT update")
            return []

        # Create Detection objects
        for bbox, score, class_id, feature in zip(bboxes, scores, class_ids, features):
            if len(bbox) != 4:
                print(f"Warning: Invalid bbox format: {bbox}")
                continue

            # Convert from [x1, y1, x2, y2] to [x, y, w, h]
            tlwh = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]

            # Create a Detection object
            detection = Detection(tlwh, score, feature, class_id=class_id)
            detections.append(detection)

        # Update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # Get outputs from tracker
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            # Get track info
            box = track.to_tlbr()  # [x1, y1, x2, y2]
            track_id = track.track_id
            class_id = track.class_id

            outputs.append({
                "bbox": box,
                "track_id": track_id,
                "class_id": class_id
            })

        return outputs