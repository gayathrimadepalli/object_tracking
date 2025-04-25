
# Object Tracking with YOLOv8 and DeepSORT

## 📌 Overview

This project implements **real-time object tracking** by combining the powerful object detection capabilities of **YOLOv8** with the reliable tracking performance of **DeepSORT**. The system can detect and track multiple objects (e.g., people, vehicles, etc.) across video frames, maintaining consistent identities over time.

## 🎯 Objectives

- Perform accurate object detection using YOLOv8
- Track objects across frames using DeepSORT
- Handle multiple objects with unique IDs
- Enable real-time or near-real-time processing of video streams

## 🧠 Components

- **YOLOv8**: A state-of-the-art object detection model developed by Ultralytics, known for its high speed and accuracy.
- **DeepSORT**: A deep learning-based extension of the SORT algorithm that uses appearance features to maintain object identity.

## 🔧 Setup Instructions

1. **Clone the repository**
2. **Install required dependencies** (YOLOv8, DeepSORT, OpenCV, etc.)
3. **Download pre-trained models**:
   - YOLOv8 model weights (from Ultralytics)
   - DeepSORT model checkpoint
4. **Configure input source**:
   - Use video files, webcam, or RTSP streams

## 🚀 Running the Tracker

- Run the main script with specified input
- View the real-time tracking results with bounding boxes and object IDs
- Optionally, save the output to a video file

## 🧪 Supported Features

- Real-time video input/output
- Class filtering (track only specific object classes)
- Frame skipping and resizing for performance tuning
- Output video recording
- FPS and performance metrics

## 📝 Notes

- Make sure your device has a CUDA-compatible GPU for optimal performance.
- YOLOv8 can be used in different modes (`n`, `s`, `m`, `l`, `x`) based on speed vs. accuracy tradeoffs.
- DeepSORT requires features from detected objects, so a re-ID model must be used.

## 📦 Dependencies

- Python 3.8+
- OpenCV
- Ultralytics YOLOv8
- NumPy
- Torch
- DeepSORT dependencies (e.g., sklearn, filterpy)

## 🧠 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [DeepSORT](https://github.com/nwojke/deep_sort)
