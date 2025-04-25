import os
from utils.download import download_yolo_model, download_from_roboflow
import argparse


def main():
    parser = argparse.ArgumentParser(description='Download necessary models and datasets')
    parser.add_argument('--model', default='yolov8n.pt', help='YOLOv8 model to download')
    args = parser.parse_args()

    # Download YOLO model
    download_yolo_model(model_name=args.model)

    # Download dataset from Roboflow
    try:
        download_from_roboflow()
        print("Dataset downloaded successfully")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please configure your Roboflow credentials in the .env file")


if __name__ == "__main__":
    main()