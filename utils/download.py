import os
import requests
from tqdm import tqdm
import yaml
from dotenv import load_dotenv
from roboflow import Roboflow


def download_yolo_model(model_name="yolov8n.pt", save_dir="models"):
    """
    Download YOLOv8 pretrained model from Ultralytics.

    Parameters
    ----------
    model_name : str
        Model name to download.
    save_dir : str
        Directory to save the model.

    Returns
    -------
    str
        Path to the downloaded model.
    """
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, model_name)

    if os.path.exists(model_path):
        print(f"Model {model_name} already exists at {model_path}")
        return model_path

    url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{model_name}"

    print(f"Downloading {model_name} from {url}...")
    response = requests.get(url, stream=True)

    if response.status_code != 200:
        raise Exception(f"Failed to download model from {url}, status code: {response.status_code}")

    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192

    with open(model_path, 'wb') as f, tqdm(
            total=total_size, unit='B', unit_scale=True,
            desc=f"Downloading {model_name}"
    ) as pbar:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    print(f"Model downloaded successfully to {model_path}")
    return model_path


def download_from_roboflow(save_dir="data/roboflow"):
    """
    Download dataset from Roboflow.

    Parameters
    ----------
    save_dir : str
        Directory to save the dataset.

    Returns
    -------
    str
        Path to the downloaded dataset.
    """
    load_dotenv()

    api_key = os.getenv("ROBOFLOW_API_KEY")
    workspace = os.getenv("ROBOFLOW_WORKSPACE")
    project = os.getenv("ROBOFLOW_PROJECT")
    version = os.getenv("ROBOFLOW_VERSION")

    if not all([api_key, workspace, project, version]):
        raise ValueError("Please set all required Roboflow environment variables in .env file")

    print(f"Downloading dataset from Roboflow: {workspace}/{project}/{version}")

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project)
    dataset = project.version(version).download("yolov8", location=save_dir)

    print(f"Dataset downloaded successfully to {save_dir}")
    return save_dir
