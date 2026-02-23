"""
Module with internally used utils
"""

import os
import pathlib
import urllib.request

FILE_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = FILE_DIR.parent


def download_model(model_file: str, path_in_repo: str) -> str:
    """
    Return the path to a model file, checking local project directory first,
    then the cache directory, and downloading from GitHub only if needed.

    Args:
        model_file (str): The name of the model (with file extension).
        path_in_repo (str): Path to the directory with model in repo.

    Returns:
        str: The full path to the model file.
    """
    # 1. Check local project path first
    local_path = os.path.join(PROJECT_ROOT, path_in_repo, model_file)
    if os.path.isfile(local_path):
        return local_path

    # 2. Check / use cache directory
    cache_dir = os.path.join(os.path.expanduser("~"), ".emotiefflib")
    os.makedirs(cache_dir, exist_ok=True)
    fpath = os.path.join(cache_dir, model_file)
    if not os.path.isfile(fpath):
        url = (
            "https://github.com/Kris-gadara/EmotiScan/blob/main/"
            + path_in_repo
            + model_file
            + "?raw=true"
        )
        print("Downloading", model_file, "from", url)
        urllib.request.urlretrieve(url, fpath)
    return fpath


def get_model_path_torch(model_name: str) -> str:
    """
    Return the local file path of a Torch model based on the given model name.

    Args:
        model_name (str): The name of the model (without file extension).

    Returns:
        str: The full path to the model file.
    """
    model_file = model_name + ".pt"
    path_in_repo = "models/affectnet_emotions/"
    return download_model(model_file, path_in_repo)


def get_model_path_onnx(model_name: str) -> str:
    """
    Return the local file path of an ONNX model based on the given model name.

    Args:
        model_name (str): The name of the model (without file extension).

    Returns:
        str: The full path to the model file.
    """
    model_file = model_name + ".onnx"
    path_in_repo = "models/affectnet_emotions/onnx/"
    return download_model(model_file, path_in_repo)


def get_engagement_classification_weights() -> str:
    """
    Return the local file path of engagement classification model.

    Returns:
        str: The full path to the model file.
    """
    model_file = "engagement_single_attention.h5"
    path_in_repo = "models/engagement_classification/"
    return download_model(model_file, path_in_repo)
