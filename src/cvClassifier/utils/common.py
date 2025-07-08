import os
from box.exceptions import BoxValueError
import yaml
from src.cvClassifier import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns ConfigBox type

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml, 'r') as f:
            content = yaml.safe_load(f)
            logger.info(f'yaml file successfully loaded from {path_to_yaml}')
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError(f'yaml file is empty: {path_to_yaml}')
    except Exception as e:
        raise e
    

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """creates the list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f'Directory created at: {path}')


@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    logger.info(f'json file saved at {path}')


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """

    with open(path, 'r') as f:
        content = json.load(f)
    logger.info(f'json file loaded from {path}')
    return ConfigBox(content)
 

@ensure_annotations
def save_bin(data: Any, path: Path):
    """saves binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value = data, filename = path)
    logger.info(f'binary file saved at {path}')

@ensure_annotations
def load_bin(path: Path) -> Any:
    """loads binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    content = joblib.load(path)
    logger.info(f'binary file loaded from {path}')
    return content

@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size = os.path.getSize(path)
    size_in_kb = round(size/1024)
    return f"~ {size_in_kb} KB"

def decodeImage(imgstring, fileName):
    """ Decodes a base64 encoded image string and saves it to a file.
    
    Args: 
        fileName: path where the image will be saved
        Imgstring (str): Base64 encoded image string
    """
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()
    logger.info(f'Image saved at {file_name}')


def encodeImageIntoBase64(croppedImagePath):
    """ Encodes an image file into a base64 string
    
    Args: 
        croppedImagePath (str): Path to image file to be encoded

    Returns:
        str: Base64 encoded string of image"""

    with open(croppedImagePath, 'rb') as f:
        return base64.b64encode(f.read())


