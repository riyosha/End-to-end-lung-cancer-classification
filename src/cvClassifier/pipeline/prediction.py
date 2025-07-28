import os
import boto3
import tempfile
from pathlib import Path

import torch
import numpy as np
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from PIL import Image

from cvClassifier.constants import *
from cvClassifier.utils.common import read_yaml, create_directories
from cvClassifier import logger


class PredictionPipeline:
    def __init__(self, filename, config_filepath = CONFIG_FILE_PATH, params_filepath = PARAMS_FILE_PATH):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.filename = filename

        # S3 configuration
        self.s3_bucket = 'chest-cancer-classifier'
        self.s3_key = "model/model.pth"
        self.local_model_path = "/tmp/model.pth"  # Cache in container

        self.transform = transforms.Compose([
            transforms.Resize(self.params.IMAGE_SIZE[:-1]),
            transforms.ToTensor()
        ])

        self.class_names = [
            'Adenocarcinoma, Stage Ib',
            'Large Cell, Stage IIIa',
            'Normal',
            'Squamous Cell, Stage IIIa'
        ]

    def load_model_from_s3(self):
        """Download model from S3 if not cached locally"""

        if os.path.exists(self.local_model_path):
            logger.info("Using cached model")
            return self.local_model_path
            
        try:
            logger.info(f"Downloading model from S3: s3://{self.s3_bucket}/{self.s3_key}")
            s3_client = boto3.client('s3')
            s3_client.download_file(self.s3_bucket, self.s3_key, self.local_model_path)
            logger.info("Model downloaded successfully")
            return self.local_model_path
            
        except Exception as e:
            logger.error(f"Failed to download model from S3: {e}")
            # Fallback to local model if available
            if os.path.exists("model/model.pth"):
                logger.info("🔄 Falling back to local model")
                return "model/model.pth"
            else:
                raise RuntimeError("Model not available locally or on S3")

    def predict(self):

        model_path = self.load_model_from_s3()
        
        model = torch.load(model_path, map_location=self.device)
        model.eval()
        img = Image.open(self.filename).convert("RGB")
        img = self.transform(img).unsqueeze(0).to(self.device)
        logger.info(f"Test image loaded")

        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            predicted_idx = predicted.item()
            logger.info(f"Predicted index: {predicted_idx}")
            predicted_class = self.class_names[predicted_idx]
            logger.info(f"Predicted class: {predicted_class}")
        return predicted_class
