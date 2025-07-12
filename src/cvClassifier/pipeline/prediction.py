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

        self.transform = transforms.Compose([
            transforms.Resize(self.params.IMAGE_SIZE[:-1]),
            transforms.ToTensor()
        ])

        self.class_names = [
            'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib',
            'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa',
            'normal',
            'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa'
        ]


    def predict(self):
        
        model = torch.load('model/model.pth', map_location=self.device)
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
