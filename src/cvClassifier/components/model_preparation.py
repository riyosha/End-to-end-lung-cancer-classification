import os

from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_Weights

from cvClassifier.entity.config_entity import ModelPreparationConfig
from cvClassifier import logger

class ModelPreparation:
    def __init__(self,config = ModelPreparationConfig):
        self.config = config
    
    def get_base_model(self):
        weights = VGG16_Weights.IMAGENET1K_V1 if self.config.params_weights == 'imagenet' else None
        self.model = models.vgg16(weights=weights)

        if not self.config.params_include_top:
            self.model = nn.Sequential(*list(self.model.features.children()))
            # *list() unpacks the list of layers in the model.features and passes them as separate arguments to nn.Sequential

        self.save_model(self.config.base_model_path, self.model)
        logger.info(f"Base model saved at {self.config.base_model_path}")

    @staticmethod
    def save_model(path: Path, model: nn.Module):
        torch.save(model, path)

    @staticmethod
    def prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for param in model.parameters():
                param.requires_grad = False
        elif (freeze_till is not None) and (freeze_till > 0):
            layers = list(model.children())
            for layer in layers[:-freeze_till]:
                for param in layer.parameters():
                    param.requires_grad = False
        # add a check here to ensure freeze_till is not larger than layer size

        num_features = 512 * 7 * 7 

        if isinstance(model, nn.Sequential):
            base_layers = list(model.children())
        else:
            # If it's a full VGG model, get only the feature layers
            base_layers = list(model.features.children())

        full_model = nn.Sequential(
            *base_layers,  # Unpack the base layers
            nn.Flatten(), 
            nn.Linear(num_features, classes), 
            # softmax is included in cross-entropy loss in PyTorch
        )


        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        logger.info(f'full_model')
        logger.info(f"Full model has {len(list(full_model.children()))} layers")
        print("Last few layers:")
        for i, layer in enumerate(list(full_model.children())[-3:]):
            print(f"  ({len(list(full_model.children()))-3+i}): {layer}")

        logger.info(f"Model prepared with {classes} classes, freeze_all={freeze_all}, freeze_till={freeze_till}, learning_rate={learning_rate}")

        return full_model, optimizer, criterion

    def update_base_model(self):
        model, optimizer, criterion = self.prepare_full_model(
            model = self.model,
            classes = self.config.params_classes,
            freeze_all = True,
            freeze_till = None,
            learning_rate = self.config.params_learning_rate
        )

        self.save_model(path = self.config.updated_base_model_path, model = model)

        logger.info(f'Updated base model saved at {self.config.updated_base_model_path}')