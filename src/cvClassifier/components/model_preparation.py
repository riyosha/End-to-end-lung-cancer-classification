import os

from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_Weights
from torchvision.models import ResNet50_Weights

from cvClassifier.entity.config_entity import ModelPreparationConfig
from cvClassifier import logger

class ModelFactory:
    @staticmethod
    def create_model(model_name: str, num_classes: int, pretrained: bool = True, include_top: bool = False, weights: str = 'imagenet'):
        """Create VGG16 or ResNet50 model"""
        
        if model_name.lower() == "vgg16":
            weights = VGG16_Weights.IMAGENET1K_V1 if weights == 'imagenet' else None
            model = models.vgg16(weights=weights)

            if not include_top:
                model = nn.Sequential(*list(model.features.children()))
                # *list() unpacks the list of layers in the model.features and passes them as separate arguments to nn.Sequential
            
            logger.info(f"Created VGG16 model with include_top={include_top}")
            
        elif model_name.lower() == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V1 if weights == 'imagenet' else None
            model = models.resnet50(weights=weights)

            #if not include_top:
                # Remove the final classification layer
                #model = nn.Sequential(*list(model.children())[:-1])
            
            
            logger.info(f"Created ResNet50 model with include_top={include_top}")
            
        else:
            raise ValueError(f"Unsupported model: {model_name}. Only 'vgg16' and 'resnet50' are supported.")
            
        return model

class ModelPreparation:
    def __init__(self,config = ModelPreparationConfig):
        self.config = config
    
    def get_base_model(self):
        """Get base model using factory pattern"""
        
        # Create model using factory
        self.model = ModelFactory.create_model(
            model_name=self.config.model_name,
            num_classes=self.config.params_classes,
            pretrained=True,
            include_top=self.config.params_include_top,
            weights=self.config.params_weights
        )
        
        logger.info(f"{self.config.model_name} model created successfully")

    @staticmethod
    def save_model(path: Path, model: nn.Module):
        torch.save(model, path)

    @staticmethod
    def prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate, model_name):
        
        if model_name.lower() == "vgg16":
            if freeze_all:
                for param in model.parameters():
                    param.requires_grad = False
            elif (freeze_till is not None) and (freeze_till > 0):
                layers = list(model.children())
                for layer in layers[:-freeze_till]:
                    for param in layer.parameters():
                        param.requires_grad = False

            num_features = 512 * 7 * 7  # VGG16 features
            
            if isinstance(model, nn.Sequential):
                base_layers = list(model.children())
            else:
                base_layers = list(model.features.children())
                
            full_model = nn.Sequential(
                *base_layers,
                nn.Flatten(), 
                nn.Linear(num_features, classes), 
            )
                
        elif model_name.lower() == "resnet50":
            if freeze_all:
                # Freeze all layers except layer4 (conv5)
                for name, param in model.named_parameters():
                    if 'layer4' not in name:
                        param.requires_grad = False
                        print(f"Frozen: {name}")
                    else:
                        param.requires_grad = True
                        print(f"Trainable: {name}")
            elif (freeze_till is not None) and (freeze_till > 0):
                for name, param in model.named_parameters():
                    if 'layer4' not in name:
                        param.requires_grad = False

            # For ResNet50, replace the fc layer
            model.fc = nn.Linear(2048, classes)
            full_model = model  # Use the full ResNet50 model
            
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(full_model.parameters(), lr=learning_rate)

        print(full_model)
        print(f"Full model has {len(list(full_model.children()))} layers")
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in full_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in full_model.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")

        logger.info(f"{model_name} model prepared with {classes} classes, freeze_all={freeze_all}, freeze_till={freeze_till}, learning_rate={learning_rate}")

        return full_model, optimizer, criterion

    def update_base_model(self):
        model, optimizer, criterion = self.prepare_full_model(
            model = self.model,
            classes = self.config.params_classes,
            freeze_all = True,
            freeze_till = None,
            learning_rate = self.config.params_learning_rate,
            model_name = self.config.model_name
        )

        self.save_model(path = self.config.updated_base_model_path, model = model)

        logger.info(f'Updated {self.config.model_name} base model saved at {self.config.updated_base_model_path}')