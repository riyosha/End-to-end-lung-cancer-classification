import os

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pathlib import Path

from cvClassifier.entity.config_entity import ModelTrainingConfig
from cvClassifier import logger

class LightningModel(pl.LightningModule):
    def __init__(self, model, learning_rate=0.01):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        
        self.log('val_loss', loss)
        self.log('val_acc', acc)
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate)


class ModelTraining:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logger.info(f"Using device: {self.device}")

    
    def get_base_model(self):
        self.model = torch.load(self.config.updated_base_model_path, map_location=self.device)
        self.model.to(self.device)

        logger.info(f"Model loaded from {self.config.updated_base_model_path}")

    def train_valid_generator(self):

        # preparing the validation dataset
        valid_transforms = transforms.Compose([
            transforms.Resize(self.config.params_image_size[:-1]),  # Resize to target size
            transforms.ToTensor(),  # Converts to tensor and scales to [0,1] (equivalent to rescale=1./255)
        ])
        
        # preparing the training dataset
        if self.config.params_is_augmentation:
            train_transforms = transforms.Compose([
                transforms.Resize(self.config.params_image_size[:-1]),
                transforms.RandomRotation(40),  # rotation_range=40
                transforms.RandomHorizontalFlip(p=0.5),  # horizontal_flip=True
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.2, 0.2),  # width_shift_range=0.2, height_shift_range=0.2
                    scale=(0.8, 1.2),  # zoom_range=0.2
                    shear=0.2  # shear_range=0.2
                ),
                transforms.ToTensor(),
            ])
        else:
            train_transforms = valid_transforms


        # load training dataset
        train_dataset = datasets.ImageFolder(
            root=self.config.training_data_path,
            transform=train_transforms
        )
        logger.info(f"Training dataset created from {self.config.training_data_path}")

        # load validation dataset
        valid_dataset = datasets.ImageFolder(
            root=self.config.validation_data_path,
            transform=valid_transforms
        )
        logger.info(f"Validation dataset created from {self.config.validation_data_path}")
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.params_batch_size,
            shuffle=True,
            num_workers=0, # more than 0 workers causes issues - need to figure out why
            pin_memory=True if self.device.type == 'cuda' else False
        )
        

        self.valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.config.params_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        
        self.train_dataset_size = len(train_dataset)
        self.valid_dataset_size = len(valid_dataset)
        
        logger.info(f"Training samples: {self.train_dataset_size}")
        logger.info(f"Validation samples: {self.valid_dataset_size}")
        logger.info(f"Number of classes: {len(train_dataset.classes)}")
        logger.info(f"Classes: {train_dataset.classes}")
        

    @staticmethod
    def save_model(path: Path, model: nn.Module):
        torch.save(model, path)


    def train(self):
        lightning_model = LightningModel(
            model=self.model,
            learning_rate=self.config.params_learning_rate
        )
        
        trainer = pl.Trainer(
            max_epochs=self.config.params_epochs,
            accelerator='auto',  
            devices='auto',      # Use all available devices
            logger=True,         # Enable logging
            enable_progress_bar=True,
            enable_model_summary=True,
            enable_checkpointing=True,
            log_every_n_steps=50,
        )
        
        logger.info("Starting training with PyTorch Lightning...")
        
        trainer.fit(
            model=lightning_model,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.valid_loader
        )
        
        # Get final metrics
        train_metrics = trainer.callback_metrics
        
        logger.info("Training completed!")
        logger.info("=" * 60)
        logger.info("FINAL TRAINING METRICS:")
        
        # Print final metrics
        for key, value in train_metrics.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"{key}: {value.item():.4f}")
            else:
                logger.info(f"{key}: {value}")
        
        logger.info("=" * 60)
        
        self.save_model(
            path=self.config.trained_model_path,
            model=lightning_model.model  
        )
        
        logger.info(f"Model trained and saved to {self.config.trained_model_path}")
        
        return trainer.callback_metrics

'''
    def train(self):

        self.steps_per_epoch = len(self.train_loader)
        self.validation_steps = len(self.valid_loader)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.params_learning_rate)
        
        for epoch in range(self.config.params_epochs):
            # Training phase
            self.model.train()
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(inputs), labels)
                loss.backward()
                optimizer.step()
            
            # Validation phase
            self.model.eval()
            with torch.no_grad():
                for inputs, labels in self.valid_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    criterion(self.model(inputs), labels) 
        
        logger.info(f"Training completed for {self.config.params_epochs} epochs")
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )

        logger.info(f"Model trained and saved to {self.config.trained_model_path}")
    '''