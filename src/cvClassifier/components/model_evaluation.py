import os

from urllib.parse import urlparse
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from pathlib import Path

import mlflow

from cvClassifier.entity.config_entity import ModelEvalConfig
from cvClassifier.utils.common import save_json
from cvClassifier import logger
from cvClassifier.components.model_training import LightningModel

from dotenv import load_dotenv
load_dotenv()

os.environ["MLFLOW_TRACKING_URI"]=os.getenv("MLFLOW_TRACKING_URI")
os.environ["MLFLOW_TRACKING_USERNAME"]=os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"]=os.getenv("MLFLOW_TRACKING_PASSWORD")


class ModelEvaluation:
    def __init__(self, config: ModelEvalConfig):
        self.config = config

    def load_model(self, path: Path) -> nn.Module:
        return torch.load(path)
        logger.info(f"Model loaded from {path}")
    
    def test_generator(self):

        # preparing the test dataset
        test_transforms = transforms.Compose([
            transforms.Resize(self.config.params_image_size[:-1]),  # Resize to target size
            transforms.ToTensor(),  # Converts to tensor and scales to [0,1]
        ])
        
        # load test dataset
        test_dataset = datasets.ImageFolder(
            root=self.config.test_data_path,
            transform=test_transforms
        )
        logger.info(f"Test dataset created from {self.config.test_data_path}")
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.params_batch_size,
            shuffle=False,
            num_workers=0
        )
        
        
        logger.info(f"Test samples: {len(test_dataset)}")
        logger.info(f"Number of classes: {len(test_dataset.classes)}")
        logger.info(f"Classes: {test_dataset.classes}")

    
    def evaluation(self):
        """Perform model evaluation using PyTorch Lightning"""

        logger.info('Starting model evaluation...')
        
        self.model = self.load_model(self.config.trained_model_path)
        self.model.eval()
        
        self.test_generator()
        
        lightning_model = LightningModel(self.model)
        
        trainer = pl.Trainer(
            accelerator='auto',
            devices='auto',
            logger=False,  # Disable logging for evaluation
            enable_progress_bar=True,
            enable_model_summary=False,
            enable_checkpointing=False,
        )
        
        test_results = trainer.test(
            model=lightning_model,
            dataloaders=self.test_loader,
            verbose=True
        )
        
        if test_results and len(test_results) > 0:
            self.scores = {
                "loss": test_results[0].get('test_loss_epoch', 0.0),
                "accuracy": test_results[0].get('test_acc_epoch', 0.0)
            }
            logger.info(f"Evaluation completed!")
            logger.info(f"Loss: {self.scores['loss']:.4f}")
            logger.info(f"Accuracy: {self.scores['accuracy']:.4f}")
            
        else:
            logger.info('No results returned from evaluation.')
        
        self.save_score(self.config.scores_path)
    
    
    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.scores['loss'], "accuracy": self.scores['accuracy']}
            )

            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.pytorch.log_model(self.model, "model", registered_model_name="VGG16Model")
            else:
                mlflow.pytorch.log_model(self.model, "model")

    def save_score(self, scores_path):
        """Save evaluation scores to JSON file"""

        save_json(path=Path(scores_path), data=self.scores)