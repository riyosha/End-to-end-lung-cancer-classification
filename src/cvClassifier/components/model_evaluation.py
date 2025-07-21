import os
import shutil

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
from cvClassifier.utils.common import save_json, load_json
from cvClassifier import logger
from cvClassifier.components.model_training import LightningModel

from torchmetrics.classification import (
    MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassAUROC, MulticlassAccuracy
)

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
        
        self.model = self.load_model(self.config.final_model_path)
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
            # test_results[0] will have all logged metrics
            self.scores = {
                "loss": test_results[0].get('test_loss', 0.0),
                "accuracy": test_results[0].get('test_accuracy', 0.0),
                "precision": test_results[0].get('test_precision', 0.0),
                "recall": test_results[0].get('test_recall', 0.0),
                "f1_score": test_results[0].get('test_f1_score', 0.0),
                "auc_roc": test_results[0].get('test_auc_roc', 0.0)
            }
            logger.info(f"Evaluation completed!")
            for k, v in self.scores.items():
                logger.info(f"{k.capitalize()}: {v:.4f}")
        else:
            logger.info('No results returned from evaluation.')
        
        self.save_score(f'{self.config.scores_path}/final_scores.json')
    
    
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

    def select_best_model(self, model_names, metric="val_f1_score"):
        best_model = None
        best_score = float('-inf')
        best_model_path = None

        for model_name in model_names:
            scores_path = Path(self.config.scores_path) / f"best_scores_{model_name}.json"
            model_path = Path(self.config.scores_path) / f"trained_model_{model_name}.pth"
            if not scores_path.exists() or not model_path.exists():
                logger.info(f"Skipping {model_name}: missing files.")
                continue

            scores = load_json(path=scores_path)
            score = scores.get(metric)
            logger.info(f"{model_name}: {metric} = {score}")

            if score is not None and score > best_score:
                best_score = score
                best_model = model_name
                best_model_path = model_path

        if best_model_path:
            # Ensure destination directory exists
            shutil.copy2(best_model_path, self.config.final_model_path)
            logger.info(f"Best model: {best_model} (score: {best_score}) copied to {self.config.final_model_path}")
            return best_model, best_score
        else:
            raise RuntimeError("No valid models found for selection.")

    def save_score(self, scores_path):
        """Save evaluation scores to JSON file"""

        save_json(path=Path(scores_path), data=self.scores)