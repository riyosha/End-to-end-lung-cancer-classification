import os

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import optuna
from pathlib import Path

from cvClassifier import logger
from cvClassifier.utils.common import get_size, read_yaml, create_directories, save_json
from cvClassifier.constants import *
from cvClassifier.entity.config_entity import ModelTrainingConfig

import mlflow
from pytorch_lightning.loggers import MLFlowLogger
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, \
                                       MulticlassF1Score, MulticlassAUROC
from pytorch_lightning.callbacks import EarlyStopping


class LightningModel(pl.LightningModule):
    def __init__(self, model, learning_rate=0.01):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()

        # For multi-class metrics:
        self.val_precision = MulticlassPrecision(num_classes=4, average='weighted', validate_args=True)
        self.val_recall = MulticlassRecall(num_classes=4, average='weighted', validate_args=True)
        self.val_f1 = MulticlassF1Score(num_classes=4, average='weighted', validate_args=True)
        self.val_auc = MulticlassAUROC(num_classes=4, average='weighted', validate_args=True)

        self.validation_step_outputs = []
        self.test_step_outputs = []
    
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

        preds_probs = torch.softmax(outputs, dim=-1) # Probabilities for AUROC and other metrics if needed
        preds_labels = outputs.argmax(dim=-1) # Predicted class labels for precision/recall/f1 if not using probabilities

        # Update with probabilities for AUROC, and with predicted labels/probabilities for others
        # Depending on the metric, it might expect one-hot encoded targets or class indices.
        # For Multiclass metrics, typically raw logits or probabilities and integer class labels are used.
        self.val_precision.update(preds_probs, labels)
        self.val_recall.update(preds_probs, labels)
        self.val_f1.update(preds_probs, labels)
        self.val_auc.update(preds_probs, labels)

        
        return {'test_loss': loss, 'test_acc': acc}

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), 
            lr=self.learning_rate,           
            weight_decay=1e-5
        )

    def on_validation_epoch_end(self):

        avg_precision = self.val_precision.compute()
        avg_recall = self.val_recall.compute()
        avg_f1 = self.val_f1.compute()
        avg_auc = self.val_auc.compute()

        self.log('val_precision', avg_precision, prog_bar=True)
        self.log('val_recall', avg_recall, prog_bar=True)
        self.log('val_f1_score', avg_f1, prog_bar=True)
        self.log('val_auc_roc', avg_auc, prog_bar=True)

        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()
        self.val_auc.reset()

        # Calculate average metrics
        if self.validation_step_outputs:
            avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
            avg_acc = torch.stack([x['val_acc'] for x in self.validation_step_outputs]).mean()
            
            self.log('avg_val_loss', avg_loss)
            self.log('avg_val_acc', avg_acc)
            
            # Clear the list for next epoch
            self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        
        # Store outputs for epoch-level metrics
        self.test_step_outputs.append({'test_loss': loss, 'test_acc': acc})
        
        self.log('test_loss', loss, on_step=True, on_epoch=True)
        self.log('test_acc', acc, on_step=True, on_epoch=True)
        
        return {'test_loss': loss, 'test_acc': acc}
    
    def on_test_epoch_end(self):
        # Calculate average metrics
        if self.test_step_outputs:
            avg_loss = torch.stack([x['test_loss'] for x in self.test_step_outputs]).mean()
            avg_acc = torch.stack([x['test_acc'] for x in self.test_step_outputs]).mean()
            
            self.log('avg_test_loss', avg_loss)
            self.log('avg_test_acc', avg_acc)
            
            # Clear the list for next epoch
            self.test_step_outputs.clear()


class ModelTraining:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if mlflow.active_run():
            logger.warning("Found active MLflow run during initialization. Ending it.")
            mlflow.end_run()
        
        # Set up MLflow
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_experiment(f"hyperparameter_tuning_{self.config.model_name}")

        logger.info(f"Using device: {self.device}")

    
    def get_base_model(self):
        self.model = torch.load(self.config.updated_base_model_path, map_location=self.device)
        self.model.to(self.device)

        logger.info(f"Model loaded from {self.config.updated_base_model_path}")

    def train_valid_generator(self, batch_size = None):

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
            batch_size=self.config.params_batch_size if batch_size == None else batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        

        self.valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.config.params_batch_size if batch_size == None else batch_size,
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

        return self.train_loader, self.valid_loader
    
    def get_hyperparameter_search_space(self):
        """Define hyperparameter search space"""
        return {
            'learning_rate': self.config.learning_rate_range,
            'batch_size': self.config.batch_size_options,
            'epochs': self.config.epochs_options
        }

    def objective(self, trial):
        """Optuna objective function for hyperparameter optimization"""
    
        search_space = self.get_hyperparameter_search_space()
    
        # Suggest hyperparameters using the search space
        learning_rate = trial.suggest_float('learning_rate', 
                                        search_space['learning_rate'][0], 
                                        search_space['learning_rate'][-1], 
                                        log=True)
        batch_size = trial.suggest_categorical('batch_size', search_space['batch_size'])
        epochs = trial.suggest_categorical('epochs', search_space['epochs'])
        
        logger.info(f"Trial {trial.number}: lr={learning_rate:.6f}, batch_size={batch_size}, epochs={epochs}")

        # Create descriptive run name
        trial_name = f"trial_{trial.number:02d}_lr{learning_rate:.4f}_bs{batch_size}_ep{epochs}"

        with mlflow.start_run(nested=True, run_name = trial_name) as run:
            
            # log hyperparameters
            mlflow.log_params({
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'epochs': epochs
            })
            # get base model
            self.get_base_model()
            # Create data loaders with suggested batch size
            train_loader, valid_loader = self.train_valid_generator(batch_size)

            # create lightning model instance
            lightning_model = LightningModel(
                model=self.model,
                learning_rate=learning_rate
            )

            # create an mlflow logger
            mlflow_logger = MLFlowLogger(
                experiment_name=f"hyperparameter_tuning_{self.config.model_name}",
                tracking_uri=self.config.mlflow_uri,
                run_id=run.info.run_id # Associated with the current MLflow run
            )

            early_stopping = EarlyStopping(
                monitor='val_loss',           
                patience=5,                   
                mode='min',                   
                verbose=True                  
            )

            # create new pytorch lightning trainer
            trainer = pl.Trainer(
                max_epochs=epochs,
                accelerator='auto',
                devices='auto',
                logger=mlflow_logger,
                callbacks=[early_stopping],
                enable_progress_bar=False, # Disable progress bar for cleaner Optuna output
                enable_model_summary=False,
                log_every_n_steps=10
            )

            # fit the model
            trainer.fit(
                model=lightning_model,
                train_dataloaders=train_loader,
                val_dataloaders=valid_loader
            )

            # Retrieve metrics from trainer.callback_metrics after training
            val_loss = trainer.callback_metrics.get('val_loss')
            val_acc = trainer.callback_metrics.get('val_acc')
            train_loss = trainer.callback_metrics.get('train_loss')
            train_acc = trainer.callback_metrics.get('train_acc')

            # log metrics to mlflow
            trial_metrics = {}
            if val_loss is not None:
                trial_metrics['val_loss'] = val_loss.item() if hasattr(val_loss, 'item') else val_loss
            if val_acc is not None:
                trial_metrics['val_acc'] = val_acc.item() if hasattr(val_acc, 'item') else val_acc
            if train_loss is not None:
                trial_metrics['train_loss'] = train_loss.item() if hasattr(train_loss, 'item') else train_loss
            if train_acc is not None:
                trial_metrics['train_acc'] = train_acc.item() if hasattr(train_acc, 'item') else train_acc

            mlflow.log_metrics(trial_metrics)

            # Return the validation loss for Optuna to minimize
            if 'val_loss' in trial_metrics:
                return trial_metrics['val_loss']
            else:
                logger.warning(f"Validation loss not found for Trial {trial.number}. Returning inf.")
                return float('inf')
    

    def optimize_hyperparameters(self):
        """Run hyperparameter optimization"""
        logger.info("Starting hyperparameter optimization...")
        
        # Create study
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Run optimization
        study.optimize(
            self.objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout
        )
        
        # Get best parameters
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best validation loss: {self.best_score:.4f}")
        
        # Save best parameters
        best_params_path = Path(self.config.root_dir)/f"best_params_{self.config.model_name}.json"
        save_json(best_params_path, self.best_params)
        
        return self.best_params

    def log_trial_to_mlflow(self, trial_params, trial_metrics, lightning_model):
        """Log trial results to MLflow"""
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run(nested=True):
            mlflow.log_params(trial_params)
            mlflow.log_metrics(trial_metrics)
            
            # Log model
            if tracking_url_type_store != "file":
                mlflow.pytorch.log_model(
                    lightning_model.model, 
                    "model", 
                    registered_model_name=self.config.model_name
                )
            else:
                mlflow.pytorch.log_model(lightning_model.model, "model")


    def train(self):
        """Train with hyperparameter optimization"""

        # End any existing active runs before starting
        if mlflow.active_run():
            logger.warning("Found active MLflow run. Ending it before starting new run.")
            mlflow.end_run()

        with mlflow.start_run(run_name=f"hyperparameter_tuning_{self.config.model_name}") as parent_run:
        
            # Log parent run metadata
            mlflow.log_param("model_architecture", self.config.model_name)
            mlflow.log_param("n_trials", self.config.n_trials)
            mlflow.log_param("timeout", self.config.timeout)
            
            logger.info("Starting hyperparameter optimization within parent run...")
        
            # Run hyperparameter optimization first
            self.optimize_hyperparameters()

            # Log best params to parent run
            mlflow.log_params({f"best_{k}": v for k, v in self.best_params.items()})
            mlflow.log_metric("best_validation_loss", self.best_score)

            final_metrics = self.train_with_best_params()

            # Log summary metrics to parent run
            if final_metrics:
                summary_metrics = {}
                for key, value in final_metrics.items():
                    if value is not None:
                        metric_name = f"final_{key}"
                        summary_metrics[metric_name] = value.item() if hasattr(value, 'item') else value
                
                mlflow.log_metrics(summary_metrics)
        
        logger.info(f"Complete training for {self.config.model_name} finished.")
        
        return final_metrics

    def train_with_best_params(self):
        """Train the final model with best hyperparameters"""
        if self.best_params is None:
            logger.warning("No best parameters found. Running optimization first...")
            self.optimize_hyperparameters()


        with mlflow.start_run(nested=True, run_name=f"final_training_{self.config.model_name}") as run:
        
            # Log parameters (same as optimization trials)
            mlflow.log_params({
                'learning_rate': self.best_params['learning_rate'],
                'batch_size': self.best_params['batch_size'], 
                'epochs': self.best_params['epochs'],
                'model_name': self.config.model_name,
                'stage': 'final_training'
            })
            
            # Get base model
            self.get_base_model()
            
            # Create data loaders with best batch size
            train_loader, valid_loader = self.train_valid_generator(self.best_params['batch_size'])
            
            # Create Lightning model with best learning rate
            lightning_model = LightningModel(
                model=self.model,
                learning_rate=self.best_params['learning_rate']
            )
            
            # Set up model checkpointing
            checkpoint_callback = ModelCheckpoint(
                dirpath=self.config.root_dir,
                filename=f"best_model_{self.config.model_name}",
                monitor="val_loss",
                mode="min",
                save_top_k=1
            )

            # Use the SAME experiment as optimization, and associate with nested run
            mlflow_logger = MLFlowLogger(
                experiment_name=f"best_models",
                tracking_uri=self.config.mlflow_uri,
                run_id=run.info.run_id  # Associate with this nested run
            )

            # Create trainer with best epochs
            trainer = pl.Trainer(
                max_epochs=self.best_params['epochs'],
                accelerator='auto',
                devices='auto',
                logger=mlflow_logger,
                callbacks=[checkpoint_callback],
                enable_progress_bar=True,
                enable_model_summary=True,
                log_every_n_steps=50,
            )
            
            # Train the model
            trainer.fit(
                model=lightning_model,
                train_dataloaders=train_loader,
                val_dataloaders=valid_loader
            )
            
            # Log final metrics to the nested run
            final_metrics = {}
            for key, value in trainer.callback_metrics.items():
                if value is not None:
                    final_metrics[key] = value.item() if hasattr(value, 'item') else value
            
            mlflow.log_metrics(final_metrics)
            
            # Save the final model
            logger.info(f"Saving final model to {self.config.trained_model_path}")
            torch.save(lightning_model.model, self.config.trained_model_path)
            logger.info(f"Final model saved to {self.config.trained_model_path}")

            # Save best scores to JSON 
            best_scores = final_metrics
            best_scores_path = Path(self.config.root_dir) / f"best_scores_{self.config.model_name}.json"
            save_json(best_scores_path, best_scores)
            
            return trainer.callback_metrics