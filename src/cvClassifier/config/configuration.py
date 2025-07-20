import os

from cvClassifier.constants import *
from cvClassifier.utils.common import read_yaml, create_directories
from cvClassifier.entity.config_entity import (DataIngestionConfig,
                                                ModelPreparationConfig,
                                                ModelTrainingConfig,
                                                ModelEvalConfig)


class ConfigurationManager:
    # this class manages the configuration of the project

    def __init__(self, config_filepath = CONFIG_FILE_PATH, params_filepath = PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        ''' Gets the config details for the data ingestion pipeline '''
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir = config.root_dir,
            source_URL = config.source_URL,
            local_data_file = config.local_data_file,
            unzip_dir = config.unzip_dir
        )

        return data_ingestion_config

    def get_model_preparation_config(self, model_name) -> ModelPreparationConfig:
        ''' Gets the config details for the model preparation ingestion pipeline '''
        config = self.config.model_preparation

        create_directories([config.root_dir])

        model_preparation_config = ModelPreparationConfig(
            root_dir = config.root_dir,
            model_name = model_name,
            base_model_path = f'{config.base_model_path}/base_model_{model_name}.pth',
            updated_base_model_path = f'{config.updated_base_model_path}/updated_base_model_{model_name}.pth',
            params_image_size = self.params.IMAGE_SIZE,
            params_include_top= self.params.INCLUDE_TOP,
            params_classes = self.params.CLASSES,
            params_weights = self.params.WEIGHTS,
            params_learning_rate = self.params.LEARNING_RATE
        )

        return model_preparation_config
    
    def get_model_training_config(self, model_name) -> ModelTrainingConfig:
        ' Gets the config details for the hyperparameter tuning pipeline '
        config = self.config.model_training
        params = self.params
        training_data = config.training_data
        validation_data = config.validation_data

        create_directories([config.root_dir])

        model_training_config = ModelTrainingConfig(
            root_dir = config.root_dir,
            model_name = model_name,
            updated_base_model_path = f'{config.updated_base_model_path}/updated_base_model_{model_name}.pth',
            training_data_path = Path(training_data),
            validation_data_path = Path(validation_data),
            trained_model_path = f'{config.trained_model_path}/trained_model_{model_name}.pth',
            best_params_path = Path(config.root_dir) / f"best_params_{model_name}.json",
            params_epochs = params.EPOCHS,
            params_batch_size = params.BATCH_SIZE,
            params_is_augmentation = params.AUGMENTATION,
            params_image_size = params.IMAGE_SIZE,
            # Hyperparameter search space
            learning_rate_range = params.LEARNING_RATE_RANGE,  # [min_lr, max_lr]
            batch_size_options = params.BATCH_SIZE_OPTIONS ,   # Different batch sizes to try
            epochs_options = params.EPOCHS_OPTIONS,        # Different epoch counts to try
            n_trials = params.N_TRIALS,                       # Number of trials for optimization
            timeout = params.TIMEOUT,                        # Timeout in seconds (1 hour)
            mlflow_uri = config.mlflow_tracking_uri
        )

        return model_training_config
    
    def get_model_eval_config(self) -> ModelEvalConfig:
        ''' Gets the config details for the model training pipeline '''
        config = self.config.model_evaluation
        params = self.params
        create_directories([config.root_dir])

        model_eval_config = ModelEvalConfig(
            root_dir = self.config.model_evaluation.root_dir,
            training_data_path = self.config.model_training.training_data,
            validation_data_path = self.config.model_training.validation_data,
            test_data_path = self.config.model_training.test_data,
            trained_model_path = self.config.model_training.trained_model_path,
            all_params = params,
            params_image_size = params.IMAGE_SIZE,
            params_batch_size = params.BATCH_SIZE,
            mlflow_uri = self.config.model_evaluation.mlflow_tracking_uri,
            scores_path = self.config.model_evaluation.scores_path
        )

        return model_eval_config