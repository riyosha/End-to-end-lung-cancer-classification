import os

from cvClassifier.constants import *
from cvClassifier.utils.common import read_yaml, create_directories 
from cvClassifier.entity.config_entity import (DataIngestionConfig,
                                                ModelPreparationConfig,
                                                ModelTrainingConfig)


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

    def get_model_preparation_config(self) -> ModelPreparationConfig:
        ''' Gets the config details for the model preparation ingestion pipeline '''
        config = self.config.model_preparation

        create_directories([config.root_dir])

        model_preparation_config = ModelPreparationConfig(
            root_dir = config.root_dir,
            base_model_path = config.base_model_path,
            updated_base_model_path = config.updated_base_model_path,
            params_image_size = self.params.IMAGE_SIZE,
            params_include_top= self.params.INCLUDE_TOP,
            params_classes = self.params.CLASSES,
            params_weights = self.params.WEIGHTS,
            params_learning_rate = self.params.LEARNING_RATE,
        )

        return model_preparation_config
    
    def get_model_training_config(self) -> ModelTrainingConfig:
        ''' Gets the config details for the model training pipeline '''
        config = self.config.model_training
        params = self.params
        training_data = os.path.join(self.config.data_ingestion.unzip_dir, "Data/train")
        validation_data = os.path.join(self.config.data_ingestion.unzip_dir, "Data/valid")
        
        create_directories([config.root_dir])

        model_training_config = ModelTrainingConfig(
            root_dir = config.root_dir,
            updated_base_model_path = config.updated_base_model_path,
            training_data_path = Path(training_data),
            validation_data_path = Path(validation_data),
            trained_model_path = config.trained_model_path,
            params_epochs = params.EPOCHS,
            params_batch_size = params.BATCH_SIZE,
            params_is_augmentation = params.AUGMENTATION,
            params_image_size = params.IMAGE_SIZE,
            params_learning_rate = self.params.LEARNING_RATE
        )

        return model_training_config