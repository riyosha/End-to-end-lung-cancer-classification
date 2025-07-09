from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class ModelPreparationConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_include_top: bool
    params_classes: int
    params_weights: str
    params_learning_rate: float

@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    updated_base_model_path: Path
    training_data_path: Path
    validation_data_path: Path
    trained_model_path: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: int
    params_learning_rate: float

@dataclass(frozen=True)
class ModelEvalConfig:
    trained_model_path: Path
    training_data_path: Path
    validation_data_path: Path
    test_data_path: Path
    all_params: dict
    params_image_size: list
    params_batch_size: int
    mlflow_uri: str
    scores_path: Path

