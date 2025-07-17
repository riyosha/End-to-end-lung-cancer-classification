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

@dataclass
class ModelTrainingConfig:
    root_dir: Path
    updated_base_model_path: Path
    training_data_path: Path
    validation_data_path: Path
    trained_model_path: Path
    best_params_path: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: int
    # Hyperparameter search space
    learning_rate_range: list  
    batch_size_options: list   
    epochs_options: list     
    n_trials: int             # Number of trials for optimization
    timeout: int              # Timeout in seconds

    mlflow_uri: str

@dataclass(frozen=True)
class ModelEvalConfig:
    root_dir: Path
    trained_model_path: Path
    training_data_path: Path
    validation_data_path: Path
    test_data_path: Path
    all_params: dict
    params_image_size: list
    params_batch_size: int
    mlflow_uri: str
    scores_path: Path

