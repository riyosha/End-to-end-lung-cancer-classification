from src.cvClassifier import logger
from src.cvClassifier.pipeline.S01_data_ingestion import DataIngestionTrainingPipeline
from src.cvClassifier.pipeline.S02_model_preparation import ModelPreparationPipeline
from src.cvClassifier.pipeline.S03_model_training import ModelTrainingPipeline
from src.cvClassifier.pipeline.S04_model_evaluation import ModelEvaluationPipeline

logger.info("Let's get started! Welcome to the End-to-end Chest Cancer Classification project.")

STAGE_NAME = "Data Ingestion Stage"

try:
        logger.info(f'<<<<<<< {STAGE_NAME} started >>>>>>>')
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f'<<<<<<< {STAGE_NAME} completed >>>>>>>\n \nx========x')

except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Model Preparation Stage"

try:
        logger.info(f'<<<<<<< {STAGE_NAME} started >>>>>>>')
        model_preparation = ModelPreparationPipeline()
        model_preparation.main()
        logger.info(f'<<<<<<< {STAGE_NAME} completed >>>>>>>\n \nx========x')

except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME = "Model Training Stage"

try:
        logger.info(f'<<<<<<< {STAGE_NAME} started >>>>>>>')
        model_training = ModelTrainingPipeline()
        model_training.main()
        logger.info(f'<<<<<<< {STAGE_NAME} completed >>>>>>>\n \nx========x')
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Model Evaluation Stage"

try:
        logger.info(f'<<<<<<< {STAGE_NAME} started >>>>>>>')
        model_evaluation = ModelEvaluationPipeline()
        model_evaluation.main()
        logger.info(f'<<<<<<< {STAGE_NAME} completed >>>>>>>\n \nx========x')
except Exception as e:
        logger.exception(e)
        raise e
