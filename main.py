from src.cvClassifier import logger
from src.cvClassifier.pipeline.S01_data_ingestion import DataIngestionTrainingPipeline

logger.info("Let's get started! Welcome to the End-to-end Chest Cancer Classification project.")

STAGE_NAME = "Data Ingestion Stage"

try:
        logger.info(f'<<<<<<< {STAGE_NAME} started >>>>>>>')
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f'<<<<<<< {STAGE_NAME} completed >>>>>>>')

except Exception as e:
    logger.exception(e)
    raise e