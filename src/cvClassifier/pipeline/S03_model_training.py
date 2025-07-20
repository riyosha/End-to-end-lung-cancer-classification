from cvClassifier.config.configuration import ConfigurationManager
from cvClassifier.components.model_training import ModelTraining
from cvClassifier import logger

STAGE_NAME = "Model Training Stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        models = ['vgg16', 'resnet50']
        for model_name in models:
            training_config = config.get_model_training_config(model_name=model_name)
            training = ModelTraining(config=training_config)
            training.get_base_model()
            training.train_valid_generator()
            training.train()
            logger.info(f"Model training for {model_name} completed successfully.")

        logger.info("All models trained successfully.")

if __name__ == '__main__':
    try:
        logger.info(f'<<<<<<< {STAGE_NAME} started >>>>>>>')
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f'<<<<<<< {STAGE_NAME} completed >>>>>>>')

    except Exception as e:
        logger.exception(e)
        raise e