from cvClassifier.config.configuration import ConfigurationManager
from cvClassifier.components.model_preparation import ModelPreparation
from cvClassifier import logger

STAGE_NAME = "Model Preparation Stage"

class ModelPreparationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        base_model_config = config.get_model_preparation_config()
        prepare_base_model = ModelPreparation(config=base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()


if __name__ == '__main__':
    try:
        logger.info(f'<<<<<<< {STAGE_NAME} started >>>>>>>')
        obj = ModelPreparationPipeline()
        obj.main()
        logger.info(f'<<<<<<< {STAGE_NAME} completed >>>>>>>')

    except Exception as e:
        logger.exception(e)
        raise e