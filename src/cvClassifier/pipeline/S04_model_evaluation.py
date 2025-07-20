from cvClassifier.config.configuration import ConfigurationManager
from cvClassifier.components.model_evaluation import ModelEvaluation
from cvClassifier import logger

STAGE_NAME = "Model Evaluation Stage"

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        models = ['vgg16', 'resnet50']
        eval_config = config.get_model_eval_config()
        model_eval = ModelEvaluation(config=eval_config)
        model_eval.select_best_model(model_names=models)
        model_eval.evaluation()
        model_eval.log_into_mlflow()


if __name__ == '__main__':
    try:
        logger.info(f'<<<<<<< {STAGE_NAME} started >>>>>>>')
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f'<<<<<<< {STAGE_NAME} completed >>>>>>>')

    except Exception as e:
        logger.exception(e)
        raise e