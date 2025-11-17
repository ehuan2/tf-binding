"""
main.py

This is the main script to run a model given the specified yaml configuration file.
"""

from models.config import Config
from models.base import BaseModel


def get_model_instance(config: Config) -> BaseModel:
    """
    Factory function to get the model instance based on the architecture specified in the config.
    """
    if config.architecture == "simple":
        from models.simple import SimpleModel

        return SimpleModel(config)
    else:
        raise ValueError(f"Unknown architecture: {config.architecture}")


if __name__ == "__main__":
    config = Config()

    # Depending on the architecture, we would instantiate different models
    model = get_model_instance(config)

    print(model)

    # TODO: train and evaluate the model
    # if config.train:
    #     print("Training the model...")
    #     model.train()

    # if config.evaluate:
    #     print("Evaluating the model...")
    #     model.evaluate()

    # model.train()
    # model.evaluate()
