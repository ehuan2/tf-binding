"""
main.py

This is the main script to run a model given the specified yaml configuration file.
"""

from models.config import Config, get_model_instance
from dataloaders import get_data_splits

if __name__ == "__main__":
    config = Config()

    # Now get the training and testing data splits
    train_dataset, test_dataset, tf_len = get_data_splits(config)
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Testing dataset size: {len(test_dataset)}")

    # Depending on the architecture, we would instantiate different models
    model = get_model_instance(config, tf_len)

    # Then we train the model and evaluate
    model.train(train_dataset)
    model.evaluate(test_dataset)
