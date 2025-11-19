"""
main.py

This is the main script to run a model given the specified yaml configuration file.
"""

from models.config import Config, get_model_instance
from dataloaders import get_data_splits
from torch.utils.data import DataLoader

if __name__ == "__main__":
    config = Config()

    # Depending on the architecture, we would instantiate different models
    model = get_model_instance(config)

    # Now get the training and testing data splits
    train_dataset, test_dataset = get_data_splits(config)
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Testing dataset size: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    for batch in train_loader:
        print("Training batch:", batch)
        break

    # Then we train the model and evaluate
    # model.train(train_loader)
    # model.evaluate(test_loader)
