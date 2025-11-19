from models.base import BaseModel


class SimpleModel(BaseModel):
    """
    Simple model that takes in the position weight matrix, the different DNA
    structural features, and combines them to classify whether a region is a TF
    binding site or not.
    """

    def __init__(self, config):
        super().__init__()

    def train(self, data):
        print("Training SimpleModel with data:", data)

    def predict(self, data):
        print("Predicting with SimpleModel on data:", data)
        return ["prediction"] * len(data)

    def evaluate(self, data):
        predictions = self.predict(data)
        print("Evaluating SimpleModel. Predictions:", predictions)
