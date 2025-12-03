from models.base import BaseModel
from sklearn.svm import LinearSVC


class SVMModel(BaseModel):
    """
    Simple model that takes in the position weight matrix, the different DNA
    structural features, and combines them to classify whether a region is a TF
    binding site or not.
    """

    def __init__(self, config):
        self.config = config
        self.model = LinearSVC(C=config.C)

    def train(self, data):
        # Fitting model
        print("Training SimpleModel with data:", data)
        self.model.fit(data.X, data.Y)

    def predict(self, data):
        print("Predicting with SimpleModel on data:", data)
        return ["prediction"] * len(data)

    def evaluate(self, data):
        predictions = self.predict(data)
        print("Evaluating SimpleModel. Predictions:", predictions)
