#from src.utils.utils import inference_mode
from src.models.definitions.house_model import HousePricesModel
import torch
from pathlib import Path


class Interference:
    def __init__(self,name,path:Path):
        # Initializing path and model name
        self.PATH=path
        self.NAME=name


    def load_model(self) -> HousePricesModel:
        """Load the trained HousePriceModel from disk."""
        model_path=self.PATH/self.NAME
        if not model_path.exists():
            raise FileNotFoundError(f"model file not found: {model_path}")
        model = HousePricesModel()

        """learning_rate = 0.001
        momentum = 0.8
        num_epochs = 200
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)"""

        model.load_state_dict(torch.load(model_path)["model_state"])
        #inference_mode(model)  # switch to eval mode
        return model

    def predict(self,input_tensor):
        """Run inference on new data."""
        model=self.load_model()

        with torch.no_grad():  # no gradients needed during prediction
            outputs = model(input_tensor)
        return outputs

