from torch import nn, Tensor
from models.definitions.base_model import PyTorchModel
from src.layers.dense_layer import DenseLayer
from src.utils.utils import inference_mode


class HousePricesModel(PyTorchModel):
    def __init__(self,
                 hidden_size: int = 13,
                 hidden_dropout: float = 1.0):
        super().__init__()
        self.dense1 = DenseLayer(13, hidden_size,
                                 activation=nn.LeakyReLU(),
                                 dropout = hidden_dropout)
        self.dense2 = DenseLayer(hidden_size, 1)
    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[1] == 13
        x = self.dense1(x)
        return self.dense2(x)
