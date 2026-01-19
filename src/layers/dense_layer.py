from torch import nn, Tensor
from layers.base_layer import PyTorchLayer
from src.utils.utils import inference_mode


class DenseLayer(PyTorchLayer):
    def __init__(self,
        input_size: int,
        neurons: int,
        dropout: float = 1.0,
        activation: nn.Module = None) -> None:
        super().__init__()
        self.linear = nn.Linear(input_size, neurons)
        self.activation = activation
        if dropout < 1.0:
            self.dropout = nn.Dropout(1 - dropout)
    def forward(self, x: Tensor,
        inference: bool = False) -> Tensor:
        if inference:
            self.apply(inference_mode)
        x = self.linear(x) # does weight multiplication + bias
        if self.activation:
            x = self.activation(x)
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        return x