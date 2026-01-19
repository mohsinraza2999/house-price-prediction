from torch import nn, Tensor

class PyTorchLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x: Tensor,
        inference: bool = False) -> Tensor:
        raise NotImplementedError()