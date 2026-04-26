import torch
from typing import Self


class ReLUActivationCustom(torch.nn.Module):
    """
    Class that represents implementation of ReLU (Rectified Linear Unit) non linear activation function.

    Formula:
    output = max(0, input)
    """

    def __init__(self) -> Self:
        """
        Method to instantiate class: `ReLUActivationCustom`

        :returns: instance of class: `ReLUActivationCustom`
        """

        super().__init__()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Method to perform forward propagation of input tensor `X` through the relu 
        non linear activation function

        :param X: input tensor

        :returns: Output of relu non linear activation function applied to input tensor `X`
        """

        return torch.max(torch.zeros_like(X), X)