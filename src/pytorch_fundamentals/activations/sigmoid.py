import torch
from typing import Self

class SigmoidActivationCustom(torch.nn.Module):
    """
    Class that represents implementation of Sigmoid non linear activation function.

    Formula:
    output = 1 / (1 + exp(-input))
    """

    def __init__(self) -> Self:
        """
        Method to instantiate class: `SigmoidActivationCustom`

        :returns: instance of class: `SigmoidActivationCustom`
        """

        super().__init__()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Method to perform forward propagation of input tensor `X` through the sigmoid 
        non linear activation function

        :param X: input tensor

        :returns: Output of sigmoid non linear activation function applied to input tensor `X`
        """

        return 1 / (1 + torch.exp(-1 * X))