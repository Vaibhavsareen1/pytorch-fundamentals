import torch 
from typing import Self


class TanhActivationCustom(torch.nn.Module):
    """
    Class that represents implementation of "Hyperbolic Tangent" (tanh) non linear activation function.

    Formula:
    output = (exp(X) - exp(-X)) / (exp(X) + exp(-X))

    Note:
    - Hyperbolic Tangent becomes computationally unstable in two cases
        - When X > 20 ( exp(X) is very close to infinity becomes computationally unstable)
        - When X < -20 (exp(X) so small that the values ~0 and becomes computationally unstable. It would be better to use 0 instead)
    - This instability is due to very small values for exp() on -X and very large values on X
    - To handle this and keep it stable you need to clip the input tensor to have X > 20 to be 20 and X < -20 to be -20.

    - Pytorch follows the same process to keep it computationally stable 
    """

    def __init__(self) -> Self:
        """
        Method to instantiate class: `TanhActivationCustom`

        :returns: instance of class: `TanhActivationCustom`
        """

        super().__init__()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Method to perform forward propagation of input tensor `X` through the tanh 
        non linear activation function

        :param X: input tensor

        :returns: Output of tanh non linear activation function applied to input tensor `X`
        """

        # Clip input tensor to be between -20 and 20 (inclusive)
        X[X > 20] = 20
        X[X < -20] = -20

        positive_x_exp = torch.exp(X)
        negative_x_exp = torch.exp(-X)

        return (positive_x_exp - negative_x_exp) / (positive_x_exp + negative_x_exp)