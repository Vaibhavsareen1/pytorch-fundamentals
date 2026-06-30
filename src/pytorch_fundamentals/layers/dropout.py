import torch


class CustomDropout2d(torch.nn.Module):
    """
    Class that represents implementation of 2 dimensional dropout layer. To know more about it
    go through the notebook `pytorch-fundamentals/notebooks/dropout_layer.ipynb`.
    """

    def __init__(self, p: float = 0.5) -> None:
        """
        Method to instantiate object of :class: CustomDropout1d

        :param p: Probability with which the neurons need to be dropped in a pass

        :returns: Instance of :class: CustomDropout1d
        """

        super().__init__()

        if 0 <= p <= 1:
            self.p = p
        else:
            raise ValueError("THe probability to drop the neurons has to be between 0 and 1")
        

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Method to propagate input tensor through CustomDropout1d layer

        :param X: input tensor

        :returns: Output with adjusted values based on probablity
        """

        if self.training:
            identity_tensor = torch.ones_like(X)

            dropout_mask = torch.rand(identity_tensor) < self.p

            identity_tensor[dropout_mask] = 0

            # Return the remaining neuron outputs by scaling them with the factor of (1 / (1 - p))
            return X * identity_tensor * (1 / (1 - self.p))
    
        return X