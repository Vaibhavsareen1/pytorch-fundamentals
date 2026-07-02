import torch


class BatchNorm2dCustom(torch.nn.Module):
    """
    Class that represents implementation of 2 dimensional batch normalization layer. To know more about it
    go through the notebook `pytorch-fundamentals/notebooks/normalization_layer.ipynb`.
    """

    def __init__(self, num_features: int, momentum: float = 0.1, eps = 0.0001) -> None:
        """
        Method to instantiate object of :class: BatchNorm2dCustom

        :param num_features: Number of channels for which normalization needs to take place
        :param momentum: The momentum with with current batch mean and variance contribute to the running mean and variance. Defaults to 1e-1
        :param eps: The minimum value to prevent computational overload. Defaults to 1e-4

        :returns: Instance of :class: BatchNorm2dCustom
        """

        super().__init__()

        self.eps = eps
        self.momentum = momentum

        # Register lambda and beta that are trainable variance and mean respectively
        self._lambda = torch.nn.Parameter(torch.ones((1, num_features, 1, 1)))
        self._beta = torch.nn.Parameter(torch.zeros((1, num_features, 1, 1)))

        # Register running mean and running variance and set them to 0 and 1 respectively per channel (num features)
        self.register_buffer("running_mean", torch.zeros(size=(1, num_features, 1, 1)))
        self.register_buffer("running_variance", torch.ones(size=(1, num_features, 1, 1)))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Method to propagate input tensor through BatchNorm2dCustom layer

        :param X: input tensor

        :returns: Batch normalized input tensor
        """

        if self.training:
            # Calculate batch's mean and variance for for each channel
            batch_mean = X.mean(dim=(0, 2, 3), keepdim=True)
            batch_variance_biased = X.var(dim=(0, 2, 3), keepdim=True, unbiased=False)

            # Calculate unbiased variance to update the running history of variance
            batch_variance_unbiased = X.var(dim=(0, 2, 3), keepdim=True, unbiased=True)
    
            # update running mean and running variance
            self.running_mean = ((1 - self.momentum) * self.running_mean) + (self.momentum * batch_mean)
            self.running_variance = ((1 - self.momentum) * self.running_mean) + (self.momentum * batch_variance_unbiased)

            # Normalize input
            X = (X - batch_mean) / torch.sqrt(batch_variance_biased + self.eps)
        else:
            X = (X - self.running_mean) / torch.sqrt(self.running_variance + self.eps)
    
        return X * self._lambda + self._beta
