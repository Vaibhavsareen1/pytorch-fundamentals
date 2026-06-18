import math
import torch
from typing import Self

class Conv2dCustom(torch.nn.Module):
    """
    Class that represents implementation of 2 dimensional convolution layer.

    To read in detail about how a 2 dimensional convolution layer convolves over the input tensor go through
    `pytorch-fundamentalsnotebooks/convolution_layers.ipynb` it will help you understand the current implementation
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 dilation: int,
                 bias: bool = True) -> None:
        """
        Method to instantiate object of :class: Conv2dCustom

        :param in_channels: Number of channels of incoming input tensor
        :param out_channels: Number of channels that should be presenting on the output tensor
        :param kernel_size: Size of the convolution filter. The filter will be a 2 dimensional filter of size kernel_size x kernel_size
        :param stride: The step size of the convolution filter
        :param padding: Number of times padding value needs to be added to the input tensor before it's convolution takes place
        :param dilation: Spacing between input tensor's elements. 1 means two adjascent elements, 2 means elements with a space of 1 element between them
        :param bias: Boolean to determine whether bias needs to be added or not. Default value `True`

        :returns: Instance of :class: Conv2dCustom
        
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self._BIAS = bias

        # Create weight tensor of shape (out_channels, in_channels, kernel_size, kernel_size)
        self.weight = torch.nn.Parameter(data=torch.randn((self.out_channels,
                                                           self.in_channels,
                                                           self.kernel_size,
                                                           self.kernel_size)))
        if self._BIAS:
            # Bias tensor will be of the shape (out_channels)
            self.register_parameter("bias", torch.nn.Parameter(data=torch.randn(self.out_channels)))
        else:
            self.register_parameter("bias", None)

    def forward(self, X) -> torch.Tensor:
        """
        Method to propagate input tensor through the convolution layer.

        :param X: input tensor of shape (N, C_in, H_in, W_in)

        :returns: Feature map of shape (N, C_out, H_out, W_out)
        """
        
        # Unpack input tensor's shape
        batch_size, c_in, h_in, w_in = X.shape

        # Calculate the dimension of the output feature map 
        h_out = math.floor((h_in + (2 * self.padding) - (self.dilation * (self.kernel_size - 1)) - 1) / self.stride) + 1
        w_out = math.floor((w_in + (2 * self.padding) - (self.dilation * (self.kernel_size - 1)) - 1) / self.stride) + 1

        # Flatten convolution filters to be used by im2col implementaion using `torch.nn.functional.unfold`
        weights_flattened = self.weight.reshape(self.out_channels, -1)

        # Unfold the input tensor into (N, C_in * kernel_size * kernel_size, L) receptive fields
        X_unfolded = torch.nn.functional.unfold(X, kernel_size=(self.kernel_size, self.kernel_size), dilation=self.dilation, padding=self.padding, stride=self.stride)

        X_convoluted = torch.matmul(weights_flattened, X_unfolded)

        if self._BIAS:
            X_convoluted = X_convoluted + self.bias.view(1, -1, 1)

        # Reshape the convolution product into the actual size of the output tensor
        return X_convoluted.reshape(batch_size, self.out_channels, h_out, w_out)