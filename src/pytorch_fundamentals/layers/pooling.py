import math
import torch


class MaxPool2dCustom(torch.nn.Module):
    """
    Class that represents implementation of 2 dimensional max pooling layer. To know more about it
    go through the notebook `pytorch-fundamentals/notebooks/pooling_layers.ipynb`.
    """

    def __init__(self, kernel_size: int, stride: int, padding: int = 0, dilation: int = 1) -> None:
        """
        Method to instantiate object of :class: MaxPool2dCustom

        :param kernel_size: Size of the convolution filter
        :param stride: sliding step size of the convolution filter
        :param padding: number of pixels to be added to the input tensor before convolving the filter around
                        input tensor
        :param dilation: Dilation step

        :returns: Instance of :class: MaxPool2dCustom
        """

        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Method to propagate input tensor through MaxPool layer

        :param X: input tensor

        :returns: Pooled output 
        """

        batch_size, c_in, h_in, w_in = X.shape

        # Unfold incoming tensor into it's receptive fields
        X_unfolded = torch.nn.functional.unfold(X,
                                                kernel_size=self.kernel_size,
                                                padding=self.padding,
                                                stride=self.stride,
                                                dilation=self.dilation)
        
        # Transpose the receptive fields so that it becomes easier to extract the maximum value 
        # from each receptive field
        X_transpose = X_unfolded.transpose(dim0=-2, dim1=-1)

        # Get the maximum value of each receptive field 
        max_values = X_transpose.max(dim=1).values

        # Get the h_out and w_out values to reshape max_values
        h_out = math.floor((h_in + (2 * self.padding) - (self.dilation * (self.kernel_size - 1)) - 1) / self.stride) + 1
        w_out = math.floor((w_in + (2 * self.padding) - (self.dilation * (self.kernel_size - 1)) - 1) / self.stride) + 1
    
        return max_values.reshape(batch_size, c_in, h_out, w_out)


class AvgPool2dCustom(torch.nn.Module):
    """
    Class that represents implementation of 2 dimensional average pooling layer. To know more about it
    go through the notebook `pytorch-fundamentals/notebooks/pooling_layers.ipynb`.
    """

    def __init__(self, kernel_size: int, stride: int, padding: int = 0, dilation: int = 1) -> None:
        """
        Method to instantiate object of :class: AvgPool2dCustom

        :param kernel_size: Size of the convolution filter
        :param stride: sliding step size of the convolution filter
        :param padding: number of pixels to be added to the input tensor before convolving the filter around
                        input tensor
        :param dilation: Dilation step

        :returns: Instance of :class: AvgPool2dCustom
        """

        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Method to propagate input tensor through AvgPool layer

        :param X: input tensor

        :returns: Pooled output 
        """

        batch_size, c_in, h_in, w_in = X.shape

        # Unfold incoming tensor into it's receptive fields
        X_unfolded = torch.nn.functional.unfold(X,
                                                kernel_size=self.kernel_size,
                                                padding=self.padding,
                                                stride=self.stride,
                                                dilation=self.dilation)
        
        # Transpose the receptive fields so that it becomes easier to extract the average value 
        # from each receptive field
        X_transpose = X_unfolded.transpose(dim0=-2, dim1=-1)

        # Get the average value of each receptive field 
        average_values = X_transpose.mean(dim=1)

        # Get the h_out and w_out values to reshape average_values
        h_out = math.floor((h_in + (2 * self.padding) - self.kernel_size) / self.stride) + 1
        w_out = math.floor((w_in + (2 * self.padding) - self.kernel_size) / self.stride) + 1
    
        return average_values.reshape(batch_size, c_in, h_out, w_out)