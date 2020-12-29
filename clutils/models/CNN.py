import torch.nn as nn
from ..globals import OUTPUT_TYPE, choose_output
from .utils import expand_output_layer, compute_conv_out_shape, compute_conv_out_shape_1d

class CNN1D(nn.Module):
    """
    1D convolutions for sequences
    """

    def __init__(self, in_channels, device, window_size, n_conv_layers=3, feed_layers=[256, 128],
        output_size=None, out_activation=None):
        """
        :param in_channels: number of input channels
        :param n_conv_layers: how many convolutional layers (output layer excluded)
        :param feed_layers: list of feedforward layers size
        :param output_size: None if output does not have to be computed. An integer otherwise. Default None.
        :param out_activation: if None last layer is linear, else out_activation is used as output function.
            out_activation must be passed in the form torch.nn.Function(args).
            If output_size is None this option has no effect. Default None.
        """

        super(CNN1D, self).__init__()

        self.output_type = OUTPUT_TYPE.ALL_OUTS
        self.is_recurrent = False

        self.in_channels = in_channels
        self.n_conv_layers = n_conv_layers
        self.feed_layers = feed_layers
        self.window_size = window_size
        self.output_size = output_size
        self.device = device
        out_chs = [self.in_channels] + \
            [ 40*(i+1) for i in range(n_conv_layers) ]

        ks = [3**(i+1) for i in range(n_conv_layers)]

        self.out_activation = out_activation

        self.layers = nn.ModuleDict()

        output_size_conv = self.window_size
        for i in range(self.n_conv_layers):
            output_size_conv = compute_conv_out_shape_1d(output_size_conv, 0, 1, ks[i], 1) # convolution
            output_size_conv = compute_conv_out_shape_1d(output_size_conv, 0, 1, 2, 1) # pooling

            self.layers.update([
                [f'conv{i}', nn.Conv1d(out_chs[i], out_chs[i+1], kernel_size=ks[i], stride=1)],
                [f'relu{i}', nn.ReLU()],
                [f'pool{i}', nn.MaxPool1d(kernel_size=2, stride=1)]
            ])

        self.layers.update([
            ['flatten', nn.Flatten()]
        ])

        for i, el in enumerate(feed_layers):
            if i == 0:
                input_size = output_size_conv * out_chs[-1]
            else:
                input_size = feed_layers[i-1]

            self.layers.update([
                [f'l{i}', nn.Linear(input_size, el, bias=True)],
                [f'relu_l{i}', nn.ReLU()],
            ])

        if self.output_size is not None:
            self.layers.update( {'out': nn.Linear(feed_layers[-1], self.output_size, bias=True) } )

        self.layers = self.layers.to(self.device)


    def forward(self, x):
        '''
        :return out: (batch_size, output_size) or (batch_size, hidden_sizes[-1]) if output_size is None.
        '''

        # reshape by putting length last
        assert(len(x.shape) == 3)
        x = x.permute(0, 2, 1)

        hs = []

        for i in range(self.n_conv_layers):
            x = self.layers[f'conv{i}'](x)
            x = self.layers[f'relu{i}'](x)
            x = self.layers[f'pool{i}'](x)
            hs.append(x)

        h = self.layers['flatten'](x)

        for i in range(len(self.feed_layers)):
            h = self.layers[f'l{i}'](h)
            hs.append(h)
            h = self.layers[f'relu_l{i}'](h)

        if self.output_size is not None:
            out = self.layers['out'](h)
            if self.out_activation is not None:
                out = self.out_activation(out)

        return choose_output(out, hs, self.output_type)


    def expand_output_layer(self, n_units=2):
        self.layers["out"] = expand_output_layer(self.layers["out"], n_units)


    def get_layers(self):
        return self.layers.values()


class CNN(nn.Module):

    def __init__(self, in_channels, device, H, W, n_conv_layers=3, feed_layers=[256, 128],
        output_size=None, out_activation=None):
        """
        :param in_channels: number of input channels
        :param n_conv_layers: how many convolutional layers (output layer excluded)
        :param feed_layers: list of feedforward layers size
        :param output_size: None if output does not have to be computed. An integer otherwise. Default None.
        :param out_activation: if None last layer is linear, else out_activation is used as output function.
            out_activation must be passed in the form torch.nn.Function(args).
            If output_size is None this option has no effect. Default None.
        """

        super(CNN, self).__init__()

        self.output_type = OUTPUT_TYPE.ALL_OUTS
        self.is_recurrent = False

        self.in_channels = in_channels
        self.n_conv_layers = n_conv_layers
        self.feed_layers = feed_layers
        self.H = H
        self.W = W
        self.output_size = output_size
        self.device = device
        out_chs = [self.in_channels] + \
            [ 2*(i+1) for i in range(n_conv_layers) ]

        ks = [2*(i+1) for i in range(n_conv_layers)]

        self.out_activation = out_activation

        self.layers = nn.ModuleDict()

        for i in range(self.n_conv_layers):
            W, H = compute_conv_out_shape(W, H, 0, 1, ks[i], 1) # convolution
            W, H = compute_conv_out_shape(W, H, 0, 1, 2, 1) # pooling

            self.layers.update([
                [f'conv{i}', nn.Conv2d(out_chs[i], out_chs[i+1], kernel_size=ks[i], stride=1)],
                [f'relu{i}', nn.ReLU()],
                [f'pool{i}', nn.MaxPool2d(kernel_size=2, stride=1)]
            ])

        self.layers.update([
            ['flatten', nn.Flatten()]
        ])

        for i, el in enumerate(feed_layers):
            if i == 0:
                input_size = W * H * out_chs[-1]
            else:
                input_size = feed_layers[i-1]

            self.layers.update([
                [f'l{i}', nn.Linear(input_size, el, bias=True)],
                [f'relu_l{i}', nn.ReLU()],
            ])

        if self.output_size is not None:
            self.layers.update( {'out': nn.Linear(feed_layers[-1], self.output_size, bias=True) } )

        self.layers = self.layers.to(self.device)


    def forward(self, x):
        '''
        :return out: (batch_size, output_size) or (batch_size, hidden_sizes[-1]) if output_size is None.
        '''
        # reshape if input is a sequence (batch-first)
        x = x.view(x.size(0), self.in_channels, self.H, self.W)
        hs = []

        for i in range(self.n_conv_layers):
            x = self.layers[f'conv{i}'](x)
            x = self.layers[f'relu{i}'](x)
            x = self.layers[f'pool{i}'](x)
            hs.append(x)

        h = self.layers['flatten'](x)

        for i in range(len(self.feed_layers)):
            h = self.layers[f'l{i}'](h)
            hs.append(h)
            h = self.layers[f'relu_l{i}'](h)

        if self.output_size is not None:
            out = self.layers['out'](h)
            if self.out_activation is not None:
                out = self.out_activation(out)

        return choose_output(out, hs, self.output_type)


    def expand_output_layer(self, n_units=2):
        self.layers["out"] = expand_output_layer(self.layers["out"], n_units)


    def get_layers(self):
        return self.layers.values()        