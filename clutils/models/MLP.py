import torch.nn as nn
from ..globals import OUTPUT_TYPE, choose_output
from ..monitors.hooks import MonitorLinear

class MLP(nn.Module):
    """
    MLP with N hidden layers

    Layers are described by the following names:
    'i2h' -> from input to first hidden layer
    'h1h2' -> from first hidden layer to second hidden layer
    'h{N-1}h{N}' -> from penultimate hidden layer to last hidden layer 
    'h2out' -> from last hidden layer to output layer (logits)

    Use dict(model.layers[layername].named_parameters()) to get {key : value} dict for parameters of layername.
    """


    def __init__(self, input_size, hidden_sizes, device, output_size=None, relu=False, out_activation=None):
        '''
        If len(hidden_sizes) = N, MLP will have N hidden layers (N weight matrixes + biases).
        If output_size is not None, MLP will have an additional output layer with 1 additional weight matrix and bias.

        :param input_size: number of input features
        :param hidden_sizes: list containing the dimension of each hidden layer
        :param output_size: None if output does not have to be computed. An integer otherwise. Default None.
        :param relu: If False, tanh activation function is used, otherwise relu is used. Default False.
        :param out_activation: if None last layer is linear, else out_activation is used as output function.
            out_activation must be passed in the form torch.nn.Function(args).
            If output_size is None this option has no effect. Default None.
    
        '''

        super(MLP, self).__init__()

        self.output_type = OUTPUT_TYPE.OUT

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.device = device

        self.monitor_linear = MonitorLinear()

        self.activation = nn.Tanh() if not relu else nn.ReLU()
        self.out_activation = out_activation

        # Input 2 hidden
        self.layers = nn.ModuleDict([
            ['i2h', nn.Linear(self.input_size, self.hidden_sizes[0], bias=True)],
        ])
        
        # Hidden 2 Hidden
        for i in range(1, len(self.hidden_sizes)):
            self.layers.update([
                [ 'h{}h{}'.format(i, i+1), nn.Linear(self.hidden_sizes[i-1], self.hidden_sizes[i], bias=True)],
            ])

        # Hidden 2 output
        if self.output_size is not None:
            self.layers.update( {'h2out': nn.Linear(self.hidden_sizes[-1], self.output_size, bias=True) } )

        self.layers = self.layers.to(self.device)

    def forward(self, x):
        '''
        :param x: (batch_size, n_features)

        :return out: (batch_size, output_size) or (batch_size, hidden_sizes[-1]) if output_size is None.
        '''

        h = self.layers['i2h'](x)
        h = self.activation(h)
        
        for i in range(1, len(self.hidden_sizes)):
            h = self.layers[f"h{i}h{i+1}"](h)
            h = self.activation(h)

        if self.output_size is not None:
            out = self.layers['h2out'](h)
            if self.out_activation is not None:
                out = self.out_activation(out)

        return choose_output(out, h, self.output_type)