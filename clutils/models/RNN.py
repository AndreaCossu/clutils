import torch
import torch.nn as nn
from ..globals import OUTPUT_TYPE, choose_output

class VanillaRNN(nn.Module):
    """
    Layers are described by the following names:
    'rnn' -> recurrent module
    'out' -> linear readout

    Use dict(model.layers[layername].named_parameters()) to get {key : value} dict for parameters of layername.
    """

    def __init__(self, input_size, hidden_size, output_size, device,
                num_layers=1, dropout=0., bidirectional=False, 
                truncated_time=0, relu=False, orthogonal=False):
        '''
        :param truncated_time: an integer representing the 
            time step to backpropagate (from the end of sequence).
        '''

        super(VanillaRNN, self).__init__()

        self.output_type = OUTPUT_TYPE.OUTH

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.activation = 'relu' if relu else 'tanh'
        self.dropout = dropout if self.num_layers > 1 else 0.
        self.bidirectional = bidirectional
        self.directions = 2 if self.bidirectional else 1
        self.device = device
        self.orthogonal = orthogonal
        self.truncated_time = truncated_time

        self.layers = nn.ModuleDict([])

        self.layers.update([
            ['rnn', nn.RNN(self.input_size, self.hidden_size,
            num_layers=self.num_layers, nonlinearity=self.activation,
            batch_first=True, dropout=self.dropout,
            bidirectional=self.bidirectional) ]
        ])

        if self.orthogonal:
            for _, hh, _, _ in self.layers['rnn'].all_weights:
                nn.init.orthogonal_(hh)


        self.layers.update([
            ['out', nn.Linear(self.directions*self.hidden_size, self.output_size) ]
        ])

        self.layers = self.layers.to(self.device)


    def forward(self, x, h=None):
        '''
        :param x: (batch_size, seq_len, input_size)
        :param h: hidden state of the recurrent module

        :return out: (batch_size, seq_len, directions*hidden_size)
        :return h: hidden state of the recurrent module
        '''

        if self.truncated_time > 0:
            with torch.no_grad():
                if h:
                    out1, h1 = self.layers['rnn'](x[:, :-self.truncated_time, :], h)
                else:
                    out1, h1 = self.layers['rnn'](x[:, :-self.truncated_time, :])

            out, h = self.layers['rnn'](x[:, -self.truncated_time:, :], h1)

        else:
            if h:
                out, h = self.layers['rnn'](x, h)
            else:
                out, h = self.layers['rnn'](x)

        out = self.layers['out'](out)
        
        if self.output_type == OUTPUT_TYPE.H:
            # take mean over layers and directions
            h = h.mean(dim=0) # (B, H)

        return choose_output(out, h, self.output_type)


    def reset_memory_state(self, batch_size):
        '''
        :param batch_size: size of current batch. 
        '''

        h = torch.zeros(self.directions*self.num_layers, batch_size, self.hidden_size,
                device=self.device)

        return h


class LSTM(nn.Module):
    """
    Layers are described by the following names:
    'rnn' -> recurrent module
    'out' -> linear readout layer

    Use dict(model.layers[layername].named_parameters()) to get {key : value} dict for parameters of layername.
    """

    def __init__(self, input_size, hidden_size, output_size, device,
                num_layers=1, dropout=0., bidirectional=False, 
                truncated_time=0, orthogonal=False):
        '''
        :param truncated_time: an integer representing the 
            time step to backpropagate (from the end of sequence).
        '''

        super(LSTM, self).__init__()

        self.output_type = OUTPUT_TYPE.OUTH

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout if self.num_layers > 1 else 0.
        self.bidirectional = bidirectional
        self.directions = 2 if self.bidirectional else 1
        self.device = device
        self.orthogonal = orthogonal
        self.truncated_time = truncated_time

        self.layers = nn.ModuleDict([])

        self.layers.update([ 
            ['rnn', nn.LSTM(self.input_size, self.hidden_size, \
                    self.num_layers, batch_first=True, dropout=self.dropout, \
                    bidirectional=self.bidirectional) ]
        ])
        if self.orthogonal:
            for _, hh, _, _ in self.layers['rnn'].all_weights:
                # lstm divides hidden matrix into 4 chunks
                # https://pytorch.org/docs/stable/nn.html?highlight=lstm#torch.nn.LSTM
                for j in range(0, hh.size(0), self.hidden_size): 
                    nn.init.orthogonal_(hh[j:j+self.hidden_size])

        self.layers.update([
            ['out', nn.Linear(self.directions*self.hidden_size, self.output_size) ]
        ])

        self.layers = self.layers.to(self.device)

    def forward(self, x, h=None):
        '''
        :param x: (batch_size, seq_len, input_size)
        :param h: hidden state of the recurrent module

        :return out: (batch_size, seq_len, directions*hidden_size)
        :return h: hidden state of the recurrent module
        '''

        if self.truncated_time > 0:
            with torch.no_grad():
                if h:
                    out1, h1 = self.layers['rnn'](x[:, :-self.truncated_time, :], h)
                else:
                    out1, h1 = self.layers['rnn'](x[:, :-self.truncated_time, :])

            out, h = self.layers['rnn'](x[:, -self.truncated_time:, :], h1)

        else:
            if h:
                out, h = self.layers['rnn'](x, h)
            else:
                out, h = self.layers['rnn'](x)

        out = self.layers['out'](out)
        
        if self.output_type == OUTPUT_TYPE.H:
            # take mean over layers and directions
            h = h.mean(dim=0) # (B, H)

        return choose_output(out, h, self.output_type)


    def reset_memory_state(self, batch_size):
        '''
        :param batch_size: size of current batch. 
        '''

        h = (
            torch.zeros(self.directions*self.num_layers, batch_size, self.hidden_size,
                device=self.device),

            torch.zeros(self.directions*self.num_layers, batch_size, self.hidden_size,
                device=self.device)
        )

        return h


class LMN(nn.Module):
    """
    Layers are described by the following names:
    'i2f' -> input to functional
    'm2f' -> memory to functional
    'f2m' -> functional to memory
    'm2m' -> memory to memory
    'f2o' -> functional to output (alternative to m2o)
    'm2o' -> memory to output (alternative to f2o)

    Use dict(model.layers[layername].named_parameters()) to get {key : value} dict for parameters of layername.
    """

    def __init__(self, input_size, functional_size, output_size, memory_size, device,
                orthogonal=False, out_activation=None, functional_out=False):

        '''
        :param orthogonal: If True initialize 'm2m' as orthogonal matrix. Default False.
        :param out_activation: torch.Function(args) or None if no output function is needed. Default None
        :param functional_out: If True output is computed from functional component, otherwise from memory. Default False.
        '''

        super(LMN, self).__init__()

        self.output_type = OUTPUT_TYPE.OUTH

        self.memory_size = memory_size
        self.functional_size = functional_size
        self.output_size = output_size
        self.device = device
        self.functional_out = functional_out
        self.out_activation = out_activation

        
        self.layers = nn.ModuleDict([
            ['i2f', nn.Linear(input_size, functional_size, bias=True)],
            ['m2f', nn.Linear(memory_size, functional_size, bias=True)],
            ['f2m', nn.Linear(functional_size, memory_size, bias=True)],
            ['m2m', nn.Linear(memory_size, memory_size, bias=True)]
        ])

        if self.functional_out:
            self.layers.update([
                ['f2o', nn.Linear(functional_size, output_size, bias=True)]
            ])
        else:
            self.layers.update([
                ['m2o', nn.Linear(memory_size, output_size, bias=True)]
            ])
            
        self.layers = self.layers.to(self.device)

        if orthogonal:
            nn.init.orthogonal_(self.layers['m2m'].weight)


    def forward(self, x, m=None):
        '''
        :param x (batch_size, sequence_len, input_size)
        :param m: (batch_size, memory_size) initial memory

        :return o, outs: last output (batch_size, output_size) 
                        and all outputs (batch_size, sequence_len, output_size)
        '''

        if m is None:
            m = torch.zeros(x.size(0), self.memory_size, device=self.device)  # initial memory

        outs = torch.empty(x.size(0), x.size(1), self.output_size, device=self.device)  # matrix of all outputs

        # for every input step
        for t in range(x.size(1)):
            f = torch.tanh( self.layers['i2f'](x[:, t, :]) + self.layers['m2f'](m) )
            m = self.layers['f2m'](f) + self.layers['m2m'](m)

            # compute output from functional or memory
            if self.functional_out:
                o = self.layers['f2o'](f)
            else:
                o = self.layers['m2o'](m)

            # activation function of output (if provided)
            if self.out_activation is not None:
                o = self.out_activation(o)

            outs[:, t, :] = o

        return choose_output(outs, o, self.output_type)
