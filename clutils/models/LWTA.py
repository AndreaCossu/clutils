import torch
import torch.nn as nn
from collections import defaultdict
from .utils import expand_output_layer, sequence_to_flat
from ..globals import OUTPUT_TYPE, choose_output

class LWTA(nn.Module):
    """
    Local Winner Takes All

    Layers are described by the following names:
    'i2h' -> from input to first hidden layer
    'h1h2' -> from first hidden layer to second hidden layer
    'h{N-1}h{N}' -> from penultimate hidden layer to last hidden layer 
    'out' -> from last hidden layer to output layer (logits)

    Use dict(model.layers[layername].named_parameters()) to get {key : value} dict for parameters of layername.
    """    

    def __init__(self, n_units_per_block, n_blocks_per_layer, device, input_size, 
            output_size=None, out_activation=None, nonlinear_activation=None):
        '''
        :param n_blocks_per_layer: list containing number of blocks for each hidden layer
        :param n_units_per_block: list containing number of units for each layers' block. 
                                blocks in the same layer have the same number of units.
                                len(n_units_per_block)==number of hidden layers==len(n_blocks_per_layer)
        '''

        super(LWTA, self).__init__()

        self.output_type = OUTPUT_TYPE.ALL_OUTS

        self.n_units_per_block = n_units_per_block
        self.n_blocks_per_layer = n_blocks_per_layer
        self.out_activation = out_activation
        self.input_size = input_size
        self.output_size = output_size
        self.device = device

        self.hidden_sizes = [ self.n_units_per_block[i] * self.n_blocks_per_layer[i] for i in range(len(self.n_blocks_per_layer)) ]
        
        if nonlinear_activation == 'none':
            self.nonlinear_activation = None
        elif nonlinear_activation == 'relu':
            self.nonlinear_activation = nn.ReLU()
        else:
            self.nonlinear_activation = nn.Hardtanh()

        # Input 2 hidden
        self.layers = nn.ModuleDict([
            ['i2h', nn.Linear(self.input_size, self.hidden_sizes[0], bias=True)]
        ])
        
        # Hidden 2 Hidden
        for i in range(1, len(self.hidden_sizes)):
            self.layers.update([
                [ 'h{}h{}'.format(i, i+1), nn.Linear(self.hidden_sizes[i-1], self.hidden_sizes[i], bias=True)]
            ])

        # Hidden 2 output
        if self.output_size is not None:
            self.layers.update( {'out': nn.Linear(self.hidden_sizes[-1], self.output_size, bias=True) } )
        
        self.layers = self.layers.to(device)

        # dictionary of units that cannot be used anymore, for each layer
        self.forbidden_units = defaultdict(list) 

    def compute_blocks(self):
        # indices of beginning of each block in each layer (list of tensors)
        idxs = []

        for l in range(len(self.n_blocks_per_layer)):
            n_units = self.n_units_per_block[l]
            hs = self.hidden_sizes[l]

            idx = torch.arange(0, hs, n_units).to(self.device)
            idxs.append(idx)

        return idxs

    def _non_maximal_indexes(self, maximal_indexes, size_h, n_blocks):
        filter_indexes = torch.ones(size_h, device=self.device)

        idxs = \
            tuple(torch.arange(size_h[0], device=self.device).view(-1,1).repeat(1, n_blocks).view(-1).tolist()), \
            tuple(maximal_indexes.view(-1).tolist())

        filter_indexes[idxs] = 0

        return filter_indexes.bool()


    def activation(self, h, l, forbidden_units=[]):
        '''
        :param h: (batch_size, hidden_size) activation vector of a layer
        :param l: hidden layer id   

        :return (batch_size, hidden_size) zero-ed non winning units for each block of a layer
        '''

        if self.nonlinear_activation is not None:
            h = self.nonlinear_activation(h) # (B, H)

        #if len(forbidden_units) > 0:
        #    h[:, torch.tensor(forbidden_units).long().to(self.device)] = 0.

        nb = self.n_blocks_per_layer[l]
        nn = self.n_units_per_block[l]

        # exploit contiguous blocks
        blocks = h.view(-1, nb, nn) # (B, blocks, units_per_block)
        # winner for each block (index local to each block)
        winners = torch.argmax(blocks, dim=2) # (B, blocks)
        # winners indexes global to the layer (add displacement relative to block start)
        global_winners = winners + torch.arange(0, nb*nn, nn, device=self.device).unsqueeze(0).repeat(h.size(0), 1)
        
        # boolean matrix with True for non maximal unit, False for maximal unit 
        filter_indexes = self._non_maximal_indexes(global_winners, h.size(), nb) # (B, H)

        h[filter_indexes] = 0.

        return h


    def forward(self, x):
        
        # reshape input if sequence (batch first)
        x = sequence_to_flat(x)

        for l in range(len(self.hidden_sizes)):

            if l == 0:
                h = self.layers['i2h'](x)
            else:
                h = self.layers['h{}h{}'.format(l, l+1)](h)

            h = self.activation(h, l, self.forbidden_units[l])

        out = self.layers['out'](h)

        if self.out_activation is not None:
            out = self.out_activation(out)
        
        return choose_output(out, h, self.output_type)
    

    def compute_frequency_activation(self, data_loader):
        """
        :param: data_loader: a dataloader with one single batch
        """
        
        assert(len(data_loader) == 1)

        with torch.no_grad():
            network_frequencies = []

            for i, (x, _) in enumerate(data_loader):
                x = x.view(x.size(0), -1).to(self.device)

                for l in range(len(self.hidden_sizes)):

                    # vector of frequencies for layer l
                    frequencies = torch.zeros(self.n_units_per_block[l] * self.n_blocks_per_layer[l], device=self.device).float()
                    if l == 0:
                        h = self.layers['i2h'](x)
                    else:                
                        h = self.layers['h{}h{}'.format(l, l+1)](h)

                    if self.nonlinear_activation is not None:
                        h = self.nonlinear_activation(h) 

                    # if len(self.forbidden_units[l]) > 0:
                    #    h[:, torch.tensor(self.forbidden_units[l]).long().to(self.device)] = 0.

                    nb = self.n_blocks_per_layer[l]
                    nn = self.n_units_per_block[l]

                    blocks = h.view(-1, nb, nn)
                    winners = torch.argmax(blocks, dim=2) 
                    global_winners = winners + torch.arange(0, nb*nn, nn, device=self.device).unsqueeze(0).repeat(h.size(0), 1)

                    for ex in global_winners: # loop over batch size
                        frequencies[ex] += 1
                    frequencies = frequencies / float(global_winners.size(0))

                    filter_indexes = self._non_maximal_indexes(global_winners, h.size(), nb)
                    h[filter_indexes] = 0.

                    network_frequencies.append(frequencies)
                    
            network_frequencies = torch.cat(network_frequencies) # (n_units_in_network)

            return network_frequencies.cpu()
    
    def expand_output_layer(self, n_units=2):
        self.layers["out"] = expand_output_layer(self.layers["out"], n_units)

    def get_layers(self):
        return self.layers.values()