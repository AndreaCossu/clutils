import torch
import torch.nn as nn


def expand_output_layer(layer, n_units):
    """
    Expand layer wiht n_units more.
    layer can be either a Linear layer or a (weight, bias) tuple of Parameters.

    Return a new torch.nn.Linear
    """

    if isinstance(layer, tuple):
        weight = layer[0]
        bias = layer[1]
    elif isinstance(layer, nn.Linear):
        weight = layer.weight
        bias = layer.bias
    else:
        raise ValueError(f"layer must be torch.nn.Linear or tuple of Parameters. Got {type(layer)}.")

    with torch.no_grad():
        # recompute output size
        old_output_size = weight.size(0)
        hidden_size = weight.size(1)
        new_output_size = old_output_size + n_units

        # copy old output layer into new one
        new_layer = nn.Linear(hidden_size, new_output_size, bias=True).to(weight.device)
        new_layer.weight[:old_output_size, :] = weight
        new_layer.bias[:old_output_size] = bias

        return new_layer


def sequence_to_flat(x):
    n_dims = len(x.size())

    if n_dims > 2:
        return x.view(x.size(0), -1)

    return x