import enum


@enum.unique
class OUTPUT_TYPE(enum.Enum):
    """
    Define what kind of output each model produces
    OUT -> all hidden states for RNNs, final readout output for feedforward / convolutional models
    H -> last hidden state for RNNs, last hidden layer activations for feedforward / convolutional models

    NOTHING does not return anything (it simply calls return with no arguments)
    """

    NOTHING = 0
    OUT = 1
    OUTH = 2
    H = 3

def choose_output(out, h, mode):
    if mode == OUTPUT_TYPE.NOTHING:
        return
    if mode == OUTPUT_TYPE.OUT:
        return out
    if mode == OUTPUT_TYPE.OUTH:
        return out, h
    if mode == OUTPUT_TYPE.H:
        return h
    
    raise ValueError(f"INVALID OUTPUT_TYPE CONSTANT. Got {mode}, expected one of NOTHING, OUT, OUTH, H")