import torch


def normalize_single_block(importance):
    """
    0-1 normalization over all parameters
    :param importance: [ (k1, p1), ..., (kn, pn)] (key, parameter) list
    """

    max_imp = -1
    min_imp = 1e7
    for _, imp in importance:
        curr_max_imp, curr_min_imp = imp.max(), imp.min()
        max_imp = max(max_imp, curr_max_imp)
        min_imp = min(min_imp, curr_min_imp)

    r = max(max_imp - min_imp, 1e-6)
    for _, imp in importance:
        imp -= min_imp
        imp /= float(r)

    return importance

def normalize_blocks(importance):
    """
    0-1 normalization over each parameter block
    :param importance: [ (k1, p1), ..., (kn, pn)] (key, parameter) list
    """

    for _, imp in importance:
        max_imp, min_imp = imp.max(), imp.min()
        imp -= min_imp
        imp /= float(max(max_imp - min_imp, 1e-6))
        
    return importance


def padded_difference(p1, p2, use_sum=False):
    """
    Return the difference between p1 and p2. Result size is size(p2).
    If p1 and p2 sizes are different, simply compute the difference 
    by cutting away additional values and zero-pad result to obtain back the p2 dimension.
    """

    assert(len(p1.size()) == len(p2.size()) < 3)

    if p1.size() == p2.size():
        return p1 + p2 if use_sum else p1 - p2


    min_size = torch.Size([
        min(a, b)
        for a,b in zip(p1.size(), p2.size())
    ])
    if len(p1.size()) == 2:
        resizedp1 = p1[:min_size[0], :min_size[1]]
        resizedp2 = p2[:min_size[0], :min_size[1]]
    else:
        resizedp1 = p1[:min_size[0]]
        resizedp2 = p2[:min_size[0]]


    difference = resizedp1 + resizedp2 if use_sum else resizedp1 - resizedp2 
    padded_difference = torch.zeros(p2.size(), device=p2.device)
    if len(p1.size()) == 2:
        padded_difference[:difference.size(0), :difference.size(1)] = difference
    else:
        padded_difference[:difference.size(0)] = difference

    return padded_difference


def zerolike_params_dict(model, return_grad=False):
    if return_grad:
        return [ ( k, torch.zeros_like(p.grad).to(p.device) ) for k,p in model.named_parameters() ]
    else:
        return [ ( k, torch.zeros_like(p).to(p.device) ) for k,p in model.named_parameters() ]

def copy_params_dict(model, copy_grad=False, detach=False):
    if copy_grad:
        return [ ( k, p.grad.clone() ) for k,p in model.named_parameters() ]
    else:
        if detach:
            return [ ( k, p.clone().detach() ) for k,p in model.named_parameters() ]
        else:
            return [ ( k, p.clone() ) for k,p in model.named_parameters() ]
