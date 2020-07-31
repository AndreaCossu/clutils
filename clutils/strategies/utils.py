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