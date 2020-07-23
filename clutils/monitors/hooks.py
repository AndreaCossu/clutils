class MonitorActivations():
    def __init__(self):
        self.hs = []

    def __call__(self, module, module_in, module_out):
        self.hs.append(module_out)

    def reset(self):
        self.hs = []
    
    def get_activations(self, model=None, x=None):
        if x:
            model(x)
        return self.hs

    def remove_hooks(self, hooks):
        for h in hooks:
            h.remove()


class MonitorGradients():
    def __init__(self):
        self.gradients = []

    def __call__(self, module, grad_input, grad_output):
        # gradients are monitored from last to first, append at the beginning
        # to respect forward order
        self.gradients.insert(0, grad_output)

    def reset(self):
        self.gradients = []
    
    def get_gradients(self):
        return self.gradients

    def remove_hooks(self, hooks):
        for h in hooks:
            h.remove()