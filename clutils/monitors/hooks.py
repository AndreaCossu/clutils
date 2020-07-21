class MonitorActivations():
    def __init__(self):
        self.hs = []

    def __call__(self, module, module_in, module_out):
        self.hs.append(module_out)

    def reset(self):
        self.hs = []
    
    def get_activations(self, model, x):
        model(x)
        return self.hs

    def remove_hooks(self, hooks):
        for h in hooks:
            h.remove()