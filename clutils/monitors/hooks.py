class MonitorLinear():
    def __init__(self):
        self.hs = []

    def __call__(self, module, module_in, module_out):
        self.hs.append(module_out)

    def reset(self):
        self.hs = []