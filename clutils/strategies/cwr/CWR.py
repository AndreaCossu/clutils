import torch
from clutils.models.utils import init_weights, zero_weight

class CWR():

    def __init__(self, head, device):
        self.head = head # will store tw
        self.cw = torch.nn.Linear(head.in_features, head.out_features).to(device)
        self.init_weights(self.cw)


    def init_weights(self, head):
        init_weights(head, zero_weight)


    def reset_tw(self):
        with torch.no_grad():
            self.init_weights(self.head) # reset tw

    
    def update_cw(self, y):
        with torch.no_grad():
            indexes = list(set(y.numpy()))
            average_weight = torch.mean(self.head.weight.data[indexes], dim=1)
            self.cw.weight.data[indexes] -= average_weight.view(-1,1)
            self.cw.bias.data[indexes] = self.head.bias.data[indexes]


    def update_head_with_cw(self):
        with torch.no_grad():
            self.head.weight.data = self.cw.weight.data
            self.head.bias.data = self.cw.bias.data
        