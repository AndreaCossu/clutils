import torch
import math
from collections import defaultdict, Counter


class CWR():

    def __init__(self, head, device):
        self.head = head # will store tw during training, cw after training on all tasks
        self.cw = torch.nn.Linear(head.in_features, head.out_features).to(device)
        self.cw.weight.data.fill_(0.)

        self.device = device

        self.classes_count = defaultdict(int)
        
    def pre_batch(self, y):
        with torch.no_grad():
            self.head.weight.data.zero_()

            current_classes = list(set(y.cpu().numpy()))
            self.head.weight.data[current_classes] = self.cw.weight.data[current_classes]
    
    def update_cw(self, y):
        with torch.no_grad():

            current_classes_count = dict(Counter(list(y.cpu().numpy())))
            current_classes = list(current_classes_count.keys())

            weigh = torch.empty(len(current_classes), device=self.device)
            for i, k in enumerate(sorted(current_classes)):
                weigh[i] = math.sqrt(float(self.classes_count[k]) / float(current_classes_count[k]))

            average_weight = torch.mean(self.head.weight.data[current_classes], dim=1)
            self.cw.weight.data[current_classes] = (\
                    (self.cw.weight.data[current_classes] * weigh.view(-1,1)) + \
                    (self.head.weight.data[current_classes] - average_weight.view(-1,1)) \
                    ) / (weigh + 1).view(-1,1)
            
            for k,v in current_classes_count.items():
                self.classes_count[k] += v

    def update_head_with_cw(self):
        with torch.no_grad():
            self.head.weight.data = self.cw.weight.data.clone()
        