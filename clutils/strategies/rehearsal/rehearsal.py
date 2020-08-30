import torch
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset


class Rehearsal():
    def __init__(self, patterns_per_class):

        self.patterns_per_class = patterns_per_class

        self.patterns = {}

    def record_patterns(self, dataloader):
        """
        Update rehearsed patterns with the current data
        """

        for x,y in dataloader:
            # loop over each minibatch
            for el, _t in zip(x,y):
                t = _t.item()
                if t not in self.patterns:
                    self.patterns[t] = el.unsqueeze(0).clone()
                elif self.patterns[t].size(0) < self.patterns_per_class:
                    self.patterns[t] = torch.cat( (self.patterns[t], el.unsqueeze(0).clone()) )
    
    def _tensorize(self):
        """
        Put the rehearsed pattern into a TensorDataset
        """

        x = []
        y = []
        for k, v in self.patterns.items():
            x.append(v)
            y.append(torch.ones(v.size(0)).long() * k)

        x, y = torch.cat(x), torch.cat(y)

        return TensorDataset(x, y)

    def augment_dataset(self, dataloader):
        """
        Add rehearsed pattern to current dataloader
        """

        if self.patterns == {}: # first task
            return dataloader
        else:
            return DataLoader( ConcatDataset((
                    dataloader.dataset, 
                    self._tensorize()
                )), shuffle=True, drop_last=True, batch_size=dataloader.batch_size)