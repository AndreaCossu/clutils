import torch
import random
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset, ConcatDataset

class RehearsalDataset(Dataset):
    def __init__(self, x, y, l):
        self.x = x
        self.y = y
        self.l = l

    def __getitem__(self, item):
        return self.x[item], self.y[item], self.l[item]

    def __len__(self):
        return len(self.x)

class Rehearsal():
    def __init__(self, patterns_per_class, patterns_per_class_per_batch=0):
        """
        :param patterns_per_class_per_batch:
            if <0 -> concatenate patterns to the entire dataloader
            if 0 -> concatenate to the current batch another batch size
                    split among existing classes
            if >0 -> concatenate to the current batch `patterns_per_class_per_batch`
                    patterns for each existing class
        """

        self.patterns_per_class = patterns_per_class
        self.patterns_per_class_per_batch = patterns_per_class_per_batch
        self.add_every_batch = patterns_per_class_per_batch >= 0

        self.patterns = defaultdict(list)
        self.lengths = defaultdict(list)

    def record_patterns(self, dataloader):
        """
        Update rehearsed patterns with the current data
        """

        counter = defaultdict(int)
        for x,y,l in dataloader:
            # loop over each minibatch
            for i, el in enumerate(x):
                t = y[i].item()
                if counter[t] < self.patterns_per_class:
                    self.patterns[t].append(el)
                    self.lengths[t].append(l[i])
                    counter[t] += 1

    def concat_to_batch(self, x,y,l):
        """
        Concatenate subset of memory to the current batch.
        """
        if not self.add_every_batch or self.patterns == {}:
            return x, y, l

        # how many replay patterns per class per batch?
        # either patterns_per_class_per_batch
        # or batch_size split equally among existing classes
        to_add = int(y.size(0) / len(self.patterns.keys())) \
                if self.patterns_per_class_per_batch == 0 \
                else self.patterns_per_class_per_batch

        rehe_x, rehe_y, rehe_l = [*x], [y], [*l]

        for (k1,v), (k2, lmem) in zip(self.patterns.items(), self.lengths.items()):
            assert(k1==k2)

            if to_add >= len(v):
                # take directly the memory
                rehe_x += v
                rehe_l += lmem
            else:
                # select at random from memory
                subset = random.sample(list(zip(v, lmem)), to_add)
                to_add_x, to_add_l = zip(*subset)
                rehe_x += list(to_add_x)
                rehe_l += list(to_add_l)

            rehe_y.append(torch.ones(min(to_add, len(v))).long() * k1)

        return rehe_x, torch.cat(rehe_y, dim=0), rehe_l


    def _tensorize(self):
        """
        Put the rehearsed pattern into a TensorDataset
        """

        x, y, l = [], [], []
        for (k1, v), (k2, lmem) in zip(self.patterns.items(), self.lengths.items()):
            assert(k1==k2)
            x += v
            l += lmem
            y.append(torch.ones(len(v)).long() * k1)

        y = torch.cat(y, dim=0)

        return RehearsalDataset(x, y, l)

    def augment_dataset(self, dataloader):
        """
        Add rehearsed pattern to current dataloader
        """
        if self.add_every_batch or self.patterns == {}:
            return dataloader
        else:
            return DataLoader( ConcatDataset((
                    dataloader.dataset, 
                    self._tensorize()
                )), shuffle=True, drop_last=True, batch_size=dataloader.batch_size)