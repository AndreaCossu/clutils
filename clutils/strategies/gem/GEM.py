import torch
import random


class AGEM():

    def __init__(self, patterns_per_step, sample_size):

        self.patterns_per_step = int(patterns_per_step)
        self.sample_size = int(sample_size)

        self.reference_gradients = None
        self.memory_x, self.memory_y = None, None

    def project_gradients(self, model):
        if self.memory_x is not None:
            for (n1, p1), (n2, refg) in zip(model.named_parameters(),
                                    self.reference_gradients):

                assert n1 == n2, "Different model parameters in AGEM projection"
                assert (p1.grad is not None and refg is not None) \
                        or (p1.grad is None and refg is None)

                if refg is None:
                    continue

                dotg = torch.dot(p1.grad.view(-1), refg.view(-1))
                dotref = torch.dot(refg.view(-1), refg.view(-1))
                if dotg < 0:
                    p1.grad -= (dotg / dotref) * refg

    def sample_from_memory(self, sample_size):
        """
        Sample a minibatch from memory.
        Return a tuple of patterns (tensor), targets (tensor).
        """

        if self.memory_x is None or self.memory_y is None:
            raise ValueError('Empty memory for AGEM.')

        if self.memory_x.size(0) <= sample_size:
            return self.memory_x, self.memory_y
        else:
            idxs = random.sample(range(self.memory_x.size(0)), sample_size)
            return self.memory_x[idxs], self.memory_y[idxs]

    @torch.no_grad()
    def update_memory(self, dataloader):
        """
        Update replay memory with patterns from current step.
        """

        tot = 0
        for x, y in dataloader:
            if tot + x.size(0) <= self.patterns_per_step:
                if self.memory_x is None:
                    self.memory_x = x.clone()
                    self.memory_y = y.clone()
                else:
                    self.memory_x = torch.cat((self.memory_x, x), dim=0)
                    self.memory_y = torch.cat((self.memory_y, y), dim=0)
            else:
                diff = self.patterns_per_step - tot
                if self.memory_x is None:
                    self.memory_x = x[:diff].clone()
                    self.memory_y = y[:diff].clone()
                else:
                    self.memory_x = torch.cat((self.memory_x, x[:diff]), dim=0)
                    self.memory_y = torch.cat((self.memory_y, y[:diff]), dim=0)
                break
            tot += x.size(0)
