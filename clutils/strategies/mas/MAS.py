import torch
from collections import defaultdict
from copy import deepcopy
from ..base_reg import BaseReg
from ..utils import normalize_blocks


class MAS(BaseReg):
    """
    Memory Aware Synapses
    """

    def __init__(self, model, device, lamb=1, normalize=True, single_batch=False):
        '''
        Task IDs are ordered integer (0,1,2,3,...)

        :param lamb: task specific hyper-parameter for Memory Aware Synapses.
        :param normalize: normalize final importance matrix in [0,1] (normalization computed among all parameters).
        :param single_batch: if True compute importance by averaging gradients pattern by pattern. 
                If False, compute importance by averaging mini batches.
        '''

        super(MAS, self).__init__(model, device, lamb, normalize, single_batch, cumulative='sum')


    def compute_importance(self, optimizer, task_id, loader, 
            update=True, truncated_time=0):
        '''
        :param update: Update MAS importance        
        :param truncated_time: 0 to compute gradients along all the sequence
                A positive value to use only last `truncated_time` sequence steps.
        '''

        self.model.train()

        # list of list
        importance = [ ( k, torch.zeros_like(p).to(self.device) ) for k,p in self.model.named_parameters() ]

        for i, (x,_) in enumerate(loader):
            x = x.to(self.device)

            if self.single_batch:
                for b in range(x.size(0)):
                    x_cur = x[b].unsqueeze(0)

                    optimizer.zero_grad()
                    if truncated_time > 0:
                        out = self.model(x_cur, truncated_time=truncated_time)
                    else:
                        out = self.model(x_cur)
                    # out = torch.softmax(out, dim=-1)
                    loss = out.norm(p=2).pow(2)
                    loss.backward()
                    for (k1,p),(k2,imp) in zip(self.model.named_parameters(), importance):
                        assert(k1==k2)
                        imp += p.grad.data.clone().abs()
            else:
                optimizer.zero_grad()
                if truncated_time > 0:
                    out = self.model(x, truncated_time=truncated_time)
                else:
                    out = self.model(x)
                # out = torch.softmax(out, dim=-1)
                loss = out.norm(p=2).pow(2)
                loss.backward()

                for (k1,p),(k2,imp) in zip(self.model.named_parameters(), importance):
                    assert(k1==k2)
                    imp += p.grad.data.clone().abs()
        
        for _, imp in importance:
            
            if self.single_batch:
                imp /= ( float(x.size(0)) * float(len(loader)))
            else:                
                imp /= float(len(loader))

        unnormalized_imp = deepcopy(importance)

        if self.normalize:
            importance = normalize_blocks(importance)

        if update:
            self.update_importance(task_id, importance)

        return importance, unnormalized_imp
