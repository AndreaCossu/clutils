import torch
from collections import defaultdict
from copy import deepcopy


class MAS():
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

        self.model = model
        self.lamb = lamb
        self.device = device
        self.normalize = normalize
        self.single_batch = single_batch

        self.saved_params = defaultdict(list)
        self.importance = defaultdict(list)


    def penalty(self, current_task_id):
        '''
        Compute EWC contribution to the total loss
        Sum the contribution over all tasks

        :param current_task_id: current task ID.
        '''

        total_penalty = torch.tensor(0, dtype=torch.float32).to(self.device)


        # for each previous task (if any)
        if current_task_id > 0:
            for (_, param), (_, saved_param), (_, importance) in zip(self.model.named_parameters(), self.saved_params[current_task_id], self.importance[current_task_id]):
                pad_difference = self._padded_difference(param, saved_param)
                total_penalty += (importance * pad_difference.pow(2)).sum()

        return self.lamb * total_penalty
        
    def _padded_difference(self, p1, p2):
        """
        Return the difference between p1 and p2. Result size is size(p2).
        If p1 and p2 sizes are different, simply compute the difference 
        by cutting away additional values and zero-pad result to obtain back the original dimension.
        """

        assert(len(p1.size()) == len(p2.size()) < 3)

        if p1.size() == p2.size():
            return p1 - p2


        min_size = torch.Size([
            min(a, b)
            for a,b in zip(p1.size(), p2.size())
        ])
        if len(p1.size()) == 2:
            resizedp1 = p1[:min_size[0], :min_size[1]]
            resizedp2 = p2[:min_size[0], :min_size[1]]
        else:
            resizedp1 = p1[:min_size[0]]
            resizedp2 = p2[:min_size[0]]


        difference = resizedp1 - resizedp2
        padded_difference = torch.zeros(p2.size(), device=p2.device)
        if len(p1.size()) == 2:
            padded_difference[:difference.size(0), :difference.size(1)] = difference
        else:
            padded_difference[:difference.size(0)] = difference

        return padded_difference

    def update_importance(self, current_task_id, importance):
        '''
        :param current_task_id: current task ID >= 0
        :importance: parameter importance matrix
        '''
        
        # store learned parameters and importance coefficients
        # no need to store all the tensor metadata, just its data (data.clone())
        self.saved_params[current_task_id] = [ ( k, param.data.clone() ) for k, param in self.model.named_parameters() ]
        
        if current_task_id > 0:
            self.importance[current_task_id] = importance + self.importance[current_task_id-1]
        else:
            self.importance[current_task_id] = importance


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
        
        max_imp = -1
        min_imp = 1e7
        for _, imp in importance:
            
            if self.single_batch:
                imp /= ( float(x.size(0)) * float(len(loader)))
            else:                
                imp /= float(len(loader))

            # compute max and min among every parameter group
            if self.normalize:
                curr_max_imp, curr_min_imp = imp.max(), imp.min()
                max_imp = max(max_imp, curr_max_imp)
                min_imp = min(min_imp, curr_min_imp)

        unnormalized_imp = deepcopy(importance)

        # max-min normalization among every parameter group
        if self.normalize:
            r = max(max_imp - min_imp, 1e-6)
            for _, imp in importance:
                imp -= min_imp
                imp /= r

        if update:
            self.update_importance(task_id, importance)

        return importance, unnormalized_imp
