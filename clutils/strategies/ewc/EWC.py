import torch
from collections import defaultdict
from copy import deepcopy


class EWC():
    def __init__(self, model, device, lamb=1, 
            normalize=True, single_batch=False, cumulative='none'):
        '''

        Task IDs are ordered integer (0,1,2,3,...)

        :param lamb: task specific hyper-parameter for EWC.
        :param normalize: normalize final fisher matrix in [0,1] (normalization computed among all parameters).
        :param single_batch: if True compute fisher by averaging gradients pattern by pattern. 
                If False, compute fisher by averaging mini batches.
        :param cumulative: possible values are 'none', 'sum'.
                Keep one separate penalty for each task if 'none'. 

        '''

        self.model = model
        self.lamb = lamb
        self.device = device
        self.normalize = normalize
        self.single_batch = single_batch
        assert(cumulative == 'none' or cumulative == 'sum')
        self.cumulative = cumulative

        self.saved_params = defaultdict(list)
        self.fisher = defaultdict(list)


    def penalty(self, current_task_id):
        '''
        Compute EWC contribution to the total loss
        Sum the contribution over all tasks

        :param current_task_id: current task ID.
        '''

        total_penalty = torch.tensor(0, dtype=torch.float32).to(self.device)

        if self.cumulative == 'none':
            for task in range(current_task_id):
                for (_, param), (_, saved_param), (_, fisher) in zip(self.model.named_parameters(), self.saved_params[task], self.fisher[task]):
                    pad_difference = self._padded_difference(param, saved_param)
                    total_penalty += (fisher * pad_difference.pow(2)).sum()
        elif self.cumulative == 'sum' and current_task_id > 0:
            for (_, param), (_, saved_param), (_, fisher) in zip(self.model.named_parameters(), self.saved_params[current_task_id], self.fisher[current_task_id]):
                pad_difference = self._padded_difference(param, saved_param)
                total_penalty += (fisher * pad_difference.pow(2)).sum()            

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

    def update_importance(self, current_task_id, fisher):
        '''
        :param current_task_id: current task ID >= 0
        :fisher: fisher diagonal
        '''
        
        # store learned parameters and fisher coefficients
        # no need to store all the tensor metadata, just its data (data.clone())
        self.saved_params[current_task_id] = [ ( k, param.data.clone() ) for k, param in self.model.named_parameters() ]
        
        if self.cumulative == 'none' or current_task_id == 0:
            self.fisher[current_task_id] = fisher

        elif self.cumulative == 'sum' and current_task_id > 0:
            self.fisher[current_task_id] = self.fisher[current_task_id-1]
            for (k1,curr_imp),(k2,imp) in zip(self.fisher[current_task_id], fisher):
                assert(k1==k2)
                curr_imp += imp

    def compute_importance(self, optimizer, criterion, task_id, loader,
            update=True, truncated_time=0):
        '''
        :param update: update EWC structure with final fisher
        :truncated_time: 0 to compute gradients along all the sequence
                A positive value to use only last `truncated_time` sequence steps.
        '''

        self.model.train()

        # list of list
        fisher_diag = [ ( k, torch.zeros_like(p).to(self.device) ) for k,p in self.model.named_parameters() ]

        for i, (x,y) in enumerate(loader):
            x, y = x.to(self.device), y.to(self.device)

            if self.single_batch:
                for b in range(x.size(0)):
                    x_cur = x[b].unsqueeze(0)
                    y_cur = y[b].unsqueeze(0)

                    optimizer.zero_grad()
                    if truncated_time > 0:
                        out = self.model(x_cur, truncated_time=truncated_time)
                    else:
                        out = self.model(x_cur)
                    loss = criterion(out, y_cur)
                    loss.backward()
                    for (k1,p),(k2,f) in zip(self.model.named_parameters(), fisher_diag):
                        assert(k1==k2)
                        f += p.grad.data.clone().pow(2)
            else:
                optimizer.zero_grad()
                if truncated_time > 0:
                    out = self.model(x, truncated_time=truncated_time)
                else:
                    out = self.model(x)
                loss = criterion(out, y)
                loss.backward()

                for (k1,p),(k2,f) in zip(self.model.named_parameters(), fisher_diag):
                    assert(k1==k2)
                    f += p.grad.data.clone().pow(2)
        
        max_f = -1
        min_f = 1e7
        for _, f in fisher_diag:
            
            if self.single_batch:
                f /= ( float(x.size(0)) * float(len(loader)))
            else:
                f /= float(len(loader))

            # compute max and min among every parameter group
            if self.normalize:
                curr_max_f, curr_min_f = f.max(), f.min()
                max_f = max(max_f, curr_max_f)
                min_f = min(min_f, curr_min_f)

        unnormalized_fisher = deepcopy(fisher_diag)

        # max-min normalization among every parameter group
        if self.normalize:
            r = max(max_f - min_f, 1e-6)
            for _, f in fisher_diag:
                f -= min_f
                f /= r

        if update:
            self.update_importance(task_id, fisher_diag)


        return fisher_diag, unnormalized_fisher
