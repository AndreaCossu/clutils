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
        :param normalize: normalize final fisher matrix in [0,1] (normalization computed among all parameters).
        :param single_batch: if True compute fisher by averaging gradients pattern by pattern. 
                If False, compute fisher by averaging mini batches.
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
        for task in range(current_task_id):
            for (_, param), (_, saved_param), (_, importance) in zip(self.model.named_parameters(), self.saved_params[task], self.importance[task]):
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
        
        # store learned parameters and fisher coefficients
        # no need to store all the tensor metadata, just its data (data.clone())
        self.saved_params[current_task_id] = [ ( k, param.data.clone() ) for k, param in self.model.named_parameters() ]
        
        #self.fisher[current_task_id] = [ param.grad.data.clone().pow(2) for param in self.model.parameters() ]
        self.importance[current_task_id] = importance


    def compute_importance(self, optimizer, task_id, loader, 
            update=True):
        '''
        :param update: Update MAS importance        '''

        self.model.train()

        # list of list
        importance = [ ( k, torch.zeros_like(p).to(self.device) ) for k,p in self.model.named_parameters() ]

        for i, (x,_) in enumerate(loader):
            x = x.to(self.device)

            if self.single_batch:
                for b in range(x.size(0)):
                    x_cur = x[b].unsqueeze(0)

                    optimizer.zero_grad()
                    out = self.model(x_cur)
                    loss = out.norm(p=2).pow(2)
                    loss.backward()
                    for (k1,p),(k2,imp) in zip(self.model.named_parameters(), importance):
                        assert(k1==k2)
                        imp += p.grad.data.clone().pow(2)
            else:
                optimizer.zero_grad()
                out = self.model(x)
                loss = out.norm(p=2).pow(2)
                loss.backward()

                for (k1,p),(k2,imp) in zip(self.model.named_parameters(), importance):
                    assert(k1==k2)
                    imp += p.grad.data.clone().pow(2)
        
        max_imp = -1
        min_imp = 1e7
        for _, imp in importance:
            
            imp /= float(len(loader))
            if self.single_batch:
                imp /= ( float(x.size(0)) * float(len(loader)))

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
