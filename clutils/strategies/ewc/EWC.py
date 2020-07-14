import torch
from collections import defaultdict
from copy import deepcopy


class EWC():
    def __init__(self, device, lamb=1):
        '''

        Task IDs are ordered integer (0,1,2,3,...)

        :param lamb: task specific hyper-parameter for EWC.
        '''
 
        self.lamb = lamb
        self.device = device

        self.saved_params = defaultdict(list)
        self.fisher = defaultdict(list)


    def ewc_loss(self, model, current_task_id):
        '''
        Compute EWC contribution to the total loss
        Sum the contribution over all tasks

        :param model: model to be optimized
        :param current_task_id: current task ID.
        '''

        total_penalty = torch.tensor(0, dtype=torch.float32).to(self.device)


        # for each previous task (if any)
        for task in range(current_task_id):
            for (_, param), (_, saved_param), (_, fisher) in zip(model.named_parameters(), self.saved_params[task], self.fisher[task]):
                pad_difference = self._padded_difference(param, saved_param)
                total_penalty += (fisher * pad_difference.pow(2)).sum()

        return self.lamb * total_penalty
        
    def _padded_difference(self, p1, p2):
        """
        Return the difference between p1 and p2. Result size is size(p2).
        If p1 and p2 sizes are different, simply compute the difference 
        by cutting away additional values and zero-pad result to obtain back the original dimension.
        """

        assert(len(p1.size()) == len(p2.size()) == 2)

        if p1.size() == p2.size():
            return p1 - p2


        min_size = torch.Size([
            min(p1.size(0), p2.size(0)),
            min(p1.size(1), p2.size(1))
        ])
        new_resized_to_old = p2[:min_size[0], min_size[1]]

        difference = p1 - new_resized_to_old
        padded_difference = torch.zeros(p2.size(), p2.device)
        padded_difference[:difference.size(0), :difference.size(1)] = difference

        return padded_difference

    def update_fisher_importance(self, model, current_task_id, fisher):
        '''
        :param model: model to be optimized
        :param current_task_id: current task ID >= 0
        :fisher: fisher diagonal
        '''
        
        # store learned parameters and fisher coefficients
        # no need to store all the tensor metadata, just its data (data.clone())
        self.saved_params[current_task_id] = [ ( k, param.data.clone() ) for k, param in model.named_parameters() ]
        
        #self.fisher[current_task_id] = [ param.grad.data.clone().pow(2) for param in model.parameters() ]
        self.fisher[current_task_id] = fisher


def compute_fisher(model, optimizer, criterion, loader, device, normalize=True, single_batch=False):
    '''
    :param normalize: normalize final fisher matrix in [0,1] (normalization computed among all parameters).
    :param single_batch: if True compute fisher by averaging gradients pattern by pattern. If False, compute fisher by averaging mini batches.
    '''

    model.train()

    # list of list
    fisher_diag = [ ( k, torch.zeros_like(p).to(device) ) for k,p in model.named_parameters() ]

    for i, (x,y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        if single_batch:
            for b in range(x.size(0)):
                x_cur = x[b].unsqueeze(0)
                y_cur = y[b].unsqueeze(0)

                optimizer.zero_grad()
                out = model(x_cur)
                loss = criterion(out, y_cur)
                loss.backward()
                for (k1,p),(k2,f) in zip(model.named_parameters(), fisher_diag):
                    assert(k1==k2)
                    f += p.grad.data.clone().pow(2)
        else:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()

            for (k1,p),(k2,f) in zip(model.named_parameters(), fisher_diag):
                assert(k1==k2)
                f += p.grad.data.clone().pow(2)
    
    max_f = -1
    min_f = 1e7
    for _, f in fisher_diag:
        
        f /= float(len(loader))
        if single_batch:
            f /= ( float(x.size(0)) * float(len(loader)))

        # compute max and min among every parameter group
        if normalize:
            curr_max_f, curr_min_f = f.max(), f.min()
            max_f = max(max_f, curr_max_f)
            min_f = min(min_f, curr_min_f)

    unnormalized_fisher = deepcopy(fisher_diag)

    # max-min normalization among every parameter group
    if normalize:
        r = max(max_f - min_f, 1e-6)
        for _, f in fisher_diag:
            f -= min_f
            f /= r

    return fisher_diag, unnormalized_fisher
