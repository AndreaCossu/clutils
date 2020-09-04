import torch
from collections import defaultdict
from .utils import padded_difference


class BaseReg():
    def __init__(self, model, device, lamb=1, 
            normalize=True, single_batch=False, cumulative='none'):
        '''

        Task IDs are ordered integer (0,1,2,3,...)

        :param lamb: task specific hyper-parameter for the penalization.
        :param normalize: normalize final importance matrix in [0,1] (normalization computed block wise).
        :param single_batch: if True compute fisher by averaging gradients pattern by pattern. 
                If False, compute fisher by averaging mini batches.
        :param cumulative: Keep a single penalty matrix for all tasks or one for each task.
                Possible values are 'none' or 'sum'.
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
        self.importance = defaultdict(list)


    def penalty(self, current_task_id):
        '''
        Compute regularization penalty.
        Sum the contribution over all tasks if importance is not cumulative.

        :param current_task_id: current task ID (0 being the first task)
        '''

        total_penalty = torch.tensor(0, dtype=torch.float32).to(self.device)

        if self.cumulative == 'none':
            for task in range(current_task_id):
                for (_, param), (_, saved_param), (_, imp) in zip(self.model.named_parameters(), self.saved_params[task], self.importance[task]):
                    pad_difference = padded_difference(param, saved_param)
                    total_penalty += (imp * pad_difference.pow(2)).sum()
        elif self.cumulative == 'sum' and current_task_id > 0:
            for (_, param), (_, saved_param), (_, imp) in zip(self.model.named_parameters(), self.saved_params[current_task_id], self.importance[current_task_id]):
                pad_difference = padded_difference(param, saved_param)
                total_penalty += (imp * pad_difference.pow(2)).sum()            

        return self.lamb * total_penalty
        

    def update_importance(self, current_task_id, importance):
        '''
        :param current_task_id: current task ID (0 being the first task)
        :importance : importance for each weight
        '''
        
        # store learned parameters and importance coefficients
        # no need to store all the tensor metadata, just its data (data.clone())
        self.saved_params[current_task_id] = [ ( k, param.data.clone() ) for k, param in self.model.named_parameters() ]
        
        if self.cumulative == 'none' or current_task_id == 0:
            self.importance[current_task_id] = importance

        elif self.cumulative == 'sum' and current_task_id > 0:
            self.importance[current_task_id] = []
            for (k1,curr_imp),(k2,imp) in zip(self.importance[current_task_id-1], importance):
                assert(k1==k2)
                self.importance[current_task_id].append( (k1, padded_difference(imp, curr_imp, use_sum=True)) )
