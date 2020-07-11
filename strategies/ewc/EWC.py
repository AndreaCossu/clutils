import torch
import torch.nn as nn
from collections import defaultdict

class EWC():
    def __init__(self, device, lamb=1):
        '''

        Task IDs are ordered integer (0,1,2,3,...)

        :param lamb: task specific hyper-parameter for EWC.
        '''
 
        self.lamb = lamb
        self.device = device

        self.saved_params = defaultdict(lambda: defaultdict(list))
        self.fisher = defaultdict(lambda: defaultdict(list))


    def ewc_loss(self, model, modelname, current_task_id):
        '''
        Compute EWC contribution to the total loss
        Sum the contribution over all tasks

        :param model: model to be optimized
        :param modelname: name of the model
        :param current_task_id: current task ID.
        '''

        total_penalty = torch.tensor(0, dtype=torch.float32).to(self.device)

        # for each previous task (if any)
        for task in range(current_task_id):
            for (_, param), (_, saved_param), (_, fisher) in zip(model.named_parameters(), self.saved_params[modelname][task], self.fisher[modelname][task]):
                total_penalty += (fisher * (param - saved_param).pow(2)).sum()

        return self.lamb * total_penalty
        

    def update_fisher_importance(self, model, modelname, current_task_id, fisher):
        '''
        :param model: model to be optimized
        :param modelname: name of the model
        :param current_task_id: current task ID >= 0
        :fisher: fisher diagonal
        '''
        
        # store learned parameters and fisher coefficients
        # no need to store all the tensor metadata, just its data (data.clone())
        self.saved_params[modelname][current_task_id] = [ ( k, param.data.clone() ) for k, param in model.named_parameters() ]
        
        #self.fisher[modelname][current_task_id] = [ param.grad.data.clone().pow(2) for param in model.parameters() ]
        self.fisher[modelname][current_task_id] = fisher
