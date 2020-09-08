import torch
from ..base_reg import BaseReg
from ..utils import zerolike_params_dict, copy_params_dict, padded_op


class SI(BaseReg):
    """
    Synaptic Intelligence
    """

    def __init__(self, model, device, lamb=1, eps=1e-3):
        """
        :param eps: tolerance value for importance denominator
        """

        super(SI, self).__init__(model, device, lamb, cumulative='sum')

        self.eps = eps
        self.before_training_step()


    def reset_task(self, task_id):

        self.omega = zerolike_params_dict(self.model)

        if task_id == 0:
            self.saved_params[task_id] = copy_params_dict(self.model)


    def compute_importance(self, task_id):

        importance = []
        self.importance[task_id] = []

        for (k1, w), (k2, curr_pars), (k3, old_pars) in zip(self.omega, self.model.named_parameters(), self.saved_params[task_id]):
            assert(k1==k2==k3)
            importance.append( 
                (k1, padded_op(w, padded_op(old_pars, curr_pars)**2 + self.eps, op='/')) 
                ) # w / ((old-curr)**2 + eps)

        if task_id > 0:
            for (k1, prev_imp), (k2, curr_imp) in zip(self.importance[task_id-1], importance):
                assert(k1==k2)
                self.importance[task_id].append( (k1, padded_op(curr_imp, prev_imp, op='+')) )
        else:
            self.importance[task_id] = importance

        self.saved_params[task_id] = copy_params_dict(self.model)

        return self.importance[task_id]


    def before_training_step(self):
        self.previous_pars = copy_params_dict(self.model)


    def save_gradients(self):
        self.omega_gradients = copy_params_dict(self.model, copy_grad=True)


    def update_omega(self, task_id):
        for (k1, w), (k2, grad), (k3, parold), (k4, par) in zip(self.omega, self.omega_gradients, self.previous_pars, self.model.named_parameters()):
            deltapar = par - parold
            if grad is not None:
                w += grad * deltapar
            else:
                print(f"None gradients found when updating omega on weights {k1}")

