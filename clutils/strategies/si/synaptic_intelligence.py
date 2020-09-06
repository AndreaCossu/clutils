import torch
from ..base_reg import BaseReg
from ..utils import zerolike_params_dict, copy_params_dict, padded_difference


class SI(BaseReg):
    """
    Synaptic Intelligence
    """

    def __init__(self, model, device, lamb=1,
            normalize=True, single_batch=False, eps=1e-3):
        """
        :param eps: tolerance value for importance denominator
        """

        self.cumulative_mode = 'sum' # 'none'

        super(SI, self).__init__(model, device, lamb, normalize, single_batch, cumulative=self.cumulative_mode)

        self.eps = eps
        self.before_training_step()


    def reset_task(self, task_id):

        self.omega = zerolike_params_dict(self.model)

        if task_id == 0:
            self.saved_params[-1] = copy_params_dict(self.model)
        else:
            self.saved_params[task_id] = copy_params_dict(self.model)


        if task_id == 0 or self.cumulative_mode == 'none':
            self.importance[task_id] = zerolike_params_dict(self.model)
        else:
            self.importance[task_id] = self.importance[task_id-1]


    def compute_importance(self, task_id):

        for (k1, w), (k2, imp), (k3, curr_pars), (k4, old_pars) in zip(self.omega, self.importance[task_id], self.model.named_parameters(), self.saved_params[task_id-1]):
                imp += w / ((padded_difference(old_pars, curr_pars))**2 + self.eps)

        self.saved_params[task_id] = copy_params_dict(self.model)

        return self.importance[task_id]


    def before_training_step(self):
        self.previous_pars = copy_params_dict(self.model)


    def save_gradients(self):
        self.omega_gradients = copy_params_dict(self.model, copy_grad=True)


    def update_omega(self, task_id):
        for (k1, w), (k2, grad), (k3, parold), (k4, par) in zip(self.omega, self.omega_gradients, self.previous_pars, self.model.named_parameters()):
            deltapar = padded_difference(par, parold)
            if grad is not None:
                w += grad * deltapar
            else:
                print(f"None gradients found when updating omega on weights {k1}")

