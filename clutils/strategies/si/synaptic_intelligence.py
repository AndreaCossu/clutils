import torch
from ..base_reg import BaseReg
from ..utils import zerolike_params_dict, copy_params_dict, padded_op


class SI(BaseReg):
    """
    Synaptic Intelligence
    """

    def __init__(self, model, device, lamb=1, eps=1e-3, cumulative='sum'):
        """
        :param eps: tolerance value for importance denominator
        """

        super(SI, self).__init__(model, device, lamb, cumulative=cumulative)

        self.eps = eps
        self.before_training_step()


    def reset_task(self, task_id):

        self.omega = zerolike_params_dict(self.model)

        if task_id == 0:
            self.saved_params[task_id] = copy_params_dict(self.model)


    def compute_importance(self, task_id, compute_for_head=True):

        importance = []

        for (k1, w), (k2, curr_pars), (k3, old_pars) in zip(self.omega, self.model.named_parameters(), self.saved_params[task_id]):
            assert(k1==k2==k3)

            if compute_for_head or (not k1.startswith('layers.out')):
                importance.append((
                    k1, padded_op(
                        w, padded_op(old_pars, curr_pars.detach().clone(), op='-')**2 + self.eps,
                        op='/')
                    )) # this means: w / ((old-curr)**2 + eps)
            else:
                importance.append( (k1, torch.zeros_like(curr_pars, device=self.device)))

        self.update_importance(task_id, importance, save_pars=True)

        return importance


    def before_training_step(self):
        self.previous_pars = copy_params_dict(self.model)


    def update_omega(self, task_id):
        for (k1, w), (k2, parold), (k3, par) in zip(self.omega, self.previous_pars, self.model.named_parameters()):
            deltapar = par.detach().clone() - parold
            if par.grad is not None:
                w += par.grad * deltapar
            else:
                print(f"None gradients found when updating omega on weights {k1}")

