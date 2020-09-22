import torch
import copy
from ..utils import distillation_loss


class LWF():
    """
    Adapted from https://github.com/vlomonaco/avalanche/blob/master/avalanche/training/strategies/lwf/lwf.py
    """

    def __init__(self, model, device, classes_per_task=2, alpha=1, temperature=2, warmup_epochs=2):
        """
        :param alpha: distillation loss hyperparameter
        :param temperature: softmax temperature for distillation
        """

        self.model = model
        self.device = device

        # LwF parameters
        self.classes_per_task = classes_per_task
        self.temperature = temperature
        self.alpha = alpha
        self.warmup_epochs = warmup_epochs

        self.prev_model = None


    def warmup(self, train_loader, task_id):
        """
        Before each task
        """

        if task_id > 0:
            opt = torch.optim.SGD(lr=0.01,
                                params=self.model.layers['out'].parameters())

            for ep in range(self.warmup_epochs):
                for x,y in train_loader:
                    opt.zero_grad()
                    x = x.to(self.device)
                    y = y.to(self.device)
                    out = self.model(x)

                    # loss computed only on the new classes
                    loss = torch.nn.functional.cross_entropy(
                        out[:, task_id*self.classes_per_task:(task_id+1)*self.classes_per_task],
                        y - task_id*self.classes_per_task, reduction='mean')

                    loss.backward()
                    opt.step()


    def penalty(self, out, x):
        if self.prev_model is not None:
            y_prev = self.prev_model(x).detach()
            dist_loss = distillation_loss(out, y_prev,
                                           self.temperature)
            return self.alpha * dist_loss
        else:
            return 0


    def save_model(self):
        """
        After each task
        """
        self.prev_model = copy.deepcopy(self.model)  