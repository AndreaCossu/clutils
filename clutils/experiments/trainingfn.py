import torch

class Trainer():

    def __init__(self, model, optimizer, criterion, device,
            eval_metric=None, clip_grad=0, penalties=None):
        """
        :param clip_grad: > 0 to clip gradient after backward. 0 not to clip.
        :param penalties: dictionary of penalties name->hyperparams. None to disable them.
        """

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.eval_metric = eval_metric
        self.clip_grad = clip_grad
        self.penalties = penalties
        self.device = device

    def train(self, x, y):
        self.model.train()

        self.optimizer.zero_grad()

        out = self.model(x)

        loss = self.criterion(out, y)
        loss += self.add_penalties()

        loss.backward()
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_grad)

        self.optimizer.step()
        metric = self.eval_metric(out, y) if self.eval_metric else None

        return loss.item(), metric


    def test(self, x, y):
        with torch.no_grad():
            self.model.eval()

            out = self.model(x)

            loss = self.criterion(out, y)
            metric = self.eval_metric(out, y) if self.eval_metric else None

            return loss.item(), metric

    def train_ewc(self, x, y, ewc, task_id):

        self.model.train()

        self.optimizer.zero_grad()

        out = self.model(x)

        loss = self.criterion(out, y)
        loss += self.add_penalties()
        loss += ewc.penalty(task_id)
        loss.backward()
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_grad)
        self.optimizer.step()

        metric = self.eval_metric(out, y) if self.eval_metric else None

        return loss.item(), metric

    def train_si(self, x, y, si, task_id):
        self.model.train()

        self.optimizer.zero_grad()

        si.before_training_step()

        out = self.model(x)

        loss = self.criterion(out, y)
        loss.backward(retain_graph=True)
        si.update_omega(task_id)
        loss += self.add_penalties()
        loss += si.penalty(task_id)
        loss.backward()

        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_grad)

        self.optimizer.step()

        metric = self.eval_metric(out, y) if self.eval_metric else None

        return loss.item(), metric

    def train_mas(self, x, y, mas, task_id):

        self.model.train()

        self.optimizer.zero_grad()

        out = self.model(x)

        loss = self.criterion(out, y)
        loss += self.add_penalties()
        loss += mas.penalty(task_id)
        loss.backward()
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_grad)
        self.optimizer.step()

        metric = self.eval_metric(out, y) if self.eval_metric else None

        return loss.item(), metric
    
    def train_slnid(self, x, y, slnid, task_id, reg=None):
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(x)
        p = slnid.penalty()
        loss = self.criterion(out, y) + p
        loss += self.add_penalties()
        if reg:
            loss += reg.penalty(task_id)
        loss.backward()
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_grad)
        self.optimizer.step()

        metric = self.eval_metric(out, y) if self.eval_metric else None

        return loss.item(), metric

    def train_jacobian(self, x,y, jac, task_id):
        self.model.train()

        self.optimizer.zero_grad()

        out = self.model(x)

        loss = self.criterion(out, y)
        loss += self.add_penalties()
        loss += jac.penalty(task_id)
        loss.backward()
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_grad)
        self.optimizer.step()

        metric = self.eval_metric(out, y) if self.eval_metric else None

        return loss.item(), metric


    def add_penalties(self):
        penalty = torch.tensor(0.).to(self.device)
        if self.penalties:
            
            if 'l1' in self.penalties.keys():
                penalty = l1_penalty(self.model, self.penalties['l1'], self.device)
            
        return penalty


def l1_penalty(model, lamb, device):

    penalty = torch.tensor(0.).to(device)

    for p in model.parameters():
        if p.requires_grad:
            penalty += torch.sum(torch.abs(p))

    return lamb * penalty
