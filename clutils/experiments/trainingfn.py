import torch

class Trainer():

    def __init__(self, model, optimizer, criterion, 
            eval_metric=None, clip_grad=0):
        """
        :param clip_grad: > 0 to clip gradient after backward. 0 not to clip.
        """

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.eval_metric = eval_metric
        self.clip_grad = clip_grad

    def train(self, x, y):
        self.model.train()

        self.optimizer.zero_grad()

        out = self.model(x)

        loss = self.criterion(out, y)
        metric = self.eval_metric(out, y) if self.eval_metric else None
        loss.backward()
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_grad)
        self.optimizer.step()

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
        loss += ewc.penalty(task_id)
        metric = self.eval_metric(out, y) if self.eval_metric else None
        loss.backward()
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_grad)        
        self.optimizer.step()

        return loss.item(), metric

    def train_mas(self, x, y, mas, task_id):

        self.model.train()

        self.optimizer.zero_grad()

        out = self.model(x)

        loss = self.criterion(out, y)
        loss += mas.penalty(task_id)
        metric = self.eval_metric(out, y) if self.eval_metric else None
        loss.backward()
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_grad)        
        self.optimizer.step()

        return loss.item(), metric
    
    def train_slnid(self, x, y, slnid, task_id, reg=None):
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(x)
        p = slnid.penalty()
        loss = self.criterion(out, y) + p
        if reg:
            loss += reg.penalty(task_id)
        metric = self.eval_metric(out, y) if self.eval_metric else None
        loss.backward()
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_grad)        
        self.optimizer.step()

        return loss.item(), metric

    def train_jacobian(self, x,y, jac, task_id):
        self.model.train()

        self.optimizer.zero_grad()

        out = self.model(x)

        loss = self.criterion(out, y)
        loss += jac.penalty(task_id)
        metric = self.eval_metric(out, y) if self.eval_metric else None
        loss.backward()
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_grad)        
        self.optimizer.step()

        return loss.item(), metric
