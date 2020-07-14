import torch

class Trainer():

    def __init__(self, model, optimizer, criterion, eval_metric=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.eval_metric = eval_metric

    def train(self, x, y):
        self.model.train()

        self.optimizer.zero_grad()

        out = self.model(x)

        loss = self.criterion(out, y)
        metric = self.eval_metric(out, y) if self.eval_metric else None
        loss.backward()
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
        loss += ewc.ewc_loss(self.model, task_id)
        metric = self.eval_metric(out, y) if self.eval_metric else None
        loss.backward()
        self.optimizer.step()

        return loss.item(), metric


def vanilla_train(model, optimizer, criterion, x, y, eval_metric=None):
    model.train()

    optimizer.zero_grad()

    out = model(x)

    loss = criterion(out, y)
    metric = eval_metric(out, y) if eval_metric else None
    loss.backward()
    optimizer.step()

    return loss.item(), metric

def vanilla_test(model, criterion, x, y, eval_metric=None):
    with torch.no_grad():
        model.eval()

        out = model(x)

        loss = criterion(out, y)
        metric = eval_metric(out, y) if eval_metric else None

        return loss.item(), metric



def train_ewc(ewc, task_id, model, optimizer, criterion, x, y, eval_metric=None):

    model.train()

    optimizer.zero_grad()

    out = model(x)

    loss = criterion(out, y)
    loss += ewc.ewc_loss(model, task_id)
    metric = eval_metric(out, y) if eval_metric else None
    loss.backward()
    optimizer.step()

    return loss.item(), metric
