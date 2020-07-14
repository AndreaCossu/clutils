import torch


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
