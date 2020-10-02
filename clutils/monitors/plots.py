import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')
import pandas as pd
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from matplotlib.ticker import MaxNLocator


def plot_learning_curves(models, result_folder, additional_metrics=['acc'], title=True, filename='training_results.csv'):
    '''
    :param models: list of modelnames to be used for plots.
    '''

    if isinstance(models, str):
        models = [models]

    with open(os.path.join(result_folder, filename), 'r') as f:
        data_csv = pd.read_csv(f)
        data_csv['task_id'] = pd.Categorical(data_csv.task_id)
        data_csv['epoch'] = data_csv['epoch'].astype(int)

        sns.set_palette(sns.color_palette("bright"))

        for modelname in models:
            data_model = data_csv[data_csv['model'] == modelname]

            for metric_type in ['loss'] + additional_metrics:
                data_plot = pd.melt(
                data_model[['epoch', 'task_id', 'train_'+metric_type, 'validation_'+metric_type]], \
                    id_vars=['epoch', 'task_id'], value_vars=['train_'+metric_type, 'validation_'+metric_type],
                    value_name=metric_type)

                rp = sns.relplot(
                    x='epoch', kind='line', y=metric_type, hue='task_id', sort=False,
                    legend='full', style='variable', markers=True,
                    data=data_plot[data_plot['epoch'] > 0])

                rp.ax.xaxis.set_major_locator(MaxNLocator(integer=True))

                if title:
                    plt.subplots_adjust(top=0.9)
                    plt.title(f"{modelname} {metric_type}")

                rp.fig.savefig(os.path.join(result_folder, f"{modelname}_{metric_type}.png"))


def create_writer(folder):
    '''
    Create Tensorboard writer
    '''

    return SummaryWriter(os.path.join(folder, 'tensorboard'))


def plot_importance(writer, modelname, importance, task_id, epoch=0):
    for paramname, imp in importance:
        if len(imp.size()) == 1: # bias
            writer.add_image(f"{modelname}-{paramname}_importance/{task_id}", imp.unsqueeze(0).cpu().data, epoch, dataformats='HW')
        else:
            writer.add_image(f"{modelname}-{paramname}_importance/{task_id}", imp.cpu().view(imp.size(0),-1).data, epoch, dataformats='HW')
        writer.add_histogram(f"{modelname}-{paramname}_importance_hist/{task_id}", imp.cpu().view(-1).data, epoch)


def plot_gradients(writer, modelname, model, task_id, epoch=0):
    for paramname, grad_matrix in model.named_parameters():
        if len(grad_matrix.size()) == 1: # bias
            writer.add_image(f"{modelname}-{paramname}/{task_id}_grad", grad_matrix.unsqueeze(0).cpu().data, epoch, dataformats='HW')
        else: # weights
            writer.add_image(f"{modelname}-{paramname}/{task_id}_grad", grad_matrix.cpu().view(grad_matrix.size(0),-1).data, epoch, dataformats='HW')
        writer.add_histogram(f"{modelname}-{paramname}_grad_hist/{task_id}", grad_matrix.cpu().view(-1).data, epoch)


def plot_weights(writer, modelname, model, task_id, epoch=0):
    for paramname, weight_matrix in model.named_parameters():
        if len(weight_matrix.size()) == 1: # bias
            writer.add_image(f"{modelname}-{paramname}/{task_id}", weight_matrix.unsqueeze(0).cpu().data, epoch, dataformats='HW')
        else: # weights
            writer.add_image(f"{modelname}-{paramname}/{task_id}", weight_matrix.cpu().view(weight_matrix.size(0),-1).data, epoch, dataformats='HW')
        try:
            writer.add_histogram(f"{modelname}-{paramname}_hist/{task_id}", weight_matrix.cpu().view(-1).data, epoch)
        except ValueError:
            print(modelname)
            print(paramname)
            print(weight_matrix.size())
            print(weight_matrix)
            raise ValueError


def plot_activations(writer, modelname, activations, task_id, epoch=0):
    """
    :param activations: list of (hidden_size)
                 or (T, hidden_size) or (batch, T, hidden_size) tensors
    """

    for i, activation in enumerate(activations):

        if len(activation.size()) == 3:
            activation = activation.mean(0)

        if len(activation.size()) == 1:
            activation = activation.unsqueeze(0)

        writer.add_image(f"{modelname}-activation{i}/{task_id}", activation.cpu().data, epoch, dataformats='HW')
        writer.add_histogram(f"{modelname}-activation{i}_hist/{task_id}", activation.cpu().view(-1).data, epoch)


def plot_importance_units(writer, modelname, importances, task_id, epoch=0):
    """
    :param importances: list of (hidden_size)
                 or (T, hidden_size) or (batch, T, hidden_size) tensors
    """

    for i, importance in enumerate(importances):

        if len(importance.size()) == 3:
            importance = importance.mean(0)

        if len(importance.size()) == 1:
            importance = importance.unsqueeze(0)

        writer.add_image(f"{modelname}-importance{i}/{task_id}", importance.cpu().data, epoch, dataformats='HW')
        writer.add_histogram(f"{modelname}-importance{i}_hist/{task_id}", importance.cpu().view(-1).data, epoch)


def get_matrix_from_modelname(model, modelname):
    if modelname == 'esn':
        label = 'h2h'
        weight_matrix = model.weights[label].data
    elif modelname == 'mlp':
        label = 'i2h'
        weight_matrix = model.layers[label].weight.data
    elif modelname == 'lwta':
        label = 'i2h'
        weight_matrix = model.layers[label].weight.data
    elif modelname == 'lmn':
        label = 'm2m'
        weight_matrix = model.layers[label].weight.data
    elif modelname == 'rnn':
        label = 'rnn'
        weight_matrix = model.layers[label].weight_hh_l0.data
    elif modelname == 'lstm':
        label = 'rnn'
        weight_matrix = model.layers[label].weight_hh_l0.data
    elif modelname == 'cnn':
        label = 'conv1'
        weight_matrix = model.layers[label].weight
    
    return weight_matrix, label


def read_accuracies(filepath, extract_last=True):
    """
    Read final test accuracies from intermediate results file.
    """

    d = pd.read_csv(filepath)
    if extract_last:
        accs = d[d['training_task']==d['training_task'].max()]['acc'].values
    else:
        accs = d['acc'].values
    return accs


def average_accuracies(filepaths, n_tasks=5):
    """
    Return average and std of final test accuracies over a set of runs
    """

    assert(len(filepaths) > 1)

    accs = np.empty( (len(filepaths), n_tasks) )
    for i, fp in enumerate(filepaths):
        accs[i, :] = read_accuracies(fp)
    means = accs.mean(axis=1)
    stds = accs.std(axis=1)

    return means, stds
