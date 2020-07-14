import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')
import pandas as pd
import os
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

    

def plot_fisher(writer, modelname, fisher, task_id):
    for paramname, f in fisher:
        writer.add_histogram(f"{modelname}-{paramname}/{task_id}", f.cpu().view(-1))


def plot_gradients(writer, modelname, grad_matrix, task_id, epoch):
    """
    if modelname == 'esn':
        grad_matrix = model.weights['h2h'].grad
    elif modelname == 'mlp':
        grad_matrix = model.layers['i2h'].weight.grad
    elif modelname == 'lwta':
        grad_matrix = model.layers['i2h'].weight.grad
    elif modelname == 'lmn':
        grad_matrix = model.layers['m2m'].weight.grad
    elif modelname == 'rnn':
        grad_matrix = model.layers['rnn'].weight_hh_l0.grad
    elif modelname == 'lstm':
        grad_matrix = model.layers['rnn'].weight_hh_l0.grad
    """

    writer.add_histogram(f"{modelname}_grad_hist/{task_id}", grad_matrix.cpu().view(-1), epoch)


def plot_weights(writer, modelname, weight_matrix, task_id, epoch):
    """
    if modelname == 'esn':
        weight_matrix = model.weights['h2h'].data
    elif modelname == 'mlp':
        weight_matrix = model.layers['i2h'].weight.data
    elif modelname == 'lwta':
        weight_matrix = model.layers['i2h'].weight.data
    elif modelname == 'lmn':
        weight_matrix = model.layers['m2m'].weight.data
    elif modelname == 'rnn':
        weight_matrix = model.layers['rnn'].weight_hh_l0.data
    elif modelname == 'lstm':
        weight_matrix = model.layers['rnn'].weight_hh_l0.data
    """

    writer.add_image(f"{modelname}_W/{task_id}", weight_matrix.cpu(), epoch, dataformats='HW')
    writer.add_histogram(f"{modelname}_hist/{task_id}", weight_matrix.cpu().view(-1), epoch)                