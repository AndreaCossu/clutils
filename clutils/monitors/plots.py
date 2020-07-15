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

    

def plot_importance(writer, modelname, importance, task_id):
    for paramname, imp in importance:
        writer.add_histogram(f"{modelname}-{paramname}_importance/{task_id}", imp.cpu().view(-1))


def plot_gradients(writer, modelname, grad_matrix, task_id, epoch, grad_matrix_name='W'):
    writer.add_histogram(f"{modelname}-{grad_matrix_name}_grad_hist/{task_id}", grad_matrix.cpu().view(-1), epoch)


def plot_weights(writer, modelname, weight_matrix, task_id, epoch, weight_matrix_name='W'):

    writer.add_image(f"{modelname}-{weight_matrix_name}/{task_id}", weight_matrix.cpu(), epoch, dataformats='HW')
    writer.add_histogram(f"{modelname}-{weight_matrix_name}_hist/{task_id}", weight_matrix.cpu().view(-1), epoch)

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
    
    return weight_matrix, label