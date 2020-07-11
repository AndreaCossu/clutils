import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')
import pandas as pd
import os
from matplotlib.ticker import MaxNLocator


def save_learning_curves(models, result_folder, additional_metrics=['acc'], title=True, filename='training_results.csv'):
    '''
    :param models: list of modelnames to be used for plots.
    '''


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