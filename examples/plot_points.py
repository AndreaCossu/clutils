import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from clutils.monitors.plots import average_accuracies


base_folder = os.path.expanduser('~/experiments/researchproject')
final_plotname = 'pointplot.png'

models = {
        'rnn': 1,
        'mlp': 3,
        'lstm': 5
    }
strategies = {
        'ewc': ['o', 'grey'], 
        'mas': ['s', 'green'],
        'slnid': ['D', 'blue']
    }

runs = 5


plt.figure()
for model, pos_x in models.items():
    for strategy, markers in strategies.items():

        filepaths = [ 
            os.path.join(base_folder, f"{model}_{strategy}_{i}/intermediate_results.csv")
            for i in range(1, runs+1) 
            ]

        mean, std = average_accuracies(filepaths, n_tasks=5)
        tasks_mean = mean[:-1].mean()
        tasks_std = std[:-1].std()

        plt.errorbar(pos_x, tasks_mean, yerr=tasks_std, marker=markers[0], c=markers[1], label=strategy)

plt.ylim(0,1.05)
plt.ylabel('Accuracy', fontsize=12)
plt.xlim(0,6.5)
plt.xticks(list(models.values()), list(models.keys()))
plt.grid(True)
plt.legend(loc='best', labels=list(strategies.keys()))
plt.savefig(os.path.join(base_folder, final_plotname))