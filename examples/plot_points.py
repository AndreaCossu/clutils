import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from clutils.monitors.plots import average_accuracies


base_folder = os.path.expanduser('~/experiments/researchproject')
final_plotname = 'pointplot.png'

models = { # models : label position on x axis
        'mlp': 1,
        'rnn': 3,
        'lstm': 5
    }
strategies = { # strategy name: [marker, color]
        'ewc': ['o', 'grey'], 
        'mas': ['s', 'green'],
        'slnid': ['D', 'blue']
    }

runs = 5


fig, ax = plt.subplots()
for model, pos_x in models.items():
    for strategy, markers in strategies.items():

        filepaths = [ 
            os.path.join(base_folder, f"{model}_{strategy}_{i}/intermediate_results.csv")
            for i in range(1, runs+1) 
            ]

        mean, std = average_accuracies(filepaths, n_tasks=5)
        tasks_mean = mean[:-1].mean()
        tasks_std = std[:-1].std()

        ax.errorbar(pos_x, tasks_mean, yerr=tasks_std, marker=markers[0], c=markers[1], label=strategy)

ax.set_ylim(0,1.05)
ax.axhline(0.1, linestyle='--', linewidth=2.0)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_xlim(0,6.5)
ax.set_xticks(list(models.values()))
ax.set_xticklabels(list(models.keys()))
ax.grid(True)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys())
fig.savefig(os.path.join(base_folder, final_plotname))
plt.close(fig=fig)