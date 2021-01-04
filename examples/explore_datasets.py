import numpy as np
import os
from matplotlib import  pyplot as plt

def plot_hist_sequence_length():
    ROOT = '/data/cossu/quickdraw'

    classes = [ el for el in os.listdir(ROOT) \
                if (not os.path.isdir(el)) and (not el.endswith('.full.npz'))]

    for cls in classes:
        print(f'Reading {cls}')
        data = np.load(os.path.join(ROOT, cls), encoding='latin1', allow_pickle=True)
        data = data['train']
        plt.figure()
        plt.hist([p.shape[0] for p in data])
        cls_f = os.path.splitext(os.path.basename(cls))[0]
        print(f'Saving {cls_f}')
        plt.title(cls_f)
        plt.savefig(os.path.join(ROOT, 'histograms', f'{cls_f}_lengths.png'))
        plt.close()

if __name__ == '__main__':
    plot_hist_sequence_length()
