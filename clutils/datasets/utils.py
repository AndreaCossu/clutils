from torch.utils.data import random_split, ConcatDataset
import os
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

def split_dataset(dataset, l1, l2):
    split_list = [int(l1), int(l2)]
    split_datasets = random_split(dataset, split_list)
    return split_datasets

def merge_datasets(dataset_list):
    """
    List of PyTorch Dataset
    """

    return ConcatDataset(dataset_list)

def compute_quickdraw_normalizer(root):
    classes = [ el for el in os.listdir(root) \
                if not os.path.isdir(el) and not el.endswith('.full.npz')
                and not el.endswith('.png')]

    normalizers = {}

    for cls in classes:
        data = np.load(os.path.join(root, cls), encoding='latin1', allow_pickle=True)
        data = data['train']
        deltas = []
        for i in range(len(data)):
            deltas += data[i][:, 0].tolist()
            deltas += data[i][:, 1].tolist()
        std = np.std(deltas)
        mu = np.mean(deltas)
        normalizers[os.path.splitext(os.path.basename(cls))[0]] = (mu, std)

    return normalizers

def collate_sequences(minibatch):
    """

    :param minibatch: a list of (x,y,length) where x (length, n_features), y is a scalar tensor, length is an integer
    :return:
    """
    x = [el[0] for el in minibatch]
    y = torch.stack([el[1] for el in minibatch], dim=0)
    l = [el[2] for el in minibatch]

    return x, y, l
