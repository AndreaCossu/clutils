from torch.utils.data import random_split

def split_dataset(dataset, l1, l2, l3=None):
    split_list = [l1, l2, l3] \
        if l3 is not None \
        else [l1, l2]

    split_datasets = random_split(dataset, split_list)

    return split_datasets
