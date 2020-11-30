from torch.utils.data import random_split, ConcatDataset


def split_dataset(dataset, l1, l2, l3=None):
    split_list = [int(l1), int(l2), int(l3)] \
        if l3 is not None \
        else [int(l1), int(l2)]

    split_datasets = random_split(dataset, split_list)

    return split_datasets


def merge_datasets(dataset_list):
    """
    List of PyTorch Dataset
    """

    return ConcatDataset(dataset_list)