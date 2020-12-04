from torch.utils.data import random_split, ConcatDataset
from sklearn.model_selection import train_test_split

def split_dataset(dataset, l1, l2):
    split_list = [int(l1), int(l2)]
    split_datasets = random_split(dataset, split_list)
    return split_datasets

def merge_datasets(dataset_list):
    """
    List of PyTorch Dataset
    """

    return ConcatDataset(dataset_list)