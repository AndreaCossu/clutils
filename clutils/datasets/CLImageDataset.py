import torch
import os
from torch.utils.data import Dataset, DataLoader
from .ImageDataset import ImageDataset
from .utils import split_dataset


class _CLImageDataset(Dataset):
    '''
    General class providing useful methods to manage subsets of classes and permutations.
    Must be inherited by a subclass as second parent class together with the target PyTorch dataset class.
    '''

    def __init__(self, input_size=1, perc_val=0.2, train_batch_size=3, test_batch_size=0,
            output_size=None, sequential=False, image_size=(28,28), normalization=None):
        '''
        :param perc_val: percentage of dataset used for validation (the same for test)
        :param train_batch_size, test_batch_size: 0 to use a full batch, otherwise > 0.
        :param sequential: True to use pixel-wise images. False to use permuted pixel-wise images.
        :param normalization: if not None, normalize input data by dividing for `normalization`.
        '''

        # to be instantiated from children classes
        self.trainval_data_all, self.trainval_targets_all = [], []
        self.mytest_data_all, self.mytest_targets_all = [], []

        self.input_size = input_size
        self.H, self.W = image_size[0], image_size[1]
        assert( (self.H*self.W) % self.input_size == 0)

        self.sequential = sequential
        self.normalization = normalization
        self.permutations = []
        self.filter_labels = []

        self.perc_val = perc_val
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.output_size = output_size

        self.dataloaders = []

    def _change_permutation(self):

        if not self.sequential:
            perm = torch.randperm( int( (self.H*self.W) / self.input_size ) )
            self.permutations.append(perm)

    def save_permutation(self, filepath):
        torch.save(self.permutations[-1], os.path.join(filepath, f"permutation{len(self.permutations) - 1}.pt"))

    def _select_digits_subset(self, labels):
        if labels is not None:
            mask_trainval = torch.sum( torch.stack([ (self.trainval_targets_all == l) for l in labels ]), dim=0).nonzero().squeeze()
            mask_test = torch.sum( torch.stack([ (self.mytest_targets_all == l) for l in labels ]), dim=0).nonzero().squeeze()

            self.trainval_targets = self.trainval_targets_all[mask_trainval]
            self.trainval_data = self.trainval_data_all[mask_trainval]
            self.mytest_targets = self.mytest_targets_all[mask_test]
            self.mytest_data = self.mytest_data_all[mask_test]
        else:
            self.trainval_targets = self.trainval_targets_all
            self.trainval_data = self.trainval_data_all
            self.mytest_targets = self.mytest_targets_all
            self.mytest_data = self.mytest_data_all


    def _get_test_loader(self, perm=None):
        '''
        :return mnist_cl_loader_test: mini or full batch loader for test set, depending on batch_size_test
        '''

        test_dataset = ImageDataset(self.mytest_data, self.mytest_targets, perm, self.input_size, self.normalization)
        if self.test_batch_size == 0:
            return DataLoader(test_dataset, batch_size=len(self.mytest_targets), shuffle=False)
        else:
            return DataLoader(test_dataset, batch_size=self.test_batch_size, shuffle=False, drop_last=True)


    def _get_train_val_loader(self, perm=None):
        '''
        :return mnist_cl_loader_train: mini-batch loader for training set
        :return mnist_cl_loader_val: mini or full batch loader for validation set, depending on batch_size_test
        '''

        trainval_dataset = ImageDataset(self.trainval_data, self.trainval_targets, perm, self.input_size, self.normalization)

        val_length = len(trainval_dataset) * self.perc_val
        train_length = len(trainval_dataset) - val_length
        train_dataset, val_dataset = split_dataset(trainval_dataset, train_length, val_length)

        train_batch_size = int(len(self.trainval_data)*(1-self.perc_val)) if self.train_batch_size == 0 else self.train_batch_size
        val_batch_size = int(len(self.trainval_data)*self.perc_val) if self.test_batch_size == 0 else self.test_batch_size

        mnist_cl_loader_train = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, drop_last=True)
        mnist_cl_loader_val = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, drop_last=True)
    
        return mnist_cl_loader_train, mnist_cl_loader_val


    def get_new_task_loaders(self, labels=None, task_id=None, change_permutation=True):
        '''
        Select a subset of dataset with provided labels and eventually permute images.
        Returns dataloaders for training, validation and test set.

        :param labels: a list containing integer representing digits to select from dataset. If None select all images.
        :param task_id: get dataloaders from a previous task id
        '''

        if task_id is not None:
            return self.dataloaders[task_id]

        # use new permutation for next task
        if not self.sequential:
            if change_permutation:
                self._change_permutation()
            perm = self.permutations[-1]
        else:
            perm = None

        self.filter_labels.append(labels)

        # select classes subset based on digit labels
        self._select_digits_subset(labels)
            
        # restrict output targets to output_size values
        if self.output_size is not None:
            self.trainval_targets = self.trainval_targets % self.output_size
            self.mytest_targets = self.mytest_targets % self.output_size

        train_loader, val_loader = self._get_train_val_loader(perm)
        test_loader = self._get_test_loader(perm)

        self.dataloaders.append( (train_loader, val_loader, test_loader) )

        return train_loader, val_loader, test_loader    