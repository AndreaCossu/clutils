import torch
import os
import pickle
import numpy as np
from torchvision import transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100
from torch.utils.data import Dataset, DataLoader
from .ImageDataset import ImageDataset
from .utils import split_dataset


class _CLImageDataset(Dataset):
    '''
    General class providing useful methods to manage subsets of classes and permutations.
    Must be inherited by a subclass as second parent class together with the target PyTorch dataset class.
    '''

    def __init__(self, input_size=1, perc_val=0.2, train_batch_size=3, test_batch_size=0,
            max_label_value=None, sequential=False, image_size=(28,28), normalization=None):
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
        self.filter_classes = []

        self.perc_val = perc_val
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.max_label_value = max_label_value

        self.dataloaders = []

        self._change_permutation()
        
    def _change_permutation(self):

        if not self.sequential:
            perm = torch.randperm( int( (self.H*self.W) / self.input_size ) )
            self.permutations.append(perm)

    def save_permutation(self, filepath):
        torch.save(self.permutations[-1], os.path.join(filepath, f"permutation{len(self.permutations) - 1}.pt"))

    def _select_digits_subset(self, classes):
        if classes is not None:
            mask_trainval = torch.nonzero( torch.sum( torch.stack([ (self.trainval_targets_all == l) for l in classes ]), dim=0), as_tuple=False).squeeze()
            mask_test = torch.nonzero(torch.sum( torch.stack([ (self.mytest_targets_all == l) for l in classes ]), dim=0), as_tuple=False).squeeze()

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

        val_length = int(len(trainval_dataset) * self.perc_val)
        train_length = len(trainval_dataset) - val_length
        train_dataset, val_dataset = split_dataset(trainval_dataset, train_length, val_length)

        train_batch_size = int(len(self.trainval_data)*(1-self.perc_val)) if self.train_batch_size == 0 else self.train_batch_size
        val_batch_size = int(len(self.trainval_data)*self.perc_val) if self.test_batch_size == 0 else self.test_batch_size

        mnist_cl_loader_train = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, drop_last=True)
        mnist_cl_loader_val = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, drop_last=True)
    
        return mnist_cl_loader_train, mnist_cl_loader_val


    def get_task_loaders(self, classes=None, task_id=None, change_permutation=True):
        '''
        Select a subset of dataset with provided classes and eventually permute images.
        Returns dataloaders for training, validation and test set.

        :param classes: a list containing integer representing digits to select from dataset. If None select all images.
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

        self.filter_classes.append(classes)

        # select classes subset based on digit classes
        self._select_digits_subset(classes)
            
        # restrict output targets to max_label_value values
        if self.max_label_value is not None:
            self.trainval_targets = self.trainval_targets % self.max_label_value
            self.mytest_targets = self.mytest_targets % self.max_label_value

        train_loader, val_loader = self._get_train_val_loader(perm)
        test_loader = self._get_test_loader(perm)

        self.dataloaders.append( (train_loader, val_loader, test_loader) )

        return train_loader, val_loader, test_loader    


class CLMNIST(MNIST, _CLImageDataset):
    def __init__(self, root, download, input_size=1, perc_val=0.2, train_batch_size=3, test_batch_size=0,
            max_label_value=None, sequential=False, image_size=(28,28), normalization=None):
        '''
        :param max_label_value: number of output units of the model
        :param test_batch_size: 0 to use a full batch, otherwise > 0. It applies to both validation and test set.
        :param sequential: True to use pixel-wise images. False to use permuted pixel-wise images.
        :param normalization: if not None, normalize input data by dividing for `normalization`.
        '''
        
        super(CLMNIST, self).__init__(root, train=True, download=download, transform=transforms.ToTensor())
        _CLImageDataset.__init__(self, input_size=input_size, perc_val=perc_val, train_batch_size=train_batch_size, test_batch_size=test_batch_size, 
            max_label_value=max_label_value, sequential=sequential, image_size=image_size, normalization=normalization)

        self.trainval_data_all, self.trainval_targets_all = torch.load(os.path.join(self.processed_folder, self.training_file))
        self.mytest_data_all, self.mytest_targets_all = torch.load(os.path.join(self.processed_folder, self.test_file))


class CLFashionMNIST(FashionMNIST, _CLImageDataset):
    def __init__(self, root, download, input_size=1, perc_val=0.2, train_batch_size=3, test_batch_size=0, 
            max_label_value=None, sequential=False, image_size=(28,28), normalization=None):
        '''
        :param max_label_value: number of output units of the model
        :param test_batch_size: 0 to use a full batch, otherwise > 0. It applies to both validation and test set.
        :param sequential: True to use pixel-wise images. False to use permuted pixel-wise images.
        :param normalization: if not None, normalize input data by dividing for `normalization`.
        '''
        
        super(CLFashionMNIST, self).__init__(root, train=True, download=download, transform=transforms.ToTensor())
        _CLImageDataset.__init__(self, input_size=input_size, perc_val=perc_val, train_batch_size=train_batch_size, test_batch_size=test_batch_size, 
            max_label_value=max_label_value, sequential=sequential, image_size=image_size, normalization=normalization)

        self.trainval_data_all, self.trainval_targets_all = torch.load(os.path.join(self.processed_folder, self.training_file))
        self.mytest_data_all, self.mytest_targets_all = torch.load(os.path.join(self.processed_folder, self.test_file))


class CLCIFAR10(CIFAR10, _CLImageDataset):
    def __init__(self, root, download, input_size=1, perc_val=0.2, train_batch_size=3, test_batch_size=0, 
            max_label_value=None, sequential=False, image_size=(32,32), normalization=None):
        '''
        :param max_label_value: number of output units of the model
        '''
        
        super(CLCIFAR10, self).__init__(root, train=True, download=download, transform=transforms.ToTensor())
        _CLImageDataset.__init__(self, input_size=input_size, perc_val=perc_val, train_batch_size=train_batch_size, test_batch_size=test_batch_size, 
            max_label_value=max_label_value, sequential=sequential, image_size=image_size, normalization=normalization)

        self.trainval_data_all, self.trainval_targets_all = torch.tensor(self.data).long(), \
                                                            torch.tensor(self.targets).long()

        for file_name, checksum in self.test_list:
            file_path = os.path.join(root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.mytest_data_all.append(entry['data'])
                if 'labels' in entry:
                    self.mytest_targets_all.extend(entry['labels'])
                else:
                    self.mytest_targets_all.extend(entry['fine_labels'])
                self.mytest_targets_all = torch.tensor(self.mytest_targets_all).long()
        self.mytest_data_all= torch.tensor(np.vstack(self.data).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)).long()


class CLCIFAR100(CIFAR100, _CLImageDataset):
    def __init__(
        self, root, download, input_size=1, perc_val=0.2, train_batch_size=3, test_batch_size=0, 
        max_label_value=None, sequential=False, image_size=(32,32), normalization=None):
        '''
        :param max_label_value: number of output units of the model
        '''
        
        super(CLCIFAR100, self).__init__(root, train=True, download=download, transform=transforms.ToTensor())
        _CLImageDataset.__init__(
            self, input_size=input_size, perc_val=perc_val, train_batch_size=train_batch_size, 
            test_batch_size=test_batch_size, max_label_value=max_label_value, sequential=sequential, 
            image_size=image_size, normalization=normalization)

        self.trainval_data_all, self.trainval_targets_all = torch.tensor(self.data).long(), \
                                                            torch.tensor(self.targets).long()

        # FROM PYTORCH 1.5.0 code
        for file_name, checksum in self.test_list:
            file_path = os.path.join(root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.mytest_data_all.append(entry['data'])
                if 'labels' in entry:
                    self.mytest_targets_all.extend(entry['labels'])
                else:
                    self.mytest_targets_all.extend(entry['fine_labels'])
                self.mytest_targets_all = torch.tensor(self.mytest_targets_all).long()

        self.mytest_data_all= torch.tensor(np.vstack(self.data).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)).long()
