import torch
import os
import pickle
import numpy as np
from torchvision import transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100
from .CLImageDataset import _CLImageDataset


class CLMNIST(MNIST, _CLImageDataset):
    def __init__(self, root, download, input_size=1, perc_val=0.2, train_batch_size=3, test_batch_size=0,
            output_size=None, sequential=False, image_size=(28,28), normalization=None):
        '''
        :param output_size: number of output units of the model
        :param test_batch_size: 0 to use a full batch, otherwise > 0. It applies to both validation and test set.
        :param sequential: True to use pixel-wise images. False to use permuted pixel-wise images.
        :param normalization: if not None, normalize input data by dividing for `normalization`.
        '''
        
        super(CLMNIST, self).__init__(root, train=True, download=download, transform=transforms.ToTensor())
        _CLImageDataset.__init__(self, input_size=input_size, perc_val=perc_val, train_batch_size=train_batch_size, test_batch_size=test_batch_size, 
            output_size=output_size, sequential=sequential, image_size=image_size, normalization=normalization)

        self.trainval_data_all, self.trainval_targets_all = torch.load(os.path.join(self.processed_folder, self.training_file))
        self.mytest_data_all, self.mytest_targets_all = torch.load(os.path.join(self.processed_folder, self.test_file))


class CLFashionMNIST(FashionMNIST, _CLImageDataset):
    def __init__(self, root, download, input_size=1, perc_val=0.2, train_batch_size=3, test_batch_size=0, 
            output_size=None, sequential=False, image_size=(28,28), normalization=None):
        '''
        :param output_size: number of output units of the model
        :param test_batch_size: 0 to use a full batch, otherwise > 0. It applies to both validation and test set.
        :param sequential: True to use pixel-wise images. False to use permuted pixel-wise images.
        :param normalization: if not None, normalize input data by dividing for `normalization`.
        '''
        
        super(CLFashionMNIST, self).__init__(root, train=True, download=download, transform=transforms.ToTensor())
        _CLImageDataset.__init__(self, input_size=input_size, perc_val=perc_val, train_batch_size=train_batch_size, test_batch_size=test_batch_size, 
            output_size=output_size, sequential=sequential, image_size=image_size, normalization=normalization)

        self.trainval_data_all, self.trainval_targets_all = torch.load(os.path.join(self.processed_folder, self.training_file))
        self.mytest_data_all, self.mytest_targets_all = torch.load(os.path.join(self.processed_folder, self.test_file))


class CLCIFAR10(CIFAR10, _CLImageDataset):
    def __init__(self, root, download, input_size=1, perc_val=0.2, train_batch_size=3, test_batch_size=0, 
            output_size=None, sequential=False, image_size=(32,32), normalization=None):
        '''
        :param output_size: number of output units of the model
        '''
        
        super(CLCIFAR10, self).__init__(root, train=True, download=download, transform=transforms.ToTensor())
        _CLImageDataset.__init__(self, input_size=input_size, perc_val=perc_val, train_batch_size=train_batch_size, test_batch_size=test_batch_size, 
            output_size=output_size, sequential=sequential, image_size=image_size, normalization=normalization)

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
        output_size=None, sequential=False, image_size=(32,32), normalization=None):
        '''
        :param output_size: number of output units of the model
        '''
        
        super(CLCIFAR100, self).__init__(root, train=True, download=download, transform=transforms.ToTensor())
        _CLImageDataset.__init__(
            self, input_size=input_size, perc_val=perc_val, train_batch_size=train_batch_size, 
            test_batch_size=test_batch_size, output_size=output_size, sequential=sequential, 
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
