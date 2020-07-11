import torch
import os
from torchaudio import transforms
from torch.utils.data import TensorDataset, DataLoader
from .utils import split_dataset


class CLSpeechWords():
    def __init__(self, root, train_batch_size, test_batch_size, n_mels=40, perc_val=0.2):

        self.root = root
        self.perc_val = perc_val
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

        self.sample_rate = 16000

        win_length = int(self.sample_rate / 1000 * 25)
        hop_length = int(self.sample_rate / 1000 * 10)
        self.mel_spectr = transforms.MelSpectrogram(sample_rate=self.sample_rate, 
            win_length=win_length, hop_length=hop_length, n_mels=n_mels)

        self.dataloaders = []

        self.current_class_id = 0
            

    def _load_data(self, classes):
        features = []
        targets = []
        for classname in classes:
            feature = torch.load(os.path.join(self.root, 'pickled', f"{classname}.pt"))
            features.append( self.preprocess_wav(feature) )
            targets.append(torch.ones(features[-1].size(0)).long() * self.current_class_id)
            self.current_class_id += 1
            
        features = torch.cat(features) # (B, L, n_mel)
        targets = torch.cat(targets)

        return TensorDataset(features, targets)


    def preprocess_wav(self, wav):
        return self.mel_spectr(wav).permute(0, 2, 1)


    def get_task_loaders(self, classes=None, task_id=None):

        if classes is not None:
            dataset = self._load_data(classes)

            len_train = int(len(dataset) - len(dataset) * self.perc_val * 2)
            len_val = int(len(dataset) * self.perc_val)
            len_test = len(dataset) - len_train - len_val

            train_d, val_d, test_d = split_dataset(dataset, len_train, len_val, len_test)

            train_batch_size = len(train_d) if self.train_batch_size == 0 else self.train_batch_size
            val_batch_size = len(val_d) if self.test_batch_size == 0 else self.test_batch_size
            test_batch_size = len(test_d) if self.test_batch_size == 0 else self.test_batch_size

            train_loader = DataLoader(train_d, batch_size=train_batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_d, batch_size=val_batch_size, shuffle=False, drop_last=True)
            test_loader = DataLoader(test_d, batch_size=test_batch_size, shuffle=False, drop_last=True)

            self.dataloaders.append( [train_loader, val_loader, test_loader] )

        elif task_id is not None:
            train_loader, val_loader, test_loader = self.dataloaders[task_id]


        return train_loader, val_loader, test_loader