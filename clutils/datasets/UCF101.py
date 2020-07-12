import torch
from torchvision.transforms import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from .utils import split_dataset
from ..video.utils import VCenterCrop


class CLUCF101():

    def __init__(self, root, annotation_paths, frames_per_clip, train_batch_size,
                test_batch_size=0, step_between_clips=1, collate_fn=None):
        
        self.root = root
        self.annotation_paths = annotation_paths
        self.frames_per_clip = frames_per_clip
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.step_between_clips = step_between_clips
        self.collate_fn = collate_fn


        mean = torch.tensor([0.485, 0.456, 0.406]).float()
        std = torch.tensor([0.229, 0.224, 0.225]).float()
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: x / 255.), # scale in [0, 1]
            transforms.Lambda(lambda x: x.sub_(mean).div_(std)), # z-normalization
            transforms.Lambda(lambda x: x.permute(0, 3, 1, 2) ), # reshape into (T, C, H, W)
            VCenterCrop((224, 224))
        ])

        self.datasets = []

    def get_task_loaders(self, task_id):

        annotation_path = self.annotation_paths[task_id]

        if len(self.datasets) < task_id + 1:
            td = datasets.UCF101(root=self.root, annotation_path=annotation_path,
                    frames_per_clip=self.frames_per_clip, step_between_clips=self.step_between_clips,
                    fold=1, train=True, transform=self.transform)
                
            train_batch_size = len(td) if self.train_batch_size == 0 else self.train_batch_size
            train_dataset = DataLoader(td, batch_size=train_batch_size, 
                shuffle=True, drop_last=True, collate_fn=self.collate_fn)

            tsdall = datasets.UCF101(root=self.root, annotation_path=annotation_path,
                    frames_per_clip=self.frames_per_clip, step_between_clips=self.step_between_clips,
                    fold=1, train=False, transform=self.transform)

            val_length = int(len(tsdall) * 0.5)
            test_length = len(tsdall) - val_length
            vd, tsd = split_dataset(tsdall, val_length, test_length)

            val_batch_size = len(vd) if self.test_batch_size == 0 else self.test_batch_size
            validation_dataset = DataLoader(vd, batch_size=val_batch_size, 
                shuffle=False, drop_last=True, collate_fn=self.collate_fn)

            test_batch_size = len(tsd) if self.test_batch_size == 0 else self.test_batch_size
            test_dataset = DataLoader(tsd, batch_size=test_batch_size, 
                shuffle=False, drop_last=True, collate_fn=self.collate_fn)

            self.datasets.append( (train_dataset, validation_dataset, test_dataset) )

            print(f"Train, validation and test length for task {task_id}:")
            print(len(train_dataset))
            print(len(validation_dataset))
            print(len(test_dataset))


        return self.datasets[task_id]
