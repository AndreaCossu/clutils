from torch.utils.data import random_split

def split_dataset(dataset, l1, l2, l3=None):
    split_list = [l1, l2, l3] \
        if l3 is not None \
        else [l1, l2]

    split_datasets = random_split(dataset, split_list)

    return split_datasets


class VCenterCrop(object):
    """Video Center Crop"""

    def __init__(self, sizes):
        self.sizes = sizes

    def __call__(self, vid):
        h, w = vid.shape[-2:]
        th, tw = self.sizes

        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))

        return vid[..., i:(i + th), j:(j + tw)]