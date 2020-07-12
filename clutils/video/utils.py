import torch

def get_video_embeddings(model, x, grad=False):
    """
    :param x: (B, T, C, H, W)

    :return outs: (B, T, H)
    """
 
    with torch.set_grad_enabled(grad):
        outs = []
        for t in range(x.size(1)):
            out = model(x[:, t, :, :, :])
            outs.append(out)
            
        outs = torch.stack(outs).permute(1, 0, 2)

        return outs


def custom_collate(batch):
    """
    Pass it as `collate_fn` to a dataloader in order to discard additional info
    from the dataset. Keeping only video and label.
    Useful in UCF101.
    """

    filtered_batch = []
    for video, _, label in batch:
        filtered_batch.append((video, label))
    return torch.utils.data.dataloader.default_collate(filtered_batch)


class VCenterCrop(object):
    """
    Video Center Crop transform. Pass it to Dataset instance as transform.
    """

    def __init__(self, sizes):
        self.sizes = sizes

    def __call__(self, vid):
        h, w = vid.shape[-2:]
        th, tw = self.sizes

        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))

        return vid[..., i:(i + th), j:(j + tw)]   