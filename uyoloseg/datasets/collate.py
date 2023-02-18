from typing import Sequence

import torch
import torch.nn.functional as F

def collate_func(batch):
    """Only collate dict value in to a list. E.g. meta data dict and img_info
    dict will be collated."""

    elem = batch[0]
    if isinstance(elem, dict):
        return {key: collate_func([d[key] for d in batch]) for key in elem}
    else:
        return batch

def stack_batch_img(img_tensors: Sequence[torch.Tensor], divisible: int = 0, pad_value: float = 0.0) -> torch.Tensor:
    """
    Args:
        img_tensors (Sequence[torch.Tensor]):
        divisible (int):
        pad_value (float): value to pad
    Returns:
        torch.Tensor.
    """
    assert len(img_tensors) > 0
    assert isinstance(img_tensors, (tuple, list))
    assert divisible >= 0
    img_heights = []
    img_widths = []
    for img in img_tensors:
        assert img.shape[:-2] == img_tensors[0].shape[:-2]
        img_heights.append(img.shape[-2])
        img_widths.append(img.shape[-1])
    max_h, max_w = max(img_heights), max(img_widths)
    if divisible > 0:
        max_h = (max_h + divisible - 1) // divisible * divisible
        max_w = (max_w + divisible - 1) // divisible * divisible

    batch_imgs = []
    for img in img_tensors:
        padding_size = [0, max_w - img.shape[-1], 0, max_h - img.shape[-2]]
        batch_imgs.append(F.pad(img, padding_size, value=pad_value))
    return torch.stack(batch_imgs, dim=0).contiguous()

def naive_collate(batch):
    d = collate_func(batch)
    return {key: stack_batch_img(d[key]) for key in d.keys()}