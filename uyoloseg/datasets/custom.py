# ==================================================================
# Copyright (c) 2023, uyolo.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the
# distribution.
# 3. All advertising materials mentioning features or use of this software
# must display the following acknowledgement:
# This product includes software developed by the uyolo Group. and
# its contributors.
# 4. Neither the name of the Group nor the names of its contributors may
# be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY UYOLO, GROUP AND CONTRIBUTORS
# ===================================================================
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io
import torchvision.transforms as transforms
from PIL import Image
from typing import Tuple
from .utils import file2list

class CustomDataset(Dataset):
    """Custom dataset for semantic segmentation. With a txt file to load image data and mask data.

    Txt contents example:
    /root/data/dir1/abcdefg.jpg, /root/data/dir1/abcdefg.png
    /root/data/dir2/abcdefg.jpg, /root/data/dir2/abcdefg.png
    /root/data/dir3/abcdefg.jpg, /root/data/dir3/abcdefg.png

    Args:
        txt_path (str): Path to txt file
        img_suffixes (list[str]): Suffixes of images. Default: ['.jpg']
        mask_suffixes (list[str]): Suffixes of masks. Default: ['.png']
        task (str): If task='test', gt wouldn't be loaded.
        ignore_index (int): The label index to be ignored. Default: 255
    """
    CLASSES = None
    PALETTE = None
    LABEL_MAP = None

    def __init__(self, txt_path='./train.txt', num_classes=2, img_suffixes=['.jpg'], mask_suffixes=['.png'], task='train', ignore_index=255, transform=None, logger=None) -> None:
        super(CustomDataset, self).__init__()
        self.num_classes = num_classes
        self.transform = transform
        self.task = task
        self.ignore_index = ignore_index
        self.files = file2list(txt_path)
        if logger:
            logger.info(f"Found {len(self.files)} {task} images.")
        else:
            print(f"Found {len(self.files)} {task} images.")

    def __len__(self) -> int:
        return len(self.files)


    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        try:
            image = io.read_image(self.files[index][0].strip())
        except RuntimeError as e:
            image = Image.open(self.files[index][0].strip())
            image = transforms.ToTensor()(image)
        
        try:
            label = io.read_image(self.files[index][1].strip())
        except RuntimeError as e:
            label = Image.open(self.files[index][1].strip())
            label = transforms.ToTensor()(label)

        assert label.shape[0] == 1, "The number of mask's channel must be 1!!!"
    
        if self.transform:
            image, label = self.transform(image, label)
        
        if self.LABEL_MAP:
            label = label.squeeze().numpy()
            label = torch.from_numpy(self.LABEL_MAP[label]).long()

        meta = {'img': image, 'mask': label}
        
        return meta
