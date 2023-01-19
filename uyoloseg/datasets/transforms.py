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
from torch import Tensor
import torchvision.transforms as transforms
import torchvision.transforms.functional as trans
from typing import Tuple, List, Union, Tuple, Optional
import random

class Normalize:
    def __init__(self, mean, std) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return trans.normalize(img.float(), self.mean, self.std), mask

class ColorJitter(transforms.ColorJitter):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0) -> None:
        super().__init__(brightness, contrast, saturation, hue)
    
    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return super().__call__(img), mask

class AdjustGamma:
    def __init__(self, gamma: float, gain: float = 1) -> None:
        """
        Args:
            gamma: Non-negative real number. gamma larger than 1 make the shadows darker, while gamma smaller than 1 make dark regions lighter.
            gain: constant multiplier
        """
        self.gamma = gamma
        self.gain = gain

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return trans.adjust_gamma(img, self.gamma, self.gain), mask

class RandomAdjustSharpness(transforms.RandomAdjustSharpness):
    def __init__(self, sharpness_factor: float, p: float = 0.5) -> None:
        super().__init__(sharpness_factor, p)

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return super().__call__(img), mask


class RandomAutocontrast(transforms.RandomAutocontrast):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__(p)

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return super().__call__(img), mask


class RandomGaussianBlur(transforms.GaussianBlur):
    def __init__(self, kernel_size: Union[int, List[int], Tuple[int]] = 3, sigma: Union[float, List[float], Tuple[float]] = (0.1, 2)) -> None:
        super().__init__(kernel_size, sigma)

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return super().__call__(img), mask


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__(p)

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return super().__call__(img), mask


class RandomVerticalFlip(transforms.RandomVerticalFlip):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__(p)

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return super().__call__(img), mask


class RandomGrayscale(transforms.RandomGrayscale):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__(p)

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return super().__call__(img), mask


class Equalize(transforms.RandomEqualize):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__(p)

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return super().__call__(img), mask


class Posterize(transforms.RandomPosterize):
    def __init__(self, bits: int = 2, p: float = 0.5) -> None:
        super().__init__(bits, p)

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return super().__call__(img), mask


class RandomAffine:
    def __init__(self, degrees=0, translate=[0, 0], scale=1.0, shear=[0, 0], seg_fill=0):
        self.img_affine = transforms.RandomAffine(degrees, translate, scale, shear, transforms.InterpolationMode.BILINEAR, 0)
        self.mask_affine = transforms.RandomAffine(degrees, translate, scale, shear, transforms.InterpolationMode.NEAREST, seg_fill)
        
    def __call__(self, img, mask):
        return self.img_affine(img), self.mask_affine(mask)


class RandomRotation:
    def __init__(self, degrees: float = 10.0, expand: bool = False, seg_fill: int = 0) -> None:
        self.img_rotate = transforms.RandomRotation(degrees, transforms.InterpolationMode.BILINEAR, expand)
        self.mask_rotate = transforms.RandomRotation(degrees, transforms.InterpolationMode.NEAREST, expand, fill=seg_fill)

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return self.img_rotate(img), self.mask_rotate(mask)
    

class CenterCrop:
    def __init__(self, size: Union[int, List[int], Tuple[int]]) -> None:
        """Crops the image at the center
        Args:
            output_size: height and width of the crop box. If int, this size is used for both directions.
        """
        self.size = (size, size) if isinstance(size, int) else size

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return trans.center_crop(img, self.size), trans.center_crop(mask, self.size)


class RandomCrop:
    def __init__(self, size: Union[int, List[int], Tuple[int]], p: float = 0.5) -> None:
        """Randomly Crops the image.
        Args:
            output_size: height and width of the crop box. If int, this size is used for both directions.
        """
        self.size = (size, size) if isinstance(size, int) else size
        self.p = p

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        H, W = img.shape[1:]
        tH, tW = self.size

        if random.random() < self.p:
            margin_h = max(H - tH, 0)
            margin_w = max(W - tW, 0)
            y1 = random.randint(0, margin_h+1)
            x1 = random.randint(0, margin_w+1)
            y2 = y1 + tH
            x2 = x1 + tW
            img = img[:, y1:y2, x1:x2]
            mask = mask[:, y1:y2, x1:x2]
        return img, mask


class Pad:
    def __init__(self, size: Union[List[int], Tuple[int], int], seg_fill: int = 0) -> None:
        """Pad the given image on all sides with the given "pad" value.
        Args:
            size: expected output image size (h, w)
            fill: Pixel fill value for constant fill. Default is 0. This value is only used when the padding mode is constant.
        """
        self.size = size
        self.seg_fill = seg_fill

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        padding = (0, 0, self.size[1]-img.shape[2], self.size[0]-img.shape[1])
        return trans.pad(img, padding), trans.pad(mask, padding, self.seg_fill)


class ResizePad:
    def __init__(self, size: Union[int, Tuple[int], List[int]], seg_fill: int = 0) -> None:
        """Resize the input image to the given size.
        Args:
            size: Desired output size (h, w). 
                If size is a sequence, the output size will be matched to this. 
                If size is an int, the smaller edge of the image will be matched to this number maintaining the aspect ratio.
        """
        self.size = size
        self.seg_fill = seg_fill

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        H, W = img.shape[1:]
        tH, tW = self.size

        # scale the image 
        scale_factor = min(tH/H, tW/W) if W > H else max(tH/H, tW/W)
        # nH, nW = int(H * scale_factor + 0.5), int(W * scale_factor + 0.5)
        nH, nW = round(H*scale_factor), round(W*scale_factor)
        img = trans.resize(img, (nH, nW), trans.InterpolationMode.BILINEAR)
        mask = trans.resize(mask, (nH, nW), trans.InterpolationMode.NEAREST)

        return Pad(self.size, self.seg_fill)(img, mask)