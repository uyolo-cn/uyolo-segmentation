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

import math
import random
import torch
from torch import Tensor
import torchvision.transforms as transforms
import torchvision.transforms.functional as trans
from typing import Tuple, List, Union, Tuple

class Normalize:
    def __init__(self, mean, std) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return trans.normalize(img.float(), self.mean, self.std, inplace=True), mask

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


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.prob = p

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if random.random() < self.prob:
            img, mask = trans.hflip(img), trans.hflip(mask)
        return img, mask


class RandomVerticalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.prob = p

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if random.random() < self.prob:
            img, mask = trans.vflip(img), trans.vflip(mask)
        return img, mask


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
    def __init__(self, p=0.2, degrees=0, translate=[0, 0], scale=1.0, shear=[0, 0], mask_fill=255):
        self.p = p
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.mask_fill = mask_fill
        
    def __call__(self, img, mask):
        random_angle = random.random() * 2 * self.degrees - self.degrees
        if random.random() < self.p:
            img = trans.affine(img, random_angle, self.translate, self.scale, self.shear, trans.InterpolationMode.BILINEAR, 0)
            mask = trans.affine(mask, random_angle, self.translate, self.scale, self.shear, trans.InterpolationMode.NEAREST, self.mask_fill)
        return img, mask


class RandomRotation:
    def __init__(self, degrees: float = 10.0, p: float = 0.2, mask_fill: int = 255, expand: bool = False) -> None:
        """Rotate the image by a random angle between -angle and angle with probability p
        Args:
            p: probability
            degrees: rotation angle value in degrees, counter-clockwise.
            expand: Optional expansion flag. 
                    If true, expands the output image to make it large enough to hold the entire rotated image.
                    If false or omitted, make the output image the same size as the input image. 
                    Note that the expand flag assumes rotation around the center and no translation.
        """
        self.p = p
        self.degrees = degrees
        self.expand = expand
        self.mask_fill = mask_fill

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        random_angle = random.random() * 2 * self.degrees - self.degrees
        if random.random() < self.p:
            img = trans.rotate(img, random_angle, trans.InterpolationMode.BILINEAR, self.expand, fill=0)
            mask = trans.rotate(mask, random_angle, trans.InterpolationMode.NEAREST, self.expand, fill=self.mask_fill)
        return img, mask
    

class CenterCrop:
    def __init__(self, size: Union[int, List[int], Tuple[int]]) -> None:
        """Crops the image at the center
        Args:
            output_size: height and width of the crop box. If int, this size is used for both directions.
        """
        self.cc = transforms.CenterCrop(size)

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return self.cc(img), self.cc(mask)


class RandomCrop:
    def __init__(self, size: Union[int, List[int], Tuple[int]], p: float = 0.5, mask_fill: int = 255) -> None:
        """Randomly Crops the image.
        Args:
            output_size: height and width of the crop box. If int, this size is used for both directions.
        """
        self.size = (size, size) if isinstance(size, int) else size
        self.p = p
        self.mask_fill = mask_fill

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        H, W = img.shape[1:]
        tH, tW = self.size

        if random.random() < self.p:
            margin_h = max(H - tH, 0)
            margin_w = max(W - tW, 0)
            y1 = random.randrange(0, margin_h+1)
            x1 = random.randrange(0, margin_w+1)
            y2 = y1 + min(tH, H)
            x2 = x1 + min(tW, W)
            img = img[:, y1:y2, x1:x2]
            mask = mask[:, y1:y2, x1:x2]

            if H < tH or W < tW:
                img, mask = Pad(self.size, self.mask_fill)(img, mask)
        return img, mask

class Pad:
    def __init__(self, size: Union[List[int], Tuple[int], int], mask_fill: int = 255) -> None:
        """Pad the given image on all sides with the given "pad" value.
        Args:
            size: expected output image size (h, w)
            fill: Pixel fill value for constant fill. Default is 0. This value is only used when the padding mode is constant.
        """
        self.size = size
        self.mask_fill = mask_fill

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        padding = (0, 0, self.size[1]-img.shape[2], self.size[0]-img.shape[1])
        return trans.pad(img, padding), trans.pad(mask, padding, self.mask_fill)


class ResizePad:
    def __init__(self, size: Union[int, Tuple[int], List[int]], mask_fill: int = 255) -> None:
        """Resize the input image to the given size.
        Args:
            size: Desired output size (h, w). 
                If size is a sequence, the output size will be matched to this. 
                If size is an int, the smaller edge of the image will be matched to this number maintaining the aspect ratio.
        """
        self.size = size
        self.mask_fill = mask_fill

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        H, W = img.shape[1:]
        tH, tW = self.size

        # scale the image 
        scale_factor = min(tH / H, tW / W) if W > H else max(tH / H, tW / W)
        # nH, nW = int(H * scale_factor + 0.5), int(W * scale_factor + 0.5)
        nH, nW = round(H * scale_factor), round(W * scale_factor)
        img = trans.resize(img, (nH, nW), trans.InterpolationMode.BILINEAR)
        mask = trans.resize(mask, (nH, nW), trans.InterpolationMode.NEAREST)

        return Pad(self.size, self.mask_fill)(img, mask)

class Resize:
    def __init__(self, size: Union[int, Tuple[int], List[int]]) -> None:
        self.img_resize = transforms.Resize(size, transforms.InterpolationMode.BILINEAR)
        self.mask_resize = transforms.Resize(size, transforms.InterpolationMode.NEAREST)
    
    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return self.img_resize(img), self.mask_resize(mask)

class RandomResizedCrop:
    def __init__(self, size: Union[int, Tuple[int], List[int]], scale: Tuple[float] = (0.5, 1.0), ratio: Tuple[float] = (0.75, 4.0 / 3.0)) -> None:
        """
        Crop a random portion of image and resize it to a given size.
        If the image is torch Tensor, it is expected to have `[..., H, W]` shape, where `...` means an arbitrary number of leading dimensions
        A crop of the original image is made: the crop has a random area (h * w) and a random aspect ratio. 
        This crop is finally resized to the given size. 
        This is popularly used to train the Inception networks.
        """
        self.out_size = size
        self.scale = scale
        self.ratio = ratio
        
    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        H, W = img.shape[1:]
        s = random.random() * (self.scale[1] - self.scale[0]) + self.scale[0]
        r = random.random() * (self.ratio[1] - self.ratio[0]) + self.ratio[0]
        crop_h, crop_w = math.sqrt((s * H * W) / r), math.sqrt((s * H * W) * r)
        img, mask = RandomCrop((crop_h, crop_w), 1.0)(img, mask)
        return Resize(self.out_size)(img, mask)
    
class SimpleCopyPaste:
    def __init__(self, label_index: List[int], prob=0.5) -> None:
        self.label_index = label_index
        self.prob = prob

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        H, W = img.shape[1:]
        if random.random() < self.prob:
            for idx in self.label_index:
                valid = (mask == idx)
                if not valid.any():
                    continue
                for i in range(4):
                    cx1, cy1 = random.random() * 3 / 4 * H, random.random() * 3 / 4 * W
                    valid = valid[:, int(cx1):int(cx1 + H / 4.0), int(cy1):int(cy1 + W / 4.0)]
                    if not valid.any():
                        continue
                    copy_img = img[:, int(cx1):int(cx1 + H / 4.0), int(cy1):int(cy1 + W / 4.0)] * valid
                    copy_mask = mask[:, int(cx1):int(cx1 + H / 4.0), int(cy1):int(cy1 + W / 4.0)] * valid
                    px1, py1 = random.random() * 3 / 4 * H, random.random() * 3 / 4 * W
                    img[:, int(px1):int(px1 + H / 4.0), int(py1):int(py1 + W / 4.0)] *= (~valid)
                    img[:, int(px1):int(px1 + H / 4.0), int(py1):int(py1 + W / 4.0)] += copy_img
                    mask[:, int(px1):int(px1 + H / 4.0), int(py1):int(py1 + W / 4.0)] *= (~valid)
                    mask[:, int(px1):int(px1 + H / 4.0), int(py1):int(py1 + W / 4.0)] += copy_mask
        return img, mask


class Compose:
    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if mask.ndim == 2:
            assert img.shape[1:] == mask.shape
        else:
            assert img.shape[1:] == mask.shape[1:]

        for transform in self.transforms:
            img, mask = transform(img, mask)

        return img, mask

if __name__ == '__main__':
    h = 230
    w = 420
    img = torch.randn(3, h, w)
    mask = torch.randn(1, h, w)
    aug = Compose([
        # RandomResizedCrop((512, 512)),
        # RandomCrop((512, 512), 125),
        # Pad((512, 512)),
        # Resize((512, 512)),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    img, mask = aug(img, mask)
    print(img.shape, mask.shape)