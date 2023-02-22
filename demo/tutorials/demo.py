import torch
import torchvision.io as io

from uyoloseg.models import build_model
from uyoloseg.datasets import build_transforms


class Predictor:
    def __init__(self, cfg, model_path, device="cuda:0") -> None:
        self.cfg = cfg
        self.device = device
        model = build_model(cfg.model)
        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(ckpt["state_dict"])
        self.model = model.to(device).eval()
        self.transform = build_transforms(cfg.val_dataset.transforms)

    def inference(self, img):
        if isinstance(img ,str):
            img = io.read_image(img)
        
        mask = torch.zeros(img.shape)

        img, mask = self.transform(img, mask)

        meta = {
            'img': img.unsqueeze(0).to(self.device), 
            'mask': mask.unsqueeze(0).to(self.device)
        }

        with torch.no_grad():
            result = self.model(meta)
        
        return result