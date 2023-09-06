from typing import Any, Optional, Tuple, Callable

import numpy as np

import torch
import torch.utils.data as td

from PIL import Image

class AnimalDataset(td.Dataset):
    
    def __init__(self, data: Any, resize: Optional[Tuple]=None, transformations: Optional[Callable]=None):

        self.data = data
        self.resize = resize
        self.transformations = transformations

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx: int):

        img_path = self.data.iloc[idx]["img_path"]
        img = Image.open(img_path)

        if self.resize is not None:
            img = img.resize((self.resize[1], self.resize[0]), resample=Image.Resampling.BILINEAR)
        
        img = np.array(img)

        if self.transformations is not None:
            aug_img = self.transformations(image=img)
            new_img = aug_img["image"]

        new_img = np.transpose(new_img, (2, 0, 1)).astype(np.float32)

        return {
            "image": torch.tensor(new_img, dtype=torch.float32)
        }
    