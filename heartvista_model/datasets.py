import os
import numpy as np
import torch
from momentum.dataset import UFData

class RegistrationDataset(UFData):
    def __init__(self, data_dir, magnitude=False, device=torch.device('cpu'), fastmri=True,
                             norm=0.99, spatial_transform=None, contrast_augs=True):
        super().__init__(data_dir, magnitude=magnitude, device=device,
                              fastmri=fastmri, random_augmentation=contrast_augs)
        
        self.spatial_transform = spatial_transform

    def __getitem__(self, index):
        #Get 2 images of same source with different contrasts
        fixed, fixed_contrast = super().__getitem__(index)
        
        #Apply Spatial Transform to Image 2 (moving input)
        if self.target_transform:
            moving = self.target_transform(fixed_contrast)

        return moving, fixed, fixed_contrast
