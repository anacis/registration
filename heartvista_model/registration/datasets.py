import os
import numpy as np
import torch
from momentum.dataset import UFData

class RegistrationDataset(UFData):
    def __init__(self, data_dir, magnitude=False, device=torch.device('cpu'), fastmri=True,
                             norm=0.99, spatial_transform=None, contrast_augs=True, augment_probability=0.9):
        super().__init__(data_dir, magnitude=magnitude, device=device,
                              fastmri=fastmri, random_augmentation=contrast_augs, augment_probability=augment_probability, crop=False)
        
        self.spatial_transform = spatial_transform

    def __getitem__(self, index):
        #Get 2 images of same source with different contrasts
        fixed, fixed_contrast = super().__getitem__(index)
        
        #Apply Spatial Transform to Image 2 (moving input)
        if self.spatial_transform:
            moving = self.spatial_transform(fixed_contrast)

        #crop the images
        fixed, offset = super().random_crop(fixed)
        fixed_contrast, _ = super().random_crop(fixed_contrast, offset=offset)
        moving, _ = super().random_crop(moving, offset=offset)

        return moving, fixed, fixed_contrast
