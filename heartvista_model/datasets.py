import os
import h5py
import numpy as np
from torch.utils.data import Dataset

def normalize(img, percent=95):
    return img/np.percentile(img, percent)

class RegistrationDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_transform=None, contrast_transform=None):
        self.data_paths = os.listdir(data_dir)
        self.data_dir = data_dir
        self.transform = transform
        self.contrast_transform = contrast_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        #Select Image 1 and apply Normalization Transforms (fixed output)
        hf = h5py.File(os.path.join(self.data_dir, self.data_paths[index]), 'r')
        im = normalize(np.abs(np.flip(np.array(hf.get('target')))))
        fixed = np.expand_dims(im, axis=2)
        if self.transform:
            fixed = self.transform(fixed)

        fixed_contrast = fixed

        percent = np.random.randint(90, 100)
        fixed_contrast = normalize(fixed_contrast, percent=percent)

        if self.contrast_transform:
            fixed_contrast = self.contrast_transform(fixed_contrast)
        
        moving = fixed_contrast
        
        #Apply Smooth Transform to Image 2 (moving input)
        if self.target_transform:
            moving = self.target_transform(moving)

        return moving, fixed, fixed_contrast
    def __len__(self):
        return len(self.data_paths)
