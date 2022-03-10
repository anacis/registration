import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

def normalize(img, percent=95):
    return img/(np.percentile(img, percent) + 1e-12)

def z_norm(img):
    return (img - torch.mean(img))/torch.std(img)

def minmax_norm(img, x): 
    return (img - img.min())/(img.max()-img.min())

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
        im = np.abs(np.flip(np.array(hf.get('target'))))
        fixed = np.expand_dims(im, axis=2)

        if self.transform:
            fixed = self.transform(fixed)

        fixed_contrast = fixed

        if self.contrast_transform:
            fixed_contrast = self.contrast_transform(fixed_contrast)
        
        moving = fixed_contrast
        
        #Apply Smooth Transform to Image 2 (moving input)
        if self.target_transform:
            moving = self.target_transform(moving)

        #Normalize everything here
        
        #clip to 99th percentile
        fixed = torch.clamp(fixed, max = np.percentile(fixed, 99))
        fixed_contrast = torch.clamp(fixed_contrast, max = np.percentile(fixed_contrast, 99))
        moving = torch.clamp(moving, max = np.percentile(moving, 99))

        percent = np.random.randint(90, 100)
        fixed = normalize(fixed, percent)
        fixed_contrast = normalize(fixed_contrast, percent)
        moving = normalize(moving, percent)

        
        return moving, fixed, fixed_contrast
    def __len__(self):
        return len(self.data_paths)
