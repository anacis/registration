import os
import h5py
import numpy as np
from torch.utils.data import Dataset

def normalize(img):
    return img/np.percentile(img, 95)

class RegistrationDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_transform=None, crop=None):
        self.data_paths = os.listdir(data_dir)
        self.data_dir = data_dir
        self.transform = transform
        self.crop = crop
        self.target_transform = target_transform

    def __getitem__(self, index):
        #Select Image 1 and apply Normalization Transforms (fixed output)
        hf = h5py.File(os.path.join(self.data_dir, self.data_paths[index]), 'r')
        im = normalize(np.abs(np.flip(np.array(hf.get('target')))))
        im = np.expand_dims(im, axis=2)
        if self.transform:
            im = self.transform(im)

        imT = im

        #crop images (#TODO: Try applying transform then crop)
        if self.crop:
            im = self.crop(im)
            imT = self.crop(imT)

        #TODO: try simpler transformation (affine or translate)

        #Apply Smooth Transform to Image 2 (moving input)
        if self.target_transform:
            imT = self.target_transform(imT)



        return imT, im
    def __len__(self):
        return len(self.data_paths)
