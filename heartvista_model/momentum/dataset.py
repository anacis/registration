import random
from glob import glob

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional

def normalize(img, percent=95):
    return img/(np.percentile(img, percent) + 1e-12)

class UFData(Dataset):

    def __init__(self, data_directory, max_offset=None, magnitude=False, device=torch.device('cpu'),
                 fastmri=False, random_augmentation=True):
        """

        Parameters
        ----------
        data_directory : str
            The directory containing the training npy data.
        max_offset : tuple
            The maximum offset to crop an image to.
        magnitude : bool
            If True, train using magnitude image as input. Otherwise, use real and imaginary image in separate channels.
        device : torch.device
            The device to load the data to.
        """
        if max_offset is None:
            if fastmri:
                max_offset = (434, 50)  # FastMRI dataset to make same size of 3D dataset slices
                # max_offset = (200, 40)  # FastMRI dataset (biggest is around 386 and smallest is 320)
            else:
                max_offset = (50, 50)  # 3D Dataset
                # max_offset = (216, 280)  # 40 by 40 patch in original dataset of 256x320

        self.image_paths = glob(f"{data_directory}/*.npy")
        self.h5_format = False
        if not self.image_paths:
            self.image_paths = glob(f"{data_directory}/*.h5")
            self.h5_format = True

        print(f"Using data from: {data_directory}\nFound {len(self.image_paths)} image paths.")
        self.device = device
        self.magnitude = magnitude
        self.random_augmentation = random_augmentation

        if fastmri:  # Fast MRI Dataset:
            self.cropped_image_size = np.array([640, 320]) - max_offset
        else:  # Original mri.org Dataset:
            self.cropped_image_size = np.array(np.load(self.image_paths[0]).shape[-2:]) - max_offset
        # self.cropped_image_size = np.array([47, 47])  # Just the image size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        """Get image at the specified index. Two cropped versions of the images will be returned: one with the previous
        cropping offset (and possibly other augmentation settings), and a new one with a new cropping offset. The new
        cropping offset will be stored for the next time this image is accessed.

        Parameters
        ----------
        index : int
            The image index.

        Returns
        -------
        previous_image : torch.Tensor
            Image cropped (augmented) at previous settings.
        new_image : torch.Tensor
            Image cropped (augmented) at new settings.
        """
        if self.h5_format:
            original_image = np.array(h5py.File(self.image_paths[index])["target"])[None]
        else:
            original_image = np.load(self.image_paths[index])[None]
        image = torch.from_numpy(original_image)
        
        if self.random_augmentation:
            image = self.augment_image(image, augment_probability=0.9, jitter_probability=0.8, noise_probability=0.8,
                      blur_probability=0.8, aliasing_probability=0)  # Model will be sensitive to this
            
            image2 = image
            jitter_probability=0.8
            noise_probability=0
            blur_probability=0.8
            aliasing_probability=0
            if random.random() < jitter_probability:
                image2 = self.random_jitter(image, max_brightness=0.8, max_contrast=0.8, max_saturation=0.8, max_hue=0.4)
            if random.random() < blur_probability:
                image2 = self.random_blur(image, max_sigma=2.0, kernel_size=5)
            if random.random() < noise_probability:  # Noise and blur or blur and noise?
                image2 = self.random_noise(image)
            if random.random() < aliasing_probability:
                image2 = self.random_aliasing(image)

            # Different random changes to each if we want to be insensitive to it
            if self.magnitude:
                image1 = torch.abs(image)  #TODO: check we have one channel
                image2 = torch.abs(image2)
            else:
                image1 = self.complex2channels(self.random_phase(image))
                image2 = self.complex2channels(self.random_phase(image2))

            
            top_clip1 = random.uniform(90, 100)
            top_clip2 = random.uniform(90, 100)
            image1 = torch.clamp(image1, max = np.percentile(image1, top_clip1))
            image2 = torch.clamp(image2, max = np.percentile(image2, top_clip2))

            bottom_clip1 = random.uniform(0, 5)
            bottom_clip2 = random.uniform(0, 5)
            image1 = torch.clamp(image1, min = np.percentile(image1, bottom_clip1))
            image2 = torch.clamp(image2, min = np.percentile(image2, bottom_clip2))

            percent = random.uniform(90, 100)
            image1 = normalize(image1, percent)
            image2 = normalize(image2, percent)
            
            return image1, image2

        else:
            if self.magnitude:
                image = torch.abs(image)
            else:
                image = self.complex2channels(image)
            return image

    def augment_image(self, image, augment_probability=0.9, jitter_probability=0.8, noise_probability=0.8,
                      blur_probability=0.8, aliasing_probability=0.8):
        # Anything we want to be sensitive to
        # Either random noise, Random blur

        # image = self.random_rotate(image)  # TODO: maybe?
        image = self.random_crop(image)


        # TODO
        #  Could also do some spiral stuff? with NUFFT
        #  Could also do off resonance? Is all you need.
        # TODO: load data from not only fastmri, but also undersampled recons with PICS

        return image

    @staticmethod
    def random_phase(image):
        """Add random phase to image"""
        return image * torch.exp(1j * 2 * np.pi * torch.rand(1))

    @staticmethod
    def complex2channels(image, dim=0):
        """Convert single from complex to channels"""
        return torch.cat((image.real, image.imag), dim=dim).float()

    def random_crop(self, image, offset=None):
        """Crop image(s) to `self.cropped_image_size` starting at the specified offset.

        Parameters
        ----------
        image : torch.Tensor
            The image to crop of shape (C, H, W)
        offset : np.array
            The offset to add of shape (2,)

        Returns
        -------
        torch.Tensor
            The cropped image of size `self.cropped_image_size`. Shape (C, H, W).
        """
        if offset is None:
            # +1 since random doesn't include max. i.e [a, b).
            offset = np.random.randint(0, image.shape[1:] - self.cropped_image_size + 1)
        stop = offset + self.cropped_image_size
        return image[:, offset[0]:stop[0], offset[1]:stop[1]]

    @staticmethod
    def random_jitter(image, max_brightness=0.1, max_contrast=0.1, max_saturation=0.2, max_hue=0.1, max_gamma=0.1):
        """
        TODO: adjust_contrast doesnt support grayscale (neither does adjust_saturation)

        Parameters
        ----------
        image
        max_brightness
        max_hue
        max_gamma

        Returns
        -------


        """
        real, imaginary = image.real, image.imag

        brightness_param = random.uniform(1 - max_brightness, 1 + max_brightness)
        real = functional.adjust_brightness(real, brightness_param)
        imaginary = functional.adjust_brightness(imaginary, brightness_param)

        contrast_param = random.uniform(1 - max_contrast, 1 + max_contrast)
        real = functional.adjust_contrast(real, contrast_param)
        imaginary = functional.adjust_contrast(imaginary, contrast_param)
        
        saturation_param = random.uniform(1 - max_saturation, 1 + max_saturation)
        real = functional.adjust_saturation(real, saturation_param)
        imaginary = functional.adjust_saturation(imaginary, saturation_param)
   
        hue_param = random.uniform(- max_hue, max_hue)
        real = functional.adjust_hue(real, hue_param)
        imaginary = functional.adjust_hue(imaginary, hue_param)

        gamma_param = random.uniform(1 - max_gamma, 1 + max_gamma)
        real = functional.adjust_gamma(real, gamma_param)
        imaginary = functional.adjust_gamma(imaginary, gamma_param)
        

        return real + 1j * imaginary

    @staticmethod
    def random_noise(image, max_std=0.2):
        std = torch.rand(1) * max_std  # Maybe make this also (abs of) gaussian
        return image * (1 - std) + torch.randn(image.size(), dtype=image.dtype) * std

    @staticmethod
    def random_blur(image, max_sigma=3, kernel_size=19):
        blur_param = random.random() * max_sigma
        real = functional.gaussian_blur(image.real, kernel_size, blur_param)
        imaginary = functional.gaussian_blur(image.imag, kernel_size, blur_param)
        return real + 1j * imaginary

    @staticmethod
    def random_aliasing(image, max_acceleration=6, center_fraction_range=(0.08, 0.16)):
        # plt.imshow(image[0].abs(), cmap="gray")
        # plt.show()
        # plt.imshow(torch.log(kspace[0].abs()) + 1, cmap="gray")
        # plt.colorbar()
        # plt.show()

        # Create the mask
        num_cols = image.shape[-1]  # TODO: Maybe choose the other direction
        acceleration = np.random.uniform(1, max_acceleration)
        center_fraction = min(np.random.uniform(*center_fraction_range), 1 / (acceleration + 1))

        num_low_freqs = int(num_cols * center_fraction)  # Floor instead of round
        prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs)
        mask = np.random.random(num_cols) < prob
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad:pad + num_low_freqs] = True
        mask = mask[None, None].astype(np.complex)

        # Mask kspace and get the image back
        kspace = torch.fft.fftshift(torch.fft.fft2(image))
        # kspace_masked = kspace * mask
        image = torch.fft.ifft2(torch.fft.ifftshift(kspace * mask))

        # plt.plot(mask[0, 0])
        # plt.show()
        # plt.imshow(torch.log(kspace_masked[0].abs() + 1), cmap="gray")
        # plt.colorbar()
        # plt.show()
        # plt.imshow(image[0].abs(), cmap="gray")
        # plt.show()
        return image


# Experiment:
#  Pick image, and then start adding noise, or blurring
#  Compare normal model vs new model with augmentation

# https://github.com/mikgroup/MoDL_framework/blob/main/MoDL_2D/patch_extraction_ufmetric.py
