import random
from glob import glob

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional
# import utils
from scipy import interpolate
from skimage.exposure import match_histograms

def normalize(img, low=0.01, high=0.99):
    high_quantile = torch.quantile(img, high)
    low_quantile = torch.quantile(img, low)
    return (img-low_quantile)/(high_quantile - low_quantile)
    # return img / 1.0

def minmaxnorm(img):
    return (img-img.min())/(img.max()-img.min())

class UFData(Dataset):

    def __init__(self, data_directory, max_offset=None, magnitude=False, device=torch.device('cpu'),
                 fastmri=False, random_augmentation=True, augment_probability = 0.9, normalization=0.99):
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

        print(f"data dir {data_directory}")

        print(f"Using data from: {data_directory}\nFound {len(self.image_paths)} image paths.")
        self.device = device
        self.magnitude = magnitude
        self.random_augmentation = random_augmentation
        self.augment_probability = augment_probability
        self.normalization = normalization

        if fastmri:  # Fast MRI Dataset:
            # self.cropped_image_size = np.array([640, 320]) - max_offset
            self.cropped_image_size = np.array([256, 256])
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
            if self.magnitude:
                image = torch.abs(image).float()
                image = minmaxnorm(image)
            
            image = self.augment_image(image)  # Model will be sensitive to this
            image = minmaxnorm(image)
             
            # sensitive_image = torch.clone(image)
            
            image2 = torch.clone(image)

            #Augmentations we want to be insensitive to
            jitter_probability= 0.9
            # noise_probability=0.8
            blur_probability= 0.9
            invert_probability = 0.5
            solarize_probability = 0

            if random.random() < self.augment_probability:
                #with 50% prob do normal aug
                if random.random() >= 0:
                    if random.random() < jitter_probability:
                        image = self.random_jitter(image)
                    if random.random() < jitter_probability:
                        image2 = self.random_jitter(image2)
                    if random.random() < blur_probability:
                        image = self.random_blur(image)
                    if random.random() < blur_probability:
                        image2 = self.random_blur(image2)
                    if random.random() < invert_probability:
                        if random.random() < 0.5:
                            image = self.random_invert(image)
                        else:
                            image2 = self.random_invert(image2)
                    if random.random() < solarize_probability:
                        image = self.random_solarize(image)
                    if random.random() < solarize_probability:
                        image2 = self.random_solarize(image2)
                #with 50% prob do curve augmentation
                else:
                    # print("this is commented out rn")
                    image2 = self.curve_aug(image2)

            image = self.random_phase(image)
            image2 = self.random_phase(image2)
            
            if self.magnitude:
                image = torch.abs(image).float()
                image2 = torch.abs(image2).float()
                image1 = normalize(image)
                image2 = normalize(image2)
            else:
                image1 = self.complex2channels(image)
                image2 = self.complex2channels(image2)
    
            #random crop image
            image1, offset = self.random_crop(image1)
            image2, _ = self.random_crop(image2, offset=offset)
            #  og, _ = self.random_crop(og, offset=offset)
            # sensitive_image, _ = self.random_crop(sensitive_image , offset=offset)

            return image1, image2#, og, sensitive_image

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
        # image = self.random_crop(image)
        # og = torch.clone(image)

        if random.random() < augment_probability:
            if random.random() < jitter_probability:
                image = self.random_jitter(image)
            if random.random() < blur_probability:
                image = self.random_blur(image)
            # if random.random() < noise_probability:  # Noise and blur or blur and noise?
            #     image = self.random_noise(image)
            # if random.random() < aliasing_probability:  #Commented out aliasing for now
            #     image = self.random_aliasing(image)

        # TODO
        #  Could also do some spiral stuff? with NUFFT
        #  Could also do off resonance? Is all you need.
        # TODO: load data from not only fastmri, but also undersampled recons with PICS

        return image#, og

    @staticmethod
    def ifft1c(tensor, dim=-1):
        tensor = torch.fft.ifftshift(tensor, dim=dim)
        tensor = torch.fft.ifft(tensor, dim=dim)
        return torch.fft.fftshift(tensor, dim=dim)

    
    def curve_aug(self, img):
        """ 
        map image to a new pixel intensity curve 
        and then equalize the augmented image to a randomly selected image in the datase
        """

        #get mapping
        kspace = torch.rand((1000, 1000,), dtype=torch.complex64) - 0.5 - 0.5j
        kx = torch.linspace(-1, 1, 1000)
        exp = torch.exp(-torch.sqrt(kx**2)/0.001)
        mapping = self.ifft1c(kspace * exp[None], dim=-1).real
        mapping -= torch.amin(mapping, dim=-1, keepdim=True)
        mapping /= torch.amax(mapping, dim=-1, keepdim=True)
    
        shape = img.shape
        img = img.reshape(-1, 1)
        f = interpolate.interp1d(torch.linspace(0, 1, 1000), mapping[0])
        new_img = f(img).reshape(shape)

        #select another image
        other_file = random.choice(self.image_paths)
        if self.h5_format:
            other_img = np.array(h5py.File(other_file)["target"])[None]
        else:
            other_img = np.load(other_file)[None]
        other_img = torch.from_numpy(other_img)
        other_img = torch.abs(other_img).float()
        other_img = minmaxnorm(other_img)
        other_img = other_img.numpy()

        #make the new mapped image match the histogram of the other image
        #these need to be np arrays
        new_img = match_histograms(new_img, other_img)
        new_img = torch.from_numpy(new_img)
    
        return new_img
    
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
        return image[:, offset[0]:stop[0], offset[1]:stop[1]], offset

    @staticmethod
    def random_jitter(image, max_brightness=0.25, max_gamma=0.5, max_hue=0, max_saturation=0):
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
        if torch.is_complex(image):
            real, imaginary = image.real, image.imag
        else:
            real = image

        brightness_param = random.uniform(1 - max_brightness, 1 + max_brightness)
        real = functional.adjust_brightness(real, brightness_param)
        if torch.is_complex(image):
            imaginary = functional.adjust_brightness(imaginary, brightness_param)

        gamma_param = random.uniform(1 - max_gamma, 1 + max_gamma)
        real = functional.adjust_gamma(real, gamma_param)
        if torch.is_complex(image):
            imaginary = functional.adjust_gamma(imaginary, gamma_param)

        # These color jitter augmentations don't make sense for grayscale - uncomment them if you are using color images
        # hue_param = random.uniform(- max_hue, max_hue)
        # real = functional.adjust_hue(real, hue_param)
        # imaginary = functional.adjust_hue(imaginary, hue_param)

        # saturation_param = random.uniform(1 - max_saturation, 1 + max_saturation)
        # real = functional.adjust_saturation(real, saturation_param)
        # imaginary = functional.adjust_saturation(imaginary, saturation_param)
        if torch.is_complex(image):
            return real + 1j * imaginary
        else:
            return real

    @staticmethod
    def random_noise(image, max_std=0.2):
        std = torch.rand(1) * max_std  # Maybe make this also (abs of) gaussian
        return image * (1 - std) + torch.randn(image.size(), dtype=image.dtype) * std

    @staticmethod
    def random_blur(image, max_sigma=3, kernel_size=7):#19):
        blur_param = random.random() * max_sigma
        if torch.is_complex(image):
            real = functional.gaussian_blur(image.real, kernel_size, blur_param)
            imaginary = functional.gaussian_blur(image.imag, kernel_size, blur_param)
            return real + 1j * imaginary
        else:
            return functional.gaussian_blur(image, kernel_size, blur_param)

    @staticmethod
    def random_invert(image):
        # real, imaginary = image.real, image.imag
        real = functional.invert(image)
        # imaginary = functional.invert(imaginary)
        return real #+ 1j * imaginary    
    
    @staticmethod
    def random_solarize(image): 
        image = minmaxnorm(image)
        thresh = random.uniform(0, 0.5)
        return functional.solarize(image, thresh)
   
   
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
