from torch.utils.data import Dataset
import data.util_2D as Util
import os
import numpy as np
from skimage import io

class MRIDataset(Dataset):
    """
    Custom dataset for loading 2D grayscale medical images and their corresponding masks for the DiffuseMorph model.
    """
    def __init__(self, dataroot, split='test'):
        self.split = split
        self.imageNum = []

        self.datapath = os.path.join(dataroot, split)
        # Construct the path to the masks
        self.maskpath = os.path.join(dataroot, f"{split}_masks")

        # Assuming moving and fixed images are named 'moving_i.png' and 'fixed_i.png'
        num_images = len([name for name in os.listdir(self.datapath) if name.startswith('moving_')])

        for i in range(num_images):
            self.imageNum.append([f'moving_{i}.png', f'fixed_{i}.png'])

        self.data_len = len(self.imageNum)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        fileInfo = self.imageNum[index]
        moving_img_name, fixed_img_name = fileInfo[0], fileInfo[1]
        
        # Construct full paths for images
        moving_img_path = os.path.join(self.datapath, moving_img_name)
        fixed_img_path = os.path.join(self.datapath, fixed_img_name)

        # Construct full paths for masks
        moving_mask_path = os.path.join(self.maskpath, moving_img_name)
        fixed_mask_path = os.path.join(self.maskpath, fixed_img_name)

        # Load images as grayscale and add a channel dimension
        moving_img = io.imread(moving_img_path, as_gray=True).astype(float)
        fixed_img = io.imread(fixed_img_path, as_gray=True).astype(float)

        # Load masks as grayscale and add a channel dimension
        moving_mask = io.imread(moving_mask_path, as_gray=True).astype(float)
        fixed_mask = io.imread(fixed_mask_path, as_gray=True).astype(float)

        # Normalize images and masks to [0, 1]
        moving_img /= 255.0
        fixed_img /= 255.0
        moving_mask /= 255.0
        fixed_mask /= 255.0
        
        moving_img = moving_img[:, :, np.newaxis]
        fixed_img = fixed_img[:, :, np.newaxis]
        moving_mask = moving_mask[:, :, np.newaxis]
        fixed_mask = fixed_mask[:, :, np.newaxis]


        # Apply transformations to both images and masks
        [moving_img, fixed_img, moving_mask, fixed_mask] = Util.transform_augment(
            [moving_img, fixed_img, moving_mask, fixed_mask], 
            split=self.split, 
            min_max=(-1, 1)
        )

        return {
            'M': moving_img, 
            'F': fixed_img, 
            'MM': moving_mask, # Add Moving Mask
            'FM': fixed_mask,  # Add Fixed Mask
            "Index": index, 
            "nS": 1, 
            'P': fileInfo
        }