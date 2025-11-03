import sys
path_to_pip_installs = "/tmp/test_env"
if path_to_pip_installs not in sys.path:
    sys.path.insert(0, path_to_pip_installs)

import os
import numpy as np
from PIL import Image
import argparse

def prepare_data(input_path, output_path, moving_contrast, fixed_contrast, masks=False):
    """
    Prepares the data for the DiffuseMorph model by converting .npy files to .png images.

    Args:
        input_path (str): The path to the input data, which should contain train, val, and test subfolders.
        output_path (str): The path to save the processed data.
        moving_contrast (str): The name of the moving contrast (e.g., 'BOLD').
        fixed_contrast (str): The name of the fixed contrast (e.g., 'DIXON').
    """

    for split in ['train', 'val', 'test']:
        # Create output directories
        print(f"Processing {split} data...")
        if masks:
            split_output = split + '_masks'
        else:
            split_output = split
        output_split_path = os.path.join(output_path, split_output)
        print(f"Saving to {output_split_path}")
        os.makedirs(output_split_path, exist_ok=True)

        # Load data from .npy files
        if masks:
            moving_contrast_load = moving_contrast + '_masks'
            fixed_contrast_load = fixed_contrast + '_masks'
        else:
            moving_contrast_load = moving_contrast
            fixed_contrast_load = fixed_contrast
        moving_data_path = os.path.join(input_path, split, moving_contrast_load + '.npy')
        fixed_data_path = os.path.join(input_path, split, fixed_contrast_load + '.npy')
        print(f"Loading moving data from {moving_data_path}")
        print(f"Loading fixed data from {fixed_data_path}")

        moving_data = np.load(moving_data_path)
        moving_data = (moving_data - np.min(moving_data)) / (np.max(moving_data) - np.min(moving_data))
        fixed_data = np.load(fixed_data_path)
        fixed_data = (fixed_data - np.min(fixed_data)) / (np.max(fixed_data) - np.min(fixed_data))

        # Save each slice as a PNG image
        for i in range(moving_data.shape[0]):
            # --- Moving Image ---
            # Normalize the slice to 0-255 and convert to uint8
            moving_slice_normalized = (moving_data[i] * 255.0).astype(np.uint8)
            #moving_slice_normalized = (255.0 * (moving_slice - np.min(moving_slice)) / (np.max(moving_slice) - np.min(moving_slice))).astype(np.uint8)
            moving_img = Image.fromarray(moving_slice_normalized)
            moving_img.save(os.path.join(output_split_path, f'moving_{i}.png'))

            # --- Fixed Image ---
            # Normalize the slice to 0-255 and convert to uint8
            fixed_slice_normalized = (fixed_data[i] * 255.0).astype(np.uint8)
            #fixed_slice_normalized = (255.0 * (fixed_slice - np.min(fixed_slice)) / (np.max(fixed_slice) - np.min(fixed_slice))).astype(np.uint8)
            fixed_img = Image.fromarray(fixed_slice_normalized)
            fixed_img.save(os.path.join(output_split_path, f'fixed_{i}.png'))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=False, help='Path to the input data containing train, val, test folders.')
    parser.add_argument('--output_path', type=str, required=False, help='Path to save the processed data.')

    args = parser.parse_args()
    input_path = "/home/students/studweilc1/MU-Diff/data/my_data2"
    output_path = "/home/students/studweilc1/DiffuseMorph/dataset/mu_diff_t1"
    input_path = args.input_path
    output_path = args.output_path
    fixed_contrast = "DIXON"

    for moving_contrast in ["T1_mapping_fl2d", "BOLD", "Diffusion"]:
        output_path_contrast = os.path.join(output_path, moving_contrast + '_to_' + fixed_contrast)
        prepare_data(input_path, output_path_contrast, moving_contrast, fixed_contrast)
        prepare_data(input_path, output_path_contrast, moving_contrast, fixed_contrast, masks=True)
