# The input is a dataset directory containing 'train' and 'test' subfolders.
# Each of these includes 'HR' and 'LR' folders with .npy files.
# The script verifies that:
# - 'train/HR' and 'train/LR' contain exactly the same filenames
# - 'test/HR' and 'test/LR' contain exactly the same filenames
# - all files within each folder have consistent sizes

import os
import sys
import numpy as np
import argparse
from glob import glob

def main(args):
    parser = argparse.ArgumentParser(description='Sanity check for dataset folders.')
    parser.add_argument('--dataset-folder', type=str, required=True, help='Path to the dataset folder containing train and test subfolders.')
    args = parser.parse_args(args)

    dataset_folder = args.dataset_folder
    train_hr_folder = os.path.join(dataset_folder, 'train', 'HR')
    train_lr_folder = os.path.join(dataset_folder, 'train', 'LR')
    test_hr_folder = os.path.join(dataset_folder, 'test', 'HR')
    test_lr_folder = os.path.join(dataset_folder, 'test', 'LR')

    # Check if all required folders exist
    for folder in [train_hr_folder, train_lr_folder, test_hr_folder, test_lr_folder]:
        if not os.path.exists(folder):
            print(f"Error: Folder {folder} does not exist.")
            return

    # Get list of files in each folder
    train_hr_files = sorted([os.path.basename(f) for f in glob(os.path.join(train_hr_folder, '*.npy'))])
    train_lr_files = sorted([os.path.basename(f) for f in glob(os.path.join(train_lr_folder, '*.npy'))])
    test_hr_files = sorted([os.path.basename(f) for f in glob(os.path.join(test_hr_folder, '*.npy'))])
    test_lr_files = sorted([os.path.basename(f) for f in glob(os.path.join(test_lr_folder, '*.npy'))])

    #print lenghts
    print("Number of files in train HR:", len(train_hr_files))
    print("Number of files in train LR:", len(train_lr_files))
    print("Number of files in test HR:", len(test_hr_files))
    print("Number of files in test LR:", len(test_lr_files))

    # Check if filenames match between HR and LR for both train and test
    if train_hr_files != train_lr_files:
        print("Error: Filenames in train HR and LR folders do not match.")
        return
    if test_hr_files != test_lr_files:
        print("Error: Filenames in test HR and LR folders do not match.")
        return

    # Check that all files in each folder occupy the same disk space, if not log the differences.
    # Does not load the files into memory, just checks their sizes on disk.
    def check_file_shapes(folder, files):
        sizes = [os.path.getsize(os.path.join(folder, f)) for f in files]

        print("average size for folder", folder, np.mean(sizes), "bytes")

        if len(set(sizes)) != 1:
            print(f"Error: Not all files in folder {folder} have the same size.")
            for f, s in zip(files, sizes):
                print(f"File: {f}, Size: {s} bytes")
            return False
        return True
    

    for folder, files in [(train_hr_folder, train_hr_files), (train_lr_folder, train_lr_files), (test_hr_folder, test_hr_files), (test_lr_folder, test_lr_files)]:
        if not check_file_shapes(folder, files):
            return
        

    print("Sanity check passed: All checks successful.")

if __name__ == '__main__':
    
    main(sys.argv[1:])
