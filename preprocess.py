import argparse
import os
import numpy as np
import pynbody
import datetime
from glob import glob

from displacement import pos2dis, dis2pos #TODO: this utilities will be probably moved into the map2map folder
from downsampling import random_sample, downsample_tricubic

import gc

import hashlib


# used to generate a determistic seed based on the folder name
def stable_hash(s):
    return int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16) % (2**32 - 1)

def process_snapshot(folder_path, train, Ng_HR, Ng_LR, output_folder, downsampling_function, use_iord, delete_files=False):
    # set seed for reproducibility, according to folder_path basename
    folder_name = os.path.basename(os.path.normpath(folder_path))

    seed = stable_hash(folder_name) % (2**32 - 1)
    np.random.seed(seed)

    s = pynbody.load(os.path.join(folder_path, 'snapdir_062', 'snap_062'))

    # use physical units
    s.physical_units() 

    boxsize = s.properties['boxsize']

    print("Boxsize:", boxsize, "type:", type(boxsize))
    print("Boxsize units:", boxsize._register_unit)

    # convert boxsize to float32
    boxsize = np.float32(boxsize.in_units('kpc'))
    print("Boxsize after conversion:", boxsize, "type:", type(boxsize))

    pos_ = s['pos']
    print("\nPositions shape:", pos_.shape, "type:", type(pos_), "dtype:", pos_.dtype, "units:", pos_.units)

    vel_ = s['vel']
    print("Velocities shape:", vel_.shape, "type:", type(vel_), "dtype:", vel_.dtype, "units:", vel_.units)

    if use_iord:
        pid_ = s['iord'] - 1 # iord starts at 1, so we subtract 1 to make it zero-indexed
        #print("Min iord:", pid_.min())
        #print("Max iord:", pid_.max())
        #print("Len iord:", len(pid_))
    else:
        pid_ = None

    



    assert len(pos_) == len(vel_) == len(pid_), "Positions, velocities and particle IDs must have the same length."

    Ng = round(len(pos_)** (1/3)) # number of particles per side of the grid

    assert Ng == 1024 # in this case we know the number of particles per side of the grid is 1024, so we can assert it just in case
    assert Ng % Ng_HR == 0 and Ng_HR <= Ng, "Ng_HR must be a divisor of Ng and less or equal than Ng."
    
    
    if use_iord:
        # using iord
        # get positions in iord order
        pos = np.empty_like(pos_)
        pos[pid_] = pos_
        pos = pos.reshape(Ng, Ng, Ng, 3)

        # get velocities in iord order
        vel = np.empty_like(vel_)
        vel[pid_] = vel_
        vel = vel.reshape(Ng, Ng, Ng, 3)
    else:
        # without using iord
        pos = pos_.reshape(Ng, Ng, Ng, 3)
        vel = vel_.reshape(Ng, Ng, Ng, 3)

    del pos_, vel_, pid_

    # convert positions to displacement field
    dis = pos2dis(pos, boxsize, Ng)
    del pos

    dis = dis.astype('f4')
    vel = vel.astype('f4')
    
    # to channel first
    dis = np.moveaxis(dis,-1,0) 
    vel = np.moveaxis(vel,-1,0)

    # NOTE: (no normalization)

    # concatenate displacement and velocity fields
    catnorm_HR = np.concatenate([dis, vel], axis=0)
    catnorm_HR = catnorm_HR.astype('f4')

    del dis, vel

    # pairs generation

    # HR
    if Ng != Ng_HR:
        factor = Ng // Ng_HR
        catnorm_HR = downsampling_function(catnorm_HR, factor)

    assert catnorm_HR.shape == (6, Ng_HR, Ng_HR, Ng_HR), f"catnorm_HR shape is {catnorm_HR.shape}, expected {(6, Ng_HR, Ng_HR, Ng_HR)}"

    # LR
    factor = Ng_HR // Ng_LR
    catnorm_LR = downsampling_function(catnorm_HR, factor)

    assert catnorm_LR.shape == (6, Ng_LR, Ng_LR, Ng_LR), f"catnorm_LR shape is {catnorm_LR.shape}, expected {(6, Ng_LR, Ng_LR, Ng_LR)}"



    # save to output folder
    output_path = os.path.join(output_folder, 'train' if train else 'test')

    # HR
    hr_folder = os.path.join(output_path, 'HR')
    np.save(os.path.join(hr_folder, f"{folder_name}.npy"), catnorm_HR)

    # LR
    lr_folder = os.path.join(output_path, 'LR')
    np.save(os.path.join(lr_folder, f"{folder_name}.npy"), catnorm_LR)

    del catnorm_HR, catnorm_LR
    del s

    # free resources
    gc.collect()

    print(f"Processed {folder_path}:\n")

    """
    if delete_files:
        print("Deleting files in the snapshot folder...\n")
        #delete files
        folder_path = os.path.join(folder_path, 'snapdir_062')
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
        print("\n")
    """




    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Converts simulation data to displacement fields and performs downsampling for training/testing pairs.')
    parser.add_argument('--train_patterns', type=str, required=True, help='Regular expression for training input folders.')
    parser.add_argument('--test_patterns', type=str, required=True, help='Regular expression for testing input folders.')
    parser.add_argument('--output_folder', type=str, required=True, help='Output folder for processed data.')
    parser.add_argument('--Ng_HR', type=int, required = True, help='Number of particles per side of the HR grid.')
    parser.add_argument('--Ng_LR', type=int, required=True, help='Number of particles per side of the LR grid.')

    # add argument for downsampling method
    parser.add_argument('--downsampling_method', type=str, choices=['random', 'tricubic'], default='random', help='Method for downsampling the data. Default is "random".')

    # add "use_iord" argument
    parser.add_argument('--use_iord', action='store_true', help='Use iord to order the particles. Default is False.')

    parser.add_argument('--delete_files', action='store_true', help='Delete original snapshot files after processing.')

    args = parser.parse_args()

    # create output directories
    os.makedirs(os.path.join(args.output_folder, 'train', 'LR'), exist_ok=True)
    os.makedirs(os.path.join(args.output_folder, 'train', 'HR'), exist_ok=True)
    os.makedirs(os.path.join(args.output_folder, 'test', 'LR'), exist_ok=True)
    os.makedirs(os.path.join(args.output_folder, 'test', 'HR'), exist_ok=True)


    print(f"Preprocessing started at {datetime.datetime.now()}\n")

    print(f"Train patterns: {args.train_patterns}")
    print(f"Test patterns: {args.test_patterns}")
    print(f"Output folder: {args.output_folder}")
    print(f"HR Ng: {args.Ng_HR}")
    print(f"LR Ng: {args.Ng_LR}")
    print(f"Downsampling method: {args.downsampling_method}")
    print(f"Use iord: {args.use_iord}")
    print(f"Delete original files after processing: {args.delete_files}")

    print(f"\n-scale factor: {args.Ng_HR // args.Ng_LR}\n")

    assert args.Ng_HR % args.Ng_LR == 0 and args.Ng_HR > args.Ng_LR, "Ng_HR must be a multiple of Ng_LR and greater than."

    downsampling_function = None

    if args.downsampling_method == 'random':
        downsampling_function = random_sample
    elif args.downsampling_method == 'tricubic':
        downsampling_function = downsample_tricubic

    # process training data
    train_folders = glob(args.train_patterns)
    print(f"Found {len(train_folders)} training folders.\n\n")
    for folder in train_folders:
        print(f"Processing training folder: {folder}\n")
        #try:
        process_snapshot(folder, True, args.Ng_HR, args.Ng_LR, args.output_folder, downsampling_function, args.use_iord, args.delete_files)
        #except Exception as e:
        #    log_file.write(f"Error processing {folder}: {e}\n")

    # process testing data
    test_folders = glob(args.test_patterns)
    print(f"Found {len(test_folders)} testing folders.\n\n")
    for folder in test_folders:
        print(f"Processing testing folder: {folder}\n")
        #try:
        process_snapshot(folder, False, args.Ng_HR, args.Ng_LR, args.output_folder, downsampling_function, args.use_iord, args.delete_files)
        #except Exception as e:
        #    log_file.write(f"Error processing {folder}: {e}\n")

    print(f"Preprocessing finished at {datetime.datetime.now()}\n\n")


