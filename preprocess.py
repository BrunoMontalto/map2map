import argparse
import os
import numpy as np
import pynbody
from glob import glob

from displacement import pos2dis, dis2pos #TODO: this utilities will be probably moved into the map2map folder
from downsampling import random_sample_fixed_seed, downsample_tricubic, average_downsample

import gc

import hashlib


# used to generate a determistic seed based on the folder name
def stable_hash(s):
    return int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16) % (2**32 - 1)

def process_snapshot(folder_path, Ng_HR, Ng_LR, output_path_LR, output_path_HR, downsampling_function, use_physical_units, use_iord, delete_files=False):
    folder_name = os.path.basename(os.path.normpath(folder_path))

    s = pynbody.load(os.path.join(folder_path, 'snapdir_062', 'snap_062'))

    # use physical units
    if use_physical_units:
        s.physical_units()

    boxsize = s.properties['boxsize']

    print('Boxsize:', boxsize, 'type:', type(boxsize))
    print('Boxsize units:', boxsize._register_unit)

    print('Boxsize in kpc/h:', boxsize.in_units('kpc a h**-1'))
    print('Boxsize in Mpc/h:', boxsize.in_units('Mpc a h**-1'))
    print('Boxsize in Gpc/h:', boxsize.in_units('Gpc a h**-1'))



    # convert boxsize to float32 and to Mpc/h
    boxsize = np.float32(boxsize) / 1000.0 # convert to Mpc a h**-1, assuming boxsize is in kpc a h**-1
    print('Boxsize after conversion:', boxsize,'type:', type(boxsize))

    pos_ = s['pos']
    # divide pos by 1000 to convert from kpc/h to Mpc/h, assuming pos is in kpc a h**-1
    pos_ = s['pos'] / 1000.0 # convert to Mpc a h**-1
    print('\nPositions shape:', pos_.shape, 'type:', type(pos_), 'dtype:', pos_.dtype, 'units:', pos_.units, 'min:', pos_.min(), 'max:', pos_.max())

    vel_ = s['vel']
    
    print('Velocities shape:', vel_.shape, 'type:', type(vel_), 'dtype:', vel_.dtype, 'units:', vel_.units, 'min:', vel_.min(), 'max:', vel_.max())


    if use_iord:
        pid_ = s['iord'] - 1 # iord starts at 1, so we subtract 1 to make it zero-indexed
        print("Min iord:", pid_.min())
        print("Max iord:", pid_.max())
        print("Len iord:", len(pid_))
        

        assert len(pos_) == len(vel_) == len(pid_), 'Positions, velocities and particle IDs must have the same length.'
    else:
        pid_ = None

        assert len(pos_) == len(vel_), 'Positions and velocities must have the same length.'

    



    

    Ng = round(len(pos_)** (1/3)) # number of particles per side of the grid

    assert Ng == 1024 # in this case we know the number of particles per side of the grid is 1024, so we can assert it just in case
    assert Ng % Ng_HR == 0 and Ng_HR <= Ng, 'Ng_HR must be a divisor of Ng and less or equal than Ng.'
    
    
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
    dis = dis.astype('f4')
    vel = vel.astype('f4')

    # pairs generation

    # HR
    if Ng != Ng_HR:
        factor = Ng // Ng_HR
        dis_HR = downsampling_function(dis, factor)
        vel_HR = downsampling_function(vel, factor)
    else:
        print('No downsampling needed for HR data, using original resolution.')
        dis_HR = dis
        vel_HR = vel

    assert dis_HR.shape == (3, Ng_HR, Ng_HR, Ng_HR), f'dis_HR shape is {dis_HR.shape}, expected {(3, Ng_HR, Ng_HR, Ng_HR)}'
    assert vel_HR.shape == (3, Ng_HR, Ng_HR, Ng_HR), f'vel_HR shape is {vel_HR.shape}, expected {(3, Ng_HR, Ng_HR, Ng_HR)}'

    # LR
    factor = Ng_HR // Ng_LR
    dis_LR = downsampling_function(dis_HR, factor)
    vel_LR = downsampling_function(vel_HR, factor)

    assert dis_LR.shape == (3, Ng_LR, Ng_LR, Ng_LR), f'dis_LR shape is {dis_LR.shape}, expected {(3, Ng_LR, Ng_LR, Ng_LR)}'
    assert vel_LR.shape == (3, Ng_LR, Ng_LR, Ng_LR), f'vel_LR shape is {vel_LR.shape}, expected {(3, Ng_LR, Ng_LR, Ng_LR)}'



    # save to output folder

    """
    # HR
    hr_folder = os.path.join(output_path, 'HR')
    np.save(os.path.join(hr_folder, f"{folder_name}.npy"), catnorm_HR)

    # LR
    lr_folder = os.path.join(output_path, 'LR')
    np.save(os.path.join(lr_folder, f"{folder_name}.npy"), catnorm_LR)
    """

    #save the first 3 channels (displacement) and the last 3 channels (velocity) separately (add a suffix to the filename)

    #HR
    os.makedirs(output_path_HR, exist_ok=True)
    np.save(os.path.join(output_path_HR, f'{folder_name}_dis.npy'), dis_HR)
    np.save(os.path.join(output_path_HR, f'{folder_name}_vel.npy'), vel_HR)

    #LR
    os.makedirs(output_path_LR, exist_ok=True)
    np.save(os.path.join(output_path_LR, f'{folder_name}_dis.npy'), dis_LR)
    np.save(os.path.join(output_path_LR, f'{folder_name}_vel.npy'), vel_LR)

    del dis, vel, dis_HR, dis_LR, vel_HR, vel_LR
    del s

    # free resources
    gc.collect()

    print(f'Processed {folder_path}:\n')

    
    if delete_files: #NOTE: this deletes all the folder contents, but not the folder itself
        #delete folder path
        print(f'Deleting folder {folder_path}...')
        for root, dirs, files in os.walk(folder_path, topdown=False):
            for name in files:
                file_path = os.path.join(root, name)
                print(f'Deleting file {file_path}...')
                os.remove(file_path)
            for name in dirs:
                dir_path = os.path.join(root, name)
                print(f'Deleting directory {dir_path}...')
                os.rmdir(dir_path)
        print(f'Folder {folder_path} deleted.\n')
    




    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Converts simulation data to displacement fields and performs downsampling for training/testing pairs.')
    parser.add_argument('--input-pattern', type=str, required=True, help='Regular expression for input folders.')

    #bool argument to specify if it is train or test data
    parser.add_argument('--is-test-data', action='store_true', help='Specify if the input data is test data. Default is False (train data).')

    parser.add_argument('--output-folder', type=str, required=True, help='Output folder for processed data.')
    parser.add_argument('--Ng-HR', type=int, required = True, help='Number of particles per side of the HR grid.')
    parser.add_argument('--Ng-LR', type=int, required=True, help='Number of particles per side of the LR grid.')

    # add argument for downsampling method
    parser.add_argument('--downsampling-method', type=str, choices=['random', 'tricubic', 'average'], default='random', help='Method for downsampling the data. Default is "random".')

    parser.add_argument('--use-physical-units', action='store_true', help='Use physical units for the simulation data. Default is False.')

    # add "use_iord" argument
    parser.add_argument('--use-iord', action='store_true', help='Use iord to order the particles. Default is False.')

    parser.add_argument('--delete-files', action='store_true', help='Delete original snapshot files after processing.')
    args = parser.parse_args()

    output_path_LR = os.path.join(args.output_folder, 'train' if not args.is_test_data else 'test', 'LR')
    output_path_HR = os.path.join(args.output_folder, 'train' if not args.is_test_data else 'test', 'HR')


    node_name = os.environ.get("SLURMD_NODENAME", "unknown")
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1"))
    print(f'Running on node: {node_name}, task ID: {task_id}')

    if task_id == 0 or task_id == -1: #task_id == -1 happens when not running as a SLURM job array
        print(f'[TASK_ID = {task_id}] Creating {output_path_LR} and {output_path_HR} folders.')

        os.makedirs(output_path_LR, exist_ok=True)
        os.makedirs(output_path_HR, exist_ok=True)

    while not os.path.exists(output_path_LR) or not os.path.exists(output_path_HR):
        print(f'Waiting for {output_path_LR} and {output_path_HR} to be created...')
        import time

        time.sleep(1)
    


    print(f'Input pattern: {args.input_pattern}')
    print(f'Is test data: {args.is_test_data}')
    print(f'Output folder: {args.output_folder}')
    print(f'HR Ng: {args.Ng_HR}')
    print(f'LR Ng: {args.Ng_LR}')
    print(f'Downsampling method: {args.downsampling_method}')
    print(f'Use physical units: {args.use_physical_units}')
    print(f'Use iord: {args.use_iord}')
    print(f'Delete original files after processing: {args.delete_files}')

    print(f'\n-scale factor: {args.Ng_HR // args.Ng_LR}\n')

    assert args.Ng_HR % args.Ng_LR == 0 and args.Ng_HR > args.Ng_LR, 'Ng_HR must be a multiple of Ng_LR and greater than.'

    downsampling_function = None

    

    if args.downsampling_method == 'tricubic':
        downsampling_function = downsample_tricubic
    elif args.downsampling_method == 'average':
        downsampling_function = average_downsample

    # process data
    input_folders = glob(args.input_pattern)
    print(f'Found {len(input_folders)} input folders.\n\n')
    for folder in input_folders:
        print(f'Processing input folder: {folder}\n')

        if args.downsampling_method == 'random':
            # set seed for reproducibility, according to folder_path basename
            folder_name = os.path.basename(os.path.normpath(folder))
            seed = stable_hash(folder_name) % (2**32 - 1)
            downsampling_function = random_sample_fixed_seed(seed)

        process_snapshot(folder, args.Ng_HR, args.Ng_LR, output_path_LR, output_path_HR, downsampling_function, args.use_physical_units, args.use_iord, args.delete_files)