import argparse
import os
import numpy as np
import pynbody
from glob import glob

from displacement import pos2dis, dis2pos #TODO: this utilities will be probably moved into the map2map folder
from downsampling import random_sample_fixed_seed, unstructured_random_sample_fixed_seed, downsample_tricubic, average_downsample

import gc

import hashlib


# used to generate a determistic seed based on the folder name
def stable_hash(s):
    return int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16) % (2**32 - 1)

"""
def id_to_ijk(particle_id, Nmesh, TileFac):
    Nbase = Nmesh // TileFac
    id0 = particle_id - 1  # zero-based

    # Offset all'interno della griglia base tiled
    kk = id0 % Nbase
    jj = (id0 // Nbase) % Nbase
    ii = (id0 // (Nbase * Nbase)) % Nbase

    # Indice del tile
    k_tile = (id0 // (Nbase**3)) % TileFac
    j_tile = (id0 // (Nbase**3 * TileFac)) % TileFac
    i_tile = (id0 // (Nbase**3 * TileFac**2)) % TileFac

    # Indice globale
    i = i_tile * Nbase + ii
    j = j_tile * Nbase + jj
    k = k_tile * Nbase + kk

    return i, j, k


def id_to_linear_index(particle_id, Nmesh, TileFac):
    i, j, k = id_to_ijk(particle_id, Nmesh, TileFac)
    return (i * Nmesh + j) * Nmesh + k
"""
def process_snapshot(folder_path, Ng_HR, Ng_LR, output_path_LR, output_path_HR, tile_fac, downsampling_function, use_positions, use_physical_units, use_iord, delete_files=False):
    #temporary#

    #load ../../lattice_id_order.npy
    lattice_id_order = np.load('../../lattice_id_order.npy')
    print('Loaded lattice_id_order.npy with shape:', lattice_id_order.shape, 'type:', type(lattice_id_order), 'dtype:', lattice_id_order.dtype)

    ###########
    
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

    pos_ = s['pos'].view(np.float32) # convert to float32 (test 8 and before don't do that)
    # divide pos by 1000 to convert from kpc/h to Mpc/h, assuming pos is in kpc a h**-1
    pos_ = pos_ / 1000.0 # convert to Mpc a h**-1
    print('\nPositions shape:', pos_.shape, 'type:', type(pos_), 'dtype:', pos_.dtype, 'units:', pos_.units, 'min:', pos_.min(), 'max:', pos_.max())

    vel_ = s['vel'].view(np.float32) # convert to float32 (test 8 and before don't do that)
    
    print('Velocities shape:', vel_.shape, 'type:', type(vel_), 'dtype:', vel_.dtype, 'units:', vel_.units, 'min:', vel_.min(), 'max:', vel_.max())


    if use_iord:
        pid_ = s['iord'] #- 1 # iord starts at 1, so we subtract 1 to make it zero-indexed, but not in test 9
        print('Particle IDs (iord) shape:', pid_.shape, 'type:', type(pid_), 'dtype:', pid_.dtype)

        #check if units attribute exists for pid_
        if hasattr(pid_, 'units'):
            print('Particle IDs (iord) units:', pid_.units)
        else:
            print('Particle IDs (iord) has no units attribute')

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
        """
        ######################### 1 #########################
        #get indices that would sort pid_
        #pid_ = np.argsort(pid_) #using this in test3. test2 is the same but without this line

        #sort pid_ in ascending order
        pid_ = np.sort(pid_) #using this in test5 (not in test3 or test2)

        
        # get positions in iord order
        pos = np.empty_like(pos_)
        pos[pid_] = pos_
        pos = pos.reshape(Ng, Ng, Ng, 3)

        # get velocities in iord order
        vel = np.empty_like(vel_)
        vel[pid_] = vel_
        vel = vel.reshape(Ng, Ng, Ng, 3)
        ########################################################
        """

        if False:
            ############################## 2 ##########################

            #using this on test6 (without the previous subblock1). test 7 is the same as test6 but with subblock1
            #test 8 is the same as test 7 but use sublock2

            """
            ### subblock1 ###
            # get positions in iord order
            pos_sorted = np.empty_like(pos_)
            pos_sorted[pid_] = pos_
            del pos_ #added after test7
            gc.collect() #added after test7

            # get velocities in iord order
            vel_sorted = np.empty_like(vel_)
            vel_sorted[pid_] = vel_
            del vel_ #added after test7
            gc.collect() #added after test7

            del vel_
            #############
            """
            ### subblock2 ###
            #use argsort to get the order of indices that would sort pid_
            print("pos_[pid_[0]]:",pos_[pid_[0]]) #checking if this is the first particle position
            print("pos_[pid_[-1]]:",pos_[pid_[-1]]) #checking if this is the last particle position
            order = np.argsort(pid_)
            #sort pid_ in ascending order
            pid_ = pid_[order]

            print("pid_[0] after sorting:", pid_[0]) #should be 0
            print("pid_[-1] after sorting:", pid_[-1]) #should be len(pid_)-1

            # reorder pos_ and vel_ according to order
            pos_sorted = pos_[order]
            vel_sorted = vel_[order]
            ################
            

            iz = pid_ % Ng 
            iy = (pid_ // Ng) % Ng
            ix = pid_ // (Ng * Ng)

            pos = np.empty((Ng, Ng, Ng, 3), dtype=np.float32)
            vel = np.empty((Ng, Ng, Ng, 3), dtype=np.float32)

            pos[ix, iy, iz, :] = pos_sorted #pos_sorted for test7,test8; pos_ for test6
            vel[ix, iy, iz, :] = vel_sorted #vel_sorted for test7,test8; vel_ for test6
            #############################################
        
        if True:
            ########################### 3 #########################

            #test 9-10 (works well but slow)
            
            #get permutation that would recover the same order of ids (id_grid_flatten), assuming we know id_grid_flatten
            val_to_idx = {val: idx for idx, val in enumerate(pid_)}
            inv_perm = np.array([val_to_idx[val] for val in lattice_id_order]) #TODO: dtype?
            print('inv_perm shape:', inv_perm.shape, 'type:', type(inv_perm), 'dtype:', inv_perm.dtype)

            pos = pos_[inv_perm].reshape(Ng, Ng, Ng, 3)
            vel = vel_[inv_perm].reshape(Ng, Ng, Ng, 3)

            del inv_perm, val_to_idx, lattice_id_order
            gc.collect()

            #######################################################

        if False:
            ######################### 4 #########################
            #faster but uses more memory

            pid_ = id_to_linear_index(pid_, Ng, tile_fac)
            inv_perm = np.argsort(pid_)
            pos = pos_[inv_perm].reshape(Ng, Ng, Ng, 3)
            vel = vel_[inv_perm].reshape(Ng, Ng, Ng, 3)
            del inv_perm, pid_
            gc.collect()
            #####################################################
        

    else:
        # without using iord
        pos = pos_.reshape(Ng, Ng, Ng, 3)
        vel = vel_.reshape(Ng, Ng, Ng, 3)

    del pos_, vel_, pid_

    # convert positions to displacement field
    if not use_positions:
        dis = pos2dis(pos, boxsize, Ng)
    else:
        print('Using positions as input, skipping conversion to displacements.')
        dis = pos #NOTE: keeping the variable name "dis"
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

    #print ranges
    print(f'Displacement HR range: min {dis_HR.min()}, max {dis_HR.max()}')
    print(f'Velocity HR range: min {vel_HR.min()}, max {vel_HR.max()}')
    print(f'Displacement LR range: min {dis_LR.min()}, max {dis_LR.max()}')
    print(f'Velocity LR range: min {vel_LR.min()}, max {vel_LR.max()}')

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

    #tilefac
    parser.add_argument('--tile-fac', type=int, default=4, help='**UNUSED** Tiling factor used in the simulation. Default is 4.')

    # add argument for downsampling method
    parser.add_argument('--downsampling-method', type=str, choices=['random', 'random_unstructured', 'tricubic', 'average'], default='random', help='Method for downsampling the data. Default is "random".')
    
    #add argument to decide wether to use positions
    parser.add_argument('--use-positions', action='store_true', help='Skips conversion from positions to displacements. Default is False.')
    
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
    print(f'Tiling factor: {args.tile_fac}')
    print(f'Downsampling method: {args.downsampling_method}')
    print(f'Use positions: {args.use_positions}')
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
        elif args.downsampling_method == 'random_unstructured':
            # set seed for reproducibility, according to folder_path basename
            folder_name = os.path.basename(os.path.normpath(folder))
            seed = stable_hash(folder_name) % (2**32 - 1)
            downsampling_function = unstructured_random_sample_fixed_seed(seed)

        process_snapshot(folder, args.Ng_HR, args.Ng_LR, output_path_LR, output_path_HR, args.tile_fac, downsampling_function, args.use_positions, args.use_physical_units, args.use_iord, args.delete_files)