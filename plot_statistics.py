import argparse
import os
import numpy as np
import glob
import datetime
import gc

import matplotlib.pyplot as plt
from displacement import dis2pos_ #TODO: this utilities will be probably moved into the map2map folder
from downsampling import random_sample

def str_list(s): # from args.py
    return s.split(',')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plots statistics and visualizations of preprocessed data.')
    parser.add_argument('--lr-patterns', type=str_list, required=True, help='Regular expression for low-resolution input folders.')
    parser.add_argument('--hr-patterns', type=str_list, required=True, help='Regular expression for high-resolution input folders.')
    parser.add_argument('--output-folder', type=str, required=True, help='Output folder for statistics and visualizations.')

    # arguments for particle distributions. The user can choose from X, Y, Z, Xvel, Yvel, Zvel
    parser.add_argument('--particle-distributions', type=str, nargs='*', choices=['X', 'Y', 'Z', 'Xvel', 'Yvel', 'Zvel'],
                        help='List of particle distributions to plot. Choose from X, Y, Z, Xvel, Yvel, Zvel. If not provided, no distributions will be plotted.')
    
    # arguments for projections. The user can choose from XY, XZ, YZ, XYvel, XZvel, YZvel
    parser.add_argument('--projections', type=str, nargs='*', choices=['XY', 'XZ', 'YZ', 'XYvel', 'XZvel', 'YZvel'],
                        help='List of projections to plot. Choose from XY, XZ, YZ, XYvel, XZvel, YZvel. If not provided, no projections will be plotted.')

    # argument for power spectra plot
    parser.add_argument('--power-spectra', action='store_true', help='If set, compute and plot power spectra of the fields.')

    # argument for "use positions" (instead of displacements) option
    parser.add_argument('--use-positions', action='store_true', help='If set, use positions instead of displacements for plotting distributions and projections.')

    # argument for boxsize, default is 1.4925373e+09 (in kpc)
    parser.add_argument('--boxsize', type=float, default=1.4925373e+09, help='Box size in kpc. Default is 1.4925373e+09.')


    args = parser.parse_args()

    # create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)


    # lr patterns will be something like 'a/b/dataset_name/train/LR/*', get dataset_name
    dataset_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(args.lr_patterns[0]))))

    # create a subfolder for the dataset in the output folder
    output_dataset_folder = os.path.join(args.output_folder, dataset_name)
    os.makedirs(output_dataset_folder, exist_ok=True)

    # create 3 subfolders for the dataset: distributions, projections, and power_spectra
    os.makedirs(os.path.join(output_dataset_folder, 'distributions'), exist_ok=True)
    os.makedirs(os.path.join(output_dataset_folder, 'projections'), exist_ok=True)
    os.makedirs(os.path.join(output_dataset_folder, 'power_spectra'), exist_ok=True)

    # print arguments
    print(f"Script started at {datetime.datetime.now()}\n")
    print(f"Low-resolution patterns: {args.lr_patterns}")
    print(f"High-resolution patterns: {args.hr_patterns}")
    print(f"Output folder: {output_dataset_folder}")
    print(f"Particle distributions to plot: {args.particle_distributions if args.particle_distributions else 'None'}")
    print(f"Projections to plot: {args.projections if args.projections else 'None'}")
    print(f"Power spectra: {'Enabled' if args.power_spectra else 'Disabled'}\n")

    #args.lr_patterns is a str_list, we assume it has 2 elements: one for dis and one for vel

    lr_files_dis = glob.glob(args.lr_patterns[0]) if len(args.lr_patterns) > 0 else []
    lr_files_vel = glob.glob(args.lr_patterns[1]) if len(args.lr_patterns) > 1 else []
    hr_files_dis = glob.glob(args.hr_patterns[0]) if len(args.hr_patterns) > 0 else []
    hr_files_vel = glob.glob(args.hr_patterns[1]) if len(args.hr_patterns) > 1 else []

    #replace empty arrays with lists of None of the same length as the other list
    if len(lr_files_dis) == 0:
        lr_files_dis = [None] * len(lr_files_vel)
    if len(lr_files_vel) == 0:
        lr_files_vel = [None] * len(lr_files_dis)
    if len(hr_files_dis) == 0:
        hr_files_dis = [None] * len(hr_files_vel)
    if len(hr_files_vel) == 0:
        hr_files_vel = [None] * len(hr_files_dis)

    for lr_file_dis, lr_file_vel, hr_file_dis, hr_file_vel in zip(lr_files_dis, lr_files_vel, hr_files_dis, hr_files_vel):
        # load the low-resolution and high-resolution data

        # get filename
        
        #try with dis 

        #lr
        if lr_file_dis is not None:
            filename_lr = os.path.basename(lr_file_dis)
        elif lr_file_vel is not None:
            filename_lr = os.path.basename(lr_file_vel)
        else:
            filename_lr = 'None'

        #hr
        if hr_file_dis is not None:
            filename_hr = os.path.basename(hr_file_dis)
        elif hr_file_vel is not None:
            filename_hr = os.path.basename(hr_file_vel)
        else:
            filename_hr = 'None'

        #check if the plot already exists
        if (args.particle_distributions and os.path.exists(os.path.join(output_dataset_folder, 'distributions', f'positions_{filename_lr}_{args.particle_distributions}_vs_{filename_hr}.png'))) or (args.projections and os.path.exists(os.path.join(output_dataset_folder, 'projections', f'positions_{filename_lr}_{args.projections}_vs_{filename_hr}.png'))):
            print(f"Skipping {filename_lr} and {filename_hr} as the plots already exist.")
            continue

        lr_dis = np.load(lr_file_dis) if lr_file_dis else None
        lr_vel = np.load(lr_file_vel) if lr_file_vel else None
        hr_dis = np.load(hr_file_dis) if hr_file_dis else None
        hr_vel = np.load(hr_file_vel) if hr_file_vel else None

        if lr_dis is not None or lr_vel is not None:
            #take the 2nd, 3rd and 4th elements of the shape of lr_dis or lr_vel (if it exists)
            Ng_LR = lr_dis.shape[1] if lr_dis is not None else lr_vel.shape[1]
        else:
            Ng_LR = None
        
        if Ng_LR is not None:
            lr_dis_vel = np.empty((6, Ng_LR, Ng_LR, Ng_LR), dtype=np.float32)
            if lr_dis is not None:
                lr_dis_vel[:3] = lr_dis
            if lr_vel is not None:
                lr_dis_vel[3:] = lr_vel

            if Ng_LR > 64:
                print(f"Downsampling LR data from {Ng_LR}^3 to 64^3.")
                # downsample the LR data to 64^3
                factor = Ng_LR // 64
                lr_dis_vel = random_sample(lr_dis_vel, factor)
                gc.collect() # clear memory after downsampling

        else:
            lr_dis_vel = None

        if hr_dis is not None or hr_vel is not None:
            #take the 2nd, 3rd and 4th elements of the shape of hr_dis or hr_vel (if it exists)
            Ng_HR = hr_dis.shape[1] if hr_dis is not None else hr_vel.shape[1]
        else:
            Ng_HR = None

        if Ng_HR is not None:
            hr_dis_vel = np.empty((6, Ng_HR, Ng_HR, Ng_HR), dtype=np.float32)
            if hr_dis is not None:
                hr_dis_vel[:3] = hr_dis
            if hr_vel is not None:
                hr_dis_vel[3:] = hr_vel

            if Ng_HR > 128:
                print(f"Downsampling HR data from {Ng_HR}^3 to 128^3.")
                # downsample the HR data to 128^3
                factor = Ng_HR // 128
                hr_dis_vel = random_sample(hr_dis_vel, factor)
                gc.collect()  # clear memory after downsampling
        else:
            hr_dis_vel = None

        if args.use_positions:
            lr_dis_vel = dis2pos_(lr_dis_vel, args.boxsize, Ng_LR) if lr_dis_vel is not None else None
            hr_dis_vel = dis2pos_(hr_dis_vel, args.boxsize, Ng_HR) if hr_dis_vel is not None else None

        # assume lr_dis_vel has shape (6, Ng_LR, Ng_LR, Ng_LR) and hr_dis_vel has shape (6, Ng_HR, Ng_HR, Ng_HR)
        
        # plot particle distributions if specified
        if args.particle_distributions:
            # do a subplot with 2 columns if both LR and HR are available, otherwise just one column
            n_cols = 2 if lr_dis_vel is not None and hr_dis_vel is not None else 1
            n_rows = len(args.particle_distributions)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 5 * n_rows), squeeze=False)

            for i, field in enumerate(args.particle_distributions):
                # map field to index
                field_index = {'X': 0, 'Y': 1, 'Z': 2, 'Xvel': 3, 'Yvel': 4, 'Zvel': 5}[field]

                if n_cols == 1:
                    ax = axes[i, 0]

                    if lr_dis_vel is not None:
                        ax.hist(lr_dis_vel[field_index, :].flatten(), bins=100, alpha=0.5, label=f'LR {field}', color='blue')
                    if hr_dis_vel is not None:
                        ax.hist(hr_dis_vel[field_index, :].flatten(), bins=100, alpha=0.5, label=f'HR {field}', color='orange')
                    ax.set_title(f'{field} Distribution')
                    ax.set_xlabel(field)
                    ax.set_ylabel('Frequency')
                    #ax.legend()

                else:
                    ax_lr = axes[i, 0]
                    ax_hr = axes[i, 1]

                    if lr_dis_vel is not None:
                        ax_lr.hist(lr_dis_vel[field_index, :].flatten(), bins=100, alpha=0.5, label=f'LR {field}', color='blue')
                        ax_lr.set_title(f'LR {field} Distribution ({Ng_LR}^3 particles)')
                        ax_lr.set_xlabel(field)
                        ax_lr.set_ylabel('Frequency')
                        #ax_lr.legend()
                    if hr_dis_vel is not None:
                        ax_hr.hist(hr_dis_vel[field_index, :].flatten(), bins=100, alpha=0.5, label=f'HR {field}', color='orange')
                        ax_hr.set_title(f'HR {field} Distribution ({Ng_HR}^3 particles)')
                        ax_hr.set_xlabel(field)
                        ax_hr.set_ylabel('Frequency')
                        #ax_hr.legend()

                

            # save the figure
            posdis = 'positions_' if args.use_positions else 'displacements_'
            fig.savefig(os.path.join(output_dataset_folder, 'distributions', f'{posdis}{filename_lr}_{args.particle_distributions}_vs_{filename_hr}.png'))
            plt.close(fig)

        # plot projections if specified (2d scatter plots)
        if args.projections:
            # do a subplot with 2 columns if both LR and HR are available, otherwise just one column
            n_cols = 2 if lr_dis_vel is not None and hr_dis_vel is not None else 1
            n_rows = len(args.projections)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 5 * n_rows), squeeze=False)

            for i, field in enumerate(args.projections):
                # map field to index
                field_index = {'XY': (0, 1), 'XZ': (0, 2), 'YZ': (1, 2), 'XYvel': (3, 4), 'XZvel': (3, 5), 'YZvel': (4, 5)}[field]

                if n_cols == 1:
                    ax = axes[i, 0]

                    if lr_dis_vel is not None:
                        ax.scatter(lr_dis_vel[field_index[0], :].flatten(), lr_dis_vel[field_index[1], :].flatten(), alpha=0.5, label=f'LR {field}', color='blue', s = 0.1)
                    if hr_dis_vel is not None:
                        ax.scatter(hr_dis_vel[field_index[0], :].flatten(), hr_dis_vel[field_index[1], :].flatten(), alpha=0.5, label=f'HR {field}', color='orange', s = 0.05)
                    ax.set_title(f'{field} Projection')
                    ax.set_xlabel(field.split('vel')[0][0])
                    ax.set_ylabel(field.split('vel')[0][1])
                    #ax.legend()

                else:
                    ax_lr = axes[i, 0]
                    ax_hr = axes[i, 1]

                    if lr_dis_vel is not None:
                        ax_lr.scatter(lr_dis_vel[field_index[0], :].flatten(), lr_dis_vel[field_index[1], :].flatten(), alpha=0.5, label=f'LR {field}', color='blue', s = 0.1)
                        ax_lr.set_title(f'LR {field} Projection')
                        ax_lr.set_xlabel(field.split('vel')[0][0])
                        ax_lr.set_ylabel(field.split('vel')[0][1])
                        #ax_lr.legend()
                    if hr_dis_vel is not None:
                        ax_hr.scatter(hr_dis_vel[field_index[0], :].flatten(), hr_dis_vel[field_index[1], :].flatten(), alpha=0.5, label=f'HR {field}', color='orange', s = 0.05)
                        ax_hr.set_title(f'HR {field} Projection')
                        ax_hr.set_xlabel(field.split('vel')[0][0])
                        ax_hr.set_ylabel(field.split('vel')[0][1])
                        #ax_hr.legend()
            # save the figure
            posdis = 'positions_' if args.use_positions else 'displacements_'
            fig.savefig(os.path.join(output_dataset_folder, 'projections', f'{posdis}{filename_lr}_{args.projections}_vs_{filename_hr}.png'))
            plt.close(fig)
        
        
                    
                    


    print(f"Script ended at {datetime.datetime.now()}\n")




# example execution for distributions of all fields
"""
python3 plot_statistics.py --lr_patterns 'DEMNUni_64_to_128/train/LR/seed_123465.npy' --hr_patterns 'DEMNUni_64_to_128/train/HR/seed_123465.npy' --output_folder 'plots' --particle_distributions X Y Z Xvel Yvel Zvel
"""

# example execution for projections of all fields
"""
python3 plot_statistics.py --lr_patterns 'DEMNUni_64_to_128/train/LR/seed_123465.npy' --hr_patterns 'DEMNUni_64_to_128/train/HR/seed_123465.npy' --output_folder 'plots' --projections XY XZ YZ XYvel XZvel YZvel
"""
