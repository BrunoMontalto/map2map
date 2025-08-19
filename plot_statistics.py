import argparse
import os
import numpy as np
import glob
import datetime

import matplotlib.pyplot as plt
from displacement import dis2pos #TODO: this utilities will be probably moved into the map2map folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plots statistics and visualizations of preprocessed data.')
    parser.add_argument('--lr_patterns', type=str, required=True, help='Regular expression for low-resolution input folders.')
    parser.add_argument('--hr_patterns', type=str, required=True, help='Regular expression for high-resolution input folders.')
    parser.add_argument('--output_folder', type=str, required=True, help='Output folder for statistics and visualizations.')

    # arguments for particle distributions. The user can choose from X, Y, Z, Xvel, Yvel, Zvel
    parser.add_argument('--particle_distributions', type=str, nargs='*', choices=['X', 'Y', 'Z', 'Xvel', 'Yvel', 'Zvel'],
                        help='List of particle distributions to plot. Choose from X, Y, Z, Xvel, Yvel, Zvel. If not provided, no distributions will be plotted.')
    
    # arguments for projections. The user can choose from XY, XZ, YZ, XYvel, XZvel, YZvel
    parser.add_argument('--projections', type=str, nargs='*', choices=['XY', 'XZ', 'YZ', 'XYvel', 'XZvel', 'YZvel'],
                        help='List of projections to plot. Choose from XY, XZ, YZ, XYvel, XZvel, YZvel. If not provided, no projections will be plotted.')

    # argument for power spectra plot
    parser.add_argument('--power_spectra', action='store_true', help='If set, compute and plot power spectra of the fields.')

    # argument for "use positions" (instead of displacements) option
    parser.add_argument('--use_positions', action='store_true', help='If set, use positions instead of displacements for plotting distributions and projections.')

    # argument for boxsize, default is 1.4925373e+09 (in kpc)
    parser.add_argument('--boxsize', type=float, default=1.4925373e+09, help='Box size in kpc. Default is 1.4925373e+09.')


    args = parser.parse_args()

    # create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)


    # lr patterns will be something like 'a/b/dataset_name/train/LR/*', get dataset_name
    dataset_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(args.lr_patterns))))

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


    lr_files = glob.glob(args.lr_patterns)
    hr_files = glob.glob(args.hr_patterns)

    if len(hr_files) == 0:
        hr_files = [None] * len(lr_files)
    elif len(lr_files) == 0:
        lr_files = [None] * len(hr_files)

    assert len(lr_files) == len(hr_files), "The number of low-resolution and high-resolution files must match. Alternatively, you can provide only one type of files (HR or LR)."

    for lr_file, hr_file in zip(lr_files, hr_files):
        # load the low-resolution and high-resolution data

        # get filename
        filename_lr = os.path.basename(lr_file) if lr_file else 'None'
        filename_hr = os.path.basename(hr_file) if hr_file else 'None'

        lr_dis_vel = np.load(lr_file) if lr_file else None
        hr_dis_vel = np.load(hr_file) if hr_file else None

        # get Ng_HR and Ng_LR from the shapes
        Ng_LR = lr_dis_vel.shape[1] if lr_dis_vel is not None else None
        Ng_HR = hr_dis_vel.shape[1] if hr_dis_vel is not None else None

        if args.use_positions:
            lr_dis_vel = dis2pos(lr_dis_vel, args.boxsize, Ng_LR) if lr_dis_vel is not None else None
            hr_dis_vel = dis2pos(hr_dis_vel, args.boxsize, Ng_HR) if hr_dis_vel is not None else None

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
                        ax_lr.set_title(f'LR {field} Distribution')
                        ax_lr.set_xlabel(field)
                        ax_lr.set_ylabel('Frequency')
                        #ax_lr.legend()
                    if hr_dis_vel is not None:
                        ax_hr.hist(hr_dis_vel[field_index, :].flatten(), bins=100, alpha=0.5, label=f'HR {field}', color='orange')
                        ax_hr.set_title(f'HR {field} Distribution')
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
