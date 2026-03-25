import argparse
import os
import numpy as np
import glob
import datetime
import gc
import torch

import matplotlib.pyplot as plt
from displacement import dis2pos, dis2posLR_average_downsample
from downsampling import average_downsample, random_sample, downsample_tricubic
from map2map.models import power

def str_list(s): # from args.py
    return s.split(',')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plots statistics and visualizations of preprocessed data.')
    parser.add_argument('--lr-patterns', type=str_list, required=True, help='Regular expression for low-resolution input folders.')
    parser.add_argument('--hr-patterns', type=str_list, required=True, help='Regular expression for high-resolution input folders.')
    parser.add_argument('--output-folder', type=str, required=True, help='Output folder for statistics and visualizations.')

    # arguments for histograms. The user can choose from X, Y, Z, Xvel, Yvel, Zvel
    parser.add_argument('--histograms', type=str, nargs='*', choices=['X', 'Y', 'Z', 'Xvel', 'Yvel', 'Zvel'],
                        help='List of histograms to plot. Choose from X, Y, Z, Xvel, Yvel, Zvel. If not provided, no histograms will be plotted.')
    
    # arguments for projections. The user can choose from XY, XZ, YZ, XYvel, XZvel, YZvel
    parser.add_argument('--scatter-plots', type=str, nargs='*', choices=['XY', 'XZ', 'YZ', 'XYvel', 'XZvel', 'YZvel', 'XYZ-3D', 'vel-3D'],
                        help='List of scatter plots to plot. Choose from XY, XZ, YZ, XYvel, XZvel, YZvel, XYZ-3D, vel-3D. If not provided, no scatter plots will be plotted.')

    # argument for power spectra plot
    parser.add_argument('--power-spectra', type=str, nargs='*', choices=['XYZ', 'vel'],
                        help='If provided, power spectra will be plotted for the specified fields. Choose from XYZ (for displacements) and XvelYvelZvel (for velocities). If not provided, no power spectra will be plotted.')
    # argument for "use positions" (instead of displacements) option
    parser.add_argument('--use-positions', action='store_true', help='If set, use positions instead of displacements for plotting distributions and projections.')

    #argument for cropping positions (if use_positions is set)
    parser.add_argument('--positions-crop', type=int, default=None, help='If set, crop positions to this size (e.g., 50). If not set, no cropping is done.')
    parser.add_argument('--field-crop', type = int, default = None)

    # argument for boxsize, default is 1000.0 (in Mpc/h)
    parser.add_argument('--boxsize', type=float, default=1000.0, help='Box size in kpc. Default is 1000.0.')

    # argument for downsampling option
    parser.add_argument('--downsampling-factor', type=int, default=1, help='If set to an integer > 1, downsample the data by this factor before plotting. Default is 1 (no downsampling).')
    parser.add_argument('--ignore-downsample-for-power-spectra', action='store_true', help='If set, do not downsample for power spectra calculation, only for histograms and projections.')

    #add arguments for cropping
    parser.add_argument('--crop-LR', type=int, default=None, help='If set, crop LR data to this size (e.g., 64). If not set, no cropping is done.')
    parser.add_argument('--crop-HR', type=int, default=None, help='If set, crop HR data to this size (e.g., 128). If not set, no cropping is done.')

    parser.add_argument('--downsampling-method', type=str, choices=['random', 'tricubic', 'average'], default='random', help='Method for downsampling the data. Default is "random".')


    args = parser.parse_args()

    downsampling_function = None
    if args.downsampling_factor > 1:
        if args.downsampling_method == 'random':
            downsampling_function = random_sample
        elif args.downsampling_method == 'tricubic':
            downsampling_function = downsample_tricubic
        elif args.downsampling_method == 'average':
            downsampling_function = average_downsample
    

    # create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)


    # lr patterns will be something like 'a/b/dataset_name/train/LR/*', get dataset_name
    dataset_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(args.lr_patterns[0]))))

    # create a subfolder for the dataset in the output folder
    output_dataset_folder = os.path.join(args.output_folder, dataset_name)
    os.makedirs(output_dataset_folder, exist_ok=True)

    # create 3 subfolders for the dataset: histograms, projections, and power_spectra
    os.makedirs(os.path.join(output_dataset_folder, 'histograms'), exist_ok=True)
    os.makedirs(os.path.join(output_dataset_folder, 'scatter_plots'), exist_ok=True)
    os.makedirs(os.path.join(output_dataset_folder, 'power_spectra'), exist_ok=True)

    # print arguments
    print(f"Script started at {datetime.datetime.now()}\n")
    print(f"Low-resolution patterns: {args.lr_patterns}")
    print(f"High-resolution patterns: {args.hr_patterns}")
    print(f"Output folder: {output_dataset_folder}")
    print(f"Histograms to plot: {args.histograms if args.histograms else 'None'}")
    print(f"Scatter plots to plot: {args.scatter_plots if args.scatter_plots else 'None'}")
    print(f"Power spectra: {args.power_spectra if args.power_spectra else 'None'}")
    print(f"Using positions instead of displacements: {'Yes' if args.use_positions else 'No'}")
    print(f"Cropping positions to: {args.positions_crop if args.positions_crop else 'No cropping'}")
    print(f"Box size: {args.boxsize} Mpc/h")
    print(f"Downsampling factor: {args.downsampling_factor}")
    print(f"Ignore downsampling for power spectra: {'Yes' if args.ignore_downsample_for_power_spectra else 'No'}\n")

    #args.lr_patterns is a str_list, we assume it has 2 elements: one for dis and one for vel

    lr_files_dis = glob.glob(args.lr_patterns[0]) if len(args.lr_patterns) > 0 else []
    lr_files_vel = glob.glob(args.lr_patterns[1]) if len(args.lr_patterns) > 1 else []
    hr_files_dis = glob.glob(args.hr_patterns[0]) if len(args.hr_patterns) > 0 else []
    hr_files_vel = glob.glob(args.hr_patterns[1]) if len(args.hr_patterns) > 1 else []

    #get max length from the 4 lists
    max_len = max(len(lr_files_dis), len(lr_files_vel), len(hr_files_dis), len(hr_files_vel))

    #replace empty arrays with lists of None of the same length as the other list
    if len(lr_files_dis) == 0:
        lr_files_dis = [None] * max_len
    if len(lr_files_vel) == 0:
        lr_files_vel = [None] * max_len
    if len(hr_files_dis) == 0:
        hr_files_dis = [None] * max_len
    if len(hr_files_vel) == 0:
        hr_files_vel = [None] * max_len

    print("lr_files_dis:", lr_files_dis)
    print("lr_files_vel:", lr_files_vel)
    print("hr_files_dis:", hr_files_dis)
    print("hr_files_vel:", hr_files_vel)

    for lr_file_dis, lr_file_vel, hr_file_dis, hr_file_vel in zip(lr_files_dis, lr_files_vel, hr_files_dis, hr_files_vel):
        gc.collect()
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

        print(f"\nProcessing {filename_lr} vs {filename_hr}")


        """
        posdis = 'positions_' if args.use_positions else 'displacements_'
            fig.savefig(os.path.join(output_dataset_folder, 'distributions', f'{posdis}{filename_lr}_{args.particle_distributions}_vs_{filename_hr}.png'))
            plt.close(fig)
        """

        posdis = 'positions' if args.use_positions else 'displacements'

        #check if the plot already exists for all the required distributions (histograms, projections, power spectra)
        hist_exists = False
        scatter_exists = False
        ps_exists = False
        if args.histograms:
            if os.path.exists(os.path.join(output_dataset_folder, 'histograms', f'{args.histograms}_{posdis}_{args.positions_crop}_{args.downsampling_factor}_{filename_lr}_vs_{filename_hr}.png')):
                hist_exists = True
        if args.scatter_plots:
            if os.path.exists(os.path.join(output_dataset_folder, 'scatter_plots', f'{args.scatter_plots}_{posdis}_{args.positions_crop}_{args.downsampling_factor}_{filename_lr}_vs_{filename_hr}.png')):
                scatter_exists = True
        if args.power_spectra:
            if os.path.exists(os.path.join(output_dataset_folder, 'power_spectra', f'{args.power_spectra}_{posdis}_{args.positions_crop}_{args.downsampling_factor}_{filename_lr}_vs_{filename_hr}.png')):
                ps_exists = True

        if hist_exists and scatter_exists and ps_exists:
            print(f"Requested histograms/scatter_plots/power_spectra for {filename_lr} vs {filename_hr} already exist, skipping...")
            continue

        if args.field_crop is None:
            lr_dis = np.load(lr_file_dis) if lr_file_dis else None
            lr_vel = np.load(lr_file_vel) if lr_file_vel else None
            hr_dis = np.load(hr_file_dis) if hr_file_dis else None
            hr_vel = np.load(hr_file_vel) if hr_file_vel else None
        else:
            #use memory mapping to read from (channels, 0,0,0) to (channels, field_crop, field_crop, field_crop)
            lr_dis = np.load(lr_file_dis, mmap_mode='r')[:, :args.field_crop, :args.field_crop, :args.field_crop] if lr_file_dis else None
            lr_vel = np.load(lr_file_vel, mmap_mode='r')[:, :args.field_crop, :args.field_crop, :args.field_crop] if lr_file_vel else None
            hr_dis = np.load(hr_file_dis, mmap_mode='r')[:, :args.field_crop*2, :args.field_crop*2, :args.field_crop*2] if hr_file_dis else None
            hr_vel = np.load(hr_file_vel, mmap_mode='r')[:, :args.field_crop*2, :args.field_crop*2, :args.field_crop*2] if hr_file_vel else None

            args.boxsize = args.boxsize * (args.field_crop / 512) 

        #Get Ng for LR and HR
        if lr_dis is not None or lr_vel is not None:
            #take the 2nd, 3rd and 4th elements of the shape of lr_dis or lr_vel (if it exists)
            Ng_LR = lr_dis.shape[1] if lr_dis is not None else lr_vel.shape[1]
        else:
            Ng_LR = None
        
        

        if hr_dis is not None or hr_vel is not None:
            #take the 2nd, 3rd and 4th elements of the shape of hr_dis or hr_vel (if it exists)
            Ng_HR = hr_dis.shape[1] if hr_dis is not None else hr_vel.shape[1]
        else:
            Ng_HR = None

        print(f"Ng_LR: {Ng_LR}, Ng_HR: {Ng_HR}")

        if args.downsampling_factor != 1 and not args.ignore_downsample_for_power_spectra: #downsample before
            if Ng_LR is not None:
                print(f"Downsampling LR data from {Ng_LR}^3 to {Ng_LR // args.downsampling_factor}^3")
                if lr_dis is not None:
                    lr_dis = downsampling_function(lr_dis, factor = args.downsampling_factor)
                if lr_vel is not None:
                    lr_vel = downsampling_function(lr_vel, factor = args.downsampling_factor)
                Ng_LR = Ng_LR // args.downsampling_factor
            if Ng_HR is not None:
                print(f"Downsampling HR data from {Ng_HR}^3 to {Ng_HR // args.downsampling_factor}^3")
                if hr_dis is not None:
                    hr_dis = downsampling_function(hr_dis, factor = args.downsampling_factor)
                if hr_vel is not None:
                    hr_vel = downsampling_function(hr_vel, factor = args.downsampling_factor)
                Ng_HR = Ng_HR // args.downsampling_factor
            gc.collect()

        if args.use_positions:
            if lr_dis is not None:
                lr_dis = dis2posLR_average_downsample(lr_dis, args.boxsize, Ng_LR, Ng_HR if Ng_HR is not None else 1024) #NOTE: assume HR is 1024
                if args.positions_crop is not None:
                    #take only positions such that x <= positions_crop, y <= positions_crop, z <= positions_crop
                    lr_dis = lr_dis[:, (lr_dis[0, :] <= args.positions_crop) & (lr_dis[1, :] <= args.positions_crop) & (lr_dis[2, :] <= args.positions_crop)]
                    #now lr_dir has shape (3, N'), with N' <= N. Add a pad that brings the second dimension to the first greater cube, then reshape to (3, Ncbrt, Ncbrt, Ncbrt)
                    N = lr_dis.shape[1]
                    Ncbrt = int(np.ceil(N ** (1/3)))
                    pad_size = Ncbrt**3 - N
                    lr_dis = np.pad(lr_dis, ((0,0),(0,pad_size)), mode='wrap')
                    lr_dis = lr_dis.reshape(3, Ncbrt, Ncbrt, Ncbrt)

                    Ng_LR = Ncbrt #update Ng_LR
                    
            if hr_dis is not None:
                hr_dis = dis2pos(hr_dis, args.boxsize, Ng_HR)
                if args.positions_crop is not None:
                    #take only positions such that x <= positions_crop, y <= positions_crop, z <= positions_crop
                    hr_dis = hr_dis[:, (hr_dis[0, :] <= args.positions_crop) & (hr_dis[1, :] <= args.positions_crop) & (hr_dis[2, :] <= args.positions_crop)]
                    #now hr_dir has shape (3, N'), with N' <= N. Add a pad that brings the second dimension to the first greater cube, then reshape to (3, Ncbrt, Ncbrt, Ncbrt)
                    N = hr_dis.shape[1]
                    Ncbrt = int(np.ceil(N ** (1/3)))
                    pad_size = Ncbrt**3 - N
                    hr_dis = np.pad(hr_dis, ((0,0),(0,pad_size)), mode='wrap')
                    hr_dis = hr_dis.reshape(3, Ncbrt, Ncbrt, Ncbrt)

                    Ng_HR = Ncbrt #update Ng_HR
            
            if lr_vel is not None and args.positions_crop is not None: #TODO: not really correct
                #crop lr_vel to the same number of particles as lr_dis
                lr_vel = lr_vel[:, :lr_dis.shape[1]]
                lr_vel = lr_vel.reshape(3, Ncbrt, Ncbrt, Ncbrt)
            if hr_vel is not None and args.positions_crop is not None: #TODO: not really correct
                #crop hr_vel to the same number of particles as hr_dis
                hr_vel = hr_vel[:, :hr_dis.shape[1]]
                hr_vel = hr_vel.reshape(3, Ncbrt, Ncbrt, Ncbrt)
            



        n_cols = 2 if (lr_dis is not None or lr_vel is not None) and (hr_dis is not None or hr_vel is not None) else 1
        
        #variable to store if the only column is LR or HR
        type_ = True if (lr_dis is not None or lr_vel is not None) else False #True for LR, False for HR
        
        if args.power_spectra and not ps_exists:
            n_rows_ps = 0

            if 'XYZ' in args.power_spectra:
                n_rows_ps += 1
            if 'vel' in args.power_spectra:
                n_rows_ps += 1

            print(f"Plotting power spectra with {n_rows_ps} rows")

            fig, axes = plt.subplots(n_rows_ps, 1, figsize=(10 * 1, 5 * n_rows_ps), squeeze=False)

            #set title
            if n_cols == 2: #use previously defined n_cols
                fig.suptitle(f'Power Spectra: LR (Ng={Ng_LR}) vs HR (Ng={Ng_HR})', fontsize=16)
            else:
                if type_:
                    fig.suptitle(f'Power Spectra: LR (Ng={Ng_LR})', fontsize=16)
                else:
                    fig.suptitle(f'Power Spectra: HR (Ng={Ng_HR})', fontsize=16)

            #do first row
            if n_rows_ps > 0:
                fields = []
                labels = []
                if lr_dis is not None:
                    #convert to torch tensor, add batch dimension as first dimension (1)
                    lr_dis_torch = torch.tensor(lr_dis).unsqueeze(0)
                    
                    fields.append(lr_dis_torch)
                    labels.append(f'LR {posdis[:-1]}')
                if hr_dis is not None:
                    #convert to torch tensor, add batch dimension as first dimension (1)
                    hr_dis_torch = torch.tensor(hr_dis).unsqueeze(0)    

                    fields.append(hr_dis_torch)
                    labels.append(f'HR {posdis[:-1]}')

                for field, label in zip(fields, labels):
                    k, P, _ = power(field)
                    k = k.cpu().numpy()
                    P = P.cpu().numpy()

                    #plot
                    axes[0, 0].loglog(k, P , label=label, alpha=0.7)
                    axes[0, 0].legend()
                    axes[0, 0].set_xlabel('unnormalized wavenumber')
                    axes[0, 0].set_ylabel('unnormalized power')

            if n_rows_ps > 1:
                fields = []
                labels = []
                if lr_vel is not None:
                    #convert to torch tensor, add batch dimension as first dimension (1)
                    lr_vel_torch = torch.tensor(lr_vel).unsqueeze(0)
                    fields.append(lr_vel_torch)
                    labels.append(f'LR vel')
                if hr_vel is not None:
                    #convert to torch tensor, add batch dimension as first dimension (1)
                    hr_vel_torch = torch.tensor(hr_vel).unsqueeze(0)
                    fields.append(hr_vel_torch)
                    labels.append(f'HR vel')

                for field, label in zip(fields, labels):
                    k, P, _ = power(field)
                    k = k.cpu().numpy()
                    P = P.cpu().numpy()

                    #plot
                    axes[1, 0].loglog(k, P , label=label, alpha=0.7)
                    axes[1, 0].legend()
                    axes[1, 0].set_xlabel('unnormalized wavenumber')
                    axes[1, 0].set_ylabel('unnormalized power')
            
            # save the figure
            fig.savefig(os.path.join(output_dataset_folder, 'power_spectra', f'{args.power_spectra}_{posdis}_{args.positions_crop}_{args.downsampling_factor}_{filename_lr}_vs_{filename_hr}.png'))
            plt.close(fig)


        if args.downsampling_factor != 1 and args.ignore_downsample_for_power_spectra: #downsample after
            if Ng_LR is not None:
                print(f"Downsampling LR data from {Ng_LR}^3 to {Ng_LR // args.downsampling_factor}^3")
                if lr_dis is not None:
                    lr_dis = downsampling_function(lr_dis, factor = args.downsampling_factor)
                if lr_vel is not None:
                    lr_vel = downsampling_function(lr_vel, factor = args.downsampling_factor)
                Ng_LR = Ng_LR // args.downsampling_factor
            if Ng_HR is not None:
                print(f"Downsampling HR data from {Ng_HR}^3 to {Ng_HR // args.downsampling_factor}^3")
                if hr_dis is not None:
                    hr_dis = downsampling_function(hr_dis, factor = args.downsampling_factor)
                if hr_vel is not None:
                    hr_vel = downsampling_function(hr_vel, factor = args.downsampling_factor)
                Ng_HR = Ng_HR // args.downsampling_factor
            gc.collect()


        # plot particle distributions if specified
        if args.histograms and not hist_exists:
            # do a subplot with 2 columns if both LR and HR are available, otherwise just one column
            n_rows_pd = len(args.histograms)
            fig, axes = plt.subplots(n_rows_pd, n_cols, figsize=(10 * n_cols, 5 * n_rows_pd), squeeze=False)


            #set title for the figure, add Ng_LR and Ng_HR if available
            if n_cols == 2:
                fig.suptitle(f'Particle Distributions: LR (Ng={Ng_LR}) vs HR (Ng={Ng_HR})', fontsize=16)
            else:
                if type_:
                    fig.suptitle(f'Particle Distributions: LR (Ng={Ng_LR})', fontsize=16)
                else:
                    fig.suptitle(f'Particle Distributions: HR (Ng={Ng_HR})', fontsize=16)

            for i, field in enumerate(args.histograms):
                # map field to index
                field_index = {'X': 0, 'Y': 1, 'Z': 2, 'Xvel': 0, 'Yvel': 1, 'Zvel': 2}[field]
                field_var = 'dis' if field in ['X', 'Y', 'Z'] else 'vel'

                if n_cols == 1:
                    ax = axes[i, 0]

                    

                    if type_:
                        if lr_dis is not None and field_var == 'dis':
                            data = lr_dis[field_index, :]
                        elif lr_vel is not None and field_var == 'vel':
                            data = lr_vel[field_index, :]
                    else:
                        if hr_dis is not None and field_var == 'dis':
                            data = hr_dis[field_index, :]
                        elif hr_vel is not None and field_var == 'vel':
                            data = hr_vel[field_index, :]
                    
                    ax.hist(data.flatten(), bins=100, alpha=0.5, label=f'{"LR" if type_ else "HR"} {field}', color='blue' if type_ else 'orange')


                    #if lr_dis_vel is not None:
                    #    ax.hist(lr_dis_vel[field_index, :].flatten(), bins=100, alpha=0.5, label=f'LR {field}', color='blue')
                    #if hr_dis_vel is not None:
                    #    ax.hist(hr_dis_vel[field_index, :].flatten(), bins=100, alpha=0.5, label=f'HR {field}', color='orange')
                    ax.set_title(f'{field} Distribution')
                    ax.set_xlabel(field)
                    ax.set_ylabel('Frequency')
                    #ax.legend()

                else:
                    ax_lr = axes[i, 0]
                    ax_hr = axes[i, 1]

                    if lr_dis is not None and field_var == 'dis':
                        data = lr_dis[field_index, :]
                        ax_lr.hist(data.flatten(), bins=100, alpha=0.5, label=f'LR {field}', color='blue')
                    elif lr_vel is not None and field_var == 'vel':
                        data = lr_vel[field_index, :]
                        ax_lr.hist(data.flatten(), bins=100, alpha=0.5, label=f'LR {field}', color='blue')
                    
                    if hr_dis is not None and field_var == 'dis':   
                        data = hr_dis[field_index, :]
                        ax_hr.hist(data.flatten(), bins=100, alpha=0.5, label=f'HR {field}', color='orange')
                    elif hr_vel is not None and field_var == 'vel':
                        data = hr_vel[field_index, :]
                        ax_hr.hist(data.flatten(), bins=100, alpha=0.5, label=f'HR {field}', color='orange')

                    ax_lr.set_title(f'LR {field} Distribution')
                    ax_lr.set_xlabel(field)
                    ax_lr.set_ylabel('Frequency')

                    ax_hr.set_title(f'HR {field} Distribution')
                    ax_hr.set_xlabel(field)
                    ax_hr.set_ylabel('Frequency')

                    #ax_lr.legend()

                

            # save the figure
            fig.savefig(os.path.join(output_dataset_folder, 'histograms', f'{args.histograms}_{posdis}_{args.positions_crop}_{args.downsampling_factor}_{filename_lr}_vs_{filename_hr}.png'))
            plt.close(fig)

        if args.use_positions and args.positions_crop is not None:
            s_LR = 1 * 50/args.positions_crop
            s_HR = 1/8 * 50/args.positions_crop
            alpha = 1 * 50/args.positions_crop
        else:
            s_LR = 0.2/(Ng_LR) if Ng_LR is not None else 0.002
            s_HR = 0.1/(Ng_HR) if Ng_HR is not None else 0.001
            alpha = 0.2

        

        # plot scatter plots if specified
        if args.scatter_plots and not scatter_exists:
            #2d projections and 3d scatter plots
            n_rows_sp = len(args.scatter_plots)
            fig, axes = plt.subplots(n_rows_sp, n_cols, figsize=(15 * n_cols, 7 * n_rows_sp), squeeze=False)
            #set title for the figure, add Ng_LR and Ng_HR if available
            if n_cols == 2:
                fig.suptitle(f'Scatter Plots: LR (Ng={Ng_LR}) vs HR (Ng={Ng_HR})', fontsize=16)
            else:
                if type_:
                    fig.suptitle(f'Scatter Plots: LR (Ng={Ng_LR})', fontsize=16)
                else:
                    fig.suptitle(f'Scatter Plots: HR (Ng={Ng_HR})', fontsize=16)
            for i, projection in enumerate(args.scatter_plots):
                if projection in ['XY', 'XZ', 'YZ']:
                    x_index, y_index = {'XY': (0, 1), 'XZ': (0, 2), 'YZ': (1, 2)}[projection]
                    x_label, y_label = projection[0], projection[1]
                    if n_cols == 1:
                        ax = axes[i, 0]

                        if type_:
                            if lr_dis is not None:
                                x = lr_dis[x_index, :]
                                y = lr_dis[y_index, :]
                            elif lr_vel is not None:
                                x = lr_vel[x_index, :]
                                y = lr_vel[y_index, :]
                        else:
                            if hr_dis is not None:
                                x = hr_dis[x_index, :]
                                y = hr_dis[y_index, :]
                            elif hr_vel is not None:
                                x = hr_vel[x_index, :]
                                y = hr_vel[y_index, :]

                        ax.scatter(x.flatten(), y.flatten(), s= (s_LR if type_ else s_HR) , alpha=alpha)
                        ax.set_title(f'{projection} Projection {"LR" if type_ else "HR"}')
                        ax.set_xlabel(x_label)
                        ax.set_ylabel(y_label)
                        ax.set_aspect('equal', adjustable='box')

                    else:
                        ax_lr = axes[i, 0]
                        ax_hr = axes[i, 1]

                        if lr_dis is not None:
                            x = lr_dis[x_index, :]
                            y = lr_dis[y_index, :]
                            ax_lr.scatter(x.flatten(), y.flatten(), s=s_LR, alpha=alpha, color='blue')
                        elif lr_vel is not None:
                            x = lr_vel[x_index, :]
                            y = lr_vel[y_index, :]
                            ax_lr.scatter(x.flatten(), y.flatten(), s=s_LR, alpha=alpha, color='blue')

                        if hr_dis is not None:
                            x = hr_dis[x_index, :]
                            y = hr_dis[y_index, :]
                            ax_hr.scatter(x.flatten(), y.flatten(), s=s_HR, alpha=alpha, color='orange')
                        elif hr_vel is not None:
                            x = hr_vel[x_index, :]
                            y = hr_vel[y_index, :]
                            ax_hr.scatter(x.flatten(), y.flatten(), s=s_HR, alpha=alpha, color='orange')

                        ax_lr.set_title(f'LR {projection} Projection')
                        ax_lr.set_xlabel(x_label)
                        ax_lr.set_ylabel(y_label)
                        ax_lr.set_aspect('equal', adjustable='box')
                        ax_hr.set_title(f'HR {projection} Projection')
                        ax_hr.set_xlabel(x_label)
                        ax_hr.set_ylabel(y_label)
                        ax_hr.set_aspect('equal', adjustable='box')
                elif projection in ['XYvel', 'XZvel', 'YZvel']:
                    x_index, y_index = {'XYvel': (0, 1), 'XZvel': (0, 2), 'YZvel': (1, 2)}[projection]
                    x_label, y_label = projection[0], projection[1]
                    if n_cols == 1:
                        ax = axes[i, 0]

                        if type_:
                            if lr_vel is not None:
                                x = lr_vel[x_index, :]
                                y = lr_vel[y_index, :]
                        else:
                            if hr_vel is not None:
                                x = hr_vel[x_index, :]
                                y = hr_vel[y_index, :]

                        ax.scatter(x.flatten(), y.flatten(), s=(s_LR if type_ else s_HR), alpha=alpha)
                        ax.set_title(f'{projection} Projection {"LR" if type_ else "HR"}')
                        ax.set_xlabel(x_label)
                        ax.set_ylabel(y_label)
                        ax.set_aspect('equal', adjustable='box')

                    else:
                        ax_lr = axes[i, 0]
                        ax_hr = axes[i, 1]

                        if lr_vel is not None:
                            x = lr_vel[x_index, :]
                            y = lr_vel[y_index, :]
                            ax_lr.scatter(x.flatten(), y.flatten(), s=s_LR, alpha=alpha, color='blue')
                        
                        if hr_vel is not None:
                            x = hr_vel[x_index, :]
                            y = hr_vel[y_index, :]
                            ax_hr.scatter(x.flatten(), y.flatten(), s=s_HR, alpha=alpha, color='orange')

                        ax_lr.set_title(f'LR {projection} Projection')
                        ax_lr.set_xlabel(x_label)
                        ax_lr.set_ylabel(y_label)
                        ax_lr.set_aspect('equal', adjustable='box')
                        ax_hr.set_title(f'HR {projection} Projection')
                        ax_hr.set_xlabel(x_label)
                        ax_hr.set_ylabel(y_label)
                        ax_hr.set_aspect('equal', adjustable='box')
                elif projection in ['XYZ-3D', 'vel-3D']:    
                    if n_cols == 1:
                        ax = axes[i, 0]
                        ax = fig.add_subplot(n_rows_sp, 1, i+1, projection='3d')

                        if type_:
                            if lr_dis is not None and projection == 'XYZ-3D':
                                x = lr_dis[0, :]
                                y = lr_dis[1, :]
                                z = lr_dis[2, :]
                            elif lr_vel is not None and projection == 'vel-3D':
                                x = lr_vel[0, :]
                                y = lr_vel[1, :]
                                z = lr_vel[2, :]
                        else:
                            if hr_dis is not None and projection == 'XYZ-3D':
                                x = hr_dis[0, :]
                                y = hr_dis[1, :]
                                z = hr_dis[2, :]
                            elif hr_vel is not None and projection == 'vel-3D':
                                x = hr_vel[0, :]
                                y = hr_vel[1, :]
                                z = hr_vel[2, :]

                        ax.scatter(x.flatten(), y.flatten(), z.flatten(), s=(s_LR if type_ else s_HR), alpha=alpha)
                        ax.set_title(f'3D Scatter Plot {"LR" if type_ else "HR"}')
                        ax.set_xlabel('X' if projection == 'XYZ-3D' else 'Xvel')
                        ax.set_ylabel('Y' if projection == 'XYZ-3D' else 'Yvel')
                        ax.set_zlabel('Z' if projection == 'XYZ-3D' else 'Zvel')
                        ax.set_box_aspect([1,1,1])
                    else:
                        ax_lr = axes[i, 0]
                        ax_hr = axes[i, 1]
                        ax_lr = fig.add_subplot(n_rows_sp, 2, i*2+1, projection='3d')
                        ax_hr = fig.add_subplot(n_rows_sp, 2, i*2+2, projection='3d')

                        if lr_dis is not None and projection == 'XYZ-3D':
                            x = lr_dis[0, :]
                            y = lr_dis[1, :]
                            z = lr_dis[2, :]
                            ax_lr.scatter(x.flatten(), y.flatten(), z.flatten(), s=s_LR, alpha=alpha, color='blue')
                        elif lr_vel is not None and projection == 'vel-3D':
                            x = lr_vel[0, :]
                            y = lr_vel[1, :]
                            z = lr_vel[2, :]
                            ax_lr.scatter(x.flatten(), y.flatten(), z.flatten(), s=s_LR, alpha=alpha, color='blue')
                        
                        if hr_dis is not None and projection == 'XYZ-3D':
                            x = hr_dis[0, :]
                            y = hr_dis[1, :]
                            z = hr_dis[2, :]
                            ax_hr.scatter(x.flatten(), y.flatten(), z.flatten(), s=s_HR, alpha=alpha, color='orange')
                        elif hr_vel is not None and projection == 'vel-3D':
                            x = hr_vel[0, :]
                            y = hr_vel[1, :]
                            z = hr_vel[2, :]
                            ax_hr.scatter(x.flatten(), y.flatten(), z.flatten(), s=s_HR, alpha=alpha, color='orange')

                        ax_lr.set_title(f'LR 3D Scatter Plot')
                        ax_lr.set_xlabel('X' if projection == 'XYZ-3D' else 'Xvel')
                        ax_lr.set_ylabel('Y' if projection == 'XYZ-3D' else 'Yvel')
                        ax_lr.set_zlabel('Z' if projection == 'XYZ-3D' else 'Zvel')
                        ax_lr.set_box_aspect([1,1,1])
                        ax_hr.set_title(f'HR 3D Scatter Plot')
                        ax_hr.set_xlabel('X' if projection == 'XYZ-3D' else 'Xvel')
                        ax_hr.set_ylabel('Y' if projection == 'XYZ-3D' else 'Yvel')
                        ax_hr.set_zlabel('Z' if projection == 'XYZ-3D' else 'Zvel')
                        ax_hr.set_box_aspect([1,1,1])
            # save the figure
            fig.savefig(os.path.join(output_dataset_folder, 'scatter_plots', f'{args.scatter_plots}_{posdis}_{args.positions_crop}_{args.downsampling_factor}_{filename_lr}_vs_{filename_hr}.png'))
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
