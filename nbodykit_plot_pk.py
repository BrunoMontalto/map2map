import argparse
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['lines.linewidth'] = 1.5
from nbodykit.lab import *
import os
from glob import glob


def find_k_max_within_percent_error(k_vals, pk_sr, pk_hr, threshold=0.01):
    """
    Returns the maximum value of k such that the percent error between pk_sr and ph_hr is under threshold.
    """
    # Interpolate pk_hr on k_sr if necessary
    pk_hr_interp = np.interp(k_vals, k_hr, pk_hr)

    relative_error = np.abs(pk_sr - pk_hr_interp) / pk_hr_interp

    # Find indices with error under the threshold
    valid_indices = np.where(relative_error < threshold)[0]

    if len(valid_indices) == 0:
        return None  # No k satisfies the condition

    # Return the maximum value of k that satisfies the condition
    return k_vals[valid_indices[-1]]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Loads gadget files and computes power spectrum.')
    
    parser.add_argument('--lr-file', type=str, required=True, help='Path to low-resolution Gadget file')
    parser.add_argument('--hr-file', type=str, required=True, help='Path to high-resolution Gadget file')
    parser.add_argument(
        "--sr-file",
        "--sr-files",
        nargs="+",
        type=str,
        dest="sr_file",
        default = []
    )
    parser.add_argument("--sr-labels", type=str, nargs="+", default=["SR"])
    parser.add_argument('--sr-colors', type=str, nargs='+', default=["magenta", "red", "yellow"])
    parser.add_argument("--compute-extra-statistics", action='store_true')

    parser.add_argument('--output-folder', type=str, default='plots_nbodykit', help='Output plot file path')

    parser.add_argument('--boxsize', type=float, default=1000.0)
    parser.add_argument('--Ng-LR', type=int, default=512)
    parser.add_argument('--Ng-HR', type=int, default=512)

    parser.add_argument('--dimensionless', action='store_true')
    parser.add_argument(
    '--plot-nyq',
    action='store_true'
    )
    parser.add_argument(
    '--plot-shotnoise',
    action='store_true',
    help='Plot the Poisson shot-noise limit expected from particle sampling'
    )

    args = parser.parse_args()
    print(vars(args))

    if len(args.sr_file) > 1 and len(args.sr_labels) == 1:
            args.sr_labels = ["SR" + str(i+1) for i in range(len(args.sr_file))]

    NG_LR = args.Ng_LR
    NG_HR = args.Ng_HR
    BOXSIZE = args.boxsize

    # Load with nbodykit
    no_files = True
    if args.lr_file:
        no_files = False


        cat_lr = Gadget1Catalog(sorted(glob(args.lr_file)))

        # Compute power spectrum
        mesh_lr = cat_lr.to_mesh(Nmesh=NG_LR, BoxSize=BOXSIZE, resampler='tsc', interlaced=True)
        r_lr = FFTPower(mesh_lr, mode='1d')#, kmin= 0.049)
        pk_lr = r_lr.power
    if args.hr_file:
        no_files = False

        cat_hr = Gadget1Catalog(sorted(glob(args.hr_file)))

        mesh_hr = cat_hr.to_mesh(Nmesh=NG_HR, BoxSize=BOXSIZE, resampler='tsc',interlaced=True)
        r_hr = FFTPower(mesh_hr, mode='1d')#, kmin = 0.024)
        pk_hr = r_hr.power
    if args.sr_file:
        pk_sr = {}
        no_files = False

        for i, path in enumerate(args.sr_file):

            cat_sr = Gadget1Catalog(sorted(glob(path)))

            mesh_sr = cat_sr.to_mesh(Nmesh=NG_HR, BoxSize=BOXSIZE, resampler='tsc',interlaced=True)
            r_sr = FFTPower(mesh_sr, mode='1d')#, kmin = 0.024)
            pk_sr[args.sr_labels[i]] = r_sr.power
    

    if no_files:
        print("no files provided")
        exit(1)
    

    # Visualization
    if args.lr_file:
        k_lr = pk_lr['k']
        shotnoise_lr = pk_lr.attrs['shotnoise']
        pk_lr = pk_lr['power'].real - shotnoise_lr

        if args.dimensionless:
            pk_lr = (k_lr**3 * pk_lr) / (2 * np.pi**2)

    if args.hr_file:
        k_hr = pk_hr['k']
        shotnoise_hr = pk_hr.attrs['shotnoise']
        pk_hr = pk_hr['power'].real - shotnoise_hr

        if args.dimensionless:
            pk_hr = (k_hr**3 * pk_hr) / (2 * np.pi**2) 

    if args.sr_file:
        k_sr = {}
        for label in pk_sr:
            k_sr[label] = pk_sr[label]['k']
            shotnoise_sr = pk_sr[label].attrs['shotnoise']
            pk_sr[label] = pk_sr[label]['power'].real - shotnoise_sr

            if args.dimensionless:
                pk_sr[label] = (k_sr[label]**3 * pk_sr[label]) / (2 * np.pi**2) 

        if args.compute_extra_statistics:
            k_max_percent_level = find_k_max_within_percent_error(k_sr, pk_sr, pk_hr, threshold=0.01)
            if k_max_percent_level is not None:
                print(f"[INFO] SR reproduces HR within '1% until k = {k_max_percent_level:.3f} h/Mpc")
                print(f"       physical scale = {1.0 / k_max_percent_level:.2f} Mpc/h")
            else:
                print("[INFO] No k values satisfies an error < 1%.")

    #
        # ---------------- RANGE COMPLETO DI k ----------------
    print("\n[INFO] k range:")

    if args.hr_file:
        print(
            f"  HR : k = [{np.min(k_hr):.5f}, {np.max(k_hr):.5f}] h/Mpc"
        )

    if args.lr_file:
        print(
            f"  LR : k = [{np.min(k_lr):.5f}, {np.max(k_lr):.5f}] h/Mpc"
        )

    if args.sr_file:
        for label in args.sr_labels:
            print(
                f"  {label} : k = "
                f"[{np.min(k_sr[label]):.5f}, {np.max(k_sr[label]):.5f}] h/Mpc"
            )

    print()
 


    ######## Plotting ########
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15), sharex=True,
                               gridspec_kw={'height_ratios': [2, 5], 'hspace': 0.05})

    # --- TOP PLOT ---
    if args.sr_file:
        for i, label in enumerate(args.sr_labels):
            ax1.loglog(k_sr[label], pk_sr[label], label=r'SR $_^3$'.replace('_', str(NG_HR)).replace('SR', label), color=args.sr_colors[i])
    if args.lr_file:
        ax1.loglog(k_lr, pk_lr, label=r'LR $_^3$'.replace('_', str(NG_LR)), color='green')

        if args.plot_nyq:
            ax1.axvline(x=NG_LR*np.pi/BOXSIZE, color='gray', linestyle='--', label=r'LR $k_{\rm nyq}$')

        #shot noise
        if args.plot_shotnoise:
            if args.dimensionless:
                shot_lr_plot = (k_hr**3 * shotnoise_lr)/(2*np.pi**2)
            else:
                shot_lr_plot = np.full_like(k_hr, shotnoise_lr)

            ax1.loglog(
                k_hr,
                shot_lr_plot,
                '--',
                color='green',
                alpha=0.6,
                label='LR shot noise'
            )

        
    if args.hr_file:
        ax1.loglog(k_hr, pk_hr, label=r'HR $_^3$'.replace('_', str(NG_HR)), color='blue')
        if args.plot_nyq:
            ax1.axvline(x=NG_HR*np.pi/BOXSIZE, color='black', linestyle='--', label=r'HR $k_{\rm nyq}$')

        if args.plot_shotnoise:
            #shot noise
            if args.dimensionless:
                shot_hr_plot = (k_hr**3 * shotnoise_hr)/(2*np.pi**2)
            else:
                shot_hr_plot = np.full_like(k_hr, shotnoise_hr)

            ax1.loglog(
                k_hr,
                shot_hr_plot,
                '--',
                color='blue',
                alpha=0.6,
                label='HR shot noise'
            )
    

    if args.dimensionless:
        ax1.set_ylabel(r'$\Delta^2(k)$', fontsize=20)
    else:
        ax1.set_ylabel(r'$P(k)\ [h^{-3} \mathrm{Mpc}^3]$', fontsize=20)
    ax1.legend(frameon=False)
    ax1.tick_params(axis="both", labelsize=20) 


    if args.hr_file and (args.lr_file or args.sr_file):
        # --- BOTTOM PLOT (ratio) ---
        if args.sr_file:
            for i, label in enumerate(args.sr_labels):
                ratio_sr = pk_sr[label] / pk_hr
                ax2.semilogx(k_sr[label], ratio_sr, label='SR/HR'.replace('SR',label), color=args.sr_colors[i])

        #hr
        ax2.semilogx(k_hr, np.ones_like(k_hr), color='blue')
        if args.plot_nyq:
            ax2.axvline(x=NG_HR*np.pi/BOXSIZE, color='black', linestyle='--')

        if args.lr_file:
            ratio_lr = pk_lr / np.interp(k_lr, k_hr, pk_hr)
            ax2.semilogx(k_lr, ratio_lr, label='LR/HR', color='green')
            if args.plot_nyq:
                ax2.axvline(x=NG_LR*np.pi/BOXSIZE, color='gray', linestyle='--')

        
        
        

        ax2.set_ylabel('ratio', fontsize=20)
        ax2.set_xlabel(r'$k \,[h\,{\rm Mpc}^{-1}]$', fontsize=20)
        ax2.set_ylim(bottom=0.9, top=1.075)
        ax2.legend(frameon=False, fontsize=24)
        ax2.tick_params(axis="both", labelsize=20) 
        #ax2.grid(True, which='both', linestyle=':', linewidth=0.7, alpha=0.7)

        plt.tight_layout()
    ######################
    

    os.makedirs(args.output_folder, exist_ok=True)
    

    import datetime
    now = datetime.datetime.now()
    fname = now.strftime("pk_%Y%m%d_%H%M%S")

    name = args.sr_labels[0]
    for label in args.sr_labels[1:]:
        name += 'vs' + label
    plt.savefig(os.path.join(args.output_folder, fname + '_' + name + ".png"))
    plt.close()
