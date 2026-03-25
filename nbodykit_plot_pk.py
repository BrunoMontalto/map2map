""" this script does pk comparison between lr and hr"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from nbodykit.lab import *
import os
from glob import glob
#from nbodykit.io import Gadget1File

def find_k_max_within_percent_error(k_vals, pk_sr, pk_hr, threshold=0.01):
    """
    Returns the maximum value of k such that the percent error between pk_sr and ph_hr is under threshold.
    """
    # Interpolate pk_hr su k_sr if necessary
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
    parser.add_argument('--sr-file', type=str, required=True, help='Path to super-resolution Gadget file')
    parser.add_argument('--output-folder', type=str, default='plots_nbodykit', help='Output plot file path')

    parser.add_argument('--boxsize', type=float, default=1000.0)
    parser.add_argument('--Ng-LR', type=int, default=512)
    parser.add_argument('--Ng-HR', type=int, default=512)

    parser.add_argument('--dimensionless', action='store_true')

    args = parser.parse_args()

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
        no_files = False

        cat_sr = Gadget1Catalog(sorted(glob(args.sr_file)))

        mesh_sr = cat_sr.to_mesh(Nmesh=NG_HR, BoxSize=BOXSIZE, resampler='tsc',interlaced=True)
        r_sr = FFTPower(mesh_sr, mode='1d')#, kmin = 0.024)
        pk_sr = r_sr.power
    

    if no_files:
        print("no files provided")
        exit(1)
    

    # Visualization
    if args.lr_file:
        k_lr = pk_lr['k']
        pk_lr = pk_lr['power'].real - pk_lr.attrs['shotnoise']

        if args.dimensionless:
            pk_lr = (k_lr**3 * pk_lr) / (2 * np.pi**2)


        #plt.loglog(k_lr, pk_lr, label='LR')
        #plt.axvline(x=NG_LR*np.pi/BOXSIZE, color='gray', linestyle='--', label='HR Nyquist')
    if args.hr_file:
        k_hr = pk_hr['k']
        pk_hr = pk_hr['power'].real - pk_hr.attrs['shotnoise']

        if args.dimensionless:
            pk_hr = (k_hr**3 * pk_hr) / (2 * np.pi**2) 

        #plt.loglog(k_hr, pk_hr, label='HR')
        #plt.axvline(x=NG_HR*np.pi/BOXSIZE, color='black', linestyle='--', label='HR Nyquist')
    #plt.legend()
    #plt.xlabel('k')
    #plt.ylabel('P(k)')
    #plt.title('Power Spectrum')

    #pk_lr_ratio (cropping to perform division)

    if args.sr_file:
        k_sr = pk_sr['k']
        pk_sr = pk_sr['power'].real - pk_sr.attrs['shotnoise']

        if args.dimensionless:
            pk_sr = (k_sr**3 * pk_sr) / (2 * np.pi**2) 


        k_max_percent_level = find_k_max_within_percent_error(k_sr, pk_sr, pk_hr, threshold=0.01)
        if k_max_percent_level is not None:
            print(f"[INFO] SR riproduce HR entro l'1% fino a k = {k_max_percent_level:.3f} h/Mpc")
            print(f"       → scala fisica = {1.0 / k_max_percent_level:.2f} Mpc/h")
        else:
            print("[INFO] Nessun valore di k soddisfa l'errore percentuale inferiore all'1%.")


    #pk_sr_ratio = pk_sr / pk_hr

    #pk_hr_cropped = pk_hr[:len(pk_lr)]
    #pk_lr_ratio = pk_lr / pk_hr_cropped

    #pk_hr_ratio = np.ones_like(pk_hr)

    ######## plot ########
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True,
                               gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})

    # --- TOP PLOT (power spectrum) ---
    if args.sr_file:
        ax1.loglog(k_sr, pk_sr, label=r'SR $_^3$'.replace('_', str(NG_HR)), color='magenta')
    if args.lr_file:
        ax1.loglog(k_lr, pk_lr, label=r'LR $_^3$'.replace('_', str(NG_LR)), color='green')
        ax1.axvline(x=NG_LR*np.pi/BOXSIZE, color='gray', linestyle='--', label=r'LR $k_{\rm nyq}$')
    if args.hr_file:
        ax1.loglog(k_hr, pk_hr, label=r'HR $_^3$'.replace('_', str(NG_HR)), color='blue')
        ax1.axvline(x=NG_HR*np.pi/BOXSIZE, color='black', linestyle='--', label=r'HR $k_{\rm nyq}$')
    

    if args.dimensionless:
        ax1.set_ylabel(r'$\Delta^2(k)$')
    else:
        ax1.set_ylabel(r'$P(k)\ [h^{-3} \mathrm{Mpc}^3]$')
    ax1.legend(frameon=False)
    #ax1.grid(True, which='both', linestyle=':', linewidth=0.7, alpha=0.7)

    if args.hr_file and (args.lr_file or args.sr_file):
        # --- BOTTOM PLOT (ratio) ---
        if args.sr_file:
            ratio_sr = pk_sr / pk_hr
            ax2.semilogx(k_sr, ratio_sr, label='SR/HR', color='magenta')

        #hr
        ax2.semilogx(k_hr, np.ones_like(k_hr), color='blue')
        ax2.axvline(x=NG_HR*np.pi/BOXSIZE, color='black', linestyle='--')

        if args.lr_file:
            ratio_lr = pk_lr / np.interp(k_lr, k_hr, pk_hr)
            ax2.semilogx(k_lr, ratio_lr, label='LR/HR', color='green')
            ax2.axvline(x=NG_LR*np.pi/BOXSIZE, color='gray', linestyle='--')

        
        
        

        ax2.set_ylabel('ratio')
        ax2.set_xlabel(r'$k \,[h\,{\rm Mpc}^{-1}]$')
        ax2.legend(frameon=False)
        #ax2.grid(True, which='both', linestyle=':', linewidth=0.7, alpha=0.7)

        plt.tight_layout()
    ######################
    

    os.makedirs(args.output_folder, exist_ok=True)
    plt.savefig(os.path.join(args.output_folder, os.path.basename(args.sr_file).split(".")[0] + ".png"))
    plt.close()
