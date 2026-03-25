import argparse
import numpy as np
import matplotlib.pyplot as plt
from nbodykit.lab import Gadget1Catalog, cosmology
from nbodykit.algorithms.fof import FOF
import os


def compute_halo_catalog(cat, linking_length=0.2, nmin=32):
    """
    Esegue Friends-of-Friends con nbodykit.algorithms.fof.FOF
    e restituisce il catalogo degli halo.
    """
    # Fix columns for compatibility
    if "Velocity" not in cat.columns and "GadgetVelocity" in cat.columns:
        cat['Velocity'] = cat['GadgetVelocity']
    if isinstance(cat.attrs['BoxSize'], (int, float)):
        cat.attrs['BoxSize'] = [cat.attrs['BoxSize']] * 3

    # Particle mass (usually MassTable[1])
    mass_np = cat['Mass'].compute()

    # Read first value and fix unit
    first_mass = mass_np[0] * 1e10

    print("first mass",type(first_mass), first_mass)
    #p_mass = cat['Mass'][0]

    # Default cosmology
    cosmo = cosmology.Planck15
    fof = FOF(cat, linking_length=linking_length, nmin=nmin)
    halos = fof.to_halos(particle_mass=first_mass, cosmo=cosmo, redshift=0.0)

    print("found", len(halos), "halos")

    return halos


def compute_hmf(halos, boxsize, bins): # with poisson error
    """
    Calcola la funzione di massa dn/dlogM da un HaloCatalog.
    Ritorna anche gli errori di Poisson.
    """
    masses = halos["Mass"].compute()
    hist, edges = np.histogram(masses, bins=bins)
    bin_centers = 0.5 * (edges[1:] + edges[:-1])

    dlogM = np.diff(np.log10(edges))
    volume = boxsize**3
    dn_dlogM = hist / (dlogM * volume)

    # Poisson error: sqrt(N) propagated on dn/dlogM
    poisson_error = np.sqrt(hist) / (dlogM * volume)

    return bin_centers, dn_dlogM, poisson_error



def find_mass_limit_within_relative_error(M_sr, hmf_sr, M_hr, hmf_hr, max_error=0.10):
    """
    Returns the minimum mass such that |hmf_sr - hmf_hr| / hmf_hr < max_error
    """
    # Filtra valori > 0
    mask_sr = hmf_sr > 0
    M_sr_valid = M_sr[mask_sr]
    hmf_sr_valid = hmf_sr[mask_sr]

    mask_hr = hmf_hr > 0
    M_hr_valid = M_hr[mask_hr]
    hmf_hr_valid = hmf_hr[mask_hr]

    # Interpolate HR on M_sr
    hmf_hr_interp = np.interp(M_sr_valid, M_hr_valid, hmf_hr_valid)

    # Compute relative error
    rel_err = np.abs(hmf_sr_valid - hmf_hr_interp) / hmf_hr_interp

    # Finx indices with error under the threshold
    valid_idx = np.where(rel_err < max_error)[0]

    if len(valid_idx) == 0:
        return None

    # Return the minimum mass where the contidion is satisfied
    return M_sr_valid[valid_idx[0]]




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Confronta halo mass function tra LR, SR e HR")
    parser.add_argument("--lr-file", type=str, help="Path al file Gadget low-resolution")
    parser.add_argument("--hr-file", type=str, help="Path al file Gadget high-resolution")
    parser.add_argument("--sr-file", type=str, help="Path al file Gadget super-resolution")
    parser.add_argument("--lr-file-npz", type=str, help="Path al file Gadget low-resolution")
    parser.add_argument("--hr-file-npz", type=str, help="Path al file Gadget high-resolution")
    parser.add_argument("--sr-file-npz", type=str, help="Path al file Gadget super-resolution")
    parser.add_argument("--output-folder", type=str, default="plots_nbodykit", help="Cartella output")
    parser.add_argument("--nbins", type=int, default=30, help="Numero di bin logaritmici per HMF")

    args = parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    hmf_data = {}

    # Caricamento HR
    bins = None

    if not args.hr_file_npz:
        if args.hr_file:
            cat_hr = Gadget1Catalog(args.hr_file)
            halos_hr = compute_halo_catalog(cat_hr)
            masses_hr = halos_hr["Mass"].compute()
            
            bins = np.logspace(np.log10(masses_hr.min()), np.log10(masses_hr.max()), args.nbins)
            M_hr, hmf_hr, err_hr = compute_hmf(halos_hr, cat_hr.attrs["BoxSize"][0], bins)
            hmf_data["HR"] = (M_hr, hmf_hr,err_hr, "blue")
    else:
        print("Caricamento HMF HR da npz...")
        data = np.load(args.hr_file_npz)
        M_hr, hmf_hr, err_hr = data["M"], data["hmf"], data["err"]
        hmf_data["HR"] = (M_hr, hmf_hr, err_hr, "blue")

    # Caricamento SR
    if not args.sr_file_npz:
        if args.sr_file:
            cat_sr = Gadget1Catalog(args.sr_file)
            halos_sr = compute_halo_catalog(cat_sr)
            masses_sr = halos_sr["Mass"].compute()
            if not args.hr_file:
                bins = np.logspace(np.log10(masses_sr.min()), np.log10(masses_sr.max()), args.nbins)
            M_sr, hmf_sr, err_sr = compute_hmf(halos_sr, cat_sr.attrs["BoxSize"][0], bins)
            hmf_data["SR"] = (M_sr, hmf_sr, err_sr, "magenta")
    else:
        print("Caricamento HMF SR da npz...")
        data = np.load(args.sr_file_npz)
        M_sr, hmf_sr,err_sr = data["M"], data["hmf"], data["err"]
        hmf_data["SR"] = (M_sr, hmf_sr, err_sr,"magenta")


    # Caricamento LR
    if not args.lr_file_npz:
        if args.lr_file:
            cat_lr = Gadget1Catalog(args.lr_file)
            halos_lr = compute_halo_catalog(cat_lr)
            masses_lr = halos_lr["Mass"].compute()
            if not (args.hr_file or args.sr_file):
                bins = np.logspace(np.log10(masses_lr.min()), np.log10(masses_lr.max()), args.nbins)
            M_lr, hmf_lr, err_lr = compute_hmf(halos_lr, cat_lr.attrs["BoxSize"][0], bins)
            hmf_data["LR"] = (M_lr, hmf_lr, err_lr, "green")
    else:
        print("Caricamento HMF LR da npz...")
        data = np.load(args.lr_file_npz)
        M_lr, hmf_lr, err_lr = data["M"], data["hmf"], data["err"]
        hmf_data["LR"] = (M_lr, hmf_lr, err_lr, "green")

        print("LR")
        print(M_lr)
        print(hmf_lr)
        print("end")

    
    M_hr, hmf_hr, err_hr, _ = hmf_data["HR"]
    if "SR" in hmf_data:
        M_sr, hmf_sr, err_sr, _ = hmf_data["SR"]
        mass_limit = find_mass_limit_within_relative_error(M_sr, hmf_sr, M_hr, hmf_hr, max_error=0.10)

        if mass_limit is not None:
            print(f"[INFO] HMF SR riproduce HR entro il 10% per M > {mass_limit:.2e} M☉/h")
        else:
            print("[INFO] Nessun intervallo di massa soddisfa l'errore < 10% tra SR e HR")

    if "HR" not in hmf_data:
        print("Serve almeno un file HR per calcolare i rapporti.")
        exit(1)

    #save results in output folder as numpy array files
    if (not (args.lr_file_npz or args.hr_file_npz or args.sr_file_npz)):
        for label, (M, hmf, err, _) in hmf_data.items():
            npz_filename = os.path.join(args.output_folder, f"{label}_hmf.npz")
            np.savez(npz_filename, M=M, hmf=hmf, err=err)
            print(f"HMF salvata in: {npz_filename}")

    """
    ######## PLOT ########
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})

    # --- PLOT SUPERIORE (HMF) ---
    for label, (M, hmf, color) in hmf_data.items():
        ax1.loglog(M, hmf, label=label, color = color)

    ax1.set_ylabel(r"$dn/d\log M \; [ (h^{-1}{\rm Mpc})^{-3} ]$")
    ax1.legend(frameon=False)

    # --- PLOT INFERIORE (ratio rispetto a HR) ---
    M_hr, hmf_hr, _ = hmf_data["HR"]
    for label, (M, hmf, color) in hmf_data.items():
        if label == "HR":
            ax2.semilogx(M_hr, np.ones_like(M_hr), label="HR/HR", color=color)
        else:
            ratio = hmf / np.interp(M, M_hr, hmf_hr)
            ax2.semilogx(M, ratio, label=f"{label}/HR", color=color)

    ax2.set_ylabel("ratio")
    ax2.set_xlabel(r"$M \; [M_\odot/h]$")
    ax2.legend(frameon=False)
    #######################################
    """

    ################
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True,
                            gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})

    # --- PLOT SUPERIORE (HMF) ---
    for label, (M, hmf, err, color) in hmf_data.items():
        # Rimuovi i punti con hmf = 0
        mask = hmf > 0
        M_plot = M[mask]
        hmf_plot = hmf[mask]
        err_plot = err[mask]
        
        if label != "HR":
            M_plot = M_plot[:-1]
            hmf_plot = hmf_plot[:-1]
        
        ax1.loglog(M_plot, hmf_plot, label=label, color=color)
        if label == "HR":
            # Aggiungo la banda di errore (Poisson)
            ax1.fill_between(M_plot,
                            hmf_plot - err_plot,
                            hmf_plot + err_plot,
                            color=color,
                            alpha=0.3)

    ax1.set_ylabel(r"$dn/d\log M \; [ (h^{-1}{\rm Mpc})^{-3} ]$")
    ax1.legend(frameon=False)

    # --- PLOT INFERIORE (ratio rispetto a HR) ---
    M_hr, hmf_hr, err_hr, _ = hmf_data["HR"]
    mask_hr = hmf_hr > 0
    M_hr_valid = M_hr[mask_hr]
    hmf_hr_valid = hmf_hr[mask_hr]
    err_hr_valid = err_hr[mask_hr]

    for label, (M, hmf, err, color) in hmf_data.items():
        # Rimuovi zeri dalla curva corrente
        mask = hmf > 0
        M_valid = M[mask]
        hmf_valid = hmf[mask]
        err_valid = err[mask]


        if label == "HR":
            ax2.semilogx(M_hr_valid, np.ones_like(M_hr_valid), label="HR/HR", color=color)
            # anche per HR posso aggiungere la banda di errore relativa
            rel_err_hr = err_hr_valid / hmf_hr_valid
            ax2.fill_between(M_hr_valid,
                            1 - rel_err_hr,
                            1 + rel_err_hr,
                            color=color,
                            alpha=0.3)
        else:
            # Interpolazione solo su valori validi
            hmf_interp = np.interp(M_valid, M_hr_valid, hmf_hr_valid)
            ratio = hmf_valid / hmf_interp

            rel_err = err_valid / hmf_valid
            rel_err_hr = np.interp(M_valid, M_hr_valid, err_hr_valid) / hmf_interp
            total_err = ratio * np.sqrt(rel_err**2 + rel_err_hr**2)

            M_valid = M_valid[:-1]
            ratio = ratio[:-1]

            ax2.semilogx(M_valid, ratio, label=f"{label}/HR", color=color)
            

    ax2.set_ylabel("ratio")
    ax2.set_xlabel(r"$M \; [M_\odot/h]$")
    ax2.legend(frameon=False)

    plt.tight_layout()


    ################

    plt.tight_layout()

    if args.sr_file:
        fname = os.path.basename(args.sr_file)
    elif args.lr_file:
        fname = os.path.basename(args.lr_file)
    elif args.hr_file:
        fname = os.path.basename(args.hr_file)
    else:
        import datetime
        now = datetime.datetime.now()
        fname = now.strftime("hmf_%Y%m%d_%H%M%S")


    outname = fname.split(".")[0] + ".png"
    plt.savefig(os.path.join(args.output_folder, outname))
    plt.close()
