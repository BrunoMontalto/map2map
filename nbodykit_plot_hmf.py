import argparse
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['lines.linewidth'] = 1.5
try:
    from nbodykit.lab import Gadget1Catalog, cosmology
    from nbodykit.algorithms.fof import FOF
except:
    print("Cannot import nbodykit")
import os
from pathlib import Path


def compute_halo_catalog(cat, linking_length=0.2, nmin=32):
    """
    Runs Friends-of-Friends with nbodykit.algorithms.fof.FOF
    and returns the halo catalog.
    """
    # Fix columns for compatibility
    if "Velocity" not in cat.columns and "GadgetVelocity" in cat.columns:
        cat['Velocity'] = cat['GadgetVelocity']
    if isinstance(cat.attrs['BoxSize'], (int, float)):
        cat.attrs['BoxSize'] = [cat.attrs['BoxSize']] * 3

    # Particle mass
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


def compute_hmf(halos, boxsize, bins, poisson_error=True): # with poisson error
    """
    Computes the halo mass function d/dlogM from the HaloCatalog.
    Also returns Poisson uncertainties.
    """
    masses = halos["Mass"].compute()
    hist, edges = np.histogram(masses, bins=bins)
    bin_centers = 0.5 * (edges[1:] + edges[:-1])

    dlogM = np.diff(np.log10(edges))
    volume = boxsize**3
    dn_dlogM = hist / (dlogM * volume)

    if poisson_error:
        # Poisson error: sqrt(N) propagated on dn/dlogM
        poisson_error = np.sqrt(hist) / (dlogM * volume)

        return bin_centers, dn_dlogM, poisson_error

    return bin_centers, dn_dlogM



import numpy as np

def find_mass_limit_within_relative_error(
    M_sr, err_sr, max_error=0.10
):
    """
    Returns the minimum mass M_min such that
    rel_err(M) < max_error for all M >= M_min.
    """


    M_sr_valid = M_sr[:-3]

    # Relative error
    rel_err = np.abs(err_sr[:-3])

    print("rel err:", rel_err)

    # True where condition is violated
    bad = rel_err >= max_error#bad = rel_err[:-1] >= max_error 

    # If never violated, return the smallest mass
    if not np.any(bad):
        return M_sr_valid[0]

    # Index of the last violation
    last_bad = np.where(bad)[0][-1]

    # If the last point still violates the condition,
    # there is no mass above which the condition is always satisfied
    if last_bad == len(M_sr_valid) - 1:
        return None

    return M_sr_valid[last_bad + 1]




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compares halo mass function between LR, SR(s) e HR")
    parser.add_argument("--lr-file", type=str, help="Path to low-resolution Gadget file")
    parser.add_argument("--hr-file", type=str, help="Path to high-resolution Gadget file")
    parser.add_argument(
    "--sr-file",
    "--sr-files",
    nargs="+",
    type=str,
    dest="sr_file",
    default = [],
    help="Path(s) to super-resolution Gadget file(s)"
    )
    parser.add_argument("--sr-labels", type=str, nargs="+", default=["SR"])
    parser.add_argument('--sr-colors', type=str, nargs='+', default=["magenta", "red", "yellow"])

    parser.add_argument("--compute-extra-statistics", action='store_true')

    parser.add_argument("--subplot-ystart", type=float, default=None)
    parser.add_argument("--subplot-yend", type=float, default=None)


    parser.add_argument("--output-folder", type=str, default="plots_nbodykit")
    parser.add_argument("--nbins", type=int, default=30)
    parser.add_argument("--max-error", type=float, default=0.1, help="")

    parser.add_argument("--title", type=str, default=None)

    args = parser.parse_args()
    args.lr_file_npz = os.path.join(args.output_folder, "LR_hmf_data.npz")
    args.hr_file_npz = os.path.join(args.output_folder, "HR_hmf_data.npz")
    args.sr_file_npz = [os.path.join(args.output_folder, f"{args.sr_labels[i]}_hmf_data.npz") for i in range(len(args.sr_labels))]

    if len(args.sr_file) > 1 and len(args.sr_labels) == 1:
        args.sr_labels = ["SR" + str(i+1) for i in range(len(args.sr_file))]

    print(vars(args))
    os.makedirs(args.output_folder, exist_ok=True)

    hmf_data = {}

    # HR loading
    bins = None

    hr_file_exists = os.path.exists(args.hr_file_npz)
    if not hr_file_exists:
        if args.hr_file:
            cat_hr = Gadget1Catalog(args.hr_file)
            halos_hr = compute_halo_catalog(cat_hr)
            masses_hr = halos_hr["Mass"].compute()
            
            bins = np.logspace(np.log10(masses_hr.min()), np.log10(masses_hr.max()), args.nbins)
            M_hr, hmf_hr, err_hr = compute_hmf(halos_hr, cat_hr.attrs["BoxSize"][0], bins)
            hmf_data["HR"] = (M_hr, hmf_hr,err_hr, "blue")

            np.savez(args.hr_file_npz, M=M_hr, hmf=hmf_hr, err=err_hr)
            print(f"HR HMF saved in: {args.hr_file_npz}")
    else:
        print("Loading HMF HR from npz...")
        data = np.load(args.hr_file_npz)
        M_hr, hmf_hr, err_hr = data["M"], data["hmf"], data["err"]
        hmf_data["HR"] = (M_hr, hmf_hr, err_hr, "blue")

        #--- recover bins ---#
        # centers in log10
        logM = np.log10(M_hr)

        # average step in log-space
        dlogM = np.diff(logM).mean()

        # edge reconstruction
        bins = np.empty(len(M_hr) + 1)
        bins[1:-1] = 10**((logM[:-1] + logM[1:]) / 2)
        bins[0]  = 10**(logM[0]  - dlogM/2)
        bins[-1] = 10**(logM[-1] + dlogM/2)

    # SR loading
    sr_file_exists = all([os.path.exists(s) for s in args.sr_file_npz]) #TODO
    if not sr_file_exists:
        if args.sr_file:
            for i, path in enumerate(args.sr_file):
                cat_sr = Gadget1Catalog(path)
                halos_sr = compute_halo_catalog(cat_sr)
                masses_sr = halos_sr["Mass"].compute()
                if not args.hr_file:
                    bins = np.logspace(np.log10(masses_sr.min()), np.log10(masses_sr.max()), args.nbins)
                M_sr, hmf_sr = compute_hmf(halos_sr, cat_sr.attrs["BoxSize"][0], bins, poisson_error=False)
                s_sr = (hmf_sr - hmf_hr)/err_hr
                
                hmf_data[args.sr_labels[i]] = (M_sr, hmf_sr, s_sr, args.sr_colors[i])

                np.savez(args.sr_file_npz[i], M=M_sr, hmf=hmf_sr, err=s_sr)
                print(f"SR HMF saved in: {args.sr_file_npz}")
    else:
        print("Loading HMF SR from npz...")
        for i,path in enumerate(args.sr_file_npz):
            data = np.load(path)
            M_sr, hmf_sr,s_sr = data["M"], data["hmf"], data["err"]
            hmf_data[Path(path).name.replace('_hmf_data.npz','')] = (M_sr, hmf_sr, s_sr, args.sr_colors[i])


    # LR loading
    lr_file_exists = os.path.exists(args.lr_file_npz)
    if not lr_file_exists:
        if args.lr_file:
            cat_lr = Gadget1Catalog(args.lr_file)
            halos_lr = compute_halo_catalog(cat_lr)
            masses_lr = halos_lr["Mass"].compute()
            if not (args.hr_file or args.sr_file):
                bins = np.logspace(np.log10(masses_lr.min()), np.log10(masses_lr.max()), args.nbins)
            M_lr, hmf_lr = compute_hmf(halos_lr, cat_lr.attrs["BoxSize"][0], bins, poisson_error=False)
            s_lr = (hmf_lr - hmf_hr) / err_hr
            hmf_data["LR"] = (M_lr, hmf_lr, s_lr, "green")

            np.savez(args.lr_file_npz, M=M_lr, hmf=hmf_lr, err=s_lr)
            print(f"LR HMF saved in: {args.lr_file_npz}")
    else:
        print("Loading HMF LR from npz...")
        data = np.load(args.lr_file_npz)
        M_lr, hmf_lr, s_lr = data["M"], data["hmf"], data["err"]
        hmf_data["LR"] = (M_lr, hmf_lr, s_lr, "green")

        print("LR")
        print(M_lr)
        print(hmf_lr)
        print("end")

    mass_limit = None
    if args.compute_extra_statistics:
        M_hr, hmf_hr, err_hr, _ = hmf_data["HR"]
        if "SR" in hmf_data:
            M_sr, hmf_sr, err_sr, _ = hmf_data["SR"]
            mass_limit = find_mass_limit_within_relative_error(M_sr, err_sr, max_error=args.max_error)

            #find max and min error before threshold
            err_sr_filtered = err_sr[M_sr < mass_limit] if mass_limit is not None else err_sr

            print("[INFO] err_sr_filtered", err_sr_filtered)

            if mass_limit is not None:
                print(f"[INFO] HMF SR reproduces HR within {args.max_error * 100}% for M > {mass_limit:.2e} M☉/h")
            else:
                print(f"[INFO] No mass interval satisfies an error < {args.max_error * 100}% between SR e HR")
            
            print(f"[INFO] min error: {min(err_sr_filtered)} a M={M_sr[np.argmin(err_sr_filtered)]}; max error: {max(err_sr_filtered)} a M={M_sr[np.argmax(err_sr_filtered)]}")

    

        #poisson error statistics
        # ---------------- SR ----------------
        for label, (M_sr, hmf_sr, S_sr, color) in hmf_data.items():

            if label == "HR" or label == "LR":
                continue

            mask_sr = np.isfinite(S_sr)

            filtered_sr = np.abs(S_sr[mask_sr])
            M_sr_f = M_sr[mask_sr]

            if len(filtered_sr) == 0:
                print(f"[INFO] {label}: no finite values.")
                continue

            id_s_max_sr = np.argmax(filtered_sr)
            s_max_sr = filtered_sr[id_s_max_sr]
            m_max_sr = M_sr_f[id_s_max_sr]

            rate_of_bins_within_1sigma_sr = np.mean(
                np.abs(S_sr[mask_sr]) < 1
            )

            print(
                f"[INFO] {label} max absolute deviation:",
                s_max_sr,
                "at bin",
                m_max_sr
            )

            print(
                f"[INFO] {label} rate of bins within 1 sigma:",
                rate_of_bins_within_1sigma_sr
            )


            #--- Relative HMF deviation ---#

            M_hr_stat, hmf_hr_stat, _, _ = hmf_data["HR"]

            mask_hr_stat = hmf_hr_stat > 0

            M_hr_stat = M_hr_stat[mask_hr_stat]
            hmf_hr_stat = hmf_hr_stat[mask_hr_stat]

            mask_common = (
                (M_sr >= M_hr_stat.min()) &
                (M_sr <= M_hr_stat.max()) &
                (hmf_sr > 0)
            )

            M_common = M_sr[mask_common]
            hmf_sr_common = hmf_sr[mask_common]

            if len(M_common) > 0:
                hmf_hr_interp = np.interp(
                    M_common,
                    M_hr_stat,
                    hmf_hr_stat
                )

                # Relative deviation:
                # (HMF_SR - HMF_HR) / HMF_HR
                relative_deviation = (
                    hmf_sr_common - hmf_hr_interp
                ) / hmf_hr_interp

                # Max absolute deviation
                id_max_rel = np.argmax(np.abs(relative_deviation))

                max_relative_deviation = np.abs(
                    relative_deviation[id_max_rel]
                )

                max_relative_mass = M_common[id_max_rel]

                # Max positive deviation
                id_max_positive = np.argmax(relative_deviation)

                max_positive_deviation = relative_deviation[id_max_positive]
                max_positive_mass = M_common[id_max_positive]

                # Min positive deviation
                id_max_negative = np.argmin(relative_deviation)

                max_negative_deviation = relative_deviation[id_max_negative]
                max_negative_mass = M_common[id_max_negative]

                print(
                    f"[INFO] {label} max |relative HMF deviation|: "
                    f"{max_relative_deviation * 100:.2f}% "
                    f"at M = {max_relative_mass:.3e} M_sun/h"
                )

                print(
                    f"[INFO] {label} max positive deviation: "
                    f"{max_positive_deviation * 100:+.2f}% "
                    f"at M = {max_positive_mass:.3e} M_sun/h"
                )

                print(
                    f"[INFO] {label} max negative deviation: "
                    f"{max_negative_deviation * 100:+.2f}% "
                    f"at M = {max_negative_mass:.3e} M_sun/h"
                )

            else:
                print(
                    f"[INFO] {label}: no common mass interval with HR."
                )

        


        # ---------------- LR ----------------
        S_lr = hmf_data["LR"][2]
        M_lr = hmf_data["LR"][0]

        mask_lr = np.isfinite(S_lr)

        filtered_lr = np.abs(S_lr[mask_lr])
        M_lr_f = M_lr[mask_lr]

        id_s_max_lr = np.argmax(filtered_lr)
        s_max_lr = filtered_lr[id_s_max_lr]
        m_max_lr = M_lr_f[id_s_max_lr]

        rate_of_bins_within_1sigma_lr = np.mean(np.abs(S_lr[mask_lr]) < 1)

        print("[INFO] LR max absolute deviation:", s_max_lr, "at bin", m_max_lr)
        print("[INFO] LR rate of bins within 1 sigma:", rate_of_bins_within_1sigma_lr)



        print("[INFO] hmf SR:", hmf_sr, "hmf HR:", hmf_hr, "err HR:", err_hr, "s SR:", s_sr, "M_sr:", M_sr)


    if "HR" not in hmf_data:
            print("HR file required to compute ratios.")
            exit(1)


    ######## Plotting ########
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15), sharex=True,
                            gridspec_kw={'height_ratios': [2, 5], 'hspace': 0.05})

    if args.title is not None:
        fig.suptitle(args.title, fontsize=20)

    # --- TOP PLOT ---
    for label, (M, hmf, err, color) in hmf_data.items():
        # Rimuovi i punti con hmf = 0
        mask = hmf > 0
        M_plot = M[mask]
        hmf_plot = hmf[mask]
        err_plot = err[mask]


        
        ax1.loglog(M_plot, hmf_plot, label=label, color=color)
        if label == "HR":
            # Poisson error bars
            ax1.fill_between(M_plot,
                            hmf_plot - err_plot,
                            hmf_plot + err_plot,
                            color=color,
                            alpha=0.3)
            ax1.fill_between(M_plot,
                            hmf_plot - err_plot*0.1,
                            hmf_plot + err_plot*0.1,
                            color="gray",
                            alpha=0.5)
            #ax1.fill_between(M_plot,
            #                hmf_plot - err_plot*0.05,
            #                hmf_plot + err_plot*0.05,
            #                color="gray",
            #                alpha=0.8)
            

    # vertical line
    if mass_limit is not None:
        ax1.axvline(mass_limit, color='gray', linestyle='--', alpha=0.7, label=r"$|\mathrm{HMF}_{SR}-\mathrm{HMF}_{HR}|/\sigma_{HR}<1.3$")

    ax1.set_ylabel(r"$dn/d\log M \; [ (h^{-1}{\rm Mpc})^{-3} ]$", fontsize=20)
    ax1.legend(frameon=False, fontsize=24)
    ax1.tick_params(axis="both", labelsize=20) 
    # --- BOTTOM PLOT (ratio) ---
    #aggiungi tic mass limit
    if args.subplot_ystart is not None and args.subplot_yend is not None:
                ax2.set_ylim(args.subplot_ystart, args.subplot_yend)

    if mass_limit is not None:
        ticks = list(ax2.get_xticks())

        # add tick
        ticks.append(mass_limit)
        ticks = sorted(ticks)

        ax2.set_xticks(ticks)

        # vertical line
        ax2.axvline(mass_limit, color='gray', linestyle='--', alpha=0.7)
        ax2.axvline(M_sr[np.argmax(err_sr_filtered)], color='red', linestyle='--', alpha=0.7)

        # custom label
        labels = []
        for t in ticks:
            if np.isclose(t, mass_limit):
                labels.append("Threshold")
            else:
                labels.append(f"{t:g}")

        ax2.set_xticklabels(labels)

    M_hr, hmf_hr, err_hr, _ = hmf_data["HR"]
    mask_hr = hmf_hr > 0
    M_hr_valid = M_hr[mask_hr]
    hmf_hr_valid = hmf_hr[mask_hr]
    err_hr_valid = err_hr[mask_hr]

    for label, (M, hmf, err, color) in hmf_data.items():
        # Remove 0s
        mask = hmf > 0
        M_valid = M[mask]
        hmf_valid = hmf[mask]
        err_valid = err[mask]


        if label == "HR":
            ax2.semilogx(M_hr_valid, np.ones_like(M_hr_valid), label="HR/HR", color=color)
            # Poisson errors
            rel_err_hr = err_hr_valid / hmf_hr_valid
            ax2.fill_between(M_hr_valid,
                            1 - rel_err_hr,
                            1 + rel_err_hr,
                            color=color,
                            alpha=0.3)
            ax2.fill_between(M_hr_valid,
                            1 - rel_err_hr*0.1,
                            1 + rel_err_hr*0.1,
                            color="gray",
                            alpha=0.5)
            #ax2.fill_between(M_hr_valid,
            #                1 - rel_err_hr*0.05,
            #                1 + rel_err_hr*0.05,
            #                color="gray",
            #                alpha=0.8)
        else:
             # Only condier masses in HR range
            mask_common = (
                (M_valid >= M_hr_valid.min()) &
                (M_valid <= M_hr_valid.max())
            )

            M_common = M_valid[mask_common]
            hmf_common = hmf_valid[mask_common]
            err_common = err_valid[mask_common]

            # Interpolate on valid masses
            hmf_interp = np.interp(M_common, M_hr_valid, hmf_hr_valid)
            ratio = hmf_common / hmf_interp

            rel_err = err_common / hmf_common
            rel_err_hr = np.interp(M_common, M_hr_valid, err_hr_valid) / hmf_interp
            total_err = ratio * np.sqrt(rel_err**2 + rel_err_hr**2)



            ax2.semilogx(M_common, ratio, label=f"{label}/HR", color=color)
            

    ax2.set_ylabel("ratio", fontsize=20)
    ax2.set_xlabel(r"$M \; [M_\odot/h]$", fontsize=20)
    ax2.legend(frameon=False, fontsize=24)
    ax2.tick_params(axis="both", labelsize=20) 

    plt.tight_layout()


    ################

    plt.tight_layout()

    
    import datetime
    now = datetime.datetime.now()
    fname = now.strftime("hmf_%Y%m%d_%H%M%S")

    name = args.sr_labels[0]
    for label in args.sr_labels[1:]:
        name += 'vs' + label
    plt.savefig(os.path.join(args.output_folder, fname + '_' + name + ".png"))
    plt.close()
