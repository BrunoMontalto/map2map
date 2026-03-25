#!/usr/bin/env python3
"""
comparison_plots.py

Script to compare three samples (lr, hr, sr) from cosmological simulations
saved as NumPy arrays with shape (3, N, N, N). The script takes as input three
.npy files (or .npz files containing a single array), cropping parameters, and
an output directory where the comparison plot will be saved (1 row x 3 columns:
lr, hr, sr).

Main behavior:
 - The provided crop indices (crop_field_start, crop_field_stop) are
   interpreted with respect to the grid at the LOWEST resolution (the smallest
   among the three files). For higher-resolution grids, the crop is scaled
   accordingly (e.g., doubled if N_file == 2 * N_min).
 - The internal block [start:stop) is read (using mmap) and converted into
   positions via the `dis2pos` function (assumed to be already implemented and
   importable).
 - The position range (x, y, z) of the block is determined.
 - An extended block is then reloaded with a margin of 20 cells around the
   original block, using periodic boundary conditions (index wrapping). Here
   too, indices are scaled for different resolutions.
 - Positions from the extended block are converted, and only the particles
   that fall within the original XY range are selected (for the XY projection).
 - A two-dimensional density estimate (2D histogram) is computed and used to
   color the particles in the XY projection.
 - A scientific-style plot (matplotlib) is created and saved in output_dir.

Note: the code assumes that the files contain exactly one array of shape
(3, N, N, N). If the files are .npz with multiple arrays, load them beforehand
or modify the code.

"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


from displacement import dis2pos, dis2posLR_average_downsample



def load_numpy_memmap(path):
    """Carica un array numpy come memmap (supporta .npy e .npz con 1 elemento).
    Restituisce l'array (mmap) e la sua shape.
    """
    assert path.endswith('.npy')
    arr = np.load(path, mmap_mode='r')
    return arr
   


def wrapped_take_3d(arr, ix, iy, iz):
    """
    Extracts a block with shape (3, len(ix), len(iy), len(iz)) using np.take with wrap.
    """

    tmp = np.take(arr, ix, axis=1, mode='wrap')
    tmp = np.take(tmp, iy, axis=2, mode='wrap')
    tmp = np.take(tmp, iz, axis=3, mode='wrap')
    return tmp


def read_block_with_periodic(arr, start, stop):
    """Reads block [start:stop) along the 3 dimensions from arr
    (shape (3,N,N,N)) applying periodic boundary conditions on the indices.
    """
    N = arr.shape[1]
    # normalize to [0, N)
    length = stop - start
    if length <= 0:
        raise ValueError("stop must be > start")
    ix = np.arange(start, start + length) % N
    iy = ix.copy()
    iz = ix.copy()
    block = wrapped_take_3d(arr, ix, iy, iz)
    return block



def compute_density_colors(xyz_positions, nbins=100):
    """Estimates the three-dimensional density using histogramdd and returns
    a density array (log-scaled) for each particle.

    Parameters
    ----------
    xyz_positions : ndarray, shape (N, 3)
        Particle positions (x, y, z).
    nbins : int
        Number of bins per axis (default 100).

    Returns
    -------
    dens_log : ndarray, shape (N,)
        Log10-normalized density per particle.
    """
    x = xyz_positions[:, 0]
    y = xyz_positions[:, 1]
    z = xyz_positions[:, 2]

    ranges = [
        (x.min(), x.max()),
        (y.min(), y.max()),
        (z.min(), z.max())
    ]

    # 3D istogram
    H, edges = np.histogramdd(xyz_positions, bins=nbins, range=ranges)

    # Find bin index for every particle
    inds = []
    for i in range(3):
        inds.append(np.searchsorted(edges[i], xyz_positions[:, i], side='right') - 1)
    inds = np.stack(inds, axis=1)

    # Clamp (for safety)
    for i in range(3):
        inds[:, i] = np.clip(inds[:, i], 0, H.shape[i] - 1)

    dens = H[inds[:, 0], inds[:, 1], inds[:, 2]].astype(float)

    # Avoid zeros
    if np.any(dens > 0):
        dens[dens <= 0] = dens[dens > 0].min()
    else:
        dens[:] = 1.0

    dens_log = np.log10(dens)
    return dens_log

def process_file(path, scale, crop_start, crop_stop, margin=20, boxsize=None, is_LR=False, original_lr_res=None):
    """
    Loads, extracts block and converts to positions in the desired range
    """
    arr = load_numpy_memmap(path)
    if arr.ndim != 4 or arr.shape[0] != 3:
        raise ValueError(f"Array in {path} deve avere shape (3, N, N, N). Got {arr.shape}")
    N = arr.shape[1]

    
    s = int(crop_start * scale)
    e = int(crop_stop * scale)

    mins = boxsize /original_lr_res * (crop_start)
    maxs = boxsize /original_lr_res * (crop_stop)


    # Read block extended with margin
    s2 = s - margin*scale
    e2 = e + margin*scale
    boxsize_scaled = boxsize * (crop_stop-crop_start + 2*margin)/original_lr_res
    block2 = read_block_with_periodic(arr, s2, e2)
    if scale == 1:
        pos_block2 = dis2posLR_average_downsample(block2, Ng_LR=(e2-s2), Ng_HR=(e2-s2)*2, boxsize=boxsize_scaled)
    else:
        pos_block2 = dis2pos(block2, Ng=(e2-s2), boxsize=boxsize_scaled)

    # Reshape
    pos_block2 = np.moveaxis(pos_block2, 0, -1).reshape(-1, 3)

    # Select particles in the range
    mask = (
        (pos_block2[:, 0] >= mins) & (pos_block2[:, 0] <= maxs) &
        (pos_block2[:, 1] >= mins) & (pos_block2[:, 1] <= maxs) &
        (pos_block2[:, 2] >= mins) & (pos_block2[:, 2] <= maxs)
    )
    selected = pos_block2[mask]
    return selected



def main():
    parser = argparse.ArgumentParser(description='Crea plot comparativo LR-HR-SR (proiezione XY).')
    parser.add_argument('--file-lr', required=True, help='Percorso file LR (.npy/.npz)')
    parser.add_argument('--file-hr', required=True, help='Percorso file HR (.npy/.npz)')
    parser.add_argument('--file-sr', required=True, help='Percorso file SR (.npy/.npz)')
    parser.add_argument('--crop-field-start', type=int, required=True, help='Start index (int) in grid units (base/min resolution)')
    parser.add_argument('--crop-field-stop', type=int, required=True, help='Stop index (int) in grid units (base/min resolution)')
    parser.add_argument('--output-dir', required=True, help='Directory where to save the figure')
    parser.add_argument('--margin', type=int, default=20, help='Margin (cells) for the extended read; default 20')
    parser.add_argument('--nbins', type=int, default=200, help='Number of bins for 2D density estimate (per axis)')
    parser.add_argument('--original-boxsize', type=float, default=1000.0)
    parser.add_argument('--particle-size-hr', type=float, default=0.01)
    parser.add_argument('--alpha-hr', type=float, default=0.5)
    parser.add_argument('--particle-size-lr', type=float, default=0.08)
    parser.add_argument('--alpha-lr', type=float, default=1.0)
    parser.add_argument('--original-lr-res', type=int, default=512)
    args = parser.parse_args()

    files = {'LR': args.file_lr, 'HR': args.file_hr, 'SR': args.file_sr}

    

    # Load shapes
    shapes = {}
    Ns = {}
    for k, p in files.items():
        arr = load_numpy_memmap(p)
        if arr.ndim != 4 or arr.shape[0] != 3:
            raise ValueError(f"{p} must be a NumPy array with shape (3,N,N,N). Got {arr.shape}.")
        Ns[k] = arr.shape[1]
        shapes[k] = arr.shape

    # Find minimum resolution and use as base to compute scale factors
    min_N = min(Ns.values())
    scales = {k: Ns[k] // min_N for k in Ns}
    for k in scales:
        if Ns[k] % min_N != 0:
            print(f"Warning: N for {k} (= {Ns[k]}) is not an integer multiple of min N (={min_N}). Using integer division for scale.")

    print("scales:", scales)
    # creiamo output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # process each file
    results = {}
    for k, p in files.items():
        print(f"Processing {k} ({p}) with scale {scales[k]} ...")
        sel = process_file(p, scale=scales[k], crop_start=args.crop_field_start, crop_stop=args.crop_field_stop, margin=args.margin, boxsize=args.original_boxsize, original_lr_res = args.original_lr_res)
        results[k] = sel
        print(f" -> selected {len(sel)} particles for {k}")

    # Prepare 1x3 plot - XY proj
    rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'figure.figsize': (15, 5),
        'legend.fontsize': 9,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'axes.linewidth': 0.8
    })

    fig, axs = plt.subplots(1, 3, constrained_layout=True)#, figsize=(18,6))
    kinds = ['LR', 'HR', 'SR']
    vmax_all = None
    dens_maps = {}

    # Compute densities and plot
    for ax, kind in zip(axs, kinds):
        ax.set_facecolor('black')
        pos = results[kind]
        if pos.size == 0:
            ax.text(0.5, 0.5, 'No particles', ha='center', va='center')
            ax.set_title(kind)
            continue
        xy = pos[:, :2]
        #dens = compute_density_colors(xy, nbins=args.nbins)
        dens = compute_density_colors(pos, nbins=args.nbins)

        # Normalize dens for colormap
        # Map dens to 0-1
        dmin, dmax = dens.min(), dens.max()
        if dmax == dmin:
            norm = (dens - dmin)
        else:
            norm = (dens - dmin) / (dmax - dmin)

        if kind == "LR":
            s = args.particle_size_lr
            alpha = args.alpha_lr
        else:
            s = args.particle_size_hr
            alpha = args.alpha_hr

        sc = ax.scatter(xy[:, 0], xy[:, 1], c=norm, s=s, alpha=alpha, cmap='viridis', rasterized=True) #viridis or plasma
        ax.set_aspect('equal')
        ax.set_title(kind)

        #ax.tick_params(colors='palegreen')  # Colore dei numeri su assi
        #ax.xaxis.label.set_color('white')  # Etichetta asse x
        #ax.yaxis.label.set_color('white')  # Etichetta asse y
        #ax.title.set_color('white')        # Titolo subplot
        ax.set_facecolor('black')  # sfondo nero
        #ax.tick_params(color='palegreen')  # tick marks (trattini) bianchi
        ax.tick_params(color='palegreen', width=1) #tick marks (trattini) bianchi con spessore
        for spine in ax.spines.values():
            spine.set_edgecolor('palegreen')  # bordi bianchi
            spine.set_linewidth(1) #spessore bordi


        ax.set_xlabel(r'$X\ \mathrm{[h^{-1}\ Mpc]}$')
        # y label only on first
        if ax is axs[0]:
            ax.set_ylabel(r'$Y\ \mathrm{[h^{-1}\ Mpc]}$')
        # Limits
        ax.set_xlim(xy[:, 0].min(), xy[:, 0].max())
        ax.set_ylim(xy[:, 1].min(), xy[:, 1].max())

    # Colorbar
    cbar = fig.colorbar(sc, ax=axs.ravel().tolist(), shrink=0.6, location='right')
    cbar.set_label('densit\u00e0 (relative, log-binned)')

    import datetime
    now = datetime.datetime.now()
    fname = now.strftime("plot_%Y%m%d_%H%M%S") + "_" + str(args.crop_field_start) + "_" + str(args.crop_field_stop)
    outpath = os.path.join(args.output_dir, fname + '.png')
    fig.savefig(outpath, dpi=300)
    print(f"Figura salvata in: {outpath}")


if __name__ == '__main__':
    main()
