#!/usr/bin/env python3
"""
Versione di comparison_plots_default.py modificata che crea due righe di plot (prima: proiezione XY completa; seconda: zoom x4
in una regione quadrata scelta con euristica per avere molte particelle).
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Rectangle

from displacement import dis2pos, dis2posLR_average_downsample

def find_top_density_centers(xy_positions, nbins=200, num_clusters=1, min_distance=None):
    """Returns up to num_clusters centers (xc, yc) of the most dense regions, discarding ones that are too close (under min_distance).
    """
    x = xy_positions[:, 0]
    y = xy_positions[:, 1]
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    H, xedges, yedges = np.histogram2d(x, y, bins=nbins, range=[[xmin, xmax], [ymin, ymax]])

    flat_indices = np.argsort(H, axis=None)[::-1]  # decrescente per densità

    centers = []
    for idx in flat_indices:
        if len(centers) >= num_clusters:
            break
        i, j = np.unravel_index(idx, H.shape)
        xc = 0.5 * (xedges[i] + xedges[i + 1])
        yc = 0.5 * (yedges[j] + yedges[j + 1])

        # Minimum distance check
        if min_distance is not None and centers:
            too_close = any(
                np.hypot(xc - cx, yc - cy) < min_distance for (cx, cy) in centers
            )
            if too_close:
                continue

        centers.append((xc, yc))
    return centers


def load_numpy_memmap(path):
    if path.endswith('.npy'):
        arr = np.load(path, mmap_mode='r')
        return arr
    elif path.endswith('.npz'):
        npz = np.load(path)
        keys = list(npz.files)
        if len(keys) == 0:
            raise ValueError(f"File npz vuoto: {path}")
        arr = npz[keys[0]]
        return arr
    else:
        arr = np.load(path, mmap_mode='r')
        return arr


def wrapped_take_3d(arr, ix, iy, iz):
    tmp = np.take(arr, ix, axis=1, mode='wrap')
    tmp = np.take(tmp, iy, axis=2, mode='wrap')
    tmp = np.take(tmp, iz, axis=3, mode='wrap')
    return tmp



def read_block_with_periodic(arr, start, stop):
    print("start:", start)
    print("stop:", stop)
    N = arr.shape[1]
    length = stop - start
    if length <= 0:
        raise ValueError("stop must be > start")
    ix = np.arange(start, start + length) % N
    iy = np.arange(start, start + length) % N
    iz = np.arange(start, start + length) % N
    block = wrapped_take_3d(arr, ix, iy, iz)
    print("block min:", block.min())
    print("block min:", block.max())
    return block

def compute_density_colors(xyz_positions, nbins=100):
    x = xyz_positions[:, 0]
    y = xyz_positions[:, 1]
    z = xyz_positions[:, 2]

    ranges = [
        (x.min(), x.max()),
        (y.min(), y.max()),
        (z.min(), z.max())
    ]

    H, edges = np.histogramdd(xyz_positions, bins=nbins, range=ranges)

    inds = []
    for i in range(3):
        inds.append(np.searchsorted(edges[i], xyz_positions[:, i], side='right') - 1)
    inds = np.stack(inds, axis=1)

    for i in range(3):
        inds[:, i] = np.clip(inds[:, i], 0, H.shape[i] - 1)

    dens = H[inds[:, 0], inds[:, 1], inds[:, 2]].astype(float)

    if np.any(dens > 0):
        dens[dens <= 0] = dens[dens > 0].min()
    else:
        dens[:] = 1.0

    dens_log = np.log10(dens)
    return dens_log


def _compute_density_colors_2d(xy_positions, nbins=200):
    x = xy_positions[:, 0]
    y = xy_positions[:, 1]
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    if xmax == xmin:
        xmax = xmin + 1.0
    if ymax == ymin:
        ymax = ymin + 1.0
    H, xedges, yedges = np.histogram2d(x, y, bins=nbins, range=[[xmin, xmax], [ymin, ymax]])
    xi = np.searchsorted(xedges, x, side='right') - 1
    yi = np.searchsorted(yedges, y, side='right') - 1
    xi = np.clip(xi, 0, H.shape[0] - 1)
    yi = np.clip(yi, 0, H.shape[1] - 1)
    dens = H[xi, yi].astype(float)
    if np.any(dens > 0):
        dens[dens <= 0] = dens[dens > 0].min()
    else:
        dens[:] = 1.0
    dens_log = np.log10(dens)
    return dens_log



def process_file(path, scale, crop_start, crop_stop, margin=20, boxsize=None, is_LR=False, original_lr_res=None):
    """
    scale: 1 for LR, 2 for HR
    crop_start & crop_end: crop starts from (crop_start*scale)^3 and ends at (crop_stop*scale)^3
    """

    # Load array
    arr = load_numpy_memmap(path)
    if arr.ndim != 4 or arr.shape[0] != 3:
        raise ValueError(f"Array in {path} deve avere shape (3, N, N, N). Got {arr.shape}")
    N = arr.shape[1]

    # Scale crop with scale argument
    s = int(crop_start * scale)
    e = int(crop_stop * scale)

    # Calculate range of allowed range of values accordingly to crop box size, scaled in cell units (boxsize / original_lr_res)
    # NOTE: boxsize is the original boxsize (should rename)
    mins = 0 #boxsize / original_lr_res * (crop_start)
    maxs = boxsize / original_lr_res * (crop_stop-crop_start)

    print("mins:", mins)
    print("maxs:", maxs)

    # Read a bigger block (according to margin) to find particles that are allocated outside the crop but are inside the cropped volume
    s2 = s - margin * scale
    e2 = e + margin * scale
    boxsize_scaled = boxsize * (crop_stop - crop_start + 2 * margin) / original_lr_res #scale boxsize accordingly to the new size
    print("boxsize_scaled:", boxsize_scaled)
    block2 = read_block_with_periodic(arr, s2, e2)
    if scale == 1:
        pos_block2 = dis2posLR_average_downsample(block2, Ng_LR=(e2 - s2), Ng_HR=(e2 - s2) * 2, boxsize=boxsize_scaled)
    else:
        pos_block2 = dis2pos(block2, Ng=(e2 - s2), boxsize=boxsize_scaled)

    pos_block2 = np.moveaxis(pos_block2, 0, -1).reshape(-1, 3)


    mask = (
        (pos_block2[:, 0] >= mins) & (pos_block2[:, 0] <= maxs) &
        (pos_block2[:, 1] >= mins) & (pos_block2[:, 1] <= maxs) &
        (pos_block2[:, 2] >= mins) & (pos_block2[:, 2] <= maxs)
    )
    selected = pos_block2[mask]
    return selected


def find_high_density_center(xy_positions, nbins=200):
    x = xy_positions[:, 0]
    y = xy_positions[:, 1]
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    H, xedges, yedges = np.histogram2d(x, y, bins=nbins, range=[[xmin, xmax], [ymin, ymax]])
    
    #idx = np.unravel_index(np.argmax(H), H.shape)
    flat_indices = np.argsort(H, axis=None)  # ordina tutti gli elementi in ordine crescente
    second_max_index = flat_indices[-4]      # -1 è il massimo, -2 il secondo massimo
    idx = np.unravel_index(second_max_index, H.shape)


    xc = 0.5 * (xedges[idx[0]] + xedges[idx[0] + 1])
    yc = 0.5 * (yedges[idx[1]] + yedges[idx[1] + 1])
    return xc, yc


def main():
    parser = argparse.ArgumentParser(description='Crea plot comparativo LR-HR-SR (proiezione XY) con zoom.')
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
    parser.add_argument('--num-clusters', type=int, default=1)
    parser.add_argument('--zoom-on-filaments', action="store_true")
    args = parser.parse_args()

    files = {'LR': args.file_lr, 'HR': args.file_hr, 'SR': args.file_sr}

    shapes = {}
    Ns = {}
    for k, p in files.items():
        arr = load_numpy_memmap(p)
        if arr.ndim != 4 or arr.shape[0] != 3:
            raise ValueError(f"{p} deve essere un array NumPy con shape (3,N,N,N). Got {arr.shape}.")
        if 'SR' in k:
            Ns[k] = Ns['HR']
        else:
            Ns[k] = arr.shape[1]
        shapes[k] = arr.shape

    min_N = min(Ns.values())
    scales = {k: Ns[k] // min_N for k in Ns}
    for k in scales:
        if Ns[k] % min_N != 0:
            print(f"Warning: N for {k} (= {Ns[k]}) is not an integer multiple of min N (={min_N}). Using integer division for scale.")

    os.makedirs(args.output_dir, exist_ok=True)

    results = {}
    for k, p in files.items():
        print(f"Processing {k} ({p}) with scale {scales[k]} ...")
        sel = process_file(p, scale=scales[k], crop_start=args.crop_field_start, crop_stop=args.crop_field_stop, margin=args.margin, boxsize=args.original_boxsize, original_lr_res = args.original_lr_res)
        results[k] = sel
        print(f" -> selected {len(sel)} particles for {k}")

    rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'figure.figsize': (15, 10),
        'legend.fontsize': 9,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'axes.linewidth': 0.8
    })

    #lr_pos = results['LR']
    #if lr_pos.size == 0:
    #    cx, cy = 0.5*args.original_boxsize, 0.5*args.original_boxsize
    #else:
    #    cx, cy = find_high_density_center(lr_pos[:, :2], nbins=min(300, args.nbins))

    lr_pos = results['LR']
    if lr_pos.size == 0:
        centers = [(0.5 * args.original_boxsize, 0.5 * args.original_boxsize)]
    else:
        # Choose min distance as 75% of boxsize
        all_pos = np.vstack([results[k] for k in results if results[k].size > 0])
        full_xlim = (all_pos[:, 0].min(), all_pos[:, 0].max())
        full_ylim = (all_pos[:, 1].min(), all_pos[:, 1].max())
        base_side = min(full_xlim[1] - full_xlim[0], full_ylim[1] - full_ylim[0]) / 10.0
        min_dist = 0.75 * base_side
        centers = find_top_density_centers(
            lr_pos[:, :2], nbins=min(300, args.nbins),
            num_clusters=args.num_clusters, min_distance=min_dist
        )


    all_pos = np.vstack([results[k] for k in results if results[k].size > 0])
    full_xlim = (all_pos[:, 0].min(), all_pos[:, 0].max())
    full_ylim = (all_pos[:, 1].min(), all_pos[:, 1].max())

    base_side = min(full_xlim[1] - full_xlim[0], full_ylim[1] - full_ylim[0]) / 10.0

    # Prepare zoom areas
    zoom_boxes = []
    for (cx, cy) in centers:
        side = base_side
        xmin = max(full_xlim[0], cx - side / 2.0)
        xmax = min(full_xlim[1], cx + side / 2.0)
        ymin = max(full_ylim[0], cy - side / 2.0)
        ymax = min(full_ylim[1], cy + side / 2.0)
        zoom_boxes.append((xmin, xmax, ymin, ymax))

    # Filament zoom
    filament_box = None
    if args.zoom_on_filaments:
        # first dense region that does not overlap with other selected clusters
        big_side = base_side * 3.0
        fc = find_top_density_centers(lr_pos[:, :2], nbins=min(300, args.nbins), num_clusters=5)
        for (fx, fy) in fc:
            candidate = (fx - big_side / 2, fx + big_side / 2, fy - big_side / 2, fy + big_side / 2)
            # Overlap check
            overlap = any(
                not (candidate[1] < xb[0] or candidate[0] > xb[1] or candidate[3] < xb[2] or candidate[2] > xb[3])
                for xb in zoom_boxes
            )
            if not overlap:
                xmin = max(full_xlim[0], candidate[0])
                xmax = min(full_xlim[1], candidate[1])
                ymin = max(full_ylim[0], candidate[2])
                ymax = min(full_ylim[1], candidate[3])
                filament_box = (xmin, xmax, ymin, ymax)
                break

    # #rows: 1  + num_clusters + 1 for filaments (optional)
    nrows = 1 + len(zoom_boxes) + (1 if filament_box else 0)
    #fig, axs = plt.subplots(nrows, 3, constrained_layout=True)
    fig, axs = plt.subplots(nrows, 3, figsize=(15, 5.5 * nrows))
    plt.tight_layout(pad=0.3, w_pad=0.1, h_pad=0.3)
    if nrows == 2:
        axs = np.array(axs).reshape(nrows, 3)

    kinds = ['LR', 'HR', 'SR']
    sc_ref = None

    # First row
    for col, kind in enumerate(kinds):
        ax = axs[0, col]
        ax.set_facecolor('black')
        pos = results[kind]
        ax.set_title(kind)
        if pos.size != 0:
            xy = pos[:, :2]
            dens = compute_density_colors(pos, nbins=args.nbins)
            dmin, dmax = dens.min(), dens.max()
            norm = (dens - dmin) / (dmax - dmin) if dmax != dmin else np.zeros_like(dens)
            s = args.particle_size_lr / 4 if kind == 'LR' else args.particle_size_hr / 4
            alpha = args.alpha_lr if kind == 'LR' else args.alpha_hr
            sc = ax.scatter(xy[:, 0], xy[:, 1], c=norm, s=s, alpha=alpha, cmap='viridis', rasterized=True)
            sc_ref = sc if sc_ref is None else sc_ref
            ax.set_xlim(full_xlim)
            ax.set_ylim(full_ylim)
            # Zoom
            for i, (xmin, xmax, ymin, ymax) in enumerate(zoom_boxes, start=1):
                rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=1.5, edgecolor='palegreen', facecolor='none')
                ax.add_patch(rect)
                ax.text(xmin, ymin, f'c{i}', color='palegreen', fontsize=9, va='top', ha='right')
            if filament_box:
                fxmin, fxmax, fymin, fymax = filament_box
                rect = Rectangle((fxmin, fymin), fxmax - fxmin, fymax - fymin,
                                 linewidth=1.5, edgecolor='orange', facecolor='none', linestyle='--')
                ax.add_patch(rect)
                ax.text(fxmin, fymax, 'filaments', color='orange', fontsize=9, va='bottom', ha='left')
        else:
            ax.text(0.5, 0.5, 'No particles', ha='center', va='center', color='white')
        ax.set_aspect('equal')
        if col == 0:
            ax.set_ylabel(r'$Y\ \mathrm{[h^{-1}\ Mpc]}$')
        ax.tick_params(color='palegreen', width=1)
        for spine in ax.spines.values():
            spine.set_edgecolor('palegreen')
            spine.set_linewidth(1)

    # Clusters rows
    for i, (xmin, xmax, ymin, ymax) in enumerate(zoom_boxes, start=1):
        for col, kind in enumerate(kinds):
            axz = axs[i, col]
            axz.set_facecolor('black')
            axz.set_title(f'{kind} (zoom c{i})')
            pos = results[kind]
            if pos.size != 0:
                maskz = (pos[:, 0] >= xmin) & (pos[:, 0] <= xmax) & (pos[:, 1] >= ymin) & (pos[:, 1] <= ymax)
                posz = pos[maskz]
                if posz.size != 0:
                    densz = compute_density_colors(posz, nbins=max(50, args.nbins // 2))
                    dzmin, dzmax = densz.min(), densz.max()
                    normz = (densz - dzmin) / (dzmax - dzmin) if dzmax != dzmin else np.zeros_like(densz)
                    sz = args.particle_size_lr * 4 if kind == 'LR' else args.particle_size_hr * 4
                    alphaz = args.alpha_lr if kind == 'LR' else args.alpha_hr
                    axz.scatter(posz[:, 0], posz[:, 1], c=normz, s=sz, alpha=alphaz, cmap='viridis', rasterized=True)
                else:
                    axz.text(0.5, 0.5, 'No particles in zoom', ha='center', va='center', color='white')
                axz.set_xlim((xmin, xmax))
                axz.set_ylim((ymin, ymax))
            axz.set_aspect('equal')
            if col == 0:
                axz.set_ylabel(r'$Y\ \mathrm{[h^{-1}\ Mpc]}$')
            axz.set_xlabel(r'$X\ \mathrm{[h^{-1}\ Mpc]}$')
            axz.tick_params(color='palegreen', width=1)
            for spine in axz.spines.values():
                spine.set_edgecolor('palegreen')
                spine.set_linewidth(1)

    # Filaments row
    if filament_box:
        i = len(zoom_boxes) + 1
        (xmin, xmax, ymin, ymax) = filament_box
        for col, kind in enumerate(kinds):
            axz = axs[i, col]
            axz.set_facecolor('black')
            axz.set_title(f'{kind} (filaments zoom)')
            pos = results[kind]
            if pos.size != 0:
                maskz = (pos[:, 0] >= xmin) & (pos[:, 0] <= xmax) & (pos[:, 1] >= ymin) & (pos[:, 1] <= ymax)
                posz = pos[maskz]
                if posz.size != 0:
                    densz = compute_density_colors(posz, nbins=max(50, args.nbins // 2))
                    dzmin, dzmax = densz.min(), densz.max()
                    normz = (densz - dzmin) / (dzmax - dzmin) if dzmax != dzmin else np.zeros_like(densz)
                    sz = args.particle_size_lr * 3 if kind == 'LR' else args.particle_size_hr * 3
                    alphaz = args.alpha_lr if kind == 'LR' else args.alpha_hr
                    axz.scatter(posz[:, 0], posz[:, 1], c=normz, s=sz, alpha=alphaz, cmap='viridis', rasterized=True)
                else:
                    axz.text(0.5, 0.5, 'No particles in zoom', ha='center', va='center', color='white')
                axz.set_xlim((xmin, xmax))
                axz.set_ylim((ymin, ymax))
            axz.set_aspect('equal')
            if col == 0:
                axz.set_ylabel(r'$Y\ \mathrm{[h^{-1}\ Mpc]}$')
            axz.set_xlabel(r'$X\ \mathrm{[h^{-1}\ Mpc]}$')
            axz.tick_params(color='orange', width=1)
            for spine in axz.spines.values():
                spine.set_edgecolor('orange')
                spine.set_linewidth(1)


    if sc_ref is not None and False:
        cbar = fig.colorbar(sc_ref, ax=axs.ravel().tolist(), shrink=0.7, location='right')
        cbar.set_label('densità (relative, log-binned)')

    import datetime
    now = datetime.datetime.now()
    fname = now.strftime("plot_zoom_%Y%m%d_%H%M%S") + os.path.basename(args.file_sr) + f"_{args.crop_field_start}_{args.crop_field_stop}"
    outpath = os.path.join(args.output_dir, fname + '.png')
    fig.savefig(outpath, dpi=300)
    print(f"Figura salvata in: {outpath}")


if __name__ == '__main__':
    main()
