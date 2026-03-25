from math import log2, log10, ceil
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm, SymLogNorm
from matplotlib.cm import ScalarMappable
#from ...displacement import dis2pos
plt.rc('text', usetex=False)

from ..models import lag2eul, power


def dis2pos(dis_field,boxsize,Ng): #from displacement.py
    """Assume 'dis_field' is in order of `pid` that aligns with the Lagrangian lattice,
    and dis_field.shape = (3,Ng,Ng,Ng)
    """
    cellsize = boxsize / Ng
    lattice = np.arange(Ng) * cellsize # assume particles are at the corner of the cell

    pos = dis_field.copy()

    pos[2] += lattice
    pos[1] += lattice.reshape(-1, 1)
    pos[0] += lattice.reshape(-1, 1, 1)

    pos[pos<0] += boxsize
    pos[pos>boxsize] -= boxsize

    return pos

def quantize(x):
    return 2 ** round(log2(x), ndigits=1)


def plt_slices(*fields, size=64, title=None, cmap=None, norm=None, **kwargs):
    """Plot slices of fields of more than 2 spatial dimensions.

    Each field should have a channel dimension followed by spatial dimensions,
    i.e. no batch dimension.
    """
    plt.close('all')

    assert all(isinstance(field, torch.Tensor) for field in fields)

    fields = [field.detach().cpu().numpy() for field in fields]

    nc = max(field.shape[0] for field in fields)
    nf = len(fields)

    if title is not None:
        assert len(title) == nf
    cmap = np.broadcast_to(cmap, (nf,))
    norm = np.broadcast_to(norm, (nf,))

    im_size = 2
    cbar_height = 0.2
    fig, axes = plt.subplots(
        nc + 1, nf,
        squeeze=False,
        figsize=(nf * im_size, nc * im_size + cbar_height),
        dpi=100,
        gridspec_kw={'height_ratios': nc * [im_size] + [cbar_height]}
    )

    for f, (field, cmap_col, norm_col) in enumerate(zip(fields, cmap, norm)):
        all_non_neg = np.all(field >= 0)
        all_non_pos = np.all(field <= 0)

        if cmap_col is None:
            if all_non_neg:
                cmap_col = 'inferno'
            elif all_non_pos:
                cmap_col = 'inferno_r'
            else:
                cmap_col = 'RdBu_r'

        if norm_col is None:
            l2, l1, h1, h2 = np.percentile(field, [2.5, 16, 84, 97.5])
            w1, w2 = (h1 - l1) / 2, (h2 - l2) / 2

            if all_non_neg:
                if h1 > 0.1 * h2 or l2 == 0:
                    norm_col = Normalize(vmin=0, vmax=quantize(h2))
                else:
                    norm_col = LogNorm(vmin=quantize(l2), vmax=quantize(h2))
            elif all_non_pos:
                if l1 < 0.1 * l2 or h2 == 0:
                    norm_col = Normalize(vmin=-quantize(-l2), vmax=0)
                else:
                    norm_col = SymLogNorm(linthresh=quantize(-h2),
                                          vmin=-quantize(-l2),
                                          vmax=-quantize(-h2))
            else:
                vlim = quantize(max(-l2, h2))
                if w1 > 0.1 * w2 or l1 * h1 >= 0:
                    norm_col = Normalize(vmin=-vlim, vmax=vlim)
                else:
                    linthresh = quantize(min(-l1, h1))
                    linscale = np.log10(vlim / linthresh)
                    norm_col = SymLogNorm(linthresh=linthresh, linscale=linscale,
                                          vmin=-vlim, vmax=vlim, base=10)

        for c in range(field.shape[0]):
            s = (c,) + tuple(d // 2 for d in field.shape[1:-2])
            if size is None:
                s += (slice(None),) * 2
            else:
                s += (
                    slice(
                        (field.shape[-2] - size) // 2,
                        (field.shape[-2] + size) // 2,
                    ),
                    slice(
                        (field.shape[-1] - size) // 2,
                        (field.shape[-1] + size) // 2,
                    ),
                )

            axes[c, f].pcolormesh(field[s], cmap=cmap_col, norm=norm_col)

            axes[c, f].set_aspect('equal')

            axes[c, f].set_xticks([])
            axes[c, f].set_yticks([])

            if c == 0 and title is not None:
                axes[c, f].set_title(title[f])

        for c in range(field.shape[0], nc):
            axes[c, f].axis('off')

        fig.colorbar(
            ScalarMappable(norm=norm_col, cmap=cmap_col),
            cax=axes[-1, f],
            orientation='horizontal',
        )

    fig.tight_layout()

    return fig


def plt_power(*fields, dis=None, label=None, **kwargs):
    """Plot power spectra of fields.

    Each field should have batch and channel dimensions followed by spatial
    dimensions.

    Optionally the field can be transformed by lag2eul first if given `dis`.

    See `map2map.models.power`.
    """
    plt.close('all')

    if label is not None:
        assert len(label) == len(fields) or len(label) == len(dis)
    else:
        label = [None] * len(fields)

    with torch.no_grad():
        if dis is not None:
            fields = lag2eul(dis, val=fields, **kwargs)

        ks, Ps = [], []
        for field in fields:
            k, P, _ = power(field)
            ks.append(k)
            Ps.append(P)

    ks = [k.cpu().numpy() for k in ks]
    Ps = [P.cpu().numpy() for P in Ps]

    #normalize powers
    #Ps = [P / np.max(P) for P in Ps]

    #print('wavenumbers:', ks)
    #print('powers:', Ps)

    fig, axes = plt.subplots(figsize=(4.8, 3.6), dpi=150)

    for k, P, l in zip(ks, Ps, label):
        axes.loglog(k, P, label=l, alpha=0.7)

    axes.legend()
    axes.set_xlabel('unnormalized wavenumber')
    axes.set_ylabel('unnormalized power')

    fig.tight_layout()

    return fig


def plt_pos_projections(*fields, boxsize, Ng, labels=None, **kwargs):
    """
    Converts displacement fields to position fields and plots their 2D projections in the X-Y, X-Z and Y-Z planes, as scatter plots.
    Each of 3 projection per field is plotted in a separate column, labeled accordingly to the labels argument.

    Each field should have a channel dimension with 3 channels, followed by 3 spatial dimensions.
    Example of field: shape: (3, 64, 64, 64) where 3 channels are the 3 components of the displacement field.
    """

    plt.close('all')

    assert all(isinstance(field, torch.Tensor) for field in fields)
    assert all(field.shape[0] == 3 for field in fields), "Each field should have 3 channels (displacement components)."

    fields = [field.detach().cpu().numpy() for field in fields]

    nf = len(fields)

    if labels is not None:
        assert len(labels) == nf
    else:
        labels = [None] * nf

    im_size = 4
    fig, axes = plt.subplots(
        1, nf,
        squeeze=False,
        figsize=(nf * im_size, im_size),
        dpi=100,
    )

    for f, (field, label) in enumerate(zip(fields, labels)):
        pos_field = dis2pos(field, boxsize, Ng)

        # XY projection
        axes[0, f].scatter(pos_field[0].flatten(), pos_field[1].flatten(), s=0.005, alpha=0.5)
        axes[0, f].set_title(f'norm XY Projection {label if label else ""}')
        axes[0, f].set_aspect('equal')  

        """
        # XZ projection
        axes[1, f].scatter(pos_field[0].flatten(), pos_field[2].flatten(), s=0.005, alpha=0.5)
        axes[1, f].set_title(f'norm XZ Projection {label if label else ""}')
        axes[1, f].set_aspect('equal')

        # YZ projection
        axes[2, f].scatter(pos_field[1].flatten(), pos_field[2].flatten(), s=0.005, alpha=0.5)
        axes[2, f].set_title(f'norm YZ Projection {label if label else ""}')
        axes[2, f].set_aspect('equal')
        """
    fig.tight_layout()
    return fig


def test_plt_projections():
    #create a random displacement field with shape (3,64,64,64)
    field = torch.randn(3, 64, 64, 64)
    field2 = torch.randn(3, 64, 64, 64) * 0.9
    field3 = torch.randn(3, 64, 64, 64) * 1.1

    boxsize = 50  # Mpc/h
    Ng = 64

    fig = plt_pos_projections(field, field2, field3, boxsize=boxsize, Ng=Ng, labels=['Random Field', 'Random Field 2', 'Random Field 3'])

    #save fig
    import os
    os.makedirs('plots_temp', exist_ok=True)
    fig.savefig('plots_temp/pos_projections.png', bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    test_plt_projections()
    quit()
    #test plt slices on DEMNUni_512_to_1024_averageDS_crop/train/
    from ..data.norms import cosmology

    """
    #load DEMNUni_512_to_1024_averageDS_crop/train/LR/seed_123465_dis_crop_0000_0000_0000.npy
    dis_LR = np.load('DEMNUni_512_to_1024_averageDS_crop/train/LR/seed_123465_dis_crop_0000_0000_0000.npy', mmap_mode='r')
    vel_LR = np.load('DEMNUni_512_to_1024_averageDS_crop/train/LR/seed_123465_vel_crop_0000_0000_0000.npy', mmap_mode='r')

    dis_HR = np.load('DEMNUni_512_to_1024_averageDS_crop/train/HR/seed_123465_dis_crop_0000_0000_0000.npy', mmap_mode='r')
    vel_HR = np.load('DEMNUni_512_to_1024_averageDS_crop/train/HR/seed_123465_vel_crop_0000_0000_0000.npy', mmap_mode='r')
    """

    dis_LR = np.load('DEMNUni_512_to_1024_average/train/LR/seed_123465_dis.npy', mmap_mode='r')
    vel_LR = np.load('DEMNUni_512_to_1024_average/train/LR/seed_123465_vel.npy', mmap_mode='r')

    dis_HR = np.load('DEMNUni_512_to_1024_average/train/HR/seed_123465_dis.npy', mmap_mode='r')
    vel_HR = np.load('DEMNUni_512_to_1024_average/train/HR/seed_123465_vel.npy', mmap_mode='r')

    #print min and max of vel_HR
    print('vel_HR min:', vel_HR.min())
    print('vel_HR max:', vel_HR.max())

    #crop a random 3x32x32x32 patch from each LR. Remember that LR dis (or vel) has shape [3, 132, 132, 132]
    crop_size = 32

    i = np.random.randint(0, 132 - crop_size)
    j = np.random.randint(0, 132 - crop_size)
    k = np.random.randint(0, 132 - crop_size)
    dis_LR = dis_LR[:, i:i + crop_size, j:j + crop_size, k:k + crop_size]
    vel_LR = vel_LR[:, i:i + crop_size, j:j + crop_size, k:k + crop_size]

    dis_HR = dis_HR[:, i*2:i*2 + crop_size*2, j*2:j*2 + crop_size*2, k*2:k*2 + crop_size*2]
    vel_HR = vel_HR[:, i*2:i*2 + crop_size*2, j*2:j*2 + crop_size*2, k*2:k*2 + crop_size*2]

    #create a copy of the read values (since we don't want to modify the file)
    dis_LR = dis_LR.copy()
    vel_LR = vel_LR.copy()
    dis_HR = dis_HR.copy()
    vel_HR = vel_HR.copy()

    #apply cosmology dis and vel
    #cosmology.dis(dis_LR)
    #cosmology.dis(dis_HR)
    #cosmology.vel(vel_LR)
    #cosmology.vel(vel_HR)

    #to torch tensor
    dis_LR = torch.from_numpy(dis_LR).float()
    vel_LR = torch.from_numpy(vel_LR).float()
    dis_HR = torch.from_numpy(dis_HR).float()
    vel_HR = torch.from_numpy(vel_HR).float()

    #concatenate in tensors with 6 channels
    dis_vel_HR = torch.cat([dis_HR, vel_HR], dim=1)
    dis_vel_LR = torch.cat([dis_LR, vel_LR], dim=1)

    from ..models.resample import resample

    dis_vel_LR = resample(dis_vel_LR,2, narrow=False)

    fig = plt_slices(
                dis_vel_LR, dis_vel_HR,
                title=['in', 'tgt'],
    )

    #save fig
    import os
    os.makedirs('plots_temp', exist_ok=True)
    fig.savefig('plots_temp/slices.png', bbox_inches='tight')
    plt.close(fig)
