import numpy as np
from scipy.ndimage import zoom

def random_sample(x, factor):
    channels = x.shape[0]
    Ng_lr = x.shape[1] // factor  # Nuova dimensione della griglia a bassa risoluzione

    # Genera un array di indici casuali per ciascun asse (x, y, z)
    offsets = np.random.randint(0, factor, size=(3, Ng_lr, Ng_lr, Ng_lr))
    
    # Crea una griglia di indici ridotti (in base al fattore 'factor')
    grid_indices = np.indices((Ng_lr, Ng_lr, Ng_lr))
    i_indices = grid_indices[0] * factor + offsets[0]
    j_indices = grid_indices[1] * factor + offsets[1]
    k_indices = grid_indices[2] * factor + offsets[2]

    # Usa gli indici per campionare l'array originale
    downsampled = x[:, i_indices, j_indices, k_indices]
    
    return downsampled


def downsample_tricubic(x, factor):
    channels, Nx, Ny, Nz = x.shape

    Nx_LR = Nx // factor
    Ny_LR = Ny // factor
    Nz_LR = Nz // factor

    zoom_factors = (1, Nx_LR / Nx, Ny_LR / Ny, Nz_LR / Nz)

    # scipy.ndimage.zoom richiede input shape senza canale, quindi facciamo un loop sui canali
    downsampled = np.empty((channels, Nx_LR, Ny_LR, Nz_LR), dtype=x.dtype)

    for c in range(channels):
        downsampled[c] = zoom(x[c], zoom=zoom_factors[1:], order=3)

    return downsampled