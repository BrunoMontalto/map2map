import numpy as np
from scipy.ndimage import zoom

def random_sample_fixed_seed(seed):
    """ returns a random_sample function with fixed seed """

    def wrapper(x, factor):
        return random_sample(x, factor, seed=seed)

    return wrapper

def random_sample(x, factor, seed=42):
    channels = x.shape[0]
    Ng_lr = x.shape[1] // factor  

    # store old random state
    old_state = np.random.get_state()

    np.random.seed(seed)
    # generate random offset for each dimension
    offsets = np.random.randint(0, factor, size=(3, Ng_lr, Ng_lr, Ng_lr))

    # restore old random state
    np.random.set_state(old_state)
    
    # sample
    grid_indices = np.indices((Ng_lr, Ng_lr, Ng_lr))
    i_indices = grid_indices[0] * factor + offsets[0]
    j_indices = grid_indices[1] * factor + offsets[1]
    k_indices = grid_indices[2] * factor + offsets[2]

    
    downsampled = x[:, i_indices, j_indices, k_indices]
    
    return downsampled


def unstructured_random_sample_fixed_seed(seed):
    """ returns an unstructured_random_sample function with fixed seed """

    def wrapper(x, factor):
        return unstructured_random_sample(x, factor, seed=seed)

    return wrapper


def unstructured_random_sample(x, factor, seed=42):
    channels, Ng, _, _ = x.shape
    Ng_lr = Ng // factor

    # store old random state
    old_state = np.random.get_state()

    np.random.seed(seed)
    # generate random offset for each dimension
    total_points = Ng_lr ** 3
    all_indices = np.array(np.meshgrid(np.arange(Ng), np.arange(Ng), np.arange(Ng), indexing='ij')).reshape(3, -1).T
    chosen_indices = all_indices[np.random.choice(all_indices.shape[0], size=total_points, replace=False)]
    
    i_indices = chosen_indices[:, 0].reshape(Ng_lr, Ng_lr, Ng_lr)
    j_indices = chosen_indices[:, 1].reshape(Ng_lr, Ng_lr, Ng_lr)
    k_indices = chosen_indices[:, 2].reshape(Ng_lr, Ng_lr, Ng_lr)

    # restore old random state
    np.random.set_state(old_state)
    
    # sample
    downsampled = x[:, i_indices, j_indices, k_indices]
    
    return downsampled


def average_downsample(x, factor):
    channels, Ng, _, _ = x.shape
    assert Ng % factor == 0, 'Ng must be divisible by factor'
    
    new_shape = (channels,
                 Ng // factor, factor,
                 Ng // factor, factor,
                 Ng // factor, factor)
    
    x_reshaped = x.reshape(new_shape)
    
    x_downsampled = x_reshaped.mean(axis=(2, 4, 6))
    
    return x_downsampled


def downsample_tricubic(x, factor):
    channels, Nx, Ny, Nz = x.shape

    Nx_LR = Nx // factor
    Ny_LR = Ny // factor
    Nz_LR = Nz // factor

    zoom_factors = (1, Nx_LR / Nx, Ny_LR / Ny, Nz_LR / Nz)

    downsampled = np.empty((channels, Nx_LR, Ny_LR, Nz_LR), dtype=x.dtype)

    for c in range(channels):
        downsampled[c] = zoom(x[c], zoom=zoom_factors[1:], order=3, mode='grid-wrap') #grid wrap is ideal for periodic BC

    return downsampled

def random_sample_with_batch(x, factor):
    batch_size, channels, N, _, _ = x.shape
    Ng_lr = N // factor

    # generate random offset for each dimension
    offsets = np.random.randint(0, factor, size=(3, Ng_lr, Ng_lr, Ng_lr))

    # sample
    grid_indices = np.indices((Ng_lr, Ng_lr, Ng_lr))
    i_indices = grid_indices[0] * factor + offsets[0]
    j_indices = grid_indices[1] * factor + offsets[1]
    k_indices = grid_indices[2] * factor + offsets[2]


    downsampled = np.empty((batch_size, channels, Ng_lr, Ng_lr, Ng_lr), dtype=x.dtype)
    for b in range(batch_size):
        downsampled[b] = x[b, :, i_indices, j_indices, k_indices]

    return downsampled


def compute_metrics(original, downsampled, factor): # for unit testing
    expected_sum = np.sum(original) / (factor ** 3)
    expected_mean = np.mean(original)
    actual_sum = np.sum(downsampled)
    actual_mean = np.mean(downsampled)
    mae = np.mean(np.abs(downsampled - expected_mean))

    ssim_scores = []
    for c in range(original.shape[0]):
        upsampled = zoom(downsampled[c], factor, order=3)
        data_range = original[c].max() - original[c].min()
        if data_range == 0:
            ssim_scores.append(1.0 if np.allclose(original[c], upsampled) else 0.0)
        else:
            ssim_score = ssim(
                original[c],
                upsampled,
                data_range=data_range
            )
            ssim_scores.append(ssim_score)
    mean_ssim = np.mean(ssim_scores)

    return {
        'expected_sum': expected_sum,
        'actual_sum': actual_sum,
        'sum_error': abs(expected_sum - actual_sum),
        'expected_mean': expected_mean,
        'actual_mean': actual_mean,
        'mean_error': abs(expected_mean - actual_mean),
        'mae_vs_expected_mean': mae,
        'ssim': mean_ssim
    }

def print_metrics(name, metrics):
    print(f"\n{name} Metrics:")
    print(f"  Expected sum:       {metrics['expected_sum']:.6f}")
    print(f"  Actual sum:         {metrics['actual_sum']:.6f}")
    print(f"  Sum error:          {metrics['sum_error']:.6f}")
    print(f"  Expected mean:      {metrics['expected_mean']:.6f}")
    print(f"  Actual mean:        {metrics['actual_mean']:.6f}")
    print(f"  Mean error:         {metrics['mean_error']:.6f}")
    print(f"  MAE vs expected μ:  {metrics['mae_vs_expected_mean']:.6f}")
    print(f"  Structural SSIM:    {metrics['ssim']:.6f}")

if __name__ == "__main__":
    from skimage.metrics import structural_similarity as ssim
    np.random.seed(42)

    print("=== Structured Test ===")
    x = np.zeros((3, 16, 16, 16))
    template = np.zeros((3, 2, 2, 2))
    template[0, 0, 0, 0] = 1
    template[0, 1, 0, 0] = 1
    template[0, 0, 1, 0] = 1
    template[0, 0, 0, 1] = 1
    for i in range(8):
        x[:, (i//4)*2:(i//4)*2+2, ((i%4)//2)*2:((i%4)//2)*2+2, (i%2)*2:(i%2)*2+2] = template

    factor = 2
    x_rs = random_sample(x, factor)
    x_tc = downsample_tricubic(x, factor)
    x_avg = average_downsample(x, factor)

    rs_metrics = compute_metrics(x, x_rs, factor)
    tc_metrics = compute_metrics(x, x_tc, factor)
    avg_metrics = compute_metrics(x, x_avg, factor)

    print_metrics("Random Sampling (Structured)", rs_metrics)
    print_metrics("Tricubic Downsampling (Structured)", tc_metrics)
    print_metrics("Average Downsampling (Structured)", avg_metrics)

    print("\n=== Gaussian Blob Test ===")
    channels, size = 3, 64
    x = np.zeros((channels, size, size, size))
    coords = np.linspace(-1, 1, size)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
    centers = [(0, 0, 0), (0.3, -0.3, 0), (-0.5, 0.5, 0.2)]
    sigma = 0.2
    for c, (cx, cy, cz) in enumerate(centers):
        x[c] = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2) / (2 * sigma ** 2))

    for factor in [2, 4]:
        print(f"\n-- Downsampling Factor: {factor} --")
        x_rs = random_sample(x, factor)
        x_tc = downsample_tricubic(x, factor)
        x_avg = average_downsample(x, factor)

        rs_metrics = compute_metrics(x, x_rs, factor)
        tc_metrics = compute_metrics(x, x_tc, factor)
        avg_metrics = compute_metrics(x, x_avg, factor)

        print_metrics(f"Random Sampling (Gaussian, factor={factor})", rs_metrics)
        print_metrics(f"Tricubic Downsampling (Gaussian, factor={factor})", tc_metrics)
        print_metrics(f"Average Downsampling (Gaussian, factor={factor})", avg_metrics)

    print("\n=== Random Data Test ===")
    x = np.random.rand(3, 64, 64, 64)
    for factor in [2, 4]:
        print(f"\n-- Downsampling Factor: {factor} --")
        x_rs = random_sample(x, factor)
        x_tc = downsample_tricubic(x, factor)
        x_avg = average_downsample(x, factor)

        rs_metrics = compute_metrics(x, x_rs, factor)
        tc_metrics = compute_metrics(x, x_tc, factor)
        avg_metrics = compute_metrics(x, x_avg, factor)

        print_metrics(f"Random Sampling (Random, factor={factor})", rs_metrics)
        print_metrics(f"Tricubic Downsampling (Random, factor={factor})", tc_metrics)
        print_metrics(f"Average Downsampling (Random, factor={factor})", avg_metrics)