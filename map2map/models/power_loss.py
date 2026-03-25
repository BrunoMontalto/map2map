import torch
import torch.nn as nn
import torch.nn.functional as F
from .lag2eul import lag2eul
from .power import power

def softbincount(x, weights, max_bin=None): #TODO temperature
    bins = torch.arange(1, max_bin + 1, device=x.device)  # list of bins [1, 2, ..., max_bin]
    bins = bins.to(torch.float32)  # ensure bins are same dtype as x
    
    #calculate a tensor of (number of values) rows and (number of bins) columns, where each element is the absolute difference between the value and the bin
    diff = x[:, None] - bins[None, :]  # (B, M) where B is the batch size and M is the number of bins


    diff = 1.7376739236 * (diff + 1/2)
    soft_counts = 1 / (diff.pow(10) + diff.pow(2) + 1)  # polynomial 2

    weighted_counts = (soft_counts * weights[:, None]).sum(dim=0)  # (B,)

    return weighted_counts

    
def power_differentiable(x):
    # get spatial dimensions
    signal_ndim = x.dim() - 2 
    signal_size = x.shape[-signal_ndim:]

    kmax = min(s for s in signal_size) // 2 # Nyquist frequency
    even = x.shape[-1] % 2 == 0 # batch size is even check

    x = torch.fft.rfftn(x, s=signal_size) #compute the rfftn (R)
    
    P = x.real.square() + x.imag.square() # compute power: real^2 + imag^2


    P = P.mean(dim=0) # average over batch
    P = P.sum(dim=0) # sum over channels

    #return None, P.flatten(), None # no binning test


    k = [torch.arange(d, dtype=torch.float32, device=P.device)
         for d in P.shape] # create a list of ranges for each dimension. if tensor has shape [64, 64, 33], then k will be a list of tensors [[0, 1, ..., 63], [0, 1, ..., 63], [0, 1, ..., 32]]
    k = [j - len(j) * (j > len(j) // 2) for j in k[:-1]] + [k[-1]] #For the first n-1 dimensions, center the range around zero, for the last dimension keep it as is. 
    
    #This is to ensure that the wavevectors are centered around zero in the first n-1 dimensions, while the last dimension remains positive (as it corresponds to frequencies in FFT).


    k = torch.meshgrid(*k) #create meshgrid from the adjusted ranges
    k = torch.stack(k, dim=0) # stack in a new dimension
    k = k.norm(p=2, dim=0) # compute magnitude of the wavevector (Euclidean norm across dimension 0)


    
    # compute N, the number of modes in each bin. A mode is
    N = torch.full_like(P, 2.0, dtype=torch.float32) #create a tensor of the same shape as P filled with 2s (for the two halves of the spectrum)
    N[..., 0] = 1 # set the first element to 1 (DC component)
    if even:
        N[..., -1] = 1 # set the last element to 1 (Nyquist frequency for even-sized signals)

    k = k.flatten()
    P = P.flatten()
    N = N.flatten()



    kk = softbincount(k, weights=k*N, max_bin=kmax)  # soft binning of k
    P = softbincount(k, weights=P*N, max_bin=kmax)  # soft binning of P
    N = softbincount(k, weights=N, max_bin=kmax)  # soft binning of N


    kk /= N 
    P /= N

    return kk, P, N



def power_loss(x, y):
    """Differentiable power spectrum loss between x and y.

    Args:
        x: Tensor of shape (B, C, H, W) or (B, C, D, H, W)
        y: Tensor of shape (B, C, H, W) or (B, C, D, H, W)
    """
    _, P_x, _ = power_differentiable(x)
    _, P_y, _ = power_differentiable(y)

    #normalize power spectra
    P_x = P_x / (P_x.sum() + 1e-8)
    P_y = P_y / (P_y.sum() + 1e-8)

    # Compute the loss as the L2 distance between the power spectra
    loss = F.mse_loss(P_x, P_y)

    return loss

def power_loss_non_diff(x, y):
    """NON-differentiable power spectrum loss between x and y.

    Args:
        x: Tensor of shape (B, C, H, W) or (B, C, D, H, W)
        y: Tensor of shape (B, C, H, W) or (B, C, D, H, W)
    """
    _, P_x, _ = power(x)
    _, P_y, _ = power(y)

    #normalize power spectra
    P_x = P_x / (P_x.sum() + 1e-8)
    P_y = P_y / (P_y.sum() + 1e-8)

    # Compute the loss as the L2 distance between the power spectra
    loss = F.mse_loss(P_x, P_y)

    return loss

class PowerLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return power_loss(x, y)

class PowerLossL2E(nn.Module):
    def __init__(self, boxsize=1000.0, meshsize = 1024, mesh_up_fac=2):
        super().__init__()
        self.mesh_up_fac = mesh_up_fac
        self.boxsize = boxsize
        self.meshsize = meshsize

    def forward(self, x, y):
        # Convert from lagrangian to eulerian space
        x = lag2eul(x,  boxsize=self.boxsize, eul_scale_factor=self.mesh_up_fac, meshsize=self.meshsize)[0]
        y = lag2eul(y,  boxsize=self.boxsize, eul_scale_factor=self.mesh_up_fac, meshsize=self.meshsize)[0]
        return power_loss(x, y)



class LogSpectralDistance(nn.Module):  # NOTE: not differentiable
    def __init__(self, eps=1e-8, boxsize=1000.0, meshsize = 1024):
        super().__init__()
        self.eps = eps  # Avoid log(0)
        self.boxsize = boxsize
        self.meshsize = meshsize

    def forward(self, x, y):
        x = lag2eul(x,  boxsize=self.boxsize, eul_scale_factor=1, meshsize=self.meshsize)[0]
        y = lag2eul(y,  boxsize=self.boxsize, eul_scale_factor=1, meshsize=self.meshsize)[0]

        x_mean = x.mean()
        y_mean = y.mean()

        x = (x-x_mean)/x_mean
        y = (y-y_mean)/y_mean

        _, P_x, _ = power(x)
        _, P_y, _ = power(y)

        log_diff = torch.log10(P_x + self.eps) - torch.log10(P_y + self.eps)

        return torch.sqrt(torch.mean(log_diff ** 2))


class PowerL2E_non_diff(nn.Module):
    def __init__(self, boxsize=1000.0, meshsize = 1024, mesh_up_fac=2):
        super().__init__()
        self.mesh_up_fac = mesh_up_fac
        self.boxsize = boxsize
        self.meshsize = meshsize

    def forward(self, x, y):
        # Convert from lagrangian to eulerian space
        x = lag2eul(x,  boxsize=self.boxsize, eul_scale_factor=self.mesh_up_fac, meshsize=self.meshsize)[0]
        y = lag2eul(y,  boxsize=self.boxsize, eul_scale_factor=self.mesh_up_fac, meshsize=self.meshsize)[0]
        return power_loss_non_diff(x, y)









### Some tests for debug
def gradient_test():
    import torch

    # Dummy input
    x = torch.randn(2, 3, 64, 64, 64, requires_grad=True)
    y = torch.randn(2, 3, 64, 64, 64)

    loss_fn = PowerLossL2E()
    loss = loss_fn(x, y)
    loss.backward()

    # Check if the gradient was computed successfully
    print("shape:", x.grad.shape,"mean:", x.grad.mean().item(), "std:", x.grad.std().item())

    
def kk_test():
    x = torch.randn(2, 3, 64, 64, 64)
    y = torch.randn(2, 3, 64, 64, 64)

    kk1,P1,_ = power(x)
    kk2,P2,_ = power(y)

    assert len(kk1) == len(kk2)
    for i in range(len(kk1)):
        assert kk1[i] == kk2[i]

    assert len(P1) == len(P2)
    print("success")


if __name__ == "__main__":
    #kk_test()
    #quit()
    #gradient_test()
    #quit()
    import numpy as np
    from .power import power
    from ..utils.figures import dis2pos, plt_slices
    from ..data.norms import cosmology

    #load ../../DEMNUni_512_to_1024_crop/train/LR/seed_123465_dis_crop_0000_0000_0000.npy
    snap_dis = np.load("DEMNUni_512_to_1024_averageDS_crop/train/HR/seed_123465_dis_crop_0264_0000_0000.npy")

    #snapdis is shape (3, 264, 264, 264). Crop a subcube of shape (3, 64, 64, 64) randomly
    start_x, start_y, start_z = 0,0,0#np.random.randint(0, 200, size=3)
    snap_dis = snap_dis[:, start_x:start_x+64, start_y:start_y+64, start_z:start_z+64]


    #to torch tensor
    snap_dis = torch.from_numpy(snap_dis).unsqueeze(0).float()
    #print type and shape
    print(type(snap_dis), snap_dis.shape)

    #calculate nyquist frequency based on snap_dis.shape[-1]
    nyquist = snap_dis.shape[-1] // 2
    print("nyquist frequency:", nyquist)

    cosmology.dis(snap_dis)

    #compute power spectrum
    l2e = lag2eul(snap_dis, boxsize=1000.0, meshsize = 1024, eul_scale_factor=2)[0]
    cosmology.dis(snap_dis, undo=True)
    #print shape of l2e
    print(l2e.shape, "range:", l2e.min().item(), l2e.max().item(), l2e.mean().item())
    
    
    #scatter plot of XY projection of snap_dis, that has shape (1, 3, 64, 64, 64), where 3 is the 3 displacement components
    #and plot of XY projection of l2e, that has shape (1, 1, 64, 64, 64), where 1 is the density value, so plot as a heatmap
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("field XY projection")
    
    #scatterplot
    snap_dis_np = snap_dis.squeeze(0).cpu().numpy()
    
    snap_pos = dis2pos(snap_dis_np, Ng=64, boxsize=64/1024 * 1000.0)
    plt.scatter(snap_pos[0].flatten(), snap_pos[1].flatten(), s=1, alpha=0.1)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')  
    plt.grid(True)
    plt.title("Positions XY projection")


    plt.subplot(1, 2, 2)
    plt.title("density XY projection")
    l2e_np = l2e.squeeze(0).squeeze(0).cpu().numpy()
    #normalize to 0-1
    l2e_np = (l2e_np - l2e_np.min()) / (l2e_np.max() - l2e_np.min())
    plt.imshow(l2e_np.sum(axis=2), cmap='viridis')
    plt.colorbar(label='Density (summed over Z)')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')

    plt.savefig("plots_temp/field_and_density_xy_projection.png")

    plt.clf()
    quit()

    kk, P, N = power(l2e)

    #print ranges
    print("power ranges")
    print(kk.min(), kk.max())
    print(P.min(), P.max())

  
    import matplotlib.pyplot as plt
 
    #compute differentiable power spectrum
    kk_, P_diff, N_ = power_differentiable(l2e)

    #assert that kk and kk_ contain the same values
    #assert torch.allclose(kk, kk_, atol=1e-5), f"kk and kk_ are not the same: {kk} vs {kk_}"

    #print ranges
    print("differentiable power ranges")
    print(kk_.min(), kk_.max())
    print(P_diff.min(), P_diff.max())
    print(N_.min(), N_.max())
    
    #plot kk vs P and vs P_diff
    plt.loglog(kk.cpu().numpy(), P.cpu().numpy(), label="power", marker='o')
    plt.loglog(kk_.cpu().numpy(), P_diff.cpu().numpy(), label="diff power", marker='o')
    #vertical line at nyquist frequency
    plt.axvline(nyquist, color='r', linestyle='--', label="nyquist")
    plt.xlabel("k [h/Mpc]")
    plt.ylabel("P(k) [(Mpc/h)^3]")
    plt.legend()
    plt.savefig("plots_temp/power_vs_diffpower.png")
    plt.clf()

    #now create a plot of the ratio between power_diff and power
    plt.semilogx(kk.cpu().numpy(), (P_diff/P).cpu().numpy(), label="diff power / power", marker='o')
    #add line at 1 for power ratio with itself
    plt.axhline(1, color='r', linestyle='--', label="power")
    plt.xlabel("k [h/Mpc]")
    plt.ylabel("P_diff(k) / P(k)")
    plt.legend()
    plt.savefig("plots_temp/power_ratio.png")
    plt.clf()

    print("Power ratio:", (P_diff/P).cpu().numpy())