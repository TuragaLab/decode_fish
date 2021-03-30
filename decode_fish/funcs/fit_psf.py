# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/15_fit_psf.ipynb (unless otherwise specified).

__all__ = ['get_peaks_3d', 'plot_detection', 'extract_roi', 'fit_psf']

# Cell
from ..imports import *
from .file_io import *
from .utils import *
from .plotting import *
from .dataset import EstimateBackground
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from ..engine.microscope import Microscope
from torch.jit import script
from ..engine.psf import LinearInterpolatedPSF
from torch import nn
from torch import distributions as D

# Cell
def get_peaks_3d(volume, threshold=500, min_distance=20):
    """Peak finding functions. Provides position estimate for bead stacks that are used as initialization for PSF fitting.

    Parameters
    ----------
    volume: 3D array
        Single bead recording
    threshold: float
        Initial threshold to identify pixel that are considered as possible peaks.
    min_distance: float
        Minimal distance between two peaks in pixels

    Returns
    -------
    peaks: array
        Array of z,y,x peak positions
    """
    assert threshold < volume.max() , "Threshold higher then max volume intensity, chose a lower threshold"

    peaks = []
    t_img = np.where(volume>threshold,volume,0)
    inds = t_img.nonzero()
    vals = t_img[inds]
    inds_zyx = [[z,y,x] for _,z,y,x in sorted(zip(vals,inds[0],inds[1],inds[2]))][::-1]

    while len(inds_zyx) > 0:

        valid = True
        zyx = inds_zyx[0]
        z,y,x = zyx
        inds_zyx.remove(zyx)

        for pyx in peaks:
            if np.sqrt((z-pyx[0])**2 + (y-pyx[1])**2 + + (x-pyx[2])**2) < min_distance:
                valid = False
                break
        if valid:
            peaks.append(zyx)

    return np.array(peaks)[:,::-1]

# Cell
def plot_detection(volume, coordinates):
    plt.figure(figsize=(10,5))
    plt.subplot(121)
    plt.imshow(volume.max(0).values)
    plt.plot(coordinates[:, 0], coordinates[:, 1], 'r.')
    plt.title('Z crossection')
    plt.subplot(122)
    plt.imshow(volume.max(2).values)
    plt.plot(coordinates[:, 1], coordinates[:, 2], 'r.')
    plt.title('X crossection')
    plt.show()

# Cell
def extract_roi(beads: torch.Tensor, coordinates: torch.Tensor, size_xy: int=10, size_z: int=10):
    res = []
    tot_size = torch.Size([1, size_z*2+1, size_xy*2+1, size_xy*2+1])
    for i in coordinates:
        x, y, z = i
        single_bead = beads[:, z-size_z: z+size_z+1,
                               y-size_xy: y+size_xy+1,
                               x-size_xy: x+size_xy+1]

        if single_bead.size() == tot_size:
            res.append(single_bead)

    return torch.cat(res, 0)

# Cell
def fit_psf(PSF, roi_list, fit_lr=1e-3, fit_iteration=10000, device='cuda'):

    ground_truth = roi_list + 0
    param_list = list(PSF.parameters())
    ground_truth= ground_truth.unsqueeze(1).to(device)

    x_os_val = nn.Parameter(D.Uniform(low=0-0.5, high=0+0.5).sample((ground_truth.shape[0], 1, 1, 1)).to(device))
    y_os_val = nn.Parameter(D.Uniform(low=0-0.5, high=0+0.5).sample((ground_truth.shape[0], 1, 1, 1)).to(device))
    z_os_val = nn.Parameter(D.Uniform(low=0-0.5, high=0+0.5).sample((ground_truth.shape[0], 1, 1, 1)).to(device))
    param_list = param_list + [x_os_val, y_os_val, z_os_val]
    loss_res = []

    optim = torch.optim.AdamW(param_list, fit_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=2000, gamma=0.5)

    for i in range(fit_iteration):
        psf_volume = PSF(x_os_val, y_os_val, z_os_val)
        loss = F.mse_loss(psf_volume, ground_truth) + 1e-5*torch.norm(psf_volume.sum(), 1)
        loss_res.append(loss.detach().cpu().item())
        loss.backward()
        optim.step()
        optim.zero_grad()

        if i%1000 == 0:
            plot_3d_projections(PSF.psf_volume[0], projection='mean', size=3);
            plt.show()
        scheduler.step()

    return loss_res