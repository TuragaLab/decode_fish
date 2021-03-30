# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/02_microscope.ipynb (unless otherwise specified).

__all__ = ['Microscope', 'place_psf', 'extractvalues']

# Cell
from ..imports import *
import torch.nn as nn
from torch.jit import script
from typing import Union, List
import torch.nn.functional as F

# Cell
class Microscope(nn.Module):
    """
    Mircoscope module takes  4 volumes 'locations', 'x_os_3d', 'y_os_3d', 'z_os_3d',
    'ints_3d'  and applies point spread function:
    1) Extracts values of intensities and offsets of given emitters
    2) Applies parametric PSF if given
    3) Applies empirical  PSF if given
    4) Combine all PSF and multiply by intenseties
    5) Normalize PSFs
    6) Places point spread function on to sampled locations  to
    generate 'x_sim' simulated image
    7) Multiplies x_sim by multipl


    Args:
        parametric_psf (torch.nn.Module): List of Paramateric PSF
        empirical_psf (torch.nn.Module): List of Emperical PSF
        noise (torch.nn.Module): Camera noise model
        scale(float): Paramter for scaling point spread functions
        multipl(float): Multiplication value that will be applied to xsim
    Shape:
        -Input: locations: Tuple(BS, Frames, H, W, D)
                x_os_val: (N_emitters,)
                y_os_val: (N_emitters,)
                z_os_val: (N_emitters,)
                ints_val: (N_emitters,)
                output_shape: Shape Tuple(BS, Frames, H, W, D)

        -Output: xsim:    (BS, C, H, W, D)
    """


    def __init__(self,
                 parametric_psf: List[torch.nn.Module]=None,
                 empirical_psf : List[torch.nn.Module]=None ,
                 noise: Union[torch.nn.Module, None]=None,
                 scale: float = 10000., multipl=100,
                 psf_noise=None, clamp_mode = 'cp'
                 ):

        super().__init__()
        self.parametric_psf = parametric_psf if parametric_psf else None
        self.empirical_psf  = empirical_psf if empirical_psf else None
#         self.scale = torch.nn.Parameter(torch.tensor(scale))
        self.scale = scale
        self.noise = noise
        self.multipl = multipl
        self.psf_noise = psf_noise
        self.clamp_mode = clamp_mode

    def add_psf_noise(self, psf_stack):

        noise = torch.distributions.Normal(loc=0, scale=self.psf_noise).sample(psf_stack.shape).to(psf_stack.device)
        return psf_stack + noise

    def forward(self, locations, x_os_val, y_os_val, z_os_val, i_val,output_shape, bg=None, eval_=None, scale_x=None, scale_y=None, scale_z=None):

        if len(locations[0]):

            if scale_x: x_os_val = x_os_val*scale_x
            if scale_y: y_os_val = y_os_val*scale_y
            if scale_z: z_os_val = z_os_val*scale_z

            psf = 0
            if self.parametric_psf:
                for param_psf_ in self.parametric_psf:
                    psf += param_psf_(x_os_val, y_os_val, z_os_val)

            if self.empirical_psf:
                for emper_psf in self.empirical_psf:
                    psf += emper_psf(x_os_val, y_os_val, z_os_val)

            if self.clamp_mode == 'cp':
                torch.clamp_min_(psf,0)
            #normalizing psf
            psf = psf.div(psf.sum(dim=[2, 3, 4], keepdim=True))
            if self.psf_noise: psf = self.add_psf_noise(psf)
            #applying intenseties (N_Emitters, C, H, W, D)

            psf = psf * i_val[:,None,None,None,None]
            xsim = place_psf(locations, psf, output_shape)
            xsim = self.scale * xsim * self.multipl
            if self.clamp_mode == 'cx':
                torch.clamp_min_(xsim,0)
            if eval_:
                return xsim, psf
            return xsim

        else:

            return torch.zeros(output_shape).cuda()

# Cell
def place_psf(locations, psf_volume, output_shape):
    """
    Places point spread functions (psf_volume) in to corresponding locations.

    Args:
        locations: torch.Tensor
        psf_volume: torch.Tensor
        output_shape: torch.Tensor

    Shape:
        -Input: locations: Tuple(BS, Frames, H, W, D)
                psf: (Num_E, C, PSF_SZ_X, PSF_SZ_Y, PSF_SZ_Z) [
                Num_E-Number of Emitters, PSF_SZ_{X, Y, Z} - PSF filter size]
                output_shape: shape of the output volume (BS, Frames, H, W, D)
        -Output: placed_psf: (BS, Frames, H, W, D)

    Returns:
        placed_psf
    """
    filter_size = psf_volume.shape[2:]
    filter_sizes = torch.cat(
        [torch.tensor((sz // 2, sz // 2 + 1)) for sz in filter_size]).reshape(
        3, 2).cuda()
    padding_sz = torch.tensor(max(filter_size) // 2 + 2).cuda()
    batch, frame, h, w, d = locations
    placed_psf = _place_psf(psf_volume, padding_sz, filter_sizes, batch, frame, h, w, d, torch.tensor(output_shape))
    assert placed_psf.shape == output_shape
    return placed_psf

# Cell
@script
def _place_psf(volume, pad_size, fz, b, c, h, w, d, output_shape):
    '''jit function for placing point spread function
    1) This function will add padding to coordinates (h, w, d) (we need padding in order to place psf on the edges)
    afterwards we will just crop out to original shape
    2) Create empty tensor with paddings loc3d_like
    3) place each individual PSFs in to the coresponding cordinates in loc3d_like
    4) unpad to original output shape


    Args:
        volume:   torch.Tensor
        pad_size: torch.Tensor
        fs:       torch.Tensor
        b:        torch.Tensor
        c:        torch.Tensor
        h:        torch.Tensor
        w:        torch.Tensor
        d:        torch.Tensor
        szs:      torch.Tensor

    Shape:
        volume: (Num_E, C, PSF_SZ_X, PSF_SZ_Y, PSF_SZ_Z)
        pad_size: (1,)
        fs: (3, 2)
        b:  (Num_E,)
        c:  (Num_E,)
        h:  (Num_E,)
        w:  (Num_E,)
        d:  (Num_E,)
        output_shape:  (BS, Frames, H, W, D)

    -Output: placed_psf: (BS, Frames, H, W, D)

    '''
    #adding padding to h, w, d
    h = h + pad_size
    w = w + pad_size
    d = d + pad_size

    #create padded tensor (bs, frame, c, h, w) We will need pad_size * 2 since we are padding from both size
    loc3d_like = torch.zeros(output_shape[0],
                             output_shape[1],
                             output_shape[2] + pad_size *2,
                             output_shape[3] + pad_size *2,
                             output_shape[4] + pad_size *2)
    loc3d_like = loc3d_like.to(d.device)
    psf_b, psf_c, psf_h, psf_w, psf_d = volume.shape
    volume = volume.reshape(-1, psf_h, psf_w, psf_d)

    for idx in range(b.shape[0]):
        loc3d_like[b[idx], c[idx],
        h[idx] - fz[0][0]:h[idx] + fz[0][1],
        w[idx] - fz[1][0]:w[idx] + fz[1][1],
        d[idx] - fz[2][0]:d[idx] + fz[2][1]] += volume[idx]

    b_sz, ch_sz, h_sz, w_sz, d_sz = loc3d_like.shape

    # unpad to original size
    placed_psf = loc3d_like[:, :, pad_size: h_sz - pad_size,
                                  pad_size: w_sz - pad_size,
                                  pad_size: d_sz - pad_size]
    return placed_psf

# Cell
def extractvalues( locs: torch.tensor,
                    x_os: torch.tensor,
                    y_os: torch.tensor,
                    z_os: torch.tensor,
                    ints:torch.tensor, dim: int=3):
    """
    Extracts Values of intensities and offsets of given emitters

     This function will take `locs`, `x_os`, `y_os`, `z_os`, `ints` all of the shapes,
     and will extract `coord` coordinate of locations where our emittors  are present.
     This `coord` will be used to extract values of `x`, `y`, `z`,
     offsets and intensities - `i` where the emitter is present

    Args:
        locs: location
        x_os: X offset
        y_os: Y offset
        z_os: Z offset
        ints: Intenseties
        dim:  Dimension 2D or 3D

    Shape:
        -Input: locs_3d: (BS, C, H, W, D)
                x_os_3d: (BS, C, H, W, D)
                y_os_3d: (BS, C, H, W, D)
                z_os_3d: (BS, C, H, W, D)
                ints_3d: (BS, C, H, W, D)

        -Output: :
                x_os_val: (Num_E, 1, 1, 1, 1)
                y_os_val: (Num_E, 1, 1, 1, 1)
                z_os_val: (Num_E, 1, 1, 1, 1)
                ints_val: (Num_E, 1, 1, 1, 1)
    """

    dim = tuple([1 for i in range(dim)])
    coord = tuple(locs.nonzero().transpose(1,0))
    x_os_val = x_os[coord].reshape(-1, *dim)
    y_os_val = y_os[coord].reshape(-1, *dim)
    z_os_val = z_os[coord].reshape(-1, *dim)
    ints_val = ints[coord].reshape(-1, *dim)
    return  x_os_val, y_os_val, z_os_val, ints_val