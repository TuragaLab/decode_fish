# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/02b_place_psfs.ipynb (unless otherwise specified).

__all__ = ['place_roi', 'CudaPlaceROI']

# Cell
# from decode_fish.funcs.utils import cpu
import torch
# import numpy as np
from torch.jit import script
import numba
from numba import cuda

# import warnings
# warnings.filterwarnings("ignore", category=numba.errors.NumbaPerformanceWarning)

# Cell
@script
def _place_psf(psf_vols, b, ch, z, y, x, output_shape):

    """jit function for placing PSFs
    1) This function will add padding to coordinates (z, y, x) (we need padding in order to place psf on the edges)
    afterwards we will just crop out to original shape
    2) Create empty tensor with paddings loc3d_like
    3) place each individual PSFs in to the corresponding cordinates in loc3d_like
    4) unpad to original output shape

    Args:
        psf_vols:   torch.Tensor
        b:        torch.Tensor
        c:        torch.Tensor
        h:        torch.Tensor
        w:        torch.Tensor
        d:        torch.Tensor
        szs:      torch.Tensor

    Shape:
        psf_vols: (Num_E, C, PSF_SZ_X, PSF_SZ_Y, PSF_SZ_Z)
        b:  (Num_E,)
        c:  (Num_E,)
        h:  (Num_E,)
        w:  (Num_E,)
        d:  (Num_E,)
        output_shape:  (BS, Frames, H, W, D)

    -Output: placed_psf: (BS, Frames, H, W, D)

    """

    psf_b, psf_c, psf_d, psf_h, psf_w = psf_vols.shape
    pad_zyx = [psf_d//2, psf_h//2, psf_w//2]
    #add padding to z, y, x

    z = z + pad_zyx[0]
    y = y + pad_zyx[1]
    x = x + pad_zyx[2]

    #create padded tensor (bs, frame, c, h, w) We will need pad_size * 2 since we are padding from both size
    loc3d_like = torch.zeros(output_shape[0],
                             output_shape[1],
                             output_shape[2] + 2*(pad_zyx[0]),
                             output_shape[3] + 2*(pad_zyx[1]),
                             output_shape[4] + 2*(pad_zyx[2])).to(x.device)

    if psf_c == 2:
        psf_ch_ind = torch.where(ch >= 8, 1, 0)
        psf_vols = psf_vols[torch.arange(len(psf_ch_ind)),psf_ch_ind]
    if output_shape[1] == 1:
        ch = torch.zeros_like(ch)

    psf_vols = psf_vols.reshape(-1, psf_d, psf_h, psf_w)

    # Take limit calculation out of the loop for 30% speed up
    z_l = z - pad_zyx[0]
    y_l = y - pad_zyx[1]
    x_l = x - pad_zyx[2]

    z_h = z + pad_zyx[0] + 1
    y_h = y + pad_zyx[1] + 1
    x_h = x + pad_zyx[2] + 1

    for idx in range(x.shape[0]):
        loc3d_like[b[idx], ch[idx],
        z_l[idx] : z_h[idx],
        y_l[idx] : y_h[idx],
        x_l[idx] : x_h[idx]] += psf_vols[idx]

    # unpad to original size
    b_sz, ch_sz, h_sz, w_sz, d_sz = loc3d_like.shape

    placed_psf = loc3d_like[:, :, pad_zyx[0]: h_sz - pad_zyx[0],
                                  pad_zyx[1]: w_sz - pad_zyx[1],
                                  pad_zyx[2]: d_sz - pad_zyx[2]]
    return placed_psf

# Cell
@cuda.jit
def place_roi(frames, roi_grads, frame_s_b, frame_s_c, frame_s_z, frame_s_y, frame_s_x, rois, roi_s_n, roi_s_z, roi_s_y, roi_s_x, b, c, z, y, x):

    kx = cuda.grid(1)
    # One thread for every pixel in the roi stack. Exit if outside
    if kx >= roi_s_n * roi_s_z * roi_s_y * roi_s_x:
        return

    # roi index
    xir = kx % roi_s_x; kx = kx // roi_s_x
    yir = kx % roi_s_y; kx = kx // roi_s_y
    zir = kx % roi_s_z; kx = kx // roi_s_z
    nir = kx % roi_s_n

    # frame index
    bif = b[nir]
    cif = c[nir]
    zif = z[nir] + zir
    yif = y[nir] + yir
    xif = x[nir] + xir

    if ((bif < 0) or (bif >= frame_s_b)): return
    if ((cif < 0) or (cif >= frame_s_c)): return
    if ((zif < 0) or (zif >= frame_s_z)): return
    if ((yif < 0) or (yif >= frame_s_y)): return
    if ((xif < 0) or (xif >= frame_s_x)): return

    cuda.atomic.add(frames, (bif, cif, zif, yif, xif), rois[nir, zir, yir, xir])
    # The gradients for the ROIs are just one if they are inside the frames and 0 otherwise. Easy to do here and then just ship to the backward function
    roi_grads[nir, zir, yir, xir] = 1
    # Alternative to atomic.add. No difference in speed
#     frames[bif, cif, zif, yif, xif] += rois[nir, zir, yir, xir]

# Cell
class CudaPlaceROI(torch.autograd.Function):

    @staticmethod
    def forward(ctx, rois, frame_s_b, frame_s_c, frame_s_z, frame_s_y, frame_s_x, roi_s_n, roi_s_z, roi_s_y, roi_s_x, b, c, z, y, x):

        frames = torch.zeros([frame_s_b, frame_s_c, frame_s_z, frame_s_y, frame_s_x]).to('cuda')
        rois_grads = torch.zeros([roi_s_n, roi_s_z, roi_s_y, roi_s_x]).to('cuda')

        threadsperblock = 256
        blocks = ((roi_s_n * roi_s_z * roi_s_y * roi_s_x) + (threadsperblock - 1)) // threadsperblock
        place_roi[blocks, threadsperblock](frames, rois_grads, frame_s_b, frame_s_c, frame_s_z, frame_s_y, frame_s_x, rois.detach(), roi_s_n, roi_s_z, roi_s_y, roi_s_x, b, c, z, y, x)

        ctx.save_for_backward(rois_grads)

        return frames

    @staticmethod
    def backward(ctx, grad_output):
        rois_grads, = ctx.saved_tensors
        return rois_grads, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None