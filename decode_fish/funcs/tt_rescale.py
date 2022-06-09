# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/27_testtime_rescale.ipynb (unless otherwise specified).

__all__ = ['rescale_train']

# Cell
from ..imports import *
from .evaluation import *
from .file_io import *
from .emitter_io import *
from .utils import *
from .dataset import *
from .output_trafo import *
from .plotting import *
from .predict import *
import torch.nn.functional as F
from torch import distributions as D
from torch.utils.data import DataLoader
import torch_optimizer
from ..engine.microscope import Microscope, get_roi_filt_inds, extract_psf_roi, mic_inp_apply_inds, add_pos_noise, concat_micro_inp
from ..engine.model import UnetDecodeNoBn
from ..engine.point_process import PointProcessUniform, get_phased_ints
from ..engine.gmm_loss import PointProcessGaussian
import shutil
import wandb
import kornia

from hydra import compose, initialize
from .merfish_eval import *
from .exp_specific import *
# from decode_fish.funcs.visualization vimport get_simulation_statistics

# Cell
def rescale_train(cfg,
          model,
          microscope,
          post_proc,
          dl,
          optim_dict):

    save_dir = Path(cfg.output.save_dir)
    model.cuda()

    # Controls which genmodel parameters are optimized
    for name, p in microscope.named_parameters():
        if name == 'channel_facs':
            p.requires_grad = True
        else:
            False

    for batch_idx in range(0, cfg.training.num_iters+1):

        optim_dict['optim_mic'].zero_grad()

        with torch.no_grad():

            ret_dict = next(iter(dl))
            x, local_rate, background = ret_dict['x'], ret_dict['local_rate'], ret_dict['background']
            if cfg.genm.microscope.col_shifts_enabled:
                zcrop, ycrop, xcrop = ret_dict['crop_z'], ret_dict['crop_y'], ret_dict['crop_x']
                zcrop, ycrop, xcrop = zcrop.flatten(), ycrop.flatten(), xcrop.flatten()
                colshift_crop = get_color_shift_inp(microscope.color_shifts, microscope.col_shifts_yx, ycrop, xcrop, cfg.sim.random_crop.crop_sz)
            else:
                zcrop, ycrop, xcrop, colshift_crop = None, None, None, None

            x = x * microscope.get_ch_mult().detach()

            out_inp = torch.concat([x,colshift_crop], 1) if colshift_crop is not None else x
            out_inp = model.tensor_to_dict(model(out_inp))
            proc_out_inp = post_proc.get_micro_inp(out_inp)

        if len(proc_out_inp[1]) > 0:

            ch_out_inp = microscope.get_single_ch_inputs(*proc_out_inp, ycrop=ycrop, xcrop=xcrop)

            # Get ch_fac loss
            ch_inds = ch_out_inp[0][1]
            int_vals = ch_out_inp[-2]

            int_means = torch.ones(cfg.genm.exp_type.n_channels).cuda()
            for i in range(cfg.genm.exp_type.n_channels):
                if i in ch_inds:
                    int_means[i] = int_vals[ch_inds == i].mean() / int_vals.mean()
#             int_means_col.append(int_means.detach())
#             print(int_means)
            ch_fac_loss = torch.sqrt(torch.mean((microscope.channel_facs - microscope.channel_facs.detach() / int_means)**2))

            ch_fac_loss.backward()
            torch.nn.utils.clip_grad_norm_(microscope.parameters(), max_norm=cfg.training.mic.grad_clip, norm_type=2)

            optim_dict['optim_mic'].step()
            optim_dict['sched_mic'].step()

#             losses.append(ch_fac_loss.item())

            # Logging
            if batch_idx % cfg.output.log_interval == 0:

                print(ch_fac_loss)
#                 print(micro.channel_facs)