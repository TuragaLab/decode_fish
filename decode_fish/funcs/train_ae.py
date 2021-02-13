# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/14_train_ae.ipynb (unless otherwise specified).

__all__ = ['train_ae']

# Cell
from ..imports import *
from .file_io import *
from .emitter_io import *
from .utils import *
from .dataset import *
from .output_trafo import *
from .evaluation import *
from .plotting import *
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from ..engine.microscope import Microscope
from ..engine.model import UnetDecodeNoBn
import shutil
from torch.utils.tensorboard import SummaryWriter
from ..engine.point_process import PointProcessUniform
from ..engine.gmm_loss import PointProcessGaussian
from .train_sl import eval_logger, load_from_eval_dict

# Cell
def train_ae(model,
             dl,
             num_iter,
             optim_net,
             optim_psf,
             sched_net,
             sched_psf,
             min_int,
             microscope,
             log_interval,
             save_dir,
             log_dir,
             psf=None,
             bl_loss_scale = 0.01,
             p_quantile=0,
             grad_clip=0.01,
             eval_dict=None):

    save_dir = Path(save_dir)
    writer = SummaryWriter(log_dir)

    if eval_dict is not None:
        eval_img, eval_df = load_from_eval_dict(eval_dict)

    model.cuda().train()

    for batch_idx in range(num_iter):
        x, local_rate, background = next(iter(dl))

        """GET SUPVERVISED LOSS"""
        locs_sl, x_os_sl, y_os_sl, z_os_sl, ints_sl, output_shape = PointProcessUniform(local_rate, min_int=min_int).sample()
        xsim = microscope(locs_sl, x_os_sl, y_os_sl, z_os_sl, ints_sl, output_shape)
        xsim_noise = microscope.noise(xsim, background).sample()

        res_sim = model(xsim_noise)
        gmm_loss = -PointProcessGaussian(logits = res_sim['logits'],xyzi_mu=res_sim['xyzi_mu'],xyzi_sigma = res_sim['xyzi_sigma']).log_prob(locs_sl,x_os_sl, y_os_sl, z_os_sl, ints_sl, p_quantile).mean()

        background_loss = F.mse_loss(res_sim['background'], background)

        loss = gmm_loss  + bl_loss_scale * background_loss
        """"""
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip, norm_type=2)

        optim_net.step()
        optim_net.zero_grad()


        res_img = model(x)

        ae_img = microscope(*model_output_to_micro_input(res_img, threshold=0.1))

        log_p_x_given_z = - microscope.noise(ae_img,res_img['background']).log_prob(x).mean()

#         print(len(locs_ae[0]))
        log_p_x_given_z.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(optim_psf.param_groups[0]['params'], max_norm=grad_clip, norm_type=2)

        optim_psf.step()
        optim_psf.zero_grad()

        if sched_net:
            sched_net.step()

        if sched_psf:
            sched_psf.step()


        writer.add_scalar('SL Losses/gmm_loss', gmm_loss.detach().cpu(), batch_idx)
        writer.add_scalar('SL Losses/bg_loss', background_loss.detach().cpu(), batch_idx)

        writer.add_scalar('AE Losses/p_x_given_z', log_p_x_given_z.detach().cpu(), batch_idx)

        if batch_idx % log_interval == 0 and writer is not None:

            with torch.no_grad():
                pred_df = model_output_to_df(res_sim, 0.1)
                target_df = sample_to_df(locs_sl, x_os_sl, y_os_sl, z_os_sl, ints_sl)

                if eval_dict is not None:
                    res_eval = model(eval_img[None].cuda())
                    ae_img = microscope(*model_output_to_micro_input(res_eval, threshold=0.1))
                    pred_eval_df = model_output_to_df(res_eval, 0.1, px_size=eval_dict['px_size'])
                    free_mem()

            if writer is not None:
                eval_logger(writer, pred_df, target_df, batch_idx, data_str='Sim. ')
                if eval_df is not None:
                    eval_logger(writer, pred_eval_df, eval_df, batch_idx, data_str='Inp. ')

            sl_fig = sl_plot(x, xsim_noise, background, res_sim)
            plt.show()

            if eval_dict is not None:
                eval_fig = gt_plot(eval_img, pred_eval_df, eval_df, eval_dict['px_size'],ae_img[0]+res_eval['background'][0])
                plt.show()

            if writer is not None:
                writer.add_figure('SL summary', sl_fig, batch_idx)
                if eval_dict is not None:
                    writer.add_figure('GT', eval_fig, batch_idx)

            #storingv
            torch.save({'state_dict':model.state_dict(), 'scaling':[model.unet.inp_scale, model.unet.inp_offset]}, str(save_dir) +'/model_ae.pkl')
            torch.save(microscope.state_dict(), str(save_dir) + '/microscope_ae.pkl')
            torch.save(psf.state_dict(), str(save_dir) + '/psf_ae.pkl' )
            torch.save(optim_psf.state_dict(), str(save_dir) + '/opt_ae.pkl' )