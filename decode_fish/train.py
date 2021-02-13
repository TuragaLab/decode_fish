from decode_fish.imports import *
from decode_fish.funcs.file_io import *
from decode_fish.funcs.emitter_io import *
from decode_fish.funcs.utils import *
from decode_fish.funcs.dataset import *
from decode_fish.funcs.output_trafo import *
from decode_fish.funcs.evaluation import *
from decode_fish.funcs.plotting import *
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from decode_fish.engine.microscope import Microscope
from decode_fish.engine.model import UnetDecodeNoBn
import shutil
from torch.utils.tensorboard import SummaryWriter
from decode_fish.engine.point_process import PointProcessUniform
from decode_fish.engine.gmm_loss import PointProcessGaussian

from decode_fish.funcs.train_sl import *
from decode_fish.funcs.train_ae import *

@hydra.main(config_path='/groups/turaga/home/speisera/Dropbox (mackelab)/Artur/WorkDB/deepstorm/decode_fish/config', config_name='train')
def my_app(cfg):

    """ General setup """
    
    img_3d, decode_dl = get_dataloader(cfg)
    
    psf, noise, micro = load_psf_micro_psf_noise(cfg)
    
    if cfg.model.inp_scale is None or cfg.model.inp_offset is None:
        inp_offset, inp_scale = get_forward_scaling(img_3d)
    else:
        inp_offset, inp_scale = cfg.model.inp_scale, cfg.model.inp_offset
    model = hydra.utils.instantiate(cfg.model, inp_scale=float(inp_scale), inp_offset=float(inp_offset))
    
    psf  .to(cfg.device.gpu_device)
    model.to(cfg.device.gpu_device)
    micro.to(cfg.device.gpu_device)
    
    if cfg.evaluation is not None:
        eval_dict = dict(cfg.evaluation)
        eval_dict['crop_sl'] = eval(eval_dict['crop_sl'],{'__builtins__': None},{'s_': np.s_})
        eval_dict['px_size'] = list(eval_dict['px_size'])
    else:
        eval_dict = None
        
    save_dir = Path(cfg.output.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    OmegaConf.save(cfg, cfg.output.save_dir + '/train.yaml')
    
    optim_net = AdamW(model.parameters(), lr=cfg.supervised.lr)
    sched_net = torch.optim.lr_scheduler.StepLR(optim_net, step_size=cfg.supervised.step_size, gamma=cfg.supervised.gamma)
    
    """ Simulator learning """
        
    train_sl(model=model, 
             dl=decode_dl, 
             num_iter=cfg.supervised.num_iter,
             optim_net=optim_net, 
             sched_net=sched_net, 
             psf=psf,
             min_int=cfg.pointprocess.min_int, 
             microscope=micro, 
             log_interval=cfg.supervised.log_interval, 
             save_dir=cfg.output.save_dir,
             log_dir=cfg.output.log_dir,
             bl_loss_scale=cfg.supervised.bl_loss_scale,
             p_quantile=cfg.supervised.p_quantile,
             grad_clip=cfg.supervised.grad_clip,
             eval_dict=eval_dict)
    
    """ Autoencoder learning """

    ae_param = list(micro.parameters())  + list(psf.parameters())
    optim_psf  = AdamW(ae_param, lr=cfg.autoencoder.lr)
    sched_psf = torch.optim.lr_scheduler.StepLR(optim_psf, step_size=cfg.autoencoder.step_size, gamma=cfg.autoencoder.gamma)
    
    if not cfg.supervised.num_iter:
    
        model_sl = load_model_state(cfg, 'model_sl.pkl').cuda()
        micro.load_state_dict(torch.load(Path(cfg.output.save_dir)/'microscope_sl.pkl'))
        optim_net.load_state_dict(torch.load(Path(cfg.output.save_dir)/'opt_sl.pkl'))
        sched_net = torch.optim.lr_scheduler.StepLR(optim_net, step_size=cfg.supervised.step_size, gamma=cfg.supervised.gamma)

    train_ae(model=model, 
             dl=decode_dl, 
             num_iter=cfg.autoencoder.num_iter,
             optim_net=optim_net, 
             optim_psf=optim_psf, 
             sched_net=sched_net, 
             sched_psf=sched_psf, 
             min_int=cfg.pointprocess.min_int, 
             psf=psf,
             microscope=micro, 
             log_interval=cfg.supervised.log_interval,  
             save_dir=cfg.output.save_dir,
             log_dir=cfg.output.log_dir,
             bl_loss_scale=cfg.supervised.bl_loss_scale,
             p_quantile=cfg.supervised.p_quantile,
             grad_clip=cfg.supervised.grad_clip,
             eval_dict=eval_dict)
        
if __name__ == "__main__":
    my_app()