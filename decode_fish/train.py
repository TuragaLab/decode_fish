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
from torch.utils.data import DataLoader
from decode_fish.engine.microscope import Microscope
from decode_fish.engine.model import UnetDecodeNoBn
import shutil
from torch.utils.tensorboard import SummaryWriter
from decode_fish.engine.point_process import PointProcessUniform
from decode_fish.engine.gmm_loss import PointProcessGaussian

from decode_fish.funcs.train_funcs import *
import wandb

@hydra.main(config_path='../config', config_name='train')
def my_app(cfg):

    """ General setup """
    
    img_3d, decode_dl = get_dataloader(cfg)
    psf, noise, micro = load_psf_noise_micro(cfg)
    
    if cfg.model.inp_scale is None or cfg.model.inp_offset is None:
        inp_offset, inp_scale = get_forward_scaling(img_3d[0])
    else:
        inp_offset, inp_scale = cfg.model.inp_scale, cfg.model.inp_offset
        
    model = hydra.utils.instantiate(cfg.model, inp_scale=float(inp_scale), inp_offset=float(inp_offset))
    post_proc = load_post_proc(cfg)
    
    psf  .to(cfg.device.gpu_device)
    model.to(cfg.device.gpu_device)
    micro.to(cfg.device.gpu_device)
    
    if cfg.evaluation is not None:
        eval_dict = dict(cfg.evaluation)
        eval_dict['crop_sl'] = eval(eval_dict['crop_sl'],{'__builtins__': None},{'s_': np.s_})
        eval_dict['px_size_zyx'] = list(eval_dict['px_size_zyx'])
    else:
        eval_dict = None
        
    save_dir = Path(cfg.output.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    OmegaConf.save(cfg, cfg.output.save_dir + '/train.yaml')
    
    _ = wandb.init(project=cfg.output.project, 
                   config=OmegaConf.to_container(cfg, resolve=True),
                   dir=cfg.output.log_dir,
                   group=cfg.output.group,
                   name=cfg.run_name,
                   mode=cfg.other.wandb_mode
              )

    optim_net = hydra.utils.instantiate(cfg.training.net.opt, params=model.unet.parameters())
    optim_mic = hydra.utils.instantiate(cfg.training.mic.opt, params=micro.parameters())
    optim_int = hydra.utils.instantiate(cfg.training.int.opt, params=model.int_dist.parameters())

    sched_net = hydra.utils.instantiate(cfg.training.net.sched, optimizer=optim_net)
    sched_mic = hydra.utils.instantiate(cfg.training.mic.sched, optimizer=optim_mic)
    sched_int = hydra.utils.instantiate(cfg.training.int.sched, optimizer=optim_int)
    
    if cfg.data_path.model_init is not None:
        print('loading')
        model = load_model_state(model, cfg.data_path.model_init).cuda()
        micro.load_state_dict(torch.load(Path(cfg.data_path.model_init)/'microscope.pkl'))

        train_state_dict = torch.load(Path(cfg.data_path.model_init)/'training_state.pkl')
        optim_net.load_state_dict(train_state_dict['optim_net'])
        optim_mic.load_state_dict(train_state_dict['optim_mic'])
        optim_int.load_state_dict(train_state_dict['optim_int'])
        sched_net.load_state_dict(train_state_dict['sched_net'])
        sched_mic.load_state_dict(train_state_dict['sched_mic'])
        sched_int.load_state_dict(train_state_dict['sched_int'])                                 
        
    train(cfg=cfg,
         model=model, 
         dl=decode_dl, 
         optim_net=optim_net, 
         optim_mic=optim_mic, 
         optim_int=optim_int, 
         sched_net=sched_net, 
         sched_mic=sched_mic, 
         sched_int=sched_int,
         psf=psf,
         post_proc=post_proc,
         microscope=micro, 
         eval_dict=eval_dict)

if __name__ == "__main__":
    my_app()