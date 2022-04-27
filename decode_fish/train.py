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
from decode_fish.engine.point_process import PointProcessUniform
from decode_fish.engine.gmm_loss import PointProcessGaussian
import torch_optimizer
from decode_fish.funcs.train_funcs import *
import wandb

@hydra.main(config_path='../config', config_name='train')
def my_app(cfg):

    """ General setup """
    print(cfg.data_path)
    seed_everything(cfg.seed)
    
    img_3d, decode_dl = get_dataloader(cfg)
    psf, noise, micro = load_psf_noise_micro(cfg)
    
    inp_offset, inp_scale = get_forward_scaling(img_3d[0])
    if cfg.network.inp_scale is not None:
        inp_scale = cfg.network.inp_scale
    if cfg.network.inp_offset is not None:
        inp_offset = cfg.network.inp_offset
        
    model = hydra.utils.instantiate(cfg.network, inp_scale=inp_scale, inp_offset=inp_offset)
    post_proc = hydra.utils.instantiate(cfg.post_proc_isi)
    
    bench_df, code_ref, targets = hydra.utils.instantiate(cfg.codebook)
    post_proc.codebook = torch.tensor(code_ref)
    
    psf  .to(cfg.device.gpu_device)
    model.to(cfg.device.gpu_device)
    micro.to(cfg.device.gpu_device)
    
    if cfg.evaluation.reconstruction.enabled:
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
                   mode=cfg.output.wandb_mode)

    optim_dict = {}
    optim_dict['optim_net'] = hydra.utils.instantiate(cfg.training.net.opt, params=model.network.parameters())
    optim_dict['optim_mic'] = hydra.utils.instantiate(cfg.training.mic.opt, params=micro.parameters())
    optim_dict['optim_int'] = hydra.utils.instantiate(cfg.training.int.opt, params=model.int_dist.parameters())

    optim_dict['sched_net'] = hydra.utils.instantiate(cfg.training.net.sched, optimizer=optim_dict['optim_net'])
    optim_dict['sched_mic'] = hydra.utils.instantiate(cfg.training.mic.sched, optimizer=optim_dict['optim_mic'])
    optim_dict['sched_int'] = hydra.utils.instantiate(cfg.training.int.sched, optimizer=optim_dict['optim_int'])

    
    if cfg.training.resume:
        cfg.data_path.model_init = cfg.output.save_dir
    
    if cfg.data_path.model_init is not None:
        print('loading')
        model = load_model_state(model, Path(cfg.data_path.model_init)/'model.pkl').cuda()
        '''temp disable'''
        # micro.load_state_dict(torch.load(Path(cfg.data_path.model_init)/'microscope.pkl'), strict=False)

        if cfg.training.net.enabled:
            train_state_dict = torch.load(Path(cfg.data_path.model_init)/'training_state.pkl')
            for k in optim_dict:
                optim_dict[k].load_state_dict(train_state_dict[k])    
            
            cfg.training.start_iter = train_state_dict['train_iter']
            
    if cfg.data_path.micro_init is not None:
        micro.load_state_dict(torch.load(cfg.data_path.micro_init), strict=False)
        
    train(cfg=cfg,
         model=model, 
         microscope=micro, 
         post_proc=post_proc,
         dl=decode_dl, 
         optim_dict=optim_dict, 
         eval_dict=eval_dict)

if __name__ == "__main__":
    my_app()
    