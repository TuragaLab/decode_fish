from decode_fish.imports import *
from decode_fish.funcs.file_io import *
from decode_fish.funcs.emitter_io import *
from decode_fish.funcs.utils import *
from decode_fish.funcs.dataset import *
from decode_fish.funcs.output_trafo import *
from decode_fish.funcs.evaluation import *
from decode_fish.funcs.plotting import *
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from decode_fish.engine.microscope import Microscope
from decode_fish.engine.model import UnetDecodeNoBn
import shutil
from decode_fish.engine.point_process import PointProcessUniform
from decode_fish.engine.gmm_loss import PointProcessGaussian

from decode_fish.funcs.fit_psf import get_peaks_3d

from decode_fish.funcs.routines import *
from decode_fish.funcs.predict import window_predict
from decode_fish.funcs.exp_specific import *
from decode_fish.funcs.tt_rescale import rescale_train
import wandb
import kornia

import h5py

from decode_fish.funcs.merfish_eval import *
# from decode_fish.funcs.merfish_codenet import *

@hydra.main(config_path='../config', config_name='merfish_eval')
def my_app(cfg):

    model_cfg = OmegaConf.load(cfg.model_cfg)
    
    psf, noise, micro = load_psf_noise_micro(model_cfg)
    post_proc = hydra.utils.instantiate(model_cfg.post_proc_isi)
    model = hydra.utils.instantiate(model_cfg.network)
    
    path = Path(model_cfg.output.save_dir)
    load_model_state(model, path/f'model.pkl')
    micro.load_state_dict(torch.load(path/'microscope.pkl'), strict=False)
    post_proc.samp_threshold = 0.5
    
    model.eval().cuda()
    
    
    image_paths = sorted(glob.glob(cfg.image_path))
    
    if cfg.crop is not None:
        crop = eval(cfg.crop,{'__builtins__': None},{'s_': np.s_})
    else:
        crop = np.s_[:,:,:,:]
        
    if cfg.scale_train:
        
        model_cfg.training.bs = 50
        model_cfg.training.num_iters = 200
        model_cfg.training.mic.opt.lr = 0.01
        model_cfg.training.mic.sched.step_size = 100
        model_cfg.data_path.image_path = cfg.image_path
        
        codebook, targets = hydra.utils.instantiate(model_cfg.codebook)
        post_proc.codebook = torch.tensor(expand_codebook(codebook)) if model_cfg.genm.exp_type.em_noise_inf else torch.tensor(codebook)
        image_vol, decode_dl = get_dataloader(model_cfg)
        optim_dict = {}
        optim_dict['optim_mic'] = hydra.utils.instantiate(model_cfg.training.mic.opt, params=micro.parameters())
        optim_dict['sched_mic'] = hydra.utils.instantiate(model_cfg.training.mic.sched, optimizer=optim_dict['optim_mic'])
        
        rescale_train(cfg=model_cfg,
             model=model, 
             microscope=micro, 
             post_proc=post_proc,
             dl=decode_dl, 
             optim_dict=optim_dict)

    else:
        image_vol = read_MOp_tiff(image_paths[0], z_to_batch=True)
        
    res_df = window_predict(model, post_proc, image_vol, window_size=[None, 128, 128], crop=crop, device='cuda', 
                             chrom_map=get_color_shift_inp(micro.color_shifts, micro.col_shifts_yx)[:,:,None], scale=micro.get_ch_mult())      
    
    #res_df = exclude_borders(res_df, border_size_zyx=[0,4000,4000], img_size=[2048*100,2048*100,2048*100])
    # res_df['gene'] = targets[res_df['code_inds']]
    
    ###
    max_vol = image_vol * micro.get_ch_mult().to(image_vol.device)
    max_p = cpu(max_vol).max(0).max(0)[0]
    fids = get_peaks(max_p, 18000, 20)
    
    res_df['zm'] = res_df['z']%100
    res_df = exclude_borders(res_df, border_size_zyx=[0,15000,15000], img_size=[2048*100,2048*100,2048*100])
    res_df = remove_fids(res_df, px_to_nm(fids), tolerance=1000)
    res_df = remove_doublets(res_df, tolerance=200)
    ###
    
    res_df.to_csv(cfg.out_file, index=False)
    
if __name__ == "__main__":
    my_app()