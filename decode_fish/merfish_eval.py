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
import torch.tensor as T

from decode_fish.funcs.fit_psf import get_peaks_3d

from decode_fish.funcs.routines import *
from decode_fish.funcs.predict import merfish_predict
from decode_fish.funcs.exp_specific import *
import wandb

import h5py

from decode_fish.funcs.merfish_eval import *
# from decode_fish.funcs.merfish_codenet import *

@hydra.main(config_path='../config', config_name='merfish_eval')
def my_app(cfg):

    model_cfg = OmegaConf.load(cfg.model_cfg)
    
#    if 'n_cols' not in model_cfg.PSF:
#        model_cfg.PSF.n_cols = 1
#    if 'phasing' not in model_cfg.exp_type:
#        model_cfg.exp_type.phasing = False
    
#    model_cfg.random_crop.crop_sz = cfg.training.crop_sz

    model, post_proc, micro, img_3d, decode_dl = load_all(model_cfg)    
    model.eval().cuda()
    
    bench_df, code_ref, targets = hydra.utils.instantiate(model_cfg.codebook)
    code_inds = np.stack([np.nonzero(c)[0] for c in code_ref])  
    
    image_paths = sorted(glob.glob(cfg.image_path))
    
    if cfg.crop is not None:
        crop = eval(cfg.crop,{'__builtins__': None},{'s_': np.s_})
    else:
        crop = np.s_[:,:,:,:]
        
    res_df = merfish_predict(model, post_proc, image_paths, window_size=[None, 128, 128], crop=crop, device='cuda')      
    
    #res_df = exclude_borders(res_df, border_size_zyx=[0,4000,4000], img_size=[2048*100,2048*100,2048*100])
    res_df['gene'] = targets[res_df['code_inds']]
    
    res_df.to_csv(cfg.out_file, index=False)
    
if __name__ == "__main__":
    my_app()