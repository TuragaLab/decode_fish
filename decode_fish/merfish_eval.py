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

from decode_fish.funcs.train_funcs import *
from decode_fish.funcs.routines import *
from decode_fish.funcs.predict import merfish_predict
import wandb

import h5py

from decode_fish.funcs.merfish_eval import *
from decode_fish.funcs.merfish_codenet import *

@hydra.main(config_path='../config', config_name='merfish_eval')
def my_app(cfg):

    model_cfg = OmegaConf.load(cfg.model_cfg)
    if 'n_cols' not in model_cfg.PSF:
        model_cfg.PSF.n_cols = 1
    if 'phasing' not in model_cfg.exp_type:
        model_cfg.exp_type.phasing = False
    
    model_cfg.random_crop.crop_sz = cfg.training.crop_sz
    model, post_proc, micro, img_3d, decode_dl = load_all(model_cfg)    
    model.eval().cuda()
    
    bench_df, code_ref, targets = get_benchmark()
    code_inds = np.stack([np.nonzero(c)[0] for c in code_ref])  
    
    image_paths = sorted(glob.glob(model_cfg.data_path.image_path))
    res_df = merfish_predict(model, post_proc, image_paths, window_size=[None, 256, 256], device='cuda')    
    
    if cfg.training.enabled:
    
        net = code_net().cuda()

        bce = torch.nn.BCEWithLogitsLoss()
        opt = hydra.utils.instantiate(cfg.training.opt, params=net.parameters())
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=500, gamma=0.5)    

        batch_size = 100

        loss_col = []

        for i in tqdm(range(cfg.training.num_iters)):

            with torch.no_grad():

                x, local_rate, background = next(iter(decode_dl))
                sim_vars = PointProcessUniform(local_rate*cfg.training.rate_fac,model.int_dist.int_conc, model.int_dist.int_rate, 
                                               model.int_dist.int_loc, channels=16, n_bits=4, sim_z=model_cfg.exp_type.pred_z, 
                                               codebook=torch.tensor(code_inds)).sample(from_code_book=True, phasing=model_cfg.exp_type.phasing)
                xsim = micro(*sim_vars, add_noise=True)
                xsimn = micro.noise(xsim, background, const_theta_sim=False).sample()

                gt_vars = sim_vars[:-1]
                gt_df = sample_to_df(*gt_vars, px_size_zyx=[1.,1.,1.])

                res_dict = model(xsimn.cuda())
                res_dict = model.tensor_to_dict(res_dict)
                pred_df = post_proc.get_df(res_dict)
                
                perf, matches, _ = matching(px_to_nm(gt_df), pred_df, tolerance=500, print_res=False)
                
                pred_df = get_code_from_ints(pred_df, code_ref, targets, int_str='')
                pred_df = pred_df.set_index('loc_idx')
                matches = get_code_from_ints(matches, code_ref, targets, int_str='_tar')
                pred_df.loc[matches['loc_idx_pred'],'matched_code'] = matches['code_inds'].values
                pred_df['gt_match'] = np.zeros(len(pred_df))
                pred_df.loc[pred_df['code_inds']==pred_df['matched_code'], 'gt_match'] += 1

            matches = matches.sample(frac=1).reset_index(drop=True)

            for b in range(len(matches)//batch_size):

                opt.zero_grad()
                
                net_inp = T(input_from_df(pred_df[b*batch_size:(b+1)*batch_size]), dtype=torch.float32).cuda()
                net_out = net(net_inp)

                net_tar = T(pred_df[b*batch_size:(b+1)*batch_size]['gt_match'].values, dtype=torch.float32).cuda()
                loss = bce(net_out, net_tar[:,None])

                loss.backward()
                opt.step()
                sched.step()

                loss_col.append(loss.item())   
    
        out_vals = []
        res_df = get_code_from_ints(res_df, code_ref, targets, int_str='')
        for b in tqdm(range(len(res_df)//100 + 1)):

            if len(res_df[b*100:(b+1)*100]):
                net_inp = T(input_from_df(res_df[b*100:(b+1)*100]), dtype=torch.float32).cuda()
                net_out = torch.sigmoid(net(net_inp))
                out_vals.append(np.array(cpu(net_out)))    

        out_arr = np.concatenate(out_vals)
        res_df['net_qual'] = out_arr    
    
    res_df = exclude_borders(res_df, border_size_zyx=[0,4000,4000], img_size=[2048*100,2048*100,2048*100])
    res_df = get_code_from_ints(res_df, code_ref, targets, vcorrcoef)
    
    res_df.to_csv(cfg.out_file, index=False)
    
if __name__ == "__main__":
    my_app()