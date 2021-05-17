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

from decode_fish.funcs.fit_psf import get_peaks_3d

from decode_fish.funcs.train_funcs import *
from decode_fish.funcs.routines import *
import wandb

import h5py

@hydra.main(config_path='/groups/turaga/home/speisera/Dropbox (mackelab)/Artur/WorkDB/deepstorm/decode_fish/config', config_name='sim_eval')
def my_app(cfg):

    model_dir = cfg.model_dir
    
    if cfg.model_names is not None:
        model_names = list(cfg.model_names)
    else:
        model_names = [s.split('/')[-1] for s in glob.glob(cfg.model_dir+'/*')]
    out_file = cfg.out_file
    
    model = hydra.utils.instantiate(cfg.model)
    model.cuda()
    
    post_proc = hydra.utils.instantiate(cfg.post_proc_isi, samp_threshold=0.5)
    
    with h5py.File(out_file, 'w') as f:
        
        if cfg.eval_routines.densities:
        
            """ Random Eval over densities """
            print('Evaluation over densities')

            g = f.create_group('eval_densities')

            sl = np.s_[:,:,:,:]
            basedir = '/groups/turaga/home/speisera/share_TUM/FishSIM/sim_density_fac1_2/'

            densities = [250,500] #  [250,500, 1000, 2000, 4000]
            n_cells = 5

            g = f['eval_densities']    
            for d in tqdm(densities):
                print('Density: ',d)

                gg = g.create_group(str(d))

                df_col = {k:DF() for k in ['GT','FQ'] + model_names}

                for ind in range(n_cells):

                    img, gt_df, fq_nog_df, fq_gmm_df = load_sim_fish(basedir, d, 'random', 'NR', ind)
                    fq_gmm_df = crop_df(fq_gmm_df, sl, px_size_zyx=[300,100,100])

                    df_col['GT'] = cat_emitter_dfs([df_col['GT'], gt_df])
                    df_col['FQ'] = cat_emitter_dfs([df_col['FQ'], fq_gmm_df])

                    with torch.no_grad():
                        for m in model_names:
                            model = load_model_state(model, model_dir + m, 'model.pkl')
                            dec_df = shift_df(post_proc(model(img[sl][None].cuda()), 'df'), [-100,-100,-300])
                            df_col[m] = cat_emitter_dfs([df_col[m], dec_df])
                            free_mem()

                for k in df_col.keys():
                    gg.create_group(k)
                    add_df_to_hdf5(gg[k], 'locations', df_col[k])

                    if 'GT' not in k:
                        perf_df, matches, shift = matching(df_col['GT'], df_col[k], print_res=False)
                        df_col[k] = shift_df(df_col[k], shift)
                        perf_df, matches, _ = matching(df_col['GT'], df_col[k], print_res=False)
                        add_df_to_hdf5(gg[k], 'performance', DF.from_records([perf_df]))
                 
        if cfg.eval_routines.foci:

            """ Foci Eval """
            print('Foci evaluation')

            g = f.create_group('eval_foci')

            basedir = '/groups/turaga/home/speisera/share_TUM/FishSIM/sim_foci_fac1_1/'

            box_sz = 10
            n_cells = 20

            count_col = {k:[] for k in ['GT','FQ'] + model_names}

            for ind in tqdm(range(n_cells)):

                df_col = {k:DF() for k in ['GT','FQ'] + model_names}

                img, gt_df, fq_nog_df, fq_gmm_df = load_sim_fish(basedir, 100, 'foci', 'strong', ind)
                df_col['GT'] = cat_emitter_dfs([df_col['GT'], nm_to_px(gt_df, px_size_zyx=[300,100,100])]) 
                df_col['FQ'] = cat_emitter_dfs([df_col['FQ'], nm_to_px(fq_gmm_df, px_size_zyx=[300,100,100])]) 

                with torch.no_grad():
                    for m in model_names:
                        model = load_model_state(model, model_dir + m, 'model.pkl')
                        res_dict = model(img[None].cuda())
                        dec_df = shift_df(post_proc(res_dict, 'df'), [-100,-100,-300])
                        df_col[m] = cat_emitter_dfs([df_col[m], nm_to_px(dec_df, px_size_zyx=[300,100,100])]) 
                        free_mem()

                try:
                    coords_xyz = get_peaks_3d(img[0], threshold=2000, min_distance=10)
                except AssertionError:
                    continue

                coords_zyx = coords_xyz[:,::-1]

                for c in coords_zyx:

                    sl = np.s_[:,c[0]-box_sz:c[0]+box_sz+1, c[1]-box_sz:c[1]+box_sz+1, c[2]-box_sz:c[2]+box_sz+1]

                    for k in df_col.keys():

                        df_crop = crop_df(df_col[k], sl)
                        count_col[k].append(len(df_crop))

            for k in count_col.keys():

                gg = g.create_group(k)
                gg.create_dataset('foci_counts', data=count_col[k])
                
        if cfg.eval_routines.microscope:

            """ Microscope save """
            print('Save microscope state')

            g = f.create_group('microscope_state')
            for m in model_names:
                
                gg = g.create_group(m)
                model = load_model_state(model, model_dir + m + '/sl_save/', 'model.pkl')
                train_cfg = OmegaConf.load(model_dir + m + '/train.yaml')
                
                gg.create_dataset('conc_pre', data=model.int_dist.int_conc.item())                
                gg.create_dataset('rate_pre', data=model.int_dist.int_rate.item())                
                gg.create_dataset('loc_pre', data=model.int_dist.int_loc.item())                
                gg.create_dataset('scale', data=train_cfg.microscope.scale) 
                
                model = load_model_state(model, model_dir + m, 'model.pkl')

                gg.create_dataset('conc_post', data=model.int_dist.int_conc.item())                
                gg.create_dataset('rate_post', data=model.int_dist.int_rate.item())                
                gg.create_dataset('loc_post', data=model.int_dist.int_loc.item())  
                
        if cfg.eval_routines.psf:

            """ PSF save """
            print('Save PSF state')

            g = f.create_group('psf_state')
            
            for i,m in enumerate(model_names):
                
                train_cfg = OmegaConf.load(model_dir + m + '/train.yaml')
                if not i:
                    g.create_dataset('gt_psf', data=cpu(load_tiff_image(train_cfg.evaluation.psf_path)[0]))
                
                gg = g.create_group(m)

                psf_init = load_psf(train_cfg)
                gg.create_dataset('init_psf', data=cpu(psf_init.psf_volume[0]))     
                
                psf_init.load_state_dict(torch.load(Path(train_cfg.output.save_dir)/'psf.pkl'))
                gg.create_dataset('fit_psf', data=cpu(psf_init.psf_volume[0]))               
                
if __name__ == "__main__":
    my_app()