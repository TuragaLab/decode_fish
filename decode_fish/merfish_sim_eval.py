from decode_fish.imports import *
from decode_fish.funcs.file_io import *
from decode_fish.funcs.emitter_io import *
from decode_fish.funcs.utils import *
from decode_fish.funcs.output_trafo import *
from decode_fish.funcs.matching import *
import shutil
from decode_fish.engine.point_process import PointProcessUniform

from decode_fish.funcs.routines import *
from decode_fish.funcs.merfish_comparison import *
import wandb

import h5py

sys.path.append('/groups/turaga/home/speisera/Mackebox/Artur/WorkDB/deepstorm/FQ/istdeco/')

from istdeco import ISTDeco
from utils import random_codebook, random_image_stack
import matplotlib.pyplot as plt
from codebook import Codebook
# from starfish.image import Filter

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

@hydra.main(config_path='../config', config_name='merfish_sim_eval')
def my_app(cfg):
    
    os.makedirs(cfg.data_dir + cfg.sweep_name, exist_ok=True)

    def crop_f(df):
        return exclude_borders(df, border_size_zyx=[0,500,500], img_size=[cfg.crop_sz*100,cfg.crop_sz*100,cfg.crop_sz*100])
        
    model_dir = cfg.model_dir
    
    if cfg.model_names is not None:
        model_names = [cfg.model_names]
    else:
        model_names = [s.split('/')[-1] for s in glob.glob(cfg.model_dir+'/*')]
        
    out_file = cfg.data_dir + cfg.sweep_name + '/' + cfg.data_file
    print(out_file)
    
    with h5py.File(cfg.data_dir + cfg.data_file, 'r') as f:
        
        xsimn = f['frames'][()]
        gt_df = get_df_from_hdf5(f['locations'])
    
    with h5py.File(out_file, 'a') as f:
        
        """Evaluate DECODE"""
        if cfg.eval_dec:
        
            for m in model_names:

                if m in f: del f[m]
                g = f.create_group(m)

                with torch.no_grad():

                    model_cfg = OmegaConf.load(f'{model_dir}/{m}/train.yaml')
                    
                    codebook, targets = hydra.utils.instantiate(model_cfg.codebook)
                    
                    model, post_proc, _, _, _ = load_all(model_cfg)
                    model.cuda()        

                    dec_df = get_prediction(model,torch.tensor(xsimn).cuda(), post_proc, cuda=True)
                    free_mem()

                    dec_df = crop_f(dec_df)

                    perf, _, _  = matching(gt_df, dec_df, match_genes=True, print_res=False)

                    add_df_to_hdf5(g, 'locations', dec_df)   
                    add_df_to_hdf5(g, 'perf', DF.from_records([perf]))
                    
        """Evaluate ISTDECO"""
        if cfg.eval_istdeco:
            
            if 'istdeco' in f: del f['istdeco']
            g = f.create_group('istdeco')

            istd_df = get_istdeco_df(xsimn, codebook.reshape([140,8,2], order='F'), psf_sig=(1.5, 1.5), n_iter=100, bg=100.)
            istd_df = crop_f(istd_df)

            q_max = 0.3*istd_df['quality'].max()
            i_max = 0.3*istd_df['intensity'].max()
            i_min = istd_df['intensity'].min()

            def objective(trial):

                qual_th = trial.suggest_uniform('qual_th', 0.1, q_max)
                int_th = trial.suggest_uniform('int_th', i_min, i_max)

                sub_df = istd_df[(istd_df['intensity'] > int_th) & (istd_df['quality'] > qual_th)]
                perf, matches, _  = matching(gt_df, sub_df, match_genes=True, print_res=False) 

                return -perf['jaccard']

            study = optuna.create_study()
            study.optimize(objective, n_trials=50)  
            
            # print(-study.best_value)

            sub_df = istd_df[(istd_df['intensity'] > study.best_params['int_th']) & (istd_df['quality'] > study.best_params['qual_th'])]
            perf, matches, _  = matching(gt_df, sub_df, match_genes=True, print_res=False) 

            add_df_to_hdf5(g, 'locations', sub_df)  
            add_df_to_hdf5(g, 'locations_raw', istd_df)  
            add_df_to_hdf5(g, 'perf', DF.from_records([perf])) 
            
        """Evaluate BARDENSR"""
        if cfg.eval_bardensr:
            
            if 'bardensr' in f: del f['bardensr']
            g = f.create_group('bardensr')
            n_iter = 300
            
            def objective(trial):

                l1_pen = trial.suggest_uniform('l1_pen', 0., 0.1)
                th     = trial.suggest_uniform('th', 0.1, 0.5)

                evd_tensors = get_bardensr_tensor(xsimn, codebook, n_iter=int(n_iter), l1_pen=l1_pen)
                bard_df = get_bardensr_df(evd_tensors, th)
                bard_df = crop_f(bard_df)

                perf, matches, _  = matching(gt_df, bard_df, match_genes=True, print_res=False) 

                return -perf['jaccard']

            study = optuna.create_study()
            study.optimize(objective, n_trials=30)
            
            evd_tensors = get_bardensr_tensor(xsimn, codebook, n_iter=int(n_iter), l1_pen=study.best_params['l1_pen'])
            bard_df = get_bardensr_df(evd_tensors, study.best_params['th'])
            bard_df = crop_f(bard_df)
            
            perf, matches, _  = matching(gt_df, bard_df, match_genes=True, print_res=False) 
            
            add_df_to_hdf5(g, 'locations', bard_df)  
            add_df_to_hdf5(g, 'perf', DF.from_records([perf])) 

                
if __name__ == "__main__":
    my_app()