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

from decode_fish.funcs.train_funcs import *
from decode_fish.funcs.routines import *
from decode_fish.funcs.predict import predict
import wandb

import h5py

@hydra.main(config_path='../config', config_name='coloc_eval')
def my_app(cfg):

    out_file = cfg.out_file
    image_paths = sorted(glob.glob(str(Path(cfg.image_path))))
    
    post_proc = hydra.utils.instantiate(cfg.post_proc_isi)
    
    ''' Get DECODE preds (2 models) '''
    
#     model_0 = hydra.utils.instantiate(cfg.model).cuda()
#     model_0 = load_model_state(model_0, Path(cfg.model_path_0)/'model.pkl')
    
#     model_1 = hydra.utils.instantiate(cfg.model).cuda()
#     model_1 = load_model_state(model_1, Path(cfg.model_path_1)/'model.pkl')    
    
#     dec_df_0 = predict(model_0, post_proc, image_paths, sm_fish_ch=0, window_size=[None, 128, 128], device='cuda')
#     dec_df_1 = predict(model_1, post_proc, image_paths, sm_fish_ch=1, window_size=[None, 128, 128], device='cuda')

    
    ''' Get DECODE preds (6 models) '''
    
    model_0 = hydra.utils.instantiate(cfg.model).cuda()
    model_1 = hydra.utils.instantiate(cfg.model).cuda()

    dec_df_0 = DF()
    dec_df_1 = DF()
    
    for p in image_paths:
        
        if '570_p1' in p:
            model_0 = load_model_state(model_0, Path(sorted(glob.glob(cfg.model_dir_0))[0])/'model.pkl')
            model_1 = load_model_state(model_1, Path(sorted(glob.glob(cfg.model_dir_1))[0])/'model.pkl')
        if '570_p2' in p:
            model_0 = load_model_state(model_0, Path(sorted(glob.glob(cfg.model_dir_0))[1])/'model.pkl')
            model_1 = load_model_state(model_1, Path(sorted(glob.glob(cfg.model_dir_1))[1])/'model.pkl')
        if '570_p3' in p:
            model_0 = load_model_state(model_0, Path(sorted(glob.glob(cfg.model_dir_0))[2])/'model.pkl')
            model_1 = load_model_state(model_1, Path(sorted(glob.glob(cfg.model_dir_1))[2])/'model.pkl')
            
        df_0 = predict(model_0, post_proc, [p], sm_fish_ch=0, window_size=[None, 128, 128], device='cuda')
        df_1 = predict(model_1, post_proc, [p], sm_fish_ch=1, window_size=[None, 128, 128], device='cuda')   
        
        dec_df_0 = append_emitter_df(dec_df_0, df_0)
        dec_df_1 = append_emitter_df(dec_df_1, df_1)

    
    ''' Get FQ preds'''
    
    base_dir = '/groups/turaga/home/speisera/Mackebox/Artur/WorkDB/deepstorm/datasets/CodFish/raw_data_for_codetection'
    
    fq0_paths = sorted(glob.glob(str(Path(base_dir + '/detections_for_codetection/YFP_ch1/*.csv'))))
    fq1_paths = sorted(glob.glob(str(Path(base_dir + '/detections_for_codetection/dlg1_ch2/*.csv'))))
    
    fq_df_0 = DF()
    fq_df_1 = DF()
    
    for g in fq0_paths:
        fq_df_0 = append_emitter_df(fq_df_0, big_fishq_to_df(g))
    for g in fq1_paths:
        fq_df_1 = append_emitter_df(fq_df_1, big_fishq_to_df(g))

    fq_df_0 = px_to_nm(fq_df_0)
    fq_df_1 = px_to_nm(fq_df_1)
    
    ''' Get shifts '''
    
    _, _, shift_dec = matching(dec_df_0[dec_df_0['frame_idx']<11], dec_df_1[dec_df_1['frame_idx']<11], print_res=False)
    _, _, shift_fq = matching(fq_df_0[fq_df_0['frame_idx']<11], fq_df_1[fq_df_1['frame_idx']<11], print_res=False)
    
    dec_df_1 = shift_df(dec_df_1, shift_dec)
    fq_df_1 = shift_df(fq_df_1, shift_fq)
    
    with h5py.File(out_file, 'w') as f:
        
        g_loc = f.create_group('localizations')
        
        add_df_to_hdf5(g_loc, 'dec_0', dec_df_0)
        add_df_to_hdf5(g_loc, 'dec_1', dec_df_1)
        add_df_to_hdf5(g_loc, 'fq_0', fq_df_0)
        add_df_to_hdf5(g_loc, 'fq_1', fq_df_1)

        ''' Full performance '''

        perf_dec_full, matches, shift = matching(dec_df_0, dec_df_1, print_res=False)
        perf_fq_full, matches, shift = matching(fq_df_0, fq_df_1, print_res=False)
        
        g_perf = f.create_group('evaluations')
        
        add_df_to_hdf5(g, 'dec_full', DF.from_records([perf_dec_full]))
        add_df_to_hdf5(g, 'fq_full', DF.from_records([perf_fq_full]))

        ''' Per image performance '''

        perf_dicts_dec = []
        perf_dicts_fq = []

        for i in range(len(image_paths)):

            fq_0_sub = fq_df_0[fq_df_0['frame_idx']==i]
            fq_1_sub = fq_df_1[fq_df_1['frame_idx']==i]

            dec_0_sub = dec_df_0[dec_df_0['frame_idx']==i]
            dec_1_sub = dec_df_1[dec_df_1['frame_idx']==i]  

            perf_df_dec, _, _ = matching(dec_0_sub, dec_1_sub, print_res=False)
            perf_df_fq, _, _ = matching(fq_0_sub, fq_1_sub, print_res=False)       

            perf_dicts_dec.append(perf_df_dec)
            perf_dicts_fq.append(perf_df_fq) 
            
        add_df_to_hdf5(g_perf, 'dec_full_pimg', DF(perf_dicts_dec))
        add_df_to_hdf5(g_perf, 'fq_full_pimg', DF(perf_dicts_fq))       

        ''' Sig filtered performance '''
    
        dec_df_0_filt = DF()
        dec_df_1_filt = DF()

        perf_dicts_dec = []
        perf_dicts_fq = []

        for i in range(len(image_paths)):

            fq_0_sub = fq_df_0[fq_df_0['frame_idx']==i]
            fq_1_sub = fq_df_1[fq_df_1['frame_idx']==i]

            dec_0_sub = dec_df_0[dec_df_0['frame_idx']==i]
            dec_1_sub = dec_df_1[dec_df_1['frame_idx']==i]  

            perc0 = len(fq_0_sub)/len(dec_0_sub)
            dec_0_sub_f = filt_perc(dec_0_sub, perc0 * 100)
            # dec_0_sub_f = filt_perc(dec_0_sub, 100 - perc0 * 100, metric='int', return_low=False)
            dec_0_sub_f['frame_idx'] = dec_0_sub_f['frame_idx']*0
            dec_df_0_filt = append_emitter_df(dec_df_0_filt, dec_0_sub_f)

            perc = len(dec_0_sub_f)/len(dec_1_sub)
            perc = np.clip(perc,0,1)
            dec_1_sub_f = filt_perc(dec_1_sub, perc * 100)
            # dec_1_sub_f = filt_perc(dec_1_sub, 100 - perc * 100, metric='int', return_low=False)
            dec_1_sub_f['frame_idx'] = dec_1_sub_f['frame_idx']*0
            dec_df_1_filt = append_emitter_df(dec_df_1_filt, dec_1_sub_f)

            perf_df_dec, matches_dec, shift_dec = matching(dec_0_sub_f, dec_1_sub_f, print_res=False)
            perf_dicts_dec.append(perf_df_dec)
            
        perf_dec_filt, matches, shift = matching(dec_df_0_filt, dec_df_1_filt, print_res=False)
        
        add_df_to_hdf5(g_perf, 'dec_filt', DF.from_records([perf_dec_filt]))
        add_df_to_hdf5(g_perf, 'dec_filt_pimg', DF(perf_dicts_dec))     
                
if __name__ == "__main__":
    my_app()