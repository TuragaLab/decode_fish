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

@hydra.main(config_path='../config', config_name='coloc_eval_p3')
def my_app(cfg):

    out_file = cfg.out_file
    image_paths = sorted(glob.glob(str(Path(cfg.image_path))))
    
    post_proc = hydra.utils.instantiate(cfg.post_proc_isi)
    
    ''' Get DECODE preds (2 models) '''
    
    model_0 = hydra.utils.instantiate(cfg.model).cuda()
    model_0 = load_model_state(model_0, Path(cfg.model_path_0)/'model.pkl')
    
    model_1 = hydra.utils.instantiate(cfg.model).cuda()
    model_1 = load_model_state(model_1, Path(cfg.model_path_1)/'model.pkl')    
    
    dec_df_0_raw = predict(model_0, post_proc, image_paths, sm_fish_ch=0, window_size=[None, 128, 128], device='cuda')
    dec_df_1_raw = predict(model_1, post_proc, image_paths, sm_fish_ch=1, window_size=[None, 128, 128], device='cuda')
    
    ''' Get FQ preds'''
    
    fq0_paths = sorted(glob.glob(str(Path(cfg.fq_path_0))))
    fq1_paths = sorted(glob.glob(str(Path(cfg.fq_path_1))))
    
    fq_df_0_raw = DF()
    fq_df_1_raw = DF()
    
    for g in fq0_paths:
        if 'txt' in g:
            fq_df_0_raw = append_emitter_df(fq_df_0_raw, matlab_fq_to_df(g))
        else:
            fq_df_0_raw = append_emitter_df(fq_df_0_raw, px_to_nm(big_fishq_to_df(g), px_size_zyx=[200,139,139]))
    for g in fq1_paths:
        if 'txt' in g:
            fq_df_1_raw = append_emitter_df(fq_df_1_raw, matlab_fq_to_df(g))
        else:
            fq_df_1_raw = append_emitter_df(fq_df_1_raw, px_to_nm(big_fishq_to_df(g), px_size_zyx=[200,139,139]))
    
    ''' Get shifts '''
    
    fq_df_0 = DF()
    fq_df_1 = DF()
    
    dec_df_0 = DF()
    dec_df_1 = DF()
    
    for  i in range(len(image_paths)):
        
        shape = load_tiff_image(image_paths[i]).shape
        
        _, _, shift_dec = matching(dec_df_0_raw[dec_df_0_raw['frame_idx']  == i], dec_df_1_raw[dec_df_1_raw['frame_idx']  == i], print_res=False)
        _, _, shift_fq = matching(fq_df_0_raw[fq_df_0_raw['frame_idx']  == i], fq_df_1_raw[fq_df_1_raw['frame_idx']  == i], print_res=False)
        
        dec_sub = shift_df(dec_df_1_raw[dec_df_1_raw['frame_idx']  == i], shift_dec)
        fq_sub = shift_df(fq_df_1_raw[fq_df_1_raw['frame_idx']  == i], shift_fq)
        dec_sub['frame_idx'] = 0
        fq_sub['frame_idx'] = 0
        
        dec_sub = exclude_borders(dec_sub, shape, px_size_zyx=[200,139,139], border_size_zyx=[1000,400,400])
        fq_sub = exclude_borders(fq_sub, shape, px_size_zyx=[200,139,139], border_size_zyx=[1000,400,400])
        
        dec_df_1 = append_emitter_df(dec_df_1, dec_sub)
        fq_df_1 = append_emitter_df(fq_df_1, fq_sub)
        
        dec_sub = exclude_borders(dec_df_0_raw[dec_df_0_raw['frame_idx']  == i], shape, px_size_zyx=[200,139,139], border_size_zyx=[1000,400,400])
        fq_sub = exclude_borders(fq_df_0_raw[fq_df_0_raw['frame_idx']  == i], shape, px_size_zyx=[200,139,139], border_size_zyx=[1000,400,400])
        dec_sub['frame_idx'] = 0
        fq_sub['frame_idx'] = 0
        
        dec_df_0 = append_emitter_df(dec_df_0, dec_sub)
        fq_df_0 = append_emitter_df(fq_df_0, fq_sub)
        
    ''' Remove Tx Sites'''
    
    peak_df = DF()

    for i,f in enumerate(image_paths):
        img = load_tiff_image(f)

        img_ch0 = img[0]
        peaks_ch0 = get_peaks_3d(img_ch0, threshold=cfg.filt_tx.threshold, min_distance=20)
        peaks_ch0['frame_idx'] = i
        peak_df = pd.concat([peak_df, peaks_ch0])

    slices = []

    for p in peak_df.iterrows():
        p=p[1]
        slices.append(np.s_[p['frame_idx']:p['frame_idx']+1, 
                            p['z']-cfg.filt_tx.tx_size[0]:p['z']+cfg.filt_tx.tx_size[0], 
                            p['y']-cfg.filt_tx.tx_size[1]:p['y']+cfg.filt_tx.tx_size[1], 
                            p['x']-cfg.filt_tx.tx_size[2]:p['x']+cfg.filt_tx.tx_size[2]])
        
    dec_0_crops = pd.concat([crop_df(dec_df_0, s, px_size_zyx=cfg.filt_tx.px_size_zyx, shift=False) for s in slices])
    dec_df_0 = pd.concat([dec_df_0, dec_0_crops]).drop_duplicates(keep=False)

    dec_1_crops = pd.concat([crop_df(dec_df_1, s, px_size_zyx=cfg.filt_tx.px_size_zyx, shift=False) for s in slices])
    dec_df_1 = pd.concat([dec_df_1, dec_1_crops]).drop_duplicates(keep=False)

    fq_0_crops = pd.concat([crop_df(fq_df_0, s, px_size_zyx=cfg.filt_tx.px_size_zyx, shift=False) for s in slices])
    fq_df_0 = pd.concat([fq_df_0, fq_0_crops]).drop_duplicates(keep=False)

    fq_1_crops = pd.concat([crop_df(fq_df_1, s, px_size_zyx=cfg.filt_tx.px_size_zyx, shift=False) for s in slices])
    fq_df_1 = pd.concat([fq_df_1, fq_1_crops]).drop_duplicates(keep=False)

    with h5py.File(out_file, 'w') as f:
        
        g_loc = f.create_group('localizations')
        
        add_df_to_hdf5(g_loc, 'dec_0', dec_df_0)
        add_df_to_hdf5(g_loc, 'dec_1', dec_df_1)
        add_df_to_hdf5(g_loc, 'fq_0', fq_df_0)
        add_df_to_hdf5(g_loc, 'fq_1', fq_df_1)

        ''' Full performance '''

#        perf_dec_full, matches, shift = matching(dec_df_0, dec_df_1, print_res=False)
#        perf_fq_full, matches, shift = matching(fq_df_0, fq_df_1, print_res=False)
        
        g_perf = f.create_group('evaluations')
        
#        add_df_to_hdf5(g_perf, 'dec_full', DF.from_records([perf_dec_full]))
#        add_df_to_hdf5(g_perf, 'fq_full', DF.from_records([perf_fq_full]))

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

#         ''' Sig filtered performance '''
    
#         dec_df_0_filt = DF()
#         dec_df_1_filt = DF()

#         perf_dicts_dec = []
#         perf_dicts_fq = []

#         for i in range(len(image_paths)):

#             fq_0_sub = fq_df_0[fq_df_0['frame_idx']==i]
#             fq_1_sub = fq_df_1[fq_df_1['frame_idx']==i]

#             dec_0_sub = dec_df_0[dec_df_0['frame_idx']==i]
#             dec_1_sub = dec_df_1[dec_df_1['frame_idx']==i]  

#             perc0 = len(fq_0_sub)/len(dec_0_sub)
#             dec_0_sub_f = filt_perc(dec_0_sub, perc0 * 100)
#             # dec_0_sub_f = filt_perc(dec_0_sub, 100 - perc0 * 100, metric='int', return_low=False)
#             dec_0_sub_f['frame_idx'] = dec_0_sub_f['frame_idx']*0
#             dec_df_0_filt = append_emitter_df(dec_df_0_filt, dec_0_sub_f)

#             perc = len(dec_0_sub_f)/len(dec_1_sub)
#             perc = np.clip(perc,0,1)
#             dec_1_sub_f = filt_perc(dec_1_sub, perc * 100)
#             # dec_1_sub_f = filt_perc(dec_1_sub, 100 - perc * 100, metric='int', return_low=False)
#             dec_1_sub_f['frame_idx'] = dec_1_sub_f['frame_idx']*0
#             dec_df_1_filt = append_emitter_df(dec_df_1_filt, dec_1_sub_f)

#             perf_df_dec, matches_dec, shift_dec = matching(dec_0_sub_f, dec_1_sub_f, print_res=False)
#             perf_dicts_dec.append(perf_df_dec)
            
#         perf_dec_filt, matches, shift = matching(dec_df_0_filt, dec_df_1_filt, print_res=False)
        
#         add_df_to_hdf5(g_perf, 'dec_filt', DF.from_records([perf_dec_filt]))
#         add_df_to_hdf5(g_perf, 'dec_filt_pimg', DF(perf_dicts_dec))     
                
if __name__ == "__main__":
    my_app()