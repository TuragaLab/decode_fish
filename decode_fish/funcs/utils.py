# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/12_utils.ipynb (unless otherwise specified).

__all__ = ['free_mem', 'crop_vol', 'center_crop', 'smooth', 'plot_tb_logs']

# Cell
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from ..imports import *
import gc

# Cell
def free_mem():
    gc.collect()
    torch.cuda.empty_cache()

def crop_vol(vol, fxyz_sl=np.s_[:,:,:,:], px_size=[1.,1.,1.]):

    vol_sl = tuple([fxyz_sl[i] for i in [0,3,2,1]])
    if vol.ndim == 3:
        vol = vol[vol_sl[-3:]]
    else:
        vol = vol[vol_sl]

    return vol

def center_crop(volume, zyx_ext):

    shape_3d = volume.shape[-3:]
    center = [s//2 for s in shape_3d]
    volume = volume[...,center[0]-math.floor(zyx_ext[0]/2):center[0]+math.ceil(zyx_ext[0]/2),
                        center[1]-math.floor(zyx_ext[1]/2):center[1]+math.ceil(zyx_ext[1]/2),
                        center[2]-math.floor(zyx_ext[2]/2):center[2]+math.ceil(zyx_ext[2]/2)]
    return volume

def smooth(x,window_len=11,window='flat'):

    if window_len<3:
        return x

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def plot_tb_logs(exps, metric='Sim. Metrics/eff_3d', window_len=1):
    all_vals = []
    for exp in exps:
        event_acc = EventAccumulator(exp)
        event_acc.Reload()
        w_times, step_nums, vals = zip(*event_acc.Scalars(metric))
        all_vals.append(vals)

    for v,e in zip(all_vals,exps):
        plt.plot(smooth(v, window_len), label=e.split('/')[-1])
#         print(np.array(v).max().round(2), np.array(v).min().round(2), len(v), e)
    plt.legend()