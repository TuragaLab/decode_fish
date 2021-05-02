# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/11_emitter_io.ipynb (unless otherwise specified).

__all__ = ['shift_df', 'sig_filt', 'nm_to_px', 'px_to_nm', 'cat_emitter_dfs', 'crop_df']

# Cell
from ..imports import *

# Cell
def shift_df(df, shift=[0.,0.,0.]):

    df_corr = df.copy()
    df_corr['x'] += shift[0]
    df_corr['y'] += shift[1]
    df_corr['z'] += shift[2]
    return df_corr

def sig_filt(df, perc = 90, return_low=True):
    if perc >= 100:
        return df
    filt_val = np.percentile(df['comb_sig'], perc)
    if return_low:
        return df[df['comb_sig'] < filt_val]
    else:
        return df[df['comb_sig'] > filt_val]

#export
def nm_to_px(df, px_size_zyx=[100.,100.,100.]):

    df_corr = df.copy()
    df_corr['x'] /= px_size_zyx[2]
    df_corr['y'] /= px_size_zyx[1]
    df_corr['z'] /= px_size_zyx[0]
    return df_corr

#export
def px_to_nm(df, px_size_zyx=[100.,100.,100.]):

    df_corr = df.copy()
    df_corr['x'] *= px_size_zyx[2]
    df_corr['y'] *= px_size_zyx[1]
    df_corr['z'] *= px_size_zyx[0]
    return df_corr

#export
def cat_emitter_dfs(df_list):
    ret_df = df_list[0]
    for df in df_list[1:]:
        dfc = df.copy()
        if len(ret_df):
            dfc['frame_idx'] += ret_df['frame_idx'].values[-1] + 1
            dfc['loc_idx'] += ret_df['loc_idx'].values[-1] + 1
        ret_df = pd.concat([ret_df, dfc], ignore_index=True)
    return ret_df

#export
def crop_df(df, fzyx_sl=np.s_[:,:,:,:], shift=True, px_size_zyx=[1.,1.,1.]):

    df_crop = df.copy()
    for sl, key, px_s in zip(fzyx_sl, ['frame_idx','z','y','x'], [1] + px_size_zyx):
        if sl.start:
            df_crop = df_crop[df_crop[key] > px_s*sl.start]
        if sl.stop:
            df_crop = df_crop[df_crop[key] < px_s*sl.stop]
        if (shift) & (sl.start is not None):
            df_crop[key] -= px_s*sl.start

    return df_crop