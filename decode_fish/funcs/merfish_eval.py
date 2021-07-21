# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/19_MERFISH_routines.ipynb (unless otherwise specified).

__all__ = ['index', 'get_bin_code', 'norm_features', 'approximate_nearest_code', 'code_from_groups',
           'get_bin_code_crossframe', 'get_bin_code_frameiter', 'get_bin_code_logorder', 'plot_gene_numbers']

# Cell
from ..imports import *
from .file_io import *
from .emitter_io import *
from .utils import *
from .dataset import *
from .plotting import *
from ..engine.noise import estimate_noise_scale
import shutil
from .visualization import *
from .predict import predict

from numba import njit
from scipy.spatial import cKDTree
from .evaluation import matching

import io, requests
from sklearn.neighbors import NearestNeighbors
from starfish import data
import pprint

# Cell
@njit
def index(array, item):
    for idx, val in np.ndenumerate(array):
        if val == item:
            return idx[0]
    return None

def get_bin_code(frame_idx, n_imgs=16):
    code = np.zeros(n_imgs, dtype='int8')
    code[frame_idx] = 1
    return code

def norm_features(code, norm_order = 2):

    norm = np.linalg.norm(code, ord=norm_order, axis=1)
    code = code / norm[:, None]

    return code

def approximate_nearest_code(ref_code, pred_code, targets):

    nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', metric='euclidean').fit(ref_code)
    metric_output, indices = nn.kneighbors(pred_code)
    gene_ids = np.ravel(targets.values[indices])

    return np.ravel(metric_output), gene_ids

# Cell
def code_from_groups(loc_df):

    loc_df = loc_df[loc_df['group_idx'] > 0]
    loc_df = loc_df.sort_values('group_idx').reset_index(drop=True)
    group_idx = loc_df['group_idx'].values
    group_idx = np.array(group_idx, dtype=np.uint32)

    inds = np.where(np.diff(group_idx))[0] + 1

    xy = loc_df.loc[:,['x','y']].values
    xy_sig = loc_df.loc[:,['x_sig','y_sig']].values
    ints = loc_df.loc[:,'int'].values
    frame_idx = loc_df.loc[:,'frame_idx'].values

    codes, rmses, cc, intsum, logs = [], [], [], [], []

    for i in tqdm(range(len(inds)-1)):

        sl = np.s_[inds[i]:inds[i+1]]

        code = get_bin_code(frame_idx[sl])
        codes.append(code)
        cc.append(code.sum())

        rmses.append(np.sqrt(np.mean((xy[sl] - xy[sl].mean(0))**2)))
        intsum.append(ints[sl].sum())

        log_dists = torch.distributions.normal.Normal(T([0,0]), torch.sqrt(T(xy_sig[sl][None])**2 + T(xy_sig[sl][:,None])**2)). \
                                             log_prob(T(xy[sl][None]) - T(xy[sl][:,None])).sum(-1)

        log_dists = torch.triu(log_dists, diagonal=1)
        logs.append(log_dists.sum()/torch.count_nonzero(log_dists))

    res_df = loc_df.iloc[inds[:-1]].copy()
    res_df['code'] = codes
    res_df['cc'] = cc
    res_df['rmses'] = rmses
    res_df['int'] = intsum
    res_df['logdist'] = np.array(logs)

    return res_df

# Cell
def get_bin_code_crossframe(pred_df, group_rad=150):

    loc_df=pred_df.copy()
    N_imgs = loc_df['frame_idx'].max() + 1

    loc_df['group_idx'] = -1

    group_count = 0

    for i in tqdm(range(0, N_imgs-1)):

        i_inds = (loc_df['frame_idx'] == i) & (loc_df['group_idx'] == -1)
        i_df = loc_df[(loc_df['frame_idx'] == i) & (loc_df['group_idx'] == -1)].reset_index(drop=True)
        loc_df.loc[i_inds,'group_idx'] = np.arange(len(i_df)) + group_count
        i_df.loc[:,'group_idx'] = np.arange(len(i_df)) + group_count
        group_count += len(i_df)

        tree = cKDTree(i_df.loc[:,['x','y']].values)

        for k in range(i+1, N_imgs):

            k_df = loc_df[(loc_df['frame_idx'] == k) & (loc_df['group_idx'] == -1)].reset_index(drop=True)
            dists, inds = tree.query(k_df.loc[:,['x','y']].values, distance_upper_bound=group_rad)

            i_inds, k_inds = np.unique(inds, return_index=True)
            k_inds = list(k_inds[:-1])
            i_inds = list(i_inds[:-1])

            loc_inds = k_df['loc_idx'].values[k_inds]
            loc_df.loc[loc_inds,'group_idx'] = i_df.loc[i_inds,'group_idx'].values

    res_df = code_from_groups(loc_df)

    return res_df

# Cell
def get_bin_code_frameiter(pred_df, group_rad=150, n_nearest=15):

    loc_df=pred_df.copy()

    N_imgs = loc_df['frame_idx'].max() + 1

    loc_df['group_idx'] = -1
    loc_df['group_logs'] = 0
    print(len(loc_df))

    group_count = 0

    for i in tqdm(range(0, N_imgs)):

        i_inds = (loc_df['frame_idx'] == i) & (loc_df['group_idx'] == -1)
        i_df = loc_df[(loc_df['frame_idx'] == i) & (loc_df['group_idx'] == -1)].reset_index(drop=True)
        i_df.loc[:,'group_idx'] = np.arange(len(i_df)) + group_count
        loc_df.loc[i_inds,'group_idx'] = np.arange(len(i_df)) + group_count
        group_count += len(i_df)

        k_df = loc_df[(loc_df['frame_idx'] > i) & (loc_df['group_idx'] == -1)].reset_index(drop=True)

        tree = cKDTree(k_df.loc[:,['x','y']].values)
        dists, inds = tree.query(i_df.loc[:,['x','y']].values, k = n_nearest, distance_upper_bound=group_rad)

        for n in range(n_nearest):

            k_inds, i_inds = np.unique(inds[:, n], return_index=True)
            k_inds = list(k_inds[:-1])
            i_inds = list(i_inds[:-1])


            loc_inds = k_df['loc_idx'].values[k_inds]
            loc_df.loc[loc_inds,'group_idx'] = i_df.loc[i_inds,'group_idx'].values

    return loc_df

# Cell
import torch.tensor as T
def get_bin_code_logorder(pred_df, group_rad=150, log_lim=-500):

    loc_df=pred_df.copy().reset_index(drop=True)

    N_imgs = loc_df['frame_idx'].max() + 1

    loc_df['group_idx'] = 0

    tree1 = cKDTree(loc_df.loc[:,['x','y']].values)
    tree2 = cKDTree(loc_df.loc[:,['x','y']].values)

    sdm = tree1.sparse_distance_matrix(tree2, group_rad, output_type='ndarray')

    frame_filt = loc_df['frame_idx'].values[sdm['i']] != loc_df['frame_idx'].values[sdm['j']]

    dists = sdm['v'][frame_filt]
    k_inds = sdm['i'][frame_filt]
    i_inds = sdm['j'][frame_filt]

    log_dists = torch.distributions.normal.Normal(T([0,0]), torch.sqrt(T(loc_df.loc[i_inds,['x_sig','y_sig']].values)**2 +
                                                                       T(loc_df.loc[k_inds,['x_sig','y_sig']].values)**2)). \
                                                                       log_prob(T(loc_df.loc[k_inds,['x','y']].values) - T(loc_df.loc[i_inds,['x','y']].values))

    log_dists = log_dists.sum(-1)
    inds = np.argsort(log_dists).flip(0)
    # Every entry is double because the pointclouds are the same.
    log_dists, k_inds, i_inds = [s[inds][::2] for s in [log_dists, k_inds, i_inds]]

    if log_lim:
        inds = log_dists > log_lim
        log_dists, k_inds, i_inds = [s[inds] for s in [log_dists, k_inds, i_inds]]

#     return group_idx, log_dists, k_inds, i_inds
#     group_idx = log_loop(np.array(group_idx), np.array(log_dists), np.array(k_inds), np.array(i_inds))

    group_count = 1
    # Operate on array for much faster calculations
    group_idx = np.array(loc_df['group_idx'].values, dtype=np.uint32)

    for d, k, i in tqdm(zip(np.array(log_dists), np.array(k_inds, dtype=np.uint32), np.array(i_inds, dtype=np.uint32))):

        i_gidx = group_idx[i]
        k_gidx = group_idx[k]

        sum_gidx = i_gidx + k_gidx

        if not sum_gidx:
            # Both not grouped, create new group
            group_idx[i] = group_count
            group_idx[k] = group_count
            group_count += 1
        else:
            if not k_gidx:
                # k not grouped, i grouped. Assign k to i group
                if np.count_nonzero(group_idx == i_gidx) < 4:
                    group_idx[k] = i_gidx
            elif not i_gidx:
                # i not grouped, k grouped. Assign i to k group
                if np.count_nonzero(group_idx == k_gidx) < 4:
                    group_idx[i] = k_gidx
            else:
                if k_gidx != i_gidx:
                    # Both grouped in different groups. Connect if len <= 5
                    if np.count_nonzero(group_idx == k_gidx) + np.count_nonzero(group_idx == i_gidx) <= 5:
                        group_idx[group_idx == k_gidx] = i_gidx

    loc_df['group_idx'] = group_idx
    return loc_df

# Cell
def plot_gene_numbers(bench_counts, res_counts, title='', log=True, corr=True):

    if corr:
        r = np.corrcoef(bench_counts, res_counts)[0, 1]
        r = np.round(r, decimals=3)
    else:
        r = np.sum(res_counts)
    x_lim = np.max([bench_counts.max(), res_counts.max()])
    x = np.linspace(0, x_lim)

    plt.scatter(bench_counts, res_counts, 50, zorder=2)
    plt.plot(x, x, '-k', zorder=1)

    plt.xlabel('Gene copy number Benchmark')
    plt.ylabel('Gene copy number DECODE')
    if log:
        plt.xscale('log')
        plt.yscale('log')
    plt.title(f'{title} r = {r}');