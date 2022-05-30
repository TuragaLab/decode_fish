# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/10_evaluation.ipynb (unless otherwise specified).

__all__ = ['matching']

# Cell
from ..imports import *
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

# Cell
def matching(target_df, pred_df, tolerance=1000, print_res=True, eff_const=0.5, match_genes=True, self_match=False, allow_multiple_matches=False, ignore_z=False):
    """Matches localizations to ground truth positions and provides assessment metrics used in the SMLM2016 challenge.
    (see http://bigwww.epfl.ch/smlm/challenge2016/index.html?p=methods#6)
    When using default parameters exactly reproduces the procedure used for the challenge (i.e. produces same numbers as the localization tool).

    Parameters
    ----------
    test_csv: str or list
        Ground truth positions with columns: 'localization', 'frame', 'x', 'y', 'z'
        Either list or str with locations of csv file.
    pred_inp: list
        List of localizations
    size_xy: list of floats
        Size of processed recording in nano meters
    tolerance: float
        Localizations are matched when they are within a circle of the given radius.
    tolerance_ax: float
        Localizations are matched when they are closer than this value in z direction. Should be ininity for 2D recordings. 500nm is used for 3D recordings in the challenge.
    border: float
        Localizations that are close to the edge of the recording are excluded because they often suffer from artifacts.
    print_res: bool
        If true prints a list of assessment metrics.
    min_int: bool
        If true only uses the brightest 75% of ground truth locations.
        This is the setting used in the leaderboard of the challenge. However this implementation does not exactly match the method used in the localization tool.

    Returns
    -------
    perf_dict, matches: dict, list
        Dictionary of perfomance metrics.
        List of all matches localizations for further evaluation in format: 'localization', 'frame', 'x_true', 'y_true', 'z_true', 'x_pred', 'y_pred', 'z_pred', 'int_true', 'x_sig', 'y_sig', 'z_sig'

    """

    perf_dict = None
    match_df = pd.DataFrame()
    TP = 0
    FP = 0.0001
    FN = 0.0001
    MSE_vol = 0

    match_list = []
    tar_cols = target_df.keys()
    pred_cols = pred_df.keys()

    if len(pred_df):

        for i in range(0, pred_df['frame_idx'].max() + 1):

            FC = 0
            sub_tar = target_df[target_df['frame_idx']==i].reset_index()
            sub_pred = pred_df[pred_df['frame_idx']==i].reset_index()

            if ignore_z:
                tar_xyz = sub_tar[['x','y']]
                pred_xyz = sub_pred[['x','y']]
            else:
                tar_xyz = sub_tar[['x','y','z']]
                pred_xyz = sub_pred[['x','y','z']]

            if match_genes:
                tar_gene = sub_tar['code_inds']
                pred_gene = sub_pred['code_inds']

            u_dists = []
            u_p_inds = []
            u_t_inds = []

            if len(tar_xyz) and len(pred_xyz):

                tar_tree = cKDTree(tar_xyz)
                pred_tree = cKDTree(pred_xyz)
                sdm = tar_tree.sparse_distance_matrix(pred_tree, tolerance, output_type='ndarray')

                sort_inds = np.argsort(sdm['v'])
                dists = sdm['v'][sort_inds]
                t_inds = sdm['i'][sort_inds]
                p_inds = sdm['j'][sort_inds]

                for d,p,t in zip(dists, p_inds, t_inds):

                    if allow_multiple_matches is False:
                        condition = p not in u_p_inds and t not in u_t_inds
                    else:
                        condition = p not in u_p_inds
                    if self_match:
                        condition = p not in u_p_inds and p not in u_t_inds and d>0

                    if condition:
                        if match_genes:
                            if pred_gene[p] != tar_gene[t]:
                                continue

                        u_dists.append(d)
                        u_p_inds.append(p)
                        u_t_inds.append(t)

            MSE_vol += (np.array(u_dists) ** 2).sum()

            TP += len(u_t_inds)

            FP += len(pred_xyz) - len(u_t_inds)
            FN += len(tar_xyz) - len(u_t_inds)

            if len(u_t_inds): match_list.append(np.concatenate([sub_tar.loc[u_t_inds,tar_cols].values, sub_pred.loc[u_p_inds,pred_cols].values], 1))

    if len(match_list): match_list = np.concatenate(match_list, 0)

    match_df = pd.DataFrame(match_list, columns = [k+'_tar' for k in tar_cols] + [k+'_pred' for k in pred_cols])

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    jaccard = TP / (TP + FP + FN)

    rmse_vol = np.sqrt(MSE_vol / (TP + 0.00001))

    eff_3d = 100-np.sqrt((100-100*jaccard)**2 + eff_const**2 * rmse_vol**2)

    rmse_x = np.nan
    rmse_y = np.nan
    rmse_z = np.nan

    x_s = np.nan
    y_s = np.nan
    z_s = np.nan

    if len(match_df):
        rmse_x = np.sqrt(((match_df['x_tar']-match_df['x_pred'])**2).mean())
        rmse_y = np.sqrt(((match_df['y_tar']-match_df['y_pred'])**2).mean())
        rmse_z = np.sqrt(((match_df['z_tar']-match_df['z_pred'])**2).mean())

        x_s = (match_df['x_tar']-match_df['x_pred']).mean()
        y_s = (match_df['y_tar']-match_df['y_pred']).mean()
        z_s = (match_df['z_tar']-match_df['z_pred']).mean()

    if print_res:
        print('{}{:0.3f}'.format('Recall: ', recall))
        print('{}{:0.3f}'.format('Precision: ', precision))
        print('{}{:0.3f}'.format('Jaccard: ', 100 * jaccard))
        print('{}{:0.3f}'.format('RMSE_vol: ', rmse_vol))
        print('{}{:0.3f}'.format('Eff_3d: ', eff_3d))
        print('FN: ' + str(np.round(FN)) + ' FP: ' + str(np.round(FP)))
        print('{}{:0.3f}'.format('Num. matches: ', len(match_df)))

        print(f'Shift: {x_s:.2f},{y_s:.2f},{z_s:.2f}')

    perf_dict = {'recall': recall, 'precision': precision, 'jaccard': jaccard, 'rmse_vol': rmse_vol,
            'rmse_x': rmse_x, 'rmse_y': rmse_y,  'rmse_z': rmse_z, 'eff_3d': eff_3d, 'n_matches':len(match_df)}

    return perf_dict, match_df, [x_s,y_s,z_s]