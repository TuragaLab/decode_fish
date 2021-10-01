# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/05_gmm_loss.ipynb (unless otherwise specified).

__all__ = ['ext_log_prob', 'PointProcessGaussian', 'get_sample_mask', 'get_true_labels', 'get_true_labels_mf_old',
           'get_true_labels_mf', 'grp_range', 'cum_count_per_group']

# Cell
from ..imports import *
from torch import distributions as D, Tensor
from torch.distributions import Distribution
from torch.distributions.utils import _sum_rightmost
from einops import rearrange
import torch.tensor as T
import torch.nn.functional as F

# Cell
def ext_log_prob(mix, x):

    x = mix._pad(x)
    log_prob_x = mix.component_distribution.base_dist.log_prob(x)
    log_prob_x = _sum_rightmost(log_prob_x, 1)

    log_mix_prob = torch.log_softmax(mix.mixture_distribution.logits, dim=-1)
    return log_prob_x, log_mix_prob

# Cell
class PointProcessGaussian(Distribution):
    def __init__(self, logits, xyzi_mu, xyzi_sigma, int_logits=None, **kwargs):
        """ Defines our loss function. Given logits, xyzi_mu and xyzi_sigma

        The count loss first constructs a Gaussian approximation to the predicted number of emitters by summing the mean and the variance of the Bernoulli detection probability map,
        and then maximizes the probability of the true number of emitters under this distribution.
        The localization loss models the distribution of sub-pixel localizations with a coordinate-wise independent Gaussian probability distribution  with a 3D standard deviation.
        For imprecise localizations, this probability is maximized for large sigmas, for precise localizations for small sigmas.
        The distribution of all localizations over the entire image is approximated as a weighted average of individual localization distributions, where the weights correspond to the probability of detection.

        Args:
            logits: shape (B,1,D,H,W)
            xyzi_mu: shape (B,4,D,H,W)
            xyzi_sigma: shape (B,4,D,H,W)
        """
        self.logits = logits.cuda()
        self.xyzi_mu = xyzi_mu.cuda()
        self.xyzi_sigma = xyzi_sigma.cuda()

    def log_prob(self, locations, x_offset, y_offset, z_offset, intensities, codes, n_bits, channels):

        gauss_dim = 3 + 1
        batch_size = self.logits.shape[0]

        xyzi, gt_codes, s_mask = get_true_labels_mf(batch_size, n_bits, channels, locations, x_offset, y_offset, z_offset, intensities, codes.cuda())

        P = torch.sigmoid(self.logits)
        count_mean = P.sum(dim=[2, 3, 4]).squeeze(-1)
        count_var = (P - P ** 2).sum(dim=[2, 3, 4]).squeeze(-1)
        count_dist = D.Normal(count_mean, torch.sqrt(count_var))

        counts = torch.zeros(count_mean.shape).cuda()
        unique_col = [gtc.unique(return_counts=True) for gtc in gt_codes]
        for i, ind_c in enumerate(unique_col):
            inds, c = ind_c
            counts[i, inds[inds>=0]] = c[inds>=0].type(torch.cuda.FloatTensor)

        count_prob =  count_dist.log_prob(counts).sum(-1) # * counts

        mixture_probs = P / P.sum(dim=[2, 3, 4], keepdim=True)

        pix_inds = torch.nonzero(P[:,:1],as_tuple=True)

        xyzi_mu = self.xyzi_mu[pix_inds[0],:,pix_inds[2],pix_inds[3],pix_inds[4]]
        xyzi_mu[:,:3] += torch.stack([pix_inds[4],pix_inds[3],pix_inds[2]], 1) + 0.5
        xyzi_mu = xyzi_mu.reshape(batch_size,-1,gauss_dim)
        xyzi_sig = self.xyzi_sigma[pix_inds[0],:,pix_inds[2],pix_inds[3],pix_inds[4]].reshape(batch_size,-1,gauss_dim)


        mix = D.Categorical(mixture_probs[torch.nonzero(P,as_tuple=True)].reshape(batch_size, 140, -1))
        mix_logits = mix.logits


        dist_normal_xyzi = D.Independent(D.Normal(xyzi_mu, xyzi_sig + 0.00001), 1)

        xyzi_inp = xyzi.transpose(0, 1)[:,:,None,:]          # reshape for log_prob()
        log_norm_prob_xyzi = dist_normal_xyzi.base_dist.log_prob(xyzi_inp) # N_gt * batch_size * n_pixel * 3

        log_cat_prob = torch.log_softmax(mix_logits, -1) # + torch.log(counts+1e-6)[:, None]       # normalized (sum to 1 over pixels) log probs for the categorical dist. of the GMM. batch_size * n_pixels

        gt_codes[gt_codes<0] = 0
        log_norm_prob_xyzi = _sum_rightmost(log_norm_prob_xyzi, 1)         # N_gt * batch_size * n_pixel
        total_prob = torch.logsumexp(log_norm_prob_xyzi + torch.gather(log_cat_prob, 1, gt_codes[...,None].expand(-1,-1,log_cat_prob.shape[-1])).transpose(0,1),-1).transpose(0, 1)

        total_prob = (total_prob * s_mask).sum(-1)  # s_mask: batch_size * N_gt. Binary mask to remove entries in all samples that have less then N_gt GT emitters.

        return count_prob, total_prob

    def log_prob_old(self, locations, x_offset, y_offset, z_offset, intensities, n_bits, channels, min_int_sig, int_fac=1):
        """ Creates the distributions for the count and localization loss and evaluates the log probability for the given set of localizations under those distriubtions.

            Args:
                locations: tuple with voxel locations of inferred emitters
                x_offset, y_offset,z_offset: continuous within pixel offsets. Has lenght of number of emitters in the whole batch.
                intensties: brightness of emitters. Has lenght of number of emitters in the whole batch.

            Returns:
                count_prob: count loss. Has langth of batch_size
                spatial_prob: localizations loss. Has langth of batch_size
        """

        gauss_dim = 3 + channels
        batch_size = self.logits.shape[0]
        if channels == 1:
            xyzi, s_mask = get_true_labels(batch_size, locations, x_offset, y_offset, z_offset, intensities)
        else:
            xyzi, s_mask = get_true_labels_mf(batch_size, n_bits, channels,
                                              locations, x_offset, y_offset, z_offset, intensities)
        counts = s_mask.sum(-1)

        P = torch.sigmoid(self.logits)
        count_mean = P.sum(dim=[2, 3, 4]).squeeze(-1)
        count_var = (P - P ** 2).sum(dim=[2, 3, 4]).squeeze(-1)
        count_dist = D.Normal(count_mean, torch.sqrt(count_var))

        count_prob =  count_dist.log_prob(counts) # * counts

        mixture_probs = P / P.sum(dim=[2, 3, 4], keepdim=True)

        pix_inds = torch.nonzero(P,as_tuple=True)

        xyzi_mu = self.xyzi_mu[pix_inds[0],:,pix_inds[2],pix_inds[3],pix_inds[4]]
        xyzi_mu[:,:3] += torch.stack([pix_inds[4],pix_inds[3],pix_inds[2]], 1) + 0.5
        xyzi_mu = xyzi_mu.reshape(batch_size,-1,gauss_dim)
        xyzi_sig = self.xyzi_sigma[pix_inds[0],:,pix_inds[2],pix_inds[3],pix_inds[4]].reshape(batch_size,-1,gauss_dim)

#         print(xyzi_mu.shape, xyzi.shape)

        mix = D.Categorical(mixture_probs[pix_inds].reshape(batch_size,-1))

        '''base 19 dim'''
        xyzi_sig[:,:,3:] = xyzi_sig[:,:,3:] + min_int_sig
        comp = D.Independent(D.Normal(xyzi_mu, xyzi_sig + 0.00001), 1)
        spatial_gmm = D.MixtureSameFamily(mix, comp)
        spatial_prob = spatial_gmm.log_prob(xyzi.transpose(0, 1)).transpose(0,1)
        total_prob = (spatial_prob * s_mask).sum(-1)
        '''split int'''
#         xyz_sl = np.s_[:,:,:3]
#         int_sl = np.s_[:,:,3:]

#         xyzi_sig[int_sl] = xyzi_sig[int_sl] + min_int_sig

#         comp_xyz = D.Independent(D.Normal(xyzi_mu[xyz_sl], xyzi_sig[xyz_sl] + 0.00001), 1)
#         comp_int = D.Independent(D.Normal(xyzi_mu[int_sl], xyzi_sig[int_sl] + 0.00001), 1)

#         spatial_gmm = D.MixtureSameFamily(mix, comp_xyz)
#         int_gmm = D.MixtureSameFamily(mix, comp_int)

#         spatial_prob, log_mix_prob = ext_log_prob(spatial_gmm, xyzi[xyz_sl].transpose(0, 1))
#         int_prob, _                = ext_log_prob(int_gmm, xyzi[int_sl].transpose(0, 1))

#         total_prob = torch.logsumexp(spatial_prob + int_prob + log_mix_prob,-1).transpose(0, 1)
#         total_prob = (total_prob * s_mask).sum(-1)

        return count_prob, total_prob

def get_sample_mask(bs, locations):

    counts_ = torch.unique(locations[0], return_counts=True)[1]
    batch_loc = torch.unique(locations[0])

    counts = torch.cuda.LongTensor(bs).fill_(0)
    counts[batch_loc] = counts_

    max_counts = counts.max()
    if max_counts==0: max_counts = 1 #if all 0 will return empty matrix of correct size
    s_arr = cum_count_per_group(locations[0])
    s_mask   = torch.cuda.FloatTensor(bs,max_counts).fill_(0)
    s_mask[locations[0],s_arr] = 1

    return s_mask, s_arr

def get_true_labels(bs, locations, x_os, y_os, z_os, *args):

    s_mask, s_arr = get_sample_mask(bs, locations)
    max_counts = s_mask.shape[1]

    x =  x_os + locations[4].type(torch.cuda.FloatTensor) + 0.5
    y =  y_os + locations[3].type(torch.cuda.FloatTensor) + 0.5
    z =  z_os + locations[2].type(torch.cuda.FloatTensor) + 0.5

    gt_vars = torch.stack([x, y, z] + [item for item in args], dim=1)
    gt_list = torch.cuda.FloatTensor(bs,max_counts,gt_vars.shape[1]).fill_(0)

    gt_list[locations[0],s_arr] = gt_vars
    return gt_list, s_mask

def get_true_labels_mf_old(bs, n_bits, channels, locations, x_os, y_os, z_os, int_ch, codes):

    # Added for phasing support. Rarely crashes here: gt_codes[xyz_locs[0],s_arr] = codes. Because of random identical x_os samples?

    b_inds = torch.cat([torch.tensor([0], device=x_os.device),((x_os[1:] - x_os[:-1]).nonzero() + 1)[:,0],
                        torch.tensor([len(x_os)], device=x_os.device)])
    n_gt = len(b_inds) - 1

    xyz_locs = [l[b_inds[:-1]] for l in locations]
    x_os = x_os[b_inds[:-1]]
    y_os = y_os[b_inds[:-1]]
    z_os = z_os[b_inds[:-1]]

    s_mask, s_arr = get_sample_mask(bs, xyz_locs)
    max_counts = s_mask.shape[1]

    x =  x_os + xyz_locs[4].type(torch.cuda.FloatTensor) + 0.5
    y =  y_os + xyz_locs[3].type(torch.cuda.FloatTensor) + 0.5
    z =  z_os + xyz_locs[2].type(torch.cuda.FloatTensor) + 0.5

    loc_idx = []
    for i in range(n_gt):
        loc_idx += [i] * (b_inds[i+1] - b_inds[i])

    intensity = torch.zeros([n_gt, channels]).to(x.device)
    intensity[loc_idx, locations[1]] = int_ch

    intensity = intensity.sum(-1)

    gt_vars = torch.stack([x, y, z, intensity], dim=1)
    gt_list = torch.cuda.FloatTensor(bs,max_counts,gt_vars.shape[1]).fill_(0)
    gt_list[xyz_locs[0],s_arr] = gt_vars

    gt_codes = torch.cuda.LongTensor(bs,max_counts).fill_(0) - 1
    gt_codes[xyz_locs[0],s_arr] = codes

    return gt_list, gt_codes, s_mask

def get_true_labels_mf(bs, n_bits, channels, locations, x_os, y_os, z_os, int_ch, codes):

    n_gt = int(len(x_os) / n_bits)

    xyz_locs = [l[::n_bits] for l in locations]
    x_os = x_os[::n_bits]
    y_os = y_os[::n_bits]
    z_os = z_os[::n_bits]

    s_mask, s_arr = get_sample_mask(bs, xyz_locs)
    max_counts = s_mask.shape[1]

    x =  x_os + xyz_locs[4].type(torch.cuda.FloatTensor) + 0.5
    y =  y_os + xyz_locs[3].type(torch.cuda.FloatTensor) + 0.5
    z =  z_os + xyz_locs[2].type(torch.cuda.FloatTensor) + 0.5

    loc_idx = torch.arange(n_gt).repeat_interleave(4)

    intensity = torch.zeros([n_gt, channels]).to(x.device)
    intensity[loc_idx, locations[1]] = int_ch

    intensity = intensity.sum(-1)

    gt_vars = torch.stack([x, y, z, intensity], dim=1)
    gt_list = torch.cuda.FloatTensor(bs,max_counts,gt_vars.shape[1]).fill_(0)
    gt_list[xyz_locs[0],s_arr] = gt_vars

    gt_codes = torch.cuda.LongTensor(bs,max_counts).fill_(0) - 1
    gt_codes[xyz_locs[0],s_arr] = codes

    return gt_list, gt_codes, s_mask

def grp_range(counts: torch.Tensor):
    assert counts.dim() == 1

    idx = counts.cumsum(0)
    id_arr = torch.ones(idx[-1], dtype=int, device=counts.device)
    id_arr[0] = 0
    id_arr[idx[:-1]] = -counts[:-1] + 1
    return id_arr.cumsum(0)

def cum_count_per_group(arr):
    """
    Helper function that returns the cumulative sum per group.
    Example:
        [0, 0, 0, 1, 2, 2, 0] --> [0, 1, 2, 0, 0, 1, 3]
    """

    if arr.numel() == 0:
        return arr

    _, cnt = torch.unique(arr, return_counts=True)
    return grp_range(cnt)[np.argsort(np.argsort(arr.cpu().numpy(), kind='mergesort'), kind='mergesort')]