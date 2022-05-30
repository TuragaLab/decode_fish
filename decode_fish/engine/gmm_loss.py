# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/05_gmm_loss.ipynb (unless otherwise specified).

__all__ = ['PointProcessGaussian', 'get_sample_mask', 'get_true_labels_mf', 'grp_range', 'cum_count_per_group']

# Cell
from ..imports import *
import torch
from torch import distributions as D, Tensor
from torch.distributions import Distribution
from torch.distributions.utils import _sum_rightmost
from einops import rearrange
# import torch.tensor as T
import torch.nn.functional as F

# Cell
class PointProcessGaussian(Distribution):
    def __init__(self, logits, xyzi_mu, xyzi_sigma, int_logits=None, **kwargs):
        """ Defines our loss function. Given logits, xyzi_mu and xyzi_sigma

        The count loss first constructs a Gaussian approximation to the predicted number of emitters by summing the mean and the variance of the Bernoulli detection probability map,
        and then maximizes the probability of the true number of emitters under this distribution.
        The localization loss models the distribution of sub-pixel localizations with a coordinate-wise independent Gaussian probability distribution  with a 3D standard deviation.
        For imprecise localizations, this probability is maximized for large sigmas, for precise localizations for small sigmas.
        The distribution of all localizations over the entire image is approximated as a weighted average of individual localization distributions, where the weights correspond to the probability of detection.

        Args (Network outputs):
            logits: shape (B,1,D,H,W)
            xyzi_mu: shape (B,4,D,H,W)
            xyzi_sigma: shape (B,4,D,H,W)
        """
        self.logits = logits.cuda()
        self.xyzi_mu = xyzi_mu.cuda()
        self.xyzi_sigma = xyzi_sigma.cuda()

    def log_prob(self, locations, x_offset, y_offset, z_offset, intensities, codes, n_bits, channels, loss_option=0, count_mult=0, cat_logits=0, slice_rec=False, int_inf='sum', z_sig_fac=0.5):

        gauss_dim = 3
        if int_inf == 'sum': gauss_dim += 1
        if int_inf == 'per_bit': gauss_dim += n_bits
        if int_inf == 'per_channel': gauss_dim += channels

        batch_size = self.logits.shape[0]
        n_codes = self.logits.shape[1]

        xyzi, gt_codes, s_mask = get_true_labels_mf(batch_size, locations, x_offset, y_offset, z_offset, intensities, codes.cuda(), slice_rec, z_sig_fac, int_inf)

        P = torch.sigmoid(self.logits)

        if loss_option == 0:
            # Calculate count loss individually for each code
            # Performs worse on sim. data. check again
            count_mean = P.sum(dim=[2, 3, 4]).squeeze(-1)
            count_var = (P - P ** 2).sum(dim=[2, 3, 4]).squeeze(-1)
            count_dist = D.Normal(count_mean, torch.sqrt(count_var))

            counts = torch.zeros(count_mean.shape).cuda()
            unique_col = [gtc.unique(return_counts=True) for gtc in gt_codes]
            for i, ind_c in enumerate(unique_col):
                inds, c = ind_c
                counts[i, inds[inds>=0]] = c[inds>=0].type(torch.cuda.FloatTensor)

            count_prob =  count_dist.log_prob(counts)
#             if count_mult:
#                 count_prob = count_prob * counts

            count_prob = count_prob.sum(-1)

        if loss_option == 1:
            # Calculate count loss by summing over all code channels
            count_mean = P.sum(dim=[1, 2, 3, 4]).squeeze(-1)
            count_var = (P - P ** 2).sum(dim=[1, 2, 3, 4]).squeeze(-1)
            count_dist = D.Normal(count_mean, torch.sqrt(count_var))

            counts = s_mask.sum(-1)
            count_prob =  count_dist.log_prob(counts)
#             if count_mult:
#                 count_prob = count_prob * counts

        pix_inds = torch.nonzero(P[:,:1],as_tuple=True)

        xyzi_mu = self.xyzi_mu[pix_inds[0],:,pix_inds[2],pix_inds[3],pix_inds[4]]
        if slice_rec: xyzi_mu[:,2] = z_sig_fac*xyzi_mu[:,2]
        # We squish the network output range in z to -0.5:0.5 beause for slice rec the true pixel inds are unqiue (i.e. cant point to the same point from different pixels)
        # Not needed if we turn slices to batches
        xyzi_mu[:,:3] += torch.stack([pix_inds[4],pix_inds[3],pix_inds[2]], 1)
        xyzi_mu = xyzi_mu.reshape(batch_size,-1,gauss_dim)
        xyzi_sig = self.xyzi_sigma[pix_inds[0],:,pix_inds[2],pix_inds[3],pix_inds[4]].reshape(batch_size,-1,gauss_dim)

        if cat_logits:
            mixture_probs = P / P.sum(dim=[2, 3, 4], keepdim=True)
            mix = D.Categorical(mixture_probs[torch.nonzero(P,as_tuple=True)].reshape(batch_size, n_codes, -1))
            mix_logits = mix.logits
        else:
            mix_logits = self.logits[torch.nonzero(P,as_tuple=True)].reshape(batch_size, n_codes, -1)

        xyzi_inp = xyzi.transpose(0, 1)[:,:,None,:]          # reshape for log_prob()

#         dist_normal_xyzi = D.Independent(D.Normal(xyzi_mu, xyzi_sig + 0.00001), 1)

#         log_norm_prob_xyzi = dist_normal_xyzi.base_dist.log_prob(xyzi_inp) # N_gt * batch_size * n_pixel * (3 + n_int_ch)
#         if int_inf == 'per_channel':
#             log_norm_prob_xyzi[...,3:] *= xyzi_inp[...,3:].ne(0)
#         log_norm_prob_xyzi = _sum_rightmost(log_norm_prob_xyzi, 1)         # N_gt * batch_size * n_pixel

        # Iterative loss calc. 15% slower but 3 times less memory!

        log_norm_prob_xyzi = 0
        for i in range(gauss_dim):
            dist_normal_xyzi = D.Independent(D.Normal(xyzi_mu[...,i], xyzi_sig[...,i] + 0.00001), 1)
#             log_norm_prob_xyzi += dist_normal_xyzi.base_dist.log_prob(xyzi_inp[...,i]) * xyzi_inp[...,i].ne(0)

            log_norm_prob_xyzi += dist_normal_xyzi.base_dist.log_prob(xyzi_inp[...,i]) * (xyzi_inp[...,i].ne(0) + count_mult * xyzi_inp[...,i].eq(0))

        if loss_option == 'bugged?':
            log_cat_prob = torch.log_softmax(mix_logits, -1) # + torch.log(counts+1e-6)[:, None]       # normalized (sum to 1 over pixels) log probs for the categorical dist. of the GMM. batch_size * n_pixels
        else:
            log_cat_prob = torch.log_softmax(mix_logits.view(batch_size, -1), -1).view(mix_logits.shape)

        gt_codes[gt_codes<0] = 0
        total_prob = torch.logsumexp(log_norm_prob_xyzi + torch.gather(log_cat_prob, 1, gt_codes[...,None].expand(-1,-1,log_cat_prob.shape[-1])).transpose(0,1),-1).transpose(0, 1)

        total_prob = (total_prob * s_mask).sum(-1)  # s_mask: batch_size * N_gt. Binary mask to remove entries in all samples that have less then N_gt GT emitters.

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

def get_true_labels_mf(bs, locations, x_os, y_os, z_os, int_ch, codes, slice_rec, z_sig_fac, int_inf='sum'):

    n_gt = len(x_os)

    s_mask, s_arr = get_sample_mask(bs, locations)
    max_counts = s_mask.shape[1]

    x =  x_os + locations[-1].type(torch.cuda.FloatTensor)
    y =  y_os + locations[-2].type(torch.cuda.FloatTensor)
    if slice_rec: z_os = z_os * z_sig_fac
    z =  z_os + locations[-3].type(torch.cuda.FloatTensor)

    loc_idx = torch.arange(n_gt).repeat_interleave(4)
    if int_inf == 'sum':
        intensity = int_ch.sum(-1)[:,None]
    if int_inf == 'per_bit':
        intensity = int_ch[int_ch.nonzero(as_tuple=True)].reshape([int_ch.shape[0],-1])
    if int_inf == 'per_channel':
        intensity = int_ch

    gt_vars = torch.cat([x[:,None], y[:,None], z[:,None], intensity], dim=1)
    gt_list = torch.cuda.FloatTensor(bs,max_counts,gt_vars.shape[1]).fill_(0)
    gt_list[locations[0],s_arr] = gt_vars

    gt_codes = torch.cuda.LongTensor(bs,max_counts).fill_(0) - 1
    gt_codes[locations[0],s_arr] = codes

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