# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/05_gmm_loss.ipynb (unless otherwise specified).

__all__ = ['PointProcessGaussian', 'img_to_coord', 'get_true_labels']

# Cell
from ..imports import *
from torch import distributions as D, Tensor
from torch.distributions import Distribution

# Cell
class PointProcessGaussian(Distribution):
    def __init__(self, logits: torch.tensor, xyzi_mu: torch.tensor,
                 xyzi_sigma: torch.tensor, **kwargs):
        "logits: BS, C, H, W, D"
        self.logits = logits
        self.xyzi_mu = xyzi_mu
        self.xyzi_sigma = xyzi_sigma

    def sample(self, N:int =1):
        if N ==1:
            return self._sample()

        res_ = [self._sample() for i in range(N)]
        locations   =  torch.cat([i[0] for i in res_], dim=1)
        x_offset    =  torch.cat([i[1] for i in res_], dim=1)
        y_offset    =  torch.cat([i[2] for i in res_], dim=1)
        z_offset    =  torch.cat([i[3] for i in res_], dim=1)
        intensities =  torch.cat([i[4] for i in res_], dim=1)
        return locations, x_offset, y_offset, z_offset, intensities



    def _sample(self):
        #ask about this sampling
        locations = D.Bernoulli(logits=self.logits).sample()
        xyzi = D.Independent(D.Normal(self.xyzi_mu, self.xyzi_sigma),
                             1).sample()
        x_offset, y_offset, z_offset, intensities = (i.unsqueeze(1) for i in
                                  torch.unbind(xyzi, dim=1))

        output_shape = tuple(locations.shape)
        locations = locations.nonzero(as_tuple=True)
        x_offset  = x_offset[locations]
        y_offset  = y_offset[locations]
        z_offset  = z_offset[locations]
        intensities = intensities[locations]
        return locations, x_offset, y_offset, z_offset, intensities, output_shape


    def log_prob(self, locations_3d, x_offset_3d, y_offset_3d, z_offset_3d, intensities_3d, p_threshold):

        batch_size = self.logits.shape[0]
        xyzi, counts, s_mask = get_true_labels(batch_size, locations_3d, x_offset_3d, y_offset_3d, z_offset_3d, intensities_3d )
        x_mu, y_mu, z_mu, i_mu = (i.unsqueeze(1) for i in
                                  torch.unbind(self.xyzi_mu, dim=1))
        x_si, y_si, z_si, i_si = (i.unsqueeze(1) for i in
                                  torch.unbind(self.xyzi_sigma, dim=1))

        P = torch.sigmoid(self.logits) + 0.00001
        count_mean = P.sum(dim=[2, 3, 4]).squeeze(-1)
        count_var = (P - P ** 2).sum(dim=[2, 3, 4]).squeeze(-1)  #avoid situation where we have perfect match
        count_dist = D.Normal(count_mean, torch.sqrt(count_var))
        count_prob = count_dist.log_prob(counts)
        mixture_probs = P / P.sum(dim=[2, 3, 4], keepdim=True)

        xyz_mu_list, _, _, i_mu_list, x_sigma_list, y_sigma_list, z_sigma_list, i_sigma_list, mixture_probs_l = img_to_coord(
            batch_size, p_threshold, P, x_mu, y_mu, z_mu, i_mu, x_si, y_si, z_si, i_si, mixture_probs)
        xyzi_mu = torch.cat((xyz_mu_list, i_mu_list), dim=-1)
        xyzi_sigma = torch.cat((x_sigma_list, y_sigma_list, z_sigma_list, i_sigma_list), dim=-1) #to avoind NAN
        mix = D.Categorical(mixture_probs_l.squeeze(-1))
        comp = D.Independent(D.Normal(xyzi_mu, xyzi_sigma + 0.00001), 1)
        spatial_gmm = D.MixtureSameFamily(mix, comp)
        spatial_prob = spatial_gmm.log_prob(xyzi.transpose(0, 1)).transpose(0,1)
        spatial_prob = (spatial_prob * s_mask).sum(-1)
        log_prob = count_prob + spatial_prob
        return log_prob

def img_to_coord(bs, p_quantile, locations, x_os, y_os, z_os, *args):
    """
    Given `locations'  will extract value of x_os, y_os, z_os where probability is more than 0.
    also generates counts of location and returns mask

    Args:
        locations: Tuple(BS, Frames, D, H, W)
        x_offsets: (N_emitters,)
        y_offsets: (N_emitters,)
        z_offsets: (N_emitters,)
        intensities: (N_emitters,)

    """

    if type(locations) == torch.Tensor:
        #this is done for model outputs since they are volumes
        if p_quantile > 0:
            threshold = torch.quantile(locations.flatten(1), p_quantile, dim=1)
            locations = torch.nonzero(locations>threshold[:,None,None,None,None],as_tuple=True)
        else:
            locations = torch.nonzero(locations,as_tuple=True)
        x_os      = x_os[locations]
        y_os      = y_os[locations]
        z_os      = z_os[locations]
        a = [item[locations].unsqueeze(-1) for item in args]
    else:
        a = [item.unsqueeze(-1) for item in args]

    # D == z 2, H == y 3, W == x 4
    x =  x_os + locations[4].type(torch.cuda.FloatTensor) + 0.5
    y =  y_os + locations[3].type(torch.cuda.FloatTensor) + 0.5
    z =  z_os + locations[2].type(torch.cuda.FloatTensor) + 0.5

    xyz =  torch.stack((x, y, z), dim=1)

    #to match where in batch we have a counts (if there is no count at this
    #position we will get 0 in tensor
    counts_ = torch.unique_consecutive(locations[0], return_counts=True)[1]
    bsz_loc = torch.unique(locations[0])
    #getting batch size
#     bs = (locations[0].max() + 1).item()
    counts = torch.cuda.LongTensor(bs).fill_(0)
    counts[bsz_loc] = counts_

    max_counts    = counts.max()
    if max_counts==0: max_counts = 1 #if all 0 will return empty matrix of correct size
    xyz_list = torch.cuda.FloatTensor(bs,max_counts,3).fill_(0)
    i_list   = [torch.cuda.FloatTensor(bs,max_counts,1).fill_(0) for i in range(len(a))]
    s_arr    = torch.cat([torch.arange(c) for c in counts], dim = 0)
    s_mask   = torch.cuda.FloatTensor(bs,max_counts).fill_(0)
    s_mask[locations[0],s_arr] = 1
    xyz_list[locations[0],s_arr] = xyz
    for i,k in zip(i_list, a): i[locations[0],s_arr] = k
    return (xyz_list, counts, s_mask) + tuple(i_list)

def get_true_labels(bs, locations, x_os, y_os, z_os, ints):
    xyz_list, counts_true, s_mask, i_1 = img_to_coord(bs, 0., locations, x_os, y_os, z_os, ints)
    xyzi_true = torch.cat((xyz_list, i_1), dim=-1)
    return xyzi_true, counts_true, s_mask