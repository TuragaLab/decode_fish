# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/04_pointsource.ipynb (unless otherwise specified).

__all__ = ['PointProcessUniform', 'list_to_locations', 'get_phased_ints']

# Cell
from ..imports import *
from torch import distributions as D, Tensor
from torch.distributions import Distribution
from ..funcs.utils import *

# Cell
class PointProcessUniform(Distribution):
    """
    This class is part of the generative model and uses the probability local_rate to generate sample locations on the voxel grid.
    For each emitter we then sample x-,y- and z-offsets uniformly in the range [-0.5,0.5] to get continuous locations.
    Intensities are sampled from a gamma distribution torch.distirubtions.gamma(int_conc, int_rate) which is shifted by int_loc.
    Together with the microscope.scale and the PSF this results in the overall brightness of an emitter.

    Args:
        local_rate torch.tensor . shape(BS, C, H, W, D): Local rate
        int_conc=0., int_rate=1., int_loc (float): parameters of the intensity gamma distribution
        sim_iters (int): instead of sampling once from local_rate, we sample sim_iters times from local_rate/sim_iters.
            This results in the same average number of sampled emitters but allows us to sample multiple emitters within one voxel.

    """
    def __init__(self, int_conc=0., int_rate=1., int_loc=1., sim_iters: int = 5, n_channels=1, sim_z=True, slice_rec=True, codebook=None, int_option=1, code_weight=None, device='cuda'):

        assert sim_iters >= 1
        self.sim_iters = sim_iters
        self.int_conc = int_conc
        self.int_rate = int_rate
        self.int_loc = int_loc
        self.n_channels = n_channels
        self.sim_z = sim_z
        self.slice_rec = slice_rec
        self.int_option = int_option
        self.device = device

        self.codebook = torch.tensor(codebook).to(self.device) if codebook is not None else None

        if self.codebook is not None:
            self.n_channels = self.codebook.shape[-1]
            self.code_weight = code_weight.to(self.device) if code_weight is not None else torch.ones(len(self.codebook)).to(self.device)

    def sample(self, local_rate, from_code_book=True):

        res_ = [self._sample(local_rate.to(self.device)/self.sim_iters, from_code_book) for i in range(self.sim_iters)]
        locations = torch.cat([i[0] for i in res_], dim=0)
        x_offset = torch.cat([i[1] for i in res_], dim=0)
        y_offset = torch.cat([i[2] for i in res_], dim=0)
        z_offset = torch.cat([i[3] for i in res_], dim=0)
        intensities = torch.cat([i[4] for i in res_], dim=0)
        codes = torch.cat([i[6] for i in res_], dim=0) if from_code_book else None

        return list(locations.T), x_offset, y_offset, z_offset, intensities, res_[0][5], codes

    def _sample(self, local_rate, from_code_book):

        output_shape = list(local_rate.shape)
        local_rate = torch.clamp(local_rate,0.,1.)
        locations = D.Bernoulli(local_rate).sample()
        n_emitter = int(locations.sum().item())
        x_offset = D.Uniform(low=-0.5, high=0.5).sample(sample_shape=[n_emitter]).to(self.device)
        y_offset = D.Uniform(low=-0.5, high=0.5).sample(sample_shape=[n_emitter]).to(self.device)
        z_offset = D.Uniform(low=-0.5, high=0.5).sample(sample_shape=[n_emitter]).to(self.device)

        if self.slice_rec: # For slice rec we use larger range
            z_offset *= 2

#         intensities = torch.zeros([n_emitter, self.n_channels]).to(self.device)
        if self.int_option == 1:
            intensities = D.Gamma(self.int_conc, self.int_rate).sample(sample_shape=[n_emitter, self.n_channels]).to(self.device) + self.int_loc
        elif self.int_option == 2:
            intensities = D.Gamma(self.int_conc, self.int_rate).sample(sample_shape=[n_emitter, 1]).to(self.device) + self.int_loc
            intensities = intensities.repeat_interleave(self.n_channels, 1)
        elif self.int_option == 3:
            intensities = D.Gamma(self.int_conc, self.int_rate).sample(sample_shape=[n_emitter, 1]).to(self.device) + self.int_loc
            intensities = intensities.repeat_interleave(self.n_channels, 1)
            int_noise = D.Uniform(low=.7, high=1.5).sample(sample_shape=intensities.shape).to(self.device)
            intensities *= int_noise

        # If 2D data z-offset is 0
        if not self.sim_z:
            z_offset *= 0

        locations = locations.nonzero(as_tuple=False)

        if self.n_channels > 1:
            code_draw = None
            if from_code_book:
                code_draw = torch.multinomial(self.code_weight, num_samples=n_emitter, replacement=True)
                ch_draw = self.codebook[code_draw]

            else:

                m_draw = torch.multinomial(torch.ones([n_emitter,self.n_channels])/self.n_channels, 1, replacement=False)
                ch_draw = torch.zeros(intensities.shape).to(self.device, dtype=torch.float32)
                ch_draw.scatter_(index=m_draw.to(self.device), dim=1, value=1)

            intensities = intensities.to(self.device) * ch_draw.to(self.device)
            output_shape.insert(1, self.n_channels)

        return locations, x_offset, y_offset, z_offset, intensities, tuple(output_shape), code_draw


def list_to_locations(locations, output_shape):
    tmp =torch.zeros(output_shape, device=locations[0].device)
    coord = torch.stack(locations).T
    #incase you have multiple emitter present
    for i in coord: tmp[tuple(i)] += 1
    return tmp

# Cell
def get_phased_ints(ints, ch_cols, n_cols):

    col_inds = []
    for i in range(n_cols):
        col_inds.append(torch.where(torch.tensor(ch_cols)==i)[0].cuda()) # Get indices of the different colors
        col_inds[i][-1] = 0 # Set last index to 0 because it won't phase

    nonz_inds = ints.nonzero()
    phased_ints = torch.zeros_like(ints)
    phased_inds = torch.zeros_like(nonz_inds)

    # Loop over colors and each pos in the col_indices. Set the phased indices to the next value
    for c in range(n_cols):
        for n in range(len(col_inds[c]) - 1):
            idx = torch.where(nonz_inds[:,1] == col_inds[c][n])[0]
            phased_inds[idx] = torch.stack([nonz_inds[idx][:,0], torch.ones(len(idx), dtype=torch.int32).cuda() * (col_inds[c][n+1])], 1)

    phased_ints[tuple([phased_inds[:,0], phased_inds[:,1]])] += ints[tuple([nonz_inds[:,0], nonz_inds[:,1]])]
    phased_ints[:,0] = 0.

    return phased_ints