# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/04_pointsource.ipynb (unless otherwise specified).

__all__ = ['PointProcessUniform', 'list_to_locations']

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
    def __init__(self, local_rate: torch.tensor, int_conc=0., int_rate=1., int_loc=1., sim_iters: int = 5, channels=1, n_bits=1, sim_z=True, codebook=None, phase_fac=0.2, int_option=1):

        assert sim_iters >= 1
        self.local_rate = local_rate
        self.device = self._get_device(self.local_rate)
        self.sim_iters = sim_iters
        self.int_conc = int_conc
        self.int_rate = int_rate
        self.int_loc = int_loc
        self.channels = channels
        self.n_bits = n_bits
        self.sim_z=sim_z
        self.codebook=codebook
        self.phase_fac=phase_fac
        self.int_option = int_option

    def sample(self, from_code_book=False, phasing=False):

        res_ = [self._sample(self.local_rate/self.sim_iters, from_code_book, phasing) for i in range(self.sim_iters)]
        locations = torch.cat([i[0] for i in res_], dim=0)
        x_offset = torch.cat([i[1] for i in res_], dim=0)
        y_offset = torch.cat([i[2] for i in res_], dim=0)
        z_offset = torch.cat([i[3] for i in res_], dim=0)
        intensities = torch.cat([i[4] for i in res_], dim=0)
        codes = torch.cat([i[6] for i in res_], dim=0)

        return list(locations.T), x_offset, y_offset, z_offset, intensities, res_[0][5], codes

    def _sample(self, local_rate, from_code_book, phasing):

        output_shape = list(local_rate.shape)
        local_rate = torch.clamp(local_rate,0.,1.)
        locations = D.Bernoulli(local_rate).sample()
        n_emitter = int(locations.sum().item())
        x_offset = D.Uniform(low=-0.5, high=0.5).sample(sample_shape=[n_emitter]).to(self.device)
        y_offset = D.Uniform(low=-0.5, high=0.5).sample(sample_shape=[n_emitter]).to(self.device)
        z_offset = D.Uniform(low=-0.5, high=0.5).sample(sample_shape=[n_emitter]).to(self.device)
        if self.int_option == 1:
            intensities = D.Gamma(self.int_conc, self.int_rate).sample(sample_shape=[n_emitter*self.n_bits]).to(self.device) + self.int_loc
        elif self.int_option == 2:
            intensities = D.Gamma(self.int_conc, self.int_rate).sample(sample_shape=[n_emitter]).to(self.device) + self.int_loc
            intensities = intensities.repeat_interleave(self.n_bits, 0)
        elif self.int_option == 3:
            intensities = D.Gamma(self.int_conc, self.int_rate).sample(sample_shape=[n_emitter]).to(self.device) + self.int_loc
            intensities = intensities.repeat_interleave(self.n_bits, 0)
            int_noise = D.Uniform(low=.7, high=1.5).sample(sample_shape=[n_emitter*self.n_bits]).to(self.device)
            intensities *= int_noise

        # If 2D data z-offset is 0
        if not self.sim_z:
            z_offset *= 0
        else:
            # If 2D data and we simulate z we use whole tanh range
            if output_shape[-3] == 1:
                 z_offset *= 2.

        locations = locations.nonzero(as_tuple=False)

        if self.channels > 1:
            code_draw = None
            if from_code_book:
                code_draw = torch.randint(0, len(self.codebook),size=[n_emitter])
                ch_draw = self.codebook[code_draw]
            else:
                ch_draw = torch.multinomial(torch.ones([n_emitter,self.channels])/self.channels, self.n_bits, replacement=False)
            locations = locations.repeat_interleave(self.n_bits, 0)
            locations[:, 1] = ch_draw.reshape(-1)

            # Exact positions are shared, but not intensities. Problems due to drift?
            x_offset = x_offset.repeat_interleave(self.n_bits, 0)
            y_offset = y_offset.repeat_interleave(self.n_bits, 0)
            z_offset = z_offset.repeat_interleave(self.n_bits, 0)

            output_shape[1] = self.channels

            if phasing:
                locations = locations.repeat_interleave(2, 0)
                locations[1::2,1] = locations[1::2,1] + 1
                x_offset = x_offset.repeat_interleave(2, 0)
                y_offset = y_offset.repeat_interleave(2, 0)
                z_offset = z_offset.repeat_interleave(2, 0)
                intensities = intensities.repeat_interleave(2, 0)

                phase_facs = torch.rand(size=intensities[1::2].shape, device=intensities.device) * self.phase_fac
                intensities[1::2] = intensities[1::2]*phase_facs

                inds = [locations[:,1] < self.channels][0]
                x_offset, y_offset, z_offset = x_offset[inds], y_offset[inds], z_offset[inds]
                intensities = intensities[inds]
                locations = locations[inds]

        return locations, x_offset, y_offset, z_offset, intensities, tuple(output_shape), code_draw


    def log_prob(self, locations, x_offset=None, y_offset=None, z_offset=None, intensities=None, output_shape=None):
        locations = list_to_locations(locations, output_shape)
        log_prob = D.Bernoulli(self.local_rate).log_prob(locations)
        return log_prob

    @staticmethod
    def _get_device(x):
        return getattr(x, 'device')


def list_to_locations(locations, output_shape):
    tmp =torch.zeros(output_shape, device=locations[0].device)
    coord = torch.stack(locations).T
    #incase you have multiple emitter present
    for i in coord: tmp[tuple(i)] += 1
    return tmp