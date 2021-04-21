# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/04_pointsource.ipynb (unless otherwise specified).

__all__ = ['PointProcessUniform', 'list_to_locations']

# Cell
from ..imports import *
from torch import distributions as D, Tensor
from torch.distributions import Distribution

# Cell
class PointProcessUniform(Distribution):
    """
    This class is part of the generative model and uses probability local_rate to generate locations `locations`  `x`, `y`, `z` offsets and `intensities` intensity of emitters. local_rate  should be `torch.tensor` scaled from 0.001 to 1), which is used by `_sample_bin` to generate `0` and `1` . `0` means that we don't have an emitter at a given pixel, and 1 means emitters is present. This map is used to generate offset in `x`, `y`, `z`, and intensities, which tells how bright is emitter or, in some cases, how many emitters are bound to given molecules.
    Args:
        local_rate (BS, C, H, W, D): Local rate
        min_int (int): minimum intensity of emitters
        bg(bool): if returns sampled backround

    """
    def __init__(self, local_rate: torch.tensor, sim_iters: int = 5):

        self.local_rate = local_rate
        self.device = self._get_device(self.local_rate)
        self.sim_iters = sim_iters

    def sample(self):

        res_ = [self._sample(self.local_rate/self.sim_iters) for i in range(self.sim_iters)]
        locations = torch.cat([i[0] for i in res_], dim=0)
        x_offset = torch.cat([i[1] for i in res_], dim=0)
        y_offset = torch.cat([i[2] for i in res_], dim=0)
        z_offset = torch.cat([i[3] for i in res_], dim=0)
        intensities = torch.cat([i[4] for i in res_], dim=0)

        return tuple(locations.T), x_offset, y_offset, z_offset, intensities, res_[0][5]

    def _sample(self, local_rate):

        local_rate = torch.clamp_max(local_rate, 1.)
        locations = D.Bernoulli(local_rate).sample()
        n_emitter = int(locations.sum().item())
        zero_point_five = torch.tensor(0.5, device=self.device)
        x_offset = D.Uniform(low=0 - zero_point_five, high=0 + zero_point_five).sample(sample_shape=[n_emitter])
        y_offset = D.Uniform(low=0 - zero_point_five, high=0 + zero_point_five).sample(sample_shape=[n_emitter])
        z_offset = D.Uniform(low=0 - zero_point_five, high=0 + zero_point_five).sample(sample_shape=[n_emitter])
        intensities = D.Normal(0, 1).sample(sample_shape=[n_emitter]).to(self.device)

        output_shape = tuple(locations.shape)
        locations = locations.nonzero(as_tuple=False)
        return locations, x_offset, y_offset, z_offset, intensities, output_shape


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