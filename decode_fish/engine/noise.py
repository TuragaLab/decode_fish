# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/03_noise.ipynb (unless otherwise specified).

__all__ = ['sCMOS']

# Cell
from ..imports import *
from torch import nn
from torch import distributions as D
from ..funcs.utils import *
from ..funcs.plotting import *
import scipy.stats as stats

# Cell
class sCMOS(nn.Module):
    """
    Generates sCMOS noise distribution which can be used for sampling and
    calculating log probabilites.

    Theta can be learned (but no the baseline)

    Args:
        theta (float): 1/theta gives the rate for torch.distributions.gamma
        baseline (float): baseline

    """
    def __init__(self, theta = 3., baseline = 0., channels = 0, sim_scale = 1.):

        super().__init__()

        if channels:
            self.theta_scale = torch.tensor(theta)
            self.theta_par = torch.nn.Parameter(torch.ones(channels))
        else:
            self.theta_scale = theta
            self.theta_par = torch.nn.Parameter(torch.tensor(1.))

        self.theta_const = (self.theta_scale.to(self.theta_par.device) * self.theta_par).detach().cuda()

        self.register_buffer('baseline', torch.tensor(baseline))
        self.channels = channels
        self.sim_scale = sim_scale

    def forward(self, x_sim, background, const_theta_sim=False):
        """ Calculates the concentration (mean / theta) of a Gamma distribution given
        the signal x_sim and background tensors.
        Also applies a shift and returns resulting the Gamma distribution
        """

        theta = (self.theta_scale.to(self.theta_par.device) * self.theta_par)
        if const_theta_sim:
            theta = self.theta_const * self.sim_scale
        else:
            theta = theta * self.sim_scale

        theta = theta[None,:,None,None,None]

        x_sim_background = x_sim + background
        x_sim_background.clamp_(1.0 + self.baseline)

        conc = (x_sim_background - self.baseline) / theta
        xsim_dist = D.Gamma(concentration=conc, rate=1 / theta)

        loc_trafo = [D.AffineTransform(loc=self.baseline, scale=1)]
        xsim_dist = D.TransformedDistribution(xsim_dist, loc_trafo)
        return xsim_dist
