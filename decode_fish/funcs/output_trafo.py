# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/09_output_trafo.ipynb (unless otherwise specified).

__all__ = ['sample_to_df', 'df_to_micro', 'SIPostProcess', 'ISIPostProcess']

# Cell
from ..imports import *
import torch.nn.functional as F
from .plotting import *
from .emitter_io import *

# Cell
def sample_to_df(locs, x_os, y_os, z_os, ints, px_size_zyx=[100,100,100]):

    n_locs = len(ints)

    x = locs[-1] + x_os + 0.5
    y = locs[-2] + y_os + 0.5
    z = locs[-3] + z_os + 0.5

    frame_idx = locs[0]
    loc_idx = torch.arange(n_locs)

    df = DF({'loc_idx': loc_idx.cpu(),
             'frame_idx': frame_idx.cpu(),
             'x': x.cpu()*px_size_zyx[2],
             'y': y.cpu()*px_size_zyx[1],
             'z': z.cpu()*px_size_zyx[0],
             'int': ints.cpu()})

    return df

def df_to_micro(df, px_size_zyx=[100,100,100]):

    locs = tuple([torch.tensor(df['frame_idx'],dtype=torch.int64).cuda(),
                 torch.zeros(len(df),dtype=torch.int64).cuda(),
                 torch.tensor((df['z']/px_size_zyx[0] + 0.5),dtype=torch.int64).cuda(),
                 torch.tensor((df['y']/px_size_zyx[1] + 0.5),dtype=torch.int64).cuda(),
                 torch.tensor((df['x']/px_size_zyx[2] + 0.5),dtype=torch.int64).cuda()])
    z = (torch.tensor(df['z'],dtype=torch.float32).cuda()-locs[2]*px_size_zyx[0])/px_size_zyx[0] - 0.5
    y = (torch.tensor(df['y'],dtype=torch.float32).cuda()-locs[3]*px_size_zyx[1])/px_size_zyx[1] - 0.5
    x = (torch.tensor(df['x'],dtype=torch.float32).cuda()-locs[4]*px_size_zyx[2])/px_size_zyx[2] - 0.5
    ints = torch.tensor(df['int']).cuda()

    return locs, x, y, z, ints

# Cell
class SIPostProcess(torch.nn.Module):

    def __init__(self, m1_threshold:float = 0.03, m2_threshold:float = 0.3, samp_threshold=0.1, px_size_zyx=[100,100,100], diag=0):

        super().__init__()
        self.m1_threshold = m1_threshold
        self.m2_threshold = m2_threshold
        self.samp_threshold = samp_threshold
        self.diag = diag
        self.px_size_zyx = px_size_zyx

        if not diag:
            d1 = 0; d2 = 0
        else:
            d1 = 1/np.sqrt(2); d2 = 1/np.sqrt(3)
#             d1 = 1; d2 = 1
        self.filt = torch.FloatTensor([[[d2,d1,d2],[d1,1,d1],[d2,d1,d2]],
                                       [[d1, 1,d1],[1, 1, 1],[d1, 1,d1]],
                                       [[d2,d1,d2],[d1,1,d1],[d2,d1,d2]]])[None,None]

    def spatial_integration(self, p):

        device = p.device

        with torch.no_grad():

            p_copy = p + 0

            # probability values > threshold are regarded as possible locations
            p_clip = torch.where(p>self.m1_threshold,p,torch.zeros_like(p))

            # localize maximum values within a 3x3 patch
            pool = F.max_pool3d(p_clip,3,1,padding=1)
            max_mask1 = torch.eq(p, pool).float()

            # Add probability values from the 4 adjacent pixels
            conv = F.conv3d(p, self.filt.to(device) ,padding=1)
            p_ps1 = (max_mask1 * conv)

            # In order do be able to identify two fluorophores in adjacent pixels we look for probablity values > 0.5 that are not part of the first mask

            p_copy *= (1-max_mask1)
            p_clip = torch.where(p_copy>self.m2_threshold, p_copy,torch.zeros_like(p_copy))
            max_mask2 = torch.where(p_copy>self.m2_threshold, torch.ones_like(p_copy),torch.zeros_like(p_copy))
            p_ps2 = max_mask2*conv

            # This is our final clustered probablity which we then threshold (normally > 0.7) to get our final discrete locations
            p_ps = p_ps1 + p_ps2

            return p_ps, None

    def forward(self, res_dict, ret='df'):

        probs =  torch.sigmoid(res_dict['logits'])
        res_dict['Probs_si'], res_dict['Iters_si'] = self.spatial_integration(probs)
        res_dict['Samples_si'] = torch.where(res_dict['Probs_si'] > self.samp_threshold, torch.ones_like(res_dict['Probs_si']), torch.zeros_like(res_dict['Probs_si']))

        if ret == 'df':

            res_dict = {k:v.cpu() for (k,v) in res_dict.items()}
            locations = res_dict['Samples_si'].nonzero(as_tuple=True)

            pos_x, pos_y, pos_z = locations[-1] ,locations[-2], locations[-3]
            x = pos_x + res_dict['xyzi_mu'][:,[0]][locations] + 0.5
            y = pos_y + res_dict['xyzi_mu'][:,[1]][locations] + 0.5
            z = pos_z + res_dict['xyzi_mu'][:,[2]][locations] + 0.5

            loc_idx = torch.arange(len(x))
            frame_idx = locations[0]

            df = DF({'loc_idx': loc_idx,
                     'frame_idx': frame_idx,
                     'x': x*self.px_size_zyx[2],
                     'y': y*self.px_size_zyx[1],
                     'z': z*self.px_size_zyx[0],
                     'prob': res_dict['Probs_si'][locations],
                     'int': res_dict['xyzi_mu'][:,[3]][locations],
                     'int_sig': res_dict['xyzi_sigma'][:,[3]][locations],
                     'x_sig': res_dict['xyzi_sigma'][:,[0]][locations]*self.px_size_zyx[0],
                     'y_sig': res_dict['xyzi_sigma'][:,[1]][locations]*self.px_size_zyx[1],
                     'z_sig': res_dict['xyzi_sigma'][:,[2]][locations]*self.px_size_zyx[2],
                     'comb_sig': torch.sqrt(res_dict['xyzi_sigma'][:,[0]][locations]**2
                                           +res_dict['xyzi_sigma'][:,[1]][locations]**2
                                           +res_dict['xyzi_sigma'][:,[2]][locations])})

            return df

        elif ret == 'micro':

            locations = res_dict['Samples_si'].nonzero(as_tuple=True)
            x_os_3d = res_dict['xyzi_mu'][:,[0]][locations]
            y_os_3d = res_dict['xyzi_mu'][:,[1]][locations]
            z_os_3d = res_dict['xyzi_mu'][:,[2]][locations]
            ints_3d = res_dict['xyzi_mu'][:,[3]][locations]
            output_shape  = res_dict['Samples_si'].shape
            comb_sig = torch.sqrt(res_dict['xyzi_sigma'][:,[0]][locations]**2
                                 +res_dict['xyzi_sigma'][:,[1]][locations]**2
                                 +res_dict['xyzi_sigma'][:,[2]][locations])
            iters_si = res_dict['Iters_si'][locations]

            return locations, x_os_3d, y_os_3d, z_os_3d, ints_3d, output_shape, comb_sig, iters_si

        elif ret == 'dict':

            return res_dict

# p_col = []

#export
class ISIPostProcess(SIPostProcess):

    def __init__(self, m1_threshold:float = 0.1, samp_threshold=0.1, px_size_zyx=[100,100,100], diag=False):

        super().__init__(m1_threshold = m1_threshold, samp_threshold=samp_threshold, px_size_zyx=px_size_zyx, diag=diag)
        self.m2_threshold = None

    def spatial_integration(self, p):

        device = p.device
        count = 0

        with torch.no_grad():

            p_ret = 0
            tot_mask = torch.ones_like(p)
            count_arr = torch.zeros_like(p)

            while True:

                count += 1
                p_copy = p + 0

                # probability values > threshold are regarded as possible locations
                p_clip = torch.where(p>self.m1_threshold,p,torch.zeros_like(p))*tot_mask

                # localize maximum values within a 3x3 patch
                pool = F.max_pool3d(p_clip,3,1,padding=1)
                max_mask1 = torch.eq(p, pool).float()
                max_mask1[p==0] = 0

                count_arr += max_mask1*count

                tot_mask *= (torch.ones_like(max_mask1) - max_mask1)

                # Add probability values from the adjacent pixels
                conv = F.conv3d(p, self.filt.to(device) ,padding=1)
                p_ps = max_mask1 * conv

                p_ret += torch.clamp_max(p_ps, 1)

                p_fac = 1/p_ps
                p_fac[torch.isinf(p_fac)] = 0
                p_fac = torch.clamp_max(p_fac, 1)
                p_proc = F.conv3d(p_fac, self.filt.to(device),padding=1)*p

#                 plt.figure(figsize=(20,5))
#                 plt.subplot(141)
#                 plt.imshow(cpu(p[0,0][sl[1:]]).sum(0))
#                 plt.subplot(142)
#                 plt.imshow(cpu(p_ps[0,0][sl[1:]]).sum(0))
#                 plt.title(cpu(p_ps[0,0][sl[1:]]).sum())
#                 plt.colorbar()
#                 plt.subplot(143)
#                 plt.imshow(cpu(p_proc[0,0][sl[1:]]).sum(0))
#                 plt.title(cpu(p_proc[0,0][sl[1:]]).sum())
#                 plt.colorbar()
#                 plt.show()

                p = p - p_proc
                torch.clamp_min_(p, 0)

                if not max_mask1.sum():
                    break

            return p_ret, count_arr