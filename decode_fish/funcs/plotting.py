# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/06_plotting.ipynb (unless otherwise specified).

__all__ = ['add_colorbar', 'sl_plot', 'gt_plot', 'plot_3d_projections', 'plot_prob_hist']

# Cell
from ..imports import *
from mpl_toolkits import axes_grid1
from matplotlib import pyplot as plt
from matplotlib import gridspec
import seaborn as sns
from .utils import *

# Cell
def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """ Add a vertical color bar to an image plot """

    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

# Cell
def sl_plot(x, xsim, pred_df, target_df, background, res):

    pred_df = pred_df[pred_df['frame_idx']==0]
    target_df = target_df[target_df['frame_idx']==0]
    with torch.no_grad():
        fig = plt.figure(figsize=(20,4))
        plt.subplot(151)
        im = plt.imshow(x[0][0].cpu().numpy().max(0))
        add_colorbar(im)
        plt.axis('off')
        plt.title('Real image')

        plt.subplot(152)
        im = plt.imshow(xsim[0][0].cpu().numpy().max(0))
        plt.scatter(target_df['x']/100, target_df['y']/100,facecolors='black', edgecolors='black', marker='x', s=25.)
        plt.scatter(pred_df['x']/100, pred_df['y']/100,facecolors='red', edgecolors='red', marker='o', s=5.)
        add_colorbar(im)
        plt.axis('off')
        plt.title('Sim. image')

        plt.subplot(153)
        im = plt.imshow(torch.sigmoid(res['logits'][0][0]).cpu().numpy().max(0))
        add_colorbar(im)
        plt.axis('off')
        plt.title('Predicted locations')

        plt.subplot(154)
        im = plt.imshow(background[0][0].cpu().numpy().max(0))
        add_colorbar(im)
        plt.axis('off')
        plt.title('Background')

        plt.subplot(155)
        im = plt.imshow(res['background'][0][0].cpu().numpy().max(0))
        add_colorbar(im)
        plt.axis('off')
        plt.title('Predicted background')

    return fig

def gt_plot(x, pred_df, gt_df, px_size, gt_rec=None, psf=None, fig_size=(24,6)):

    with torch.no_grad():
        fig = plt.figure(figsize=fig_size)
        plt.subplot(141)

        x = x[0].cpu().numpy()

        max_proj = x.max(0)
        vmax = max_proj.max()
        im = plt.imshow(max_proj, vmax=vmax)
        add_colorbar(im)
        plt.scatter(pred_df['x']/px_size[0], pred_df['y']/px_size[1],facecolors='red', edgecolors='red', marker='+', s=20)
#         plt.scatter(gt_df['x']/px_size[0], gt_df['y']/px_size[1],facecolors='none', edgecolors='black', marker='o', s=20)
        plt.axis('off')
        plt.title('Real image')

        if gt_rec is not None:

            gt_rec = gt_rec[0].cpu().numpy()

            plt.subplot(142)
            im = plt.imshow(gt_rec.max(0), vmax=vmax)
            add_colorbar(im)
            plt.axis('off')
            plt.title('RMSE ' + str(np.round(np.sqrt(((x-gt_rec)**2).mean()),2)))

            plt.subplot(143)
            im = plt.imshow(abs(x - gt_rec).max(0))
            add_colorbar(im)
            plt.axis('off')

            if psf is not None:
                plt.subplot(144)
                im = plt.imshow(psf.psf_volume[0].detach().cpu().numpy().mean(1))
                plt.axis('off')

    return fig

def plot_3d_projections(volume, projection='mean', size=6, vmax=None):

    if torch.is_tensor(volume):
        plot_vol = volume.detach().cpu().numpy()
    else:
        plot_vol = volume

    fig, axes = plt.subplots(1,3, figsize=(3*size, size))
    for i in range(3):
        if 'mean' in projection:
            im = axes[i].imshow(plot_vol.mean(i),vmax=vmax)
        if 'max' in projection:
            im = axes[i].imshow(plot_vol.max(i),vmax=vmax)
        add_colorbar(im)

    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('z')
    axes[2].set_xlabel('y')
    axes[2].set_ylabel('z')

    return axes

def plot_prob_hist(res_dict):
    fig = plt.figure()
    plt.hist(cpu(torch.sigmoid(res_dict['logits'])).reshape(-1), bins=np.linspace(0.01,1,100))
    return fig
