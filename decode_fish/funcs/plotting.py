# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/06_plotting.ipynb (unless otherwise specified).

__all__ = ['add_colorbar', 'sl_plot', 'gt_plot', 'plot_3d_projections', 'scat_3d_projections', 'plot_prob_hist',
           'combine_figures']

# Cell
from ..imports import *
from mpl_toolkits import axes_grid1
from matplotlib import pyplot as plt
from matplotlib import gridspec
import seaborn as sns
from .utils import *
from .emitter_io import *
from matplotlib.backends.backend_agg import FigureCanvas

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
        plt.scatter(target_df['x'], target_df['y'],facecolors='black', edgecolors='black', marker='x', s=25.)
        plt.scatter(pred_df['x'], pred_df['y'],facecolors='red', edgecolors='red', marker='o', s=5.)
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
        plt.scatter(pred_df['x'], pred_df['y'],facecolors='red', edgecolors='red', marker='+', s=20)
#         plt.scatter(gt_df['x'], gt_df['y'],facecolors='none', edgecolors='black', marker='o', s=20)
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

def plot_3d_projections(volume, proj_func=np.max, size=6, vmax=None, display=True):

    if torch.is_tensor(volume):
        plot_vol = volume.detach().cpu().numpy()
    else:
        plot_vol = volume

    for _ in range(plot_vol.ndim - 3):
        plot_vol = plot_vol.squeeze(0)

    x,y,z = plot_vol.shape[::-1]

    size_y = size * ((y+z)/y)
    size_x = size * ((1.1*x+z)/x)

    fig, ((ax_yx, ax_yz, ax_c), (ax_zx, ax_t1, ax_t2)) = plt.subplots(2, 3, figsize=(size_x,size_y), sharex='col', sharey=False,
                                                 gridspec_kw={'height_ratios': [y, z], 'width_ratios':[x, z, x/20]})
    plt.subplots_adjust(hspace=0.0,wspace=0.05)

    im = ax_yx.imshow(proj_func(plot_vol, 0),vmax=vmax)
    ax_yz.imshow(proj_func(plot_vol, 2).T,vmax=vmax)
    ax_zx.imshow(proj_func(plot_vol, 1),vmax=vmax)

    ax_t1.axis('off')
    ax_t2.axis('off')
#     ax_yz.axis('off')

    fig.colorbar(im, cax=ax_c)

    ax_yx.set_ylabel('y')
    ax_zx.set_xlabel('x')
    ax_zx.set_ylabel('z')

    ax_yz.set_xlabel('z')
    ax_yz.set_yticklabels([])

    plt.tight_layout()

    if not display: plt.close(fig)

    return fig, [ax_yx,ax_zx,ax_yz]

def scat_3d_projections(axes, dfs, px_size_zyx=[1.,1.,1], s_fac=1.):
    colors = ['red','black','orange']
    markers = ['o','+','x']
    if not isinstance(dfs, list):
        dfs = [dfs]
    for i,df in enumerate(dfs):
        df = nm_to_px(df, px_size_zyx)
        axes[0].scatter(df['x'],df['y'], color=colors[i], marker=markers[i], s=10*s_fac, label=f'DF {i}')
        axes[1].scatter(df['x'],df['z'], color=colors[i], marker=markers[i], s=10*s_fac)
        axes[2].scatter(df['z'],df['y'], color=colors[i], marker=markers[i], s=10*s_fac)
    axes[0].legend()


def plot_prob_hist(res_dict):
    fig = plt.figure()
    plt.hist(cpu(torch.sigmoid(res_dict['logits'])).reshape(-1), bins=np.linspace(0.01,1,100))
    return fig

# Cell
def combine_figures(figures, titles, nrows=1, ncols=2, figsize=(10,5)):

    imgs = []
    for f in figures:
        canvas = FigureCanvas(f)
        canvas.draw()
        imgs.append(np.array(canvas.renderer.buffer_rgba()))

    figure = plt.figure(figsize=figsize)
    axes = figure.subplots(nrows, ncols)
    plt.subplots_adjust(hspace=0.,wspace=0.)

    for i in range(len(imgs)):
        axes[i].imshow(imgs[i])
        axes[i].axis('off')
        if len(titles) >= i-1:
            axes[i].set_title(titles[i])

    return figure