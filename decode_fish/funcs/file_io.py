# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/07_file_io.ipynb (unless otherwise specified).

__all__ = ['tiff_imread', 'load_model_state', 'simfish_to_df', 'load_sim_fish', 'big_fishq_to_df', 'load_tiff_image',
           'load_psf_noise_micro', 'load_post_proc']

# Cell
from ..imports import *
from .utils import *
from tifffile import imread
from ..engine.microscope import Microscope
from ..engine.psf import crop_psf

# Cell
def tiff_imread(path):
    '''helper function to read tiff file with pathlib object or str'''
    if isinstance(path, str) : return imread(path)
    if isinstance(path, Path): return imread(str(path))

# Cell
def load_model_state(model, path, file_name ='model.pkl'):
    model_dict = torch.load(Path(path)/file_name)
    model.load_state_dict(model_dict['state_dict'])
    model.unet.inp_scale = model_dict['scaling'][0]
    model.unet.inp_offset = model_dict['scaling'][1]
    return model

# Cell
def simfish_to_df(sim_file, px_size=np.array([100.,100.,300.]), frame_idx=0):

    yxz = []
    with open(sim_file) as f:
        read = False
        for line in f:
            if 'Pos_Y' in line:
                read = True
                continue
            if 'SPOTS_END' in line: break
            if read: yxz.append([float(s) for s in line.split()[:3]])

    yxz = np.array(yxz)/px_size
    loc_idx = np.arange(len(yxz))

    df = pd.DataFrame({'loc_idx': loc_idx,
                       'frame_idx': frame_idx,
                       'x': yxz[:,1]*px_size[0],
                       'y': yxz[:,0]*px_size[1],
                       'z': yxz[:,2]*px_size[2],
                       'prob': np.ones_like(loc_idx),
                       'int': np.ones_like(loc_idx),
                       'int_sig': np.ones_like(loc_idx),
                       'x_sig': np.ones_like(loc_idx),
                       'y_sig': np.ones_like(loc_idx),
                       'z_sig': np.ones_like(loc_idx)})

    return df

#export
def load_sim_fish(basedir, mrna_lvl=200, shape='cell3D', exp_strength='strong'):

    spec_dir = f'/mRNAlevel_200/{shape}/{exp_strength}/'
#     file_name = f'/w1_HelaKyoto_Gapdh_2597_p01_cy3__Cell_CP_{cell_cp}__{shape}__{nr}'
    img_path = glob.glob(basedir + spec_dir + 'w1*.tif')[0]
    print(img_path)
    img = load_tiff_image(img_path)
    gt_df = simfish_to_df(glob.glob(basedir + spec_dir + '*.txt')[0])
    fq_nog_df = simfish_to_df(glob.glob(basedir + '/_results_detection/' + spec_dir + '/results_noGMM/' + 'w1*.txt')[0])
    fq_gmm_df = simfish_to_df(glob.glob(basedir + '/_results_detection/' + spec_dir + '/results_GMM/' + 'w1*.txt')[0])

    return img, gt_df, fq_nog_df, fq_gmm_df

#export
def big_fishq_to_df(file_str):

    csv = pd.read_csv(file_str,sep=';',names=['z','y','x'], index_col=False)

    zyx = np.array(csv)
    loc_idx = np.arange(len(zyx))

    df = pd.DataFrame({'loc_idx': loc_idx,
                       'frame_idx': np.zeros_like(loc_idx),
                       'x': zyx[:,2],
                       'y': zyx[:,1],
                       'z': zyx[:,0],
                       'prob': np.ones_like(loc_idx),
                       'int': np.ones_like(loc_idx),
                       'int_sig': np.ones_like(loc_idx),
                       'x_sig': np.ones_like(loc_idx),
                       'y_sig': np.ones_like(loc_idx),
                       'z_sig': np.ones_like(loc_idx)})

    return df

# Cell
def load_tiff_image(image_path: str):
    "Given tiff stack path, loads the stack and converts it to a tensor. If necessary adds a dimension for the batch size"
    image_path = Path(image_path)
    image  = torch.tensor(tiff_imread(image_path).astype('float32'))
    if len(image.shape) == 3: image.unsqueeze_(0)
    assert len(image.shape) == 4, 'the shape of image must be 4, (1, Z, X, Y)'
    #removing minum values of the image
    return image

# Cell
def load_psf_noise_micro(cfg):
    psf_state = torch.load(cfg.data_path.psf_path)

    psf = hydra.utils.instantiate(cfg.PSF, size_zyx=psf_state['psf_volume'].shape[-3:])
    psf.load_state_dict(psf_state)

    if cfg.microscope.psf_extent_zyx:
        psf = crop_psf(psf,cfg.microscope.psf_extent_zyx)

    noise = hydra.utils.instantiate(cfg.noise)
    micro = Microscope(parametric_psf=[psf], noise=noise, multipl=cfg.microscope.multipl, psf_noise=cfg.microscope.psf_noise, clamp_mode=cfg.microscope.clamp_mode).cuda()

    return psf, noise, micro

def load_post_proc(cfg):
    if cfg.other.pp == 'si':
        return hydra.utils.instantiate(cfg.post_proc_si)
    if cfg.other.pp == 'isi':
        return hydra.utils.instantiate(cfg.post_proc_isi)