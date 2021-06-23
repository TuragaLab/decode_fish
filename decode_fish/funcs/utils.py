# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/12_utils.ipynb (unless otherwise specified).

__all__ = ['seed_everything', 'free_mem', 'center_crop', 'smooth', 'gaussian_sphere', 'tiff_imread', 'load_tiff_image',
           'gpu', 'cpu', 'zip_longest_special', 'param_iter']

# Cell
from ..imports import *
from itertools import product as iter_product
from tifffile import imread

import gc
import random

# Cell
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def free_mem():
    gc.collect()
    torch.cuda.empty_cache()

def center_crop(volume, zyx_ext):

    shape_3d = volume.shape[-3:]
    center = [s//2 for s in shape_3d]
    volume = volume[...,center[0]-math.floor(zyx_ext[0]/2):center[0]+math.ceil(zyx_ext[0]/2),
                        center[1]-math.floor(zyx_ext[1]/2):center[1]+math.ceil(zyx_ext[1]/2),
                        center[2]-math.floor(zyx_ext[2]/2):center[2]+math.ceil(zyx_ext[2]/2)]
    return volume

def smooth(x,window_len=11,window='flat'):

    if window_len<3:
        return x

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def gaussian_sphere(shape, radius, position):
    grid = [slice(-x0, dim - x0) for x0, dim in zip(position, shape)]
    position = np.ogrid[grid]
    arr = np.exp(-position[0]**2 / (2 * (radius[0] ** 2))) * np.exp(-position[1]**2 / (2 * (radius[1] ** 2))) * np.exp(-position[2]**2 / (2 * (radius[2] ** 2))) / (2 * np.pi * (radius[0] * radius[1] * radius[2]))
    return arr

# Cell
def tiff_imread(path):
    '''helper function to read tiff file with pathlib object or str'''
    if isinstance(path, str) : return imread(path)
    if isinstance(path, Path): return imread(str(path))

def load_tiff_image(image_path: str):
    "Given tiff stack path, loads the stack and converts it to a tensor. If necessary adds a dimension for the batch size"
    image_path = Path(image_path)
    image  = torch.tensor(tiff_imread(image_path).astype('float32'))
    if len(image.shape) == 3: image.unsqueeze_(0)
    assert len(image.shape) == 4, 'the shape of image must be 4, (1, Z, X, Y)'
    #removing minum values of the image
    return image

# Cell
def gpu(x):
    '''Transforms numpy array or torch tensor torch torch.cuda.FloatTensor'''
    return FloatTensor(x).cuda()

def cpu(x):
    '''Transforms torch tensor into numpy array'''
    if torch.is_tensor(x):
        return x.cpu().detach().numpy()
    else:
        return x

# Cell
def zip_longest_special(*iterables):
    def filter(items, defaults):
        return tuple(d if i is sentinel else i for i, d in zip(items, defaults))
    sentinel = object()
    iterables = itertools.zip_longest(*iterables, fillvalue=sentinel)
    first = next(iterables)
    yield filter(first, [None] * len(first))
    for item in iterables:
        yield filter(item, first)

class param_iter(object):

    def __init__(self):

        self.keys = []
        self.vals = []

    def add(self, name, *args):

        self.keys.append(name)
        self.vals.append(args)

    def param_product(self):

        all_params = []
        for values in iter_product(*self.vals):

            params = dict()
            for i,val in zip(self.keys,values):
                params.update({i : val })

            all_params.append(params)

        return all_params

    def param_zip(self):

        all_params = []
        for values in zip_longest_special(*self.vals):

            params = dict()
            for i,val in zip(self.keys,values):
                params.update({i : val })

            all_params.append(params)

        return all_params