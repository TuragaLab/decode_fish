# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/24_exp_specific.ipynb (unless otherwise specified).

__all__ = ['simfish_to_df', 'matlab_fq_to_df', 'load_sim_fish', 'big_fishq_to_df', 'rsfish_to_df', 'read_MOp_tiff',
           'read_starfish_tiff', 'get_benchmark_from_starfish', 'get_starfish_benchmark', 'get_starfish_codebook',
           'get_istdeco', 'get_mop_codebook', 'get_mop_benchmark', 'get_mop_fov', 'get_mop_colors',
           'get_train_eval_benchmark_MOp', 'get_train_eval_benchmark_starfish', 'df_pp_mop', 'df_pp_starfish']

# Cell
from ..imports import *
from .utils import *
from tifffile import imread
from .emitter_io import *
import pandas as pd

# Cell
def simfish_to_df(sim_file, frame_idx=0, int_fac=1.05):

    yxzi = []
    with open(sim_file) as f:
        read = False
        for line in f:
            if 'Pos_Y' in line:
                read = True
                continue
            if 'SPOTS_END' in line: break
            if read: yxzi.append([float(s) for s in line.split()])

    yxzi = np.array(yxzi)#/px_size
    loc_idx = np.arange(len(yxzi))
    # Number calculated by taking into account their (or my?) normalization (by max and not by sum)
    if yxzi.shape[1] == 4:
        # PSF.max() | PSF.sum() | 3**3 (superres)  | PSF.max() | microscope scale
        # ints = yxzi[:,3] * 65535.0 * 156772560.0 / 27 / 65535.0 / 10000.0
        ints = yxzi[:,3] * 65535.0 / 100.0 / int_fac
    else:
        ints = np.ones_like(loc_idx)

    df = pd.DataFrame({'loc_idx': loc_idx,
                       'frame_idx': frame_idx,
                       'x': yxzi[:,1],
                       'y': yxzi[:,0],
                       'z': yxzi[:,2],
                       'prob': np.ones_like(loc_idx),
                       'int': ints,
                       'int_sig': np.ones_like(loc_idx),
                       'x_sig': np.ones_like(loc_idx),
                       'y_sig': np.ones_like(loc_idx),
                       'z_sig': np.ones_like(loc_idx)})

    return df


def matlab_fq_to_df(resfile, frame_idx=0):

    ind_dict = {'x':1, 'y':0, 'z':2, 'int':3, 'x_sig':6, 'y_sig':7, 'z_sig':8}

    yxzi = []
    with open(resfile) as f:
        read = False
        for line in f:
            if 'Pos_Y' in line:
                read = True
                continue
            if 'SPOTS_END' in line: break
            if read: yxzi.append([float(s) for s in line.split()])

    yxzi = np.array(yxzi)#/px_size
    loc_idx = np.arange(len(yxzi))

    df = pd.DataFrame({'loc_idx': loc_idx,
                       'frame_idx': frame_idx,
                       'x': yxzi[:,ind_dict['x']],
                       'y': yxzi[:,ind_dict['y']],
                       'z': yxzi[:,ind_dict['z']],
                       'prob': np.ones_like(loc_idx),
                       'int': yxzi[:,ind_dict['int']],
                       'int_sig': np.ones_like(loc_idx),
                       'x_sig': yxzi[:,ind_dict['x_sig']],
                       'y_sig': yxzi[:,ind_dict['y_sig']],
                       'z_sig': yxzi[:,ind_dict['z_sig']],
                       'comb_sig' : np.sqrt(yxzi[:,ind_dict['x_sig']]**2
                                              +yxzi[:,ind_dict['y_sig']]**2
                                              +yxzi[:,ind_dict['z_sig']]**2)})

    return df

def load_sim_fish(basedir, mrna_lvl=200, shape='cell3D', exp_strength='strong', cell_nr=0, shift=[-38,-38,-110], int_fac=1.05):

    spec_dir = f'/mRNAlevel_{mrna_lvl}/{shape}/{exp_strength}/'
    img_path = sorted(glob.glob(basedir + spec_dir + 'w1*.tif'))[cell_nr]
    cellname = Path(img_path).name.split('.')[0]
#     print(name)
    img = load_tiff_image(img_path)
    gt_df = simfish_to_df(img_path.split('.')[0] + '.txt', int_fac=int_fac)
    fq_nog_df = fq_gmm_df = DF()
    if os.path.exists(basedir + '/_results_detection/'):
        nog_path = Path(basedir + '/_results_detection/' + spec_dir + '/results_noGMM/' + cellname + '_res_NO_GMM.txt')
        gmm_path = Path(basedir + '/_results_detection/' + spec_dir + '/results_GMM/' + cellname + '_res_GMM.txt')

        if nog_path.is_file():
            fq_nog_df = simfish_to_df(nog_path, int_fac=int_fac)
            fq_nog_df = shift_df(fq_nog_df, shift)
        if gmm_path.is_file():
            fq_gmm_df = simfish_to_df(gmm_path, int_fac=int_fac)
            fq_gmm_df = shift_df(fq_gmm_df, shift)

    return img, gt_df, fq_nog_df, fq_gmm_df

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

def rsfish_to_df(file_str):

    csv = pd.read_csv(file_str,sep='  ',names=['x','y','z','?','??'], index_col=False)

    xyz = np.array(csv)
    loc_idx = np.arange(len(xyz))

    df = pd.DataFrame({'loc_idx': loc_idx,
                       'frame_idx': np.zeros_like(loc_idx),
                       'x': xyz[:,1],
                       'y': xyz[:,0],
                       'z': xyz[:,2],
                       'prob': np.ones_like(loc_idx),
                       'int': np.ones_like(loc_idx),
                       'int_sig': np.ones_like(loc_idx),
                       'x_sig': np.ones_like(loc_idx),
                       'y_sig': np.ones_like(loc_idx),
                       'z_sig': np.ones_like(loc_idx)})

    return df

# Cell
def read_MOp_tiff(image_path,  z_to_batch=False, sort_cols=False):
    img_stack = imread(image_path, key=range(0,7*22))
    img_stack = img_stack.reshape([22,7,2048,2048])
    img_stack = torch.tensor(img_stack.astype('float32'))
    if sort_cols:
        ch_cols = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0])
        img_stack  = torch.cat([img_stack[ch_cols==0], img_stack[ch_cols==1]], 0)

    if z_to_batch:
        return img_stack[None].permute([2,1,0,3,4])
    else:
        return img_stack[None] # Add batch dim.

def read_starfish_tiff(image_path):

    return load_tiff_image(image_path)[None]

# Cell
def get_benchmark_from_starfish(magnitude_threshold=10**0.75*4):

    bench = pd.read_csv(
        io.BytesIO(requests.get('https://d2nhj9g34unfro.cloudfront.net/MERFISH/benchmark_results.csv').content),
        dtype={'barcode': object})

    #See Fig. S4 https://www.pnas.org/content/113/39/11046

    bench_df = bench.copy()
    bench_df = bench_df[bench_df['total_magnitude']>magnitude_threshold]
    bench_df = bench_df[bench_df['area']>3]

    print(len(bench_df))

    experiment = data.MERFISH(use_test_data=True)
    codebook = experiment.codebook.data.reshape([140,-1], order='F')
    targets = experiment.codebook.indexes['target']

    bench_df.loc[:,'frame_idx'] = 0
    bench_df.loc[:,'loc_idx'] = np.arange(len(bench_df))
    bench_df.loc[:,'int'] = bench_df['total_magnitude']
    bench_df.loc[:,'z'] = 50/100
    bench_df = px_to_nm(bench_df)

    return bench_df, codebook, targets

def get_starfish_benchmark(magnitude_threshold=10**0.75*4):

    bench = pd.read_csv(base_path + '/decode_fish/data/merfish_bench_df.csv')

    #See Fig. S4 https://www.pnas.org/content/113/39/11046

    bench_df = bench.copy()
    bench_df = bench_df[bench_df['total_magnitude']>magnitude_threshold]
    bench_df = bench_df[bench_df['area']>3]

    print(len(bench_df))

    bench_df.loc[:,'frame_idx'] = 0
    bench_df.loc[:,'code_inds'] = bench_df['barcode_id'].values - 1
    bench_df.loc[:,'loc_idx'] = np.arange(len(bench_df))
    bench_df.loc[:,'int'] = bench_df['total_magnitude']
    bench_df.loc[:,'z'] = 50/100
    bench_df = px_to_nm(bench_df)
    bench_df = bench_df.drop('barcode_id', axis=1)
    return bench_df

def get_starfish_codebook():
    codebook = np.load(base_path + '/decode_fish/data/merfish_code_ref.npz')['arr_0']
    targets = np.load(base_path + '/decode_fish/data/merfish_targets.npz', allow_pickle=True)['arr_0']
    return codebook, targets

def get_istdeco():
    istdeco_df = pd.read_csv(base_path + '/decode_fish/data/results/ISTDECO.csv')
    istdeco_df = istdeco_df.rename(columns={'target_name':'gene'})
    fov = [40,40,2008,2008]
    istdeco_df = istdeco_df[ (istdeco_df['x'] >= fov[0]) & (istdeco_df['x'] <= fov[2])  & (istdeco_df['y'] >= fov[1]) & (istdeco_df['y'] <= fov[3])]

    istdeco_df.loc[:,'frame_idx'] = 0
    istdeco_df.loc[:,'code_inds'] = istdeco_df['target_id'].values - 1
    istdeco_df.loc[:,'loc_idx'] = np.arange(len(istdeco_df))
    istdeco_df.loc[:,'int'] = istdeco_df['intensity']
    istdeco_df.loc[:,'z'] = 50/100
    istdeco_df = px_to_nm(istdeco_df)
    istdeco_df = istdeco_df.drop('target_id', axis=1)

    print(len(istdeco_df))
    return istdeco_df

def get_mop_codebook():
    mop_path = base_path +'/datasets/CodFish/MERFISH/MOp/'
    codebook = pd.read_csv(mop_path + '/additional_files/codebook.csv')
    codebook = codebook.loc[:,codebook.columns[2:]].values.astype('int8')
    targets = open(mop_path + '/additional_files/genes_combinatorial.txt').read().split('\n')
    targets = np.concatenate([targets, [f'Blank-{i}' for i in range(1,11)]])
    targets = np.array(targets).astype('object')
    return codebook, targets

def get_mop_benchmark(mouse=1, sample=1, z_to_batch=True):

    mop_path = base_path +'/datasets/CodFish/MERFISH/MOp/'
    codebook, targets = get_mop_codebook()

    bench_df = pd.read_csv(mop_path + f'spots_mouse{mouse}sample{sample}.csv')
    bench_df.columns = ['loc_idx', 'x', 'y', 'z', 'gene']
    bench_df['frame_idx'] = 0
    bench_df[['x','y','z']] = 1000*bench_df[['x','y','z']] # um to nm

    t_dict = {}
    for i,k in enumerate(targets):
        t_dict[k] = i

    code_inds = [t_dict[x] for x in bench_df['gene']]
    bench_df['code_inds'] = code_inds

    if z_to_batch:
        bench_df['frame_idx'] = np.array(bench_df['z'].values / 1500, dtype='int16')
        bench_df['z'] *= 0

    print(len(bench_df))

    return bench_df

def get_mop_fov(bench_df, img_nr, mouse=1, sample=1):

    px_size = 108.5
    n_px = 2048

    mop_path = base_path +'/datasets/CodFish/MERFISH/MOp/'
    fov_pos = np.genfromtxt(mop_path + f'/additional_files/fov_positions/mouse{mouse}_sample{sample}.txt', delimiter=',')

    fov_x = fov_pos[img_nr,0]*1000 # um to nm
    fov_y = fov_pos[img_nr,1]*1000
    bench_sub = crop_df(bench_df, np.s_[:,:,fov_y:fov_y+n_px*px_size,fov_x:fov_x+n_px*px_size])

    return bench_sub

def get_mop_colors():
#     cols = (pd.read_csv('/groups/turaga/home/speisera/Mackebox/Artur/WorkDB/deepstorm/datasets/CodFish/MERFISH/MOp/additional_files/data_organization_raw.csv',encoding='latin')['color (nm)'].values == 750)
    return np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0,
       0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0])

# Cell
from .predict import window_predict
from .evaluation import matching


def get_train_eval_benchmark_MOp(datapath, crop):

    fov = int(''.join(filter(str.isdigit, datapath[-10:])))

    bench_df = get_mop_benchmark(z_to_batch=True)
    bench_df = get_mop_fov(bench_df, fov)
    bench_df = nm_to_px(bench_df, [1.085,1.085,1.085])
    bench_df['x'] += 70
    bench_df['y'] += 70

    return crop_df(bench_df, crop[-4:], px_size_zyx=[100., 100., 100.])

def get_train_eval_benchmark_starfish(crop):

    bench_df = get_starfish_benchmark()
    bench_df = bench_df[bench_df['gene'] != 'MALAT1']

    return bench_df

def df_pp_mop(res_df):

    res_df = remove_doublets(res_df, 200)
    return res_df

def df_pp_starfish(res_df):

    res_df = res_df[res_df['gene'] != 'MALAT1']
    res_df = exclude_borders(res_df, border_size_zyx=[0,4000,4000], img_size=[2048*100,2048*100,2048*100])
    res_df = remove_doublets(res_df, 200)
    res_df['x'] += 100
    res_df['y'] += 100
    return res_df