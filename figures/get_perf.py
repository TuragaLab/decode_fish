from decode_fish.imports import *
from decode_fish.funcs.file_io import *
from decode_fish.funcs.emitter_io import *
from decode_fish.funcs.utils import *
from decode_fish.funcs.dataset import *
from decode_fish.funcs.output_trafo import *
from decode_fish.funcs.evaluation import *
from decode_fish.funcs.plotting import *
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from decode_fish.engine.microscope import Microscope
from decode_fish.engine.model import UnetDecodeNoBn
import shutil
from torch.utils.tensorboard import SummaryWriter
from decode_fish.engine.point_process import PointProcessUniform
from decode_fish.engine.gmm_loss import PointProcessGaussian

from decode_fish.funcs.visualization import *
import ipyvolume as ipv

exp = sys.argv[1]
cfg = OmegaConf.load(f'/groups/turaga/home/speisera/Mackebox/Artur/WorkDB/deepstorm/models/fishcod/Fig_sim_density/nb_run/{exp}/train.yaml')

path = Path(cfg.output.save_dir)
model = hydra.utils.instantiate(cfg.model)
model = load_model_state(model, path, 'model.pkl')
post_proc = hydra.utils.instantiate(cfg.post_proc_isi, samp_threshold=0.5)

basedir = '/groups/turaga/home/speisera/share_TUM/FishSIM/' + sys.argv[2] + '/'
densities = [250,500,1000,2000,4000]
n_cells = 5

model.cuda()

int_threshold = float(sys.argv[3])

# Run once to get shift

dec_df = []
gtc_df = []
for i in range(n_cells):
    img, gt_df, fq_nog_df, fq_gmm_df = load_sim_fish(basedir, 500, 'random', 'NR', i)
    gtc_df.append(gt_df)
    with torch.no_grad():
        curr_df = shift_df(post_proc(model(img[None].cuda()), 'df'), [-100,-100,-100])
        dec_df.append(shift_df(curr_df, [0,0,0]))
        
gtc_df = cat_emitter_dfs(gtc_df)
dec_df = cat_emitter_dfs(dec_df)
dec_df = dec_df[dec_df['int'] > int_threshold]
perf_df, matches, shift = matching(gtc_df, dec_df, print_res=True)

# Run over all densities

dec_col = []

for d in densities:
    print(d)
    gtc_df = []
    dec_df = []
    for i in range(n_cells):
        img, gt_df, fq_nog_df, fq_gmm_df = load_sim_fish(basedir, d, 'random', 'NR', i)
        gtc_df.append(gt_df)
        with torch.no_grad():
            curr_df = shift_df(post_proc(model(img[None].cuda()), 'df'), [-100,-100,-100])
            dec_df.append(shift_df(curr_df, shift))
        
    gtc_df = cat_emitter_dfs(gtc_df)
    dec_df = cat_emitter_dfs(dec_df)
    
    dec_df = dec_df[dec_df['int'] > int_threshold]
        
    perf_df, matches, _ = matching(gtc_df, dec_df, print_res=True)
    dec_col.append(perf_df)
    
with open(basedir + f'dec{sys.argv[4]}_perf_dfs.pkl', 'wb') as f:
    pickle.dump({'dec':dec_col, 'densities':[250,500,1000,2000,4000]}, f)