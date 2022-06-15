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
from torch.utils.data import DataLoader
from decode_fish.engine.microscope import Microscope
from decode_fish.engine.model import UnetDecodeNoBn
import shutil
from decode_fish.engine.point_process import PointProcessUniform
from decode_fish.engine.gmm_loss import PointProcessGaussian
import torch_optimizer
from decode_fish.funcs.gen_train_funcs import *
import wandb
from decode_fish.funcs.merfish_codenet import *

@hydra.main(config_path='../config', config_name='metricnettrain')
def my_app(cfg):

    model_cfg = OmegaConf.load(cfg.model_cfg)
    test_csv = pd.read_csv(cfg.test_csv)
    
    model_cfg.training.bs = cfg.bs
    model, post_proc, micro, img_3d, decode_dl = load_all(model_cfg)
    
    cfg = OmegaConf.merge(model_cfg, cfg)
    codebook, targets = hydra.utils.instantiate(cfg.codebook)
    
    post_proc.codebook = expand_codebook(codebook)
    
    net = conv_net(5, codebook.shape[1], bn=cfg.batch_norm).cuda()
    # net = code_net(53).cuda()
    
    code_weight = torch.ones(len(post_proc.codebook))
    code_weight[len(codebook):] *= cfg.genm.emitter_noise.rate_fac
    point_process = PointProcessUniform(int_conc=cfg.genm.intensity_dist.int_conc, int_rate=cfg.genm.intensity_dist.int_rate, int_loc=cfg.genm.intensity_dist.int_loc, 
                                       sim_iters=5, n_channels=cfg.genm.exp_type.n_channels, sim_z=cfg.genm.exp_type.pred_z, slice_rec=cfg.genm.exp_type.slice_rec, 
                                       codebook=post_proc.codebook, int_option=cfg.training.int_option, code_weight=code_weight)
    
    _ = wandb.init(project=cfg.output.project, 
                   config=OmegaConf.to_container(cfg, resolve=True),
                   dir=cfg.output.log_dir,
                   group=cfg.output.group,
                   name=cfg.run_name)


    train_metric_net(net, 
                     model, 
                     decode_dl, 
                     post_proc, 
                     micro, 
                     point_process, 
                     cfg)

if __name__ == "__main__":
    my_app()
    