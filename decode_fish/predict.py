from decode_fish.imports import *
from decode_fish.funcs.file_io import *
from decode_fish.funcs.emitter_io import *
from decode_fish.funcs.utils import *
from decode_fish.funcs.dataset import *
from decode_fish.funcs.output_trafo import *
from decode_fish.engine.model import UnetDecodeNoBn
from decode_fish.funcs.predict import predict
import wandb

import h5py

@hydra.main(config_path='../config', config_name='predict')
def my_app(cfg):

    model = hydra.utils.instantiate(cfg.model)
    model = load_model_state(model, cfg.model_path)
    post_proc = hydra.utils.instantiate(cfg.post_proc_isi)
    
    image_paths = sorted(glob.glob(cfg.image_path))
    pred_df = predict(model, post_proc, image_paths, window_size=[None,cfg.predict.window_size_xy,cfg.predict.window_size_xy], device='cuda')
    pred_df.to_csv(cfg.out_file, index=False)
    
if __name__ == "__main__":
    my_app()