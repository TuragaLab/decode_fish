import os, io, sys, glob, pickle, collections, itertools, time, math, warnings
import matplotlib, copy
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame as DF
import scipy
from tqdm import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver("add", lambda *numbers: sum(numbers))
OmegaConf.register_new_resolver("bool_fac", lambda bfac, num: bool(bfac)*num)

default_conf = '../config/train.yaml'
# simfish_dir = '/groups/turaga/home/speisera/share_TUM/FishSIM/'
# temp_dir = '/groups/turaga/home/speisera/Mackebox/Artur/WorkDB/deeppop/temp_save/'
# exp_dir = '/groups/turaga/home/speisera/Mackebox/Artur/WorkDB/deeppop/DiSAE_nbdev/experiments/'
base_path = '/groups/turaga/home/speisera/Mackebox/Artur/WorkDB/deepstorm/'
