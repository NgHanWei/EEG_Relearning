from cgi import test
from vae.utils import *
from vae_models import vanilla_vae
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import random
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
from cmath import inf
from braindecode.torch_ext.util import set_random_seeds
import h5py
set_random_seeds(seed=20200205, cuda=True)

import vae_subj_select

parser = argparse.ArgumentParser(
    description='Subject-adaptative classification with KU Data')
parser.add_argument('datapath', type=str, help='Path to the h5 data file')
parser.add_argument('-start', type=int,
                    help='Start of the subject index', default=1)
parser.add_argument(
    '-end', type=int, help='End of the subject index (not inclusive)', default=55)
parser.add_argument('-subj', type=int,
                    help='Target Subject for Subject Selection, will override start and end')
parser.add_argument('-trial', type=int, default = 7,
                    help='How many trials to use for few-shot, set to 0 to use validation results')

args = parser.parse_args()

start = args.start
end = args.end
assert(start < end)
subjs = args.subj if args.subj else range(start, end)
trials = args.trial

for subj in subjs:
    baseline_subject = vae_subj_select.vae_select(subj,trials-1,args.datapath)
    baseline_subject.run()