#!/usr/bin/env python
# coding: utf-8
'''Subject-adaptative classification with KU Data,
using Deep ConvNet model from [1].

References
----------
.. [1] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J.,
   Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F. & Ball, T. (2017).
   Deep learning with convolutional neural networks for EEG decoding and
   visualization.
   Human Brain Mapping , Aug. 2017. Online: http://dx.doi.org/10.1002/hbm.23730
'''
import argparse
import logging
import sys
from os.path import join as pjoin
import pandas as pd

import vae_phase_select
import vae_phase_select_exclude

import numpy as np
import h5py
import torch
import torch.nn.functional as F
from braindecode.models.deep4 import Deep4Net
from braindecode.torch_ext.optimizers import AdamW
from braindecode.torch_ext.util import set_random_seeds
from torch import nn

logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                    level=logging.INFO, stream=sys.stdout)

parser = argparse.ArgumentParser(
    description='Subject-adaptative classification with KU Data')
parser.add_argument('datapath', type=str, help='Path to the h5 data file')
parser.add_argument('modelpath', type=str,
                    help='Path to the base model folder')
parser.add_argument('outpath', type=str, help='Path to the result folder')
parser.add_argument('-scheme', type=int, help='Adaptation scheme', default=4)
parser.add_argument(
    '-trfrate', type=int, help='The percentage of data for adaptation', default=100)
parser.add_argument('-lr', type=float, help='Learning rate', default=0.0005)
parser.add_argument('-gpu', type=int, help='The gpu device to use', default=0)
parser.add_argument('-trial', type=int, default = 1,
                    help='How many trials to use for few-shot')
parser.add_argument('-exclude', default=False, action='store_true')

args = parser.parse_args()
datapath = args.datapath
outpath = args.outpath
modelpath = args.modelpath
scheme = args.scheme
rate = args.trfrate
lr = args.lr
trials = args.trial
exclude = args.exclude

dfile = h5py.File(datapath, 'r')
torch.cuda.set_device(args.gpu)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.set_num_threads(2)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
set_random_seeds(seed=20200205, cuda=True)

BATCH_SIZE = 16
TRAIN_EPOCH = 10

# Randomly shuffled subject.
subjs = [35, 47, 46, 37, 13, 27, 12, 32, 53, 54, 4, 40, 19, 41, 18, 42, 34, 7,
         49, 9, 5, 48, 29, 15, 21, 17, 31, 45, 1, 38, 51, 8, 11, 16, 28, 44, 24,
         52, 3, 26, 39, 50, 6, 23, 2, 14, 25, 20, 10, 33, 22, 43, 36, 30]

# Get data from single subject.
def get_data(subj):
    dpath = 's' + str(subj)
    X = dfile[dpath]['X']
    Y = dfile[dpath]['Y']
    return X, Y

def get_multi_data(subjs):
    Xs = []
    Ys = []
    for s in subjs:
        x, y = get_data(s)
        Xs.append(x[:])
        Ys.append(y[:])
    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    return X, Y

def get_multi_phase(subjs,phases):
    Xs = []
    Ys = []
    for i in range(len(subjs)):
        s = subjs[i]
        phase = phases[i]
        x, y = get_data(s)
        Xs.append(x[(phase-1)*100:phase*100])
        Ys.append(y[(phase-1)*100:phase*100])
    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    return X, Y

def reset_conv_pool_block(network, block_nr):
    suffix = "_{:d}".format(block_nr)
    conv = getattr(network, 'conv' + suffix)
    kernel_size = conv.kernel_size
    n_filters_before = conv.in_channels
    n_filters = conv.out_channels
    setattr(network, 'conv' + suffix,
            nn.Conv2d(
                n_filters_before,
                n_filters,
                kernel_size,
                stride=(1, 1),
                bias=False,
            ))
    setattr(network, 'bnorm' + suffix,
            nn.BatchNorm2d(
                n_filters,
                momentum=0.1,
                affine=True,
                eps=1e-5,
            ))
    # Initialize the layers.
    conv = getattr(network, 'conv' + suffix)
    bnorm = getattr(network, 'bnorm' + suffix)
    nn.init.xavier_uniform_(conv.weight, gain=1)
    nn.init.constant_(bnorm.weight, 1)
    nn.init.constant_(bnorm.bias, 0)


def reset_model(checkpoint,model):
    # Load the state dict of the model.
    model.network.load_state_dict(checkpoint['model_state_dict'])

    if scheme != 5:
        # Freeze all layers.
        for param in model.network.parameters():
            param.requires_grad = False

        if scheme in {1, 2, 3, 4}:
            # Unfreeze the FC layer.
            for param in model.network.conv_classifier.parameters():
                param.requires_grad = True

        if scheme in {2, 3, 4}:
            # Unfreeze the conv4 layer.
            for param in model.network.conv_4.parameters():
                param.requires_grad = True
            for param in model.network.bnorm_4.parameters():
                param.requires_grad = True

        if scheme in {3, 4}:
            # Unfreeze the conv3 layer.
            for param in model.network.conv_3.parameters():
                param.requires_grad = True
            for param in model.network.bnorm_3.parameters():
                param.requires_grad = True

        if scheme == 4:
            # Unfreeze the conv2 layer.
            for param in model.network.conv_2.parameters():
                param.requires_grad = True
            for param in model.network.bnorm_2.parameters():
                param.requires_grad = True

    # Only optimize parameters that requires gradient.
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.network.parameters()),
                      lr=lr, weight_decay=0.5*0.001)
    model.compile(loss=F.nll_loss, optimizer=optimizer,
                  iterator_seed=20200205, )

    return model

cutoff = int(rate * 200 / 100)
# Use only session 1 data for training
assert(cutoff <= 200)

# total_loss = []

baseline = []
normal_adapt = []
few_shot = []

def update_model(update_subjs,subj,trial_num,update_phases):
    n_classes = 2
    set_random_seeds(seed=20200205, cuda=True)

    if subj in update_subjs:
        np.delete(update_subjs,np.where(update_subjs == subj))

    # Model update data        
    X, Y = get_multi_data(update_subjs)
    X_train_update, Y_train_update = X[:], Y[:]

    # Model phase update data
    X, Y = get_multi_phase(update_subjs,update_phases)
    X_train_update, Y_train_update = X[:], Y[:]

    X1, Y1 = get_data(subj)
    X_val, Y_val = X1[200:300], Y1[200:300]
    X_test_N, Y_test_N = X1[300:301+trial_num], Y1[300:301+trial_num]

    ### Adaptation on Baseline
    XB, YB = get_data(subjs[0])
    n_classes = 2
    in_chans = XB.shape[1]

    model = Deep4Net(in_chans=XB.shape[1], n_classes=n_classes,
                    input_time_length=XB.shape[2],
                    final_conv_length='auto').cuda()

    checkpoint = torch.load(pjoin(modelpath, 'subj_' + str(subj) + '.pt'),
                            map_location='cuda:' + str(args.gpu))
    model = reset_model(checkpoint,model)

    X1, Y1 = get_data(subj)

    X_train, Y_train = X1[:cutoff], Y1[:cutoff]
    X_test, Y_test = X1[300:], Y1[300:]
    model.fit(X_train, Y_train, epochs=200,
              batch_size=BATCH_SIZE, scheduler='cosine',
              validation_data=(X_val, Y_val), remember_best_column='valid_loss')
    suffix = '_s' + str(subj) + '_normal'
    model.epochs_df.to_csv(pjoin(outpath, 'epochs' + suffix + '.csv'))

    base_adapt_loss = model.evaluate(X_test[1+trial_num:], Y_test[1+trial_num:])
    base_adapt_loss = 100 * ((100-trial_num-1) * (1- base_adapt_loss["misclass"]))/(100-trial_num-1)

    print("Accuracy using normal adapt : " + str(base_adapt_loss))
    normal_adapt.append(base_adapt_loss)

    ### Baseline Accuracy
    X, Y = get_data(subjs[0])
    n_classes = 2
    in_chans = X.shape[1]
    # final_conv_length = auto ensures we only get a single output in the time dimension
    model = Deep4Net(in_chans=in_chans, n_classes=n_classes,
                    input_time_length=X.shape[2],
                    final_conv_length='auto').cuda()

    # Dummy train data to set up the model.
    X_train = np.zeros(X[:2].shape).astype(np.float32)
    Y_train = np.zeros(Y[:2].shape).astype(np.int64)

    checkpoint = torch.load(pjoin(modelpath, 'subj_' + str(subj) + '.pt'),
                            map_location='cuda:' + str(args.gpu))
    # Set up the model.
    model = reset_model(checkpoint,model)
    model.fit(X_train, Y_train, 0, BATCH_SIZE)

    X, Y = get_data(subj)
    X_test, Y_test = X[300:], Y[300:]
    test_loss = model.evaluate(X_test, Y_test)
    baseline_acc = 100 * (1 - test_loss["misclass"])

    test_loss = model.evaluate(X_test[1+trial_num:], Y_test[1+trial_num:])
    baseline_acc = 100 * ((100-trial_num-1) * (1 - test_loss["misclass"]))/(100-trial_num-1)

    print("Baseline Accuracy on subj " + str(subj) + " : "  + str(baseline_acc))
    baseline.append(baseline_acc)

    ### Accuracy on first N trials before adaptation
    First_N_loss = model.evaluate(X_test_N, Y_test_N)
    print("Wrong trials for first N: " + str(First_N_loss["misclass"] * (trial_num + 1)))

    ## Update the Model
    model = Deep4Net(in_chans=X1.shape[1], n_classes=n_classes,
                 input_time_length=X1.shape[2],
                 final_conv_length='auto').cuda()

    checkpoint = torch.load(pjoin(modelpath, 'subj_' + str(subj) + '.pt'),
                        map_location='cuda:' + str(args.gpu))
    model = reset_model(checkpoint,model)

    exp = model.fit(X_train_update, Y_train_update, epochs=TRAIN_EPOCH,
                batch_size=BATCH_SIZE, scheduler='cosine',
                validation_data=(X_val, Y_val), remember_best_column='valid_loss')
    suffix = '_s' + str(subj) + '_updated'
    model.epochs_df.to_csv(pjoin(outpath, 'epochs' + suffix + '.csv'))

    rememberer = exp.rememberer
    base_model_param = {
        'epoch': rememberer.best_epoch,
        'model_state_dict': rememberer.model_state_dict,
        'optimizer_state_dict': rememberer.optimizer_state_dict,
        'loss': rememberer.lowest_val
    }
    torch.save(base_model_param, pjoin(outpath, 'subj_{}.pt'.format(subj)))

    remaining_loss = model.evaluate(X1[301+trial_num:],Y1[301+trial_num:])

    remaining_loss = remaining_loss["misclass"] * (100 - trial_num - 1)

    print("Loss on Remaining trials: " + str(remaining_loss))

    total_acc = remaining_loss + First_N_loss["misclass"] * (trial_num + 1)
    total_acc = 100 * (100-trial_num-1 - remaining_loss)/(100-trial_num-1)

    print("Overall acc: " + str(total_acc))
    few_shot.append(total_acc)

# For all Subjects
for subj in range(1,55):
    
    store_subjs = [55 , 56 , 57, 58, 59, 60, 61 , 62, 63,64]
 
    ## Select Best subject based on trial

    if exclude:
        loadmodel = vae_phase_select_exclude.vae_select(subj,trials-1,args.datapath)
    else:
        loadmodel = vae_phase_select.vae_select(subj,trials-1,args.datapath)
    loadmodel.run()

    load_string = './trial_phase_lists/test_' + str(subj) + '_list.npy'
    load_string_2 = './trial_phase_lists/test_phase_' + str(subj) + '_list.npy'

    ## Load best X phases for adaptation
    with open(load_string, 'rb') as f:
        update_subjs = np.load(f)
        update_subjs = update_subjs[:100]
    print(update_subjs)
    with open(load_string_2, 'rb') as f:
        update_phases = np.load(f)
        update_phases = update_phases[:100]
    print(update_phases)
    
    ## Perform adaptation
    update_model(update_subjs,subj,trials-1,update_phases)


## Save results to an excel sheet
dict1 = {"Baseline": baseline, "Normal Adaptation": normal_adapt, "Few Shot Unsupervised": few_shot}
df = pd.DataFrame(data=dict1)
df.index += 1
print (df)

df.to_excel(outpath + '/Results.xlsx',index_label='Subject')
