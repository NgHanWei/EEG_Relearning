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
import json
import logging
import sys
from os.path import join as pjoin
import os

import dcvae_subj_select

import numpy as np
import h5py
import torch
import torch.nn.functional as F
from braindecode.models.deep4 import Deep4Net
from braindecode.torch_ext.optimizers import AdamW
from braindecode.torch_ext.util import set_random_seeds
from torch import nn

# python dual_adapt_while_test.py D:/DeepConvNet/pre-processed/KU_mi_smt.h5 D:/adapt_eeg/baseline_models D:/adapt_eeg/results_adapt -scheme 5 -trfrate 10 -subj $subj

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
parser.add_argument('-subj', type=int,
                    help='Target Subject for Subject Selection', required=True)

args = parser.parse_args()
datapath = args.datapath
outpath = args.outpath
modelpath = args.modelpath
scheme = args.scheme
rate = args.trfrate
lr = args.lr
dfile = h5py.File(datapath, 'r')
torch.cuda.set_device(args.gpu)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
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

X, Y = get_data(subjs[0])
n_classes = 2
in_chans = X.shape[1]
# final_conv_length = auto ensures we only get a single output in the time dimension
model = Deep4Net(in_chans=in_chans, n_classes=n_classes,
                 input_time_length=X.shape[2],
                 final_conv_length='auto').cuda()

# Deprecated.


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


def reset_model(checkpoint):
    # Load the state dict of the model.
    model.network.load_state_dict(checkpoint['model_state_dict'])

    # # Resets the last conv block
    # reset_conv_pool_block(model.network, block_nr=4)
    # reset_conv_pool_block(model.network, block_nr=3)
    # reset_conv_pool_block(model.network, block_nr=2)
    # # Resets the fully-connected layer.
    # # Parameters of newly constructed modules have requires_grad=True by default.
    # n_final_conv_length = model.network.conv_classifier.kernel_size[0]
    # n_prev_filter = model.network.conv_classifier.in_channels
    # n_classes = model.network.conv_classifier.out_channels
    # model.network.conv_classifier = nn.Conv2d(
    #     n_prev_filter, n_classes, (n_final_conv_length, 1), bias=True)
    # nn.init.xavier_uniform_(model.network.conv_classifier.weight, gain=1)
    # nn.init.constant_(model.network.conv_classifier.bias, 0)

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
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.compile(loss=F.nll_loss, optimizer=optimizer,
                  iterator_seed=20200205, )

cutoff = int(rate * 200 / 100)
# Use only session 1 data for training
assert(cutoff <= 200)

fold = args.subj
subj = args.subj

total_loss = []
suffix = '_s' + str(subj) + '_f' + str(fold)

checkpoint = torch.load(pjoin(modelpath, 'subj_' + str(fold) + '.pt'),
                    map_location='cuda:' + str(args.gpu))
reset_model(checkpoint)

def update_model(update_subjs,subj,trial_num):

    if subj in update_subjs:
        # update_subjs.remove(subj)
        # print("ADAPTING WITH TRAINING SESS")
        np.delete(update_subjs,np.where(update_subjs == subj))
        
        X, Y = get_multi_data(update_subjs)
        X2,Y2 = get_data(subj)
        X_train, Y_train = X[:], Y[:]

        ## Adaptation On
        # X_train = np.concatenate((X_train, X2[:cutoff]), axis=0)
        # Y_train = np.concatenate((Y_train,Y2[:cutoff]),axis=0)
    else:
        X, Y = get_multi_data(update_subjs)
        X_train, Y_train = X[:], Y[:]

    # X_val, Y_val = X[200:300], Y[200:300]
    if trial_num < 100:
        X1, Y1 = get_data(subj)
        X_val, Y_val = X1[200:300], Y1[200:300]
        X_test, Y_test = X1[300+trial_num:301+trial_num], Y1[300+trial_num:301+trial_num]

    ## Adaptation on Baseline
    # if trial_num == 0:
    checkpoint = torch.load(pjoin(modelpath, 'subj_' + str(fold) + '.pt'),
                    map_location='cuda:' + str(args.gpu))
    reset_model(checkpoint)
    X1, Y1 = get_data(subj)
    model.fit(X1[:cutoff], Y1[:cutoff], epochs=200,
        batch_size=BATCH_SIZE, scheduler='cosine',
        validation_data=(X_val, Y_val), remember_best_column='valid_loss')
    print("If normal adaptation on baseline: " + str(model.evaluate(X1[300:],Y1[300:])))

    baseline_loss = model.evaluate(X1[300:],Y1[300:])
    baseline_loss = 100 * (1 - baseline_loss["misclass"])

    f = open("dualadapt_baseline.txt", "a")
    f.write(f"{baseline_loss}\n")
    f.close()

    # if trial_num == 0:
    ## Accuracy on the first N trials
    checkpoint = torch.load(pjoin(modelpath, 'subj_' + str(fold) + '.pt'),
                    map_location='cuda:' + str(args.gpu))
    reset_model(checkpoint)
    X1, Y1 = get_data(subj)
    X_dummy = np.zeros(X[:2].shape).astype(np.float32)
    Y_dummy = np.zeros(Y[:2].shape).astype(np.int64)
    model.fit(X_dummy, Y_dummy, 0, BATCH_SIZE)
    ## Evaluate accuracy of first N trials
    test_loss = model.evaluate(X_test, Y_test)
    # total_loss.append(test_loss["misclass"])
    print("Accuracy on first N trial no adaptation: " + str(test_loss["misclass"]))

    ## Update the Model
    checkpoint = torch.load(pjoin(modelpath, 'subj_' + str(fold) + '.pt'),
                        map_location='cuda:' + str(args.gpu))
    reset_model(checkpoint)

    model.fit(X_train, Y_train, epochs=TRAIN_EPOCH,
                batch_size=BATCH_SIZE, scheduler='cosine',
                validation_data=(X_val, Y_val), remember_best_column='valid_loss')
    model.epochs_df.to_csv(pjoin(outpath, 'epochs' + suffix + '.csv'))

    remaining_loss = model.evaluate(X1[301+trial_num:],Y1[301+trial_num:])

    remaining_loss = remaining_loss["misclass"] * (100 - trial_num - 1)

    print("Accuracy on Remaining trials: " + str(remaining_loss))

    total_acc = remaining_loss + test_loss["misclass"] * (trial_num + 1)

    print("Overall acc: " + str(100 - total_acc))
    f = open("dualadapt.txt", "a")
    f.write(f"{100-total_acc}\n")
    f.close()

# For all Subjects
for subj in range(1,55):
    store_subjs = [55 , 56 , 57, 58, 59, 60, 61 , 62, 63,64]
    for trial in range(2,3):


        ## Select Best subject based on trial
        loadmodel = dcvae_subj_select.dcvae_select(subj, trial, args.datapath)
        loadmodel.run()
        load_string = 'test_' + str(subj) + '_list.npy'

        ## Load best 15 subjects for adaptation
        with open(load_string, 'rb') as f:
            update_subjs = np.load(f)
            update_subjs = update_subjs[:15]

        ## Test Trial data
        X1, Y1 = get_data(subj)
        X_test, Y_test = X1[300+trial:301+trial], Y1[300+trial:301+trial]

        ## Check if newly selected subjects same as previous
        same_elements = True
        for element in update_subjs:
            if element not in store_subjs:
                same_elements = False

        ## If not the same, update the model again, else remains the same
        if same_elements == False:
            update_model(update_subjs,subj,trial)

        ## Perform inference
        # if trial > 0:
        #     test_loss = model.evaluate(X_test, Y_test)
        #     total_loss.append(test_loss["misclass"])

        # store_subjs = update_subjs
        # print(total_loss)