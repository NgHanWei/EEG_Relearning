#!/usr/bin/env python
# coding: utf-8
'''Subject-specific classification with KU Data,
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

import numpy as np
import h5py
import torch
import torch.nn.functional as F
from braindecode.models.deep4 import Deep4Net
from braindecode.torch_ext.optimizers import AdamW
from braindecode.torch_ext.util import set_random_seeds

# python train_custom.py D:/DeepConvNet/pre-processed/KU_mi_smt.h5 D:/eeg-adapt/results -subj $subj

logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                    level=logging.INFO, stream=sys.stdout)
parser = argparse.ArgumentParser(
    description='Subject-specific classification with KU Data')
parser.add_argument('datapath', type=str, help='Path to the h5 data file')
parser.add_argument('outpath', type=str, help='Path to the result folder')
parser.add_argument('listpath', type=str, help='Path to the list folder')
parser.add_argument('-gpu', type=int,
                    help='The gpu device index to use', default=0)
parser.add_argument('-start', type=int,
                    help='Start of the subject index', default=1)
parser.add_argument(
    '-end', type=int, help='End of the subject index (not inclusive)', default=55)
parser.add_argument('-subj', type=int, nargs='+',
                    help='Explicitly set the subject number. This will override the start and end argument')
args = parser.parse_args()

datapath = args.datapath
outpath = args.outpath
listpath = args.listpath
start = args.start
end = args.end
assert(start < end)
subjs = args.subj if args.subj else range(start, end)
dfile = h5py.File(datapath, 'r')
torch.cuda.set_device(args.gpu)
set_random_seeds(seed=20200205, cuda=True)


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

def get_multi_data_custom(subjs):
    Xs = []
    Ys = []
    X1s = []
    Y1s = []
    for s in subjs:
        x, y = get_data(s)
        Xs.append(x[100:])
        Ys.append(y[100:])
        X1s.append(x[:100])
        Y1s.append(y[:100])
    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    X2 = np.concatenate(X1s, axis = 0)
    Y2 = np.concatenate(Y1s,axis=0)
    return X, Y, X2, Y2

for subj in subjs:

    ## Load Chosen Subjects
    save_string = listpath + '/subj_' + str(args.subj[0]) + '_list.npy'
    with open(save_string, 'rb') as f:
        list = np.load(f)

    # Select unchosen subjects to be validation
    # list = [19, 44, 15, 2, 21, 31, 41, 49, 52, 53, 33, 23, 18, 13, 43, 11, 20, 7, 16, 29, 42, 46, 25, 6, 8, 10, 22, 47, 45, 14, 28, 27, 4, 12, 36, 39, 35, 26, 24, 48, 38, 3, 1]
    valid_list = []
    train_list = []
    for list_i in range(1,55):
        if list_i not in list:
            valid_list.append(list_i)
    train_list = list
    valid_list.remove(subjs[0])
    print(valid_list)

    # Get data for within-subject classification
    X, Y = get_data(subj)
    X2, Y2 = get_multi_data(train_list)
    Valid_X, Valid_Y = get_multi_data(valid_list)
    # Valid_X, Valid_Y = X[200:300], Y[200:300]
    # X2,Y2 = X2[200:] , Y2[200:]
    X_train, Y_train = X2, Y2
    # print(Y_train)

    # X_train = np.concatenate((X_train, X2), axis=0)
    # Y_train = np.concatenate((Y_train,Y2),axis=0)
    # print(X_train.shape)
    # print(Y_train.shape)

    # X_val, Y_val =  X[200:300], Y[200:300]
    X_val, Y_val = Valid_X, Valid_Y
    X_test, Y_test = X[300:], Y[300:]

    suffix = 's' + str(subj)
    n_classes = 2
    in_chans = X.shape[1]

    # final_conv_length = auto ensures we only get a single output in the time dimension
    model = Deep4Net(in_chans=in_chans, n_classes=n_classes,
                     input_time_length=X.shape[2],
                     final_conv_length='auto').cuda()

    # these are good values for the deep model
    optimizer = AdamW(model.parameters(), lr=1 * 0.01, weight_decay=0.5*0.001)
    model.compile(loss=F.nll_loss, optimizer=optimizer, iterator_seed=1, )

    exp = model.fit(X_train, Y_train, epochs=200, batch_size=8, scheduler='cosine',
              validation_data=(X_val, Y_val), remember_best_column='valid_loss')

    rememberer = exp.rememberer
    base_model_param = {
        'epoch': rememberer.best_epoch,
        'model_state_dict': rememberer.model_state_dict,
        'optimizer_state_dict': rememberer.optimizer_state_dict,
        'loss': rememberer.lowest_val
    }
    torch.save(base_model_param, pjoin(outpath, 'subj_{}.pt'.format(subj)))

    test_loss = model.evaluate(X_test, Y_test)
    print(test_loss["misclass"])
    model.epochs_df.to_csv(pjoin(outpath, 'epochs_' + suffix + '.csv'))
    with open(pjoin(outpath, 'test_subj_' + str(subj) + '.json'), 'w') as f:
        json.dump(test_loss, f)

dfile.close()
