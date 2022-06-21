from cmath import inf
from vrae.utils import *
from vae_models import dualchain_vae
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import pandas as pd
import sys
import torch
import torch.nn as nn
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

from braindecode.torch_ext.util import set_random_seeds
import h5py
set_random_seeds(seed=0, cuda=True)

parser = argparse.ArgumentParser(
    description='VAE Subject Selection')
parser.add_argument('-subj', type=int,
                    help='Target Subject for Feature Extraction', required=True)
parser.add_argument('-trial', type=int, default = -1,
                    help='Current Trial Number for Subject Selection', required=False)
parser.add_argument('-epochs', type=int, default= 100,
                    help='Number of Epochs', required=False)
parser.add_argument('-features', type=int, default= 16,
                    help='Number of Latent Features', required=False)
parser.add_argument('-alpha', type=float, default= 0.5,
                    help='Alpha Hyperparameter for First Chain KL Divergence', required=False)
parser.add_argument('-beta', type=float, default= 0.5,
                    help='Beta Hyperparameter for Second Chain KL Divergence', required=False)
parser.add_argument('-lr', type=float, default= 0.0005,
                    help='Set Learning Rate', required=False)
parser.add_argument('-clip', type=float, default= 0,
                    help='Set Gradient Clipping Threshold', required=False)
parser.add_argument('-loss', type=str, default= "default",
                    help='Sets the various lower bound loss functions to train DCVAE. Default - Default DCVAE loss. full - Entire reconstruction loss. indiv - Individual chain loss.', required=False)                    
parser.add_argument('-data', type=str, default= 'eeg',
                    help='Choose Type of Data: eeg or semg', required=False)
parser.add_argument('-datapath', type=str, help='Path to data',required=True)
parser.add_argument('-all', default=False, action='store_true')
parser.add_argument('-flow', default=False, action='store_true')
args = parser.parse_args()

if args.loss not in ['default', 'full', 'indiv']:
    parser.print_help()
    sys.exit(1)

targ_subj = args.subj
targ_trial = args.trial

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

# Randomly shuffled subject.
if args.data == 'eeg':
    datapath = args.datapath
else:
    datapath = args.datapath + '/semg_flexex_smt.h5'

dfile = h5py.File(datapath, 'r')
torch.cuda.set_device(0)
set_random_seeds(seed=20200205, cuda=True)

if args.data == 'eeg':

    X_train_all , y_train_all = get_multi_data([targ_subj])
    X_train_all = np.expand_dims(X_train_all,axis=1)

    if (args.all == False) and (targ_trial == -1):
        X_test , y_test = get_multi_data([targ_subj])
        X_test = np.expand_dims(X_test,axis=1)
        X_valid = X_train_all[200:300]
        X_test = X_train_all[200:300]
        X_train = X_train_all[200:300]
        y_train = y_test[200:300]
        y_test = y_test[200:300]

    elif (targ_trial >= 0) and (args.all == False):
        print("Running Trial Subject Selection")
        X_test , y_test = get_multi_data([targ_subj])
        X_test = np.expand_dims(X_test,axis=1)
        X_valid = X_train_all[300:301+targ_trial]
        X_test = X_train_all[300:301+targ_trial]
        X_train = X_train_all[300:301+targ_trial]
        y_train = y_test[300:301+targ_trial]
        y_test = y_test[300:301+targ_trial]

    else:
        # Data visualisation
        X_train = X_train_all
        X_test = X_train_all
        X_valid = X_train_all
        y_train = y_train_all
        y_test = y_train_all
## sEMG data
else:
    subject_list = list(range(1,41))
    subject_list.remove(targ_subj)
    if args.all == False:
        X_train, y_train = get_multi_data(subject_list[3:])
        X_valid, y_valid = get_multi_data(subject_list[:3])
        X_test, y_test = get_multi_data([targ_subj])
        X_train = np.expand_dims(X_train,axis=1)
        X_valid = np.expand_dims(X_valid,axis=1)
        X_test = np.expand_dims(X_test,axis=1)
    else:
        X_train, y_train = get_multi_data(subject_list)
        X_valid, y_valid = get_multi_data(subject_list)
        X_test, y_test = get_multi_data(subject_list)
        X_train = np.expand_dims(X_train,axis=1)
        X_valid = np.expand_dims(X_valid,axis=1)
        X_test = np.expand_dims(X_test,axis=1)

# print(X_train.shape)
X_train = torch.from_numpy(X_train)
X_train = X_train.to('cuda')
X_valid = torch.from_numpy(X_valid)
X_valid = X_valid.to('cuda')
X_test = torch.from_numpy(X_test)
X_test = X_test.to('cuda')

# VAE model
input_shape=(X_train.shape[1:])
batch_size = 16
kernel_size = 5
filters = 8
features = int(args.features / 2)
data_load = torch.split(X_train,batch_size)
channels = len(X_train[0,0,:,0])

print("Number of Features: " + str(features * 2))
if args.clip > 0:
    print("Gradient Clipping: " + str(args.clip))
else:
    print("No Gradient Clipping")

if args.data == 'eeg':
    print("Data Loaded: " + args.data)
else:
    print("Data Loaded: semg")

# leanring parameters
epochs = args.epochs
lr = 0.0005
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Model
model = dualchain_vae.DCVAE(filters=filters,channels=channels,features=features,data_type=args.data,flow=args.flow).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr,betas=(0.5, 0.999),weight_decay=0.5*lr)
criterion = nn.BCELoss(reduction='sum')

## Number of trainable params
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of trainable params: " + str(pytorch_total_params))

def recon_loss(outputs,targets):

    outputs = torch.flatten(outputs)
    targets = torch.flatten(targets)

    loss = nn.MSELoss()

    recon_loss = loss(outputs,targets)

    return recon_loss

def fit(model):
    model.train()
    running_loss = 0.0
    # For each batch
    for batch in tqdm(range(0,len(data_load))):
        optimizer.zero_grad()
        reconstruction, mu, logvar, new_inputs, reconstruction_2,mu_2,logvar_2 = model(data_load[batch])
        # print(reconstruction.shape)
        # print(X_train.shape)
        # bce_loss = criterion(reconstruction, X_train)
        bce_loss = recon_loss(reconstruction,data_load[batch])
        kl_1_loss = args.alpha * -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        mask_loss = recon_loss(reconstruction_2,new_inputs)
        full_loss = recon_loss(reconstruction+reconstruction_2,data_load[batch])
        kl_2_loss = args.beta * -0.5 * torch.sum(1 + logvar_2 - mu_2.pow(2) - logvar_2.exp())
        # loss = loss + mask_loss + bce_loss + kl_2_loss
        if args.loss == 'full':
            loss = kl_1_loss + kl_2_loss + full_loss
        elif args.loss == 'indiv':
            loss = kl_1_loss + kl_2_loss + bce_loss + mask_loss
        else:
            loss = kl_1_loss + mask_loss + kl_2_loss
        running_loss += loss.item()
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

    train_loss = running_loss/len(X_train)

    return train_loss

def validate(model,data):
    model.eval()
    running_loss = 0.0
    full_recon_loss = 0.0
    mask_recon_loss = 0.0
    with torch.no_grad():
        # For each image in batch
        # for batch in range(0,len(data_load)):
        reconstruction, mu, logvar, new_inputs, reconstruction_2,mu_2,logvar_2 = model(data)
        # bce_loss = criterion(reconstruction, X_train)
        bce_loss = recon_loss(reconstruction,data)
        kl_1_loss = args.alpha * -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        mask_loss = recon_loss(reconstruction_2,new_inputs)
        full_loss = recon_loss(reconstruction+reconstruction_2,data)
        kl_2_loss = args.beta * -0.5 * torch.sum(1 + logvar_2 - mu_2.pow(2) - logvar_2.exp())
        if args.loss == 'full':
            loss = kl_1_loss + kl_2_loss + full_loss
        elif args.loss == 'indiv':
            loss = kl_1_loss + kl_2_loss + bce_loss + mask_loss
        else:
            loss = kl_1_loss + mask_loss + kl_2_loss
        running_loss += loss.item()
        full_recon_loss += full_loss.item()
        mask_recon_loss += mask_loss.item()


    val_loss = running_loss/len(data)
    full_recon_loss = full_recon_loss/len(data)
    mask_recon_loss = mask_recon_loss/len(data)
    print(f"2nd Chain Recon Loss: {mask_recon_loss:.4f}")
    print(f"Full Recon Loss: {full_recon_loss:.4f}")
    return val_loss, full_recon_loss

# Save file name
file_name = "./dual_vae_torch" +  '_' + str(args.data) + '_' + str(targ_subj) + '_' + str(filters) + '_' + str(channels) + '_' + str(args.features) + ".pt"

train_loss = []
val_loss = []
eval_loss = []
recon_loss_array = []
best_val_loss = inf
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = fit(model)
    val_epoch_loss, full_recon_loss = validate(model,X_valid)
    eval_epoch_loss, _ = validate(model,X_test)

    #Save best model
    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        torch.save(model.state_dict(),file_name)
        print(f"Saving Model... Best Val Loss: {best_val_loss:.4f}")    
    
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    eval_loss.append(eval_epoch_loss)
    recon_loss_array.append(full_recon_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {val_epoch_loss:.4f}")


# Results
model = dualchain_vae.DCVAE(filters=filters,channels=channels,features=features,data_type=args.data,flow=args.flow).to(device)
model.load_state_dict(torch.load(file_name))

loss_list = []
for subj in range(1,55):
    X_test , y_test = get_multi_data([subj])
    X_test = np.expand_dims(X_test,axis=1)
    X_test = torch.from_numpy(X_test)
    X_test = X_test.to('cuda')
    model.eval()
    with torch.no_grad():
        reconstruction, mu, logvar, new_inputs, reconstruction_2,mu_2,logvar_2 = model(X_test)

    full_loss = recon_loss(reconstruction+reconstruction_2,X_test)

    loss_list.append(full_loss.item())

    print(full_loss)

sort_list = loss_list.copy()
sort_list.sort()

index_list = []
for subj in sort_list:
    sub_index = loss_list.index(subj) + 1
    index_list.append(sub_index)

index_list.remove(args.subj)

print(index_list[:43])

## Save Array
if targ_trial == -1:
    save_string = 'subj_' + str(args.subj) + '_list.npy'
else:
    save_string = 'test_' + str(args.subj) + '_list.npy'
with open(save_string, 'wb') as f:
    np.save(f, np.array(index_list[:43]))