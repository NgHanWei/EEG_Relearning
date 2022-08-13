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

class vae_select():
    def __init__(self, subj, trial, datapath, epochs = 100, features = 16, alpha = 0.5, beta = 0.5, lr = 0.0005, clip = 0, loss = 'default', data = 'eeg', all=False, flow=False):
        super(vae_select, self).__init__()

        self.subj = subj
        self.trial = trial
        self.datapath = datapath
        self.epochs = epochs
        self.features = features
        self.alpha = alpha
        self.beta = beta
        self.lr = lr
        self.clip = clip
        self.loss = loss
        self.data = data
        self.all = all
        self.flow = flow

    def run(self):

        targ_subj = self.subj
        targ_trial = self.trial

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
        datapath = self.datapath

        dfile = h5py.File(datapath, 'r')
        torch.cuda.set_device(0)
        set_random_seeds(seed=20200205, cuda=True)

        # Obtain Data
        if self.data == 'eeg':

            X_train_all , y_train_all = get_multi_data([targ_subj])
            X_train_all = np.expand_dims(X_train_all,axis=1)

            if (self.all == False) and (targ_trial == -1):
                X_test , y_test = get_multi_data([targ_subj])
                X_test = np.expand_dims(X_test,axis=1)
                X_valid = X_train_all[200:300]
                X_test = X_train_all[200:300]
                X_train = X_train_all[200:300]
                y_train = y_test[200:300]
                y_test = y_test[200:300]

            elif (targ_trial >= 0) and (self.all == False):
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
            if self.all == False:
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
        features = self.features

        data_load = torch.split(X_train,batch_size)
        channels = len(X_train[0,0,:,0])

        print("Number of Features: " + str(features))
        if self.clip > 0:
            print("Gradient Clipping: " + str(self.clip))
        else:
            print("No Gradient Clipping")

        if self.data == 'eeg':
            print("Data Loaded: " + self.data)
        else:
            print("Data Loaded: semg")


        # learning parameters
        epochs = self.epochs
        lr = 0.0005
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        ## Define Model
        model = vanilla_vae.VanillaVAE(filters=filters,channels=channels,features=features,data_type=self.data,data_length=len(data_load[0][:,0,0,0])).to(device)
        # print(model)
        optimizer = optim.Adam(model.parameters(), lr=lr,betas=(0.5, 0.999),weight_decay=0.5*lr)
        criterion = nn.BCELoss(reduction='sum')

        ## Number of trainable params
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Total number of trainable params: " + str(pytorch_total_params))

        # Save file name
        if os.path.exists("./trained_vae/") == False:
            os.makedirs("./trained_vae/")
        file_name = "./trained_vae/vae_torch" +  '_' + str(self.data)  + '_' + str(targ_subj) + '_' + str(filters) + '_' + str(channels) + '_' + str(features) + ".pt"

        def recon_loss(outputs,targets):

            outputs = torch.flatten(outputs)
            targets = torch.flatten(targets)

            loss = nn.MSELoss()

            recon_loss = loss(outputs,targets)

            return recon_loss

        def final_loss(bce_loss, mu, logvar):
            BCE = bce_loss 
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return BCE + KLD

        def fit(model):
            model.train()
            running_loss = 0.0
            # For each batch
            for batch in tqdm(range(0,len(data_load))):
                optimizer.zero_grad()
                reconstruction, mu, logvar = model(data_load[batch])
                bce_loss = recon_loss(reconstruction,data_load[batch])
                loss = final_loss(bce_loss, mu, logvar)
                running_loss += loss.item()
                loss.backward()
                if self.clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip)
                optimizer.step()

            train_loss = running_loss/len(X_train)

            return train_loss

        def validate(model):
            model.eval()
            running_loss = 0.0
            full_recon_loss = 0.0
            with torch.no_grad():
                # For each image in batch
                # for batch in range(0,len(X_valid)):
                reconstruction, mu, logvar = model(X_valid)
                bce_loss = recon_loss(reconstruction,X_valid)
                loss = final_loss(bce_loss, mu, logvar)
                running_loss += loss.item()
                full_recon_loss += bce_loss.item()

            val_loss = running_loss/len(X_valid)
            full_recon_loss = full_recon_loss/len(X_valid)
            print(f"Recon Loss: {full_recon_loss:.4f}")
            return val_loss, full_recon_loss

        def eval(model):
            model.eval()
            nll_loss_eval = 0
            with torch.no_grad():
                reconstruction, mu, logvar = model(X_test)
                recon_1 = recon_loss(reconstruction,X_test)
                KLD_1 = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                nll_loss_eval = (recon_1 + KLD_1)/len(X_test)
                return nll_loss_eval.item()

        # Train the Model
        train_loss = []
        val_loss = []
        eval_loss = []
        best_val_loss = inf
        for epoch in range(epochs):
            print(f"Epoch {epoch+1} of {epochs}")
            train_epoch_loss = fit(model)
            val_epoch_loss, full_recon_loss = validate(model)
            eval_epoch_loss = eval(model)
            
            train_loss.append(train_epoch_loss)
            val_loss.append(val_epoch_loss)
            eval_loss.append(eval_epoch_loss)

            #Save best model
            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                torch.save(model.state_dict(),file_name)
                print(f"Saving Model... Best Val Loss: {best_val_loss:.4f}")

            print(f"Train Loss: {train_epoch_loss:.4f}")
            print(f"Val Loss: {val_epoch_loss:.4f}")

        model = vanilla_vae.VanillaVAE(filters=filters,channels=channels,features=features,data_type=self.data,data_length=len(data_load[0][:,0,0,0])).to(device)
        model.load_state_dict(torch.load(file_name))

        # Subject selection
        loss_list = []
        for subj in range(1,55):
            X_test , y_test = get_multi_data([subj])
            X_test = np.expand_dims(X_test,axis=1)
            X_test = torch.from_numpy(X_test)
            X_test = X_test.to('cuda')
            model.eval()
            with torch.no_grad():
                reconstruction, mu, logvar = model(X_test)

            full_loss = recon_loss(reconstruction,X_test)

            loss_list.append(full_loss.item())

            print(full_loss)

        sort_list = loss_list.copy()
        sort_list.sort()

        # Get subjet index list
        index_list = []
        for subj in sort_list:
            sub_index = loss_list.index(subj) + 1
            index_list.append(sub_index)

        # Remove Current Subject
        index_list.remove(self.subj)

        print(index_list[:43])

        ## Save Array
        if targ_trial == -1:
            if os.path.exists("./subj_lists/") == False:
                os.makedirs("./subj_lists/")
            save_string = './subj_lists/subj_' + str(self.subj) + '_list.npy'
        else:
            if os.path.exists("./trial_lists/") == False:
                os.makedirs("./trial_lists/")
            save_string = './trial_lists/test_' + str(self.subj) + '_list.npy'
        with open(save_string, 'wb') as f:
            np.save(f, np.array(index_list[:43]))