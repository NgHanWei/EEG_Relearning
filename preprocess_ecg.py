import pandas as pd
import numpy as np
import wfdb
import ast
import matplotlib.pyplot as plt
import random
import argparse
import h5py
from os.path import join as pjoin

def load_raw_data(df,sampling_rate,path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

parser = argparse.ArgumentParser(
    description='Preprocessor for ECG Data')
parser.add_argument('source', type=str, help='Path to raw ECG data', default='D:/ecg_data/')
parser.add_argument('target', type=str, help='Path to pre-processed ECG data', default='D:/eeg_vrae/')
args = parser.parse_args()

out = args.target
path = args.source

sampling_rate = 100

# Load and convert annotation data
Y = pd.read_csv(path + 'ptbxl_database.csv',index_col = 'ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load raw signal data
X = load_raw_data(Y, sampling_rate, path)

# Load scp_statements.csv for dianostic aggregation
agg_df = pd.read_csv(path+'scp_statements.csv',index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

# Apply diagnostic superclass
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

print(X.shape)
print(Y.shape)

# Split data into train and Test
test_fold = 10
# Train
X_train = X[np.where(Y.strat_fold != test_fold)]
X_train = np.moveaxis(X_train, 2,1)
y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
# Test
X_test = X[np.where(Y.strat_fold == test_fold)]
X_test = np.moveaxis(X_test, 2,1)
y_test = Y[(Y.strat_fold == test_fold)].diagnostic_superclass

X = np.concatenate((X_train,X_test),axis=0)
Y_tmp = np.concatenate((y_train,y_test),axis=0)

Y = np.zeros(len(Y_tmp))

for i in range(0,len(Y_tmp)):
    if Y_tmp[i] == ['NORM']:
        Y[i] = 0
    if Y_tmp[i] == ['CD']:
        Y[i] = 1
    if Y_tmp[i] == ['MI']:
        Y[i] = 2
    if Y_tmp[i] == ['HYP']:
        Y[i] = 3
    if Y_tmp[i] == ['STTC']:
        Y[i] = 4

X = X.astype(np.float32)
Y = Y.astype(np.int64)
print(X.shape)
print(Y.shape)

# plt.imshow(X_train[0,:,:],cmap='gray', interpolation='nearest')
# plt.show()

# fig, axs = plt.subplots(12, sharex=True, sharey=True, gridspec_kw={'hspace': 0})
# fig.suptitle('Learned Latent Features')
# hex_colors = []
# for _ in range(0,12):
#     hex_colors.append('#%06X' % random.randint(0, 0xFFFFFF))
# colors = [hex_colors[int(i)] for i in range(0,12)]
# for i in range(0,12):
#     axs[i].plot(X_train[0,i,:], linewidth=3, color=colors[i])

# plt.show()

with h5py.File(pjoin(out, 'ecg_smt.h5'), 'w') as f:
    f.create_dataset('ecg/X', data=X)
    f.create_dataset('ecg/Y', data=Y)