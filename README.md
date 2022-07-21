# EEG_Relearning
Unsupervised One-Shot Adaptation through Model Re-Learning

## Overview

## Resources
Raw Dataset: Link[http://gigadb.org/dataset/100542]

## Dependencies

It is recommended to create a virtual environment with python version 3.7 and activate it before running the following:

```
pip install -r requirements.txt
```

## Run

### Obtain the raw dataset

Download the raw dataset from the resources above, and save them to the same folder. To conserve space, you may only download files that ends with EEG_MI.mat.

### Pre-process the raw dataset

The following command will read the raw dataset from the $source folder, and output the pre-processed data KU_mi_smt.h5 into the $target folder.

```
python preprocess_h5_smt.py $source $target
```

### Getting baseline models
Either: Run normal train_base.py for all 54 subjects.

Or: vae_subj_select.py --> Generate subj_list folder containing npy files using validation trials 200:300 or trial_list using target trials 300:300+X for closest 43 subject representations. 

Run dual_train_custom.py with subj_\#_list.npy containing each subjects' closest subjects for training baseline. Baseline models will be saved as subj_#.pt

### Running the code

dual_adapt_phase_while_test.py 
Runs adaptation for different number of trials. Saves in results.xlsx file (1) Baseline (2) Baseline + normal adapt (3) Baseline + proposed adapt.

vae_phase_select.py
Selects based on phases

vae_subj_select.py
Selects based on overall subject

VAE_visualisation.py
Visualisation of TSNE and PCA plots.

### Folder structures

Trial_phase_list
test_#_list contains subject index (1:54)
test_phase_#_list contains phase index (1:4)

trained_vae
Trained vaes

adapt_models
proposed adapted models

results_adapt
Epochs performance of proposed adapted

baseline_models
folder containing baseline SI model for each subject

results
folder containing training results of baseline SI model

subj_list
folder containing npy files of 43 subjects closest to validation of target subj, used to train baseline model using dual_train_custom.py
