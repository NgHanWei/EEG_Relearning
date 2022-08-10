# EEG_Relearning
Unsupervised One-Shot Adaptation through Model Re-Learning

## Overview

## Resources
Raw Dataset: [Link](http://gigadb.org/dataset/100542)

## Dependencies

It is recommended to create a virtual environment with python version 3.7 and activate it before running the following:

```
pip install -r requirements.txt
```

## Run

### Obtain the raw dataset

Download the raw dataset from the resources above, and save them to the same `$source` folder. To conserve space, you may only download files that ends with `EEG_MI.mat`.

### Pre-process the raw dataset

The following command will read the raw dataset from the `$source` folder, and output the pre-processed data `KU_mi_smt.h5` into the `$target` folder.

```
python preprocess_h5_smt.py $source $target
```

### Getting baseline models

#### Either: 
Run normal `train_base.py` for all 54 subjects.

#### Or:
To generate baseline models using subject selection method, run `get_list_vae.py`. The selection is based on the validation trials of the target subject, which is assumed to be known during the adaptation process to select the best adaptation performance.
```
usage: python get_list_vae.py [DATAPATH][-start START][-end END][-subj SUBJ][-trial TRIAL]

Arguments:
-datapath DATAPATH                  Datapath for the pre-processed EEG signals
-start START                        Set start of range for subjects, minimum 1 and maximum 54
-end END                            Set end of range for subjects, minimum 2 and maximum 55
-subj SUBJ                          Set the subject number to run feature extraction on, will override the -start and -end functions if used
-trial TRIAL (REQUIRED)             Set the number of test trials from target subject to create baseline. Set number of trials to 0 to use target validation data
```
A list of 43 subjects with the closest latent representations to the target subject will be produced and saved. If validation data was used, lists will be saved in the `subj_lists` folder as `subj_\#Subj\_list.npy`. If target trial data was used, it will be saved in the `trial_lists` folder as `test_\#Subj\_list.npy`. In total there will be 54 subject lists corresponding to each subject of the EEG data.

Run `dual_train_custom.py` 
```
usage: python dual_train_custom.py [DATAPATH][OUTPATH][-gpu GPU][-start START][-end END][-subj SUBJ]

Arguments:
-datapath DATAPATH                  Path for the pre-processed EEG signals
-outpath OUTPATH                    Path to save the trained model and results in
-listpath LISTPATH                  Path to lists, either `./subj_lists` or `./trial_lists` based on `get_list_vae.py`
-gpu GPU                            Set gpu to use, default is 0
-start START                        Set start of range for subjects, minimum 1 and maximum 54
-end END                            Set end of range for subjects, minimum 2 and maximum 55
-subj SUBJ                          Set the subject number to run feature extraction on, will override the -start and -end functions if used
```
Baseline models will be saved as `subj_\#Subj\.pt` in the `$outpath` directory.

### Running the code
With baseline models and pre-processed eeg file, run:

`dual_adapt_phase_while_test.py`
Runs adaptation for different number of trials. Saves in `Results.xlsx` file the final test accuracy for each of the subjects across 3 categories: (1) Baseline (2) Baseline + normal adapt (3) Baseline + proposed adapt.

To options for selecting adaptation:
```
`vae_phase_select.py`   #Selects based on phases, which phases of known subjects best suit target

`vae_subj_select.py`    #Selects based on overall subject, which subjects overall data best suit target
```

### Visualisation
`VAE_visualisation.py`
Visualisation of TSNE and PCA plots.

### Folder structures

`Trial_phase_list`
test_\#Subj\_list contains subject index (1:54)
test_phase_\#Subj\_list contains phase index (1:4)

`trained_vae`
Trained vaes

`adapt_models`
proposed adapted models

`results_adapt`
Epochs performance of proposed adapted

`$outpath$`
folder containing baseline SI model for each subject and training results of baseline SI model

`subj_list`
folder containing npy files of 43 subjects closest to validation of target subj, used to train baseline model using dual_train_custom.py

`trial_list`
folder containing npy files of 43 subjects closest to test trial(s) of target subj, used to train baseline model using dual_train_custom.py
