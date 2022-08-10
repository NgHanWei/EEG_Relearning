# EEG_Relearning
Adaptive Few-Shot Unsupervised Machine Re-Learning for EEG Motor Imagery Classification via Deep Representation Learning

## Overview

<!-- Insert Table of Results here -->

## Resources
Raw Dataset: [Link](http://gigadb.org/dataset/100542)

The pre-processing code of the dataset and a variation of training of the baseline models is provided by Kaishuo et al.: [Link](https://github.com/zhangks98/eeg-adapt)

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
Run normal `train_base.py` for all 54 subjects as per [Kaishuo et al.](https://github.com/zhangks98/eeg-adapt).

#### Or:
To generate baseline models using subject selection method, run `get_list_vae.py`. The selection is based on the validation trials of the target subject, which is assumed to be known during the adaptation process to select the best adaptation performance.
```
usage: python get_list_vae.py [DATAPATH][-start START][-end END][-subj SUBJ][-trial TRIAL]

Obtain list of subjects to use as training data for subject-indepdendent baseline model

Positional Arguments:
    DATAPATH                            Datapath for the pre-processed EEG signals

Optional Arguments:
    -start START                        Set start of range for subjects, minimum 1 and maximum 54
    -end END                            Set end of range for subjects, minimum 2 and maximum 55
    -subj SUBJ                          Set the subject number to run feature extraction on, will override the -start and -end functions if used
    -trial TRIAL                        Set the number of test trials from target subject to create baseline. Set number of trials to 0 to use target validation data
```
A list of 43 subjects with the closest latent representations to the target subject will be produced and saved. If validation data was used, lists will be saved in the `subj_lists` folder as `subj_\#Subj\_list.npy`. If target trial data was used, it will be saved in the `trial_lists` folder as `test_\#Subj\_list.npy`. In total there will be 54 subject lists corresponding to each subject of the EEG data.

Run `dual_train_custom.py`
```
usage: python dual_train_custom.py [DATAPATH] [OUTPATH] [LISTPATH] [-gpu GPU] [-start START] [-end END] [-subj SUBJ]

Training a custom subject-indepdendent baseline model using pre-determined training-validation split

Positional Arguments:
    DATAPATH                            Path for the pre-processed EEG signals
    OUTPATH                             Path to save the trained model and results in
    LISTPATH                            Path to lists, either `./subj_lists` or `./trial_lists` based on `get_list_vae.py`

Optional Arguments:
    -gpu GPU                            Set gpu to use, default is 0
    -start START                        Set start of range for subjects, minimum 1 and maximum 54
    -end END                            Set end of range for subjects, minimum 2 and maximum 55
    -subj SUBJ                          Set the subject number to run feature extraction on, will override the -start and -end functions if used
```
Baseline models will be saved as `subj_\#Subj\.pt` in the `$outpath` directory.

### Running the code
With baseline models and pre-processed eeg file, run:

`dual_adapt_phase_while_test.py`
Runs adaptation for different number of trials. Saves output in `Results.xlsx` file the final test accuracy for each of the subjects across 3 categories: (1) Baseline (2) Baseline + normal adapt (3) Baseline + proposed adapt.

```
usage: python dual_train_custom.py [DATAPATH] [MODELPATH] [OUTPATH_ADAPT] [-scheme SCHEME] [-trfrate TRFRATE] [-lr LR] [-gpu GPU] [-start START] [-end END] [-subj SUBJ] [-trial TRIAL]

Perform adaptation on subject-independent baseline model 

Positional Arguments:
    DATAPATH                            Path for the pre-processed EEG signals
    MODELPATH                           Path containing the baseline classification models for adaptation
    OUTPATH_ADAPT                       Path to save the trained model and results in

Optional Arguments:
    -scheme SCHEME                      Model layer freezing scheme
    -trfrate TRFRATE                    The percentage of data for adaptation
    -lr LR                              Learning rate
    -gpu GPU                            Set gpu to use, default is 0
    -start START                        Set start of range for subjects, minimum 1 and maximum 54
    -end END                            Set end of range for subjects, minimum 2 and maximum 55
    -subj SUBJ                          Set the subject number to run feature extraction on, will override the -start and -end functions if used
    -trial TRIAL                        How many trials from target subject to use for adaptation
```

Options for selecting adaptation:
```
`vae_phase_select.py`           Selects based on phases including the additional labelled data, which phases of known subjects best suit target

vae_phase_select_exclude.py`    Selects based on phases excluding the additional labelled data, which phases of known subjects best suit target

`vae_subj_select.py`            Selects based on overall subject, which subjects excluding target subject best suit target
```

### Visualisation
`VAE_visualisation.py`
Visualisation of TSNE and PCA plots.

### Folder structures

`Trial_phase_list`
test_\#Subj\_list contains subject index (1:54)

test_phase_\#Subj\_list contains phase index (1:4) from vae_phase_select.py or from vae_subj_select_exclude.py,,

`trained_vae`
Trained vaes on each subject validation or test trials from get_list_vae.py

`$output_adapt`
Epochs performance of proposed adapted and proposed adapted models for dual_adapt_phase_while_test.py

`$outpath`
folder containing baseline SI model for each subject and training results of baseline SI model for dual_train_custom.py

`subj_lists`
folder containing npy files of 43 subjects closest to validation of target subj, from vae_subj_select.py, used to train baseline model using dual_train_custom.py

`trial_lists`
folder containing npy files of 43 subjects closest to test trial(s) of target subj, from vae_subj_select.py, used to train baseline model using dual_train_custom.py