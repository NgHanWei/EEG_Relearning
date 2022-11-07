# Adaptive Few-Shot Unsupervised Machine Re-Learning for EEG Motor Imagery Classification via Deep Representation Learning

Codes for performing one/few-shot unsupervised adaptation of a subject-independent deep convolutional neural network (CNN) based electroencephalography (EEG)-BCI system for decoding hand motor imagery (MI). In the unsupervised adaptation, a few unlabelled MI trials are seen. Subsequently known data with the closest latent representations are selected and grouped together to finetune the pre-trained baseline model to improve performance towards a target subject.

## Results Overview

| Methodology | Mean (SD) | Median | Range (Max-Min) |
|-|-|-|-|
| Subject-Independent | 84.44 (11.93) | 86.32 | 42.11 (100-57.89) |
| Subject-Adaptive Deep CNN | 85.05 (11.20) | 86.32 | 40 (100-60) |
| Subject-Adaptive Siamese-VAE<br>(Including Extra Labels) | 85.82 (11.05) | 89.36 | 39.36 (100-60.64) |
| Subject-Adaptive Siamese-VAE<br>(Excluding Extra Labels) | 86.63 (11.79) | 90.10 | 38.54 (100-61.46) |

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

Pre-trained baseline models created using the proposed method is included under the folder `baseline_models`. The folder can be used to replace `$MODELPATH` when running the code for the EEG model re-learning. Otherwise, baseline models can be trained using either one of the two following methods.

#### Either:
Run normal `train_base.py` for all 54 subjects as per [Kaishuo et al.](https://github.com/zhangks98/eeg-adapt).

#### Or:
#### Obtaining train/validation split using unsupervised learning

To generate baseline models using subject selection method, run `get_list_vae.py`. The selection is based on the validation trials of the target subject, which is assumed to be known during the adaptation process to select the best adaptation performance.
```
usage: python get_list_vae.py [DATAPATH][-start START][-end END][-subj SUBJ][-trial TRIAL]

Obtain list of subjects to use as training data for subject-indepdendent baseline model

Positional Arguments:
    DATAPATH                            Datapath for the pre-processed EEG signals file

Optional Arguments:
    -start START                        Set start of range for subjects, minimum 1 and maximum 54
    -end END                            Set end of range for subjects, minimum 2 and maximum 55
    -subj SUBJ                          Set the subject number to run feature extraction on, will override the -start and -end functions if used
    -trial TRIAL                        Set the number of test trials from target subject to create baseline. Set number of trials to 0 to use target validation data
```

To run the code for learning best train/validation splits using target validation data:
```
python get_list_vae.py DATAPATH -trial 0
```

A list of 43 subjects with the closest latent representations to the target subject will be produced and saved. If validation data was used, lists will be saved in the `subj_lists` folder as `subj_\#Subj\_list.npy`. If target trial data was used, it will be saved in the `trial_lists` folder as `test_\#Subj\_list.npy`. In total there will be 54 subject lists corresponding to each subject of the EEG data.

#### Obtaining baseline models

Run `dual_train_custom.py`
```
usage: python dual_train_custom.py [DATAPATH] [OUTPATH] [LISTPATH] [-gpu GPU] [-start START] [-end END] [-subj SUBJ]

Training a custom subject-indepdendent baseline model using pre-determined training-validation split

Positional Arguments:
    DATAPATH                            Path for the pre-processed EEG signals file
    OUTPATH                             Path to folder for saving the trained model and results in
    LISTPATH                            Path to folder of lists, either `./subj_lists` or `./trial_lists` based on `get_list_vae.py`

Optional Arguments:
    -gpu GPU                            Set gpu to use, default is 0
    -start START                        Set start of range for subjects, minimum 1 and maximum 54
    -end END                            Set end of range for subjects, minimum 2 and maximum 55
    -subj SUBJ                          Set the subject number to run feature extraction on, will override the -start and -end functions if used
```

To run for all subjects after obtaining lists:
```
python dual_train_custom.py DATAPATH OUTPATH LISTPATH
```

Baseline models will be saved as `subj_#Subj.pt` in the `$OUTPATH` directory. The baseline models will be used for the adaptation.

## Performing Adaptation
With baseline models saved in `$MODELPATH` and pre-processed eeg file in `$DATAPATH`, run:

`dual_adapt_phase_while_test.py`
Runs adaptation for different number of trials. Saves output in `Results.xlsx` file the final test accuracy for each of the subjects across 3 categories: (1) Baseline (2) Baseline + normal adapt (3) Baseline + proposed adapt.

```
usage: python dual_adapt_phase_while_test.py [DATAPATH] [MODELPATH] [OUTPATH_ADAPT] [-scheme SCHEME] [-trfrate TRFRATE] [-lr LR] [-gpu GPU] [-start START] [-end END] [-subj SUBJ] [-trial TRIAL] [-exclude]

Perform adaptation on subject-independent baseline model 

Positional Arguments:
    DATAPATH                            Path for the pre-processed EEG signals file
    MODELPATH                           Path to folder containing the baseline classification models for adaptation
    OUTPATH_ADAPT                       Path to folder for saving the trained model and results in

Optional Arguments:
    -scheme SCHEME                      Model layer freezing scheme
    -trfrate TRFRATE                    The percentage of data for adaptation
    -lr LR                              Learning rate
    -gpu GPU                            Set gpu to use, default is 0
    -start START                        Set start of range for subjects, minimum 1 and maximum 54, default set at 1
    -end END                            Set end of range for subjects, minimum 2 and maximum 55, default set at 55
    -subj SUBJ                          Set the subject number to run feature extraction on, will override the -start and -end functions if used
    -trial TRIAL                        How many trials from target subject to use for adaptation
    -exclude                            Set data selection based on phases to either include or exclude additional labelled data, default included
```

An example command line to run the proposed method for 54 subjects excluding the additional target subject data at scheme 5 and 4 unlabelled target trials, while comparing against previous method which uses 100% of additional target subject adaptation data:
```
python dual_adapt_phase_while_test.py $DATAPATH $MODELPATH $OUTPATH_ADAPT -exclude
```

For testing on scheme 4 with 80% of adaptation data for both proposed and comparison method, with 1 unlabelled target trial for the proposed method:
``` 
python dual_adapt_phase_while_test.py $DATAPATH $MODELPATH $OUTPATH_ADAPT -scheme 4 -trfrate 80 -trial 1
```

## Folder structures
    .
    ├── ...
    ├── vae                             # Contains utility functions
    ├── vae_models                      # Contains the unsupervised representation learning technique (vanilla VAE is used for this)
    ├── $MODELPATH                      # Folder containing baseline models. Pre-trained baseline models are given in the folder `baseline_models`
    ├── trained_vae                     # Folder containing the trained VAE models if `get_list_vae.py` is used to generate baseline models
    ├── subj_lists                      # Folder containing npy files of 43 subjects closest to validation of target subject
    ├── trial_lists                     # Folder containing npy files of 43 subjects closest to test trial(s) of target subject
    ├── trial_phase_list                # Folder containing selected subject and phases when running main adaptation code
    │   ├── test_#Subj_list             # Subject index ranging from 1 to 54
    │   └── test_phase_#Subj_list       # Phase index ranging from 1 to 4
    ├── $OUTPUT_ADAPT                   # Folder containing the adapted model using unsupervised re-learning and the performance of the model while training
    │   ├── epochs_s#Subj_normal.csv    # Performance for the normal model
    │   ├── epochs_s#Subj_updated.csv   # Performance for the adaptation model
    │   └── subj_#Subj.pt               # Adapted model
    └── ...
