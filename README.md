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

# Obtain the raw dataset

Download the raw dataset from the resources above, and save them to the same folder. To conserve space, you may only download files that ends with EEG_MI.mat.

# Pre-process the raw dataset

The following command will read the raw dataset from the $source folder, and output the pre-processed data KU_mi_smt.h5 into the $target folder.

```
python preprocess_h5_smt.py $source $target
```
