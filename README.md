# Replication Package for the Paper "Personalized First Issue Recommender for Newcomers in Open Source Projects"

This repository contains a replication package for the paper titled "Personalized First Issue Recommender for Newcomers in Open Source Projects." This package includes a dataset of 68,858 issues from 100 GitHub projects, records of 123 manually labeled issue samples, and Python scripts for analyzing the data and evaluating models.

## Required Environment

We recommend setting up the required environment on a commodity Linux machine with at least 1 CPU Core, 8GB Memory, and 100GB empty storage space. Our experiments were conducted on an Ubuntu 20.04 server with two Intel Xeon Gold CPUs, 320GB memory, and 36TB RAID 5 Storage.

## Files and Replicating Results

We used the GFI-bot database and the GitHub GraphQL API to collect features of 68,858 candidate issues and restore historical states of resolvers of 11,615 FIs (first issues). As the dataset is too large for git, please download the data file separately from [Zenodo](https://zenodo.org/record/7915841#.ZFp3zexByCc) and place it under `ReplicationPackage/`. We will provide the Zenodo link in the paper upon acceptance.

The followings are the files and replicating results:

Dataset:

The raw data of newcomer-issue pairs' features are stored in `ReplicationPackage/data/dataset_{bertmodel}_{num}.pkl`, where {bertmodel} is one of the four BERT-based language models: SIMCSE, RoBERTa, CodeBERT, and BERTOverflow, corresponding to the dataset whose textual features are extracted by one of the four language models. And {num} is 0 to 19, corresponding to the 20 chronological folds.
The training sets of the GFI-Bot approach are contained in `ReplicationPackage/data/training_set_recgfi_simcse_{num}.pkl`.
`ReplicationPackage/data/newcomerdata.json` contains first issues' title and description and their resolvers' total commit number and number of commits in the latest month, and `ReplicationPackage/data/processeddata.pkl` contains the 37 developers' features for the empirical study.
`ReplicationPackage/data/isstexts.json` contains issues titles and descriptions for Stanik et al.'s approach.

Python scripts:

`ReplicationPackage/empirical.py` is the script for reproducing all the results in Section III of the paper.
`ReplicationPackage/model.py` is the script for reproducing all the results in Section IV of the paper. 

Records:

`ReplicationPackage/PFIs.csv` records the manually labeled issues for the empirical study.

Figures:

By running `ReplicationPackage/empirical.py` and `ReplicationPackage/model.py`, you can get all the figures in the fold `ReplicationPackage/figures/`. Besides the figures in the paper, `ReplicationPackage/figures/` also contains `typedis_{num}.png`, and `domaindis_{num}.png`, {num} is 1 to 4, representing additional results of newcomer features for Figure 4 in the paper.
