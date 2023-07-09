

<img src=docs/calf_logo.jpg align="right" width="100" height="100"/>

# Calf

<br clear="left"/>

A simple framework for deep learning.

Our primary objective is to enhance the efficiency of implementing deep learning experiments, preferably by utilizing a single Python file for each experiment.


## Usage


### Currently supported features and functionality

- Unified access to Huggingface models and tokenizers

- PyTorch dataset-based access to [Universal Dependencies (UD)](https://universaldependencies.org/) Treebanks

- [OmegaConf](https://omegaconf.readthedocs.io/en/2.3_branch/)-based hierarchical configuration system

- Automatic GPU discovery for distributed training and testing (with multiple GPUs on a single machine currently)

- A flexible event logging system for applications

- Automatically generating command line interfaces with a single line of code through [Fire](https://github.com/google/python-fire)

## Acknowledgement

This repository is partly inspired by [Flair](https://github.com/flairNLP/flair) and [SuPar](https://github.com/yzhangcs/parser).