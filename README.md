

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


### Experiments

[2023-10-19] **NEW**: Measuring the syntactic difference between languages encoded in Multilingual BERT (mBERT, [`bert-base-multilingual-cased`](https://huggingface.co/bert-base-multilingual-cased)).

- Code is available at [`exps/cl_syntactic_diff_mbert`](./exps/cl_syntactic_diff_mbert).

- Dataset by default is Universal Dependencies (UD) [v2.10](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-4758).

- Specify the experiments by the configuration (`yaml`) files, which can be overridden by using command-line arguments.
  - Default settings have been specified in the main config [`exps/cl_syntactic_diff_mbert/config.yaml`](./exps/cl_syntactic_diff_mbert/config.yaml) and the config file for measurement [`exps/cl_syntactic_diff_mbert/syndiff_cfg.yaml`](./exps/cl_syntactic_diff_mbert/syndiff_cfg.yaml).
  - Set the pair of language for measurement (`lang1` and `lang2`) in [`exps/cl_syntactic_diff_mbert/config.yaml`](./exps/cl_syntactic_diff_mbert/config.yaml), together with the corpora needed for measurement (`corpora`) in [`exps/cl_syntactic_diff_mbert/syndiff_cfg.yaml`](./exps/cl_syntactic_diff_mbert/syndiff_cfg.yaml). The default languages are set to English (en) and Spanish (es).
  - Set the layer to use in [`exps/cl_syntactic_diff_mbert/syndiff_cfg.yaml`](./exps/cl_syntactic_diff_mbert/syndiff_cfg.yaml). The default is set to layer 7.

- Compute the syntactic difference between a pair of languages by running
      
      python -m exps.cl_syntactic_diff_mbert.run compute_syndiff



[2023-07-31] Probing mBERT for grammatical relations.

- Code is available at [`exps/cl_syntactic_diff_mbert`](./exps/cl_syntactic_diff_mbert).

- Dataset by default is Universal Dependencies (UD) [v2.10](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-4758).

- Specify the experiments by the configuration (`yaml`) files, which can be overridden by using command-line arguments.
  - Default settings have been specified in the main config [`exps/cl_syntactic_diff_mbert/config.yaml`](./exps/cl_syntactic_diff_mbert/config.yaml) and the config file for the probing experiment [`exps/cl_syntactic_diff_mbert/probe_cfg.yaml`](./exps/cl_syntactic_diff_mbert/probe_cfg.yaml).
  - Set the language for probing (`probe_lang`) in [`exps/cl_syntactic_diff_mbert/config.yaml`](./exps/cl_syntactic_diff_mbert/config.yaml), together with the corpora needed for probing (`corpora`) in [`exps/cl_syntactic_diff_mbert/probe_cfg.yaml`](./exps/cl_syntactic_diff_mbert/probe_cfg.yaml). The default language is set to English.
  - Set the layer to be probed in [`exps/cl_syntactic_diff_mbert/probe_cfg.yaml`](./exps/cl_syntactic_diff_mbert/probe_cfg.yaml). The default is set to layer 7.

- Probe mBERT layers by running
      
      python -m exps.cl_syntactic_diff_mbert.run probe


[2023-07-19] Zero-shot cross-lingual dependency parsing with Multilingual BERT.

- Code is available at [`exps/cl_syntactic_diff_mbert`](./exps/cl_syntactic_diff_mbert).

- Dataset by default is Universal Dependencies (UD) [v2.10](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-4758).

- Specify the experiments by the configuration (`yaml`) files, which can be overridden by using command-line arguments.
  - Default settings have been specified in the main config [`exps/cl_syntactic_diff_mbert/config.yaml`](./exps/cl_syntactic_diff_mbert/config.yaml) and the config file for training the parser [`exps/cl_syntactic_diff_mbert/parser_cfg.yaml`](./exps/cl_syntactic_diff_mbert/parser_cfg.yaml).
  - Set the source language (`src_lang`) and target languages (`tgt_langs`) in [`exps/cl_syntactic_diff_mbert/config.yaml`](./exps/cl_syntactic_diff_mbert/config.yaml)

- Fine-tune mBERT for dependency parsing by running

        python -m exps.cl_syntactic_diff_mbert.run parser --mode=train

- Test the mBERT fine-tuned on a source language by running

        python -m exps.cl_syntactic_diff_mbert.run parser --mode=test


## Acknowledgement

This repository is partly inspired by [Flair](https://github.com/flairNLP/flair) and [SuPar](https://github.com/yzhangcs/parser).

We use the [public source code](https://github.com/microsoft/otdd) of Alvarez-Melis and Fusi (2020) to compute the Optimal Transport Dataset Distance (OTDD) for our experiments in Xu et al. (2022).

## References

Alvarez-Melis, D., & Fusi, N. (2020). [Geometric Dataset Distances via Optimal Transport.](https://proceedings.neurips.cc/paper/2020/hash/f52a7b2610fb4d3f74b4106fb80b233d-Abstract.html) _Advances in Neural Information Processing Systems, 33_, 21428–21439. 

Xu, N., Gui, T., Ma, R., Zhang, Q., Ye, J., Zhang, M., & Huang, X. J. (2022). [Cross-Linguistic Syntactic Difference in Multilingual BERT: How Good is It and How Does It Affect Transfer?.](https://aclanthology.org/2022.emnlp-main.552/) In _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing_ (pp. 8073-8092).