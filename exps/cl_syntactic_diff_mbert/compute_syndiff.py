from omegaconf import DictConfig
from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader as TorchDataLoader

from calf import OUTPUT, CHECKPOINT, LOG, device, logger, parse_config
from .model import MBertDPModel
from .utils.hdf5_dataset import HDF5Dataset
from calf.utils import enumeration as enum
from calf.utils.log import log_line

from .otdd.pytorch.utils import random_index_split
from .otdd.pytorch.distance import DatasetDistance


def compute_dist(cfg: DictConfig, src_lang: str, tgt_lang: str):
    cfg = parse_config(
        cfg=cfg, config_path=str(Path(__file__).parent), config_file="syndiff_cfg.yaml"
    )
    # get hdf5 directory where embeddings are saved
    model_name = f"{cfg.model.encoder.name}_{cfg.model.encoder.get('revision', 'main')}"
    hdf5_path = OUTPUT / "embedding" / model_name
    hdf5_path.mkdir(parents=True, exist_ok=True)

    # load model
    model = MBertDPModel(cfg.model).to(device)
    if cfg.model.encoder.finetune:
        checkpoint_path = CHECKPOINT / cfg.exp_name / cfg.src_lang
        assert (checkpoint_path / "best_model.pt").is_file(), \
            f"Cannot find saved checkpoint of mbert_for_dependency_parsing at {checkpoint_path}."
        model.load_state_dict(model.load(checkpoint_path / "best_model.pt").state_dict())
    model.eval()

    # save embeddings to hdf5 file
    log_line(logger=logger)
    logger.info("Saving embeddings to hdf5_file.")
    for corpus_params in cfg.corpora:
        hdf5_file = hdf5_path / f"{corpus_params.lang}-{corpus_params.genre}-{corpus_params.split}.hdf5"
        if not hdf5_file.is_file():
            model.predict(corpus_params=corpus_params)
    log_line(logger=logger)

    # set log_path
    log_path = LOG / cfg.exp_name
    log_path.mkdir(exist_ok=True)

    # compute OTDD
    result_dict = dict()
    log_line(logger=logger)
    logger.info(f"Computing OTDD [src = {src_lang}, tgt = {tgt_lang}]")

    src_dataset_cfg = {
        "lang": src_lang,
        "genre": cfg.all_langs.criteria.genres[src_lang][0],
        "split": enum.Split.TRAIN
    }
    tgt_dataset_cfg = {
        "lang": tgt_lang,
        "genre": cfg.all_langs.criteria.genres[tgt_lang][0],
        "split": enum.Split.TRAIN
    }
    src_hdf5 = HDF5Dataset(cfg=cfg, hdf5_cfg=src_dataset_cfg, layer_index=cfg.model.encoder.embedding.layer).dataset
    tgt_hdf5 = HDF5Dataset(cfg=cfg, hdf5_cfg=tgt_dataset_cfg, layer_index=cfg.model.encoder.embedding.layer).dataset

    src = load_dataset(src_hdf5, maxsize=cfg.maxsize, shuffle=True)
    tgt = load_dataset(tgt_hdf5, maxsize=cfg.maxsize, shuffle=True)

    dist = DatasetDistance(src['train'], tgt['train'], inner_ot_method='exact',
                           debiased_loss=True, inner_ot_debiased=True,
                           p=2, inner_ot_p=2, entreg=cfg.ot_entreg, inner_ot_entreg=cfg.inner_ot_entreg,
                           device=device)

    d = dist.distance(maxsamples=cfg.maxsamples, return_coupling=False).cpu().numpy().item()
    label_dist = dist.label_distances.cpu().numpy().tolist()
    src_classes = dist.classes1
    tgt_classes = dist.classes2
    result_dict[f"{src_lang}_to_{tgt_lang}"] = dict()
    result_dict[f"{src_lang}_to_{tgt_lang}"]["otdd"] = d
    result_dict[f"{src_lang}_to_{tgt_lang}"]["label_dist"] = label_dist
    result_dict[f"{src_lang}_to_{tgt_lang}"]["src_classes"] = src_classes
    result_dict[f"{src_lang}_to_{tgt_lang}"]["tgt_classes"] = tgt_classes
    log_line(logger=logger)

    with open(log_path / f"results_{src_lang}.json", mode="a", encoding="utf-8") as fh:
        print(json.dumps(result_dict), file=fh)

    if cfg.delete_hdf5_after_probing:
        # delete hdf5 file after obtaining the probing results
        for corpus_params in cfg.corpora:
            hdf5_file = hdf5_path / f"{corpus_params.lang}-{corpus_params.genre}-{corpus_params.split}.hdf5"
            hdf5_file.unlink(missing_ok=True)


def load_dataset(
        data, valid_size=0, splits=None, maxsize=2000,
        shuffle=True, batch_size=16, num_workers=0,
):
    fold_idxs = {}
    if splits is None and valid_size == 0:
        fold_idxs['train'] = np.arange(len(data))
    elif splits is None and valid_size > 0:
        train_idx, valid_idx = random_index_split(
            len(data), 1 - valid_size, (maxsize, None)
        )  # No maxsize for validation
        fold_idxs['train'] = train_idx
        fold_idxs['valid'] = valid_idx

    for k, idxs in fold_idxs.items():
        if maxsize and maxsize < len(idxs):
            fold_idxs[k] = np.sort(np.random.choice(idxs, maxsize, replace=False))

    sampler_class = SubsetRandomSampler if shuffle else SubsetSampler
    fold_samplers = {k: sampler_class(idxs) for k, idxs in fold_idxs.items()}
    dataloader_args = dict(batch_size=batch_size, num_workers=num_workers)
    fold_loaders = {k: TorchDataLoader(data, sampler=sampler, **dataloader_args)
                    for k, sampler in fold_samplers.items()}
    return fold_loaders


class SubsetSampler(torch.utils.data.Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)
