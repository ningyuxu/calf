from omegaconf import DictConfig
from pathlib import Path
import json

import numpy as np
import torch
from sklearn.linear_model import SGDClassifier

from calf import OUTPUT, CHECKPOINT, LOG, device, logger, parse_config
from .model import MBertDPModel
from .utils.hdf5_dataset import HDF5Dataset
from calf.utils import enumeration as enum
from calf.utils.log import log_line


def get_probe_acc(
        lang: str,
        cfg: DictConfig,
        alpha_list=np.logspace(-9, -2, num=8, base=10),
):
    train_dataset_cfg = {
        "lang": lang,
        "genre": cfg.all_langs.criteria.genres[lang][0],
        "split": enum.Split.TRAIN
    }
    test_dataset_cfg = {
        "lang": lang,
        "genre": cfg.all_langs.criteria.genres[lang][0],
        "split": enum.Split.TEST
    }
    train_dataset = HDF5Dataset(cfg=cfg, hdf5_cfg=train_dataset_cfg, layer_index=cfg.model.encoder.embedding.layer)
    test_dataset = HDF5Dataset(cfg=cfg, hdf5_cfg=test_dataset_cfg, layer_index=cfg.model.encoder.embedding.layer)
    data_train = train_dataset.data
    data_test = test_dataset.data

    train_X = torch.tensor(np.array(data_train["representations"]), dtype=torch.float)
    train_Y = np.array(data_train["labels"])

    test_X = torch.tensor(np.array(data_test["representations"]), dtype=torch.float)
    test_Y = np.array(data_test["labels"])

    accs = []

    for alpha in alpha_list:
        clf = SGDClassifier(
            loss="log",
            alpha=alpha,
            max_iter=10000,
            shuffle=True,
            verbose=False,
            random_state=25,
            early_stopping=True
        )
        clf.fit(train_X, train_Y)
        train_acc = np.mean(clf.predict(train_X) == train_Y)
        test_acc = np.mean(clf.predict(test_X) == test_Y)

        if cfg.verbose:
            logger.info('[ alpha = %f ] train acc: %f  test acc: %f' % (alpha, train_acc.item(), test_acc.item()))
        accs.append(test_acc)
    return accs


def probing(cfg: DictConfig, lang: str):
    cfg = parse_config(
        cfg=cfg, config_path=str(Path(__file__).parent), config_file="probe_cfg.yaml"
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

    # probing
    result_dict = dict()
    log_line(logger=logger)
    logger.info(f"Probing [lang = {lang}]")
    accs = get_probe_acc(lang=lang, cfg=cfg)
    result_dict[lang] = accs
    log_line(logger=logger)

    with open(log_path / f"results_{lang}.json", mode="a", encoding="utf-8") as fh:
        print(json.dumps(result_dict), file=fh)

    if cfg.delete_hdf5_after_probing:
        # delete hdf5 file after obtaining the probing results
        for corpus_params in cfg.corpora:
            hdf5_file = hdf5_path / f"{corpus_params.lang}-{corpus_params.genre}-{corpus_params.split}.hdf5"
            hdf5_file.unlink(missing_ok=True)
