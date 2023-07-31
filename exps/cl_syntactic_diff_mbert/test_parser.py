from pathlib import Path
from omegaconf import DictConfig

import json

from calf import parse_config, device, CHECKPOINT, LOG, logger
from calf.utils.log import log_line
from calf.utils.corpus import Corpus, UDTreebankCorpus
from calf.utils.dataset import Dataset, build_dataloader
from calf.utils import enumeration as enum

from .ud_transform import UDTreebankTransform
from .model import MBertDPModel
from .trainer.dp_trainer import DPTrainer


def train_parser(cfg: DictConfig, lang: str = "en"):
    cfg = parse_config(
        cfg=cfg, config_path=str(Path(__file__).parent), config_file="parser_cfg.yaml"
    )
    model = MBertDPModel(cfg.model).to(device)
    tokenizer = model.encoder.tokenizer
    transform = UDTreebankTransform(tokenizer=tokenizer)
    trainer = DPTrainer(cfg=cfg, transform=transform, model=model)
    dataset_cfg = {
        "lang": lang,
        "genre": cfg.all_langs.criteria.genres[lang][0],
    }
    outputs = trainer.train(dataset_cfg=dataset_cfg)
    results = outputs["result_dict"]
    log_path = LOG / cfg.exp_name / f"{lang}"
    log_path.mkdir(exist_ok=True)
    with open(log_path / f"results_{lang}.json", mode="a", encoding="utf-8") as fh:
        print(json.dumps(results), file=fh)


def test_parser(cfg: DictConfig, tgt_langs: list, src_lang: str = "en"):
    cfg = parse_config(
        cfg=cfg, config_path=str(Path(__file__).parent), config_file="parser_cfg.yaml"
    )
    model = MBertDPModel(cfg.model).to(device)
    checkpoint_path = CHECKPOINT / cfg.exp_name / src_lang

    # Fine-tune mBERT for dependency parsing if checkpoint does not exist
    if not (checkpoint_path / "best_model.pt").is_file():
        log_line(logger=logger)
        logger.info(f"Cannot find saved checkpoint of mbert_for_dependency_parsing. \nStart Training on {src_lang}.")
        log_line(logger=logger)
        train_parser(cfg=cfg, lang=src_lang)

    # Load the model fine-tuned for dependency parsing
    model.load_state_dict(model.load(checkpoint_path / "best_model.pt").state_dict())

    model.eval()
    tokenizer = model.encoder.tokenizer
    transform = UDTreebankTransform(tokenizer=tokenizer)

    log_path = LOG / cfg.exp_name / f"{src_lang}_zs"
    log_path.mkdir(exist_ok=True)

    log_line(logger=logger)
    logger.info(f"Start Testing [src_lang = {src_lang}, tgt_langs = {', '.join(tgt_langs)}].")
    log_line(logger=logger)

    for tgt_lang in tgt_langs:
        result_dict = dict()

        genre = cfg.all_langs.criteria.genres[tgt_lang][0]
        test_corpus = UDTreebankCorpus(lang=tgt_lang, genre=genre, split=enum.Split.TEST)
        dataloader, dataset, sampler = build_dataloader(
            corpus=test_corpus, transform=transform, max_len=cfg.transform.max_len,
            drop_last=False, batch_size=cfg.trainer.eval_batch_size, training=False
        )
        test_metric_result, test_loss_result = model.evaluate(dataloader=dataloader)

        log_line(logger=logger)
        logger.info(f"TEST ({src_lang} to {tgt_lang})")
        logger.info(test_metric_result.log_header)
        logger.info(test_metric_result.log_line)
        log_line(logger=logger)

        result_dict["test_las"] = test_metric_result.metric_detail["las"]
        result_dict["test_uas"] = test_metric_result.metric_detail["uas"]

        with open(log_path / f"results_{src_lang}_to_{tgt_lang}.json", mode="a", encoding="utf-8") as fh:
            print(json.dumps(result_dict), file=fh)

