from pathlib import Path
from omegaconf import DictConfig
from calf import logger
from calf.utils.corpus import (IterableCorpus,
                               UDTreebankCorpus,
                               SST2Corpus,
                               WikiText103Corpus)
from calf import parse_config


def log_corpus(corpus):
    logger.info(corpus.name)
    logger.info(len(corpus))
    logger.info(corpus.fields)
    for i, data in enumerate(corpus.data):
        if i == 0:
            data = dict(zip(corpus.fields, data))
            logger.info(f"{i}, {data}")
            break


def ud_corpus(cfg: DictConfig) -> None:
    cfg = parse_config(cfg=cfg,
                       config_path=str(Path(__file__).parent),
                       config_file="test_corpus.yaml")
    for corpus_params in cfg.ud_corpora:
        corpus = UDTreebankCorpus(lang=corpus_params.lang,
                                  genre=corpus_params.genre,
                                  split=corpus_params.split,
                                  reload=cfg.reload)
        log_corpus(corpus)


def iterable_corpus(cfg: DictConfig) -> None:
    iterable = ["Dan Morgan told himself he would forget Ann Turner.",
                "He was well rid of her.",
                "He certainly didn't want a wife who was fickle as Ann.",
                "If he had married her , he'd have been asking for trouble."]
    corpus = IterableCorpus(values=iterable)
    log_corpus(corpus)


def sst_corpus(cfg: DictConfig) -> None:
    cfg = parse_config(cfg=cfg,
                       config_path=str(Path(__file__).parent),
                       config_file="test_corpus.yaml")
    corpus = SST2Corpus(reload=cfg.reload)
    log_corpus(corpus)


def wiki_corpus(cfg: DictConfig) -> None:
    cfg = parse_config(cfg=cfg,
                       config_path=str(Path(__file__).parent),
                       config_file="test_corpus.yaml")
    corpus = WikiText103Corpus(reload=cfg.reload)
    log_corpus(corpus)
