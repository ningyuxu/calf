import sys
import fire
from pathlib import Path
from omegaconf import DictConfig
from calf import run_experiment


def calf(cfg: DictConfig):
    if cfg.module == "init":
        from .test_calf import init
        init(cfg)


def corpus(cfg: DictConfig):
    if cfg.module == "ud":
        from .test_corpus import ud_corpus
        ud_corpus(cfg)
    elif cfg.module == "iter":
        from .test_corpus import iterable_corpus
        iterable_corpus(cfg)
    elif cfg.module == "sst":
        from .test_corpus import sst_corpus
        sst_corpus(cfg)
    elif cfg.module == "wiki":
        from .test_corpus import wiki_corpus
        wiki_corpus(cfg)


def bert(cfg: DictConfig):
    if cfg.module == "init":
        from .test_bert import init
        init(cfg)
    elif cfg.module == "transform":
        from .test_bert import transform
        transform(cfg)
    elif cfg.module == "dataset":
        from .test_bert import dataset
        dataset(cfg)
    elif cfg.module == "tokenize":
        from .test_bert import tokenize
        tokenize(cfg)
    elif cfg.module == "embed":
        from .test_bert import embed
        embed(cfg)


def pythia(cfg: DictConfig):
    if cfg.module == "init":
        from .test_pythia import init
        init(cfg)
    elif cfg.module == "transform":
        from .test_pythia import transform
        transform(cfg)
    elif cfg.module == "dataset":
        from .test_pythia import dataset
        dataset(cfg)
    elif cfg.module == "tokenize":
        from .test_pythia import tokenize
        tokenize(cfg)
    elif cfg.module == "embed":
        from .test_pythia import embed
        embed(cfg)


def llama(cfg: DictConfig):
    if cfg.module == "init":
        from .test_llama import init
        init(cfg)
    elif cfg.module == "transform":
        from .test_llama import transform
        transform(cfg)
    elif cfg.module == "dataset":
        from .test_llama import dataset
        dataset(cfg)
    elif cfg.module == "tokenize":
        from .test_llama import tokenize
        tokenize(cfg)
    elif cfg.module == "embed":
        from .test_llama import embed
        embed(cfg)


def baichuan(cfg: DictConfig):
    if cfg.module == "init":
        from .test_baichuan import init
        init(cfg)
    elif cfg.module == "transform":
        from .test_baichuan import transform
        transform(cfg)
    elif cfg.module == "dataset":
        from .test_baichuan import dataset
        dataset(cfg)
    elif cfg.module == "tokenize":
        from .test_baichuan import tokenize
        tokenize(cfg)
    elif cfg.module == "embed":
        from .test_baichuan import embed
        embed(cfg)


def start(cfg: DictConfig):
    getattr(sys.modules[__name__], cfg.command)(cfg)


def main(command: str, config_path: str = None, config_file: str = None, **kwargs) -> None:
    run_experiment(
        experiment=Path(__file__).parent.stem,
        command=command,
        config_path=config_path if config_path else str(Path(__file__).parent),
        config_file=config_file if config_file else "config.yaml",
        callback=start,
        **kwargs
    )


if __name__ == "__main__":
    fire.Fire(main)
