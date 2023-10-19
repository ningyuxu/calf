import sys
import fire
from pathlib import Path
from omegaconf import DictConfig
from calf import run_experiment


def parser(cfg: DictConfig):
    from .test_parser import train_parser, test_parser
    if cfg.mode == "train":
        train_parser(cfg=cfg, lang=cfg.src_lang)
    elif cfg.mode == "test":
        test_parser(cfg=cfg, src_lang=cfg.src_lang, tgt_langs=cfg.tgt_langs)


def probe(cfg: DictConfig):
    from .probe import probing
    probing(cfg=cfg, lang=cfg.probe_lang)


def compute_syndiff(cfg: DictConfig):
    from .compute_syndiff import compute_dist
    compute_dist(cfg=cfg, src_lang=cfg.lang1, tgt_lang=cfg.lang2)


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
