import sys
from pathlib import Path
import fire
from omegaconf import DictConfig
from calf import run_experiment


def start(cfg: DictConfig):
    getattr(sys.modules[__name__], cfg.command)(cfg)


def main(command: str, config_path: str = None, config_file: str = None, **kwargs) -> None:
    run_experiment(
        experiment=Path(__file__).parent.stem,
        command=command,
        config_path=config_path,
        config_file=config_file,
        callback=start,
        # can add some more parameters and define default value here,
        # for example, we need to add four more parameters for llama
        # ckpt_dir=str(MODEL / "llama/7B"),
        # tokenizer_path=str(MODEL / "llama/tokenizer.model"),
        # temperature=0.8,
        # top_p=0.95
        # parameters will be merged into elephant.cfg, which can be accessed anywhere
        **kwargs
    )


if __name__ == "__main__":
    fire.Fire(main)
