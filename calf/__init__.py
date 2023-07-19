import os
import sys
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Callable
from omegaconf import OmegaConf, DictConfig
import torch
from torch import distributed as dist
from torch import multiprocessing as mp
from calf.utils.cuda import get_free_gpus
from calf.utils.distributed import get_free_port, is_master
from calf.utils.log import get_logger, init_logger

__version__ = "1.0.0"

cfg = OmegaConf.create()
device = "cpu"
logger = get_logger()

ROOT = None
DATA = None
CORPUS = None
CACHE = None
MODEL = None
OUTPUT = None
CHECKPOINT = None
LOG = None


def get_root_path() -> Path:
    return Path(__file__).parent.parent.resolve()


def get_data_root() -> Path:
    data_root = get_root_path() / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    return data_root


def get_corpus_root() -> Path:
    corpus_root = get_data_root() / "corpus"
    corpus_root.mkdir(parents=True, exist_ok=True)
    return corpus_root


def get_cache_root() -> Path:
    cache_root = get_data_root() / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root


def get_model_root() -> Path:
    model_root = get_root_path() / "model"
    model_root.mkdir(parents=True, exist_ok=True)
    return model_root


def get_output_root(experiment: str) -> Path:
    output_root = get_root_path() / "output" / experiment
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


def get_checkpoint_path(experiment: str) -> Path:
    checkpoint_path = get_output_root(experiment) / "checkpoint"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    return checkpoint_path


def get_log_path(experiment: str) -> Path:
    log_path = get_output_root(experiment) / "log"
    log_path.mkdir(parents=True, exist_ok=True)
    return log_path


def get_calf_device(cfg: DictConfig) -> str:
    return f"cuda:{cfg.local_rank}" if cfg.world_size > 0 else "cpu"


def get_calf_logger(cfg: DictConfig) -> logging.Logger:
    log_file = str(get_log_path(cfg.experiment) / f"{cfg.command}_{cfg.local_rank}.log")
    return init_logger(logger=logger,
                       log_file=log_file if cfg.log_to_file else None,
                       mode='a' if cfg.checkpoint else 'w',
                       verbose=cfg.verbose,
                       non_master_level=logging.ERROR if cfg.log_master_only else logging.INFO)


def parse_config(cfg: DictConfig = None,
                 config_path: str = None,
                 config_file: str = None,
                 args: Dict = None) -> DictConfig:
    config = OmegaConf.create()
    if cfg is not None:
        config.merge_with(cfg)
    if config_path and config_file:
        file = Path(config_path) / config_file
        if file.is_file():
            config.merge_with(OmegaConf.load(str(file)))
    if args is not None:
        config.merge_with(args)
    # Assign variables their default values if they are not specified in the configuration or arguments
    config.nproc = config.get("nproc", 0)
    config.gid = config.get("gid", None)
    config.gpu_mem_threshold = config.get("gpu_mem_threshold", 9216)
    config.seed = config.get("seed", 25)
    config.threads = config.get("threads", 16)
    config.verbose = config.get("verbose", True)
    config.log_to_file = config.get("log_to_file", False)
    config.checkpoint = config.get("checkpoint", False)
    config.use_available_gpu = config.get("use_available_gpu", False)
    config.log_master_only = config.get("log_master_only", True)
    return config


def setup_os_environment() -> None:
    # disable parallelism to avoid deadlocks
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # disable transformers warning
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # debug, info, warning, error, critical
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    # disable warnings
    warnings.filterwarnings("ignore")


def compete_for_gpus(cfg: DictConfig) -> List:
    # setup parallel environment
    if cfg.gid:
        gpus_to_use = [cfg.gid]
    else:
        gpus_to_use = get_free_gpus(cfg.gpu_mem_threshold, cfg.nproc) if cfg.nproc > 0 else []
        assert (
                cfg.use_available_gpu or len(gpus_to_use) == cfg.nproc
        ), f"{cfg.nproc} GPU(s) required, but only {len(gpus_to_use)} available"
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(g) for g in gpus_to_use])
    return gpus_to_use


def run_experiment(experiment: str,
                   command: str,
                   config_path: str,
                   config_file: str,
                   callback: Callable = None,
                   **kwargs) -> None:
    # parse configuration
    args = locals()
    args.pop("callback")
    args.pop("kwargs")
    config_path = args.pop("config_path")
    config_file = args.pop("config_file")
    for k, v in kwargs.items():
        args[k] = v
    cfg = parse_config(config_path=config_path, config_file=config_file, args=args)
    # setup os environment
    setup_os_environment()
    # get available gpus
    gpus = compete_for_gpus(cfg)
    # run experiment
    world_size = len(gpus)
    if world_size > 1:
        os.environ["MASTER_ADDR"] = "tcp://127.0.0.1"
        os.environ["MASTER_PORT"] = get_free_port()
        mp.spawn(start, args=(world_size, callback, cfg), nprocs=world_size)
    else:
        start(gpus[0] if world_size == 1 else -1, world_size, callback, cfg)


def start(local_rank: int, world_size: int, fn: Callable, config: DictConfig) -> None:
    global cfg, device, logger
    global ROOT, DATA, CORPUS, CACHE, MODEL, OUTPUT, CHECKPOINT, LOG
    # init dist
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.threads)
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            init_method=f"{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
            world_size=world_size,
            rank=local_rank
        )
    config.local_rank = local_rank
    config.world_size = world_size
    # cfg
    cfg.merge_with(config)
    # paths
    ROOT = get_root_path()
    DATA = get_data_root()
    CORPUS = get_corpus_root()
    CACHE = get_cache_root()
    MODEL = get_model_root()
    OUTPUT = get_output_root(cfg.experiment)
    CHECKPOINT = get_checkpoint_path(cfg.experiment)
    LOG = get_log_path(cfg.experiment)
    # device
    device = get_calf_device(config)
    # logger
    logger = get_calf_logger(config)
    # print only master node
    if not is_master() and cfg.log_master_only:
        sys.stdout = open(os.devnull, 'w')
    # callback
    if fn is not None:
        fn(config)
