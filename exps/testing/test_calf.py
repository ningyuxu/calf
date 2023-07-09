from omegaconf import DictConfig
from calf import logger, device
from calf import ROOT, DATA, CORPUS, CACHE, MODEL, OUTPUT, CHECKPOINT, LOG


def init(cfg: DictConfig) -> None:
    logger.info(f"root_path: {ROOT}")
    logger.info(f"data_root: {DATA}")
    logger.info(f"corpus_root: {CORPUS}")
    logger.info(f"cache_root: {CACHE}")
    logger.info(f"model_root: {MODEL}")
    logger.info(f"output_root: {OUTPUT}")
    logger.info(f"checkpoint_path: {CHECKPOINT}")
    logger.info(f"log_path: {LOG}")
    logger.info(f"local_rank: {cfg.local_rank}")
    logger.info(f"world_size: {cfg.world_size}")
    logger.info(f"device: {device}")
