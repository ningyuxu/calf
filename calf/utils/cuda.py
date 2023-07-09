import time
import subprocess
from typing import List, Dict
import torch


def get_rng_state() -> Dict[str, torch.Tensor]:
    state = {"rng_state": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state()
    return state


def set_rng_state(state: Dict) -> None:
    torch.set_rng_state(state["rng_state"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda_rng_state"])


def get_free_gpus(gpu_mem_threshold: int = 4096,
                  max_gpus: int = 0,
                  wait: bool = False,
                  sleep_time: int = 30) -> List[int]:
    """
    Borrowed from https://gist.github.com/afspies/7e211b83ca5a8902849b05ded9a10696
    Args:
        gpu_mem_threshold (int):
            A GPU is considered free if the vram usage is no less than the threshold.
            Defaults to 4096(MiB).
        max_gpus (int):
            Max GPUs is the maximum number of gpus to assign. Defaults to 2.
        wait (bool):
            Whether to wait until a GPU is free. Defaults to False.
        sleep_time (int):
            Sleep time (in seconds) to wait before checking GPUs, if wait=True. Defaults to 10.
    Returns:
        A list of gpus with the largest available memory.
    """
    def _check():
        smi_query_result = subprocess.check_output("nvidia-smi -q -d Memory | grep -A4 GPU",
                                                   shell=True)
        gpu_info = smi_query_result.decode("utf-8").split("\n")
        total_mem = list(filter(lambda info: "Total" in info, gpu_info))
        total_mem = [int(x.split(":")[1].replace("MiB", "").strip()) for x in total_mem]
        used_mem = list(filter(lambda info: "Used" in info, gpu_info))
        used_mem = [int(x.split(":")[1].replace("MiB", "").strip()) for x in used_mem]
        free_mem = [total - used for total, used in zip(total_mem, used_mem)]
        available = [(i, mem) for i, mem in enumerate(free_mem) if mem >= gpu_mem_threshold]
        available_gpus = [a[0] for a in sorted(available, key=lambda k: k[1], reverse=True)]
        available_gpus = available_gpus[: min(max_gpus, len(available_gpus))]
        return available_gpus

    free_gpus = []
    if torch.cuda.is_available():
        while True:
            free_gpus = _check()
            if free_gpus or not wait:
                break
            print(f"No free GPUs found, retrying in {sleep_time}s ...")
            time.sleep(sleep_time)
    return free_gpus
