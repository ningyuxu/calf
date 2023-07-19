from transformers import AutoConfig
from calf import cfg, MODEL


class HuggingfaceConfig:

    @staticmethod
    def from_params(model_name: str, **params):
        config_path = MODEL / f"huggingface/config/{model_name.replace('/', '--')}"
        config_path.mkdir(parents=True, exist_ok=True)
        if not any(config_path.iterdir()):
            config = AutoConfig.from_pretrained(model_name, **params)
            config.save_pretrained(str(config_path))
        else:
            config = AutoConfig.from_pretrained(str(config_path), **params)
        return config
