from dataclasses import dataclass
import torch
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM
from transformers.configuration_utils import PretrainedConfig
from calf import MODEL
from calf.utils.enumeration import Enumberation
from ..scalar_mix import ScalarMix


@dataclass
class ModelType(Enumberation):
    CausalLM: str = "CausalLM"
    MaskedLM: str = "MaskedLM"


class HuggingfaceModel(torch.nn.Module):
    """
    Args:
        name (str):
            Path or name of the pretrained models registered, e.g., ``'bert-base-cased'``.
        n_layers (int):
            The number of model layers to use. If 0, uses all layers.
        n_out (int):
            The requested size of the embeddings. If 0, uses the size of the pretrained embedding
            model. Default: 0.
        stride (int):
            A sequence longer than max length will be split into several small pieces with a
            window size of ``stride``. Default: 10.
        mix_dropout (float):
            The dropout ratio of model layers. This value will be passed into the
            :class:`ScalarMix` layer. Default: 0.
        finetune (bool):
            If ``True``, the model parameters will be updated together with the downstream task.
            Default: ``False``.
    """

    def __init__(self,
                 model_name: str,
                 model_type: str,
                 revision: str = None,
                 trust_remote_code: bool = False,
                 n_layers: int = 1,
                 n_out: int = 0,
                 stride: int = 256,
                 mix_dropout: float = .0,
                 max_seq_len: int = 512,
                 finetune: bool = False,
                 config: PretrainedConfig = None,
                 **kwargs) -> None:
        super().__init__()

        self.name = model_name.replace('/', '--')

        # init backend model
        self.backend = self.init_model(model_name=model_name,
                                       model_type=model_type,
                                       cache_dir=str(MODEL / "huggingface/model"),
                                       revision=revision,
                                       trust_remote_code=trust_remote_code,
                                       config=config,
                                       **kwargs)
        self.backend = self.backend.requires_grad_(finetune)

        self.revision = revision

        self.n_layers = n_layers or self.backend.config.num_hidden_layers

        # hidden size
        if hasattr(self.backend.config, "hidden_size"):
            self.hidden_size = self.backend.config.hidden_size
        elif hasattr(self.backend.config, "d_model"):
            self.hidden_size = self.backend.config.d_model
        else:
            self.hidden_size = 0

        # output dimension
        self.n_out = n_out if n_out else self.hidden_size

        self.mix_dropout = mix_dropout
        self.finetune = finetune

        # max sequence length
        if hasattr(self.backend.config, "max_position_embeddings"):
            self.max_len = self.backend.config.max_position_embeddings - 2
        elif hasattr(self.backend.config, "max_seq_len"):
            self.max_len = self.backend.config.max_seq_len - 2
        elif hasattr(self.backend.config, "n_positions"):
            self.max_len = self.backend.config.n_positions - 2
        else:
            self.max_len = max_seq_len - 2

        self.stride = min(stride, self.max_len)

        self.scalar_mix = ScalarMix(self.n_layers, mix_dropout)

        if self.hidden_size == self.n_out:
            self.projection = torch.nn.Identity()
        else:
            self.projection = torch.nn.Linear(in_features=self.hidden_size,
                                              out_features=self.n_out,
                                              bias=False)

    @staticmethod
    def init_model(model_name: str,
                   model_type: str,
                   cache_dir: str,
                   revision: str = "main",
                   trust_remote_code: bool = False,
                   config: PretrainedConfig = None,
                   **kwargs):
        assert (
            model_type in [ModelType.CausalLM, ModelType.MaskedLM]
        ), f"Unrecoginzed model type {model_type}"

        AutoModelClass = {ModelType.CausalLM: AutoModelForCausalLM,
                          ModelType.MaskedLM: AutoModelForMaskedLM}[model_type]
        try:
            model = AutoModelClass.from_pretrained(model_name,
                                                   local_files_only=True,
                                                   cache_dir=cache_dir,
                                                   revision=revision,
                                                   trust_remote_code=trust_remote_code,
                                                   config=config,
                                                   **kwargs)
        except Exception:
            model = AutoModelClass.from_pretrained(model_name,
                                                   local_files_only=False,
                                                   cache_dir=cache_dir,
                                                   revision=revision,
                                                   trust_remote_code=trust_remote_code,
                                                   config=config,
                                                   **kwargs)
        return model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # return the hidden states of all layers
        x = self.backend(input_ids=input_ids[:, :self.max_len],
                         attention_mask=attention_mask[:, :self.max_len].float())["hidden_states"]
        # [batch_size, max_len, hidden_size]
        x = self.scalar_mix(x[-self.n_layers:])
        # [batch_size, n_tokens, hidden_size]
        for i in range(
                self.stride,
                (input_ids.shape[1]-self.max_len+self.stride-1)//self.stride*self.stride+1,
                self.stride
        ):
            part = self.backend(
                input_ids=input_ids[:, i:i+self.max_len],
                attention_mask=attention_mask[:, i:i+self.max_len].float()
            )["hidden_states"]
            x = torch.cat(
                (x, self.scalar_mix(part[-self.n_layers:])[:, self.max_len-self.stride:]), 1)
        return self.projection(x)
