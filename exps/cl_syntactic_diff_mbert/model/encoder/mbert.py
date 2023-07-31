from typing import Dict, List
import h5py

import numpy as np
import torch
import torch.nn as nn

from calf.utils.log import progress_bar

from calf.utils.corpus import UDTreebankCorpus
from calf.utils.dataset import build_dataloader
from calf.modules import HuggingfaceConfig, HuggingfaceModel, HuggingfaceTokenizer
from calf import device, OUTPUT, logger

from exps.cl_syntactic_diff_mbert.ud_transform import UDTreebankTransform


upos_vocab = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON",
              "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]
deprel_vocab = ["acl", "advcl", "advmod", "amod", "appos", "aux", "case", "cc", "ccomp", "clf",
                "compound", "conj", "cop", "csubj", "dep", "det", "discourse", "dislocated",
                "expl", "fixed", "flat", "goeswith", "iobj", "list", "mark", "nmod", "nsubj",
                "nummod", "obj", "obl", "orphan", "parataxis", "punct", "reparandum", "root",
                "vocative", "xcomp"]


class MBertEncoder(nn.Module):
    def __init__(self, encoder_cfg):
        super(MBertEncoder, self).__init__()
        self.encoder_cfg = encoder_cfg

        self.tokenizer = self.get_tokenizer()
        self.pretrained_model = self.get_pretrained_model()
        self.freeze_pretrained_model(self.encoder_cfg.freeze_layer)

    @property
    def embedding_dim(self) -> int:
        return self.pretrained_model.config.hidden_size

    def get_tokenizer(self):
        return HuggingfaceTokenizer(model_name=self.encoder_cfg.name,
                                    trust_remote_code=True,
                                    bos=True, eos=True,
                                    revision=self.encoder_cfg.get("revision", None))

    def get_config(self):
        return HuggingfaceConfig.from_params(model_name=self.encoder_cfg.name,
                                             output_hidden_states=True,
                                             output_attentions=False,
                                             return_dict=True,
                                             trust_remote_code=True,
                                             max_seq_len=self.encoder_cfg.max_seq_len,
                                             revision=self.encoder_cfg.get("revision", None))

    def get_pretrained_model(self):
        config = self.get_config()
        model = HuggingfaceModel(model_name=self.encoder_cfg.name,
                                 model_type="MaskedLM",
                                 n_layers=self.encoder_cfg.n_layers,
                                 trust_remote_code=True,
                                 max_seq_len=self.encoder_cfg.max_seq_len,
                                 finetune=self.encoder_cfg.finetune,
                                 torch_dtype=torch.float32,
                                 revision=self.encoder_cfg.get("revision", None),
                                 config=config)
        # model.apply(model._init_weights)
        model.to(device)
        return model

    def freeze_pretrained_model(self, freeze_layer: int = -1) -> None:
        if freeze_layer == -1:
            return  # do not freeze
        if freeze_layer >= 0:
            for i in range(freeze_layer + 1):
                if i == 0:
                    self._freeze_embedding_layer()
                else:
                    self._freeze_encoder_layer(i)

    def activate_pretrained_model(self) -> None:
        for i in range(self.num_hidden_layers + 1):
            if i == 0:
                self._freeze_embedding_layer(freeze=False)
            else:
                self._freeze_encoder_layer(layer=i, freeze=False)

    def _freeze_embedding_layer(self, freeze: bool = True) -> None:
        freeze_module(self.pretrained_model.embeddings, freeze=freeze)

    def _freeze_encoder_layer(self, layer: int, freeze: bool = True) -> None:
        freeze_module(self.pretrained_model.encoder.layer[layer - 1], freeze=freeze)

    # --------------------------------------------------------------------------------------------
    # encode data
    # --------------------------------------------------------------------------------------------
    def embed(self, batch: Dict) -> Dict:
        """
        Add word embedding
        """
        # lang = batch["lang"]
        # genre = batch["genre"]
        # split = batch["split"]
        # seqid = batch["seqid"]
        tokens = batch["form"]
        length = batch["num_tokens"]
        token_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        # token_type_ids = batch["token_type_ids"]
        postag_ids = batch["postag_ids"]
        head_ids = batch["head_ids"]
        deprel_ids = batch["deprel_ids"]
        segment_spans = batch["segment_spans"]
        output = self.pretrained_model.backend(
            input_ids=token_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids
        )
        hidden_states = torch.stack(output["hidden_states"])
        embedding = hidden_states[self.encoder_cfg.embedding.layer]

        batch_with_embedding = {
            # "lang": lang,
            # "genre": genre,
            # "split": split,
            # "seqid": seqid,
            "tokens": tokens,
            "length": length,
            "token_ids": token_ids,
            "attention_mask": attention_mask,
            # "token_type_ids": token_type_ids,
            "postag_ids": postag_ids,
            "head_ids": head_ids,
            "deprel_ids": deprel_ids,
            "embedding": embedding,
            "segment_spans": segment_spans,
        }

        return batch_with_embedding

    def predict(self, corpus_params):
        # for corpus_params in cfg.corpora:
        ud_corpus = UDTreebankCorpus(lang=corpus_params.lang,
                                     genre=corpus_params.genre,
                                     split=corpus_params.split)
        transform = UDTreebankTransform(tokenizer=self.tokenizer)
        dataloader, _, _ = build_dataloader(corpus=ud_corpus,
                                            transform=transform,
                                            cache=True,
                                            batch_size=self.encoder_cfg.predict_batch_size)
        hdf5_file = get_embedding_file(self.encoder_cfg, corpus_params)
        # if Path(hdf5_file).is_file():
        #     continue
        sid = 0
        for batch in progress_bar(
                logger=logger,
                iterator=dataloader,
                desc=f'{self.encoder_cfg.name}-'
                     f'{corpus_params["lang"]}-'
                     f'{corpus_params["genre"]}-'
                     f'{corpus_params["split"]}'
        ):
            with torch.no_grad():
                output = self.pretrained_model.backend(batch["input_ids"], batch["attention_mask"])
                hidden_states = torch.stack(output["hidden_states"])
                hidden_states = torch.permute(hidden_states, (1, 0, 2, 3))
                for idx in range(len(batch["num_tokens"])):
                    length_s = batch["num_tokens"][idx]
                    tokens_s = batch["form"][idx][:length_s]
                    token_ids_s = batch["input_ids"][idx][:length_s]
                    postag_ids_s = batch["postag_ids"][idx][:length_s]
                    head_ids_s = batch["head_ids"][idx][:length_s]
                    deprel_ids_s = batch["deprel_ids"][idx][:length_s]
                    hidden_states_s = hidden_states[idx][:, :length_s]
                    sid += 1
                    result = {"seqid": sid,
                              "length": length_s,
                              "tokens": tokens_s,
                              "token_ids": token_ids_s,
                              "postag_ids": postag_ids_s,
                              "head_ids": head_ids_s,
                              "deprel_ids": deprel_ids_s,
                              "embeddings": hidden_states_s}
                    with h5py.File(hdf5_file, 'a') as f:
                        save_as_hdf5(result, f)


def freeze_module(module: torch.nn.Module, freeze=True):
    """
    Freezes all weights of the model.
    """
    for param in module.parameters():
        param.requires_grad = not freeze


def get_embedding_file(model_params, corpus_params):
    model = f"{model_params.name}_{model_params.get('revision', 'main')}"
    hdf5_path = OUTPUT / "embedding" / model
    hdf5_path.mkdir(parents=True, exist_ok=True)
    # corpus_signture = {"lang": corpus_params.lang,
    #                    "genre": corpus_params.genre,
    #                    "split": corpus_params.split}
    # hdf5_file = f"{get_signature(corpus_signture)}.hdf5"
    hdf5_file = f"{corpus_params.lang}-{corpus_params.genre}-{corpus_params.split}.hdf5"
    return str(hdf5_path / hdf5_file)


def save_as_hdf5(output, fp):
    grp = fp.create_group(str(output["seqid"]))
    for k, v in output.items():
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
            grp.create_dataset(k, data=v)
        elif isinstance(v, str):
            dt = h5py.string_dtype(encoding="utf-8")
            grp.create_dataset(k, data=v, dtype=dt)
        elif isinstance(v, int):
            grp.create_dataset(k, data=v)
        else:
            assert (
                isinstance(v, list) or isinstance(v, tuple) or isinstance(v, np.ndarray)
            ), "Unsupported data type."
            if isinstance(v[0], str):
                v = [w.encode() for w in v]
                dt = h5py.special_dtype(vlen=str)
                grp.create_dataset(k, shape=(len(v),), dtype=dt, data=v)
            else:
                v = np.array(v)
                grp.create_dataset(k, data=v)


def get_label_ids(vocab, values) -> List:
    stoi = {s: i for i, s in enumerate(vocab)}
    return torch.tensor([stoi[v] for v in values]).long()
