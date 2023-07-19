from typing import Iterable, Dict, List
import itertools
import numpy as np
import torch
from calf import cfg
from calf.modules.huggingface import HuggingfaceTokenizer
from calf.utils.tensor import pad
from calf.utils.corpus import Corpus
from calf.utils.transform import Transform


class UDTreebankTransform(Transform):

    def __init__(self, tokenizer: HuggingfaceTokenizer = None) -> None:
        super().__init__(name="doword_ud_transform")

        self.tokenizer = tokenizer
        if not self.tokenizer.pad_token:
            special_tokens = {"pad_token": tokenizer.eos_token}
            self.tokenizer.add_special_tokens(special_tokens)

    def load(self, corpus: Corpus) -> Iterable[Dict]:
        fields = corpus.fields
        for data in corpus.data:
            text = dict(zip(fields, data))
            yield text

    def transform(self, dataset: Iterable[Dict]) -> Iterable[Dict]:
        for data in dataset:
            doc_id = data["doc_id"]
            par_id = data["par_id"]
            sent_id = data["sent_id"]
            form = data["form"]
            upos = data["upos"]
            head = data["head"]
            deprel = data["deprel"]
            t_result = self.tokenizer(form)
            mask_for_labels = self._get_special_tokens_mask(token_ids=t_result["input_ids"])
            segment_spans = t_result["segment_spans"]
            head_ids = self._update_head_for_subword(head, segment_spans)
            head_ids = self.tokenizer.backend.build_inputs_with_special_tokens(head_ids)
            head_ids = head_ids * (1 - mask_for_labels) + cfg.pad_head_id * mask_for_labels
            upos_values = cfg.corpus.upos_values
            postag2ids = {label: i for i, label in enumerate(upos_values)}
            postag_ids = [postag2ids[t] for t in upos]
            postag_ids = self._update_label_for_subword(postag_ids, segment_spans)
            postag_ids = self.tokenizer.backend.build_inputs_with_special_tokens(postag_ids)
            postag_ids = postag_ids * (1 - mask_for_labels) + cfg.pad_label_id * mask_for_labels
            deprel_values = cfg.corpus.deprel_values
            deprel2ids = {label: i for i, label in enumerate(deprel_values)}
            deprel_ids = [deprel2ids[t] for t in deprel]
            deprel_ids = self._update_label_for_subword(deprel_ids, segment_spans)
            deprel_ids = self.tokenizer.backend.build_inputs_with_special_tokens(deprel_ids)
            deprel_ids = deprel_ids * (1 - mask_for_labels) + cfg.pad_label_id * mask_for_labels

            yield {"doc_id": doc_id,
                   "par_id": par_id,
                   "sent_id": sent_id,
                   "form": form,
                   **t_result,
                   "postag_ids": postag_ids,
                   "head_ids": head_ids,
                   "deprel_ids": deprel_ids}

    def compose(self, batch: List[Dict]) -> Dict:
        batch = {k: [data[k] for data in batch] for k in batch[0]}
        batch["input_ids"] = pad(tensors=batch["input_ids"],
                                 padding_value=self.tokenizer.pad_token_id,
                                 padding_side="right")
        batch["attention_mask"] = pad(tensors=batch["attention_mask"],
                                      padding_value=0,
                                      padding_side="right")
        batch["special_token_mask"] = pad(tensors=batch["special_token_mask"],
                                          padding_value=1,
                                          padding_side="right")

        postag_ids = [torch.tensor(d, dtype=torch.long) for d in batch["postag_ids"]]
        batch["postag_ids"] = pad(tensors=postag_ids, padding_value=cfg.pad_label_id, padding_side="right")

        head_ids = [torch.tensor(d, dtype=torch.long) for d in batch["head_ids"]]
        batch["head_ids"] = pad(tensors=head_ids, padding_value=cfg.pad_head_id, padding_side="right")

        deprel_ids = [torch.tensor(d, dtype=torch.long) for d in batch["deprel_ids"]]
        batch["deprel_ids"] = pad(tensors=deprel_ids, padding_value=cfg.pad_label_id, padding_side="right")

        return batch

    @staticmethod
    def get_label_ids(vocab, values) -> List:
        stoi = {s: i for i, s in enumerate(vocab)}
        return torch.tensor([stoi[v] for v in values]).long()

    def _get_special_tokens_mask(self, token_ids: List) -> np.array:
        mask = self.tokenizer.backend.get_special_tokens_mask(token_ids, already_has_special_tokens=True)
        mask = np.array(mask)
        unk_token_id = self.tokenizer.unk_token_id
        unk_mask = [1 if t == unk_token_id else 0 for t in token_ids]
        unk_mask = np.array(unk_mask)
        special_tokens_mask = mask & (~unk_mask)

        return special_tokens_mask

    def _update_head_for_subword(self, heads: np.array, segment_spans: List) -> List:  # noqa
        word_position = np.cumsum([0, 1] + [(span[1] - span[0]) for span in segment_spans[1:-1]])
        new_head = []
        for span, head in zip(segment_spans[1:-1], heads):
            assert head >= 0, "Error: head < 0"
            for i in range(span[1] - span[0]):
                h = word_position[int(head)] if i == 0 else cfg.pad_head_id
                new_head.append(h)
        return new_head

    def _update_label_for_subword(self, labels: np.array, segment_spans: List) -> List:  # noqa
        new_label = []
        for span, label in zip(segment_spans[1:-1], labels):
            for i in range(span[1] - span[0]):
                t = label if i == 0 else cfg.pad_label_id
                new_label.append(t)
        return new_label

