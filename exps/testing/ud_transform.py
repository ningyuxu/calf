from typing import Iterable, Dict, List
import torch
from calf.modules.huggingface import HuggingfaceTokenizer
from calf.utils.tensor import pad
from calf.utils.corpus import Corpus
from calf.utils.transform import Transform


class UDTreebankTransform(Transform):

    def __init__(self, tokenizer: HuggingfaceTokenizer = None) -> None:
        super().__init__(name="doword_ud_transform")

        self.tokenizer = tokenizer
        if not self.tokenizer.pad_token_id:
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
            yield {"doc_id": doc_id,
                   "par_id": par_id,
                   "sent_id": sent_id,
                   "form": form,
                   **t_result,
                   "upos": upos,
                   "head": head,
                   "deprel": deprel}

    def compose(self, batch: List[Dict]) -> Dict:
        batch = {k: [data[k] for data in batch] for k in batch[0]}
        batch["input_ids"] = pad(tensors=batch["input_ids"],
                                 padding_value=self.tokenizer.pad_token_id,
                                 padding_side="left")
        batch["attention_mask"] = pad(tensors=batch["attention_mask"],
                                      padding_value=0,
                                      padding_side="left")
        batch["special_token_mask"] = pad(tensors=batch["special_token_mask"],
                                          padding_value=1,
                                          padding_side="left")
        return batch

    @staticmethod
    def get_label_ids(vocab, values) -> List:
        stoi = {s: i for i, s in enumerate(vocab)}
        return torch.tensor([stoi[v] for v in values]).long()
