from omegaconf import DictConfig

import numpy as np
import torch
from typing import Dict, List, Union
import warnings
from pathlib import Path

from collections import defaultdict
from collections import namedtuple
from tqdm import tqdm

import h5py

from calf import OUTPUT
from torch.utils.data import Dataset as TorchDataset


class HDF5Dataset:
    DATA_FIELDS = ["seqid", "tokens", "token_ids", "length", "postag_ids", "head_ids", "deprel_ids", "embeddings"]

    def __init__(
            self,
            cfg: DictConfig,
            hdf5_cfg: Dict,
            filepath: Union[Path, str] = None,
            layer_index: int = 12,
            control_size: bool = False,
            rel_labels: List = None,
            label2id: Dict = None,
            num_sentences: int = 1000,
            e_samples_per_label: int = 1000,
            sample_strategy: str = "sentence",
            e_min_samples_per_label: int = 1,
            designated_indices: List = None,
            smooth_all: bool = True,
            rand: bool = False,
    ):
        super(HDF5Dataset, self).__init__()
        self.cfg = cfg
        self.hdf5_cfg = hdf5_cfg
        self.lang = self.hdf5_cfg["lang"]
        self.filepath = self.get_embedding_file(filepath=filepath)
        if rel_labels is None:
            rel_labels = cfg.corpus.deprel_values
        if label2id is None:
            label2id = {l: v for (v, l) in enumerate(rel_labels)}
        self.rel_labels = rel_labels
        self.num_classes = len(rel_labels)
        self.label2id = label2id
        self.layer_index = layer_index
        self.control_size = control_size
        self.num_sentences = num_sentences
        self.e_samples_per_label = e_samples_per_label
        self.e_min_samples_per_label = e_min_samples_per_label
        self.sample_strategy = sample_strategy
        self.designated_indices = designated_indices
        self.smooth_all = smooth_all
        self.samples_each_label = dict()
        self.rand = rand
        self.observation_class = self.get_observation_class(self.DATA_FIELDS)
        self.data = self.prepare_dataset()
        self.dataset = ObservationIterator(cfg=self.cfg, data=self.data, lang=self.lang)

    @property
    def data_fields(self) -> List[str]:
        return self.DATA_FIELDS

    def get_embedding_file(self, filepath: Union[Path, str] = None):
        if filepath is None:
            model = f"{self.cfg.model.encoder.name}_{self.cfg.model.encoder.get('revision', 'main')}"
            hdf5_path = OUTPUT / "embedding" / model
            hdf5_file = hdf5_path / f"{self.hdf5_cfg['lang']}-{self.hdf5_cfg['genre']}-{self.hdf5_cfg['split']}.hdf5"
        else:
            hdf5_file = Path(filepath)
        assert hdf5_file.is_file(), f"Check if hdf5 file exists [{hdf5_file}]"
        return hdf5_file

    def prepare_dataset(self):
        observations = self.load_dataset_group()
        root_id = self.label2id["root"]
        label_representations = self.get_label_representations(observations, omit_ids=[root_id])
        all_ids = []
        if self.control_size:
            if self.sample_strategy == "sentence":
                total_sentence_ids = np.unique(label_representations["seqids"])
                if self.designated_indices:
                    if len(self.designated_indices) < self.num_sentences:
                        warnings.warn(
                            f"Designated sentence ids less than expected "
                            f"({len(self.designated_indices)} < {self.num_sentences})"
                        )
                    assert all(item in total_sentence_ids for item in self.designated_indices), \
                        f"Mismatch between designated sentence_ids and sentence_ids loaded from the hdf5 file."
                    s_ids = self.designated_indices
                else:
                    if self.num_sentences < len(total_sentence_ids):
                        s_ids = np.random.choice(total_sentence_ids, size=self.num_sentences, replace=False).tolist()
                    else:
                        s_ids = total_sentence_ids.tolist()
                add_random_sentences = 0
                for s in tqdm(s_ids, desc="[packing sentences]"):
                    samples_ids = (np.array(label_representations["seqids"]) == s).nonzero()[0].tolist()
                    if samples_ids:
                        all_ids.extend(samples_ids)
                    else:  # if selected sentence ids do not retrieve any sentence
                        add_random_sentences += 1
                assert add_random_sentences < 1, "Designated sentence ids retrieve fewer sentences than expected."
                if add_random_sentences:
                    warnings.warn("Designated sentence ids less than expected.")
                    left_sentence_ids = total_sentence_ids[np.in1d(total_sentence_ids, s_ids, invert=True)].tolist()
                    random_sentence_ids = np.random.choice(left_sentence_ids, add_random_sentences).tolist()
                    for s in tqdm(random_sentence_ids, desc="[packing sentences]"):
                        samples_ids = (np.array(label_representations["seqids"]) == s).nonzero()[0].tolist()
                        all_ids.extend(samples_ids)

                smooth = False
                for c in range(self.num_classes):
                    num_class_all = (np.array(label_representations["labels"]) == c).sum()
                    num_class_samples = (np.array(label_representations["labels"])[all_ids] == c).sum()
                    if num_class_samples < self.e_min_samples_per_label <= num_class_all:
                        smooth = True
                        break
                if smooth:
                    ids = np.arange(len(label_representations["labels"]))
                    left_ids = ids[np.in1d(ids, all_ids, invert=True)].tolist()
                    add_ids = []
                    for c in tqdm(range(self.num_classes), desc="[adding samples for smoothing]"):
                        if not self.smooth_all:
                            num_class_samples_ex = (np.array(label_representations["labels"])[all_ids] == c).sum()
                            if num_class_samples_ex > self.e_min_samples_per_label:
                                continue
                        num_class_samples = (np.array(label_representations["labels"])[left_ids] == c).sum()
                        class_ids = (np.array(label_representations["labels"]) == c).nonzero()[0].tolist()
                        candidates = list(set(class_ids) & set(left_ids))
                        if num_class_samples >= self.e_min_samples_per_label:
                            add_ids.extend(
                                np.random.choice(candidates, size=self.e_min_samples_per_label, replace=False).tolist()
                            )
                        else:
                            add_ids.extend(candidates)
                    all_ids.extend(add_ids)
            else:
                for c in tqdm(range(self.num_classes), desc="[packing dataset wrt labels]"):
                    num_class_samples = (np.array(label_representations["labels"]) == c).sum()
                    class_ids = (np.array(label_representations["labels"]) == c).nonzero()[0].tolist()
                    if num_class_samples > self.e_samples_per_label:
                        class_ids = np.random.choice(class_ids, size=self.e_samples_per_label, replace=False).tolist()
                    all_ids.extend(class_ids)
            all_ids = np.array(all_ids)
        else:
            all_ids = np.arange(len(label_representations["labels"]))

        # Get Label Statistics
        for c in range(self.num_classes):
            num_class_all = (np.array(label_representations["labels"]) == c).sum()
            num_class_samples = (np.array(label_representations["labels"])[all_ids] == c).sum()
            self.samples_each_label[self.rel_labels[c]] = dict()
            self.samples_each_label[self.rel_labels[c]]["num_samples"] = int(num_class_samples)
            self.samples_each_label[self.rel_labels[c]]["num_entire_dataset"] = int(num_class_all)

        outputs = defaultdict(list)
        if self.rand:
            print(">>> Assign random labels to representations with a uniform distribution over all labels.")
            unique_labels = np.unique(label_representations["labels"])
            labels_rand = np.random.choice(unique_labels, len(all_ids))
            to_add = {
                f"labels": labels_rand,
                f"representations": np.array(label_representations["representations"])[all_ids].tolist(),
            }
        else:
            to_add = {
                f"labels": np.array(label_representations["labels"])[all_ids].tolist(),
                f"representations": np.array(label_representations["representations"])[all_ids].tolist(),
            }
        for target in to_add:
            outputs[target] += list(to_add[target])
        return outputs

    @staticmethod
    def get_observation_class(fieldnames):
        return namedtuple('Observation', fieldnames, defaults=(None,) * len(fieldnames))

    def load_dataset_group(self):
        observations = []
        data = dict()
        hdf5_file = h5py.File(self.filepath, 'r')
        indices = list(hdf5_file.keys())
        for idx in tqdm(sorted([int(x) for x in indices]), desc='[loading observations]'):
            to_del = 0
            length = int(hdf5_file[str(idx)]["length"][()])
            for key in self.DATA_FIELDS:
                if key == "embeddings":
                    assert len(hdf5_file[str(idx)][key][()][self.layer_index]) == int(length)
                    data[key] = hdf5_file[str(idx)][key][()][self.layer_index]
                elif key in ["seqid", "length"]:
                    data[key] = int(hdf5_file[str(idx)][key][()])
                else:
                    # assert len(hdf5_file[str(idx)][key][()]) == int(length)
                    data[key] = hdf5_file[str(idx)][key][()]
            observation = self.observation_class(**data)
            for head in observation.head_ids:  # noqa
                if head >= observation.length - 1:  # noqa
                    to_del = 1
            if to_del:
                continue
            else:
                observations.append(observation)
        return observations

    def get_label_representations(self, observations, omit_ids: List = None):
        outputs = defaultdict(list)
        for observation in tqdm(observations, desc='[computing labels & representations]'):
            seqids = []
            labels = []
            rel_representations = []
            for i in range(observation.length):
                if observation.deprel_ids[i] == self.cfg.pad_label_id:
                    continue
                label = observation.deprel_ids[i]
                if label in omit_ids:
                    continue
                rel_representation = observation.embeddings[observation.head_ids[i]] - observation.embeddings[i]
                labels.append(label)
                rel_representations.append(rel_representation)
                seqids.append(observation.seqid)
            to_add = {
                "seqids": seqids,
                "labels": labels,
                "representations": rel_representations,
            }
            for target in to_add:
                outputs[target] += list(to_add[target])
        return outputs


class ObservationIterator(TorchDataset):
    def __init__(self, cfg, data, lang, labels=None, targets=None):
        self.cfg = cfg
        self.xs = torch.tensor(data["representations"], dtype=torch.float)
        self.ys = torch.LongTensor(data["labels"])
        self.lang = lang
        self._labels = labels
        self._targets = targets

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, idx):
        x = self.xs[idx]
        y = self.ys[idx]
        return x, y

    @property
    def targets(self):
        if self._targets is None:
            self._targets = self.ys.long()
        return self._targets

    @property
    def classes(self):
        if self._labels is None:
            self._labels = self.cfg.corpus.deprel_values
        return self._labels
