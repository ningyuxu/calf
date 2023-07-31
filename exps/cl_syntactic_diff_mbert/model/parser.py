from tqdm import tqdm
from typing import Tuple, Dict, Union
from pathlib import Path
from omegaconf import DictConfig

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from calf import device
from calf.utils.result import Result

from .modules import BiaffineModel
from .encoder import MBertEncoder


class MBertDPModel(nn.Module):
    def __init__(self, model_cfg):
        super(MBertDPModel, self).__init__()
        self.model_cfg = model_cfg

        self.encoder = MBertEncoder(self.model_cfg.encoder)
        self.parser = BiaffineModel(self.model_cfg.task)

    def embed_data(self, batch: Dict) -> Dict:
        """
        Call encoder to embed data
        """
        batch = self.encoder.embed(batch)
        return batch

    def forward_loss(self, batch: Dict) -> Tuple[Result, int]:
        batch = self.encoder.embed(batch)

        loss, count, detail_loss = self.parser.forward_loss(batch=batch)
        log_header = "{:<20} | {:<20} | {:<20} | {:<20}".format(
            "total_loss", "pos_loss", "arc_loss", "rel_loss",
        )
        log_line = "{:<20} | {:<20} | {:<20} | {:<20}".format(
            loss.item(),
            detail_loss["pos_loss"],
            detail_loss["arc_loss"],
            detail_loss["rel_loss"],
        )
        loss_result = Result(
            metric_score=loss,
            log_header=log_header,
            log_line=log_line,
            metric_detail=detail_loss,
        )
        return loss_result, count

    def evaluate(
            self, dataloader: DataLoader
    ) -> Tuple[Result, Result]:
        unlabeled_correct = 0.0
        full_unlabeled_correct = 0.0
        labeled_correct = 0.0
        full_labeled_correct = 0.0

        total_sentences = 0.0
        total_words = 0.0

        pos_loss = 0.0
        arc_loss = 0.0
        rel_loss = 0.0

        unlabeled_attachment_score = 0.0
        labeled_attachment_score = 0.0
        unlabeled_full_match = 0.0
        labeled_full_match = 0.0

        with torch.no_grad():
            seen_batch = 0
            for batch in tqdm(dataloader, leave=False):
                batch = self.encoder.embed(batch)

                metric, loss = self.parser.evaluate(batch=batch)

                unlabeled_correct += metric["unlabeled_correct"]
                full_unlabeled_correct += metric["full_unlabeled_correct"]
                labeled_correct += metric["labeled_correct"]
                full_labeled_correct += metric["full_labeled_correct"]

                total_sentences += metric["total_sentences"]
                total_words += metric["total_words"]

                pos_loss += loss["pos_loss"]
                arc_loss += loss["arc_loss"]
                rel_loss += loss["rel_loss"]

                seen_batch += 1

            if total_words > 0.0:
                unlabeled_attachment_score = unlabeled_correct / total_words
                labeled_attachment_score = labeled_correct / total_words
            if total_sentences > 0.0:
                unlabeled_full_match = full_unlabeled_correct / total_sentences
                labeled_full_match = full_labeled_correct / total_sentences

        accuracy_score = {
            "uas": unlabeled_attachment_score,
            "las": labeled_attachment_score,
            "ufm": unlabeled_full_match,
            "lfm": labeled_full_match,
        }

        accuracy_log_header = "{:<20} | {:<20} | {:<20} | {:<20}".format(
            f"UAS",
            f"LAS",
            f"UFM",
            f"LFM",
        )
        accuracy_log_line = "{:<20} | {:<20} | {:<20} | {:<20}".format(
            accuracy_score["uas"],
            accuracy_score["las"],
            accuracy_score["ufm"],
            accuracy_score["lfm"],
        )

        loss_score = {
            "pos_loss": pos_loss / seen_batch,
            "arc_loss": arc_loss / seen_batch,
            "rel_loss": rel_loss / seen_batch,
        }
        total_loss = loss_score["pos_loss"] + loss_score["arc_loss"] + loss_score["rel_loss"]
        loss_log_header = "{:<20} | {:<20} | {:<20} | {:<20}".format(
            f"TOTAL_LOSS", f"POS_LOSS", f"ARC_LOSS", f"REL_LOSS",
        )
        loss_log_line = "{:<20} | {:<20} | {:<20} | {:<20}".format(
            total_loss, loss_score["pos_loss"], loss_score["arc_loss"], loss_score["rel_loss"]
        )

        accuracy_result = Result(
            metric_score=labeled_attachment_score,
            log_header=accuracy_log_header,
            log_line=accuracy_log_line,
            metric_detail=accuracy_score
        )

        loss_result = Result(
            metric_score=total_loss,
            log_header=loss_log_header,
            log_line=loss_log_line,
            metric_detail=loss_score
        )

        return accuracy_result, loss_result

    def predict(self, corpus_params):
        self.encoder.predict(corpus_params=corpus_params)

    def save(self, model_file: Union[str, Path]) -> None:
        model_state = self._get_state_dict()
        # save model
        torch.save(model_state, str(model_file), pickle_protocol=4)

    def load(self, model_file: Union[str, Path]):
        with open(model_file, mode="rb") as fh:
            model_state = torch.load(fh, map_location="cpu")

        model = self._init_model_with_state_dict(model_state, self.model_cfg)
        model.eval()
        model.to(device)
        return model

    def _get_state_dict(self) -> Dict:
        state_dict = {"state_dict": self.state_dict()}
        return state_dict

    @classmethod
    def _init_model_with_state_dict(cls, state: Dict, model_cfg):
        model = cls(model_cfg)
        model.load_state_dict(state["state_dict"])
        return model
