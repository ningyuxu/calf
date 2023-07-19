from typing import Dict, Tuple, Optional, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F  # noqa

from calf import device, cfg
from calf.modules.mlp import MLP
from exps.cl_syntactic_diff_mbert.model.modules.tree import decode_mst


class BiaffineModel(nn.Module):

    POS_TO_IGNORE = {"``", "''", ":", ",", ".", "PU", "PUNCT", "SYM"}

    def __init__(self, task_cfg):
        super(BiaffineModel, self).__init__()
        self.task_cfg = task_cfg

        self.num_pos_tags = len(self.task_cfg.upos_values)
        self.embedding_dim = self.task_cfg.embedding_dim
        self.parser_pos_dim = self.task_cfg.parser_pos_dim
        self.parser_rel_dim = self.task_cfg.parser_rel_dim
        self.parser_arc_dim = self.task_cfg.parser_arc_dim
        self.parser_use_pos = self.task_cfg.parser_use_pos
        self.parser_predict_pos = self.task_cfg.parser_predict_pos
        self.parser_use_predict_pos = self.task_cfg.parser_use_predict_pos
        self.parser_dropout = self.task_cfg.parser_dropout
        # self.parser_predict_rel_label = self.task_cfg.parser_predict_rel_label
        self.num_rels = len(self.task_cfg.deprel_values)

        if self.parser_use_pos or self.parser_use_predict_pos:
            if self.parser_use_pos:
                num_pos_tags = self.num_pos_tags + 1
                padding_idx = cfg.pad_label_id
            elif self.parser_use_predict_pos:
                assert (
                    self.parser_predict_pos
                ), "when parser_use_predict_pos is True, parser_predict_pos should also be True."
                num_pos_tags = self.num_pos_tags
                padding_idx = None
            else:
                raise ValueError("parser_use_pos and parser_use_predict_pos are mutually exclusive.")
            self.pos_embed = torch.nn.Embedding(
                num_pos_tags, self.parser_pos_dim, padding_idx=padding_idx
            )
            self.embedding_dim += self.parser_pos_dim

        if self.parser_predict_pos:
            self.pos_tagger = torch.nn.Linear(self.task_cfg.embedding_dim, self.num_pos_tags)

        self.head_arc_mlp = MLP(self.embedding_dim, self.parser_arc_dim, self.parser_dropout)
        self.child_arc_mlp = MLP(self.embedding_dim, self.parser_arc_dim, self.parser_dropout)
        self.head_rel_mlp = MLP(self.embedding_dim, self.parser_rel_dim, self.parser_dropout)
        self.child_rel_mlp = MLP(self.embedding_dim, self.parser_rel_dim, self.parser_dropout)

        self.arc_attention = BiLinearAttention(
            self.parser_arc_dim, self.parser_arc_dim, use_input_biases=True
        )
        self.rel_bilinear = torch.nn.Bilinear(
            self.parser_rel_dim, self.parser_rel_dim, self.num_rels
        )

        punctuation_tag_indices = {
            pos_tag: index
            for index, pos_tag in enumerate(self.task_cfg.upos_values) if pos_tag in self.POS_TO_IGNORE
        }
        self.pos_to_ignore = set(punctuation_tag_indices.values())

    def inference(self, x: torch.Tensor, postag_ids):
        if self.parser_predict_pos:
            postag_logits = self.pos_tagger(x)
            postags = F.log_softmax(postag_logits, dim=-1)
        else:
            postags = None

        if self.parser_use_pos:
            pos = self.pos_embed(postag_ids.masked_fill(postag_ids < 0, self.num_pos_tags))
            x = torch.cat((x, pos), dim=-1)
        elif self.parser_use_predict_pos:
            assert postags is not None
            pos = F.linear(postags.exp().detach(), self.pos_embed.weight.t())
            x = torch.cat((x, pos), dim=-1)

        head_arc = self.head_arc_mlp(x)
        child_arc = self.child_arc_mlp(x)
        score_arc = self.arc_attention(head_arc, child_arc)

        head_rel = self.head_rel_mlp(x)
        child_rel = self.child_rel_mlp(x)

        return score_arc, head_rel, child_rel, postags

    def forward_loss(self, batch: Dict) -> Tuple[torch.Tensor, int, Optional[Dict]]:
        x = batch["embedding"]
        postag_ids = batch["postag_ids"]
        assert x.size(1) == postag_ids.size(-1), \
            f"Size of embedding [{x.size()}] does not match size of labels [{postag_ids.size()}]"
        head_ids = batch["head_ids"]
        deprel_ids = batch["deprel_ids"]
        first_subword_mask = head_ids != cfg.pad_head_id

        score_arc, head_rel, child_rel, postags = self.inference(x, postag_ids)

        minus_inf = -1e8
        minus_mask = ~first_subword_mask * minus_inf
        score_arc = score_arc + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

        if self.parser_predict_pos:
            pos_nll = self.loss_of_pos_tagging(postags, postag_ids)
        else:
            pos_nll = torch.tensor(0.0, device=device)

        arc_nll, rel_nll = self.loss_of_dependency_parsing(
            head_rel=head_rel,
            child_rel=child_rel,
            score_arc=score_arc,
            head_indices=head_ids,
            head_tags=deprel_ids,
            mask=first_subword_mask,  # noqa
        )

        # if not self.parser_predict_rel_label:
        #     rel_nll = torch.zeros(1).squeeze(0).to(device)

        loss = pos_nll + arc_nll + rel_nll
        detail_loss = {
            "pos_loss": pos_nll, "arc_loss": arc_nll, "rel_loss": rel_nll
        }
        return loss, x.size(0), detail_loss

    def evaluate(self, batch: Dict) -> Tuple[Dict, Dict]:
        x = batch["embedding"]
        postag_ids = batch["postag_ids"]
        head_ids = batch["head_ids"]
        deprel_ids = batch["deprel_ids"]
        first_subword_mask = head_ids != cfg.pad_head_id

        score_arc, head_rel, child_rel, postags = self.inference(x, postag_ids)

        if self.parser_predict_pos:
            pos_nll = self.loss_of_pos_tagging(postags, postag_ids)
        else:
            pos_nll = torch.tensor(0.0, device=device)

        arc_nll, rel_nll = self.loss_of_dependency_parsing(
            head_rel=head_rel,
            child_rel=child_rel,
            score_arc=score_arc,
            head_indices=head_ids,
            head_tags=deprel_ids,
            mask=first_subword_mask,  # noqa
        )
        # if not self.parser_predict_rel_label:
        #     rel_nll = torch.zeros(1).squeeze(0).to(device)

        lengths = batch["length"]

        predict_heads, predict_labels = self._mst_decode(
            head_rel, child_rel, score_arc, first_subword_mask, lengths  # noqa
        )

        if self.task_cfg.ignore_pos_punct:
            mask = self._get_mask_for_eval(first_subword_mask, postag_ids)  # noqa
        else:
            mask = first_subword_mask

        head_ids = head_ids.detach().cpu()
        deprel_ids = deprel_ids.detach().cpu()
        predict_heads = predict_heads.detach().cpu()
        predict_labels = predict_labels.detach().cpu()
        mask = mask.detach().cpu()

        correct_heads = predict_heads.eq(head_ids).long() * mask
        unlabeled_full_match = (correct_heads + (1 - mask.long())).prod(dim=-1)
        correct_labels = predict_labels.eq(deprel_ids).long() * mask
        correct_labels_and_indices = correct_heads * correct_labels
        labeled_full_match = (correct_labels_and_indices + (1 - mask.long())).prod(dim=-1)

        unlabeled_correct = correct_heads.sum().item()
        full_unlabeled_correct = unlabeled_full_match.sum().item()
        labeled_correct = correct_labels_and_indices.sum().item()
        full_labeled_correct = labeled_full_match.sum().item()
        total_sentences = correct_heads.size(0)
        total_words = correct_heads.numel() - (1 - mask.long()).sum().item()

        metric = {
            "unlabeled_correct": unlabeled_correct,
            "full_unlabeled_correct": full_unlabeled_correct,
            "labeled_correct": labeled_correct,
            "full_labeled_correct": full_labeled_correct,
            "total_sentences": total_sentences,
            "total_words": total_words
        }
        loss = {
            "pos_loss": pos_nll, "arc_loss": arc_nll, "rel_loss": rel_nll
        }

        return metric, loss

    def _get_mask_for_eval(self, mask: torch.Tensor, pos_tags: torch.Tensor) -> torch.Tensor:
        """
        Dependency evaluation excludes words are punctuation. Here, we create
        a new mask to exclude word indices which have a "punctuation-like"
        part of speech tag.

        Parameters
        ----------
        mask : `torch.BoolTensor`, required.
            The original mask.
        pos_tags : `torch.LongTensor`, required.
            The pos tags for the sequence.

        Returns
        -------
        A new mask, where any indices equal to labels we should be ignoring
        are masked.
        """
        new_mask = mask.detach()
        for label in self.pos_to_ignore:
            label_mask = pos_tags.eq(label)
            new_mask = new_mask & ~label_mask
        return new_mask.bool()

    def loss_of_pos_tagging(self, predict: torch.Tensor, target: torch.Tensor):
        pos_nll = F.nll_loss(
            predict.view(-1, self.num_pos_tags), target.view(-1), ignore_index=cfg.pad_label_id,
        )
        return pos_nll

    def loss_of_dependency_parsing(
            self,
            head_rel: torch.Tensor,
            child_rel: torch.Tensor,
            score_arc: torch.Tensor,
            head_indices: torch.Tensor,
            head_tags: torch.Tensor,
            mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the arc and tag loss for a sequence.

        Parameters
        ----------
        head_rel : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, tag_dim), which
            will be used to generate predictions for the dependency tags for
            the given arcs.
        child_rel : `torch.Tensor`, required
            A tensor of shape (batch_size, sequence_length, tag_dim), which
            will be used to generate predictions for the dependency tags for
            the given arcs.
        score_arc : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, sequence_length)
            used to generate a distribution over attachments of a given word
            to all other words.
        head_indices : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length). The indices of
            the heads for every word.
        head_tags : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length). The dependency
            labels of the heads for every word.
        mask : `torch.BoolTensor`, required.
            A mask of shape (batch_size, sequence_length), denoting un-padded
            elements in the sequence.

        Returns
        -------
        arc_nll : `torch.Tensor`, required.
            The negative log likelihood from the arc loss.
        tag_nll : `torch.Tensor`, required.
            The negative log likelihood from the arc tag loss.
        """
        batch_size, sequence_length, _ = score_arc.size()
        # shape (batch_size, 1)
        range_vector = torch.arange(
            batch_size, device=score_arc.device
        ).unsqueeze(1)
        # shape (batch_size, sequence_length, sequence_length)
        normalised_arc_logits = (
                masked_log_softmax(score_arc, mask) * mask.unsqueeze(2) * mask.unsqueeze(1)
        )
        # shape (batch_size, sequence_length, num_head_tags)
        head_tag_logits = self._get_head_tags(
            head_rel, child_rel, head_indices
        )
        normalised_head_tag_logits = masked_log_softmax(
            head_tag_logits, mask.unsqueeze(-1)
        ) * mask.unsqueeze(-1)
        # index matrix with shape (batch, sequence_length)
        timestep_index = torch.arange(
            sequence_length, device=score_arc.device
        )
        child_index = (
            timestep_index.view(
                1, sequence_length
            ).expand(batch_size, sequence_length).long()
        )
        # shape (batch_size, sequence_length)
        arc_loss = normalised_arc_logits[
            range_vector, child_index, head_indices
        ]
        tag_loss = normalised_head_tag_logits[
            range_vector, child_index, head_tags
        ]
        # We don't care about predictions for the symbolic ROOT
        # token's head, so we remove it from the loss.
        arc_loss = arc_loss[:, 1:]
        tag_loss = tag_loss[:, 1:]

        # The number of valid positions is equal to the number of
        # unmasked elements minus 1 per sequence in the batch, to
        # account for the symbolic HEAD token.
        valid_positions = mask.sum() - batch_size
        if valid_positions > 0:
            arc_nll = -arc_loss.sum() / valid_positions.float()
            tag_nll = -tag_loss.sum() / valid_positions.float()
        else:
            arc_nll = -arc_loss.sum() * 0
            tag_nll = -tag_loss.sum() * 0

        return arc_nll, tag_nll

    def _get_head_tags(
            self,
            head_tag: torch.Tensor,
            child_tag: torch.Tensor,
            head_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decodes the head tags given the head and child tag representations and
        a tensor of head indices to compute tags for. Note that these are either
        gold or predicted heads, depending on whether this function is being
        called to compute the loss, or if it's being called during inference.

        # Parameters
        head_tag : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, tag_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag : `torch.Tensor`, required
            A tensor of shape (batch_size, sequence_length, tag_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        head_indices : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length). The indices of the
            heads for every word.

        # Returns
        head_tag_logits : `torch.Tensor`
            A tensor of shape (batch_size, sequence_length, num_head_tags),
            representing logits for predicting a distribution over tags
            for each arc.
        """
        batch_size = head_tag.size()[0]
        # shape (batch_size, 1)
        range_vector = torch.arange(
            batch_size, device=head_tag.device
        ).unsqueeze(1)

        # This next statement is quite a complex piece of indexing, which you
        # really need to read the docs to understand. See here:
        # https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html#advanced-indexing
        # In effect, we are selecting the indices corresponding to the heads
        # of each word from the sequence length dimension for each element in
        # the batch.

        # shape (batch_size, sequence_length, tag_dim)
        selected_head_tag = head_tag[range_vector, head_indices]
        selected_head_tag = selected_head_tag.contiguous()
        # shape (batch_size, sequence_length, num_head_tags)
        head_tag_logits = self.rel_bilinear(selected_head_tag, child_tag)
        return head_tag_logits

    def _mst_decode(
            self,
            head_tag: torch.Tensor,
            child_tag: torch.Tensor,
            score_arc: torch.Tensor,
            mask: torch.BoolTensor,
            lengths: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decodes the head and head tag predictions using the Edmonds' Algorithm
        for finding minimum spanning trees on directed graphs. Nodes in the
        graph are the words in the sentence, and between each pair of nodes,
        there is an edge in each direction, where the weight of the edge
        corresponds to the most likely dependency label probability for that
        arc. The MST is then generated from this directed graph.

        Parameters
        ----------
        head_tag : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, tag_dim), which
            will be used to generate predictions for the dependency tags for
            the given arcs.
        child_tag : `torch.Tensor`, required
            A tensor of shape (batch_size, sequence_length, tag_dim), which
            will be used to generate predictions for the dependency tags for
            the given arcs.
        score_arc : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, sequence_length)
            used to generate a distribution over attachments of a given word
            to all other words.

        Returns
        -------
        heads : `torch.Tensor`
            A tensor of shape (batch_size, sequence_length) representing the
            greedily decoded heads of each word.
        head_tags : `torch.Tensor`
            A tensor of shape (batch_size, sequence_length) representing the
            dependency tags of the optimally decoded heads of each word.
        """
        batch_size, sequence_length, tag_dim = head_tag.size()

        expanded_shape = [
            batch_size, sequence_length, sequence_length, tag_dim
        ]
        head_tag = head_tag.unsqueeze(2)
        head_tag = head_tag.expand(*expanded_shape).contiguous()
        child_tag = child_tag.unsqueeze(1)
        child_tag = child_tag.expand(*expanded_shape).contiguous()

        # Shape (batch_size, sequence_length, sequence_length, num_head_tags)
        pairwise_head_logits = self.rel_bilinear(head_tag, child_tag)

        # Note that this log_softmax is over the tag dimension, and we don't consider pairs of tags
        # which are invalid (e.g. are a pair which includes a padded element) anyway below.
        # Shape (batch, num_labels, sequence_length, sequence_length)
        normalized_pairwise_head_logits = F.log_softmax(
            pairwise_head_logits, dim=3
        ).permute(0, 3, 1, 2)

        # Mask padded tokens, because we only want to consider actual words as
        # heads.
        minus_inf = -1e8
        minus_mask = ~mask * minus_inf
        score_arc = score_arc + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

        # Shape (batch_size, sequence_length, sequence_length)
        normalized_arc_logits = F.log_softmax(score_arc, dim=2).transpose(1, 2)

        # Shape (batch_size, num_head_tags, sequence_length, sequence_length)
        # This energy tensor expresses the following relation:
        # energy[i,j] = "Score that i is the head of j".
        # In this case, we have heads pointing to their children.
        # if self.parser_predict_rel_label:
        batch_energy = torch.exp(
            normalized_arc_logits.unsqueeze(1) + normalized_pairwise_head_logits
        )
        # else:
        #     batch_energy = torch.exp(normalized_arc_logits.unsqueeze(1))
        return self._run_mst_decoding(batch_energy, lengths)

    @staticmethod
    def _run_mst_decoding(
            batch_energy: torch.Tensor,
            lengths: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        heads = []
        head_tags = []
        for energy, length in zip(batch_energy.detach().cpu(), lengths):
            scores, tag_ids = energy.max(dim=0)
            # Although we need to include the root node so that the MST includes it,
            # we do not want any word to be the parent of the root node.
            # Here, we enforce this by setting the scores for all word -> ROOT
            # edges to be 0.
            scores[0, :] = 0
            # Decode the heads. Because we modify the scores to prevent
            # adding in word -> ROOT edges, we need to find the labels ourselves.
            instance_heads, _ = decode_mst(scores.numpy(), length, has_labels=False)

            # Find the labels which correspond to the edges in the max spanning tree.
            instance_head_tags = []
            for child, parent in enumerate(instance_heads):
                instance_head_tags.append(tag_ids[parent, child].item())
            # We don't care what the head or tag is for the root token, but by default
            # it's not necessarily the same in the batched vs un-batched case, which is
            # annoying. Here we'll just set them to zero.
            instance_heads[0] = 0
            instance_head_tags[0] = 0
            heads.append(instance_heads)
            head_tags.append(instance_head_tags)
        return (
            torch.from_numpy(np.stack(heads)).to(batch_energy.device),
            torch.from_numpy(np.stack(head_tags)).to(batch_energy.device),
        )


class BiLinearAttention(torch.nn.Module):
    """
    Computes attention between two matrices using a bi-linear attention
    function. This function has a matrix of weights ``W`` and a bias ``b``, and
    the similarity between the two matrices ``X`` and ``Y`` is computed as
    ``X W Y^T + b``.

    Input:
        - mat1: ``(batch_size, num_rows_1, mat1_dim)``
        - mat2: ``(batch_size, num_rows_2, mat2_dim)``

    Output:
        - ``(batch_size, num_rows_1, num_rows_2)``
    """

    def __init__(self, mat1_dim: int, mat2_dim: int, use_input_biases: bool = False) -> None:
        super(BiLinearAttention, self).__init__()

        if use_input_biases:
            mat1_dim += 1
            mat2_dim += 1

        self.weight = torch.nn.Parameter(torch.zeros(1, mat1_dim, mat2_dim))
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self._use_input_biases = use_input_biases
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        self.bias.data.fill_(0)

    def forward(self, mat1: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
        if self._use_input_biases:
            bias1 = mat1.new_ones(mat1.size()[:-1] + (1,))
            bias2 = mat2.new_ones(mat2.size()[:-1] + (1,))

            mat1 = torch.cat([mat1, bias1], -1)
            mat2 = torch.cat([mat2, bias2], -1)

        intermediate = torch.matmul(mat1.unsqueeze(1), self.weight)
        final = torch.matmul(intermediate, mat2.unsqueeze(1).transpose(2, 3))
        return final.squeeze(1) + self.bias


class Biaffine(nn.Module):
    """
    Biaffine layer for first-order scoring :cite:`dozat-etal-2017-biaffine`.

    This function has a tensor of weights :math:`W` and bias terms if needed.
    The score :math:`s(x, y)` of the vector pair :math:`(x, y)` is computed as :math:`x^T W y / d^s`,
    where `d` and `s` are vector dimension and scaling factor respectively.
    :math:`x` and :math:`y` can be concatenated with bias terms.

    Args:
        n_in (int):
            The size of the input feature.
        n_out (int):
            The number of output channels.
        n_proj (int):
            If specified, applies MLP layers to reduce vector dimensions. Default: ``None``.
        dropout (float):
            If specified, applies a :class:`SharedDropout` layer with the ratio on MLP outputs. Default: 0.
        scale (float):
            Factor to scale the scores. Default: 0.
        bias_x (bool):
            If ``True``, adds a bias term for tensor :math:`x`. Default: ``True``.
        bias_y (bool):
            If ``True``, adds a bias term for tensor :math:`y`. Default: ``True``.
        decompose (bool):
            If ``True``, represents the weight as the product of 2 independent matrices. Default: ``False``.
        init (Callable):
            Callable initialization method. Default: `nn.init.zeros_`.

    """

    def __init__(
        self,
        n_in: int,
        n_out: int = 1,
        n_proj: Optional[int] = None,
        dropout: Optional[float] = 0,
        scale: int = 0,
        bias_x: bool = True,
        bias_y: bool = True,
        decompose: bool = False,
        init: Callable = nn.init.zeros_
    ):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.n_proj = n_proj
        self.dropout = dropout
        self.scale = scale
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.decompose = decompose
        self.init = init

        if n_proj is not None:
            self.mlp_x, self.mlp_y = MLP(n_in, n_proj, dropout), MLP(n_in, n_proj, dropout)

        self.n_model = n_proj or n_in

        if not decompose:
            self.weight = nn.Parameter(torch.Tensor(n_out, self.n_model + bias_x, self.n_model + bias_y))
        else:
            self.weight = nn.ParameterList(
                (nn.Parameter(torch.Tensor(n_out, self.n_model + bias_x)),
                 nn.Parameter(torch.Tensor(n_out, self.n_model + bias_y)))
            )

        self.reset_parameters()

    def __repr__(self):
        s = f"n_in={self.n_in}"
        if self.n_out > 1:
            s += f", n_out={self.n_out}"
        if self.n_proj is not None:
            s += f", n_proj={self.n_proj}"
        if self.dropout > 0:
            s += f", dropout={self.dropout}"
        if self.scale != 0:
            s += f", scale={self.scale}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"
        if self.decompose:
            s += f", decompose={self.decompose}"
        return f"{self.__class__.__name__}({s})"

    def reset_parameters(self):
        if self.decompose:
            for i in self.weight:
                self.init(i)
        else:
            self.init(self.weight)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): ``[batch_size, seq_len, n_in]``.
            y (torch.Tensor): ``[batch_size, seq_len, n_in]``.

        Returns:
            torch.Tensor:
                A scoring tensor of shape ``[batch_size, n_out, seq_len, seq_len]``.
                If ``n_out=1``, the dimension for ``n_out`` will be squeezed automatically.
        """

        if hasattr(self, 'mlp_x'):
            x, y = self.mlp_x(x), self.mlp_y(y)
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        if self.decompose:
            wx = torch.einsum('bxi,oi->box', x, self.weight[0])
            wy = torch.einsum('byj,oj->boy', y, self.weight[1])
            s = torch.einsum('box,boy->boxy', wx, wy)
        else:
            s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        return s.squeeze(1) / self.n_in ** self.scale


class Triaffine(nn.Module):
    """
    Triaffine layer for second-order scoring :cite:`zhang-etal-2020-efficient,wang-etal-2019-second`.

    This function has a tensor of weights :math:`W` and bias terms if needed.
    The score :math:`s(x, y, z)` of the vector triple :math:`(x, y, z)` is computed as :math:`x^T z^T W y / d^s`,
    where `d` and `s` are vector dimension and scaling factor respectively.
    :math:`x` and :math:`y` can be concatenated with bias terms.

    Args:
        n_in (int):
            The size of the input feature.
        n_out (int):
            The number of output channels.
        n_proj (int):
            If specified, applies MLP layers to reduce vector dimensions. Default: ``None``.
        dropout (float):
            If specified, applies a :class:`SharedDropout` layer with the ratio on MLP outputs. Default: 0.
        scale (float):
            Factor to scale the scores. Default: 0.
        bias_x (bool):
            If ``True``, adds a bias term for tensor :math:`x`. Default: ``False``.
        bias_y (bool):
            If ``True``, adds a bias term for tensor :math:`y`. Default: ``False``.
        decompose (bool):
            If ``True``, represents the weight as the product of 3 independent matrices. Default: ``False``.
        init (Callable):
            Callable initialization method. Default: `nn.init.zeros_`.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int = 1,
        n_proj: Optional[int] = None,
        dropout: Optional[float] = 0,
        scale: int = 0,
        bias_x: bool = False,
        bias_y: bool = False,
        decompose: bool = False,
        init: Callable = nn.init.zeros_
    ):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.n_proj = n_proj
        self.dropout = dropout
        self.scale = scale
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.decompose = decompose
        self.init = init

        if n_proj is not None:
            self.mlp_x = MLP(n_in, n_proj, dropout)
            self.mlp_y = MLP(n_in, n_proj, dropout)
            self.mlp_z = MLP(n_in, n_proj, dropout)
        self.n_model = n_proj or n_in
        if not decompose:
            self.weight = nn.Parameter(torch.Tensor(n_out, self.n_model + bias_x, self.n_model, self.n_model + bias_y))
        else:
            self.weight = nn.ParameterList((nn.Parameter(torch.Tensor(n_out, self.n_model + bias_x)),
                                            nn.Parameter(torch.Tensor(n_out, self.n_model)),
                                            nn.Parameter(torch.Tensor(n_out, self.n_model + bias_y))))

        self.reset_parameters()

    def __repr__(self):
        s = f"n_in={self.n_in}"
        if self.n_out > 1:
            s += f", n_out={self.n_out}"
        if self.n_proj is not None:
            s += f", n_proj={self.n_proj}"
        if self.dropout > 0:
            s += f", dropout={self.dropout}"
        if self.scale != 0:
            s += f", scale={self.scale}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"
        if self.decompose:
            s += f", decompose={self.decompose}"
        return f"{self.__class__.__name__}({s})"

    def reset_parameters(self):
        if self.decompose:
            for i in self.weight:
                self.init(i)
        else:
            self.init(self.weight)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): ``[batch_size, seq_len, n_in]``.
            y (torch.Tensor): ``[batch_size, seq_len, n_in]``.
            z (torch.Tensor): ``[batch_size, seq_len, n_in]``.

        Returns:
            ~torch.Tensor:
                A scoring tensor of shape ``[batch_size, n_out, seq_len, seq_len, seq_len]``.
                If ``n_out=1``, the dimension for ``n_out`` will be squeezed automatically.
        """

        if hasattr(self, 'mlp_x'):
            x, y, z = self.mlp_x(x), self.mlp_y(y), self.mlp_z(y)
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len, seq_len]
        if self.decompose:
            wx = torch.einsum('bxi,oi->box', x, self.weight[0])
            wz = torch.einsum('bzk,ok->boz', z, self.weight[1])
            wy = torch.einsum('byj,oj->boy', y, self.weight[2])
            s = torch.einsum('box,boz,boy->bozxy', wx, wz, wy)
        else:
            w = torch.einsum('bzk,oikj->bozij', z, self.weight)
            s = torch.einsum('bxi,bozij,byj->bozxy', x, w, y)
        return s.squeeze(1) / self.n_in ** self.scale


def masked_log_softmax(
        vector: torch.Tensor, mask: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """
    ``torch.nn.functional.log_softmax(vector)`` does not work if some elements
    of ``vector`` should be masked.  This performs a log_softmax on just the
    non-masked portions of ``vector``.  Passing ``None`` in for the mask is also
    acceptable; you'll just get a regular log_softmax. ``vector`` can have an
    arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcast-able to ``vector's`` shape.  If ``mask`` has fewer dimensions than
    ``vector``, we will un-squeeze on dimension 1 until they match.  If you need
    a different un-squeezing of your mask, do it yourself before passing the
    mask into this function.

    In the case that the input vector is completely masked, the return value of
    this function is arbitrary, but not ``nan``.  You should be masking the
    result of whatever computation comes out of this in that case, anyway, so
    the specific values returned shouldn't matter.  Also, the way that we deal
    with this case relies on having single-precision floats; mixing
    half-precision floats with fully-masked vectors will likely give you
    ``nans``.

    If your logits are all extremely negative (i.e., the max value in your logit
    vector is -50 or lower), the way we handle masking here could mess you up.
    But if you've got logit values that extreme, you've got bigger problems than
    this.
    """
    if mask is not None:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # vector + mask.log() is an easy way to zero out masked elements in
        # logspace, but it results in nans when the whole vector is masked.  We
        # need a very small value instead of a zero in the mask for these cases.
        # log(1 + 1e-45) is still basically 0, so we can safely just add 1e-45
        # before calling mask.log().  We use 1e-45 because 1e-46 is so small it
        # becomes 0 - this is just the smallest value we can actually use.
        vector = vector + (mask + 1e-45).log()
    return F.log_softmax(vector, dim=dim)


def masked_softmax(
        vector: torch.Tensor, mask: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    if mask is not None:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        vector = vector + (mask + 1e-45).log()
    return F.softmax(vector, dim=dim)
