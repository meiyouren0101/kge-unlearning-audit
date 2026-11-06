"""Eq.5 pairwise logistic objective with the source L1 distance kept explicit.

A_original is the unchanged public implementation. B changes its full objective,
not merely a softmax axis; no author-confirmed bug claim is implied. The paper
writes L2 whereas both audited TransE source paths here retain normalized L1.
"""

import torch
import torch.nn.functional as F


def distance_vectors(model, positive_triples, negative_triples):
    """Use the same model.score_fun and phase-reference embeddings in both arms."""
    if positive_triples.ndim != 2 or positive_triples.shape[1] != 3:
        raise ValueError("Positive triples must have shape [B, 3].")
    if negative_triples.shape != positive_triples.shape or not positive_triples.shape[0]:
        raise ValueError("Positive/negative triple shapes must agree and be nonempty.")
    if positive_triples.dtype != torch.long or negative_triples.dtype != torch.long:
        raise ValueError("Triple indices must use torch.long.")
    entities, relations = model.embedding()
    old_entities, old_relations = model.old_embeddings()
    if old_entities.requires_grad or old_relations.requires_grad:
        raise ValueError("Phase-reference tensors must remain detached.")

    def score(triples, entity_table, relation_table):
        return model.score_fun(
            torch.index_select(entity_table, 0, triples[:, 0]),
            torch.index_select(relation_table, 0, triples[:, 1]),
            torch.index_select(entity_table, 0, triples[:, 2]),
        )

    return (
        score(positive_triples, entities, relations),
        score(negative_triples, entities, relations),
        score(positive_triples, old_entities, old_relations),
        score(negative_triples, old_entities, old_relations),
    )


def pairwise_from_distances(current_positive, current_negative, old_positive, old_negative):
    """Declared beta=1, alpha=.1 pairwise logsigmoid(-distance) comparison."""
    vectors = (current_positive, current_negative, old_positive, old_negative)
    if any(vector.ndim != 1 or vector.shape != vectors[0].shape for vector in vectors):
        raise ValueError("Four distance vectors must have the same one-dimensional shape.")
    if not current_positive.numel():
        raise ValueError("A pairwise objective cannot use an empty batch.")
    log_f_positive = F.logsigmoid(-current_positive)
    log_f_negative = F.logsigmoid(-current_negative)
    log_f_old_positive = F.logsigmoid(-old_positive)
    log_f_old_negative = F.logsigmoid(-old_negative)
    difference = (
        log_f_positive - log_f_old_positive
        - log_f_negative + log_f_old_negative
    )
    return -0.1 * F.logsigmoid(difference).mean()


def pairwise_dpo_loss(self, positive_triples, negative_triples):
    """Only method overridden by B; source model, losses and distance stay fixed."""
    self.args.dpo_beta = 1
    self.args.dpo_lamba = 0.1
    return pairwise_from_distances(*distance_vectors(self, positive_triples, negative_triples))


def pairwise_class(original_class):
    """Make an isolated subclass rather than modifying the original class in place."""
    return type("GraphDPOPairwise", (original_class,), {"dpo_loss": pairwise_dpo_loss})
