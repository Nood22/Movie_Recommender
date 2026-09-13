"""Experimental scoring policies, evaluated offline and not used in serving."""
import torch


def bounded_align_scores(logits: torch.Tensor, genre_membership: dict[str, torch.Tensor], preferences):
    """Soft genre coverage; at most one top-100 score spread per direction.

    Unlike hard tiers, a tag cannot rescue a candidate arbitrarily far below
    the model's leaders. Inverse catalog frequency gives specific genres more
    weight than ubiquitous ones, and partial coverage gets a partial bonus.
    Call with excluded candidates already masked so they cannot set the scale.
    """
    positive = torch.zeros_like(logits)
    negative = torch.zeros_like(logits)
    for names, target in ((preferences["preferred_genres"], positive),
                          (preferences["avoided_genres"], negative)):
        weight_sum = 0.0
        for genre in names:
            if genre not in genre_membership:
                continue
            membership = genre_membership[genre].to(device=logits.device, dtype=logits.dtype)
            weight = float(torch.log((membership.numel() + 1) / (membership.sum() + 1)).item())
            target.add_(membership * weight)
            weight_sum += weight
        if weight_sum:
            target.div_(weight_sum)
    if not torch.any(positive) and not torch.any(negative):
        return logits.clone()
    result = logits.clone()
    for row in range(logits.shape[0]):
        finite = logits[row][torch.isfinite(logits[row])]
        if finite.numel() < 2:
            continue
        head = finite.topk(min(100, finite.numel())).values
        scale = head[0] - head[-1]
        result[row] += scale * (positive[row] - negative[row])
    return result


def similarity_align_scores(logits: torch.Tensor, genre_membership: dict[str, torch.Tensor], preferences):
    """Bounded weighted-Jaccard genre similarity, with an explicit-dislike penalty."""
    preferred = set(preferences["preferred_genres"])
    avoided = set(preferences["avoided_genres"])
    intersection = torch.zeros_like(logits)
    union = torch.zeros_like(logits)
    negative = torch.zeros_like(logits, dtype=torch.bool)
    for name, membership in genre_membership.items():
        member = membership.to(device=logits.device, dtype=logits.dtype)
        weight = float(torch.log((member.numel() + 1) / (member.sum() + 1)).item())
        if name in preferred:
            intersection += member * weight
            union += weight
        else:
            union += member * weight
        if name in avoided:
            negative |= member.bool()
    adjustment = intersection / union.clamp_min(torch.finfo(logits.dtype).eps) - negative.to(logits.dtype)
    result = logits.clone()
    for row in range(logits.shape[0]):
        finite = logits[row][torch.isfinite(logits[row])]
        if finite.numel() < 2:
            continue
        head = finite.topk(min(100, finite.numel())).values
        result[row] += (head[0] - head[-1]) * adjustment[row]
    return result
