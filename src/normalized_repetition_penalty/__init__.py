"""Shift-invariant repetition penalty logits processors for HuggingFace Transformers.

The stock `RepetitionPenaltyLogitsProcessor` branches on the sign of each raw logit
(divide if positive, multiply if negative). The softmax is invariant to adding a
constant to every logit, so the sign of a raw logit is not a property of the model's
distribution; it depends on the model's arbitrary logit zero-point and on whether the
processor chain is running on logits (greedy/sampling) or log-probabilities (beam
search). The effective penalty strength therefore changes with the model and with the
decoding mode.

The processors here apply the same sign-branched rule to the row's log-probabilities,
where the zero-point is fixed (log-probs are always <= 0), and then shift the result
back onto the input scale so downstream processors see exactly what they would have
seen otherwise. Under beam search the input is already log-probs and the shift is a
no-op, so this is the same computation beam search has always done.

See https://arxiv.org/abs/2607.09791 and
https://github.com/huggingface/transformers/pull/48642.
"""

from __future__ import annotations

import torch
from transformers.generation.logits_process import LogitsProcessor

__all__ = [
    "NormalizedRepetitionPenaltyLogitsProcessor",
    "NormalizedEncoderRepetitionPenaltyLogitsProcessor",
]
__version__ = "0.1.0"


def _check_penalty(penalty: float) -> None:
    if not isinstance(penalty, float) or not (penalty > 0):
        raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")


def _penalize_normalized(
    scores: torch.FloatTensor, token_ids: torch.LongTensor, penalty: float
) -> torch.FloatTensor:
    """Apply the sign-branched multiplicative penalty to `token_ids` on the log-prob scale.

    `scores` is `(batch, vocab)`; `token_ids` is `(batch, n)` and may contain ids that are
    `>= vocab` (added tokens the model cannot emit), which are ignored. Returns a new
    tensor on the same scale as `scores`; entries not in `token_ids` are unchanged.
    """
    vocab_size = scores.shape[-1]
    lse = torch.logsumexp(scores, dim=-1, keepdim=True)

    # Absorb out-of-vocab ids into a scratch column so gather/scatter stay in range.
    padded = torch.nn.functional.pad(scores, (0, 1))
    token_ids = torch.where(token_ids >= vocab_size, torch.full_like(token_ids, vocab_size), token_ids)

    score = torch.gather(padded, 1, token_ids) - lse
    # log-probs are <= 0 so this always multiplies; the branch is kept so the rule is
    # literally the stock one, just evaluated at a fixed zero-point.
    score = torch.where(score < 0, score * penalty, score / penalty)
    score = score + lse

    return padded.scatter(1, token_ids, score)[..., :vocab_size]


class NormalizedRepetitionPenaltyLogitsProcessor(LogitsProcessor):
    r"""Repetition penalty applied on the log-probability scale.

    Drop-in alternative to `transformers.RepetitionPenaltyLogitsProcessor`. Same
    `penalty` semantics (> 1.0 penalizes previously seen tokens, < 1.0 rewards them),
    but the effect no longer depends on the model's logit offset or on greedy vs beam.

    Args:
        penalty (`float`):
            The parameter for repetition penalty. 1.0 means no penalty. Above 1.0
            penalizes previously generated tokens; between 0.0 and 1.0 rewards them.
        prompt_ignore_length (`int`, *optional*):
            The original input ids sequence length, which if provided, will not be used
            in the penalty calculation.

    Example:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM
    >>> from normalized_repetition_penalty import NormalizedRepetitionPenaltyLogitsProcessor

    >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
    >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
    >>> inputs = tokenizer(["I'm not going to"], return_tensors="pt")

    >>> # leave `repetition_penalty` unset (1.0) and pass the processor explicitly
    >>> outputs = model.generate(
    ...     **inputs,
    ...     logits_processor=[NormalizedRepetitionPenaltyLogitsProcessor(1.3)],
    ... )
    ```
    """

    def __init__(self, penalty: float, prompt_ignore_length: int | None = None):
        _check_penalty(penalty)
        if prompt_ignore_length is not None and (
            not isinstance(prompt_ignore_length, int) or prompt_ignore_length < 0
        ):
            raise ValueError(
                f"`prompt_ignore_length` has to be a positive integer, but is {prompt_ignore_length}"
            )
        self.penalty = penalty
        self.prompt_ignore_length = prompt_ignore_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.prompt_ignore_length:
            input_ids = input_ids[:, self.prompt_ignore_length :]

        if scores.dim() == 3:
            # (batch, seq, vocab): penalize the last position only, as the stock processor does.
            last = _penalize_normalized(scores[:, -1, :], input_ids, self.penalty)
            scores = scores.clone()
            scores[:, -1, :] = last
            return scores

        return _penalize_normalized(scores, input_ids, self.penalty)


class NormalizedEncoderRepetitionPenaltyLogitsProcessor(LogitsProcessor):
    r"""Encoder repetition ("hallucination") penalty applied on the log-probability scale.

    Drop-in alternative to `transformers.EncoderRepetitionPenaltyLogitsProcessor`.
    Rewards tokens that appear in `encoder_input_ids`.

    Args:
        penalty (`float`):
            The parameter for repetition penalty. 1.0 means no penalty. Above 1.0
            rewards prompt tokens; between 0.0 and 1.0 penalizes them.
        encoder_input_ids (`torch.LongTensor`):
            The encoder_input_ids that should be repeated within the decoder ids.
    """

    def __init__(self, penalty: float, encoder_input_ids: torch.LongTensor):
        _check_penalty(penalty)
        self.penalty = 1 / penalty
        self.encoder_input_ids = encoder_input_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        return _penalize_normalized(scores, self.encoder_input_ids, self.penalty)
