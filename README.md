# normalized-repetition-penalty

Shift-invariant repetition penalty logits processors for HuggingFace Transformers.

```bash
pip install normalized-repetition-penalty
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from normalized_repetition_penalty import NormalizedRepetitionPenaltyLogitsProcessor

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
inputs = tok("I'm not going to", return_tensors="pt")

out = model.generate(
    **inputs,
    max_new_tokens=40,
    # leave `repetition_penalty` unset and pass the processor explicitly
    logits_processor=[NormalizedRepetitionPenaltyLogitsProcessor(1.3)],
)
```

Same `penalty` semantics as `transformers.RepetitionPenaltyLogitsProcessor` (> 1.0 penalizes
seen tokens, < 1.0 rewards them), and the same `prompt_ignore_length` option.
`NormalizedEncoderRepetitionPenaltyLogitsProcessor` is the matching replacement for
`EncoderRepetitionPenaltyLogitsProcessor`.

## Why

The stock processor branches on the sign of each raw logit:

```python
score = torch.where(score < 0, score * penalty, score / penalty)
```

The softmax is invariant to adding a constant to every logit, so the sign of a raw logit is
not a property of the model's distribution. It depends on the model's arbitrary logit
zero-point (gpt2's logits are all negative; pythia-160m's are all positive) and on whether
the processor chain is running on logits (greedy/sampling) or log-probabilities (beam
search). The effective penalty strength therefore changes with the model and with the
decoding mode, and the same `repetition_penalty=1.3` means different things on different
models.

This was noted when the sign branch was introduced in 2019
([huggingface/transformers#2303](https://github.com/huggingface/transformers/pull/2303#issuecomment-587962571)),
and it is measured across models and inference stacks in
[arXiv:2607.09791](https://arxiv.org/abs/2607.09791).

## What this does

The processors here apply the same rule to the row's log-probabilities, where the
zero-point is fixed, and shift the result back onto the input scale:

```python
lse = torch.logsumexp(scores, dim=-1, keepdim=True)
score = torch.gather(scores, 1, input_ids) - lse      # log-probs, always <= 0
score = torch.where(score < 0, score * penalty, score / penalty)
score = score + lse                                   # back to the input scale
```

Consequences:

- The output distribution is invariant to a constant shift of the logits.
- Tokens not in `input_ids` are returned bit-identical; downstream processors see the same
  scale they would have seen without the penalty.
- Under beam search the input is already log-probs, `lse == 0`, and this is exactly the
  stock computation. Greedy/sampling now gets the same treatment beam search always had.

The penalty is applied identically whether the chain is on logits or log-probs, so a given
`penalty` value has the same effect across models and decoding modes.

## Status

This is the standalone form of
[huggingface/transformers#48642](https://github.com/huggingface/transformers/pull/48642).
It is verified against that branch's `normalize=True` implementation (tests in `tests/`) and
against the paper's gauge-probe harness: run through `generate()` on gpt2 with the processor
passed via `logits_processor=`, the output streams agree token-for-token with the paper's
stored `fix` streams under ±5 logit shifts.

Requires `transformers>=4.40` and `torch>=2.0`. No `trust_remote_code`.

## License

Apache-2.0.
