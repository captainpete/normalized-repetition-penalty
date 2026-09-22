import pytest
import torch

from normalized_repetition_penalty import (
    NormalizedEncoderRepetitionPenaltyLogitsProcessor,
    NormalizedRepetitionPenaltyLogitsProcessor,
)

try:  # the upstream PR branch (huggingface/transformers#48642) carries a `normalize=True` flag
    import inspect

    from transformers import EncoderRepetitionPenaltyLogitsProcessor, RepetitionPenaltyLogitsProcessor

    HAVE_UPSTREAM_FLAG = "normalize" in inspect.signature(RepetitionPenaltyLogitsProcessor.__init__).parameters
except ImportError:  # pragma: no cover
    HAVE_UPSTREAM_FLAG = False


def uniform_logits(batch_size, vocab_size):
    return torch.ones((batch_size, vocab_size), dtype=torch.float) / vocab_size


def test_repetition_penalty_normalized():
    input_ids = torch.tensor([[0, 1], [5, 0]], dtype=torch.long)
    vocab_size = 10
    scores = uniform_logits(2, vocab_size)
    scores[0, 0] = -(1 / vocab_size)
    scores[1, 5] = 4 / vocab_size

    processed = NormalizedRepetitionPenaltyLogitsProcessor(penalty=2.0)(input_ids, scores)

    # seen tokens are penalized in log-probability space, then shifted back
    log_probs = torch.log_softmax(scores, dim=-1)
    lse = torch.logsumexp(scores, dim=-1)
    assert processed[0, 0].item() == pytest.approx(log_probs[0, 0].item() * 2 + lse[0].item(), abs=1e-4)
    assert processed[0, 1].item() == pytest.approx(log_probs[0, 1].item() * 2 + lse[0].item(), abs=1e-4)
    assert processed[1, 5].item() == pytest.approx(log_probs[1, 5].item() * 2 + lse[1].item(), abs=1e-4)

    # unseen tokens are bit-identical
    assert processed[0, 2].item() == scores[0, 2].item()
    assert torch.equal(processed[1, [1, 2, 3, 4, 6, 7, 8, 9]], scores[1, [1, 2, 3, 4, 6, 7, 8, 9]])

    # not in-place
    assert not torch.all(scores == processed)


def test_repetition_penalty_normalized_reward():
    input_ids = torch.tensor([[0, 1]], dtype=torch.long)
    scores = uniform_logits(1, 10)
    scores[0, 0] = 0.5

    processed = NormalizedRepetitionPenaltyLogitsProcessor(penalty=0.5)(input_ids, scores)

    log_probs = torch.log_softmax(scores, dim=-1)
    lse = torch.logsumexp(scores, dim=-1)
    assert processed[0, 0].item() == pytest.approx(log_probs[0, 0].item() * 0.5 + lse[0].item(), abs=1e-4)
    assert processed[0, 1].item() == pytest.approx(log_probs[0, 1].item() * 0.5 + lse[0].item(), abs=1e-4)

    # rewarded tokens move toward 0 in log-prob space, i.e. their probability increases
    probs = torch.softmax(processed, dim=-1)
    assert probs[0, 0] > torch.softmax(scores, dim=-1)[0, 0]


def test_gauge_invariance():
    """softmax is invariant to a constant logit shift; the normalized penalty must be too."""
    input_ids = torch.tensor([[0, 1]], dtype=torch.long)
    scores = (torch.arange(10, dtype=torch.float) / 10 - 0.5).unsqueeze(0)
    shifted = scores + 100.0

    proc = NormalizedRepetitionPenaltyLogitsProcessor(penalty=1.3)
    probs = torch.softmax(proc(input_ids, scores), dim=-1)
    shifted_probs = torch.softmax(proc(input_ids, shifted), dim=-1)
    assert torch.allclose(probs, shifted_probs, rtol=1e-4, atol=1e-6)

    # and the output stays on the input scale: shift in == shift out
    assert torch.allclose(proc(input_ids, shifted) - proc(input_ids, scores), torch.full_like(scores, 100.0), atol=1e-4)


def test_log_prob_input_is_fixed_point():
    """Under beam search the chain runs on log-probs; there lse == 0 and the shift is a no-op."""
    input_ids = torch.tensor([[0, 1]], dtype=torch.long)
    logits = torch.randn(1, 10)
    log_probs = torch.log_softmax(logits, dim=-1)

    proc = NormalizedRepetitionPenaltyLogitsProcessor(penalty=1.3)
    # same distribution whether fed logits or log-probs
    assert torch.allclose(
        torch.softmax(proc(input_ids, logits), dim=-1), torch.softmax(proc(input_ids, log_probs), dim=-1), atol=1e-6
    )
    # and on log-prob input it is literally the stock rule (multiply, since all entries are <= 0)
    expected = log_probs.clone()
    expected[0, [0, 1]] = log_probs[0, [0, 1]] * 1.3
    assert torch.allclose(proc(input_ids, log_probs), expected, atol=1e-6)


def test_out_of_vocab_ids_are_ignored():
    """Added tokens can have ids >= vocab_size (huggingface/transformers#48846)."""
    input_ids = torch.tensor([[0, 12]], dtype=torch.long)
    scores = torch.randn(1, 10)
    processed = NormalizedRepetitionPenaltyLogitsProcessor(penalty=2.0)(input_ids, scores)
    assert processed.shape == scores.shape
    assert torch.equal(processed[0, 1:], scores[0, 1:])
    assert processed[0, 0] != scores[0, 0]


def test_prompt_ignore_length():
    input_ids = torch.tensor([[3, 4, 0, 1]], dtype=torch.long)
    scores = torch.randn(1, 10)
    full = NormalizedRepetitionPenaltyLogitsProcessor(penalty=2.0)(input_ids, scores)
    partial = NormalizedRepetitionPenaltyLogitsProcessor(penalty=2.0, prompt_ignore_length=2)(input_ids, scores)
    assert torch.equal(partial[0, [3, 4]], scores[0, [3, 4]])
    assert full[0, 3] != scores[0, 3]
    assert torch.equal(partial[0, [0, 1]], full[0, [0, 1]])


def test_3d_scores():
    input_ids = torch.tensor([[0, 1], [5, 0]], dtype=torch.long)
    scores = torch.randn(2, 3, 10)
    proc = NormalizedRepetitionPenaltyLogitsProcessor(penalty=2.0)
    processed = proc(input_ids, scores)
    assert torch.equal(processed[:, :-1, :], scores[:, :-1, :])
    assert torch.equal(processed[:, -1, :], proc(input_ids, scores[:, -1, :]))


def test_encoder_repetition_penalty_normalized():
    input_ids = torch.tensor([[0, 1], [5, 0]], dtype=torch.long)
    vocab_size = 10
    scores = uniform_logits(2, vocab_size)
    scores[0, 0] = -(1 / vocab_size)
    scores[1, 5] = 4 / vocab_size

    processed = NormalizedEncoderRepetitionPenaltyLogitsProcessor(penalty=2.0, encoder_input_ids=input_ids)(
        input_ids, scores
    )

    # prompt tokens are rewarded in log-prob space regardless of the sign of their raw logit
    log_probs = torch.log_softmax(scores, dim=-1)
    lse = torch.logsumexp(scores, dim=-1)
    assert processed[0, 0].item() == pytest.approx(log_probs[0, 0].item() / 2 + lse[0].item(), abs=1e-4)
    assert processed[1, 5].item() == pytest.approx(log_probs[1, 5].item() / 2 + lse[1].item(), abs=1e-4)
    assert processed[0, 2].item() == scores[0, 2].item()


def test_encoder_gauge_invariance():
    input_ids = torch.tensor([[0, 1]], dtype=torch.long)
    scores = (torch.arange(10, dtype=torch.float) / 10 - 0.5).unsqueeze(0)
    proc = NormalizedEncoderRepetitionPenaltyLogitsProcessor(penalty=2.0, encoder_input_ids=input_ids)
    probs = torch.softmax(proc(input_ids, scores), dim=-1)
    shifted_probs = torch.softmax(proc(input_ids, scores + 100.0), dim=-1)
    assert torch.allclose(probs, shifted_probs, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("penalty", [0.7, 1.0, 1.3, 2.0])
def test_invalid_penalty_and_valid_penalty(penalty):
    NormalizedRepetitionPenaltyLogitsProcessor(penalty=penalty)
    with pytest.raises(ValueError):
        NormalizedRepetitionPenaltyLogitsProcessor(penalty=-penalty)
    with pytest.raises(ValueError):
        NormalizedRepetitionPenaltyLogitsProcessor(penalty=int(penalty))


@pytest.mark.skipif(not HAVE_UPSTREAM_FLAG, reason="needs the huggingface/transformers#48642 branch")
@pytest.mark.parametrize("penalty", [0.7, 1.3, 2.0])
def test_matches_upstream_pr_branch(penalty):
    torch.manual_seed(0)
    for offset in (-100.0, 0.0, 840.0):
        scores = torch.randn(4, 50, dtype=torch.float32) * 5 + offset
        input_ids = torch.randint(0, 50, (4, 12))
        ours = NormalizedRepetitionPenaltyLogitsProcessor(penalty=penalty)(input_ids, scores)
        theirs = RepetitionPenaltyLogitsProcessor(penalty=penalty, normalize=True)(input_ids, scores)
        assert torch.allclose(ours, theirs, atol=1e-5, rtol=0), (penalty, offset)

        enc = torch.randint(0, 50, (4, 6))
        ours = NormalizedEncoderRepetitionPenaltyLogitsProcessor(penalty=penalty, encoder_input_ids=enc)(
            input_ids, scores
        )
        theirs = EncoderRepetitionPenaltyLogitsProcessor(penalty=penalty, encoder_input_ids=enc, normalize=True)(
            input_ids, scores
        )
        assert torch.allclose(ours, theirs, atol=1e-5, rtol=0), (penalty, offset)
