# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

"""
Simple Rejection Sampler for NPU - Extracted from vLLM RejectionSampler
Implements modified rejection sampling as described in "Accelerating Large
Language Model Decoding with Speculative Sampling"
https://arxiv.org/pdf/2302.01318.pdf.
"""

from typing import Dict, List, Optional, Union
import torch
from functools import cached_property
from transformers import AutoTokenizer

# Optional NPU support
try:
    import torch_npu

    HAS_NPU = True
except ImportError:
    HAS_NPU = False

# Constants
FP32_EPS = 2 ** -24
UNINITIALIZED_CACHED_K_NUM = -1


class SimpleRejectSampler:
    """Apply modified rejection sampling for speculative decoding on NPU.

    This is a simplified, NPU-optimized version extracted from vLLM's RejectionSampler
    that removes vLLM dependencies while preserving the core algorithm logic.
    """

    def __init__(self,
                 strict_mode: bool = False,
                 device: Union[str, torch.device] = "npu:0",
                 dtype: torch.dtype = torch.float32):
        """Create a rejection sampler.

        Args:
            strict_mode: Whether or not to perform shape/device/dtype checks
                during sampling. This catches correctness issues but adds
                nontrivial latency.
            device: Device to run computations on (NPU device)
            dtype: Data type for probability computations
        """
        self._strict_mode = strict_mode
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.probs_dtype = dtype
        self.token_id_dtype = torch.int64

        # NOTE: A "bonus token" is accepted iff all proposal tokens are
        # accepted. There is always only one possible bonus token.
        self._num_bonus_tokens = 1

        # NPU optimizations: cached tensors to avoid repeated allocations
        self.int64_neg_one = torch.tensor(-1, device=self.device, dtype=self.token_id_dtype)
        self.cached_indices = None
        self.cached_k_tensor = None
        self.cached_k = UNINITIALIZED_CACHED_K_NUM

        # Metrics tracking
        self.num_accepted_tokens = torch.tensor(0, dtype=torch.long, device=self.device)
        self.num_emitted_tokens = torch.tensor(0, dtype=torch.long, device=self.device)
        self.num_draft_tokens = 0
        self.enable_spec_metric = True

    def forward(
            self,
            target_with_bonus_probs: torch.Tensor,
            bonus_token_ids: torch.Tensor,
            draft_probs: torch.Tensor,
            draft_token_ids: torch.Tensor,
            seeded_seqs: Optional[Dict[int, torch.Generator]] = None,
    ) -> torch.Tensor:
        """Sample token ids using rejection sampling. This accepts or rejects
        tokens proposed by the draft model using the probability of each token
        according to the draft and target models.

        In the worst case where all draft tokens are rejected, it is guaranteed
        one correct token will be emitted.

        In the case where all draft tokens are accepted, a bonus token will be
        accepted as its cheap to have the target model score this speculative
        sequence.

        Args:
            target_with_bonus_probs: The probability distribution
                over token ids given context according to the target model.
            shape = [batch_size, num_speculative_tokens + 1, vocab_size]

            bonus_token_ids: The "bonus" token ids that are accepted iff all
                speculative tokens in a sequence are accepted.
            shape = [batch_size, num_bonus_tokens]

            draft_probs: The probability distribution over token ids given
                context according to the draft model.
            shape = [batch_size, num_speculative_tokens, vocab_size]

            draft_token_ids: The token ids that were sampled from the draft
                probabilities.
            shape = [batch_size, num_speculative_tokens]

            seeded_seqs: Dict of batch row index to torch generator, for
                sequences using seeded generation.

        Returns:
            output_token_ids: The token ids sampled via rejection sampling,
                or -1 if unable to sample a token because the previous token
                was rejected.
            shape = [batch_size, num_speculative_tokens + num_bonus_tokens]
        """
        # Only perform shape/dtype/device checking in strict mode, as it adds
        # overhead.
        if self._strict_mode:
            self._raise_if_incorrect_input(target_with_bonus_probs,
                                           draft_token_ids, bonus_token_ids,
                                           draft_probs)

        batch_size, k, _ = draft_probs.shape

        # batch_size = 0 when all requests in the batch are
        # non_spec requests. In this case, output_token_ids is
        # just an empty tensor.
        if batch_size == 0:
            return torch.empty(0, k + 1, device=draft_probs.device, dtype=self.token_id_dtype)

        # Perform modified rejection sampling
        accepted, recovered_token_ids = self._batch_modified_rejection_sampling(
            target_with_bonus_probs[:, :-1],
            draft_probs,
            draft_token_ids,
            seeded_seqs,
        )

        output_token_ids = self._create_output(
            accepted,
            recovered_token_ids,
            draft_token_ids,
            bonus_token_ids,
        )

        return output_token_ids

    def _batch_modified_rejection_sampling(
            self,
            target_probs: torch.Tensor,  # [batch_size, k, vocab_size]
            draft_probs: torch.Tensor,  # [batch_size, k, vocab_size]
            draft_token_ids: torch.Tensor,  # [batch_size, k]
            seeded_seqs: Optional[Dict[int, torch.Generator]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform modified rejection sampling on each sequence.

        Returns:
            A tuple of two tensors:
            0: A bool tensor of which tokens in each sequence is accepted.
                shape = [batch_size, k]
            1: Token ids sampled from a recovered distribution, to be used
                when a token is rejected.
                shape = [batch_size, k]
        """
        batch_size, k, vocab_size = draft_probs.shape

        # shape [batch_size, k]
        accepted = self._get_accepted(target_probs, draft_probs,
                                      draft_token_ids, seeded_seqs)

        recovered_probs = self._get_recovered_probs(
            target_probs, draft_probs).reshape(batch_size * k, vocab_size)

        # NOTE: the recovered_probs are overwritten by this method.
        recovered_token_ids = _multinomial(
            recovered_probs,
            num_samples=1,
            k=k,
            seeded_seqs=seeded_seqs or {},
        ).reshape(batch_size, k)

        return accepted, recovered_token_ids

    def _create_uniform_samples(self,
                                seeded_seqs: Optional[Dict[int, torch.Generator]],
                                batch_size: int, k: int,
                                device: torch.device) -> torch.Tensor:
        """
        Generates a batch of uniform random samples, with optional seeding
        for specific sequences.

        This method creates a tensor of shape `(batch_size, k + 1)` filled
        with uniform random values in the range [0, 1). If `seeded_seqs`
        is provided, the sequences corresponding to specific indices
        will be generated using the provided `torch.Generator` for
        reproducibility. The other sequences will be generated without
        a seed.

        Args:
            seeded_seqs : Optional[Dict[int, torch.Generator]]
                A dictionary mapping indices in the batch to
                `torch.Generator` objects. If `None`, all samples are
                generated without a seed.
            batch_size : int
                The number of sequences to generate.
            k : int
                The number of random samples per sequence.
            device : torch.device
                The device on which to allocate the tensor.

        Returns:
            uniform_rand : torch.Tensor
                A tensor of shape `(batch_size, k + 1)` containing uniform
                random values in the range [0, 1).
        """
        if not seeded_seqs:
            return torch.rand(batch_size, k + 1, device=device)

        uniform_rand = torch.empty(batch_size, k + 1, device=device)

        non_seeded_indices = []
        for idx in range(batch_size):
            generator = seeded_seqs.get(idx)
            if generator is None:
                non_seeded_indices.append(idx)
            else:
                uniform_rand[idx, :] = torch.rand(1,
                                                  k + 1,
                                                  dtype=self.probs_dtype,
                                                  device=device,
                                                  generator=generator)
        if non_seeded_indices:
            uniform_rand[non_seeded_indices, :] = torch.rand(
                len(non_seeded_indices),
                k + 1,
                dtype=self.probs_dtype,
                device=device)
        return uniform_rand

    def _get_accepted(
            self,
            target_probs: torch.Tensor,  # [batch_size, k, vocab_size]
            draft_probs: torch.Tensor,  # [batch_size, k, vocab_size]
            draft_token_ids: torch.Tensor,  # [batch_size, k]
            seeded_seqs: Optional[Dict[int, torch.Generator]],
    ) -> torch.Tensor:
        r"""Create bool matrix over the proposed draft tokens. If
        True, then a token can be accepted, else it should be
        rejected.

        Given $q(\hat{x}_{n+1}|x_1, \dots, x_n)$, the probability of
        $\hat{x}_{n+1}$ given context $x_1, \dots, x_n$ according
        to the target model, and $p(\hat{x}_{n+1}|x_1, \dots, x_n)$, the
        same conditional probability according to the draft model, the token
        is accepted with probability:

        $$
        \min\left(1, \frac{q(\hat{x}_{n+1}|x_1, \dots, x_n)}
                        {p(\hat{x}_{n+1}|x_1, \dots, x_n)}\right)
        $$

        This implementation does not apply causality. When using the output,
        if a token is rejected, subsequent tokens should not be used.

        Returns a bool tensor of shape [batch_size, k] specifying which tokens
        are accepted.
        """
        batch_size, k, _ = draft_probs.shape

        uniform_rand = self._create_uniform_samples(seeded_seqs, batch_size,
                                                    k - 1, target_probs.device)

        # NPU optimization: replace index_select with gather for better performance
        draft_token_ids = draft_token_ids.view(batch_size, k, 1)
        selected_draft_probs = torch.gather(draft_probs, dim=-1, index=draft_token_ids).view(batch_size, k)
        selected_target_probs = torch.gather(target_probs, dim=-1, index=draft_token_ids).view(batch_size, k)

        # NPU optimization: use in-place operations
        selected_target_probs.div_(selected_draft_probs).clamp_max_(1)

        accepted = uniform_rand < selected_target_probs
        return accepted

    def _get_recovered_probs(
            self,
            target_probs: torch.Tensor,  # [batch_size, k, vocab_size]
            draft_probs: torch.Tensor,  # [batch_size, k, vocab_size]
    ) -> torch.Tensor:
        r"""Create a probability distribution for each proposed token which can
        be sampled if the proposed token is rejected.

        When this routine is applied sequentially, the true distribution of the
        target model is recovered (within hardware numerics).

        The probability distribution used in this rejection case is constructed
        as follows. Given $q(x|x_1, \dots, x_n)$, the probability of
        $x$ given context $x_1, \dots, x_n$ according to the target
        model and $p(x|x_1, \dots, x_n)$, the same conditional probability
        according to the draft model:

        $$
        x_{n+1} \sim (q(x|x_1, \dots, x_n) - p(x|x_1, \dots, x_n))_+
        $$

        where $(f(x))_+$ is defined as:

        $$
        (f(x))_+ = \frac{\max(0, f(x))}{\sum_x \max(0, f(x))}
        $$

        See https://github.com/vllm-project/vllm/pull/2336 for a visualization
        of the draft, target, and recovered probability distributions.

        Returns a tensor of shape [batch_size, k, vocab_size].

        Note:
            This batches operations on GPU and thus constructs the recovered
            distribution for all tokens, even if they are accepted. This causes
            division-by-zero errors, so we use self._smallest_positive_value to
            avoid that. This introduces some drift to the distribution.
        """
        _, k, _ = draft_probs.shape

        # NPU optimization: use inplace operations for better performance
        target_probs.sub_(draft_probs).clamp_min_(self._smallest_positive_value)
        recovered_probs = target_probs / torch.sum(target_probs, dim=-1).view(-1, k, 1)

        return recovered_probs

    def _create_output(
            self,
            accepted: torch.Tensor,  # [batch_size, k]
            substitute_token_ids: torch.Tensor,  # [batch_size, k]
            draft_token_ids: torch.Tensor,  # [batch_size, k]
            bonus_token_ids: torch.Tensor,  # [batch_size]
    ) -> torch.Tensor:
        """Format output. Returns a matrix of token ids. When
        a token is rejected via sampling, all subsequent token ids are
        set to -1 for the sequence.

        Args:
            accepted: A boolean tensor indicating if the corresponding
            draft token in draft_token_ids should be accepted or not.
            substitute_token_ids: A tensor of token_ids that can be used
            as substitutes for the draft token ids if the proposed token
            is rejected.
            draft_token_ids: A tensor of token ids speculated by the
            draft model.
            bonus_token_ids: Token ids to use as the bonus token if
            all the draft tokens are accepted.
        Returns:
            A tensor containing the accepted token ids. The shape of the
            tensor is [batch_size, k + num_bonus_tokens]
        """
        batch_size, k = substitute_token_ids.shape
        bonus_token_ids = bonus_token_ids.squeeze()

        # Determine the index of the first False value for each row.
        accepted_equal_zero_mask = accepted == 0
        limits = accepted_equal_zero_mask.max(1).indices

        # NPU optimization: cache tensor to avoid repeated allocations
        mask = accepted_equal_zero_mask.any(1)
        if self.cached_k_tensor is None or self.cached_k != k:
            self.cached_k_tensor = torch.tensor(k, dtype=limits.dtype, device=limits.device)
            self.cached_k = k
        limits = torch.where(mask, limits, self.cached_k_tensor)

        # Create masks using the indices.
        if self.cached_indices is None or self.cached_indices.shape[1] != k:
            self.cached_indices = torch.arange(k, device=accepted.device).unsqueeze(0)
        accepted_mask = self.cached_indices < limits.unsqueeze(1)
        after_false_mask = self.cached_indices == limits.unsqueeze(1)

        # Create an extended output tensor
        output_with_bonus_tokens = torch.full(
            (batch_size, k + self._num_bonus_tokens),
            fill_value=-1,
            dtype=self.token_id_dtype,
            device=accepted.device)
        output = output_with_bonus_tokens[:, :k]

        # Fill in the first k columns of the output tensor using masks and data tensors.
        # NPU optimization: remove index select, use torch.where directly
        torch.where(accepted_mask,
                    draft_token_ids,
                    self.int64_neg_one,
                    out=output)

        # Fill the last column.
        # We check output directly as accepted may have True values inconsistent
        # with causal acceptance.
        # NPU optimization: avoid memory copy
        output_with_bonus_tokens[:, -1] = torch.where(output[:, -1] != self.int64_neg_one,
                                                      bonus_token_ids, self.int64_neg_one)

        # Fill the recovered token ids.
        output.mul_(~after_false_mask).add_(
            substitute_token_ids.mul(after_false_mask))

        # NPU optimization: disable log metric when disable_logprobs is True.
        if self.enable_spec_metric:
            self.num_accepted_tokens += accepted.sum()
            self.num_emitted_tokens += (output_with_bonus_tokens != -1).sum()
            self.num_draft_tokens += batch_size * k

        return output_with_bonus_tokens

    @cached_property
    def _smallest_positive_value(self) -> float:
        """Return the smallest positive value representable by the probs dtype.
        This value is used when constructing a distribution from which to sample
        recovered tokens in the first rejection case.

        See _get_recovered_probs for more details

        Note that this isn't actually the smallest positive value representable
        by float32, but the smallest positive normal value.
        See https://en.wikipedia.org/wiki/Subnormal_number for more information.
        """
        return torch.finfo(self.probs_dtype).tiny

    def _raise_if_incorrect_input(
            self,
            target_with_bonus_probs: torch.Tensor,
            draft_token_ids: torch.Tensor,
            bonus_token_ids: torch.Tensor,
            draft_probs: torch.Tensor,
    ) -> None:
        """Raise exceptions if input is malformed."""
        batch_size, k, vocab_size = draft_probs.shape

        # Check shapes
        assert target_with_bonus_probs.shape == (batch_size, k + 1, vocab_size), \
            f"target_with_bonus_probs shape mismatch: expected {(batch_size, k + 1, vocab_size)}, got {target_with_bonus_probs.shape}"
        assert draft_token_ids.shape == (batch_size, k), \
            f"draft_token_ids shape mismatch: expected {(batch_size, k)}, got {draft_token_ids.shape}"
        assert bonus_token_ids.shape[0] == batch_size, \
            f"bonus_token_ids batch size mismatch: expected {batch_size}, got {bonus_token_ids.shape[0]}"

        # Check devices
        assert target_with_bonus_probs.device == draft_probs.device
        assert draft_token_ids.device == draft_probs.device
        assert bonus_token_ids.device == draft_probs.device

        # Check dtypes
        assert target_with_bonus_probs.dtype == draft_probs.dtype
        assert draft_token_ids.dtype == self.token_id_dtype
        assert bonus_token_ids.dtype == self.token_id_dtype

        # Check token bounds
        assert torch.all(draft_token_ids >= 0)
        assert torch.all(bonus_token_ids >= 0)


# NPU-optimized multinomial sampling function
def _multinomial(
        probs: torch.Tensor,
        num_samples: int,
        k: int,
        seeded_seqs: Dict[int, torch.Generator],
) -> torch.Tensor:
    """NPU-optimized multinomial sampling.

    torch.multinomial forces a GPU<->CPU sync.
    Therefore, we use an optimized implementation instead that skips the sync.
    Note that we always sample with replacement.
    probs will be modified in place, but this is fine, as we pass
    in a copy already.
    """
    if num_samples > 1:
        # This is equivalent to torch.repeat_interleaved (which also
        # forces a GPU<->CPU sync).
        probs = probs[:, None, :].expand(probs.shape[0], num_samples,
                                         probs.shape[1]).contiguous().view(
            -1, probs.shape[1])
    q = torch.empty_like(probs)
    if not seeded_seqs:
        q.exponential_(1.0)
    else:
        # NPU optimization: handle non-seeded indices more efficiently
        non_seeded_indices: List[int] = []
        start = 0
        for idx in range(len(q) // k):
            end = start + k
            generator = seeded_seqs.get(idx)
            if generator is None:
                non_seeded_indices.extend(list(range(start, end)))
            else:
                q[start:end].exponential_(1.0, generator=generator)
            start = end
        q[non_seeded_indices].exponential_(1.0)

    # NPU optimization: add FP32_EPS to avoid division by zero
    q.add_(FP32_EPS)
    return probs.div_(q).argmax(dim=1).view(-1, num_samples)


# Example usage and testing functions
def create_test_tensors(batch_size: int = 1, k: int = 1, vocab_size: int = 129280, device: str = "cpu"):
    """Create test tensors for SimpleRejectSampler."""
    device = torch.device(device)

    # Create random probability distributions
    target_with_bonus_logits = torch.load("20251107_090619_255_rej_target_logits_from_npu1.pt", map_location=device)
    draft_logtis = torch.load("20251107_090619_255_rej_draft_logits_from_npu1.pt", map_location=device)

    target_with_bonus_logits = torch.reshape(target_with_bonus_logits, (batch_size, k + 1, vocab_size))
    draft_logtis = torch.reshape(draft_logtis, (batch_size, k, vocab_size))
    print(f"target_with_bonus_logits = f{target_with_bonus_logits.shape}")
    print(f"draft_logtis = f{draft_logtis.shape}")

    target_with_bonus_probs = torch.softmax(target_with_bonus_logits, dim=-1)
    draft_probs = torch.softmax(draft_logtis, dim=-1)

    # Create random token ids
    draft_token_ids = torch.argmax(draft_probs, dim=-1)
    bonus_token_ids = torch.argmax(target_with_bonus_probs[:, -1, :], dim=-1)

    print(f"draft_token_ids = f{draft_token_ids}")
    print(f"bonus_token_ids = f{bonus_token_ids}")

    return target_with_bonus_probs, draft_probs, draft_token_ids, bonus_token_ids


def test_simple_reject_sampler():
    """Test function for SimpleRejectSampler."""
    print("Testing SimpleRejectSampler...")

    # Create sampler
    sampler = SimpleRejectSampler(device="cpu")

    # Create test data
    target_probs, draft_probs, draft_tokens, bonus_tokens = create_test_tensors()

    # Run sampling
    output = sampler.forward(target_probs, bonus_tokens, draft_probs, draft_tokens)

    tokenizer = AutoTokenizer.from_pretrained(
        "d:\\Onebox\\重点工作\\日常工作\\2025-11-06 拒绝采样问题定位\\tensor_log\\tensor_log")

    print(f"Input shapes:")
    print(f"  target_probs: {target_probs.shape}")
    print(f"  draft_probs: {draft_probs.shape}")
    print(f"  draft_tokens: {draft_tokens.shape}")
    print(f"  bonus_tokens: {bonus_tokens.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output tokens:\n{output}")
    print(f"Accepted tokens: {sampler.num_accepted_tokens.item()}")
    print(f"Emitted tokens: {sampler.num_emitted_tokens.item()}")
    print("Test completed successfully!")

    for token in output.flatten().tolist():
        if token != -1:
            print("new text:", tokenizer.decode(token))

    print("参考输出：", tokenizer.decode(64740))

if __name__ == "__main__":
    test_simple_reject_sampler()
