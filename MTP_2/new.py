# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import numpy as np

from .decoding_policy import DecodingPolicy
from ..plugin import Plugin
from ....utils.log.logging import logger
from ....utils.tensor import backend


class MtpPlugin(Plugin):
    def __init__(self, generator_backend, cache_manager, input_manager, output_filter, plugin_data_param, **kwargs):
        super().__init__()
        self.pad_token_id = 0
        self.generator_backend = generator_backend
        self.model_wrapper = self.generator_backend.model_wrapper
        self.cache_manager = cache_manager
        self.input_manager = input_manager
        self.cache = self.input_manager.cache
        self.output_filter = output_filter
        kv_device = self.model_wrapper.device
        kv_dtype = self.cache_manager.dtype
        device_and_type = (kv_device, kv_dtype)
        self.plugin_data_param = plugin_data_param
        self.num_speculative_tokens = kwargs.get('num_speculative_tokens')
        self.model_role = kwargs.get('model_role', 'standard')
        self.decoding_policy = DecodingPolicy(generator_backend, self.input_manager, self.model_wrapper,
                                              self.num_speculative_tokens, device_and_type, plugin_data_param,
                                              self.model_role)
        self.rank = generator_backend.rank
        self.mapping = self.model_wrapper.mapping
        self.cp_rank = 0
        self.sp_rank = 0
        self.sp_size = 1
        self.cp_size = 1
        if self.mapping.has_attn_inner_sp():
            self.sp_rank = self.mapping.attn_inner_sp.rank
            self.sp_size = self.mapping.attn_inner_sp.group_size
        if self.mapping.has_attn_cp():
            self.cp_rank = self.mapping.attn_cp.rank
            self.cp_size = self.mapping.attn_cp.group_size

    def model_inputs_update(self, model_inputs, input_metadata, sampling_metadata, cache_ids, input_len_mask, **kwargs):
        logger.info(f"[MTP] model_inputs_update started with cache_ids={cache_ids}, is_prefill={input_metadata.is_prefill}")
        hit_mask = kwargs.get('hit_mask')
        if hit_mask is not None:
            logger.info(f"[MTP] Hit mask provided with {np.sum(hit_mask)} hits out of {len(hit_mask)} sequences")
        
        model_inputs, q_len, attention_mask = (
            self.decoding_policy.handle_input(model_inputs, input_metadata, cache_ids, hit_mask=hit_mask))
        input_len_mask = (q_len, attention_mask)
        logger.info(f"[MTP] Processed model inputs, query length shape: {q_len.shape if hasattr(q_len, 'shape') else len(q_len)}")

        self.decoding_policy.handle_sampling(sampling_metadata)
        logger.info("[MTP] model_inputs_update completed")

        return model_inputs, input_len_mask

    def prepare_masks_for_filling(
        self,
        model_inputs,
        current_dp_sequence_ids,
        current_all_sequence_ids,
        last_all_sequence_ids
    ):
        masks = {}
        if last_all_sequence_ids is not None:
            speculative_length = self.num_speculative_tokens + 1
            hit_mask = np.isin(current_dp_sequence_ids, last_all_sequence_ids)
            if hit_mask.any():
                hit_sequence_ids = current_dp_sequence_ids[hit_mask]
                hit_indices = np.where(hit_sequence_ids[:, None] == last_all_sequence_ids[None, :])[1]
                hit_mask_per_token = np.repeat(hit_mask, speculative_length)
                hit_size = len(hit_indices)
                hit_increments = np.repeat(
                    np.arange(speculative_length).reshape(-1, 1), hit_size, axis=1).transpose().reshape(-1)
                hit_indices_per_token = np.repeat(hit_indices * speculative_length, speculative_length) + hit_increments
                masks['hit_mask'] = hit_mask
                masks['hit_mask_tensor'] = self.generator_backend.to_tensor(hit_mask)
                all_hit_mask = np.isin(current_all_sequence_ids, last_all_sequence_ids)
                all_hit_sequence_ids = current_all_sequence_ids[all_hit_mask]
                masks['all_hit_mask_tensor'] = self.generator_backend.to_tensor(all_hit_mask)
                all_hit_indices = np.where(all_hit_sequence_ids[:, None] == last_all_sequence_ids[None, :])[1]
                masks['all_hit_indices'] = all_hit_indices
                masks['hit_indices'] = hit_indices
                masks['hit_indices_per_token_tensor'] = self.generator_backend.to_tensor(hit_indices_per_token)
                masks['hit_mask_per_token_tensor'] = self.generator_backend.to_tensor(hit_mask_per_token)
                masks['hit_speculative_length'] = np.full(len(all_hit_indices), speculative_length)
                masks['hit_dp_speculative_length'] = np.full(len(hit_indices), speculative_length)
                hit_arange = np.arange(hit_size)
                masks['hit_arange_tensor'] = self.generator_backend.to_tensor(hit_arange)
                hit_mask_mod = self.generator_backend.to_tensor(
                    np.arange(len(hit_mask_per_token)) % speculative_length != 0)
                masks['hit_mask_mod'] = hit_mask_mod
                hit_block_tables = model_inputs.block_tables_array[hit_mask]
                candidate_slots = self.input_manager.all_slots[hit_block_tables].reshape(hit_size, -1)
                hit_block_indices = np.repeat(hit_arange, speculative_length)
                masks['candidate_slots'] = self.generator_backend.to_tensor(candidate_slots)
                masks['hit_block_indices'] = self.generator_backend.to_tensor(hit_block_indices)
                masks['hit_increments'] = self.generator_backend.to_tensor(hit_increments)
        return masks

    def fill_in_model_result(self, input_metadata, model_inputs, model_kwargs, model_output_wrapper, filling_masks, cache_ids):
        logger.info(f"[MTP] fill_in_model_result started with cache_ids={cache_ids}")
        speculative_length = self.num_speculative_tokens + 1
        logger.info(f"[MTP] Speculative length (num_tokens + 1): {speculative_length}")
        
        mtp_model_inputs = model_kwargs.get('sub_model_inputs')
        hidden_states = model_kwargs.get('hidden_states')
        lm_head_local_dp = model_kwargs.get('lm_head_local_dp', None)
        input_lengths_sp = model_kwargs.get('input_lengths_sp', None)
        hit_mask = filling_masks.get('hit_mask')
        
        if hit_mask is not None:
            hit_count = np.sum(hit_mask)
            logger.info(f"[MTP] Processing {hit_count} hit sequences out of {len(hit_mask)} total sequences")
            model_output_hidden_states = model_output_wrapper.model_output.hidden_states
            sampling_output = model_output_wrapper.sampling_output

            # Move the hit_token_ids of mtp model to device.
            hit_indices = filling_masks.get('hit_indices')
            hit_token_ids = sampling_output.token_ids[hit_indices]
            hit_token_ids_cols = hit_token_ids.shape[1]
            logger.info(f"[MTP] Extracted hit token IDs, shape: {hit_token_ids.shape}, expected cols: {speculative_length}")
            
            if hit_token_ids_cols < speculative_length:
                padding_width = ((0, 0), (0, speculative_length - hit_token_ids_cols))
                hit_token_ids = np.pad(hit_token_ids, padding_width, 'constant', constant_values=0)
                logger.info(f"[MTP] Padded hit token IDs from {hit_token_ids_cols} to {speculative_length} columns")
            elif hit_token_ids_cols > speculative_length:
                logger.warning('[MTP] Found the number of output tokens exceeds the speculative length, '
                               'which will be truncated forcibly.')
                hit_token_ids = hit_token_ids[:, :speculative_length]
                logger.info(f"[MTP] Truncated hit token IDs from {hit_token_ids_cols} to {speculative_length} columns")
            
            hit_token_ids = self.generator_backend.to_tensor_async(hit_token_ids)

            # Get the device tensor of subtrahend of lmhead indices.
            # The default prefill_head_indices is calculated assuming the number of tokens is speculative_length.
            # The true head indices equals to the default prefill_head_indices minus head_indices_subtrahend.
            all_hit_indices = filling_masks.get('all_hit_indices')
            hit_speculative_length = filling_masks.get('hit_speculative_length')
            all_hit_num_tokens = sampling_output.num_new_tokens[all_hit_indices]
            head_indices_subtrahend = hit_speculative_length - all_hit_num_tokens
            head_indices_subtrahend = self.generator_backend.to_tensor_async(head_indices_subtrahend)

            # Move the hit_num_tokens and hit_num_tokens_per_token to device.
            hit_num_tokens = sampling_output.num_new_tokens[hit_indices]
            hit_num_tokens_tensor = self.generator_backend.to_tensor_async(hit_num_tokens)
            hit_num_tokens_per_token = backend.repeat_interleave(hit_num_tokens_tensor, speculative_length)

            # Get masks.
            hit_mask_tensor = filling_masks.get('hit_mask_tensor')
            hit_mask_per_token = filling_masks.get('hit_mask_per_token_tensor')
            all_hit_mask_tensor = filling_masks.get('all_hit_mask_tensor')
            hit_indices_per_token = filling_masks.get('hit_indices_per_token_tensor')
            hit_arange_tensor = filling_masks.get('hit_arange_tensor')
            hit_mask_mod = filling_masks.get('hit_mask_mod')

            # assignation
            logger.info("[MTP] Starting model input assignments for hit sequences")
            hit_token_ids_tensor = hit_token_ids.reshape(-1)
            mtp_model_inputs.input_ids[hit_mask_per_token] = hit_token_ids_tensor
            logger.info(f"[MTP] Updated input_ids with hit token IDs, reshaped to {hit_token_ids_tensor.shape}")
            
            mtp_model_inputs.prefill_head_indices[all_hit_mask_tensor] -= head_indices_subtrahend
            logger.info("[MTP] Adjusted prefill_head_indices by subtracting head_indices_subtrahend")
            
            hit_hidden_states = model_output_hidden_states[hit_indices_per_token]
            hidden_states[hit_mask_per_token] = hit_hidden_states
            logger.info(f"[MTP] Updated hidden_states with hit hidden states, shape: {hit_hidden_states.shape}")
            
            if lm_head_local_dp is not None and not (len(lm_head_local_dp) == 1 and lm_head_local_dp[0] == 0):
                hit_dp_speculative_length = filling_masks.get('hit_dp_speculative_length')
                lm_head_local_dp[hit_mask] -= \
                    self.generator_backend.to_tensor(hit_dp_speculative_length - hit_num_tokens)
                logger.info("[MTP] Adjusted lm_head_local_dp for distributed processing")

            model_inputs.position_ids[hit_mask_per_token] += hit_num_tokens_per_token
            model_inputs.context_length[hit_mask] += hit_num_tokens
            model_inputs.input_lengths[hit_mask_tensor] += hit_num_tokens_tensor
            logger.info(f"[MTP] Updated position_ids, context_length, and input_lengths by hit_num_tokens")
            
            model_inputs.max_seq_len = max(model_inputs.context_length)
            logger.info(f"[MTP] Updated max_seq_len to {model_inputs.max_seq_len}")
            if self.cache.scp_size == 1:
                logger.info("[MTP] Processing slots for single context parallelism (scp_size=1)")
                offset_start_indices = model_inputs.input_lengths[hit_mask_tensor] - speculative_length
                hit_increments = filling_masks.get('hit_increments')
                block_offsets = backend.repeat_interleave(offset_start_indices, speculative_length) + hit_increments
                candidate_slots = filling_masks.get('candidate_slots')
                hit_block_indices = filling_masks.get('hit_block_indices')
                model_inputs.slots[hit_mask_per_token] = candidate_slots[hit_block_indices, block_offsets]
                logger.info("[MTP] Updated slots using block offsets and candidate slots")
            else:
                logger.info(f"[MTP] Processing slots for multi context parallelism (scp_size={self.cache.scp_size})")
                model_inputs.cached_context_length[hit_mask] += hit_num_tokens
                indices = np.where(hit_mask)[0]
                logger.info(f"[MTP] Recalculating slots for {len(indices)} hit sequences")
                for _, idx in enumerate(indices):
                    sp_tokens, tmp_slots = self.decoding_policy.sp_token_and_slot_calc_by_context_length(model_inputs.cached_context_length[idx], self.cache.cached_seq_block_rank_id[cache_ids[idx]], input_metadata.block_tables[idx], speculative_length)
                    model_inputs.slots[idx * speculative_length:(idx + 1) * speculative_length] = self.generator_backend.to_tensor_async(tmp_slots)
                    model_inputs.sp_tokens[idx] = sp_tokens
                    input_lengths_sp[idx] = sp_tokens[self.cp_rank * self.sp_size:(self.cp_rank + 1) * self.sp_size][self.mapping.attn_inner_sp.rank]

            hit_mask_per_token[hit_mask_mod] = False
            model_inputs.input_ids[hit_mask_per_token] = hit_token_ids[hit_arange_tensor, hit_num_tokens_tensor - 1]
            logger.info("[MTP] fill_in_model_result completed successfully")

    def sample_preprocess(self, logits, result, sampling_metadata, input_metadata):
        logger.info("[MTP] sample_preprocess started")
        self.sampling_param = sampling_metadata
        self.decoding_policy.sampling_param = self.sampling_param
        self.input_metadata = input_metadata
        if isinstance(result, tuple):
            logits = result[0]
            logger.info(f"[MTP] Result is tuple, extracted logits with shape: {logits.shape}")
        else:
            logits = result
            logger.info(f"[MTP] Result is tensor, logits shape: {logits.shape}")
        
        if sampling_metadata is None or sampling_metadata.is_prefill:
            logger.info(f"[MTP] Prefill mode detected (is_prefill={sampling_metadata.is_prefill if sampling_metadata else 'N/A'}), returning logits directly")
            return logits
        
        draft_tokens = result[2].cpu()
        logger.info(f"[MTP] Draft tokens extracted from result[2], shape: {draft_tokens.shape}")
        
        all_sequence_ids = sampling_metadata.all_sequence_ids
        batch_size = len(all_sequence_ids)
        logger.info(f"[MTP] Processing batch_size={batch_size}, sequence_ids={all_sequence_ids}")

        logits_num_per_batch = [self.num_speculative_tokens + 1] * batch_size
        logger.info(f"[MTP] Speculative tokens per batch: {self.num_speculative_tokens}, total logits per batch: {logits_num_per_batch[0]}")
        
        input_ids_pad = self.decoding_policy.all_token_ids_padding(sampling_metadata, logits_num_per_batch,
                                                                   batch_size, draft_tokens)
        sampling_metadata.all_token_ids = input_ids_pad
        logger.info(f"[MTP] Token IDs padded, new all_token_ids shape: {input_ids_pad.shape}")

        req_id_new = np.concatenate([[req_id] * n for req_id, n in zip(all_sequence_ids, logits_num_per_batch)])
        sampling_metadata.all_sequence_ids = req_id_new
        sampling_metadata.parent_sequence_ids = req_id_new
        logger.info(f"[MTP] Updated sequence_ids from {len(all_sequence_ids)} to {len(req_id_new)} entries (expanded by speculative length)")
        logger.info("[MTP] sample_preprocess completed")

        return logits

    def plugin_verify(self, sampling_output, cache_ids, result):
        logger.info(f"[MTP] plugin_verify started with cache_ids={cache_ids}")
        sampling_output.repeating_indices = np.arange(len(cache_ids))
        logger.info(f"[MTP] Set repeating_indices for {len(cache_ids)} sequences")
        
        if self.input_metadata.is_prefill:
            logger.info("[MTP] Prefill mode, skipping verification and returning directly")
            return
        
        draft_token = result[2].cpu()
        logger.info(f"[MTP] Extracted draft tokens from result[2], shape: {draft_token.shape}")
        
        next_tokens_uncheck = sampling_output.token_ids
        logger.info(f"[MTP] Unverified next tokens from sampling_output, total tokens: {len(next_tokens_uncheck)}")
        
        input_metadata = self.input_metadata
        next_tokens_indices = []
        out_seq_len = 1 if input_metadata.is_prefill else (self.num_speculative_tokens + 1)
        logger.info(f"[MTP] Output sequence length per batch: {out_seq_len} (speculative_tokens={self.num_speculative_tokens})")
        
        start_pos = 0
        draft_token_num_per_batch = self.num_speculative_tokens
        logger.info(f"[MTP] Starting verification for batch_size={input_metadata.batch_size}")
        
        for batch in range(input_metadata.batch_size):
            end = start_pos + out_seq_len
            next_guess_by_batch = next_tokens_uncheck[start_pos:end]

            verify_guess_tokens = \
                draft_token[batch * draft_token_num_per_batch: (batch + 1) * draft_token_num_per_batch].view(-1)
            
            logger.info(f"[MTP] Batch {batch}: verifying {len(verify_guess_tokens)} draft tokens against {len(next_guess_by_batch)} predicted tokens")

            indices = self.decoding_policy.verify_greedy_one_batch(verify_guess_tokens, next_guess_by_batch)
            accepted_tokens = indices + 1  # +1 because we also accept the final verified token
            next_tokens_indices.append(list(range(start_pos, start_pos + accepted_tokens)))
            logger.info(f"[MTP] Batch {batch}: accepted {accepted_tokens} tokens (matched {indices} draft tokens)")
            
            start_pos += out_seq_len
        
        logger.info(f"[MTP] Verification completed for all batches. Accepted token indices: {next_tokens_indices}")
        
        output_token_len = self.input_manager.cache.output_len_count[cache_ids]
        output_space_left1 = self.input_metadata.batch_max_output_lens - output_token_len
        output_space_left2 = self.input_manager.cache.cache_config.max_seq_len - \
            self.input_manager.cache.cached_seq_lens[cache_ids]
        output_space_left = np.minimum(output_space_left1, output_space_left2)
        logger.info(f"[MTP] Output space remaining: max_output_lens constraint={output_space_left1}, "
                    f"max_seq_len constraint={output_space_left2}, effective={output_space_left}")
        
        self.decoding_policy.stop_criteria(sampling_output, output_space_left, next_tokens_indices)
        logger.info("[MTP] Applied stop criteria based on output space constraints")
        
        self.reshape_speculative_outputs(sampling_output, next_tokens_indices)
        logger.info("[MTP] Reshaped speculative outputs and plugin_verify completed")

    def plugin_cache_update(self, cache_ids, sampling_output, la_cache_input, is_prefill=False):
        logger.info(f"[MTP] plugin_cache_update started with cache_ids={cache_ids}, is_prefill={is_prefill}")
        result, _ = la_cache_input
        if isinstance(result, tuple):
            hidden_states = result[1]
            logger.info(f"[MTP] Extracted hidden_states from result tuple, shape: {hidden_states.shape}")
        
        if is_prefill:
            token_alias_len = [1] * len(cache_ids)
            logger.info(f"[MTP] Prefill mode: token_alias_len set to 1 for all {len(cache_ids)} sequences")
        else:
            token_alias_len = [self.num_speculative_tokens + 1] * len(cache_ids)
            logger.info(f"[MTP] Decode mode: token_alias_len set to {self.num_speculative_tokens + 1} for all {len(cache_ids)} sequences")
        
        self.decoding_policy.mtp_cache.cache_update(cache_ids, hidden_states,
                                                    sampling_output, is_prefill, token_alias_len)
        logger.info("[MTP] plugin_cache_update completed, cache updated successfully")

    def plugin_cache_clear(self, cache_ids, finish_reason):
        self.input_manager.sampling_cache.clear()
        pass
