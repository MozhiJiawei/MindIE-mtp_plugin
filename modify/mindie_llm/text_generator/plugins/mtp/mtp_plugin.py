# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import numpy as np
from collections import deque
import torch
import torch.nn.functional as F

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

        # 投机接受率统计相关变量
        self.window_size = 500
        self.print_interval = 100
        self.iteration_count = 0
        self.first_token_acceptance_window = deque(maxlen=self.window_size)
        self.second_token_acceptance_window = deque(maxlen=self.window_size)
    
    def model_inputs_update(self, model_inputs, input_metadata, sampling_metadata, cache_ids, input_len_mask, **kwargs):
        hit_mask = kwargs.get('hit_mask')
        model_inputs, q_len, attention_mask = (
            self.decoding_policy.handle_input(model_inputs, input_metadata, cache_ids, hit_mask=hit_mask))
        input_len_mask = (q_len, attention_mask)

        self.decoding_policy.handle_sampling(sampling_metadata)

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
        speculative_length = self.num_speculative_tokens + 1
        mtp_model_inputs = model_kwargs.get('sub_model_inputs')
        hidden_states = model_kwargs.get('hidden_states')
        lm_head_local_dp = model_kwargs.get('lm_head_local_dp', None)
        input_lengths_sp = model_kwargs.get('input_lengths_sp', None)
        hit_mask = filling_masks.get('hit_mask')
        if hit_mask is not None:
            model_output_hidden_states = model_output_wrapper.model_output.hidden_states
            sampling_output = model_output_wrapper.sampling_output

            # Move the hit_token_ids of mtp model to device.
            hit_indices = filling_masks.get('hit_indices')
            hit_token_ids = sampling_output.token_ids[hit_indices]
            hit_token_ids_cols = hit_token_ids.shape[1]
            if hit_token_ids_cols < speculative_length:
                padding_width = ((0, 0), (0, speculative_length - hit_token_ids_cols))
                hit_token_ids = np.pad(hit_token_ids, padding_width, 'constant', constant_values=0)
            elif hit_token_ids_cols > speculative_length:
                logger.warning('Found the number of output tokens exceeds the speculative length, '
                               'which will be truncated forcibly.')
                hit_token_ids = hit_token_ids[:, :speculative_length]
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
            hit_token_ids_tensor = hit_token_ids.reshape(-1)
            mtp_model_inputs.input_ids[hit_mask_per_token] = hit_token_ids_tensor
            mtp_model_inputs.prefill_head_indices[all_hit_mask_tensor] -= head_indices_subtrahend
            hit_hidden_states = model_output_hidden_states[hit_indices_per_token]
            hidden_states[hit_mask_per_token] = hit_hidden_states
            if lm_head_local_dp is not None and not (len(lm_head_local_dp) == 1 and lm_head_local_dp[0] == 0):
                hit_dp_speculative_length = filling_masks.get('hit_dp_speculative_length')
                lm_head_local_dp[hit_mask] -= \
                    self.generator_backend.to_tensor(hit_dp_speculative_length - hit_num_tokens)

            model_inputs.position_ids[hit_mask_per_token] += hit_num_tokens_per_token
            model_inputs.context_length[hit_mask] += hit_num_tokens
            model_inputs.input_lengths[hit_mask_tensor] += hit_num_tokens_tensor
            model_inputs.max_seq_len = max(model_inputs.context_length)
            if self.cache.scp_size == 1:
                offset_start_indices = model_inputs.input_lengths[hit_mask_tensor] - speculative_length
                hit_increments = filling_masks.get('hit_increments')
                block_offsets = backend.repeat_interleave(offset_start_indices, speculative_length) + hit_increments
                candidate_slots = filling_masks.get('candidate_slots')
                hit_block_indices = filling_masks.get('hit_block_indices')
                model_inputs.slots[hit_mask_per_token] = candidate_slots[hit_block_indices, block_offsets]
            else:
                model_inputs.cached_context_length[hit_mask] += hit_num_tokens
                indices = np.where(hit_mask)[0]
                for _, idx in enumerate(indices):
                    sp_tokens, tmp_slots = self.decoding_policy.sp_token_and_slot_calc_by_context_length(model_inputs.cached_context_length[idx], self.cache.cached_seq_block_rank_id[cache_ids[idx]], input_metadata.block_tables[idx], speculative_length)
                    model_inputs.slots[idx * speculative_length:(idx + 1) * speculative_length] = self.generator_backend.to_tensor_async(tmp_slots)
                    model_inputs.sp_tokens[idx] = sp_tokens
                    input_lengths_sp[idx] = sp_tokens[self.cp_rank * self.sp_size:(self.cp_rank + 1) * self.sp_size][self.mapping.attn_inner_sp.rank]

            hit_mask_per_token[hit_mask_mod] = False
            model_inputs.input_ids[hit_mask_per_token] = hit_token_ids[hit_arange_tensor, hit_num_tokens_tensor - 1]

    def sample_preprocess(self, logits, result, sampling_metadata, input_metadata):
        self.sampling_param = sampling_metadata
        self.decoding_policy.sampling_param = self.sampling_param
        self.input_metadata = input_metadata
        if isinstance(result, tuple):
            logits = result[0]
        else:
            logits = result
        if sampling_metadata is None or sampling_metadata.is_prefill:
            return logits
        draft_tokens = result[2].cpu()
        all_sequence_ids = sampling_metadata.all_sequence_ids
        batch_size = len(all_sequence_ids)

        logits_num_per_batch = [self.num_speculative_tokens + 1] * batch_size
        input_ids_pad = self.decoding_policy.all_token_ids_padding(sampling_metadata, logits_num_per_batch,
                                                                   batch_size, draft_tokens)
        sampling_metadata.all_token_ids = input_ids_pad

        req_id_new = np.concatenate([[req_id] * n for req_id, n in zip(all_sequence_ids, logits_num_per_batch)])
        sampling_metadata.all_sequence_ids = req_id_new
        sampling_metadata.parent_sequence_ids = req_id_new

        return logits

    def rejection_sampling_verify(self, draft_logits, target_logits, draft_tokens, temperature=1.0):
        """
        实现拒绝采样算法（Algorithm 4 GenSpecSample）
        
        Args:
            draft_logits: Draft model的logits, shape [num_tokens, vocab_size]
            target_logits: Target model的logits, shape [num_tokens+1, vocab_size]
            draft_tokens: Draft model生成的tokens, shape [num_tokens]
            temperature: 采样温度
            
        Returns:
            accepted_tokens: 接受的token列表
            num_accepted: 接受的token数量
        """
        device = draft_logits.device
        vocab_size = draft_logits.size(-1)
        
        print(f"[Rejection Sampling] Input shapes - draft_logits: {draft_logits.shape}, "
                   f"target_logits: {target_logits.shape}, draft_tokens: {draft_tokens.shape}, "
                   f"vocab_size: {vocab_size}, temperature: {temperature}")
        print(f"[Rejection Sampling] Draft tokens: {draft_tokens.tolist()}")

        # 应用温度并计算概率分布
        draft_probs = F.softmax(draft_logits / temperature, dim=-1)
        target_probs = F.softmax(target_logits / temperature, dim=-1)
        
        print(f"[Rejection Sampling] Probability distribution computed - "
                   f"draft_probs range: [{draft_probs.min().item():.6f}, {draft_probs.max().item():.6f}], "
                   f"target_probs range: [{target_probs.min().item():.6f}, {target_probs.max().item():.6f}]")

        accepted_tokens = []
        num_tokens = draft_tokens.size(0)
        rejected = False  # 标记是否发生了拒绝

        print(f"[Rejection Sampling] Starting verification loop for {num_tokens} tokens")

        for j in range(num_tokens):
            # 获取draft token的概率
            draft_token = draft_tokens[j].item()
            q_x = draft_probs[j, draft_token].item()
            p_x = target_probs[j, draft_token].item()
            
            # 计算接受概率: min{1, p(x)/q(x)}
            accept_prob = min(1.0, p_x / (q_x + 1e-10))  # 添加小值避免除零
            
            print(f"[Rejection Sampling] Token {j}: draft_token={draft_token}, "
                       f"q_x={q_x:.6f}, p_x={p_x:.6f}, accept_prob={accept_prob:.6f}")

            # 伯努利采样决定是否接受
            random_value = np.random.random()
            if random_value < accept_prob:
                # 接受draft token
                accepted_tokens.append(draft_token)
                print(f"[Rejection Sampling] Token {j} ACCEPTED (random={random_value:.6f} < {accept_prob:.6f})")
            else:
                # 拒绝，从残差分布采样
                # residual(x) = norm(max{0, p(x) - q(x)})
                residual = torch.clamp(target_probs[j] - draft_probs[j], min=0.0)
                residual_sum = residual.sum()
                
                print(f"[Rejection Sampling] Token {j} REJECTED (random={random_value:.6f} >= {accept_prob:.6f}), "
                           f"residual_sum={residual_sum.item():.6f}")

                if residual_sum > 1e-10:
                    # 归一化残差分布
                    residual_probs = residual / residual_sum
                    # 从残差分布采样新token
                    new_token = torch.multinomial(residual_probs, num_samples=1).item()
                    print(f"[Rejection Sampling] Sampled new token from residual: {new_token}")
                else:
                    # 如果残差分布为空，直接从target分布采样
                    new_token = torch.multinomial(target_probs[j], num_samples=1).item()
                    print(f"[Rejection Sampling] Sampled new token from target (residual empty): {new_token}")

                accepted_tokens.append(new_token)
                rejected = True  # 标记发生了拒绝
                # 拒绝后停止
                print(f"[Rejection Sampling] Stopping after rejection at position {j}")
                break
        
        # 只有当所有draft tokens都被接受（没有发生拒绝）时，才从最后的target分布采样bonus token
        if not rejected:
            # 使用最后一个target logits采样bonus token
            last_token = torch.multinomial(target_probs[-1], num_samples=1).item()
            accepted_tokens.append(last_token)
            # num_accepted 表示实际接受的draft tokens数量，不包括bonus token
            num_accepted = num_tokens
            print(f"[Rejection Sampling] All {num_tokens} draft tokens accepted, bonus token sampled: {last_token}")
            print(f"[Rejection Sampling] Total tokens to use: {len(accepted_tokens)} (draft + bonus)")
        else:
            # 发生了拒绝，最后一个token是从残差分布采样的新token
            # num_accepted 表示被接受的draft tokens数量（拒绝位置之前的tokens）
            num_accepted = len(accepted_tokens) - 1
            print(f"[Rejection Sampling] Rejection occurred, {num_accepted} draft tokens accepted before rejection")
            print(f"[Rejection Sampling] Total tokens to use: {len(accepted_tokens)} (accepted + resampled)")

        print(f"[Rejection Sampling] Final result - accepted_tokens: {accepted_tokens}, "
                   f"num_accepted (draft only): {num_accepted}, total_tokens: {len(accepted_tokens)}")

        return accepted_tokens, num_accepted

    def plugin_verify(self, sampling_output, cache_ids, result):
        sampling_output.repeating_indices = np.arange(len(cache_ids))
        if self.input_metadata.is_prefill:
            print("[Plugin Verify] Skipping verification for prefill phase")
            return
        
        print(f"[Plugin Verify] Starting verification - cache_ids: {cache_ids}, "
                   f"result length: {len(result)}, result types: {[type(r) for r in result]}")

        # 提取draft tokens和logits
        # result格式取决于是否使用 forward_mtp_decoding_v2
        if len(result) >= 4 and isinstance(result[3], torch.Tensor):
            # forward_mtp_decoding_v2 返回 (logits, hidden_states, draft_tokens, draft_logits)
            target_logits = result[0]  # Target model logits
            draft_tokens = result[2]   # Draft tokens
            draft_logits = result[3]   # Draft model logits
            use_rejection_sampling = True
            print(f"[Plugin Verify] Using rejection sampling - target_logits shape: {target_logits.shape}, "
                       f"draft_tokens shape: {draft_tokens.shape}, draft_logits shape: {draft_logits.shape}")
        else:
            # 兼容旧格式
            draft_tokens = result[2].cpu() if torch.is_tensor(result[2]) else result[2]
            use_rejection_sampling = False
            print(f"[Plugin Verify] Using greedy verification (fallback) - draft_tokens: {draft_tokens}")

        draft_token = result[2].cpu()
        next_tokens_uncheck = sampling_output.token_ids
        input_metadata = self.input_metadata
        next_tokens_indices = []
        out_seq_len = 1 if input_metadata.is_prefill else (self.num_speculative_tokens + 1)
        start_pos = 0
        draft_token_num_per_batch = self.num_speculative_tokens

        print(f"[Plugin Verify] Processing batches - batch_size: {input_metadata.batch_size}, "
                   f"out_seq_len: {out_seq_len}, draft_token_num_per_batch: {draft_token_num_per_batch}")
        print(f"[Plugin Verify] Initial next_tokens_uncheck: {next_tokens_uncheck}")

        for batch in range(input_metadata.batch_size):
            end = start_pos + out_seq_len
            print(f"[Plugin Verify] Batch {batch} - start_pos: {start_pos}, end: {end}")

            if use_rejection_sampling:
                # 使用拒绝采样算法
                batch_draft_tokens = draft_tokens[
                    batch * draft_token_num_per_batch: (batch + 1) * draft_token_num_per_batch]
                batch_draft_logits = draft_logits[
                    batch * draft_token_num_per_batch: (batch + 1) * draft_token_num_per_batch]
                batch_target_logits = target_logits[start_pos:end]
                
                print(f"[Plugin Verify] Batch {batch} - batch_draft_tokens shape: {batch_draft_tokens.shape}, "
                           f"batch_draft_logits shape: {batch_draft_logits.shape}, "
                           f"batch_target_logits shape: {batch_target_logits.shape}")

                # 执行拒绝采样
                accepted_tokens, num_accepted = self.rejection_sampling_verify(
                    batch_draft_logits, 
                    batch_target_logits,
                    batch_draft_tokens
                )
                
                # num_accepted 是被接受的draft tokens数量
                # len(accepted_tokens) 是总共要使用的tokens数量（包括bonus token或重采样token）
                total_tokens = len(accepted_tokens)

                print(f"[Plugin Verify] Batch {batch} - accepted_tokens: {accepted_tokens}, "
                           f"num_accepted (draft): {num_accepted}, total_tokens: {total_tokens}")

                # 更新sampling_output中的tokens
                original_tokens = [next_tokens_uncheck[start_pos + i] for i in range(min(total_tokens, len(next_tokens_uncheck) - start_pos))]
                for i, token in enumerate(accepted_tokens):
                    if start_pos + i < len(next_tokens_uncheck):
                        sampling_output.token_ids[start_pos + i] = token
                print(f"[Plugin Verify] Batch {batch} - updated tokens from {original_tokens} to {accepted_tokens}")

                # indices 用于统计接受率，使用 num_accepted（只计算draft tokens）
                indices = num_accepted
                # next_tokens_indices 应该包含所有实际使用的tokens（包括bonus token）
                next_tokens_indices.append(list(range(start_pos, start_pos + total_tokens)))
            else:
                # 原始的贪婪验证方法（作为fallback）
                next_guess_by_batch = next_tokens_uncheck[start_pos:end]
                verify_guess_tokens = draft_tokens[
                    batch * draft_token_num_per_batch: (batch + 1) * draft_token_num_per_batch]
                if torch.is_tensor(verify_guess_tokens):
                    verify_guess_tokens = verify_guess_tokens.view(-1)

                print(f"[Plugin Verify] Batch {batch} - greedy verification: "
                           f"next_guess_by_batch: {next_guess_by_batch}, "
                           f"verify_guess_tokens: {verify_guess_tokens}")

                indices = self.decoding_policy.verify_greedy_one_batch(verify_guess_tokens, next_guess_by_batch)
                next_tokens_indices.append(list(range(start_pos, start_pos + indices + 1)))
                print(f"[Plugin Verify] Batch {batch} - greedy indices: {indices}")

            # 统计第一个和第二个Token的接受情况
            print(f"[Plugin Verify] Batch {batch} - updating acceptance statistics with indices: {indices}")
            self._update_acceptance_statistics(indices)
            
            start_pos += out_seq_len
        
        print(f"[Plugin Verify] All batches processed - next_tokens_indices: {next_tokens_indices}")

        # 更新迭代计数并打印接受率
        self._update_and_print_acceptance_rate()
        
        output_token_len = self.input_manager.cache.output_len_count[cache_ids]
        output_space_left1 = self.input_metadata.batch_max_output_lens - output_token_len
        output_space_left2 = self.input_manager.cache.cache_config.max_seq_len - \
            self.input_manager.cache.cached_seq_lens[cache_ids]
        output_space_left = np.minimum(output_space_left1, output_space_left2)

        print(f"[Plugin Verify] Output space calculation - output_token_len: {output_token_len}, "
                   f"batch_max_output_lens: {self.input_metadata.batch_max_output_lens}, "
                   f"output_space_left1: {output_space_left1}, output_space_left2: {output_space_left2}, "
                   f"output_space_left: {output_space_left}")

        print(f"[Plugin Verify] Before stop_criteria - next_tokens_indices: {next_tokens_indices}")
        self.decoding_policy.stop_criteria(sampling_output, output_space_left, next_tokens_indices)
        print(f"[Plugin Verify] After stop_criteria - next_tokens_indices: {next_tokens_indices}")
        self.reshape_speculative_outputs(sampling_output, next_tokens_indices)

        print(f"[Plugin Verify] Final sampling_output.token_ids: {sampling_output.token_ids}")
        print(f"[Plugin Verify] Final sampling_output.num_new_tokens: {sampling_output.num_new_tokens}")

    def plugin_cache_update(self, cache_ids, sampling_output, la_cache_input, is_prefill=False):
        result, _ = la_cache_input
        if isinstance(result, tuple):
            hidden_states = result[1]
        if is_prefill:
            token_alias_len = [1] * len(cache_ids)
        else:
            token_alias_len = [self.num_speculative_tokens + 1] * len(cache_ids)
        
        self.decoding_policy.mtp_cache.cache_update(cache_ids, hidden_states,
                                                    sampling_output, is_prefill, token_alias_len)

    def plugin_cache_clear(self, cache_ids, finish_reason):
        self.input_manager.sampling_cache.clear()
        pass
    
    def _update_acceptance_statistics(self, accepted_indices):
        """
        更新第一个和第二个Token的接受情况统计
        
        Args:
            accepted_indices: 验证后接受的Token数量索引
        """
        # 第一个投机Token是否被接受 (accepted_indices >= 1表示至少接受了第一个Token)
        first_token_accepted = 1 if accepted_indices >= 1 else 0
        self.first_token_acceptance_window.append(first_token_accepted)
        
        # 第二个投机Token是否被接受 (accepted_indices >= 2表示至少接受了前两个Token)
        second_token_accepted = 1 if accepted_indices >= 2 else 0
        self.second_token_acceptance_window.append(second_token_accepted)
    
    def _update_and_print_acceptance_rate(self):
        """
        更新迭代计数，并在达到打印间隔时打印接受率统计信息
        """
        self.iteration_count += 1
        
        # 每100轮迭代打印一次
        if self.iteration_count % self.print_interval == 0:
            if len(self.first_token_acceptance_window) > 0:
                first_token_rate = sum(self.first_token_acceptance_window) / len(self.first_token_acceptance_window)
                second_token_rate = sum(self.second_token_acceptance_window) / len(self.second_token_acceptance_window)
                
                logger.info(f"[MTP接受率统计] 迭代次数: {self.iteration_count}, "
                           f"滑窗大小: {len(self.first_token_acceptance_window)}, "
                           f"第一个Token接受率: {first_token_rate:.4f} ({first_token_rate*100:.2f}%), "
                           f"第二个Token接受率: {second_token_rate:.4f} ({second_token_rate*100:.2f}%)")
