# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 许可证：Apache-2.0
# 版权信息：vLLM 项目贡献者所有

"""
简化的拒绝采样实现
Simplified Rejection Sampling Implementation

该实现仅使用 numpy，实现了推测解码（Speculative Decoding）中的拒绝采样算法。
This implementation uses only numpy and implements the rejection sampling
algorithm used in Speculative Decoding.

参考论文：https://arxiv.org/abs/2211.17192
Reference paper: https://arxiv.org/abs/2211.17192
"""

import numpy as np


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """计算 softmax 函数
    Compute softmax function
    
    Args:
        logits: 输入 logits
        axis: 计算 softmax 的维度
    
    Returns:
        概率分布
    """
    # 减去最大值以提高数值稳定性
    # Subtract max for numerical stability
    logits_max = np.max(logits, axis=axis, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    return exp_logits / np.sum(exp_logits, axis=axis, keepdims=True)


def sample_from_probs(probs: np.ndarray) -> int:
    """从概率分布中采样一个 token
    Sample a token from probability distribution
    
    Args:
        probs: [vocab_size] 概率分布
    
    Returns:
        采样的 token ID
    """
    return np.random.choice(len(probs), p=probs)


def sample_recovered_token(draft_prob: np.ndarray, target_prob: np.ndarray, 
                           draft_token_id: int) -> int:
    """采样恢复的 token
    Sample a recovered token
    
    当草稿 token 被拒绝时，从调整后的概率分布中采样。
    调整后的分布：max(target_prob - draft_prob, 0)
    When a draft token is rejected, sample from the adjusted distribution.
    Adjusted distribution: max(target_prob - draft_prob, 0)
    
    Args:
        draft_prob: [vocab_size] 草稿概率分布
        target_prob: [vocab_size] 目标概率分布
        draft_token_id: 被拒绝的草稿 token ID
    
    Returns:
        恢复的 token ID
    """
    # 计算调整后的概率：max(target_prob - draft_prob, 0)
    # Compute adjusted probability: max(target_prob - draft_prob, 0)
    adjusted_prob = np.maximum(target_prob - draft_prob, 0.0)
    
    # 归一化概率分布
    # Normalize the probability distribution
    prob_sum = np.sum(adjusted_prob)
    if prob_sum > 0:
        adjusted_prob = adjusted_prob / prob_sum
    else:
        # 如果所有概率都是 0，则使用均匀分布
        # If all probabilities are 0, use uniform distribution
        adjusted_prob = np.ones_like(adjusted_prob) / len(adjusted_prob)
    
    # 从调整后的分布中采样
    # Sample from the adjusted distribution
    return sample_from_probs(adjusted_prob)


def rejection_sample(draft_logits: np.ndarray, 
                    target_logits: np.ndarray) -> list[int]:
    """拒绝采样函数
    Rejection sampling function
    
    该函数实现了推测解码中的拒绝采样算法。算法流程：
    1. 首先从草稿模型的概率分布中采样 draft tokens
    2. 对每个 draft token，计算接受概率 = min(1, target_prob / draft_prob)
    3. 使用均匀随机数决定是否接受该 token
    4. 如果接受，将该 token 加入输出；如果拒绝，从调整后的分布中采样 recovered token
    5. 一旦有 token 被拒绝，停止处理后续的 draft tokens
    
    This function implements the rejection sampling algorithm in speculative decoding:
    1. First sample draft tokens from the draft model's probability distribution
    2. For each draft token, compute acceptance probability = min(1, target_prob / draft_prob)
    3. Use uniform random number to decide whether to accept the token
    4. If accepted, add to output; if rejected, sample a recovered token from adjusted distribution
    5. Once a token is rejected, stop processing subsequent draft tokens
    
    Args:
        draft_logits: np.ndarray
            草稿模型的 logits，形状为 [num_tokens, vocab_size]
            Draft model's logits with shape [num_tokens, vocab_size]
        target_logits: np.ndarray
            目标模型的 logits，形状为 [num_tokens, vocab_size]
            Target model's logits with shape [num_tokens, vocab_size]
    
    Returns:
        list[int]: 采样得到的 token ID 列表
                   List of sampled token IDs
    
    示例 Example:
        >>> draft_logits = np.random.randn(5, 1000)  # 5个draft tokens，词汇表大小1000
        >>> target_logits = np.random.randn(5, 1000)
        >>> output_tokens = rejection_sample(draft_logits, target_logits)
        >>> print(f"输出了 {len(output_tokens)} 个 tokens")
    """
    assert draft_logits.ndim == 2, "draft_logits 必须是 2 维张量 [num_tokens, vocab_size]"
    assert target_logits.ndim == 2, "target_logits 必须是 2 维张量 [num_tokens, vocab_size]"
    assert draft_logits.shape == target_logits.shape, "draft_logits 和 target_logits 形状必须相同"
    
    num_tokens, vocab_size = draft_logits.shape
    
    # 步骤 1: 将 logits 转换为概率分布
    # Step 1: Convert logits to probability distributions
    draft_probs = softmax(draft_logits, axis=-1)  # [num_tokens, vocab_size]
    target_probs = softmax(target_logits, axis=-1)  # [num_tokens, vocab_size]
    
    # 步骤 2: 从草稿概率分布中采样 draft tokens
    # Step 2: Sample draft tokens from draft probability distribution
    draft_token_ids = np.array([
        sample_from_probs(draft_probs[i]) 
        for i in range(num_tokens)
    ])
    
    # 步骤 3: 执行拒绝采样
    # Step 3: Perform rejection sampling
    output_tokens = []
    
    for i in range(num_tokens):
        draft_token_id = draft_token_ids[i]
        
        # 获取该 draft token 在两个分布中的概率
        # Get the probability of this draft token in both distributions
        draft_prob = draft_probs[i, draft_token_id]
        target_prob = target_probs[i, draft_token_id]
        
        # 计算接受概率：min(1, target_prob / draft_prob)
        # Compute acceptance probability: min(1, target_prob / draft_prob)
        if draft_prob > 0:
            acceptance_prob = min(1.0, target_prob / draft_prob)
        else:
            # 如果 draft_prob 为 0，拒绝该 token
            # If draft_prob is 0, reject the token
            acceptance_prob = 0.0
        
        # 生成均匀随机数进行接受/拒绝判断
        # Generate uniform random number for accept/reject decision
        uniform_rand = np.random.uniform(0, 1)
        
        if uniform_rand < acceptance_prob:
            # 接受该 draft token
            # Accept the draft token
            output_tokens.append(draft_token_id)
        else:
            # 拒绝该 draft token，从调整后的分布中采样 recovered token
            # Reject the draft token, sample recovered token from adjusted distribution
            recovered_token_id = sample_recovered_token(
                draft_probs[i], 
                target_probs[i], 
                draft_token_id
            )
            output_tokens.append(recovered_token_id)
            
            # 一旦拒绝，停止处理后续 tokens（这是拒绝采样的关键）
            # Once rejected, stop processing subsequent tokens (key to rejection sampling)
            break
    
    return output_tokens


def rejection_sample_with_bonus(draft_logits: np.ndarray, 
                                target_logits: np.ndarray,
                                bonus_logits: np.ndarray) -> list[int]:
    """带有奖励 token 的拒绝采样
    Rejection sampling with bonus token
    
    如果所有 draft tokens 都被接受，则从 bonus_logits 中采样一个额外的 token。
    If all draft tokens are accepted, sample an additional token from bonus_logits.
    
    Args:
        draft_logits: [num_tokens, vocab_size] 草稿模型的 logits
        target_logits: [num_tokens, vocab_size] 目标模型的 logits
        bonus_logits: [vocab_size] 奖励 token 的 logits
    
    Returns:
        list[int]: 采样得到的 token ID 列表
    """
    # 先执行标准的拒绝采样
    # First perform standard rejection sampling
    output_tokens = rejection_sample(draft_logits, target_logits)
    
    # 如果所有 draft tokens 都被接受，添加 bonus token
    # If all draft tokens are accepted, add bonus token
    num_draft_tokens = draft_logits.shape[0]
    if len(output_tokens) == num_draft_tokens:
        bonus_probs = softmax(bonus_logits)
        bonus_token = sample_from_probs(bonus_probs)
        output_tokens.append(bonus_token)
    
    return output_tokens


# 使用示例 Usage Example
if __name__ == "__main__":
    print("=" * 60)
    print("拒绝采样示例 Rejection Sampling Example")
    print("=" * 60)
    
    # 设置随机种子以保证可重现性
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # 创建示例数据
    # Create example data
    num_draft_tokens = 5
    vocab_size = 100
    
    print(f"\n配置 Configuration:")
    print(f"  - 草稿 token 数量 (Number of draft tokens): {num_draft_tokens}")
    print(f"  - 词汇表大小 (Vocabulary size): {vocab_size}")
    
    # 生成随机的 logits
    # Generate random logits
    draft_logits = np.random.randn(num_draft_tokens, vocab_size)
    target_logits = np.random.randn(num_draft_tokens, vocab_size)
    
    print(f"\n输入形状 Input shapes:")
    print(f"  - draft_logits: {draft_logits.shape}")
    print(f"  - target_logits: {target_logits.shape}")
    
    # 执行拒绝采样
    # Perform rejection sampling
    print(f"\n执行拒绝采样 Performing rejection sampling...")
    output_tokens = rejection_sample(draft_logits, target_logits)
    
    print(f"\n结果 Results:")
    print(f"  - 输出 token 数量 (Number of output tokens): {len(output_tokens)}")
    print(f"  - 输出 tokens (Output tokens): {output_tokens}")
    
    # 多次运行以展示随机性
    # Run multiple times to show randomness
    print(f"\n多次运行结果 Multiple runs:")
    for run in range(5):
        np.random.seed(100 + run)
        draft_logits = np.random.randn(num_draft_tokens, vocab_size)
        target_logits = np.random.randn(num_draft_tokens, vocab_size)
        tokens = rejection_sample(draft_logits, target_logits)
        print(f"  Run {run + 1}: 接受了 {len(tokens)} 个 tokens")
    
    # 测试带奖励 token 的版本
    # Test version with bonus token
    print(f"\n测试带奖励 token 的拒绝采样 Testing rejection sampling with bonus token:")
    np.random.seed(42)
    draft_logits = np.random.randn(3, vocab_size)
    target_logits = np.random.randn(3, vocab_size)
    bonus_logits = np.random.randn(vocab_size)
    
    tokens_with_bonus = rejection_sample_with_bonus(
        draft_logits, target_logits, bonus_logits
    )
    print(f"  - 输出 tokens: {tokens_with_bonus}")
    print(f"  - Token 数量: {len(tokens_with_bonus)}")
    
    print(f"\n" + "=" * 60)
    print("完成！Done!")
    print("=" * 60)

