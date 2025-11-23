import torch
import numpy as np
import os
torch.manual_seed(1234)

# --- Settings ---
N = 4               # number of tokens
num_experts = 128
top_k = 8
threshold = 0.8
dtype = torch.float16
pad_bytes = 32
elements_per_pad = pad_bytes // 2  # 16 elements per token (float16 = 2 bytes)

# --- Utility for file naming ---
def fname(base: str) -> str:
    return f"{base}_N{N}.bin"

# --- 1) Generate routing scores for all tokens ---
full_routing_scores = torch.randn(N, 128) * 2.0  # [N, 128]
full_routing_scores = torch.softmax(full_routing_scores, dim=-1)

# --- 2) Select top-k experts for each token ---
topk_vals, topk_idx = torch.topk(full_routing_scores, k=top_k, dim=-1)
routing_weights_in = topk_vals.clone()       # [N, top_k]
selected_experts_in = topk_idx.clone()       # [N, top_k]

print("Input (unnormalized) top-k routing weights:\n", routing_weights_in)
print("\nInput selected expert indices:\n", selected_experts_in)

# --- 3) Normalize among top-k experts ---
norm_routing_weights = routing_weights_in / routing_weights_in.sum(dim=-1, keepdim=True)
print("\nNormalized routing weights (sum per token ~1):")
print(norm_routing_weights)
print("Sum per token:", norm_routing_weights.sum(dim=-1))

# --- 4) Dynamic Expert Selection (DES) ---
cumsum = norm_routing_weights.cumsum(dim=-1)
keep_num = (cumsum >= threshold).float().argmax(dim=-1, keepdim=True) + 1  # [N,1]
max_keep = keep_num.max().item()

routing_weights_out = torch.zeros(N, max_keep, dtype=dtype)
selected_experts_out = torch.full((N, max_keep), float(777), dtype=dtype)

for i in range(N):
    k = keep_num[i].item()
    routing_weights_out[i, :k] = norm_routing_weights[i, :k]
    selected_experts_out[i, :k] = selected_experts_in[i, :k]

print("\n--- After DES ---")
print("Keep num per token:", keep_num.squeeze(-1))
print("Output routing weights (unpadded):\n", routing_weights_out)
print("Output selected experts (unpadded):\n", selected_experts_out)

# --- 5) Pad outputs to exactly 32 bytes (16 elements) per token ---
def pad_tensor(tensor, pad_value, target_len):
    pad_len = max(0, target_len - tensor.shape[1])
    if pad_len > 0:
        pad_block = torch.full((tensor.shape[0], pad_len), pad_value, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, pad_block], dim=-1)
    else:
        return tensor[:, :target_len]

routing_weights_out_padded = pad_tensor(routing_weights_out, 0.0, elements_per_pad)
selected_experts_out_padded = pad_tensor(selected_experts_out, float(777), elements_per_pad)

print("\n--- After Padding ---")
print("Padded routing weights shape:", routing_weights_out_padded.shape)
print("Padded selected experts shape:", selected_experts_out_padded.shape)

# --- 6) Save required tensors as .bin (float16) ---
routing_weights_in.cpu().numpy().astype('float16').tofile(fname("input_routing_weights"))
norm_routing_weights.cpu().numpy().astype('float16').tofile(fname("normalized_routing_weights"))
routing_weights_out_padded.cpu().numpy().astype('float16').tofile(fname("output_routing_weights"))
selected_experts_out_padded.cpu().numpy().astype('float16').tofile(fname("output_selected_experts"))

print("\nSaved .bin files:")
print(f" - {fname('input_routing_weights')}      [N, top_k]")
print(f" - {fname('normalized_routing_weights')} [N, top_k]")
print(f" - {fname('output_routing_weights')}     [N, 16] (32 bytes/token)")
print(f" - {fname('output_selected_experts')}    [N, 16] (32 bytes/token)")

# --- 7) Reload and verify ---
def load_bin(filename, shape):
    return np.fromfile(filename, dtype=np.float16).reshape(shape)

inp = load_bin(fname("input_routing_weights"), (N, top_k))
norm = load_bin(fname("normalized_routing_weights"), (N, top_k))
out_w = load_bin(fname("output_routing_weights"), (N, elements_per_pad))
out_idx = load_bin(fname("output_selected_experts"), (N, elements_per_pad))

print("\n--- Reloaded from .bin ---")
print("Input routing weights:\n", inp)
print("Normalized routing weights:\n", norm)
print("Output routing weights (padded):\n", out_w)
print("Output selected experts (padded):\n", out_idx)

# --- 8) Optional: file size check ---
for f in [fname("output_routing_weights"), fname("output_selected_experts")]:
    size = os.path.getsize(f)
    print(f"File {f}: {size} bytes (expected {N*pad_bytes})")

# --- Sanity checks ---
print("\nSanity checks:")
print("Input == saved:", np.allclose(inp, routing_weights_in.cpu().numpy(), atol=1e-3))
print("Norm == saved:", np.allclose(norm, norm_routing_weights.cpu().numpy(), atol=1e-3))
