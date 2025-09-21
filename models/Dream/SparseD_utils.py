import torch
import torch.nn.functional as F

from torch.nn.attention.flex_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE,
    create_block_mask,
    create_mask,
    flex_attention
)

def create_block_mask_cached(score_mod, B, H, M, N, device="cuda", _compile=True):
    block_mask = create_block_mask(
        score_mod, B, H, M, N, 
        device=device, 
        _compile=_compile,
    )
    return block_mask

def customize_mask(mask, block_size=128):
    def process(b, h, q_idx, kv_id):
        return mask[b, h, q_idx//block_size, kv_id//block_size]
    return process

def create_attention_block_mask(attn_weights, block_size=128, keep_ratio=0.5):
    bsz, num_heads, q_len, kv_len = attn_weights.shape
    num_blocks_q = (q_len + block_size - 1) // block_size
    num_blocks_kv = (kv_len + block_size - 1) // block_size
    pad_q = (num_blocks_q * block_size - q_len) % block_size
    pad_kv = (num_blocks_kv * block_size - kv_len) % block_size
    padded_attn = F.pad(attn_weights, (0, pad_kv, 0, pad_q))
    
    blocks = padded_attn.unfold(2, block_size, block_size).unfold(3, block_size, block_size)
    block_importance = blocks.abs().mean(dim=(-1, -2))  # (bsz, num_heads, num_blocks_q, num_blocks_kv)
    keep_per_q = max(1, int(num_blocks_kv * keep_ratio)+1)  
    
    _, topk_indices = torch.topk(block_importance, k=keep_per_q, dim=-1, sorted=False)  # (bsz, num_heads, num_blocks_q, keep_per_q)
    
    block_mask = torch.zeros_like(block_importance, dtype=torch.bool) 
    block_mask.scatter_(dim=-1, index=topk_indices, value=True)  # (bsz, num_heads, num_blocks_q, num_blocks_kv)
    
    expanded_mask = block_mask.unsqueeze(-1).unsqueeze(-1)
    
    # (bsz, num_heads, num_blocks_q, num_blocks_kv, block_size, block_size)
    expanded_mask = expanded_mask.expand(-1, -1, -1, -1, block_size, block_size)
    merged_mask = expanded_mask.permute(0, 1, 2, 4, 3, 5).contiguous()
    merged_mask = merged_mask.reshape(bsz, num_heads, num_blocks_q * block_size, num_blocks_kv * block_size)
    final_mask = merged_mask[:, :, :q_len, :kv_len]
    
    final_mask = F.pad(final_mask, (0, pad_kv, 0, pad_q), mode='constant', value=True)
    final_mask = final_mask.unfold(2, block_size, block_size).unfold(3, block_size, block_size)
    final_mask = torch.all(final_mask, dim=(-1, -2))
    return final_mask
