import torch

import triton
import triton.language as tl


@triton.jit
def _get_ntk_position(
    cos_cached, 
    sin_cached, 
    b_start_loc,
    b_seq_len,
    infer_ntk_id,
    position_ids,
    cache_stride_0, cache_stride_1, cache_stride_2,
    output_cos, output_sin,
    output_stride_0, output_stride_1,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    start_s = tl.program_id(1)

    cur_ntk_id = tl.load(infer_ntk_id + cur_batch)
    cur_start_loc = tl.load(b_start_loc + cur_batch)
    cur_seq_len = tl.load(b_seq_len + cur_batch)

    offs_s = start_s * BLOCK_SIZE  + tl.arange(0, BLOCK_SIZE) 
    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_pos_id = tl.load(position_ids + cur_start_loc + offs_s, mask=offs_s < cur_seq_len)

    offs_cached = cur_ntk_id * cache_stride_0 + cur_pos_id[:, None] * cache_stride_1 + offs_d[None, :] * cache_stride_2

    cos_cached_ptr = cos_cached + offs_cached
    sin_cached_ptr = sin_cached + offs_cached

    cur_cos_cached = tl.load(cos_cached_ptr, mask=offs_s[:, None] < cur_seq_len)
    cur_sin_cached = tl.load(sin_cached_ptr, mask=offs_s[:, None] < cur_seq_len)

    offs_out = (cur_start_loc + start_s * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)) * output_stride_0 
    output_ptr = offs_out[:, None] + offs_d[None, :]

    tl.store(output_cos + output_ptr, cur_cos_cached, mask=offs_s[:, None] < cur_seq_len)
    tl.store(output_sin + output_ptr, cur_sin_cached, mask=offs_s[:, None] < cur_seq_len)
    



def ntk_position(cos_cached, sin_cached, b_start_loc, b_seq_len, infer_ntk_id, position_ids, max_input_len, output_cos, output_sin):
    
    BLOCK = 64
    head_dim = cos_cached.shape[-1]    
    batch_size = b_seq_len.shape[0]
    grid = (batch_size, triton.cdiv(max_input_len, BLOCK))
    num_warps = 4
    _get_ntk_position[grid](cos_cached, sin_cached, b_start_loc, b_seq_len, infer_ntk_id, position_ids,
                            cos_cached.stride(0), cos_cached.stride(1), cos_cached.stride(2),
                            output_cos, output_sin,
                            output_cos.stride(0), output_cos.stride(1),
                            BLOCK_SIZE=BLOCK, BLOCK_DMODEL=head_dim, num_warps=num_warps)
    return

def test_ntk():
    # create data
    cos_cached = torch.randn(4, 20, 128).cuda()
    sin_cached = torch.randn(4, 20, 128).cuda()
    infer_ntk_id = torch.tensor([0, 1]).cuda().long()
    b_seq_len = torch.tensor([3, 4]).cuda().long()
    b_start_loc = torch.tensor([0, 3]).cuda().long()
    position_ids = torch.tensor([0,1,2,3,4,5,6]).cuda().long()
    max_input_len = 4

    output_cos = torch.randn(7, 128).cuda()
    output_sin = torch.randn(7, 128).cuda()

    origin_cos = torch.cat([cos_cached[0][[0,1,2]], cos_cached[1][[3,4,5,6]]], dim=0)
    origin_sin = torch.cat([sin_cached[0][[0,1,2]], sin_cached[1][[3,4,5,6]]], dim=0)

    ntk_position(cos_cached, sin_cached, b_start_loc, b_seq_len, infer_ntk_id, position_ids, max_input_len, output_cos, output_sin)

    # compare
    print("type:", output_sin.dtype, origin_sin.dtype)
    print("max delta:", torch.max(torch.abs(output_sin[0:3] - origin_sin[0:3])))
    return