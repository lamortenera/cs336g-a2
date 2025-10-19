"""Implementation of a FlashAttention2 kernel."""

from torch import autograd, Tensor
from jaxtyping import Float, Bool, Int
import torch
from einops import einsum, rearrange
import math

import triton
import triton.language as tl


def ceiling_division(num, denom):
    return -(num // (-denom))


class FlashAttentionFuncPytorch(autograd.Function):
    @staticmethod
    def forward(ctx, Q: Float[Tensor, "... seq_len d"],
                K: Float[Tensor, "... seq_len d"],
                V: Float[Tensor, "... seq_len d"],
                is_causal: bool = False, Q_tile_size: int = 16, K_tile_size: int = 32) -> Float[Tensor,
                                                                                                "... seq_len d"]:
        # d <= 64 based on assignment description
        # 4(b**2 + 4*b*d + 2b) <= (L1 cache size)
        # L1 cache size in my crappy laptop 64kb, but it doesn't get way
        # bigger (128kb on L4 and 196 on really good GPUs)
        # b**2 + 514*b <= 16000
        # it's about b <= 29 on my crappy laptop, 32 should work in most cases.
        block_size = 32

        assert Q.shape == K.shape
        assert V.shape == Q.shape

        orig_shape = Q.shape

        Q = Q.reshape(-1, orig_shape[-2], orig_shape[-1])
        K = K.reshape(-1, orig_shape[-2], orig_shape[-1])
        V = V.reshape(-1, orig_shape[-2], orig_shape[-1])

        batch_size, N, d = Q.shape

        # Q_tile_size = N
        # K_tile_size = N

        O = torch.empty(batch_size, N, d, dtype=Q.dtype)
        L = torch.empty(batch_size, N, dtype=torch.float32)

        # Do work
        Q_tiles = ceiling_division(N, Q_tile_size)
        for b in range(batch_size):
            for i in range(Q_tiles):
                Q_i = Q[b, i*Q_tile_size:(i+1)*Q_tile_size, :]
                O_i = O[b, i*Q_tile_size:(i+1)*Q_tile_size, :]
                L_i = L[b, i*Q_tile_size:(i+1)*Q_tile_size]
                FlashAttentionFuncPytorch._inner_loop(
                    Q_i, K[b], V[b], K_tile_size, O_i, L_i)

        ctx.save_for_backward(L)
        O = O.reshape(orig_shape)
        return O

    @staticmethod
    def _inner_loop(Q: Float[Tensor, "Q_tile_size d"],
                    K: Float[Tensor, "N d"],
                    V: Float[Tensor, "N d"],
                    K_tile_size: int,
                    O: Float[Tensor, "Q_tile_size d"],
                    L: Float[Tensor, "Q_tile_size"]):
        Q_tile_size, d = Q.shape
        N, _ = K.shape
        K_tiles = ceiling_division(N, K_tile_size)
        m = torch.full((Q_tile_size, ), float("-inf"), dtype=torch.float32)
        l = torch.zeros(Q_tile_size, dtype=torch.float32)
        O_curr = torch.zeros(Q_tile_size, d, dtype=torch.float32)
        for j in range(K_tiles):
            K_j = K[j*K_tile_size:(j+1)*K_tile_size]
            V_j = V[j*K_tile_size:(j+1)*K_tile_size]
            S = (Q @ K_j.T) / math.sqrt(d)
            next_m = torch.maximum(m, torch.max(S, dim=-1).values)

            P = torch.exp(S - next_m[:, None])
            l_j = P.sum(axis=-1, dtype=torch.float32)

            correction = torch.exp(m - next_m)

            l[...] = l * correction + l_j
            m[...] = next_m
            O_curr[...] = O_curr * correction[:, None] + P @ V_j

        O[...] = O_curr / l[:, None]
        L[...] = m + torch.log(l)


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    m = tl.full((Q_TILE_SIZE, ), float("-inf"), dtype=torch.float32)
    l = tl.zeros(Q_TILE_SIZE, dtype=torch.float32)
    O_curr = tl.zeros(Q_TILE_SIZE, D, dtype=torch.float32)

    Q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    for _ in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        S = tl.dot(Q, K_j.T, out_dtype=tl.float32) / scale
        next_m = tl.maximum(m, tl.max(S, axis=-1))

        P = tl.exp(S - next_m[:, None])
        l_j = P.sum(axis=-1, dtype=tl.float32)

        correction = tl.exp(m - next_m)

        l *= correction
        l += l_j
        m[...] = next_m
        O_curr *= correction[:, None]
        tl.dot(P.to(V_j.dtype), V_j, acc=O_curr)

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    O_curr /= l[:, None]

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    tl.store(
        O_block_ptr, O_curr.to(O_block_ptr.type.element_ty),
        boundary_check=(0, 1))

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES),
        strides=(stride_lq),
        offsets=(query_tile_index * Q_TILE_SIZE),
        block_shape=(Q_TILE_SIZE),
        order=(0,)
    )

    tl.store(L_block_ptr, m + torch.log(l), boundary_check=(0,))


class FlashAttentionFunc(autograd.Function):
    @staticmethod
    def forward(ctx, Q: Float[Tensor, "... seq_len d"],
                K: Float[Tensor, "... seq_len d"],
                V: Float[Tensor, "... seq_len d"],
                is_causal: bool = False) -> Float[Tensor, "... seq_len d"]:
        # d <= 64 based on assignment description
        # 4(b**2 + 4*b*d + 2b) <= (L1 cache size)
        # L1 cache size in my crappy laptop 64kb, but it doesn't get way
        # bigger (128kb on L4 and 196 on really good GPUs)
        # b**2 + 514*b <= 16000
        # it's about b <= 29 on my crappy laptop, 32 should work in most cases.
        ctx.Q_TILE_SIZE = 32
        ctx.K_TILE_SIZE = 32


        assert Q.shape == K.shape
        assert V.shape == Q.shape
        assert Q.is_cuda and K.is_cuda and V.is_cuda
        assert Q.is_contiguous() and K.is_contiguous and V.is_contiguous

        input_shape = Q.shape
        Q = rearrange(Q, "... seq_len d -> (...) seq_len d")
        K = rearrange(K, "... seq_len d -> (...) seq_len d")
        K = rearrange(V, "... seq_len d -> (...) seq_len d")
        
        batch_size, N, d = Q.shape

        O = torch.empty(batch_size, N, d, dtype=Q.dtype, device=Q.device)
        L = torch.empty(batch_size, N, dtype=torch.float32, device=Q.device)

        Q_tiles = tl.cdiv(N, ctx.Q_TILE_SIZE)
        flash_fwd_kernel[(Q_tiles, batch_size)](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES=N, N_KEYS=N,
            scale=math.sqrt(d),
            Q_TILE_SIZE=ctx.Q_TILE_SIZE,
            K_TILE_SIZE=ctx.K_TILE_SIZE)

        ctx.save_for_backward(L)
        O = O.reshape(input_shape)
        return O

    @staticmethod
    def backward(ctx):
        # Note: You will need to update the signature of this method
        raise NotImplementedError(
            "TODO: Implement FlashAttentionFunc.backward"
        )
