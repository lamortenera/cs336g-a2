import copy
import argparse
import torch.cuda.nvtx as nvtx
import torch
import timeit
import math
import pandas as pd
from torch import Tensor
from einops import rearrange, einsum
from jaxtyping import Float, Bool, Int
from cs336_basics.nn_utils import softmax


def parse_bool(s):
    s = s.strip().lower()
    assert s in ["true", "false"]
    return s == "true"


@nvtx.range("scaled dot product attention")
def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """Scaled dot-product attention.

    This function implements Eq. 1 of the Transformer paper.

    Args:
        Q: Tensor of queries, may have any number of leading dimensions.
        K: Tensor of keys, sharing leading dimensions with Q.
        V: Tensor of values, sharding leading dimensions with Q and K.
        mask: An (optional) mask of shape (..., seq_len, seq_len).
            Attention scores for positions with a mask value of `False` should
            be masked out, i.e., not affect the softmaxed attention probabilities.

    Returns:
        torch.FloatTensor of shape (..., seq_len, value_dimension)
        with the output of running your scaled dot product attention
        implementation with the provided key, query, and value tensors.
    """

    d_k = K.shape[-1]
    with nvtx.range("computing attention scores"):
        attention_scores = einsum(
            Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx.range("computing softmax"):
        # Softmax over the key dimension
        attention_weights = softmax(attention_scores, dim=-1)

    with nvtx.range("final matmul"):
        result = einsum(
            attention_weights, V,
            "... query key, ... key d_v ->  ... query d_v")
    return result


def get_mem(device_index):
    if device_index is None:
        return 0
    return torch.cuda.device_memory(device_index)


def inner_loop(Q, K, V, device_index):
    t_start = timeit.default_timer()
    m_start = get_mem(device_index)
    with nvtx.range("Forward"):
        O = scaled_dot_product_attention(Q, K, V)
        loss = O.sum()
        torch.cuda.synchronize()
    t_fend = timeit.default_timer()
    m_fend = get_mem(device_index)

    with nvtx.range("Backward"):
        loss.backward()
        torch.cuda.synchronize()

    t_bend = timeit.default_timer()
    m_bend = get_mem(device_index)

    return {"t_forward": t_fend - t_start,
            "t_backward": t_bend - t_fend,
            "m_start": m_start,
            "m_forward": m_fend,
            "m_backward": m_bend
            }


def to_series(df):
    stacked = df.stack().reset_index().sort_values("level_1")
    stacked = stacked.set_index(
        stacked["level_1"].str.cat(stacked["level_0"], sep="_"))
    return stacked[0]


def benchmark(args):

    Q = torch.nn.Parameter(torch.randn(args.batch_size, args.context_length,
                                       args.d_model)).to("cuda")
    K = torch.nn.Parameter(torch.randn(args.batch_size, args.context_length,
                                       args.d_model)).to("cuda")
    V = torch.nn.Parameter(torch.randn(args.batch_size, args.context_length,
                                       args.d_model)).to("cuda")

    device_index = None
    if parse_bool(args.memory_profile):
        device_index = torch.cuda.device("cuda").idx

    with nvtx.range("Warmup steps"):
        for _ in range(args.warmup_steps):
            _ = inner_loop(Q, K, V, device_index)

    with nvtx.range("Forward benchmarking steps"):
        stats = []
        for _ in range(args.benchmark_steps):
            stat = inner_loop(Q, K, V, device_index)
            stats.append(stat)

    stats = pd.DataFrame(stats)
    stats = stats.describe()
    print("Stats for: ", args.label)
    print(stats)

    return to_series(stats.loc[["mean", "std"]]).to_dict()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--warmup_steps", help="Number of warmup steps",
        type=int, default=5)
    parser.add_argument(
        "--benchmark_steps",
        help="Steps for benchmarking",
        type=int, default=100)
    parser.add_argument(
        "--memory_profile", help="If true, profile memory", type=str,
        default="False")
    parser.add_argument(
        "--memory_output",
        help="If present, materialized detailed mem stats here")
    parser.add_argument("--output", help="Path to json output", type=str)
    parser.add_argument("--batch_size", help="Batch size", type=int, default=8)
    parser.add_argument("--context_length",
                        help="Context length", type=int, nargs="+")
    parser.add_argument(
        "--d_model", help="Model embedding size", type=int, nargs="+")
    args = parser.parse_args()

    if args.memory_output:
        torch.cuda.memory._record_memory_history(max_entries=1000000)

    all_stats = []
    for context_length in args.context_length:
        for d_model in args.d_model:
            label = f"context_length={context_length},d_model={d_model}"
            single_arg = copy.deepcopy(args)
            single_arg.label = label
            single_arg.context_length = context_length
            single_arg.d_model = d_model
            print(f"Processing {label}")
            try:
                stats_dict = {"label": label}
                stats_dict.update(benchmark(single_arg))
                all_stats.append(stats_dict)
            except Exception as e:
                print(f"Failed loop for {label} with error:\n{e}")
    all_stats = pd.DataFrame(all_stats)

    if args.memory_output:
        torch.cuda.memory._dump_snapshot(args.memory_output)
        torch.cuda.memory._record_memory_history(enabled=None)

    if args.output:
        all_stats.to_csv(args.output)
