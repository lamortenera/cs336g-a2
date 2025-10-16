from cs336_basics import model as model_module
from cs336_basics import optimizer as optimizer_module
from cs336_basics import nn_utils
import argparse
import torch
import timeit
import pandas as pd
import copy
import math
import torch.cuda.nvtx as nvtx

from einops import rearrange, einsum
from torch import Tensor
from jaxtyping import Float, Bool, Int
from cs336_basics.nn_utils import softmax


def parse_bool(s):
    s = s.strip().lower()
    assert s in ["true", "false"]
    return s == "true"


@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
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


def inner_loop(model, inputs, labels, forward_only=False, optimizer=None):
    with nvtx.range("Forward pass"):
        outputs = model(inputs)
        loss = nn_utils.cross_entropy(outputs, labels)

    if not forward_only:
        with nvtx.range("Backward pass"):
            loss.backward()

    if optimizer is not None:
        with nvtx.range("Optimizer step"):
            optimizer.step()
            optimizer.zero_grad()

    torch.cuda.synchronize()
    print(f"Loss: {loss.item():.3f}")


def benchmark(args):
    model_module.scaled_dot_product_attention = annotated_scaled_dot_product_attention

    m = model_module.BasicsTransformerLM(
        vocab_size=args.vocab_size, num_layers=args.num_layers,
        d_model=args.d_model, num_heads=args.num_heads, d_ff=args.d_ff,
        context_length=args.context_length, rope_theta=args.rope_theta).to(
        args.device)
    print("Model created")

    optimizer = None
    if parse_bool(args.with_optimizer):
        optimizer = optimizer_module.AdamW(m.parameters())

    tokens = torch.randint(
        args.vocab_size, (args.batch_size, args.context_length + 1)).to(args.device)
    inputs, labels = tokens[:, :-1], tokens[:, 1:]

    forward_only = parse_bool(args.forward_only)

    print("Warmup steps")
    with nvtx.range("Warmup steps"):
        for _ in range(args.warmup_steps):
            inner_loop(m, inputs, labels, forward_only, optimizer)

    print("Benchmarking steps")
    with nvtx.range("Profiling steps"):
        timings = []
        for _ in range(args.benchmark_steps):
            t_start = timeit.default_timer()
            inner_loop(m, inputs, labels, forward_only, optimizer)
            t_end = timeit.default_timer()
            timings.append(t_end - t_start)

        timings = pd.Series(timings)
    stats = timings.describe()
    print("Stats for: ", args.label)
    print(stats)

    return stats.to_dict()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--warmup_steps", help="Number of warmup steps",
        type=int, default=5)
    parser.add_argument(
        "--benchmark_steps",
        help="Steps for benchmarking",
        type=int, default=10)
    parser.add_argument(
        "--forward_only",
        help="Benchmark only the forward pass",
        type=str, default="False")
    parser.add_argument(
        "--with_optimizer",
        help="Add an optimizer step",
        type=str, default="False")
    parser.add_argument("--output", help="Path to json output", type=str)
    parser.add_argument(
        "--vocab_size",
        help="Total number of tokens, it will be assumed the largest token is vocab_size -1",
        type=int, default=10000)
    parser.add_argument("--batch_size", help="Batch size", type=int, default=4)
    parser.add_argument("--context_length",
                        help="Context length", type=int, nargs="+")
    parser.add_argument(
        "--num_layers", help="Number of layers", type=int, nargs="+")
    parser.add_argument(
        "--num_heads", help="Number of heads", type=int, nargs="+")
    parser.add_argument(
        "--d_model", help="Model embedding size", type=int, nargs="+")
    parser.add_argument(
        "--d_ff", help="Dimension of feed-forward layer", type=int, nargs="+")
    parser.add_argument(
        "--label", help="Label for the model configuration", nargs="+")
    parser.add_argument(
        "--rope_theta",
        help="Theta param for rotary position embeddings",
        type=float, default=10000)
    parser.add_argument("--device", help="The device to use", default="cpu")
    parser.add_argument("--dtype", help="The dtype to use", default="float32")
    args = parser.parse_args()

    num_configs = len(args.label)
    repeated_args = ["context_length", "num_layers",
                     "num_heads", "d_model", "d_ff"]
    for arg in repeated_args:
        val_list = getattr(args, arg)
        assert ((len(val_list) == num_configs) or (len(val_list) == 1)), f"Expected: {
            num_configs}, found: {len(getattr(args, arg))}  "

    all_stats = []
    for i in range(num_configs):
        config_args = copy.deepcopy(args)
        config_args.label = args.label[i]
        d = {"label": args.label[i]}
        for arg in repeated_args:
            val_list = getattr(args, arg)
            val = val_list[0] if len(val_list) == 1 else val_list[i]
            d[arg] = val
            setattr(config_args, arg, val)
        with nvtx.range(config_args.label):
            perf_stats = benchmark(config_args)
        d.update(perf_stats)
        all_stats.append(d)
    all_stats = pd.DataFrame(all_stats)
    print(all_stats)

    if args.output:
        all_stats.to_csv(args.output)
