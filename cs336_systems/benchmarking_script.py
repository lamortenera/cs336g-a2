from cs336_basics import model, nn_utils
import argparse
import torch
import timeit
import pandas as pd


def parse_bool(s):
    s = s.strip().lower()
    assert s in ["true", "false"]
    return s == "true"


def inner_loop(model, inputs, labels, forward_only=False):
    outputs = model(inputs)
    if not forward_only:
        loss = nn_utils.cross_entropy(outputs, labels)
        loss.backward()
    torch.cuda.synchronize()


def benchmark(args):
    m = model.BasicsTransformerLM(
        vocab_size=args.vocab_size, num_layers=args.num_layers,
        d_model=args.d_model, num_heads=args.num_heads, d_ff=args.d_ff,
        context_length=args.context_length, rope_theta=args.rope_theta).to(
        args.device)
    print("Model created")

    tokens = torch.randint(
        args.vocab_size, (args.batch_size, args.context_length + 1)).to(args.device)
    inputs, labels = tokens[:, :-1], tokens[:, 1:]

    forward_only = parse_bool(args.forward_only)

    print("Warmup steps")
    for _ in range(args.warmup_steps):
        inner_loop(m, inputs, labels, forward_only)

    print("Benchmarking steps")
    timings = []
    for _ in range(args.benchmark_steps):
        t_start = timeit.default_timer()
        inner_loop(m, inputs, labels, forward_only)
        t_end = timeit.default_timer()
        timings.append(t_end - t_start)
    
    timings = pd.Series(timings)
    print(timings.describe())

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
        "--vocab_size",
        help="Total number of tokens, it will be assumed the largest token is vocab_size -1",
        type=int, default=10000)
    parser.add_argument("--batch_size", help="Batch size", type=int, default=32)
    parser.add_argument("--context_length",
                        help="Context length", type=int, default=256)
    parser.add_argument(
        "--num_layers", help="Number of layers", type=int, default=4)
    parser.add_argument(
        "--num_heads", help="Number of heads", type=int, default=16)
    parser.add_argument(
        "--d_model", help="Model embedding size", type=int, default=512)
    parser.add_argument(
        "--d_ff", help="Dimension of feed-forward layer", type=int,
        default=1344)
    parser.add_argument(
        "--rope_theta",
        help="Theta param for rotary position embeddings",
        type=float, default=10000)
    parser.add_argument("--device", help="The device to use", default="cpu")
    parser.add_argument("--dtype", help="The dtype to use", default="float32")
    args = parser.parse_args()

    benchmark(args)
