#!/usr/bin/env python3
"""Benchmark concurrent serving throughput on long prompt and response for Ornith 1.5 9B."""

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

LONG_CONTEXT_BASE = """
Modern artificial intelligence architectures have converged on sequence transduction paradigms 
rooted in transformer architectures and linear recurrent mechanisms. 

1. Attention Mechanisms and Quadratic Complexity
The standard scaled dot-product attention computes Attention(Q, K, V) = softmax(Q K^T / sqrt(d)) V.
While this formulation enables all-to-all token interactions and achieves exceptional empirical performance,
its quadratic time and memory complexity O(T^2) with respect to sequence length T presents severe computational
bottlenecks during prefill and decode phases for ultra-long context windows exceeding hundreds of thousands of tokens.
Specifically, computing the full attention matrix across sequence length T requires T^2 operations per head,
and caching key-value representations across decoding steps requires O(T * L * H * D) memory storage.

2. Linear Attention and Recurrent Formulations
To circumvent the quadratic complexity, linear attention reformulations compute attention via kernel mappings
or recurrent state updates S_t = S_{t-1} + k_t^T v_t, where S_t in R^{d_k x d_v} acts as a fixed-size recurrent memory.
During autoregressive generation, the hidden state size remains constant regardless of sequence position O(1),
eliminating the memory growth of the KV cache. However, standard linear attention lacks the expressive power
to selectively forget and overwrite obsolete information, leading to retrieval degradation over long sequences.

3. Gated DeltaNet (GDN) and State Space Models
The Gated DeltaNet architecture introduces input-dependent retention and forgetting gates, combining the linear
complexity of RNNs with high expressivity. In GDN, the update rule incorporates a delta error correction mechanism:
S_t = alpha_t * S_{t-1} + beta_t * (v_t - S_{t-1} k_t) k_t^T.
This allows the recurrent state to selectively overwrite previous associations when new information updates a key.
Furthermore, depthwise 1D convolutions precede the projection layers to provide local context mixing.

4. Hybrid Layer Stacking
In models such as Qwen 3.5 and Ornith 1.5, full attention and gated linear attention are stacked in a 1:3 ratio
(e.g., 3 linear attention layers followed by 1 full attention layer across 32 decoder layers).
This hybrid topology preserves the strong in-context associative recall of full softmax attention while
reducing the overall KV cache footprint and prefill FLOPs by approximately 75% across the backbone.

5. Speculative Decoding with Multi-Token Prediction (MTP)
To accelerate autoregressive token generation, Multi-Token Prediction (MTP) appends dedicated draft heads
or lightweight transformer layers trained to predict future tokens k+1, k+2, ... concurrently.
During inference, candidate draft tokens are generated and validated in parallel through the base model in a single forward pass,
significantly improving token generation throughput per round when acceptance rates remain high.
"""

def generate_long_prompt(target_word_count: int = 1200) -> str:
    repetitions = (target_word_count // len(LONG_CONTEXT_BASE.split())) + 1
    body = (LONG_CONTEXT_BASE + "\n") * repetitions
    prompt = (
        f"{body}\n\n"
        "Question: Based on the technical exposition above, explain the operational trade-offs "
        "between quadratic softmax attention, linear recurrence, and hybrid topologies in modern language models. "
        "Discuss memory consumption, prefill computational complexity, autoregressive decoding latency, and "
        "the role of speculative multi-token prediction (MTP) in maximizing hardware throughput on consumer GPUs."
    )
    return prompt


def wait_for_port(host: str, port: int, timeout: float = 180.0) -> bool:
    start = time.time()
    url = f"http://{host}:{port}/health"
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(1.0)
    return False


def send_chat_completion(port: int, prompt: str, max_tokens: int = 256) -> dict:
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = {
        "model": "ornith-1.5-9b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300.0) as resp:
        t1 = time.perf_counter()
        body = json.loads(resp.read().decode("utf-8"))
        elapsed = t1 - t0
        content = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", len(content.split()))
        return {
            "elapsed": elapsed,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "content": content,
            "tok_per_sec": completion_tokens / max(elapsed, 1e-4),
        }


def run_concurrency_wave(port: int, concurrency: int, prompt_template: str, max_tokens: int = 256) -> dict:
    print(f"\n>>> Running wave C={concurrency} with {concurrency} parallel request(s)...")
    prompts = [prompt_template + f"\n[Session Lane Index: {i+1}]" for i in range(concurrency)]

    t_batch_start = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(send_chat_completion, port, prompts[i], max_tokens)
            for i in range(concurrency)
        ]
        for future in as_completed(futures):
            results.append(future.result())
    t_batch_end = time.perf_counter()

    batch_elapsed = t_batch_end - t_batch_start
    total_gen_tokens = sum(r["completion_tokens"] for r in results)
    prompt_tokens_each = results[0]["prompt_tokens"]
    aggregate_throughput = total_gen_tokens / max(batch_elapsed, 1e-4)

    print(f"C={concurrency} completed in {batch_elapsed:.3f}s:")
    print(f"  Prompt tokens: {prompt_tokens_each} per request")
    print(f"  Total tokens generated: {total_gen_tokens}")
    print(f"  Aggregate throughput: {aggregate_throughput:.2f} tok/s")
    for idx, res in enumerate(results):
        print(f"    Lane {idx+1}: {res['completion_tokens']} tok in {res['elapsed']:.2f}s ({res['tok_per_sec']:.2f} tok/s)")

    return {
        "concurrency": concurrency,
        "prompt_tokens": prompt_tokens_each,
        "total_gen_tokens": total_gen_tokens,
        "batch_elapsed": batch_elapsed,
        "aggregate_tok_per_sec": aggregate_throughput,
        "sample_content": results[0]["content"][:120].strip(),
    }


def main():
    parser = argparse.ArgumentParser(description="Concurrent Long Prompt/Response Benchmark")
    parser.add_argument("--model", default="models/ornith_1_5_9b.ninfer")
    parser.add_argument("--concurrencies", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--kv-dtype",
        choices=("bf16", "int8", "fp8", "nvfp4", "k8v4"),
        default="int8",
        help="KV-cache storage dtype (default: int8)",
    )
    args = parser.parse_args()

    max_c = max(args.concurrencies)
    cmd = [
        "./build/apps/ninfer-serve",
        args.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--max-context",
        "4096",
        "--kv-capacity",
        "32768",
        "--max-concurrency",
        str(max_c),
        "--prefill-chunk",
        "4096",
        "--kv-dtype",
        args.kv_dtype,
        "--spec",
        "mtp",
        "--draft-tokens",
        "3",
        "--lm-head-draft",
        "--model-id",
        "ornith-1.5-9b",
        "--default-max-tokens",
        str(args.max_tokens),
        "--cors",
    ]

    print("=" * 80)
    print(f"Starting single persistent NInfer server on port {args.port} (max-concurrency={max_c})...")
    print("Command:", " ".join(cmd))
    print("=" * 80)

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    try:
        print("Waiting for server to become ready...")
        if not wait_for_port("127.0.0.1", args.port, timeout=180.0):
            print("ERROR: Server timed out starting up!")
            proc.terminate()
            out, _ = proc.communicate(timeout=5)
            print(out)
            sys.exit(1)

        print(f"Server ready! Generating long prompt template...")
        prompt = generate_long_prompt(target_word_count=1200)

        # Warm-up request
        print("Sending short warm-up request...")
        warmup = send_chat_completion(args.port, "Hello world.", max_tokens=16)
        print(f"Warm-up complete ({warmup['completion_tokens']} tok generated in {warmup['elapsed']:.2f}s).")

        summaries = []
        for c in args.concurrencies:
            s = run_concurrency_wave(args.port, c, prompt, max_tokens=args.max_tokens)
            summaries.append(s)
            time.sleep(2)

        print("\n" + "=" * 85)
        print("          ORNITH 1.5 9B CONCURRENCY BENCHMARK RESULTS (RTX 5060 Ti 16GB)          ")
        print("=" * 85)
        c1_tps = summaries[0]["aggregate_tok_per_sec"] if summaries else 1.0
        print(f"{'Concurrency':<12} | {'Prompt Tok':<11} | {'Gen Tok':<9} | {'Batch Time':<12} | {'Throughput':<15} | {'Scaling':<10}")
        print("-" * 85)
        for s in summaries:
            scaling = s["aggregate_tok_per_sec"] / c1_tps
            print(
                f"C={s['concurrency']:<10} | {s['prompt_tokens']:<11} | {s['total_gen_tokens']:<9} | "
                f"{s['batch_elapsed']:>8.3f} s   | {s['aggregate_tok_per_sec']:>9.2f} tok/s | {scaling:>7.2f}x"
            )
        print("=" * 85)
        print("\nSample Output from C=8 Lane 1:")
        print(f"\"{summaries[-1]['sample_content']}...\"\n")

    finally:
        print(f"Shutting down server (PID {proc.pid})...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        print("Server shutdown complete.")


if __name__ == "__main__":
    main()
