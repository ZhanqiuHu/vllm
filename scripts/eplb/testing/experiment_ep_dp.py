#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Experiment script to understand EP, DP, and EPLB.
Run different configurations and observe behavior.

Usage:
    # Experiment 1: EP only (2 GPUs)
    gr 2 python experiment_ep_dp.py --mode ep_only

    # Experiment 2: EP + DP (4 GPUs)
    gr 4 python experiment_ep_dp.py --mode ep_dp

    # Experiment 3: Compare with/without EPLB
    gr 2 python experiment_ep_dp.py --mode compare_eplb
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import requests

MODEL = "deepseek-ai/DeepSeek-V2-Lite"
PORT = 8000
URL = f"http://localhost:{PORT}"


def start_server(tp_size, dp_size=1, enable_ep=True, enable_eplb=False, extra_args=""):
    """Start vLLM server with given configuration."""
    cmd = f"""vllm serve {MODEL} \
        --tensor-parallel-size {tp_size} \
        --port {PORT}"""

    if dp_size > 1:
        cmd += f" --data-parallel-size {dp_size}"

    if enable_ep:
        cmd += " --enable-expert-parallel"

    if enable_eplb:
        cmd += """ --enable-eplb --eplb-config '{"log_balancedness": true, "log_balancedness_interval": 10, "step_interval": 100}'"""

    cmd += f" {extra_args}"

    print(f"\n{'=' * 60}")
    print("Starting server with:")
    print(f"  TP={tp_size}, DP={dp_size}, EP={enable_ep}, EPLB={enable_eplb}")
    print(f"{'=' * 60}")
    print(f"Command: {cmd}\n")

    return cmd


def wait_for_server(timeout=300):
    """Wait for server to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{URL}/health", timeout=5)
            if r.status_code == 200:
                print("Server is ready!")
                return True
        except:
            pass
        time.sleep(5)
        print("Waiting for server...")
    return False


def send_request(prompt, max_tokens=50):
    """Send a completion request."""
    try:
        r = requests.post(
            f"{URL}/v1/completions",
            json={"model": MODEL, "prompt": prompt, "max_tokens": max_tokens},
            timeout=60,
        )
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def run_load_test(num_requests=100, concurrent=10):
    """Run concurrent requests and measure throughput."""
    prompts = [
        "Explain machine learning",
        "What is Python?",
        "Describe neural networks",
        "How does attention work?",
        "What is expert parallelism?",
    ]

    print(f"\nRunning load test: {num_requests} requests, {concurrent} concurrent")

    start = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=concurrent) as executor:
        futures = [
            executor.submit(send_request, prompts[i % len(prompts)])
            for i in range(num_requests)
        ]
        for f in futures:
            results.append(f.result())

    elapsed = time.time() - start
    success = sum(1 for r in results if "error" not in r)

    print("Results:")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Successful: {success}/{num_requests}")
    print(f"  Throughput: {num_requests / elapsed:.2f} req/s")

    return {
        "elapsed": elapsed,
        "success": success,
        "throughput": num_requests / elapsed,
    }


def check_metrics():
    """Check Prometheus metrics."""
    try:
        r = requests.get(f"{URL}/metrics", timeout=5)
        lines = r.text.split("\n")

        # Only show a few key metrics
        print("\n--- Key Metrics ---")
        shown = 0
        for line in lines:
            if line.startswith("#"):
                continue
            if (
                "num_requests_running" in line
                or "num_requests_waiting" in line
                or "gpu_cache_usage" in line
            ):
                print(f"  {line}")
                shown += 1
            if shown >= 5:
                break

    except Exception as e:
        print(f"Error fetching metrics: {e}")


def explain_config(mode):
    """Explain what each configuration does."""
    explanations = {
        "ep_only": """
EP Only (Expert Parallelism):
- Experts are distributed across GPUs
- Each GPU holds different experts (e.g., GPU0: E0-31, GPU1: E32-63)
- All tokens go through all GPUs for attention (replicated)
- Tokens are routed via All2All to the GPU holding their expert
- Good for: Memory efficiency, single-user latency
""",
        "ep_dp": """
EP + DP (Expert + Data Parallelism):
- Experts distributed across ALL GPUs (EP group = TP x DP)
- Requests are split across DP groups
- Each DP group handles different requests independently
- Within each DP group, experts are distributed via EP
- Good for: High throughput, many concurrent users
""",
        "compare_eplb": """
EPLB (Expert Parallel Load Balancer):
- Without EPLB: Fixed expert placement, some experts may be overloaded
- With EPLB: Dynamic rebalancing based on actual load
  - Monitors which experts are "hot" (receive many tokens)
  - Creates redundant copies of hot experts
  - Rebalances every step_interval steps
- Good for: Production workloads with skewed expert usage
""",
    }
    print(explanations.get(mode, "Unknown mode"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["ep_only", "ep_dp", "compare_eplb"], default="ep_only"
    )
    parser.add_argument("--num-requests", type=int, default=50)
    args = parser.parse_args()

    explain_config(args.mode)

    if args.mode == "ep_only":
        cmd = start_server(tp_size=2, enable_ep=True, enable_eplb=False)
        print("\nTo run this experiment:")
        print(f"  1. Start server: gr 2 {cmd}")
        print(
            f"  2. Run load test: python {__file__} --mode ep_only --num-requests 100"
        )
        print("  3. Watch server logs for expert routing")

    elif args.mode == "ep_dp":
        cmd = start_server(tp_size=1, dp_size=4, enable_ep=True, enable_eplb=False)
        print("\nTo run this experiment:")
        print(f"  1. Start server: gr 4 {cmd}")
        print("  2. Run load test with high concurrency")
        print("  3. Compare throughput vs EP-only")

    elif args.mode == "compare_eplb":
        print("\nTo compare with/without EPLB:")
        print("1. Run WITHOUT EPLB:")
        cmd1 = start_server(tp_size=2, enable_ep=True, enable_eplb=False)
        print(f"   gr 2 {cmd1}")
        print("\n2. Run WITH EPLB:")
        cmd2 = start_server(tp_size=2, enable_ep=True, enable_eplb=True)
        print(f"   gr 2 {cmd2}")
        print("\n3. Compare server logs - look for:")
        print("   - 'EPLB step: X: avg_tokens=..., max_tokens=..., balancedness=...'")
        print("   - Balancedness closer to 1.0 = better balanced")

    # If server is running, run the actual test
    try:
        r = requests.get(f"{URL}/health", timeout=2)
        if r.status_code == 200:
            print("\n" + "=" * 60)
            print("Server detected! Running load test...")
            print("=" * 60)
            run_load_test(args.num_requests)
            check_metrics()
    except:
        print("\n(Server not running - start it first with the commands above)")


if __name__ == "__main__":
    main()
