#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Simple vLLM metrics visualization."""

import argparse
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import requests


def fetch_metrics(url):
    """Fetch and parse Prometheus metrics."""
    resp = requests.get(f"{url}/metrics", timeout=5)
    metrics = {}
    for line in resp.text.split("\n"):
        if line and not line.startswith("#"):
            parts = line.split()
            if len(parts) >= 2:
                name = parts[0].split("{")[0]  # strip labels
                try:
                    metrics[name] = float(parts[-1])
                except ValueError:
                    pass
    return metrics


def live_plot(url, interval=2, duration=60):
    """Live plot of key metrics over time."""
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"vLLM Metrics - {url}")

    history = defaultdict(list)
    timestamps = []

    start = time.time()
    while time.time() - start < duration:
        try:
            m = fetch_metrics(url)
            timestamps.append(time.time() - start)

            # Track key metrics
            history["requests"].append(m.get("vllm:num_requests_running", 0))
            history["tokens_total"].append(m.get("vllm:generation_tokens_total", 0))
            history["gpu_cache"].append(m.get("vllm:gpu_cache_usage_perc", 0) * 100)
            history["ttft"].append(m.get("vllm:time_to_first_token_seconds_sum", 0))

            # Clear and redraw
            for ax in axes.flat:
                ax.clear()

            axes[0, 0].plot(timestamps, history["requests"], "b-")
            axes[0, 0].set_title("Running Requests")
            axes[0, 0].set_ylabel("Count")

            axes[0, 1].plot(timestamps, history["tokens_total"], "g-")
            axes[0, 1].set_title("Total Tokens Generated")
            axes[0, 1].set_ylabel("Tokens")

            axes[1, 0].plot(timestamps, history["gpu_cache"], "r-")
            axes[1, 0].set_title("GPU Cache Usage")
            axes[1, 0].set_ylabel("%")
            axes[1, 0].set_ylim(0, 100)

            axes[1, 1].plot(timestamps, history["ttft"], "m-")
            axes[1, 1].set_title("TTFT (cumulative)")
            axes[1, 1].set_ylabel("Seconds")

            for ax in axes.flat:
                ax.set_xlabel("Time (s)")
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.pause(interval)

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(interval)

    plt.ioff()
    plt.show()


def snapshot(url):
    """Print current metrics snapshot."""
    m = fetch_metrics(url)

    print(f"\n=== vLLM Metrics Snapshot ({url}) ===\n")

    print("Requests:")
    print(f"  Running:    {m.get('vllm:num_requests_running', 'N/A')}")
    print(f"  Waiting:    {m.get('vllm:num_requests_waiting', 'N/A')}")

    print("\nTokens:")
    print(f"  Generated:  {m.get('vllm:generation_tokens_total', 'N/A')}")
    print(f"  Prompt:     {m.get('vllm:prompt_tokens_total', 'N/A')}")

    print("\nCache:")
    gpu = m.get("vllm:gpu_cache_usage_perc", 0) * 100
    print(f"  GPU Cache:  {gpu:.1f}%")

    print("\nEPLB (RFC #30696 - not yet in metrics):")
    eplb_metrics = {k: v for k, v in m.items() if "eplb" in k.lower()}
    if eplb_metrics:
        for k, v in eplb_metrics.items():
            print(f"  {k}: {v}")
    else:
        print("  (no EPLB metrics - check server logs for balancedness)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize vLLM metrics")
    parser.add_argument(
        "--url", default="http://localhost:8000", help="vLLM server URL"
    )
    parser.add_argument("--live", action="store_true", help="Live plotting mode")
    parser.add_argument(
        "--duration", type=int, default=60, help="Duration for live plot (seconds)"
    )
    parser.add_argument(
        "--interval", type=float, default=2, help="Poll interval (seconds)"
    )
    args = parser.parse_args()

    if args.live:
        live_plot(args.url, args.interval, args.duration)
    else:
        snapshot(args.url)
