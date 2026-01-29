#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Simple vLLM Dashboard - links to Prometheus for time series."""

import gradio as gr
import requests

VLLM_URL = "http://localhost:8000"
PROM_URL = "http://localhost:9090"
MODEL = "deepseek-ai/DeepSeek-V2-Lite"


def send_request(prompt, max_tokens):
    try:
        r = requests.post(
            f"{VLLM_URL}/v1/completions",
            json={"model": MODEL, "prompt": prompt, "max_tokens": int(max_tokens)},
            timeout=60,
        )
        return r.json().get("choices", [{}])[0].get("text", "No response")
    except Exception as e:
        return f"Error: {e}"


with gr.Blocks(title="vLLM Dashboard") as app:
    gr.Markdown("# vLLM Dashboard")

    gr.Markdown("""
### Prometheus Time Series (real-time)
- [Tokens/sec](http://localhost:9090/graph?g0.expr=rate(vllm%3Ageneration_tokens_total%5B30s%5D)&g0.tab=0)
- [Requests Running](http://localhost:9090/graph?g0.expr=vllm%3Anum_requests_running&g0.tab=0)
- [Requests Waiting](http://localhost:9090/graph?g0.expr=vllm%3Anum_requests_waiting&g0.tab=0)
- [KV Cache Usage](http://localhost:9090/graph?g0.expr=vllm%3Akv_cache_usage_perc&g0.tab=0)

### Grafana Dashboard
- [Grafana](http://localhost:3000) (admin/admin)
""")

    gr.Markdown("### Send Test Request")
    with gr.Row():
        prompt = gr.Textbox(value="Hello", label="Prompt")
        max_tokens = gr.Slider(10, 100, value=20, label="Max Tokens")
    btn = gr.Button("Send")
    output = gr.Textbox(label="Response", lines=3)

    btn.click(send_request, [prompt, max_tokens], output)

if __name__ == "__main__":
    app.launch()
