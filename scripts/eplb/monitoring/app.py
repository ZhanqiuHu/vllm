#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Simple real-time metrics app."""

import requests
from flask import Flask, jsonify, render_template_string

app = Flask(__name__)
PROM = "http://localhost:9090"

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>vLLM Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: sans-serif; margin: 20px; }
        .container { display: flex; gap: 40px; }
        .left { flex: 1; }
        .right { flex: 1; }
        .chart-container { width: 100%; height: 300px; }
        textarea { width: 100%; height: 60px; }
        input, button { margin: 5px 0; padding: 8px; }
        #response { background: #f5f5f5; padding: 10px; min-height: 100px; white-space: pre-wrap; }
    </style>
</head>
<body>
    <h1>vLLM Dashboard</h1>
    <div class="container">
        <div class="left">
            <h3>Send Request</h3>
            <textarea id="prompt" placeholder="Enter prompt...">Hello, tell me a joke</textarea><br>
            <label>Max Tokens: <input type="number" id="max_tokens" value="50" min="1" max="500"></label>
            <button onclick="sendRequest()">Send</button>
            <h4>Response:</h4>
            <div id="response">-</div>
        </div>
        <div class="right">
            <h3>Real-time Metrics</h3>
            <div class="chart-container"><canvas id="chart"></canvas></div>
        </div>
    </div>
    
    <script>
    async function sendRequest() {
        const prompt = document.getElementById('prompt').value;
        const max_tokens = document.getElementById('max_tokens').value;
        document.getElementById('response').innerText = 'Loading...';
        const res = await fetch('/api/complete', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({prompt, max_tokens: parseInt(max_tokens)})
        });
        const data = await res.json();
        document.getElementById('response').innerText = data.text || data.error;
    }
    
    const ctx = document.getElementById('chart').getContext('2d');
    const chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Tokens/sec',
                data: [],
                borderColor: 'green',
                fill: false
            }, {
                label: 'Requests Running',
                data: [],
                borderColor: 'blue',
                fill: false
            }]
        },
        options: {
            scales: {
                x: { title: { display: true, text: 'Time' }},
                y: { title: { display: true, text: 'Value' }, beginAtZero: true }
            },
            animation: false
        }
    });
    
    async function update() {
        const res = await fetch('/api/data');
        const data = await res.json();
        
        chart.data.labels = data.timestamps;
        chart.data.datasets[0].data = data.tokens_rate;
        chart.data.datasets[1].data = data.running;
        chart.update();
    }
    
    update();
    setInterval(update, 1000);
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/complete", methods=["POST"])
def complete():
    from flask import request

    data = request.json
    try:
        r = requests.post(
            "http://localhost:8000/v1/completions",
            json={
                "model": "deepseek-ai/DeepSeek-V2-Lite",
                "prompt": data.get("prompt", ""),
                "max_tokens": data.get("max_tokens", 50),
            },
            timeout=60,
        )
        return jsonify({"text": r.json().get("choices", [{}])[0].get("text", "")})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/data")
def data():
    import time

    now = int(time.time())
    start = now - 300  # 5 minutes ago

    def query_range(q):
        r = requests.get(
            f"{PROM}/api/v1/query_range",
            params={"query": q, "start": start, "end": now, "step": 5},
            timeout=5,
        )
        d = r.json()
        if d["status"] == "success" and d["data"]["result"]:
            return d["data"]["result"][0]["values"]
        return []

    tokens = query_range("rate(vllm:generation_tokens_total[30s])")
    running = query_range("vllm:num_requests_running")

    if not tokens:
        return jsonify({"timestamps": [], "tokens_rate": [], "running": []})

    from datetime import datetime

    def fmt(ts):
        return datetime.fromtimestamp(ts).strftime("%H:%M:%S")

    return jsonify(
        {
            "timestamps": [fmt(t[0]) for t in tokens],
            "tokens_rate": [float(t[1]) for t in tokens],
            "running": [float(r[1]) for r in running] if running else [0] * len(tokens),
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
