# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Integration edge-case tests for MultiConnector (NixlConnector + OffloadingConnector).

Tests cover:
  - Output correctness across block-size boundaries (proxy vs direct prefill).
  - Prometheus metrics validation (local_cache_hit, external_kv_transfer,
     local_compute) for cold/warm/partial cache scenarios.

Requires running servers started by run_multi_connector_edge_case_test.sh.
"""

import os
import time
import urllib.request

import openai
import regex as re

# ── Server configuration from environment ─────────────────────────────────

PREFILL_HOST = os.getenv("PREFILL_HOST", "localhost")
PREFILL_PORT = os.getenv("PREFILL_PORT")
DECODE_HOST = os.getenv("DECODE_HOST", "localhost")
DECODE_PORT = os.getenv("DECODE_PORT")
PROXY_HOST = os.getenv("PROXY_HOST", "localhost")
PROXY_PORT = os.getenv("PROXY_PORT")
BLOCK_SIZE = int(os.getenv("BLOCK_SIZE", "128"))

if not all([PREFILL_PORT, DECODE_PORT, PROXY_PORT]):
    raise ValueError("PREFILL_PORT, DECODE_PORT, and PROXY_PORT must be set.")

# ── OpenAI clients ────────────────────────────────────────────────────────

decode_client = openai.OpenAI(
    api_key="EMPTY",
    base_url=f"http://{DECODE_HOST}:{DECODE_PORT}/v1",
)
prefill_client = openai.OpenAI(
    api_key="EMPTY",
    base_url=f"http://{PREFILL_HOST}:{PREFILL_PORT}/v1",
)
proxy_client = openai.OpenAI(
    api_key="EMPTY",
    base_url=f"http://{PROXY_HOST}:{PROXY_PORT}/v1",
)

_MODEL = None


def _get_model() -> str:
    global _MODEL
    if _MODEL is None:
        models = decode_client.models.list()
        _MODEL = models.data[0].id
    return _MODEL


def _complete(client: openai.OpenAI, prompt: str, max_tokens: int = 20):
    """Send a completion request and return (text, prompt_tokens)."""
    resp = client.completions.create(
        model=_get_model(),
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0,
    )
    return resp.choices[0].text, resp.usage.prompt_tokens


# ── Prometheus metrics helpers ────────────────────────────────────────────

_METRIC_RE = re.compile(
    r'vllm:prompt_tokens_by_source_total\{.*?source="([^"]+)".*?\}\s+'
    r"([\d.eE+\-]+)"
)


def _fetch_decode_metrics() -> dict[str, float]:
    """Scrape prompt_tokens_by_source counters from the decode server."""
    body = (
        urllib.request.urlopen(f"http://{DECODE_HOST}:{DECODE_PORT}/metrics")
        .read()
        .decode()
    )
    result = {
        "local_compute": 0.0,
        "local_cache_hit": 0.0,
        "external_kv_transfer": 0.0,
    }
    for m in _METRIC_RE.finditer(body):
        source, val = m.group(1), float(m.group(2))
        if source in result:
            result[source] = val
    return result


def _metrics_delta(before: dict, after: dict) -> dict[str, float]:
    return {k: after.get(k, 0) - before.get(k, 0) for k in before}


# ── Prompts (unique per test to avoid cross-test cache interference) ──────

SHORT_PROMPT = "Red Hat is "

MEDIUM_PROMPT = (
    "Red Hat is the best company in the world to work for because it works "
    "on open source software, which means that all the contributions are "
    "delivered to the community. As a result,"
)


def _make_prompt(n_tokens: int) -> str:
    """Build a prompt of ~n_tokens tokens (1 word ~ 1 token)."""
    return "word " * n_tokens


BLOCK_BOUNDARY_PROMPT = _make_prompt(BLOCK_SIZE)
ABOVE_BOUNDARY_PROMPT = _make_prompt(BLOCK_SIZE + 2)
MULTI_BLOCK_PROMPT = _make_prompt(BLOCK_SIZE * 4)

FULL_CACHE_HIT_PROMPT = (  # noqa: E501
    "The history of computing begins with Charles Babbage who designed the "
    "Analytical Engine in the 1830s which is considered the first general "
    "purpose computer design in history. Ada Lovelace is widely regarded as "
    "the first computer programmer for her work on the Analytical Engine. "
    "The modern era of computing began with Alan Turing who formalized the "
    "concept of computation with his Turing machine in 1936. During World "
    "War Two Turing worked at Bletchley Park to break the Enigma cipher. "
    "After the war the first electronic computers were built including ENIAC "
    "at the University of Pennsylvania and Colossus at Bletchley Park. "
    "These early machines filled entire rooms and used vacuum tubes for logic. "
    "The invention of the transistor at Bell Labs in 1947 revolutionized "
    "computing by making smaller and more reliable machines possible. "
    "The integrated circuit followed in the late 1950s combining multiple "
    "transistors on a single chip. This led to the microprocessor in the 1970s "
    "and eventually to the personal computer revolution of the 1980s."
)

PARTIAL_CACHE_PREFIX = (  # noqa: E501
    "Machine learning has transformed the field of artificial intelligence "
    "by enabling computers to learn patterns from data without being "
    "explicitly programmed for every task. The field has evolved dramatically "
    "since its inception in the 1950s when Arthur Samuel coined the term while "
    "working at IBM. Early approaches focused on symbolic reasoning and expert "
    "systems that encoded human knowledge as rules. The statistical revolution "
    "of the 1990s shifted the paradigm toward data driven methods. Support "
    "vector machines and random forests became popular for classification tasks. "
    "The breakthrough of deep learning in 2012 with AlexNet winning ImageNet "
    "changed everything. Neural networks with many layers could automatically "
    "learn hierarchical feature representations from raw data."
)
PARTIAL_CACHE_EXTENDED = PARTIAL_CACHE_PREFIX + (
    " Transformers have become the dominant architecture for natural language "
    "processing tasks including translation, summarization, and generation. "
    "The attention mechanism allows models to weigh the importance of different "
    "parts of the input sequence. Large language models like GPT and BERT "
    "demonstrated that pre-training on massive text corpora followed by fine "
    "tuning on specific tasks could achieve state of the art results across "
    "a wide range of benchmarks. Scaling laws suggest that larger models "
    "trained on more data continue to improve in capability."
)

# ═══════════════════════════════════════════════════════════════════════════
# Output correctness across block-size boundaries
#
# Each test verifies proxy output matches prefill_direct at temperature=0,
# and that NIXL was actually used (not silently falling back to local compute).
# ═══════════════════════════════════════════════════════════════════════════


def test_short_prompt_correctness():
    """Short prompt (< block_size): output matches prefill, NIXL used."""
    m0 = _fetch_decode_metrics()
    proxy_text, _ = _complete(proxy_client, SHORT_PROMPT)
    time.sleep(1)
    m1 = _fetch_decode_metrics()
    d = _metrics_delta(m0, m1)

    prefill_text, _ = _complete(prefill_client, SHORT_PROMPT)
    print(f"SHORT PROMPT: {proxy_text=}")
    assert proxy_text == prefill_text
    assert d["external_kv_transfer"] > 0, (
        "NIXL transfer did not occur — decode may have silently fallen back "
        "to local compute"
    )


def test_block_boundary_correctness():
    """Exactly block_size tokens: output matches prefill, NIXL used."""
    m0 = _fetch_decode_metrics()
    proxy_text, pt = _complete(proxy_client, BLOCK_BOUNDARY_PROMPT)
    time.sleep(1)
    m1 = _fetch_decode_metrics()
    d = _metrics_delta(m0, m1)

    prefill_text, _ = _complete(prefill_client, BLOCK_BOUNDARY_PROMPT)
    print(f"BLOCK BOUNDARY: {pt} prompt tokens")
    assert proxy_text == prefill_text
    assert d["external_kv_transfer"] > 0, (
        "NIXL transfer did not occur — decode may have silently fallen back "
        "to local compute"
    )


def test_above_block_boundary_correctness():
    """Just above block_size (partial second block): output matches prefill."""
    m0 = _fetch_decode_metrics()
    proxy_text, pt = _complete(proxy_client, ABOVE_BOUNDARY_PROMPT)
    time.sleep(1)
    m1 = _fetch_decode_metrics()
    d = _metrics_delta(m0, m1)

    prefill_text, _ = _complete(prefill_client, ABOVE_BOUNDARY_PROMPT)
    print(f"ABOVE BOUNDARY: {pt} prompt tokens")
    assert proxy_text == prefill_text
    assert d["external_kv_transfer"] > 0, (
        "NIXL transfer did not occur — decode may have silently fallen back "
        "to local compute"
    )


def test_multi_block_correctness():
    """Multi-block prompt (~4x block_size): output matches prefill."""
    m0 = _fetch_decode_metrics()
    proxy_text, pt = _complete(proxy_client, MULTI_BLOCK_PROMPT)
    time.sleep(1)
    m1 = _fetch_decode_metrics()
    d = _metrics_delta(m0, m1)

    prefill_text, _ = _complete(prefill_client, MULTI_BLOCK_PROMPT)
    print(f"MULTI BLOCK: {pt} prompt tokens")
    assert proxy_text == prefill_text
    assert d["external_kv_transfer"] > 0, (
        "NIXL transfer did not occur — decode may have silently fallen back "
        "to local compute"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Decode-side KV source validation via Prometheus metrics
#
# These tests scrape vllm:prompt_tokens_by_source_total from the decode
# server to verify which code path was exercised for each request.
# ═══════════════════════════════════════════════════════════════════════════


def test_cold_decode_no_cache_hit_metrics():
    """Cold decode: external_kv_transfer==P, local_cache_hit==0."""
    m0 = _fetch_decode_metrics()
    proxy_text, P = _complete(proxy_client, MEDIUM_PROMPT)
    time.sleep(1)
    m1 = _fetch_decode_metrics()
    d = _metrics_delta(m0, m1)

    prefill_text, _ = _complete(prefill_client, MEDIUM_PROMPT)
    print(f"COLD DECODE: {P} prompt tokens, metrics delta: {d}")
    assert proxy_text == prefill_text, (
        f"output mismatch: {proxy_text=!r}, {prefill_text=!r}"
    )
    assert d["external_kv_transfer"] == P, (
        f"expected external_kv_transfer={P}, got {d['external_kv_transfer']}"
    )
    assert d["local_compute"] == 1, (
        f"expected local_compute=1, got {d['local_compute']}"
    )
    assert d["local_cache_hit"] == 0, (
        f"expected local_cache_hit=0, got {d['local_cache_hit']}"
    )


def test_full_decode_gpu_cache_hit_metrics():
    """Prime decode, resend via proxy: local_cache_hit==cached blocks."""
    decode_text, _ = _complete(decode_client, FULL_CACHE_HIT_PROMPT)

    m0 = _fetch_decode_metrics()
    proxy_text, P = _complete(proxy_client, FULL_CACHE_HIT_PROMPT)
    time.sleep(1)
    m1 = _fetch_decode_metrics()
    d = _metrics_delta(m0, m1)

    cached = (P // BLOCK_SIZE) * BLOCK_SIZE
    expected_nixl = P - cached

    print(f"FULL CACHE HIT: {P} tokens, cached={cached}, nixl={expected_nixl}")
    print(f"  metrics delta: {d}")
    assert len(proxy_text) > 0, "proxy returned empty response"
    assert d["local_cache_hit"] == cached, (
        f"expected local_cache_hit={cached}, got {d['local_cache_hit']}"
    )
    assert d["external_kv_transfer"] == expected_nixl, (
        f"expected external_kv_transfer={expected_nixl}, "
        f"got {d['external_kv_transfer']}"
    )
    assert d["local_compute"] == 1, (
        f"expected local_compute=1 (recomputed last token), got {d['local_compute']}"
    )


def test_partial_decode_gpu_cache_hit_metrics():
    """Prime with prefix, extend via proxy: partial local_cache_hit."""
    _, prefix_tokens = _complete(decode_client, PARTIAL_CACHE_PREFIX)
    cached = (prefix_tokens // BLOCK_SIZE) * BLOCK_SIZE
    assert cached >= BLOCK_SIZE, (
        f"PARTIAL_CACHE_PREFIX too short ({prefix_tokens} tokens) for partial "
        f"cache hit test with block_size={BLOCK_SIZE}"
    )

    m0 = _fetch_decode_metrics()
    proxy_text, P = _complete(proxy_client, PARTIAL_CACHE_EXTENDED)
    time.sleep(1)
    m1 = _fetch_decode_metrics()
    d = _metrics_delta(m0, m1)

    prefill_text, _ = _complete(prefill_client, PARTIAL_CACHE_EXTENDED)
    expected_nixl = P - cached

    print(f"PARTIAL CACHE HIT: {P} tokens, cached={cached}, nixl={expected_nixl}")
    print(f"  metrics delta: {d}")
    assert len(proxy_text) > 0, "proxy returned empty response"
    assert d["external_kv_transfer"] == expected_nixl, (
        f"expected external_kv_transfer={expected_nixl}, "
        f"got {d['external_kv_transfer']}"
    )
    assert d["local_cache_hit"] == cached, (
        f"expected local_cache_hit={cached}, got {d['local_cache_hit']}"
    )
    assert d["local_compute"] == 1, (
        f"expected local_compute=1 (recomputed last token), got {d['local_compute']}"
    )


def test_decode_direct_all_local_compute():
    """Direct decode (no proxy): local_compute==P, no transfers."""
    prompt = "The speed of light is approximately"
    m0 = _fetch_decode_metrics()
    text, P = _complete(decode_client, prompt)
    time.sleep(1)
    m1 = _fetch_decode_metrics()
    d = _metrics_delta(m0, m1)

    print(f"DIRECT DECODE: {text!r} ({P} tokens), metrics delta: {d}")
    assert len(text.strip()) > 0, "empty output from direct decode"
    assert d["local_compute"] == P, (
        f"expected local_compute={P}, got {d['local_compute']}"
    )
    assert d["external_kv_transfer"] == 0, (
        f"expected external_kv_transfer=0, got {d['external_kv_transfer']}"
    )
