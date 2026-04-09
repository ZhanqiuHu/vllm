# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from tests.evals.gsm8k.gsm8k_eval import evaluate_gsm8k_offline
from vllm import LLM
from vllm.config import SpeculativeConfig
from vllm.distributed import cleanup_dist_env_and_memory

MODEL_PATH = "nm-testing/dflash-qwen3-8b-speculators"

EXPECTED_GSM8K_ACCURACY = 0.885
ACCURACY_RTOL = 0.03
EXPECTED_ACCEPTANCE_LEN = 1.84
ACCEPTANCE_LEN_RTOL = 0.15


def compute_acceptance_len(metrics) -> float:
    name2metric = {m.name: m for m in metrics}
    n_drafts = name2metric["vllm:spec_decode_num_drafts"].value
    n_accepted = name2metric["vllm:spec_decode_num_accepted_tokens"].value
    if n_drafts == 0:
        return 1.0
    return 1 + (n_accepted / n_drafts)


def test_dflash_speculators_model(vllm_runner, example_prompts, monkeypatch):
    """
    Test DFlash speculators model properly initializes speculative decoding.

    Verifies:
    1. Speculative config is automatically initialized from speculators config
    2. Method is detected as 'dflash'
    3. The draft model path is correctly set
    4. Speculative tokens count is valid (num_speculative_tokens=8)
    5. Text generation works with speculative decoding enabled
    """
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    with vllm_runner(
        MODEL_PATH, dtype=torch.bfloat16, enforce_eager=True
    ) as vllm_model:
        vllm_config = vllm_model.llm.llm_engine.vllm_config

        assert isinstance(vllm_config.speculative_config, SpeculativeConfig), (
            "Speculative config should be initialized for speculators model"
        )

        spec_config = vllm_config.speculative_config
        assert spec_config.method == "dflash", (
            f"Expected method='dflash', got '{spec_config.method}'"
        )
        assert spec_config.num_speculative_tokens > 0, (
            f"Expected positive speculative tokens, "
            f"got {spec_config.num_speculative_tokens}"
        )
        assert spec_config.model == MODEL_PATH, (
            f"Draft model should be {MODEL_PATH}, got {spec_config.model}"
        )

        vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens=20)
        assert vllm_outputs, f"No outputs generated for speculators model {MODEL_PATH}"


def test_dflash_speculators_correctness():
    """
    E2E correctness test for DFlash via the speculators auto-detect path.

    Evaluates GSM8k accuracy to ensure the speculators-format model produces
    correct outputs, and checks that acceptance length does not collapse under
    batched inference (lm-eval style).

    Observed per-position acceptance rates on magpie (200 prompts):
        pos 0: 0.478, pos 1: 0.181, pos 2: 0.069, pos 3: 0.023,
        pos 4: 0.007, pos 5: 0.002, pos 6: 0.001, pos 7: 0.000
    Observed mean AL: 1.77 (batch-size-1, magpie dataset)
    """
    spec_llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        max_model_len=4096,
        max_num_seqs=128,
        gpu_memory_utilization=0.85,
        enforce_eager=False,
        disable_log_stats=False,
    )

    results = evaluate_gsm8k_offline(spec_llm)
    accuracy = results["accuracy"]
    accuracy_threshold = EXPECTED_GSM8K_ACCURACY * (1 - ACCURACY_RTOL)
    assert accuracy >= accuracy_threshold, (
        f"Expected GSM8K accuracy >= {accuracy_threshold:.3f}, got {accuracy:.3f}"
    )

    current_metrics = spec_llm.get_metrics()
    acceptance_len = compute_acceptance_len(current_metrics)

    al_threshold = EXPECTED_ACCEPTANCE_LEN * (1 - ACCEPTANCE_LEN_RTOL)
    assert acceptance_len >= al_threshold, (
        f"DFlash speculators acceptance length too low: "
        f"{acceptance_len:.2f} < {al_threshold:.2f}"
    )

    del spec_llm
    torch.accelerator.empty_cache()
    cleanup_dist_env_and_memory()
