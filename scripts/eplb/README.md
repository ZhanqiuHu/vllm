# EPLB (Expert-Parallel Load Balancing) Scripts

Scripts for testing and monitoring EPLB in vLLM MoE models.

## Background

EPLB dynamically replicates heavily-used experts across EP ranks to balance load.
See [RFC #30696](https://github.com/vllm-project/vllm/issues/30696) for proposed
Prometheus metrics.

## Directory Structure

```
eplb/
├── serve_moe_eplb.sh      # Start MoE server with EPLB enabled
├── testing/
│   └── load_test.sh       # Generate load to trigger EPLB rebalancing
└── monitoring/
    └── check_metrics.sh   # Check Prometheus metrics
```

## Quick Start

```bash
# 1. Start server with EPLB (uses DeepSeek-V2-Lite by default)
./serve_moe_eplb.sh

# 2. In another terminal, run load test
./testing/load_test.sh 200

# 3. Check metrics
./monitoring/check_metrics.sh
```

## EPLB Configuration

Key `--eplb-config` options:
- `log_balancedness`: Enable stdout logging of balancedness stats
- `log_balancedness_interval`: Log every N steps (default: 100)
- `step_interval`: Rebalance every N steps (default: 3000)
- `num_redundant_experts`: Extra expert replicas for load balancing
- `use_async`: Non-blocking EPLB transfers

## Expected EPLB Log Output

When running with `log_balancedness: true`, server logs show:
```
EPLB step: 100: avg_tokens=512.00, max_tokens=1024, balancedness=0.5000
```

Where:
- `avg_tokens`: Average tokens per EP rank
- `max_tokens`: Maximum tokens across EP ranks  
- `balancedness`: Ratio of avg/max (1.0 = perfectly balanced)

## Models for Testing

| Model | Total Params | Experts | Top-K | Notes |
|-------|-------------|---------|-------|-------|
| DeepSeek-V2-Lite | 16B | 64 | 6 | Quick testing |
| Mixtral-8x7B | 47B | 8 | 2 | Smaller MoE |
| DeepSeek-V2 | 236B | 160 | 6 | Full scale |
