# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for TP mapping and transfer plan utilities."""

from __future__ import annotations

from types import SimpleNamespace

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    TPMapping,
    compute_tp_mapping,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec


def _compute_mapping(
    tp_rank: int = 0,
    tp_size: int = 1,
    remote_tp_size: int = 1,
    is_mla: bool = False,
    num_kv_heads: int = 8,
    group_spec_types: tuple[type, ...] = (FullAttentionSpec,),
) -> TPMapping:
    transfer_topology = SimpleNamespace(
        tp_rank=tp_rank,
        tp_size=tp_size,
        is_mla=is_mla,
        total_num_kv_heads=num_kv_heads,
    )
    return compute_tp_mapping(
        transfer_topology=transfer_topology,
        remote_tp_size=remote_tp_size,
        group_spec_types=group_spec_types,
    )


class TestTPMappingStructure:
    def test_source_ranks_homogeneous(self):
        mapping = _compute_mapping(tp_size=2, tp_rank=1, remote_tp_size=2)
        assert mapping.all_source_ranks == (1,)

    def test_source_ranks_d_gt_p(self):
        mapping = _compute_mapping(tp_size=4, tp_rank=2, remote_tp_size=2)
        assert mapping.all_source_ranks == (1,)

    def test_source_ranks_p_gt_d(self):
        mapping = _compute_mapping(tp_size=1, tp_rank=0, remote_tp_size=2)
        assert mapping.all_source_ranks == (0, 1)
