# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for TP mapping and transfer plan utilities.

These tests verify that TP mapping produces correct outputs
(source ranks, split handles, desc IDs).
No GPU or NIXL required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
    NixlConnectorWorker,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MambaSpec,
    MLAAttentionSpec,
    ShardRange,
    SlidingWindowMLASpec,
    TPTransferSlice,
)

# ======================================================================
# Test fixtures / helpers
# ======================================================================


def _make_fa_spec(num_kv_heads: int = 4):
    return FullAttentionSpec(
        block_size=16,
        num_kv_heads=num_kv_heads,
        head_size=128,
        head_size_v=128,
        dtype=torch.float16,
    )


def _get_slices(
    tp_rank: int = 0,
    tp_size: int = 1,
    remote_tp_size: int = 1,
    total_num_kv_heads: int = 8,
    spec=None,
) -> dict[int, TPTransferSlice]:
    """Call get_tp_transfer_slices on the given spec (or a default FA spec)."""
    if spec is None:
        num_kv_heads = max(1, total_num_kv_heads // tp_size)
        spec = _make_fa_spec(num_kv_heads)
    return spec.get_tp_transfer_slices(
        tp_rank, tp_size, remote_tp_size, total_num_kv_heads
    )


def _source_ranks_from_slices(
    *group_slices: dict[int, TPTransferSlice],
) -> tuple[int, ...]:
    """Derive deduplicated sorted source ranks from multiple group slices."""
    return tuple(sorted({r for slices in group_slices for r in slices}))


# ======================================================================
# TP mapping structure tests
# ======================================================================


class TestTPMappingStructure:
    def test_source_ranks_homogeneous(self):
        slices = _get_slices(tp_size=2, tp_rank=1, remote_tp_size=2)
        assert _source_ranks_from_slices(slices) == (1,)

    def test_source_ranks_d_gt_p(self):
        slices = _get_slices(tp_size=4, tp_rank=2, remote_tp_size=2)
        assert _source_ranks_from_slices(slices) == (1,)

    def test_source_ranks_p_gt_d(self):
        slices = _get_slices(tp_size=1, tp_rank=0, remote_tp_size=2)
        assert _source_ranks_from_slices(slices) == (0, 1)

    def test_per_group_slices(self):
        slices = _get_slices(tp_size=2, tp_rank=0, remote_tp_size=4)
        assert len(slices) == 2
        assert 0 in slices
        assert 1 in slices

    def test_has_rank_in_group(self):
        slices = _get_slices(tp_size=1, tp_rank=0, remote_tp_size=2)
        assert 0 in slices
        assert 1 in slices
        assert 2 not in slices

    def test_gqa_dedup_load_balanced(self):
        """With total_heads=2, remote_tp=4: picks aligned remote ranks."""
        slices_r0 = _get_slices(
            tp_size=2, tp_rank=0, remote_tp_size=4, total_num_kv_heads=2
        )
        slices_r1 = _get_slices(
            tp_size=2, tp_rank=1, remote_tp_size=4, total_num_kv_heads=2
        )
        assert 0 in slices_r0
        assert 2 in slices_r1


# ======================================================================
# Split handle tests
# ======================================================================


def _make_mock_worker_for_splits(
    group_specs: list,
    tp_mappings: tuple,
    source_ranks: tuple[int, ...],
    engine_id: str = "remote_0",
):
    """Build a mock NixlConnectorWorker with the fields _build_local_splits needs."""
    worker = object.__new__(NixlConnectorWorker)
    kv_cache_groups = []
    for spec in group_specs:
        group = MagicMock()
        group.kv_cache_spec = spec
        kv_cache_groups.append(group)
    kv_cache_config = MagicMock()
    kv_cache_config.kv_cache_groups = kv_cache_groups
    worker.kv_cache_config = kv_cache_config
    worker.tp_mappings = {engine_id: tp_mappings}
    worker.source_ranks = {engine_id: source_ranks}
    worker.transfer_topo = MagicMock()
    return worker


class TestBuildSrcSplitHandles:
    @pytest.mark.parametrize("remote_tp_size", [2, 4])
    def test_split_shape(self, remote_tp_size):
        """Each split has correct number of descs with correct chunk size."""
        tp_rank = 0
        tp_size = 1
        total_num_kv_heads = 8
        engine_id = "remote_0"

        fa_spec = _make_fa_spec(num_kv_heads=total_num_kv_heads // tp_size)
        fa_slices = fa_spec.get_tp_transfer_slices(
            tp_rank, tp_size, remote_tp_size, total_num_kv_heads
        )
        source_ranks = _source_ranks_from_slices(fa_slices)

        worker = _make_mock_worker_for_splits(
            group_specs=[fa_spec],
            tp_mappings=(fa_slices,),
            source_ranks=source_ranks,
            engine_id=engine_id,
        )
        src_blocks_data = [(0x2000 + i * 1024, 1024, 0) for i in range(8)]
        num_fa_descs = len(src_blocks_data)
        splits = list(
            worker._build_local_splits(engine_id, src_blocks_data, num_fa_descs)
        )

        assert len(splits) == remote_tp_size
        for handle in splits:
            assert len(handle) == len(src_blocks_data)
            for _, length, _ in handle:
                assert length == 1024 // remote_tp_size

    @pytest.mark.parametrize(
        "remote_tp_size,total_num_kv_heads",
        [(2, 4), (2, 8), (4, 8)],
    )
    def test_fa_offsets_p_gt_d(self, remote_tp_size, total_num_kv_heads):
        """Verify concrete FA offsets for multi-head P>D (the previously buggy path).

        With local_tp=1, the full local block covers all heads. Each remote
        rank's slice should land at the correct byte offset proportional to
        its position in the local tensor.
        """
        tp_rank = 0
        tp_size = 1
        engine_id = "remote_0"
        local_block_len = 1024

        fa_spec = _make_fa_spec(num_kv_heads=total_num_kv_heads // tp_size)
        fa_slices = fa_spec.get_tp_transfer_slices(
            tp_rank, tp_size, remote_tp_size, total_num_kv_heads
        )
        source_ranks = _source_ranks_from_slices(fa_slices)

        worker = _make_mock_worker_for_splits(
            group_specs=[fa_spec],
            tp_mappings=(fa_slices,),
            source_ranks=source_ranks,
            engine_id=engine_id,
        )
        base_addr = 0x4000
        src_blocks_data = [(base_addr, local_block_len, 0)]
        splits = list(worker._build_local_splits(engine_id, src_blocks_data, 1))

        assert len(splits) == remote_tp_size
        chunk = local_block_len // remote_tp_size
        for idx, (rank, sl) in enumerate(sorted(fa_slices.items())):
            expected_offset = (
                sl.local_write_offset * local_block_len // len(sl.local_shard)
            )
            # Offsets should tile the local block without overlap
            assert expected_offset == idx * chunk
            addr, length, dev = splits[idx][0]
            assert addr == base_addr + expected_offset
            assert length == chunk
            assert dev == 0


class TestMambaPlanSplitHandles:
    """Verify split handles for Mamba with FA/SSM distinction."""

    def test_fa_and_ssm_different_split_factors(self):
        """Section 0 split by num_attn_reads, section 1 by abs_tp."""
        engine_id = "remote_0"
        # total_kv_heads=1 < remote_tp=2 triggers GQA dedup:
        # only remote rank 0 holds unique FA data.
        total_num_kv_heads = 1

        fa_spec = _make_fa_spec(num_kv_heads=1)
        mamba_spec = MagicMock(spec=MambaSpec)

        # local_tp=1, remote_tp=2
        # FA: 1 unique slice (reads from remote 0, GQA dedup skips rank 1)
        # Mamba: 2 slices (reads from remote 0 and 1)
        fa_slices = fa_spec.get_tp_transfer_slices(0, 1, 2, total_num_kv_heads)

        shard_mamba = ShardRange(0, 1, 1)
        ssm_slices = {
            0: TPTransferSlice(
                source_rank=0,
                source_shard=shard_mamba,
                local_shard=shard_mamba,
                transfer_range=shard_mamba,
            ),
            1: TPTransferSlice(
                source_rank=1,
                source_shard=shard_mamba,
                local_shard=shard_mamba,
                transfer_range=shard_mamba,
            ),
        }
        source_ranks = _source_ranks_from_slices(fa_slices, ssm_slices)

        worker = _make_mock_worker_for_splits(
            group_specs=[fa_spec, mamba_spec],
            tp_mappings=(fa_slices, ssm_slices),
            source_ranks=source_ranks,
            engine_id=engine_id,
        )

        # 2 FA descs + 1 SSM desc
        src_blocks_data = [
            (1000, 200, 0),  # FA desc 0
            (2000, 200, 0),  # FA desc 1
            (3000, 400, 0),  # SSM desc 0
        ]

        splits = list(worker._build_local_splits(engine_id, src_blocks_data, 2))

        assert len(splits) == 2  # 2 source ranks

        # Rank 0 is in fa_slices -> uses local_write_offset for FA offset
        fa_chunk = 200 // len(fa_slices)
        ssm_chunk = 400 // len(ssm_slices)

        # Rank 0 (source_idx=0):
        # FA: chunk=200//1=200 (only 1 FA slice)
        # offset = local_write_offset * local_block_len // len(local_shard)
        # SSM: chunk=400//2=200, offset = source_idx(0) * 200
        sl = fa_slices[0]
        fa_offset_r0 = sl.local_write_offset * 200 // len(sl.local_shard)
        assert splits[0][0] == (1000 + fa_offset_r0, fa_chunk, 0)
        assert splits[0][1] == (2000 + fa_offset_r0, fa_chunk, 0)
        assert splits[0][2] == (3000 + 0 * ssm_chunk, ssm_chunk, 0)

        # Rank 1 (source_idx=1):
        # FA: rank 1 NOT in fa_slices -> GQA-deduped placeholder (addr, chunk, dev)
        # SSM: chunk=400//2=200, offset = source_idx(1) * 200
        assert splits[1][0] == (1000, fa_chunk, 0)
        assert splits[1][1] == (2000, fa_chunk, 0)
        assert splits[1][2] == (3000 + 1 * ssm_chunk, ssm_chunk, 0)


# ======================================================================
# slice_for_tp_transfer tests
# ======================================================================

NUM_BLOCKS = 4
BLOCK_SIZE = 16
HEAD_SIZE = 128


def _make_meta_tensor(num_kv_heads: int) -> torch.Tensor:
    """Create a meta tensor with shape [B, H, N, C] for planning."""
    return torch.empty(
        NUM_BLOCKS,
        num_kv_heads,
        BLOCK_SIZE,
        2 * HEAD_SIZE,
        device="meta",
    )


class TestSliceForTPTransferGQA:
    """Test AttentionSpec.slice_for_tp_transfer for GQA head slicing."""

    def test_homogeneous_tp(self):
        """Same TP on both sides: each rank returns its own full shard."""
        spec = _make_fa_spec(num_kv_heads=4)
        tensor = _make_meta_tensor(num_kv_heads=4)
        # Local and remote spec match
        slices = spec.slice_for_tp_transfer(tensor, 2, 0, spec, 2, 0)
        assert len(slices) == 1
        assert slices[0].shape == (NUM_BLOCKS, 4, BLOCK_SIZE, 2 * HEAD_SIZE)

        slices_other = spec.slice_for_tp_transfer(tensor, 2, 0, spec, 2, 1)
        assert len(slices_other) == 0

    def test_d_gt_p(self):
        """D_TP=4, P_TP=2: src rank has wider shard, dst reads a sub-range."""
        src_spec = _make_fa_spec(num_kv_heads=4)
        dst_spec = _make_fa_spec(num_kv_heads=2)
        tensor = _make_meta_tensor(num_kv_heads=4)

        slices = src_spec.slice_for_tp_transfer(tensor, 2, 1, dst_spec, 4, 2)
        assert len(slices) == 1
        assert slices[0].shape[1] == 2

        slices_miss = src_spec.slice_for_tp_transfer(tensor, 2, 0, dst_spec, 4, 2)
        assert len(slices_miss) == 0

    def test_p_gt_d(self):
        """P_TP=4, D_TP=2: multiple src ranks contribute to one dst rank."""
        src_spec = _make_fa_spec(num_kv_heads=2)
        dst_spec = _make_fa_spec(num_kv_heads=4)
        tensor = _make_meta_tensor(num_kv_heads=2)

        slices_r0 = src_spec.slice_for_tp_transfer(tensor, 4, 0, dst_spec, 2, 0)
        slices_r1 = src_spec.slice_for_tp_transfer(tensor, 4, 1, dst_spec, 2, 0)
        assert len(slices_r0) == 1
        assert len(slices_r1) == 1
        assert slices_r0[0].shape[1] == 2
        assert slices_r1[0].shape[1] == 2

        slices_miss = src_spec.slice_for_tp_transfer(tensor, 4, 2, dst_spec, 2, 0)
        assert len(slices_miss) == 0

    def test_gqa_dedup_replicated_heads(self):
        """total_heads=4, P_TP=4, D_TP=2, kv_heads=1 per rank.

        src_rank 0 has head [0,1), src_rank 1 has [1,2), etc.
        dst_rank 0 (kv_heads=2) needs heads [0,2) -> reads from src 0 and 1.
        src_rank 2,3 are filtered by src_tp_rank >= dst_tp_size.
        """
        src_spec = _make_fa_spec(num_kv_heads=1)
        dst_spec = _make_fa_spec(num_kv_heads=2)
        tensor = _make_meta_tensor(num_kv_heads=1)

        contributing = []
        for r in range(4):
            slices = src_spec.slice_for_tp_transfer(tensor, 4, r, dst_spec, 2, 0)
            if slices:
                contributing.append(r)

        assert contributing == [0, 1]

    def test_single_head_full_replication(self):
        """total_heads=1, all TP ranks hold the same head."""
        src_spec = _make_fa_spec(num_kv_heads=1)
        dst_spec = _make_fa_spec(num_kv_heads=1)
        tensor = _make_meta_tensor(num_kv_heads=1)

        slices = src_spec.slice_for_tp_transfer(tensor, 4, 0, dst_spec, 4, 0)
        assert len(slices) == 1
        assert slices[0].shape[1] == 1

    @pytest.mark.parametrize(
        "src_tp,dst_tp,total_heads",
        [(1, 2, 8), (2, 4, 8), (1, 4, 4), (4, 1, 8), (2, 1, 8)],
    )
    def test_equivalence_with_get_tp_transfer_slices(self, src_tp, dst_tp, total_heads):
        """Verify slice_for_tp_transfer selects the same source ranks
        as get_tp_transfer_slices for the destination rank."""
        for dst_rank in range(dst_tp):
            dst_kv_heads = max(1, total_heads // dst_tp)
            dst_spec = _make_fa_spec(num_kv_heads=dst_kv_heads)
            old_slices = dst_spec.get_tp_transfer_slices(
                dst_rank, dst_tp, src_tp, total_heads
            )
            old_source_ranks = set(old_slices.keys())

            src_kv_heads = max(1, total_heads // src_tp)
            src_spec = _make_fa_spec(num_kv_heads=src_kv_heads)
            tensor = _make_meta_tensor(num_kv_heads=src_kv_heads)

            new_source_ranks = set()
            for src_rank in range(src_tp):
                slices = src_spec.slice_for_tp_transfer(
                    tensor, src_tp, src_rank, dst_spec, dst_tp, dst_rank
                )
                if slices:
                    new_source_ranks.add(src_rank)

            assert new_source_ranks == old_source_ranks, (
                f"src_tp={src_tp}, dst_tp={dst_tp}, dst_rank={dst_rank}: "
                f"old={old_source_ranks}, new={new_source_ranks}"
            )

            for src_rank in old_source_ranks:
                old_slice = old_slices[src_rank]
                new_slices = src_spec.slice_for_tp_transfer(
                    tensor, src_tp, src_rank, dst_spec, dst_tp, dst_rank
                )
                assert len(new_slices) == 1
                assert new_slices[0].shape[1] == old_slice.num_elements, (
                    f"src_rank={src_rank}: head count mismatch "
                    f"old={old_slice.num_elements}, "
                    f"new={new_slices[0].shape[1]}"
                )


class TestSliceForTPTransferMLA:
    """Test MLAAttentionSpec.slice_for_tp_transfer."""

    def _make_mla_spec(self):
        return MLAAttentionSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.float16,
        )

    def test_aligned_rank_returns_full(self):
        spec = self._make_mla_spec()
        tensor = torch.empty(NUM_BLOCKS, 1, BLOCK_SIZE, 512, device="meta")
        slices = spec.slice_for_tp_transfer(tensor, 2, 0, spec, 2, 0)
        assert len(slices) == 1
        assert slices[0] is tensor

    def test_non_aligned_returns_empty(self):
        spec = self._make_mla_spec()
        tensor = torch.empty(NUM_BLOCKS, 1, BLOCK_SIZE, 512, device="meta")
        slices = spec.slice_for_tp_transfer(tensor, 2, 1, spec, 2, 0)
        assert len(slices) == 0

    def test_hetero_tp_load_balance(self):
        """D_TP=4, P_TP=2: each P rank serves 2 D ranks."""
        spec = self._make_mla_spec()
        tensor = torch.empty(NUM_BLOCKS, 1, BLOCK_SIZE, 512, device="meta")

        served_by_r0 = sum(
            1 for d in range(4) if spec.slice_for_tp_transfer(tensor, 2, 0, spec, 4, d)
        )
        served_by_r1 = sum(
            1 for d in range(4) if spec.slice_for_tp_transfer(tensor, 2, 1, spec, 4, d)
        )
        assert served_by_r0 == 2
        assert served_by_r1 == 2


class TestSliceForTPTransferSlidingWindowMLA:
    """Test SlidingWindowMLASpec.slice_for_tp_transfer."""

    def test_aligned_rank_returns_full(self):
        spec = SlidingWindowMLASpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.float16,
            sliding_window=4096,
        )
        tensor = torch.empty(NUM_BLOCKS, 1, BLOCK_SIZE, 512, device="meta")
        slices = spec.slice_for_tp_transfer(tensor, 2, 0, spec, 2, 0)
        assert len(slices) == 1
        assert slices[0] is tensor

    def test_non_aligned_returns_empty(self):
        spec = SlidingWindowMLASpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.float16,
            sliding_window=4096,
        )
        tensor = torch.empty(NUM_BLOCKS, 1, BLOCK_SIZE, 512, device="meta")
        slices = spec.slice_for_tp_transfer(tensor, 2, 1, spec, 2, 0)
        assert len(slices) == 0


class TestSliceForTPTransferMamba:
    """Test MambaSpec.slice_for_tp_transfer with sub-projection slicing.

    Uses realistic Mamba2 shapes for TP=2:
      conv: (conv_dim_local=160, conv_rows=3) → 480 elements
      ssm:  (num_heads_local=8, head_dim=16)  → 128 elements
      total C = 608 elements

    Decomposition at TP=2:
      intermediate_size = 8*2*16 = 256, groups_ss = (160*2 - 256)/2 = 32
      proj_dims = (x=128, B=16, C=16), conv_rows=3
      conv sub-proj elements: (384, 48, 48)
    """

    CONV_DIM_LOCAL = 160
    CONV_ROWS = 3
    SSM_HEADS_LOCAL = 8
    SSM_HEAD_DIM = 16
    TOTAL_C = CONV_DIM_LOCAL * CONV_ROWS + SSM_HEADS_LOCAL * SSM_HEAD_DIM

    def _make_mamba_spec(self, tp: int = 2):
        conv_dim = self.CONV_DIM_LOCAL * tp // tp  # stays per-rank
        ssm_heads = self.SSM_HEADS_LOCAL * tp // tp
        return MambaSpec(
            block_size=1,
            shapes=((conv_dim, self.CONV_ROWS), (ssm_heads, self.SSM_HEAD_DIM)),
            dtypes=(torch.float16, torch.float16),
        )

    def _make_tensor(self):
        return torch.empty(NUM_BLOCKS, 1, 1, self.TOTAL_C, device="meta")

    def test_homo_tp_returns_4_slices(self):
        """Same TP: returns 4 slices covering the full C dimension."""
        spec = self._make_mamba_spec(tp=2)
        tensor = self._make_tensor()
        slices = spec.slice_for_tp_transfer(tensor, 2, 0, spec, 2, 0)

        assert len(slices) == 4
        total_elements = sum(s.shape[3] for s in slices)
        assert total_elements == self.TOTAL_C

        # proj_dims at TP=2: (128, 16, 16), conv_rows=3
        assert slices[0].shape[3] == 128 * 3  # x
        assert slices[1].shape[3] == 16 * 3  # B
        assert slices[2].shape[3] == 16 * 3  # C
        assert slices[3].shape[3] == 8 * 16  # SSM

    def test_homo_tp_non_aligned_empty(self):
        spec = self._make_mamba_spec(tp=2)
        tensor = self._make_tensor()
        slices = spec.slice_for_tp_transfer(tensor, 2, 1, spec, 2, 0)
        assert len(slices) == 0

    def test_d_gt_p_sub_projection_slicing(self):
        """D_TP=4, P_TP=2: dst reads half of each src sub-projection."""
        src_spec = self._make_mamba_spec(tp=2)
        # At TP=4, shapes are halved: conv_dim=80, ssm_heads=4
        dst_spec = MambaSpec(
            block_size=1,
            shapes=((80, self.CONV_ROWS), (4, self.SSM_HEAD_DIM)),
            dtypes=(torch.float16, torch.float16),
        )
        tensor = self._make_tensor()

        # dst_rank 0 reads from src_rank 0 (aligned)
        slices = src_spec.slice_for_tp_transfer(tensor, 2, 0, dst_spec, 4, 0)
        assert len(slices) == 4

        # dst proj_dims at TP=4: intermediate=4*4*16=256, conv_dim*4=320,
        # groups_ss=(320-256)/2=32, x=256/4=64, b=32/4=8 → (64, 8, 8)
        assert slices[0].shape[3] == 64 * 3  # x (half of src's 128*3)
        assert slices[1].shape[3] == 8 * 3  # B (half of src's 16*3)
        assert slices[2].shape[3] == 8 * 3  # C
        assert slices[3].shape[3] == 4 * 16  # SSM (half of src's 8*16)

        # dst_rank 1 also reads from src_rank 0, second half
        slices_r1 = src_spec.slice_for_tp_transfer(tensor, 2, 0, dst_spec, 4, 1)
        assert len(slices_r1) == 4
        assert slices_r1[0].shape[3] == 64 * 3

        # dst_rank 2 reads from src_rank 1 (different src)
        slices_miss = src_spec.slice_for_tp_transfer(tensor, 2, 0, dst_spec, 4, 2)
        assert len(slices_miss) == 0

    def test_p_gt_d_reads_full_src(self):
        """P_TP=4, D_TP=2: dst reads entire src sub-projections."""
        # src at TP=4: conv_dim=80, ssm_heads=4
        src_spec = MambaSpec(
            block_size=1,
            shapes=((80, self.CONV_ROWS), (4, self.SSM_HEAD_DIM)),
            dtypes=(torch.float16, torch.float16),
        )
        dst_spec = self._make_mamba_spec(tp=2)
        src_c = 80 * 3 + 4 * 16
        tensor = torch.empty(NUM_BLOCKS, 1, 1, src_c, device="meta")

        # src_rank 0 contributes to dst_rank 0
        slices = src_spec.slice_for_tp_transfer(tensor, 4, 0, dst_spec, 2, 0)
        assert len(slices) == 4

        # src proj_dims at TP=4: (64, 8, 8), conv_rows=3
        # abs_ratio=2, so remote elements = local // 2
        assert slices[0].shape[3] == 64 * 3 // 2  # x
        assert slices[1].shape[3] == 8 * 3 // 2  # B
        assert slices[2].shape[3] == 8 * 3 // 2  # C
        assert slices[3].shape[3] == 4 * 16 // 2  # SSM

        # src_rank 1 also contributes to dst_rank 0
        slices_r1 = src_spec.slice_for_tp_transfer(tensor, 4, 1, dst_spec, 2, 0)
        assert len(slices_r1) == 4

        # src_rank 2 does NOT contribute to dst_rank 0
        slices_miss = src_spec.slice_for_tp_transfer(tensor, 4, 2, dst_spec, 2, 0)
        assert len(slices_miss) == 0

    def test_source_rank_selection_matches_old_api(self):
        """Verify source rank selection matches get_tp_transfer_slices."""
        spec = self._make_mamba_spec(tp=2)
        tensor = self._make_tensor()

        for dst_tp in [1, 2, 4]:
            for dst_rank in range(dst_tp):
                old_slices = spec.get_tp_transfer_slices(dst_rank, dst_tp, 2, 8)
                old_ranks = set(old_slices.keys())

                new_ranks = {
                    r
                    for r in range(2)
                    if spec.slice_for_tp_transfer(tensor, 2, r, spec, dst_tp, dst_rank)
                }
                assert new_ranks == old_ranks
