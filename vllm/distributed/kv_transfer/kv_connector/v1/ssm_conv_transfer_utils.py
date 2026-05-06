# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Conv-state sub-projection decomposition for the 3-read transfer.

With DS conv state layout (dim, state_len), sub-projections are
contiguous in memory.  Each D rank reads its slices via 3 separate
RDMA transfers — no P-side permutation needed.

Supported model types:
  - Mamba2: conv = [x, B, C], temporal = (num_heads, head_dim)
  - GDN (Gated Delta Net): conv = [K, K, V], temporal = (num_v_heads, v_dim, k_dim)
"""

import math
from dataclasses import dataclass

import torch

from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.v1.kv_cache_interface import MambaSpec

_SUPPORTED_MAMBA_TYPES = ("mamba2", "gdn_attention")


@dataclass(frozen=True)
class MambaConvSplitInfo:
    """Per-rank byte sizes of the 3 conv sub-projections.

    Used by both P and D sides for NIXL descriptor registration.
    All fields are LOCAL to this engine's TP (already divided by TP size).

    The conv state has 3 contiguous sub-projections in DS layout:
      Mamba2: |-- x --|- B -|- C -|  (B == C)
      GDN:    |- K -|- K -|-- V --|  (K and V may differ)
    """

    conv_rows: int  # conv_kernel - 1 (typically 3)
    local_proj_dims: tuple[int, int, int]  # per-rank column counts per sub-proj
    conv_dtype_size: int  # bytes per element (e.g. 2 for float16)
    ssm_sizes: tuple[int, int]  # (conv_state_bytes, ssm_state_bytes)

    @property
    def local_conv_dim(self) -> int:
        """Total conv columns per rank."""
        return sum(self.local_proj_dims)

    @property
    def proj_bytes(self) -> tuple[int, int, int]:
        """Byte sizes of the 3 sub-projections for one rank."""
        row_bytes = self.conv_rows * self.conv_dtype_size
        return tuple(d * row_bytes for d in self.local_proj_dims)  # type: ignore[return-value]

    @property
    def local_conv_offsets(self) -> list[tuple[int, int]]:
        """(byte_offset, byte_size) of each sub-projection within this
        engine's page.

        Used by both P and D for local descriptor registration.
        """
        conv0, conv1, conv2 = self.proj_bytes
        return [(0, conv0), (conv0, conv1), (conv0 + conv1, conv2)]

    def remote_conv_offsets(
        self, local_rank_offset: int, tp_ratio: int
    ) -> list[tuple[int, int]]:
        """(byte_offset, byte_size) of this D rank's sub-projection slices
        within one P page.

        Used by D side only, during remote descriptor registration.

        Args:
            local_rank_offset: which slice this D rank reads.
            tp_ratio: signed TP ratio.
                >= 1:  D_TP >= P_TP — P page is larger, D reads its slice.
                < 0:   P_TP > D_TP — P pages are smaller, D reads entire
                       P page.  Local dims are scaled down by |tp_ratio|
                       to get P-sized offsets.
        """
        conv0, conv1, conv2 = self.proj_bytes
        if tp_ratio >= 1:
            remote_conv0 = conv0 * tp_ratio
            remote_conv1 = conv1 * tp_ratio
            return [
                (local_rank_offset * conv0, conv0),
                (remote_conv0 + local_rank_offset * conv1, conv1),
                (remote_conv0 + remote_conv1 + local_rank_offset * conv2, conv2),
            ]
        else:
            abs_ratio = -tp_ratio
            remote_conv0 = conv0 // abs_ratio
            remote_conv1 = conv1 // abs_ratio
            remote_conv2 = conv2 // abs_ratio
            return [
                (0, remote_conv0),
                (remote_conv0, remote_conv1),
                (remote_conv0 + remote_conv1, remote_conv2),
            ]


def _compute_ssm_byte_sizes(
    mamba_spec: MambaSpec,
) -> tuple[int, int, int, int]:
    """Return (conv_dtype_size, ssm_dtype_size, conv_bytes, ssm_bytes)."""
    conv_dtype_size = torch.tensor(
        [],
        dtype=mamba_spec.dtypes[0],  # type: ignore[misc]
    ).element_size()
    ssm_dtype_size = torch.tensor(
        [],
        dtype=mamba_spec.dtypes[1],  # type: ignore[misc]
    ).element_size()
    conv_state_bytes = torch.Size(mamba_spec.shapes[0]).numel() * conv_dtype_size
    ssm_state_bytes = torch.Size(mamba_spec.shapes[1]).numel() * ssm_dtype_size
    return conv_dtype_size, ssm_dtype_size, conv_state_bytes, ssm_state_bytes


def _derive_mamba2_conv_split(
    mamba_spec: MambaSpec,
    local_tp: int,
    local_conv_dim: int,
    conv_rows: int,
) -> MambaConvSplitInfo:
    """Mamba2 decomposition: conv = [x, B, C] where B == C."""
    head_dim = mamba_spec.shapes[1][1]
    local_num_heads = mamba_spec.shapes[1][0]
    intermediate_size = local_num_heads * local_tp * head_dim

    remainder = local_conv_dim * local_tp - intermediate_size
    assert remainder > 0 and remainder % 2 == 0, (
        f"Conv dim ({local_conv_dim}*tp={local_tp}) doesn't decompose into "
        f"intermediate_size={intermediate_size} + 2*groups_ss. "
        f"remainder={remainder}"
    )
    groups_ss = remainder // 2

    conv_dtype_size, _, conv_state_bytes, ssm_state_bytes = _compute_ssm_byte_sizes(
        mamba_spec
    )

    x_local = intermediate_size // local_tp
    b_local = groups_ss // local_tp
    return MambaConvSplitInfo(
        conv_rows=conv_rows,
        local_proj_dims=(x_local, b_local, b_local),
        conv_dtype_size=conv_dtype_size,
        ssm_sizes=(conv_state_bytes, ssm_state_bytes),
    )


def _derive_gdn_conv_split(
    mamba_spec: MambaSpec,
    local_conv_dim: int,
    conv_rows: int,
) -> MambaConvSplitInfo:
    """GDN decomposition: conv = [K, K, V].

    GDN conv_dim = key_dim*2 + value_dim (all global, divided by TP).
    The temporal state shape is (num_v_heads/TP, head_v_dim, head_k_dim).
    We recover value_dim_local from the temporal state, then derive
    key_dim_local from the conv remainder.
    """
    temporal_shape = mamba_spec.shapes[1]
    num_v_heads_local = temporal_shape[0]
    head_v_dim = temporal_shape[1]
    value_dim_local = num_v_heads_local * head_v_dim

    remainder = local_conv_dim - value_dim_local
    assert remainder > 0 and remainder % 2 == 0, (
        f"GDN conv dim ({local_conv_dim}) doesn't decompose into "
        f"2*key_dim_local + value_dim_local={value_dim_local}. "
        f"remainder={remainder}"
    )
    key_dim_local = remainder // 2

    conv_dtype_size, _, conv_state_bytes, ssm_state_bytes = _compute_ssm_byte_sizes(
        mamba_spec
    )

    return MambaConvSplitInfo(
        conv_rows=conv_rows,
        local_proj_dims=(key_dim_local, key_dim_local, value_dim_local),
        conv_dtype_size=conv_dtype_size,
        ssm_sizes=(conv_state_bytes, ssm_state_bytes),
    )


def derive_mamba_conv_split(
    mamba_spec: MambaSpec,
    local_tp: int,
) -> MambaConvSplitInfo:
    """Derive per-rank sub-projection byte sizes from a MambaSpec.

    Called once at init on both P and D.  Decomposes the conv dimension
    into its 3 sub-projection parts based on the model type.

    Args:
        mamba_spec: MambaSpec whose shapes are:
            shapes[0] = conv state: (local_conv_dim, conv_rows) in DS layout.
            shapes[1] = temporal state (model-specific shape).
        local_tp: this engine's tensor-parallel size.

    Returns:
        MambaConvSplitInfo with per-rank sub-projection dims, conv_rows,
        conv_dtype_size, and ssm_sizes (conv_state_bytes, ssm_state_bytes).
    """
    if mamba_spec.mamba_type not in _SUPPORTED_MAMBA_TYPES:
        raise NotImplementedError(
            f"3-read conv transfer supports {_SUPPORTED_MAMBA_TYPES}, "
            f"got mamba_type={mamba_spec.mamba_type!r}."
        )

    conv_shape = mamba_spec.shapes[0]
    assert len(conv_shape) == 2, f"Expected 2D conv state shape, got {conv_shape}"

    assert is_conv_state_dim_first(), "3-read requires DS conv state layout"
    local_conv_dim = conv_shape[0]  # DS: (local_conv_dim, conv_rows)
    conv_rows = conv_shape[1]

    if mamba_spec.mamba_type == "mamba2":
        info = _derive_mamba2_conv_split(
            mamba_spec, local_tp, local_conv_dim, conv_rows
        )
    else:
        info = _derive_gdn_conv_split(mamba_spec, local_conv_dim, conv_rows)

    return info


def compute_physical_blocks_per_logical(
    ssm_sizes: tuple[int, ...], block_len: int
) -> int:
    """Derive _physical_blocks_per_logical_kv_block from remote metadata.

    The remote engine's ratio is not sent directly in the handshake, so we
    reconstruct it: total mamba state per logical block / block_len.

    Args:
        ssm_sizes: (conv_state_bytes, ssm_state_bytes) from NixlAgentMetadata.
        block_len: the engine's block_len in bytes (from block_lens[0]).
    """
    return math.ceil((ssm_sizes[0] + ssm_sizes[1]) / block_len)
