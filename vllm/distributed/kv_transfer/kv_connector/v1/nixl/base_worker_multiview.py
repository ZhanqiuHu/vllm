# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired local/remote descriptor construction for the NIXL connector."""

from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.distributed.kv_transfer.kv_connector.utils import (
    BlockIds,
    EngineTransferInfo,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    NixlBaseConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlAgentMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    TPMapping,
)
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

_DIM4_B = 0


def build_region_meta(
    spec: KVCacheSpec,
    num_blocks: int,
    block_size: int,
    block_stride_bytes: int,
    region_content_bytes: int,
    kv_cache_layout: str = "HND",
) -> torch.Tensor:
    """Build a ``(B, H, N, C)`` metadata tensor for a cache region."""
    dtype = getattr(spec, "dtype", torch.int8)
    elem = get_dtype_size(dtype)
    num_heads, num_tokens, content_dim = spec.compute_transfer_shape(
        region_content_bytes, block_size
    )
    if kv_cache_layout in ("NHD", "NHC"):
        inner_strides = (
            content_dim,
            num_heads * content_dim,
            1,
        )
    else:
        inner_strides = (
            num_tokens * content_dim,
            content_dim,
            1,
        )
    return torch.as_strided(
        torch.empty(1, dtype=dtype, device="meta"),
        size=(num_blocks, num_heads, num_tokens, content_dim),
        stride=(block_stride_bytes // elem, *inner_strides),
        storage_offset=0,
    )


def _jointly_contiguous_subviews(
    local_view: torch.Tensor,
    remote_view: torch.Tensor,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    if local_view.is_contiguous() and remote_view.is_contiguous():
        yield local_view, remote_view
        return

    split_dim = next(dim for dim, size in enumerate(local_view.shape) if size > 1)
    for local_part, remote_part in zip(
        local_view.unbind(split_dim), remote_view.unbind(split_dim)
    ):
        yield from _jointly_contiguous_subviews(local_part, remote_part)


def jointly_contiguous_chunks(
    local_view: torch.Tensor,
    remote_view: torch.Tensor,
) -> Iterator[tuple[int, int, int]]:
    """Yield byte ranges that are contiguous in both logical views.

    The block dimension is excluded. Returned offsets are relative to block zero
    and include each view's storage offset.
    """
    assert local_view.shape[1:] == remote_view.shape[1:]
    assert local_view.element_size() == remote_view.element_size()

    elem = local_view.element_size()
    local_block = local_view.select(_DIM4_B, 0)
    remote_block = remote_view.select(_DIM4_B, 0)

    pending: tuple[int, int, int] | None = None
    for local_part, remote_part in _jointly_contiguous_subviews(
        local_block, remote_block
    ):
        current = (
            local_part.storage_offset() * elem,
            remote_part.storage_offset() * elem,
            local_part.numel() * elem,
        )
        if pending is None:
            pending = current
            continue
        local_start, remote_start, length = pending
        if current[0] == local_start + length and current[1] == remote_start + length:
            pending = (local_start, remote_start, length + current[2])
        else:
            yield pending
            pending = current
    if pending is not None:
        yield pending


class NixlBaseConnectorWorkerMultiview(NixlBaseConnectorWorker):
    """Build matching local and remote descriptor lists during handshake."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, engine_id, kv_cache_config)
        self._descs_per_block_per_group: tuple[int, ...] = ()

    def _group_specs(self) -> list[KVCacheSpec]:
        specs: list[KVCacheSpec] = []
        for group in self.kv_cache_config.kv_cache_groups:
            spec = group.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = next(iter(spec.kv_cache_specs.values()))
            specs.append(spec)
        return specs

    def _can_build_descriptor_pairs(self) -> bool:
        local_bases = self.kv_caches_base_addr.get(self.engine_id, {})
        return (
            self.tp_rank in local_bases
            and bool(local_bases[self.tp_rank])
            and bool(self.spec_per_region)
            and len(self.block_stride_per_layer) == len(self.spec_per_region)
        )

    def _group_num_blocks(
        self,
        spec: KVCacheSpec,
        kernel_num_blocks: int,
        physical_blocks_per_logical: int,
    ) -> int:
        if isinstance(spec, MambaSpec):
            return kernel_num_blocks // physical_blocks_per_logical
        return kernel_num_blocks

    def _compute_desc_ids(
        self,
        block_ids: BlockIds,
        dst_num_blocks: int,
        block_size_ratio: float | None,
        physical_blocks_per_logical: int,
        descs_per_block_per_group: tuple[int, ...] | None = None,
    ) -> np.ndarray:
        layout = descs_per_block_per_group or self._descs_per_block_per_group
        if self._is_packed_kv or not layout:
            return super()._compute_desc_ids(
                block_ids,
                dst_num_blocks,
                block_size_ratio,
                physical_blocks_per_logical,
                descs_per_block_per_group,
            )

        assert len(layout) == len(block_ids)
        kernel_num_blocks = dst_num_blocks
        if block_size_ratio is not None:
            kernel_num_blocks = int(kernel_num_blocks * block_size_ratio)

        offset = 0
        all_descs: list[np.ndarray] = []
        for group_idx, (spec, group_block_ids) in enumerate(
            zip(self._group_specs(), block_ids)
        ):
            num_blocks = self._group_num_blocks(
                spec, kernel_num_blocks, physical_blocks_per_logical
            )
            num_streams = layout[group_idx]
            all_descs.append(
                (
                    offset
                    + np.arange(num_streams)[:, None] * num_blocks
                    + np.asarray(group_block_ids)[None, :]
                ).flatten()
            )
            offset += num_streams * num_blocks
        return np.concatenate(all_descs)

    @staticmethod
    def _view_to_nixl_descriptors(
        view: torch.Tensor,
        base_addr: int,
        device_id: int,
    ) -> np.ndarray:
        block_stride = view.stride(_DIM4_B) * view.element_size()
        block_offsets = np.arange(view.shape[_DIM4_B], dtype=np.uint64) * block_stride
        parts: list[np.ndarray] = []
        for offset, _, length in jointly_contiguous_chunks(view, view):
            parts.append(
                NixlBaseConnectorWorker._stack_descs(
                    base_addr + offset + block_offsets,
                    length,
                    device_id,
                )
            )
        return np.concatenate(parts)

    @staticmethod
    def _views_to_nixl_descriptors(
        local_view: torch.Tensor,
        remote_view: torch.Tensor,
        local_base_addr: int,
        remote_base_addr: int,
        local_device_id: int,
        remote_device_id: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        local_block_stride = local_view.stride(_DIM4_B) * local_view.element_size()
        remote_block_stride = remote_view.stride(_DIM4_B) * remote_view.element_size()
        local_block_offsets = (
            np.arange(local_view.shape[_DIM4_B], dtype=np.uint64) * local_block_stride
        )
        remote_block_offsets = (
            np.arange(remote_view.shape[_DIM4_B], dtype=np.uint64) * remote_block_stride
        )
        assert len(local_block_offsets) == len(remote_block_offsets)

        local_parts: list[np.ndarray] = []
        remote_parts: list[np.ndarray] = []
        for local_offset, remote_offset, length in jointly_contiguous_chunks(
            local_view, remote_view
        ):
            local_parts.append(
                NixlBaseConnectorWorker._stack_descs(
                    local_base_addr + local_offset + local_block_offsets,
                    length,
                    local_device_id,
                )
            )
            remote_parts.append(
                NixlBaseConnectorWorker._stack_descs(
                    remote_base_addr + remote_offset + remote_block_offsets,
                    length,
                    remote_device_id,
                )
            )
        return np.concatenate(local_parts), np.concatenate(remote_parts)

    def _local_region_meta(
        self,
        spec: KVCacheSpec,
        region_idx: int,
        block_size_ratio: int,
    ) -> torch.Tensor:
        if isinstance(spec, MambaSpec):
            assert block_size_ratio == 1
            num_blocks = self._logical_num_blocks
            stride = (
                self.block_len_per_layer[region_idx]
                * self._physical_blocks_per_logical_kv_block
            )
            content = sum(self._mamba_ssm_size)
        elif isinstance(spec, AttentionSpec):
            num_blocks = self.num_blocks * block_size_ratio
            stride = self.block_stride_per_layer[region_idx] // block_size_ratio
            content = self.block_len_per_layer[region_idx] // block_size_ratio
        else:
            raise ValueError(f"Unsupported KV cache spec: {type(spec).__name__}")
        kv_cache_layout = (
            self.host_buffer_kv_cache_layout
            if self.use_host_buffer
            else self.kv_cache_layout
        )
        return build_region_meta(
            spec,
            num_blocks,
            self.block_size,
            stride,
            content,
            kv_cache_layout,
        )

    def _remote_region_meta(
        self,
        spec: KVCacheSpec,
        region_idx: int,
        metadata: NixlAgentMetadata,
        physical_blocks_per_logical: int,
    ) -> torch.Tensor:
        if isinstance(spec, MambaSpec):
            num_blocks = metadata.num_blocks // physical_blocks_per_logical
            stride = metadata.block_strides[region_idx] * physical_blocks_per_logical
            content = sum(metadata.ssm_sizes)
        elif isinstance(spec, AttentionSpec):
            num_blocks = metadata.num_blocks
            stride = metadata.block_strides[region_idx]
            content = metadata.block_lens[region_idx]
        else:
            raise ValueError(f"Unsupported KV cache spec: {type(spec).__name__}")
        return build_region_meta(
            spec,
            num_blocks,
            metadata.block_size,
            stride,
            content,
            metadata.kv_cache_layout,
        )

    def _build_descriptor_pair(
        self,
        metadata: NixlAgentMetadata,
        block_size_ratio: int,
        physical_blocks_per_logical: int,
        remote_tp_size: int,
        remote_tp_rank: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build descriptor lists in one pass and verify pairwise compatibility."""
        local_parts: list[np.ndarray] = []
        remote_parts: list[np.ndarray] = []
        descs_per_block_per_group: list[int] = []
        local_bases = self.kv_caches_base_addr[self.engine_id][self.tp_rank]
        local_size_and_rank = (self.world_size, self.tp_rank)
        remote_size_and_rank = (remote_tp_size, remote_tp_rank)

        for spec in self._group_specs():
            group_descs_per_block = 0
            for region_idx, local_base in enumerate(local_bases):
                local_meta = self._local_region_meta(spec, region_idx, block_size_ratio)
                remote_meta = self._remote_region_meta(
                    spec,
                    region_idx,
                    metadata,
                    physical_blocks_per_logical,
                )
                local_slices = spec.slice_for_tp_transfer(
                    local_meta,
                    *remote_size_and_rank,
                    *local_size_and_rank,
                    self.model_config,
                )
                remote_slices = spec.slice_for_tp_transfer(
                    remote_meta,
                    *local_size_and_rank,
                    *remote_size_and_rank,
                    self.model_config,
                )
                assert len(local_slices) == len(remote_slices)

                for local_slice, remote_slice in zip(local_slices, remote_slices):
                    local_descs, remote_descs = self._views_to_nixl_descriptors(
                        local_slice,
                        remote_slice,
                        local_base,
                        metadata.kv_caches_base_addr[region_idx],
                        self.device_id,
                        metadata.device_id,
                    )
                    assert len(local_descs) == len(remote_descs)
                    assert np.array_equal(local_descs[:, 1], remote_descs[:, 1])
                    group_descs_per_block += len(local_descs) // len(local_slice)
                    local_parts.append(local_descs)
                    remote_parts.append(remote_descs)
            descs_per_block_per_group.append(group_descs_per_block)

        descriptor_layout = tuple(descs_per_block_per_group)
        self.xfer_descs_per_block_by_remote[metadata.engine_id][remote_tp_rank] = (
            descriptor_layout
        )
        return np.concatenate(local_parts), np.concatenate(remote_parts)

    def register_local_xfer_handler(
        self,
        block_size: int,
    ) -> tuple[int, np.ndarray]:
        if self._is_packed_kv:
            return super().register_local_xfer_handler(block_size)

        block_size_ratio = self.block_size // block_size
        parts: list[np.ndarray] = []
        descs_per_block_per_group: list[int] = []
        local_bases = self.kv_caches_base_addr[self.engine_id][self.tp_rank]
        for spec in self._group_specs():
            group_descs_per_block = 0
            for region_idx, base_addr in enumerate(local_bases):
                meta = self._local_region_meta(spec, region_idx, block_size_ratio)
                slices = spec.slice_for_tp_transfer(
                    meta,
                    self.world_size,
                    self.tp_rank,
                    self.world_size,
                    self.tp_rank,
                    self.model_config,
                )
                for view in slices:
                    view_descs = self._view_to_nixl_descriptors(
                        view, base_addr, self.device_id
                    )
                    group_descs_per_block += len(view_descs) // len(view)
                    parts.append(view_descs)
            descs_per_block_per_group.append(group_descs_per_block)

        self._descs_per_block_per_group = tuple(descs_per_block_per_group)
        blocks_data = np.concatenate(parts)
        descs = self.nixl_wrapper.get_xfer_descs(blocks_data, self.nixl_memory_type)
        handle = self.nixl_wrapper.prep_xfer_dlist("NIXL_INIT_AGENT", descs)
        return handle, blocks_data

    def _build_xfer_descs(
        self,
        nixl_agent_meta: NixlAgentMetadata,
        plan: TPMapping,
        block_size_ratio: int,
        tp_ratio: int,
        transfer_info: EngineTransferInfo,
        physical_blocks_per_logical: int,
        remote_tp_rank: int,
        remote_tp_size: int,
    ) -> tuple[np.ndarray | None, np.ndarray]:
        if (
            self._is_packed_kv
            or block_size_ratio != 1
            or not self._can_build_descriptor_pairs()
        ):
            return super()._build_xfer_descs(
                nixl_agent_meta,
                plan,
                block_size_ratio,
                tp_ratio,
                transfer_info,
                physical_blocks_per_logical,
                remote_tp_rank,
                remote_tp_size,
            )

        return self._build_descriptor_pair(
            nixl_agent_meta,
            block_size_ratio,
            physical_blocks_per_logical,
            remote_tp_size,
            remote_tp_rank,
        )

    def _build_local_splits_from_plan(
        self,
        plan: TPMapping,
        src_blocks_data: np.ndarray,
        num_fa_descs: int | None = None,
    ) -> Iterator[list[tuple[int, int, int]]]:
        if self._is_packed_kv or not self._can_build_descriptor_pairs():
            yield from super()._build_local_splits_from_plan(
                plan,
                src_blocks_data,
                num_fa_descs if num_fa_descs is not None else 0,
            )
        # Non-packed local slices are built directly with their remote peers.
