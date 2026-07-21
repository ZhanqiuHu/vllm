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
    stride_order: tuple[int, ...],
    layers_per_region: int = 1,
) -> torch.Tensor:
    """Build a logical cache-region view with the requested stride order."""
    dtype = getattr(spec, "dtype", torch.int8)
    elem = get_dtype_size(dtype)
    H, N, C = spec.compute_transfer_shape(
        region_content_bytes // layers_per_region, block_size
    )
    size = (
        (num_blocks, H, layers_per_region, N, C)
        if layers_per_region > 1
        else (num_blocks, H, N, C)
    )
    assert sorted(stride_order) == list(range(len(size)))
    assert stride_order[0] == _DIM4_B

    strides = [0] * len(size)
    running_stride = 1
    for dim in reversed(stride_order):
        strides[dim] = running_stride
        running_stride *= size[dim]
    strides[_DIM4_B] = block_stride_bytes // elem
    return torch.as_strided(
        torch.empty(1, dtype=dtype, device="meta"),
        size=size,
        stride=strides,
        storage_offset=0,
    )


def _stride_order_for_layout(
    kv_cache_layout: str, layers_per_region: int
) -> tuple[int, ...]:
    if layers_per_region > 1:
        if kv_cache_layout in ("NHD", "NHC"):
            return (0, 2, 3, 1, 4)
        return (0, 1, 2, 3, 4)
    if kv_cache_layout in ("NHD", "NHC"):
        return (0, 2, 1, 3)
    return (0, 1, 2, 3)


def jointly_contiguous_chunks(
    *views: torch.Tensor,
) -> Iterator[tuple[torch.Tensor, ...]]:
    """Yield minimal jointly contiguous chunks for dense cache views."""
    assert len(views) >= 2
    assert all(view.shape == views[0].shape for view in views[1:])
    assert all(view.element_size() == views[0].element_size() for view in views[1:])
    if all(view.is_contiguous() for view in views):
        yield views
        return

    split_dim = max(
        (dim for dim, size in enumerate(views[0].shape) if size > 1),
        key=lambda dim: max(view.stride(dim) for view in views),
    )
    for parts in zip(*(view.unbind(split_dim) for view in views), strict=True):
        yield from jointly_contiguous_chunks(*parts)


def _stack_nixl_descriptors(
    addrs: np.ndarray, length: int, device_id: int
) -> np.ndarray:
    descriptors = np.empty((addrs.shape[0], 3), dtype=np.uint64)
    descriptors[:, 0] = addrs
    descriptors[:, 1] = length
    descriptors[:, 2] = device_id
    return descriptors


class NixlBaseConnectorWorkerMultiview(NixlBaseConnectorWorker):
    """Build matching local and remote descriptor lists during handshake."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, engine_id, kv_cache_config)

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
        spec_type: type[KVCacheSpec],
        kernel_num_blocks: int,
        physical_blocks_per_logical: int,
    ) -> int:
        if issubclass(spec_type, MambaSpec):
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
        assert descs_per_block_per_group is not None
        layout = descs_per_block_per_group
        assert len(layout) == len(block_ids)
        kernel_num_blocks = dst_num_blocks
        if block_size_ratio is not None:
            kernel_num_blocks = int(kernel_num_blocks * block_size_ratio)

        offset = 0
        all_descs: list[np.ndarray] = []
        for group_idx, group_block_ids in enumerate(block_ids):
            num_blocks = self._group_num_blocks(
                self._group_spec_types[group_idx],
                kernel_num_blocks,
                physical_blocks_per_logical,
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
    def _views_to_nixl_descriptors(
        local_view: torch.Tensor,
        remote_view: torch.Tensor,
        local_base_addr: int,
        remote_base_addr: int,
        local_device_id: int,
        remote_device_id: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        return NixlBaseConnectorWorkerMultiview._mapped_views_to_nixl_descriptors(
            [local_view],
            remote_view,
            local_base_addr,
            remote_base_addr,
            local_device_id,
            remote_device_id,
        )

    @staticmethod
    def _mapped_views_to_nixl_descriptors(
        local_views: list[torch.Tensor],
        remote_view: torch.Tensor,
        local_base_addr: int,
        remote_base_addr: int,
        local_device_id: int,
        remote_device_id: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert local_views
        local_num_blocks = local_views[0].shape[_DIM4_B]
        assert all(view.shape[_DIM4_B] == local_num_blocks for view in local_views)
        local_block_stride = (
            local_views[0].stride(_DIM4_B) * local_views[0].element_size()
        )
        remote_block_stride = remote_view.stride(_DIM4_B) * remote_view.element_size()
        local_block_offsets = (
            np.arange(local_num_blocks, dtype=np.uint64) * local_block_stride
        )
        remote_block_offsets = (
            np.arange(remote_view.shape[_DIM4_B], dtype=np.uint64) * remote_block_stride
        )

        local_blocks = tuple(view.select(_DIM4_B, 0) for view in local_views)
        remote_block = remote_view.select(_DIM4_B, 0)

        local_parts: list[np.ndarray] = []
        remote_parts: list[np.ndarray] = []
        for *local_chunks, remote_chunk in jointly_contiguous_chunks(
            *local_blocks, remote_block
        ):
            remote_offset = remote_chunk.storage_offset() * remote_chunk.element_size()
            length = remote_chunk.numel() * remote_chunk.element_size()
            subblock_offsets = np.asarray(
                [
                    chunk.storage_offset() * chunk.element_size()
                    for chunk in local_chunks
                ],
                dtype=np.uint64,
            )
            local_addrs = (
                local_base_addr
                + local_block_offsets[:, None]
                + subblock_offsets[None, :]
            ).flatten()
            local_parts.append(
                _stack_nixl_descriptors(
                    local_addrs,
                    length,
                    local_device_id,
                )
            )
            remote_parts.append(
                _stack_nixl_descriptors(
                    remote_base_addr + remote_offset + remote_block_offsets,
                    length,
                    remote_device_id,
                )
            )
        return np.concatenate(local_parts), np.concatenate(remote_parts)

    def _local_region_metas(
        self,
        spec: KVCacheSpec,
        region_idx: int,
        block_size_ratio: int,
    ) -> list[torch.Tensor]:
        if isinstance(spec, MambaSpec):
            assert block_size_ratio == 1
            num_blocks = self._logical_num_blocks
            stride = (
                self.block_len_per_layer[region_idx]
                * self._physical_blocks_per_logical_kv_block
            )
            content = sum(self._mamba_ssm_size)
        elif isinstance(spec, AttentionSpec):
            num_blocks = self.num_blocks
            stride = self.block_stride_per_layer[region_idx]
            content = self.block_len_per_layer[region_idx]
        else:
            raise ValueError(f"Unsupported KV cache spec: {type(spec).__name__}")
        kv_cache_layout = (
            self.host_buffer_kv_cache_layout
            if self.use_host_buffer
            else self.kv_cache_layout
        )
        layers_per_region = (
            len(self.kv_cache_config.kv_cache_tensors)
            if self.transfer_topo is not None and self.transfer_topo.cross_layers_blocks
            else 1
        )
        meta = build_region_meta(
            spec,
            num_blocks,
            self.block_size,
            stride,
            content,
            _stride_order_for_layout(kv_cache_layout, layers_per_region),
            layers_per_region,
        )
        if block_size_ratio == 1:
            return [meta]

        transfer_block_size = self.block_size // block_size_ratio
        token_dim = meta.ndim - 2
        return [
            meta.narrow(
                token_dim,
                subblock_idx * transfer_block_size,
                transfer_block_size,
            )
            for subblock_idx in range(block_size_ratio)
        ]

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
        layers_per_region = (
            len(self.kv_cache_config.kv_cache_tensors)
            if self.transfer_topo is not None and self.transfer_topo.cross_layers_blocks
            else 1
        )
        return build_region_meta(
            spec,
            num_blocks,
            metadata.block_size,
            stride,
            content,
            _stride_order_for_layout(metadata.kv_cache_layout, layers_per_region),
            layers_per_region,
        )

    def _build_packed_descriptor_pair(
        self,
        metadata: NixlAgentMetadata,
        remote_tp_rank: int,
        remote_tp_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self.world_size == remote_tp_size or self.use_mla, (
            "Heterogeneous TP for non-replicated packed KV requires semantic "
            "per-layer metadata"
        )
        assert len(self.block_len_per_layer) == 1
        assert len(metadata.block_lens) == 1
        local_content = self.block_len_per_layer[0]
        remote_content = metadata.block_lens[0]
        assert local_content == remote_content

        local_meta = torch.as_strided(
            torch.empty(1, dtype=torch.uint8, device="meta"),
            size=(self.num_blocks, 1, 1, local_content),
            stride=(self.block_stride_per_layer[0], local_content, local_content, 1),
        )
        remote_meta = torch.as_strided(
            torch.empty(1, dtype=torch.uint8, device="meta"),
            size=(metadata.num_blocks, 1, 1, remote_content),
            stride=(metadata.block_strides[0], remote_content, remote_content, 1),
        )
        local_descs, remote_descs = self._views_to_nixl_descriptors(
            local_meta,
            remote_meta,
            self.kv_caches_base_addr[self.engine_id][self.tp_rank][0],
            metadata.kv_caches_base_addr[0],
            self.device_id,
            metadata.device_id,
        )
        num_groups = len(self.kv_cache_config.kv_cache_groups)
        local_descs = np.concatenate([local_descs] * num_groups)
        remote_descs = np.concatenate([remote_descs] * num_groups)
        self.xfer_descs_per_block_by_remote[metadata.engine_id][remote_tp_rank] = tuple(
            1 for _ in range(num_groups)
        )
        return local_descs, remote_descs

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
                local_metas = self._local_region_metas(
                    spec, region_idx, block_size_ratio
                )
                remote_meta = self._remote_region_meta(
                    spec,
                    region_idx,
                    metadata,
                    physical_blocks_per_logical,
                )
                local_slices_per_subblock = [
                    spec.slice_for_tp_transfer(
                        local_meta,
                        *remote_size_and_rank,
                        *local_size_and_rank,
                        self.model_config,
                    )
                    for local_meta in local_metas
                ]
                remote_slices = spec.slice_for_tp_transfer(
                    remote_meta,
                    *local_size_and_rank,
                    *remote_size_and_rank,
                    self.model_config,
                )
                assert all(
                    len(local_slices) == len(remote_slices)
                    for local_slices in local_slices_per_subblock
                )

                for slice_idx, remote_slice in enumerate(remote_slices):
                    local_slices = [
                        slices[slice_idx] for slices in local_slices_per_subblock
                    ]
                    local_descs, remote_descs = self._mapped_views_to_nixl_descriptors(
                        local_slices,
                        remote_slice,
                        local_base,
                        metadata.kv_caches_base_addr[region_idx],
                        self.device_id,
                        metadata.device_id,
                    )
                    local_num_blocks = len(local_slices[0]) * len(local_slices)
                    local_descs_per_block = len(local_descs) // local_num_blocks
                    remote_descs_per_block = len(remote_descs) // len(remote_slice)
                    assert local_descs_per_block == remote_descs_per_block
                    assert np.array_equal(
                        local_descs[::local_num_blocks, 1],
                        remote_descs[:: len(remote_slice), 1],
                    )
                    group_descs_per_block += local_descs_per_block
                    local_parts.append(local_descs)
                    remote_parts.append(remote_descs)
            descs_per_block_per_group.append(group_descs_per_block)

        descriptor_layout = tuple(descs_per_block_per_group)
        self.xfer_descs_per_block_by_remote[metadata.engine_id][remote_tp_rank] = (
            descriptor_layout
        )
        return np.concatenate(local_parts), np.concatenate(remote_parts)

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
        if self._is_packed_kv:
            assert block_size_ratio == 1, (
                "Heterogeneous block sizes for packed KV require semantic "
                "per-layer metadata"
            )
            return self._build_packed_descriptor_pair(
                nixl_agent_meta, remote_tp_rank, remote_tp_size
            )
        assert self._can_build_descriptor_pairs()

        return self._build_descriptor_pair(
            nixl_agent_meta,
            block_size_ratio,
            physical_blocks_per_logical,
            remote_tp_size,
            remote_tp_rank,
        )
