# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in, correctness-first Qwen GDN/DFlash pull prototype.

Descriptors are built per request, by layer identity rather than allocation
region identity. This intentionally trades CPU overhead for inspectable mapping.
"""

import os
from dataclasses import dataclass

from vllm.logger import init_logger

logger = init_logger(__name__)


def enabled(config):
    return os.getenv("VLLM_NIXL_EXPERIMENTAL_LAYER_MAPPING") == "1" or bool(
        config.kv_transfer_config
        and config.kv_transfer_config.get_from_extra_config(
            "experimental_layer_mapping", False
        )
    )


@dataclass
class LayerGeometry:
    group: int
    base: int
    stride: int
    blocks: int
    device: int
    kind: str
    block_tokens: int
    heads: int
    cell_bytes: int
    window: int
    signature: str
    # Conv Q/K/V followed by temporal state; offsets within a logical page.
    segments: list[tuple[int, int]]


def validate_pair(local, remote, ratio):
    if (local.kind, local.signature, local.window) != (
        remote.kind,
        remote.signature,
        remote.window,
    ):
        raise ValueError("Layer type, dtype, window or state geometry differs")
    if ratio < 1:
        raise ValueError("Prototype requires D TP >= P TP")
    if local.kind == "ssm":
        if len(local.segments) != len(remote.segments) or any(
            rsize != lsize * ratio
            for (_, lsize), (_, rsize) in zip(local.segments, remote.segments)
        ):
            raise ValueError("SSM components must have matching TP-scaled shapes")
    elif local.cell_bytes != remote.cell_bytes or remote.heads not in (
        local.heads,
        local.heads * ratio,
    ):
        raise ValueError("Unsupported per-layer attention head geometry")


def plan_layer(local, remote, local_ids, remote_ids, tokens, ratio, rank_slot):
    """Return paired byte descriptors; attention IDs retain absolute positions."""
    validate_pair(local, remote, ratio)
    if not 0 <= rank_slot < ratio or tokens <= 0:
        raise ValueError("Invalid rank slot or transferred token count")
    dst, src = [], []

    def append_pair(di, doff, si, soff, length):
        if not (0 < di < local.blocks and 0 < si < remote.blocks):
            raise ValueError("Missing/null/out-of-range block in required token range")
        if length <= 0 or min(doff, soff) < 0:
            raise ValueError("Invalid transfer extent")
        if doff + length > local.stride or soff + length > remote.stride:
            raise ValueError("Transfer extends beyond its layer page")
        dst.append((local.base + di * local.stride + doff, length, local.device))
        src.append((remote.base + si * remote.stride + soff, length, remote.device))

    if local.kind == "ssm":
        if len(local_ids) != 1 or len(remote_ids) != 1:
            raise ValueError("Prototype requires one running SSM state per layer")
        for (doff, size), (soff, _) in zip(local.segments, remote.segments):
            append_pair(
                local_ids[0], doff, remote_ids[0], soff + rank_slot * size, size
            )
        return dst, src

    head_start = 0 if local.heads == remote.heads else rank_slot * local.heads
    pos = max(0, tokens - local.window) if local.window else 0
    while pos < tokens:
        db, dt = divmod(pos, local.block_tokens)
        sb, st = divmod(pos, remote.block_tokens)
        if db >= len(local_ids) or sb >= len(remote_ids):
            raise ValueError("Block table does not cover the transferred token range")
        count = min(tokens - pos, local.block_tokens - dt, remote.block_tokens - st)
        for head in range(local.heads):
            doff = (head * local.block_tokens + dt) * local.cell_bytes
            soff = ((head_start + head) * remote.block_tokens + st) * remote.cell_bytes
            append_pair(
                local_ids[db], doff, remote_ids[sb], soff, count * local.cell_bytes
            )
        pos += count
    return dst, src


def register(worker, caches):
    import msgspec

    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        NixlAgentMetadata,
        NixlHandshakePayload,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils import (
        derive_mamba_conv_split,
    )
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVQuantMode,
        MambaSpec,
        SlidingWindowSpec,
    )

    cfg = worker.vllm_config
    logger.info(
        "NIXL_MAP_V1 source=%s engine=%s rank=%s TP/PP/DCP/PCP=%s "
        "layout=%s block_tokens=%s physical_per_logical=%s transfer_config=%s",
        __file__, worker.engine_id, worker.tp_rank,
        (worker.world_size, worker.pp_size, worker.dcp_size, worker.pcp_size),
        worker.kv_cache_layout, worker.block_size,
        worker._physical_blocks_per_logical_kv_block, cfg.kv_transfer_config,
    )
    group_by_layer = worker.kv_cache_config.transfer_group_index_by_layer
    for gid, group in enumerate(worker.kv_cache_config.kv_cache_groups):
        transfer_gid = (
            group_by_layer.get(group.layer_names[0])
            if group.layer_names else None
        )
        logger.info(
            "NIXL_MAP_GROUP engine=%s rank=%s kv_group=%s transfer_group=%s "
            "is_eagle_group=%s role=%s spec=%r layers=%s",
            worker.engine_id, worker.tp_rank, gid, transfer_gid,
            group.is_eagle_group, group.role, group.kv_cache_spec, group.layer_names,
        )
    if (
        worker._TRANSFER_MODE != "pull"
        or worker.kv_cache_layout != "LBHNC"
        or worker.pp_size != 1
        or worker.dcp_size != 1
        or worker.pcp_size != 1
        or worker.use_host_buffer
        or cfg.cache_config.enable_prefix_caching
        or worker.enable_permute_local_kv
        or worker._is_csa_linear
        or worker._physical_blocks_per_logical_kv_block != 1
        or cfg.model_config.hf_text_config.model_type != "qwen3_5_moe_text"
    ):
        raise ValueError(
            "Layer-map prototype requires Qwen3.6 MoE, LBHNC, pull, "
            "PP/PCP/DCP=1, no prefix/host cache, no kernel block splitting"
        )
    layers, storage_ranges = {}, {}
    for name in sorted(worker._layer_specs):
        spec = worker._layer_specs[name]
        cache = caches[name]
        if cache.device.type != "cuda":
            raise ValueError("Prototype requires device-resident layer caches")
        group = worker.kv_cache_config.transfer_group_index_by_layer[name]
        stride = (
            spec.page_size_bytes
            if cache.ndim == 1
            else (cache.stride(0) * cache.element_size())
        )
        blocks = worker._logical_num_blocks
        storage = cache.untyped_storage()
        storage_ranges[storage.data_ptr()] = (storage.nbytes(), cache.get_device())
        segments = []
        if isinstance(spec, MambaSpec):
            if spec.mamba_cache_mode != "none" or spec.tp_replicated:
                raise ValueError("Prototype requires unreplicated, single-state GDN")
            split = derive_mamba_conv_split(spec, worker.world_size)
            conv, ssm = split.ssm_sizes
            segments = [*split.local_conv_offsets, (conv, ssm)]
            # DS conv's state length and temporal per-head shape must match.
            signature = str(
                (
                    tuple(str(d) for d in spec.dtypes),
                    tuple(s[1:] for s in spec.shapes),
                    str(spec.mamba_type),
                )
            )
            kind, heads, cell, window = "ssm", 0, 0, 0
            extent = conv + ssm
        elif type(spec) in (FullAttentionSpec, SlidingWindowSpec):
            if (
                spec.kv_quant_mode != KVQuantMode.NONE
                or spec.tokens_per_state != 1
                or spec.num_head_slots is not None
                or spec.state_content_bytes is not None
            ):
                raise ValueError("Prototype only supports dense unquantized attention")
            heads, cell = spec.num_kv_heads, spec.state_content_size_bytes
            window = (
                spec.sliding_window + spec.extra_retained_tokens
                if isinstance(spec, SlidingWindowSpec)
                else 0
            )
            kind = "attention"
            signature = str(
                (type(spec).__name__, str(spec.dtype), spec.head_size, spec.head_size_v)
            )
            extent = spec.unpadded_page_size_bytes
            if cache.ndim != 1 and (
                cache.ndim != 4
                or not cache[0].is_contiguous()
                or cache[0].nbytes != extent
            ):
                raise ValueError("Expected dense LBHNC layer view")
        else:
            raise ValueError(f"Unsupported layer spec: {type(spec).__name__}")
        if extent > stride or cache.data_ptr() + (blocks - 1) * stride + extent > (
            storage.data_ptr() + storage.nbytes()
        ):
            raise ValueError(f"Layer {name} exceeds its backing allocation")
        layers[name] = LayerGeometry(
            group,
            cache.data_ptr(),
            stride,
            blocks,
            cache.get_device(),
            kind,
            spec.block_size,
            heads,
            cell,
            window,
            signature,
            segments,
        )
        logger.info(
            "NIXL_MAP_LAYER engine=%s rank=%s layer=%s spec=%r shape=%s "
            "dtype=%s strides_elements=%s storage_base=%s storage_bytes=%s "
            "offset_bytes=%s page_bytes=%s payload_bytes=%s padding_bytes=%s "
            "geometry=%r",
            worker.engine_id, worker.tp_rank, name, spec, tuple(cache.shape),
            cache.dtype, cache.stride(), storage.data_ptr(), storage.nbytes(),
            cache.data_ptr() - storage.data_ptr(), spec.page_size_bytes,
            extent, spec.page_size_bytes - extent, layers[name],
        )

    worker._layer_mapping_local = layers
    worker._layer_mapping_remote = {}
    worker._layer_mapping_traced = set()
    worker.device_kv_caches = caches
    for base, (size, device) in storage_ranges.items():
        desc = worker.nixl_wrapper.get_reg_descs(
            [(base, size, device, "")], worker.nixl_memory_type
        )
        worker.nixl_wrapper.register_memory(desc, backends=worker.nixl_backends)
        worker._registered_descs.append(desc)
    meta = NixlAgentMetadata(
        engine_id=worker.engine_id,
        agent_metadata=worker.nixl_wrapper.get_agent_metadata(),
        kv_caches_base_addr=[],
        device_id=next(iter(layers.values())).device,
        num_blocks=worker.num_blocks,
        block_lens=[],
        block_strides=[],
        kv_cache_layout=worker.kv_cache_layout,
        block_size=worker.block_size,
        ssm_sizes=worker._mamba_ssm_size,
        attn_backend_name=worker.backend_name,
        physical_blocks_per_logical_kv_block=1,
        layer_geometry=layers,
    )
    worker.xfer_handshake_metadata = NixlHandshakePayload(
        compatibility_hash=worker.compat_hash,
        agent_metadata_bytes=msgspec.msgpack.encode(meta),
    )


def add_remote(worker, meta, rank, tp, dcp):
    from vllm.distributed.kv_transfer.kv_connector.utils import EngineTransferInfo
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
        compute_tp_mapping,
    )

    logger.info(
        "NIXL_MAP_HANDSHAKE local_engine=%s local_rank=%s remote_engine=%s "
        "remote_rank=%s remote_TP/DCP/PCP=%s layout=%s block_tokens=%s "
        "num_blocks=%s physical_per_logical=%s backend=%s ssm_sizes=%s",
        worker.engine_id, worker.tp_rank, meta.engine_id, rank,
        (tp, dcp, meta.pcp_size), meta.kv_cache_layout, meta.block_size,
        meta.num_blocks, meta.physical_blocks_per_logical_kv_block,
        meta.attn_backend_name, meta.ssm_sizes,
    )
    if (
        dcp != 1
        or meta.pcp_size != 1
        or meta.kv_cache_layout != "LBHNC"
        or meta.physical_blocks_per_logical_kv_block != 1
        or worker.world_size < tp
        or worker.world_size % tp
    ):
        raise ValueError("Unsupported remote topology/layout for layer-map prototype")
    local_names = set(worker._layer_mapping_local)
    remote_names = set(meta.layer_geometry)
    if local_names != remote_names:
        raise ValueError(
            "P/D transferable layer names differ: "
            f"local_only={sorted(local_names - remote_names)}, "
            f"remote_only={sorted(remote_names - local_names)}"
        )
    ratio = worker.world_size // tp
    if rank != worker.tp_rank // ratio:
        raise ValueError("Unexpected producer rank")
    for name, local in worker._layer_mapping_local.items():
        logger.info(
            "NIXL_MAP_PAIR layer=%s P(engine=%s rank=%s geometry=%r) -> "
            "D(engine=%s rank=%s geometry=%r) tp_ratio=%s rank_slot=%s",
            name, meta.engine_id, rank, meta.layer_geometry[name],
            worker.engine_id, worker.tp_rank, local, ratio, worker.tp_rank % ratio,
        )
        validate_pair(local, meta.layer_geometry[name], ratio)
    info = EngineTransferInfo(
        remote_tp_size=tp,
        remote_block_size=meta.block_size,
        remote_block_len=0,
        remote_physical_blocks_per_logical=1,
        remote_dcp_size=1,
    )
    worker.transfer_topo.register_remote_engine(meta.engine_id, info)
    worker.tp_mappings[meta.engine_id] = compute_tp_mapping(
        transfer_topology=worker.transfer_topo,
        remote_tp_size=tp,
        group_spec_types=worker._group_spec_types,
        remote_dcp_size=1,
    )
    worker._layer_mapping_remote[(meta.engine_id, rank)] = meta.layer_geometry
    worker._layer_mapping_traced.discard((meta.engine_id, rank))
    return worker.nixl_wrapper.add_remote_agent(meta.agent_metadata)


def read(worker, req_id, meta):
    """Use the normal async completion/error/notification lifecycle."""
    remote = meta.remote
    plan = worker.tp_mappings[remote.engine_id]
    if len(plan.all_source_ranks) != 1:
        raise ValueError("Prototype requires a single producer rank per consumer")
    rank = plan.all_source_ranks[0]
    agent = worker._remote_agents[remote.engine_id][(0, rank)]
    notif = f"{remote.request_id}:{plan.local_consumers}".encode()
    handle = None
    name = None
    try:
        if not any(meta.local_block_ids):
            meta.region_blocks_to_zero = ()
            worker.nixl_wrapper.send_notif(agent, notif_msg=notif)
            if meta.awaiting_kvs:
                worker._recving_transfers.setdefault(req_id, [])
            return
        if any(meta.local_num_computed_blocks):
            raise ValueError("Prototype does not support cached prefixes")
        tokens = remote.num_tokens
        if not tokens:
            raise ValueError("Layer mapping requires remote_num_tokens")
        ratio = (
            worker.world_size
            // worker.transfer_topo.get_engine_info(remote.engine_id).remote_tp_size
        )
        src, dst = [], []
        geometry = worker._layer_mapping_remote[(remote.engine_id, rank)]
        trace = (remote.engine_id, rank) not in worker._layer_mapping_traced
        if trace:
            logger.info(
                "NIXL_MAP_REQUEST req=%s remote_req=%s remote_engine=%s "
                "P_rank=%s D_rank=%s tokens=%s consumers=%s "
                "P_block_ids=%s D_block_ids=%s computed_blocks=%s",
                req_id, remote.request_id, remote.engine_id, rank, worker.tp_rank,
                tokens, plan.local_consumers, remote.block_ids,
                meta.local_block_ids, meta.local_num_computed_blocks,
            )
        for name, local in worker._layer_mapping_local.items():
            other = geometry[name]
            d, s = plan_layer(
                local,
                other,
                meta.local_block_ids[local.group],
                remote.block_ids[other.group],
                tokens,
                ratio,
                worker.tp_rank % ratio,
            )
            dst.extend(d)
            src.extend(s)
            if trace:
                logger.info(
                    "NIXL_MAP_COPY req=%s rank=%s layer=%s P_group=%s D_group=%s "
                    "descriptors(address,bytes,device) P=%s D=%s",
                    req_id, worker.tp_rank, name, other.group, local.group, s, d,
                )
        local_desc = worker.nixl_wrapper.get_xfer_descs(dst, worker.nixl_memory_type)
        remote_desc = worker.nixl_wrapper.get_xfer_descs(src, worker.nixl_memory_type)
        handle = worker.nixl_wrapper.initialize_xfer(
            "READ",
            local_desc,
            remote_desc,
            agent,
            notif_msg=notif,
        )
        worker.nixl_wrapper.transfer(handle)
        worker._recving_transfers[req_id].append(handle)
        if trace:
            logger.info(
                "NIXL_MAP_SUBMITTED req=%s rank=%s descriptors=%s bytes=%s "
                "completion=pending",
                req_id, worker.tp_rank, len(src), sum(length for _, length, _ in src),
            )
            worker._layer_mapping_traced.add((remote.engine_id, rank))
        # Descriptors already land in local layout; no group-index postprocessing.
        meta.region_blocks_to_zero = ()
    except Exception as exc:
        worker._log_failure(
            failure_type="transfer_setup_failed",
            req_id=req_id,
            msg=f"Experimental layer mapping failed at layer={name}",
            error=exc,
            dst_engine_id=remote.engine_id,
            remote_rank=rank,
        )
        if not worker._handle_failed_transfer(req_id, handle, worker._recv_failures):
            worker._recving_transfers[req_id].append(handle)
