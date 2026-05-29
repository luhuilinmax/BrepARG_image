#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


def install_numpy_pickle_compat() -> None:
    """Allow NumPy-2 pickles to load in environments exposing numpy.core only."""
    try:
        import numpy.core as np_core

        sys.modules.setdefault("numpy._core", np_core)
        sys.modules.setdefault("numpy._core.multiarray", np_core.multiarray)
        sys.modules.setdefault("numpy._core.numeric", np_core.numeric)
    except Exception:
        pass


def load_pickle(path: str) -> Any:
    install_numpy_pickle_compat()
    with open(path, "rb") as f:
        return pickle.load(f)


def resolve_sample_path(args: argparse.Namespace) -> str:
    if args.sample_pkl:
        return remap_path(args.sample_pkl, args.path_prefix_from, args.path_prefix_to)

    split_data = load_pickle(args.split_pkl)
    if args.split not in split_data:
        raise KeyError(f"Split '{args.split}' not found in {args.split_pkl}")
    paths = split_data[args.split]
    if not paths:
        raise ValueError(f"Split '{args.split}' is empty")
    if args.sample_index < 0 or args.sample_index >= len(paths):
        raise IndexError(f"sample_index {args.sample_index} out of range for split length {len(paths)}")
    return remap_path(paths[args.sample_index], args.path_prefix_from, args.path_prefix_to)


def remap_path(path: str, old_prefix: str, new_prefix: str) -> str:
    if os.path.exists(path):
        return path
    if old_prefix and new_prefix and path.startswith(old_prefix):
        remapped = new_prefix + path[len(old_prefix):]
        if os.path.exists(remapped):
            return remapped
    return path


def require_array(cad: Dict[str, Any], key: str, shape_tail: Tuple[int, ...]) -> np.ndarray:
    if key not in cad:
        raise KeyError(f"Missing required field '{key}'")
    arr = np.asarray(cad[key], dtype=np.float32)
    if arr.ndim != len(shape_tail) + 1 or tuple(arr.shape[1:]) != shape_tail:
        raise ValueError(f"Field '{key}' has shape {arr.shape}, expected (N, {', '.join(map(str, shape_tail))})")
    return arr


def load_cad_sample(sample_path: str) -> Dict[str, Any]:
    cad = load_pickle(sample_path)
    if not isinstance(cad, dict):
        raise TypeError(f"Expected dict pkl, got {type(cad).__name__}")

    surf_ncs = require_array(cad, "surf_ncs", (32, 32, 3))
    edge_ncs = require_array(cad, "edge_ncs", (32, 3))
    surf_bbox_wcs = require_array(cad, "surf_bbox_wcs", (6,))
    edge_bbox_wcs = require_array(cad, "edge_bbox_wcs", (6,))

    if "edgeFace_adj" not in cad:
        raise KeyError("Missing required field 'edgeFace_adj'")
    edge_face_adj = np.asarray(cad["edgeFace_adj"], dtype=np.int64)
    if edge_face_adj.ndim != 2 or edge_face_adj.shape[1] < 2:
        raise ValueError(f"edgeFace_adj has shape {edge_face_adj.shape}, expected (E, 2)")

    return {
        "surf_ncs": surf_ncs,
        "edge_ncs": edge_ncs,
        "surf_bbox_wcs": surf_bbox_wcs,
        "edge_bbox_wcs": edge_bbox_wcs,
        "edgeFace_adj": edge_face_adj[:, :2],
        "raw": cad,
    }


def prepare_se_batch(surf_ncs: np.ndarray, edge_ncs: np.ndarray) -> np.ndarray:
    surf_data = surf_ncs.astype(np.float32)
    edge_data = edge_ncs.astype(np.float32)
    edge_expanded = np.tile(edge_data[:, :, np.newaxis, :], (1, 1, 32, 1))
    combined = np.concatenate([surf_data, edge_expanded], axis=0)
    return combined.transpose(0, 3, 1, 2)


def extract_token_indices(indices: Any) -> torch.Tensor:
    token_indices = (
        indices[2] if isinstance(indices, tuple) and len(indices) > 2
        else indices[0] if isinstance(indices, tuple)
        else indices
    )
    if not isinstance(token_indices, torch.Tensor):
        raise TypeError(f"Unexpected quantizer index type: {type(token_indices).__name__}")
    return token_indices


def encode_se_tokens(model: torch.nn.Module, se_data: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        se_tensor = torch.tensor(se_data, dtype=torch.float32, device=device)
        h = model.encoder(se_tensor)
        h = model.quant_conv(h)
        _, _, indices = model.quantize(h)
        token_indices = extract_token_indices(indices)
    tokens_per_element = int(token_indices.numel() // len(se_data))
    return token_indices.detach().cpu().reshape(len(se_data), tokens_per_element).numpy().astype(np.int64)


def dfs_face_ordering_from_core(edge_face_pairs: List[Tuple[int, int]], num_faces: int) -> Tuple[List[int], Dict[int, int]]:
    """
    Match 2sequence.py: start each component from the highest-degree face, then DFS
    through low-degree neighbors first.
    """
    nbrs = [set() for _ in range(num_faces)]
    for f1, f2 in edge_face_pairs:
        if 0 <= f1 < num_faces and 0 <= f2 < num_faces and f1 != f2:
            nbrs[f1].add(f2)
            nbrs[f2].add(f1)

    deg = [len(n) for n in nbrs]
    visited = [False] * num_faces
    face_order: List[int] = []
    seeds = sorted(range(num_faces), key=lambda x: (-deg[x], x))

    def dfs(u: int) -> None:
        visited[u] = True
        face_order.append(u)
        unvisited_neighbors = [v for v in nbrs[u] if not visited[v]]
        unvisited_neighbors.sort(key=lambda x: (deg[x], x))
        for v in unvisited_neighbors:
            if not visited[v]:
                dfs(v)

    for seed in seeds:
        if not visited[seed]:
            dfs(seed)

    face_position_map = {face_idx: pos for pos, face_idx in enumerate(face_order)}
    return face_order, face_position_map


def lexicographic_edge_ordering(edge_face_pairs: List[Tuple[int, int]]) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Match 2sequence.py MAX-IDX-A style ordering: sort by (max(face), min(face)).
    """
    edge_sort_info = []
    for edge_idx, pair in enumerate(edge_face_pairs):
        if len(pair) < 2:
            continue
        f1, f2 = int(pair[0]), int(pair[1])
        edge_sort_info.append(((max(f1, f2), min(f1, f2)), edge_idx, (f1, f2)))

    edge_sort_info.sort(key=lambda item: item[0])
    edge_order = [item[1] for item in edge_sort_info]
    ordered_edge_face_pairs = [item[2] for item in edge_sort_info]
    return edge_order, ordered_edge_face_pairs


def build_vocab_info(dataset_type: str, se_tokens_per_element: int) -> Dict[str, int]:
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)[dataset_type].copy()

    cfg["special_token_size"] = 4
    cfg["bbox_tokens_per_element"] = 6
    cfg["face_index_offset"] = 0
    cfg["se_token_offset"] = cfg["face_index_size"]
    cfg["bbox_token_offset"] = cfg["se_token_offset"] + cfg["se_codebook_size"]
    special_start = cfg["bbox_token_offset"] + cfg["bbox_index_size"]
    cfg["START_TOKEN"] = special_start
    cfg["SEP_TOKEN"] = special_start + 1
    cfg["END_TOKEN"] = special_start + 2
    cfg["PAD_TOKEN"] = special_start + 3
    cfg["vocab_size"] = special_start + cfg["special_token_size"]
    cfg["se_tokens_per_element"] = se_tokens_per_element
    return cfg


def build_roundtrip_sequence(
    surf_tokens: np.ndarray,
    edge_tokens: np.ndarray,
    surf_bbox_wcs: np.ndarray,
    edge_bbox_wcs: np.ndarray,
    edge_face_adj: np.ndarray,
    vocab_info: Dict[str, int],
    scale_factor: float,
    seed: int,
    reindex_offset: int,
) -> Tuple[List[int], Dict[str, Any]]:
    num_faces = len(surf_tokens)
    if num_faces > vocab_info["face_index_size"]:
        raise ValueError(
            f"Sample has {num_faces} faces, but config supports only {vocab_info['face_index_size']} face-index tokens"
        )
    if len(edge_tokens) != len(edge_face_adj):
        raise ValueError(f"Edge token count {len(edge_tokens)} does not match edgeFace_adj count {len(edge_face_adj)}")

    edge_face_pairs = [(int(pair[0]), int(pair[1])) for pair in np.asarray(edge_face_adj)[:, :2]]
    face_order, face_position_map = dfs_face_ordering_from_core(edge_face_pairs, num_faces)

    surf_tokens = surf_tokens[face_order]
    surf_bbox_wcs = surf_bbox_wcs[face_order]

    remapped_edge_face_pairs = [
        (face_position_map[f1], face_position_map[f2])
        for f1, f2 in edge_face_pairs
    ]
    edge_order, ordered_edge_face_pairs = lexicographic_edge_ordering(remapped_edge_face_pairs)
    edge_tokens = edge_tokens[edge_order]
    edge_bbox_wcs = edge_bbox_wcs[edge_order]

    face_index_size = int(vocab_info["face_index_size"])
    # 2sequence.py uses N = min(face_index_size, max_face). For this script the
    # config face_index_size is the same default cap (50 for ABC/DeepCAD).
    max_reindex_faces = face_index_size
    if max_reindex_faces <= 0:
        cyclic_offset = 0
    elif reindex_offset >= 0:
        cyclic_offset = reindex_offset % max_reindex_faces
    else:
        rng = np.random.default_rng(seed)
        cyclic_offset = int(rng.integers(0, max_reindex_faces))

    face_index_map = {i: (i + cyclic_offset) % face_index_size for i in range(num_faces)}

    bbox_offset = vocab_info["bbox_token_offset"]
    se_offset = vocab_info["se_token_offset"]
    face_offset = vocab_info["face_index_offset"]

    surf_bbox_tokens = quantize_bbox_local(surf_bbox_wcs * scale_factor, vocab_info["bbox_index_size"])
    edge_bbox_tokens = quantize_bbox_local(edge_bbox_wcs * scale_factor, vocab_info["bbox_index_size"])

    tokens: List[int] = [vocab_info["START_TOKEN"]]
    for face_idx in range(num_faces):
        tokens.extend((bbox_offset + int(x)) for x in surf_bbox_tokens[face_idx])
        tokens.extend((se_offset + int(x)) for x in surf_tokens[face_idx])
        tokens.append(face_offset + face_index_map[face_idx])

    tokens.append(vocab_info["SEP_TOKEN"])
    for edge_idx, (src, dst) in enumerate(ordered_edge_face_pairs):
        tokens.append(face_offset + face_index_map[int(src)])
        tokens.append(face_offset + face_index_map[int(dst)])
        tokens.extend((bbox_offset + int(x)) for x in edge_bbox_tokens[edge_idx])
        tokens.extend((se_offset + int(x)) for x in edge_tokens[edge_idx])

    tokens.append(vocab_info["END_TOKEN"])
    ordering_info = {
        "face_order": [int(x) for x in face_order],
        "edge_order": [int(x) for x in edge_order],
        "ordered_edge_face_pairs": [[int(a), int(b)] for a, b in ordered_edge_face_pairs],
        "face_index_map": {str(k): int(v) for k, v in face_index_map.items()},
        "cyclic_offset": int(cyclic_offset),
        "seed": int(seed),
    }
    return tokens, ordering_info


def quantize_bbox_local(bbox_coords: np.ndarray, num_tokens: int) -> np.ndarray:
    normalized = (bbox_coords + 1.0) / 2.0
    normalized = np.clip(normalized, 0.0, 1.0)
    return np.round(normalized * (num_tokens - 1)).astype(np.int64)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def write_brep_outputs(solid: Any, output_dir: str, prefix: str) -> Dict[str, Any]:
    from OCC.Extend.DataExchange import write_step_file, write_stl_file
    from utils import check_brep_validity

    step_path = os.path.join(output_dir, f"{prefix}.step")
    stl_path = os.path.join(output_dir, f"{prefix}.stl")
    write_step_file(solid, step_path)
    write_stl_file(solid, stl_path, linear_deflection=0.001, angular_deflection=0.5)
    return {
        "step_path": step_path,
        "stl_path": stl_path,
        "brep_valid": bool(check_brep_validity(step_path)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one-sample VQ-VAE encode-decode CAD reconstruction.")
    parser.add_argument("--sample_pkl", type=str, default="", help="Path to one preprocessed CAD pkl")
    parser.add_argument("--split_pkl", type=str, default="", help="Optional split pkl used when sample_pkl is omitted")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--path_prefix_from", type=str, default="/workspace/data")
    parser.add_argument("--path_prefix_to", type=str, default="/data/project/ly/data")
    parser.add_argument("--se_vqvae", type=str, required=True, help="Path to ABC SE VQ-VAE checkpoint")
    parser.add_argument("--dataset_type", type=str, default="abc", choices=["abc", "deepcad"])
    parser.add_argument("--output_dir", type=str, default="result/vqvae_reconstruct")
    parser.add_argument("--filename_prefix", type=str, default="vqvae_recon")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--scale_factor", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42, help="Seed used when cyclic face re-index offset is sampled")
    parser.add_argument(
        "--reindex_offset",
        type=int,
        default=-1,
        help="Cyclic face re-index offset; -1 samples one from --seed like 2sequence.py",
    )
    parser.add_argument("--skip_brep", action="store_true", help="Only write token/NCS diagnostics")
    args = parser.parse_args()

    if not args.sample_pkl and not args.split_pkl:
        raise ValueError("Provide either --sample_pkl or --split_pkl")

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    os.makedirs(args.output_dir, exist_ok=True)
    print("Resolving sample path...", flush=True)
    sample_path = resolve_sample_path(args)
    print("Loading CAD sample...", flush=True)
    cad = load_cad_sample(sample_path)

    print(f"Sample: {sample_path}")
    print(f"Faces: {len(cad['surf_ncs'])}, Edges: {len(cad['edge_ncs'])}")
    print(f"Device: {device}")

    print("Importing project VQ-VAE utilities...", flush=True)
    from utils import decode_tokens_to_ncs, load_se_vqvae_model, reconstruct_cad_from_sequence

    print("Loading VQ-VAE checkpoint...", flush=True)
    model = load_se_vqvae_model(args.se_vqvae, False, args.dataset_type, device)
    if model is None:
        raise RuntimeError(f"Failed to load VQ-VAE checkpoint: {args.se_vqvae}")

    print("Encoding surface/edge NCS to codebook tokens...", flush=True)
    se_data = prepare_se_batch(cad["surf_ncs"], cad["edge_ncs"])
    se_tokens = encode_se_tokens(model, se_data, device)
    num_faces = len(cad["surf_ncs"])
    surf_tokens = se_tokens[:num_faces]
    edge_tokens = se_tokens[num_faces:]
    tokens_per_element = int(se_tokens.shape[1])

    print("Decoding codebook tokens back to NCS...", flush=True)
    surf_recon = np.asarray(decode_tokens_to_ncs(surf_tokens.tolist(), model, "face", tokens_per_element, device))
    edge_recon = np.asarray(decode_tokens_to_ncs(edge_tokens.tolist(), model, "edge", tokens_per_element, device))

    metrics = {
        "sample_path": sample_path,
        "checkpoint": args.se_vqvae,
        "dataset_type": args.dataset_type,
        "num_faces": int(num_faces),
        "num_edges": int(len(edge_tokens)),
        "tokens_per_element": tokens_per_element,
        "token_min": int(se_tokens.min()) if se_tokens.size else None,
        "token_max": int(se_tokens.max()) if se_tokens.size else None,
        "face_mse": mse(cad["surf_ncs"], surf_recon) if len(surf_recon) else None,
        "edge_mse": mse(cad["edge_ncs"], edge_recon) if len(edge_recon) else None,
        "brep": None,
    }

    print("Saving round-trip diagnostics...", flush=True)
    npz_path = os.path.join(args.output_dir, f"{args.filename_prefix}_roundtrip.npz")
    np.savez_compressed(
        npz_path,
        surf_ncs=cad["surf_ncs"],
        edge_ncs=cad["edge_ncs"],
        surf_ncs_recon=surf_recon,
        edge_ncs_recon=edge_recon,
        surf_tokens=surf_tokens,
        edge_tokens=edge_tokens,
        surf_bbox_wcs=cad["surf_bbox_wcs"],
        edge_bbox_wcs=cad["edge_bbox_wcs"],
        edgeFace_adj=cad["edgeFace_adj"],
    )
    metrics["npz_path"] = npz_path

    vocab_info = build_vocab_info(args.dataset_type, tokens_per_element)
    sequence, ordering_info = build_roundtrip_sequence(
        surf_tokens,
        edge_tokens,
        cad["surf_bbox_wcs"],
        cad["edge_bbox_wcs"],
        cad["edgeFace_adj"],
        vocab_info,
        args.scale_factor,
        args.seed,
        args.reindex_offset,
    )
    metrics["ordering"] = ordering_info
    sequence_path = os.path.join(args.output_dir, f"{args.filename_prefix}_sequence.json")
    with open(sequence_path, "w", encoding="utf-8") as f:
        json.dump(sequence, f)
    ordering_path = os.path.join(args.output_dir, f"{args.filename_prefix}_ordering.json")
    with open(ordering_path, "w", encoding="utf-8") as f:
        json.dump(ordering_info, f, indent=2)
    metrics["sequence_path"] = sequence_path
    metrics["ordering_path"] = ordering_path
    metrics["sequence_length"] = len(sequence)

    if not args.skip_brep:
        if device.type != "cuda":
            metrics["brep"] = {"error": "Skipped: utils.joint_optimize currently calls .cuda() internally"}
        else:
            try:
                print("Reconstructing B-rep from the round-trip sequence...", flush=True)
                solid, debug_info = reconstruct_cad_from_sequence(
                    sequence=sequence,
                    vocab_info=vocab_info,
                    se_vqvae_model=model,
                    device=device,
                    scale_factor=args.scale_factor,
                    verbose=True,
                    return_debug=True,
                )
                debug_path = os.path.join(args.output_dir, f"{args.filename_prefix}_brep_debug.pkl")
                with open(debug_path, "wb") as f:
                    pickle.dump(debug_info, f)
                if solid is None:
                    metrics["brep"] = {"error": "reconstruct_cad_from_sequence returned None", "debug_path": debug_path}
                else:
                    metrics["brep"] = write_brep_outputs(solid, args.output_dir, args.filename_prefix)
                    metrics["brep"]["debug_path"] = debug_path
            except Exception as exc:
                metrics["brep"] = {"error": repr(exc)}

    metrics_path = os.path.join(args.output_dir, f"{args.filename_prefix}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
