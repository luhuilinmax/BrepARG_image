import os
import numpy as np
import torch
import pickle
import argparse
from tqdm import tqdm
import random
from collections import deque

# ===== Utility functions from your project =====
from utils import (
    quantize_bbox,
    bbox_corners,
    get_bbox,
    load_se_vqvae_model,
    rotate_axis,
)

# ==============================
#        Data Flow Tracer
# ==============================

class DataTracer:
    """Trace data transformations: type, shape, dtype, min/max/mean at each step."""

    def __init__(self, enabled=False):
        self.enabled = enabled
        self.records = []
        self.details = []
        self._step = 0

    def trace(self, name, data, note=""):
        if not self.enabled:
            return
        self._step += 1
        info = {"step": self._step, "name": name, "note": note}

        if isinstance(data, np.ndarray):
            info["type"] = "ndarray"
            info["shape"] = str(data.shape)
            info["dtype"] = str(data.dtype)
            info["min"] = f"{float(data.min()):.6g}"
            info["max"] = f"{float(data.max()):.6g}"
            info["mean"] = f"{float(data.mean()):.6g}"
        elif isinstance(data, torch.Tensor):
            info["type"] = "Tensor"
            info["shape"] = str(tuple(data.shape))
            info["dtype"] = str(data.dtype).replace("torch.", "")
            d = data.detach().float()
            info["min"] = f"{d.min().item():.6g}"
            info["max"] = f"{d.max().item():.6g}"
            info["mean"] = f"{d.mean().item():.6g}"
        elif isinstance(data, list):
            info["type"] = "list"
            info["shape"] = f"len={len(data)}"
            info["dtype"] = "-"
            flat = self._flatten_numeric(data)
            if flat:
                info["min"] = f"{min(flat):.6g}"
                info["max"] = f"{max(flat):.6g}"
                info["mean"] = f"{sum(flat)/len(flat):.6g}"
            else:
                info["min"] = info["max"] = info["mean"] = "-"
        elif isinstance(data, dict):
            info["type"] = "dict"
            info["shape"] = f"keys={len(data)}"
            info["dtype"] = "-"
            info["min"] = info["max"] = info["mean"] = "-"
        elif isinstance(data, (int, float)):
            info["type"] = type(data).__name__
            info["shape"] = "scalar"
            info["dtype"] = "-"
            info["min"] = info["max"] = info["mean"] = f"{data}"
        else:
            info["type"] = type(data).__name__
            info["shape"] = "-"
            info["dtype"] = "-"
            info["min"] = info["max"] = info["mean"] = "-"

        self.records.append(info)

    def detail(self, text):
        """Append a free-form text block to the report (shown after the table)."""
        if not self.enabled:
            return
        self.details.append(text)

    def _flatten_numeric(self, lst):
        out = []
        for item in lst:
            if isinstance(item, (int, float)):
                out.append(float(item))
            elif isinstance(item, (list, tuple)):
                out.extend(self._flatten_numeric(item))
        return out

    def _build_report(self):
        parts = []

        if not self.records and not self.details:
            return "No trace records.\n"

        # --- summary table ---
        if self.records:
            cols = ["Step", "Name", "Type", "Shape", "Dtype", "Min", "Max", "Mean", "Note"]
            widths = [len(c) for c in cols]
            rows = []
            for r in self.records:
                row = [
                    str(r["step"]),
                    r["name"],
                    r["type"],
                    r["shape"],
                    r["dtype"],
                    r["min"],
                    r["max"],
                    r["mean"],
                    r["note"],
                ]
                rows.append(row)
                for i, v in enumerate(row):
                    widths[i] = max(widths[i], len(v))

            def fmt_row(vals):
                return " | ".join(v.ljust(widths[i]) for i, v in enumerate(vals))

            sep = "-+-".join("-" * w for w in widths)
            tw = sum(widths) + 3 * (len(widths) - 1)
            parts.append("=" * tw)
            parts.append("  Data Flow Trace  --  Summary Table (1st sample)")
            parts.append("=" * tw)
            parts.append(fmt_row(cols))
            parts.append(sep)
            for row in rows:
                parts.append(fmt_row(row))
            parts.append("=" * tw)

        # --- detail blocks ---
        if self.details:
            parts.append("")
            for block in self.details:
                parts.append(block)

        return "\n".join(parts) + "\n"

    def print_report(self):
        print(self._build_report())

    def save_report(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self._build_report())
        print(f"[Trace] Report saved to: {path}")

    def register_vqvae_hooks(self, model):
        """Register forward hooks on VQ-VAE's key modules."""
        if not self.enabled:
            return []
        handles = []

        def make_hook(layer_name):
            def hook_fn(module, inp, out):
                if isinstance(out, torch.Tensor):
                    self.trace(f"vqvae.{layer_name} output", out, f"[hook] {layer_name}")
                elif isinstance(out, tuple):
                    for idx, o in enumerate(out):
                        if isinstance(o, torch.Tensor):
                            self.trace(f"vqvae.{layer_name} out[{idx}]", o, f"[hook] {layer_name} tuple[{idx}]")
            return hook_fn

        handles.append(model.encoder.register_forward_hook(make_hook("encoder")))
        handles.append(model.quant_conv.register_forward_hook(make_hook("quant_conv")))
        handles.append(model.quantize.register_forward_hook(make_hook("quantize")))
        return handles


# ==============================
#          Utils
# ==============================

def prepare_surface_edge_batch_for_vqvae(surf_ncs, edge_ncs, edgeFace_adj, use_type_flag=False):
    """
    Combine face/edge data into SE VQ-VAE input (N, C, 32, 32)
    - Face: (num_face, 32, 32, 3)
    - Edge: (num_edge, 32, 3) -> expand to (num_edge, 32, 32, 3)
    """
    surf_data = surf_ncs.astype(np.float32)  # (F, 32, 32, 3)
    num_face = len(surf_data)

    edge_data = edge_ncs.astype(np.float32)  # (E, 32, 3)
    edge_expanded = np.tile(edge_data[:, :, np.newaxis, :], (1, 1, 32, 1))  # (E, 32, 32, 3)
    num_edge = len(edge_data)

    # Edge-face correspondence (for subsequent BFS ordering)
    edge_face_pairs = []
    if len(edgeFace_adj) > 0:
        for edge_adj in edgeFace_adj:
            if len(edge_adj) >= 2:
                face1_idx, face2_idx = edge_adj[0], edge_adj[1]
                edge_face_pairs.append((face1_idx, face2_idx))

    if use_type_flag:
        surf_flags = np.zeros((num_face, 32, 32, 1), dtype=np.float32)
        surf_with_flags = np.concatenate([surf_data, surf_flags], axis=-1)
        edge_flags = np.ones((num_edge, 32, 32, 1), dtype=np.float32)
        edge_with_flags = np.concatenate([edge_expanded, edge_flags], axis=-1)
        combined_data = np.concatenate([surf_with_flags, edge_with_flags], axis=0)
        combined_data = combined_data.transpose(0, 3, 1, 2)
    else:
        combined_data = np.concatenate([surf_data, edge_expanded], axis=0)
        combined_data = combined_data.transpose(0, 3, 1, 2)  # (F+E, 3, 32, 32)

    return combined_data, num_face, num_edge, edge_face_pairs


def calculate_tokens_per_element(se_vqvae_model, device):
    """
    Detect the number of tokens per element for SE-VQ; bbox is fixed at 6 (min/max xyz)
    """
    try:
        in_channels = se_vqvae_model.encoder.conv_in.weight.shape[1]
    except Exception:
        in_channels = 3
    se_random_data = np.random.rand(in_channels, 32, 32).astype(np.float32)

    with torch.no_grad():
        x = torch.tensor(se_random_data, dtype=torch.float32).unsqueeze(0).to(device)
        h = se_vqvae_model.encoder(x)
        h = se_vqvae_model.quant_conv(h)
        _, _, indices = se_vqvae_model.quantize(h)
        token_indices = (
            indices[2] if isinstance(indices, tuple) and len(indices) > 2
            else indices[0] if isinstance(indices, tuple)
            else indices
        )
        se_tokens = int(token_indices.numel())

    bbox_tokens = 6
    return se_tokens, bbox_tokens

def dfs_face_ordering_from_core(edge_face_pairs, num_faces):
    """
    Face ordering strategy: Depth-First Search (DFS), prioritizing low-degree neighbors
    1. Find the face with highest degree as starting point
    2. Execute DFS from starting point, prioritizing unvisited neighbors with lowest degree
    3. Generate face sequence in visit order
    
    Returns:
        face_order: [face_idx0, face_idx1, ...]  New ordering of original faces
        face_position_map: {original_face_idx: new_position}
    """
    # Build graph & degrees
    nbrs = [set() for _ in range(num_faces)]
    for f1, f2 in edge_face_pairs:
        if 0 <= f1 < num_faces and 0 <= f2 < num_faces and f1 != f2:
            nbrs[f1].add(f2); nbrs[f2].add(f1)
    deg = [len(n) for n in nbrs]

    visited = [False]*num_faces
    face_order = []

    # Use (degree descending, id ascending) as component starting point selection order
    seeds = sorted(range(num_faces), key=lambda x: (-deg[x], x))
    
    def dfs(u):
        """Depth-first search, prioritizing low-degree neighbors"""
        visited[u] = True
        face_order.append(u)
        
        # Get unvisited neighbors, sort by (degree ascending, id ascending)
        unvisited_neighbors = [v for v in nbrs[u] if not visited[v]]
        unvisited_neighbors.sort(key=lambda x: (deg[x], x))
        
        # Recursively visit each neighbor
        for v in unvisited_neighbors:
            if not visited[v]:  # Double check to prevent being visited during ordering
                dfs(v)
    
    # Execute DFS for each connected component
    for s in seeds:
        if not visited[s]:
            dfs(s)

    face_position_map = {f:i for i,f in enumerate(face_order)}
    return face_order, face_position_map


def lexicographic_edge_ordering(edge_face_pairs):
    """
    Edge ordering strategy: (max, min) lexicographic ordering
    1. For each edge's face pair (f1, f2), use 0-based position indices after ordering
    2. Calculate sort key: Key = (max(f1, f2), min(f1, f2))
    3. Sort in ascending order by sort key
    
    Args:
        edge_face_pairs: [(f1, f2), ...] Face position pairs (0-based positions after DFS ordering)
    
    Returns:
        edge_order: [eidx0, eidx1, ...]  New ordering of edges
        ordered_edge_face_pairs: [(f1, f2), ...] Aligned with edge_order
    """
    # Validate input: check if face indices are in valid range starting from 0
    if len(edge_face_pairs) > 0:
        all_face_indices = set()
        for pair in edge_face_pairs:
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                all_face_indices.add(pair[0])
                all_face_indices.add(pair[1])
        
        if len(all_face_indices) > 0:
            min_face_idx = min(all_face_indices)
            max_face_idx = max(all_face_indices)
            # Assert: face indices should start from 0 and be continuous (or at least in reasonable range)
            assert min_face_idx >= 0, f"Face index must >= 0, current minimum: {min_face_idx}"
            assert max_face_idx <= 50, f"Face index exceeds reasonable range, current maximum: {max_face_idx}"
    
    # Build edge ordering information
    edge_sort_info = []
    
    for eidx, pair in enumerate(edge_face_pairs):
        if not (isinstance(pair, (list, tuple)) and len(pair) >= 2):
            continue
        
        f1, f2 = pair[0], pair[1]
        
        # Calculate sort key: (max_idx, min_idx)
        # Use 0-based face position indices
        max_idx = max(f1, f2)
        min_idx = min(f1, f2)
        
        sort_key = (max_idx, min_idx)
        edge_sort_info.append((sort_key, eidx, pair))
    
    # Sort by sort key in ascending order
    edge_sort_info.sort(key=lambda x: x[0])
    
    # Extract sorting results
    edge_order = [item[1] for item in edge_sort_info]
    ordered_edge_face_pairs = [item[2] for item in edge_sort_info]
    
    return edge_order, ordered_edge_face_pairs

# ==============================
#    Preprocessor (group version)
# ==============================

class ARDataPreprocessor:
    def __init__(self, data_list, se_vqvae_model, args, tracer=None):
        self.data_list = data_list
        self.se_vqvae_model = se_vqvae_model
        self.args = args
        self.device = next(se_vqvae_model.parameters()).device

        self.tracer = tracer or DataTracer(enabled=False)

        # Detect token count (before registering hooks to avoid stray trace records)
        self.se_tokens_per_element, self.bbox_tokens_per_element = calculate_tokens_per_element(
            se_vqvae_model, self.device
        )

        self._hook_handles = self.tracer.register_vqvae_hooks(se_vqvae_model)
        self._trace_done_pending = False

        # Read data list (path collection)
        with open(data_list, 'rb') as f:
            ds = pickle.load(f)
        self.train_paths = ds['train']
        self.val_paths = ds.get('val', [])
        self.test_paths = ds.get('test', [])

        # Vocabulary/offsets
        self.face_index_size = 50
        self.se_codebook_size = 8192
        self.bbox_index_size = 2048
        self.special_token_size = 4

        self.face_index_offset = 0
        self.se_token_offset = self.face_index_offset + self.face_index_size
        self.bbox_token_offset = self.se_token_offset + self.se_codebook_size

        self.vocab_size = (
            self.face_index_size + self.se_codebook_size + self.bbox_index_size + self.special_token_size
        )
        special_token_offset = self.bbox_token_offset + self.bbox_index_size
        self.START_TOKEN = special_token_offset
        self.SEP_TOKEN = special_token_offset + 1
        self.END_TOKEN = special_token_offset + 2
        self.PAD_TOKEN = special_token_offset + 3

        self.group_cache = []
        self._process_all_data()

    # ---------- Main processing ----------
    def _process_all_data(self):
        for path in tqdm(self.train_paths, desc="Processing train"):
            g = self._process_single_cad(path, 'train')
            if g:
                self.group_cache.append(('train', g))
                self._finish_trace_if_needed()
        for path in tqdm(self.val_paths, desc="Processing val"):
            g = self._process_single_cad(path, 'val')
            if g:
                self.group_cache.append(('val', g))
                self._finish_trace_if_needed()
        for path in tqdm(self.test_paths, desc="Processing test"):
            g = self._process_single_cad(path, 'test')
            if g:
                self.group_cache.append(('test', g))
                self._finish_trace_if_needed()

    def _finish_trace_if_needed(self):
        """Print and save the trace report after the first successful sample, then disable."""
        if not getattr(self, '_trace_done_pending', False) and not self.tracer.enabled:
            return
        if self.tracer.enabled:
            self.tracer.enabled = False
        self._trace_done_pending = False
        self.tracer.print_report()
        trace_path = getattr(self.args, 'trace_output', 'data/trace_report.txt')
        self.tracer.save_report(trace_path)
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

    # ---------- Encode a rotation version ----------
    def _encode_single_rotation(
        self,
        surf_ncs, edge_ncs,
        surf_bbox_wcs, edge_bbox_wcs,
        edgeFace_adj,
        rotation_angle
    ):
        """
        Returns: (tokens:list[int], attention_mask:list[int])
        """
        t = self.tracer

        # Deep copy
        current_surf_ncs = surf_ncs.copy()
        current_edge_ncs = edge_ncs.copy()
        current_surf_bbox_wcs = surf_bbox_wcs.copy()
        current_edge_bbox_wcs = edge_bbox_wcs.copy()
        current_edgeFace_adj = [adj[:] for adj in edgeFace_adj]

        # Rotation
        if rotation_angle % 360 != 0:
            surfpos_corners = bbox_corners(current_surf_bbox_wcs)
            edgepos_corners = bbox_corners(current_edge_bbox_wcs)
            surfpos_corners = rotate_axis(surfpos_corners, rotation_angle, 'z', normalized=True)
            edgepos_corners = rotate_axis(edgepos_corners, rotation_angle, 'z', normalized=True)
            current_surf_ncs = rotate_axis(current_surf_ncs, rotation_angle, 'z', normalized=False)
            current_edge_ncs = rotate_axis(current_edge_ncs, rotation_angle, 'z', normalized=False)
            current_surf_bbox_wcs = get_bbox(surfpos_corners).reshape(len(current_surf_bbox_wcs), 6)
            current_edge_bbox_wcs = get_bbox(edgepos_corners).reshape(len(current_edge_bbox_wcs), 6)
            t.trace("rotated_surf_ncs", current_surf_ncs, f"after rotate {rotation_angle} deg")
            t.trace("rotated_edge_ncs", current_edge_ncs, f"after rotate {rotation_angle} deg")
            t.trace("rotated_surf_bbox", current_surf_bbox_wcs, f"bbox after rotate {rotation_angle} deg")
            t.trace("rotated_edge_bbox", current_edge_bbox_wcs, f"bbox after rotate {rotation_angle} deg")

        # VQ encoding: prepare face and edge data
        se_data, num_face, num_edge, edge_face_pairs = prepare_surface_edge_batch_for_vqvae(
            current_surf_ncs, current_edge_ncs, current_edgeFace_adj, use_type_flag=False
        )
        t.trace("se_data", se_data, f"VQ-VAE input (F={num_face}+E={num_edge}, C, 32, 32)")
        t.trace("edge_face_pairs", edge_face_pairs, "extracted edge-face pairs")
        
        # ===== New three-stage ordering strategy =====
        # Stage 1: Face ordering (DFS from edge to core)
        face_order, face_position_map = dfs_face_ordering_from_core(edge_face_pairs, num_face)
        t.trace("face_order", face_order, "DFS face visit order")
        
        # Rearrange face-related data according to new face ordering
        current_surf_ncs = current_surf_ncs[face_order]
        current_surf_bbox_wcs = current_surf_bbox_wcs[face_order]
        
        # Update face portion in SE data (first num_face elements)
        se_data[:num_face] = se_data[face_order]
        t.trace("se_data_reordered", se_data, "se_data after DFS face reorder")
        
        # Update edge-face adjacency relations using new face positions (0-based indices)
        updated_edge_face_pairs = []
        for f1, f2 in edge_face_pairs:
            new_f1 = face_position_map[f1]
            new_f2 = face_position_map[f2]
            updated_edge_face_pairs.append((new_f1, new_f2))
        edge_face_pairs = updated_edge_face_pairs
        t.trace("remapped_edge_face_pairs", edge_face_pairs, "edge-face pairs with new face indices")
        
        # Stage 2: Edge ordering (lexicographic: max-min ordering, using 0-based indices)
        edge_order, ordered_edge_face_pairs = lexicographic_edge_ordering(edge_face_pairs)
        t.trace("edge_order", edge_order, "MAX-IDX-A edge ordering")
        t.trace("ordered_edge_face_pairs", ordered_edge_face_pairs, "edge-face pairs after edge sort")
        
        # Stage 3: Face index cyclic offset (re-indexing at the end)
        max_faces = self.args.max_face
        num_faces = num_face
        N = min(self.face_index_size, max_faces)  # This is 50
        r = random.randint(0, N - 1) if N > 0 else 0
        face_index_map = {i: (i + r) % N for i in range(num_faces)} if N > 0 else {i: i for i in range(num_faces)}
        t.trace("face_index_map", list(face_index_map.values()), f"re-index offset r={r}, N={N}")

        # bbox quantization (note empty check)
        surf_bbox_indices = []
        edge_bbox_indices = []
        if len(current_surf_bbox_wcs) > 0:
            surf_bbox_scaled = np.array(current_surf_bbox_wcs) * float(self.args.scale)
            t.trace("surf_bbox_scaled", surf_bbox_scaled, f"face bbox * scale={self.args.scale}")
            surf_bbox_indices = quantize_bbox(
                surf_bbox_scaled,
                num_tokens=self.bbox_index_size
            ).tolist()
            t.trace("surf_bbox_indices", surf_bbox_indices, f"face position tokens, range [0,{self.bbox_index_size-1}]")
        if len(current_edge_bbox_wcs) > 0:
            edge_bbox_scaled = np.array(current_edge_bbox_wcs) * float(self.args.scale)
            t.trace("edge_bbox_scaled", edge_bbox_scaled, f"edge bbox * scale={self.args.scale}")
            edge_bbox_indices = quantize_bbox(
                edge_bbox_scaled,
                num_tokens=self.bbox_index_size
            ).tolist()
            t.trace("edge_bbox_indices", edge_bbox_indices, f"edge position tokens, range [0,{self.bbox_index_size-1}]")

        # SE encoding (hooks auto-trace encoder/quant_conv/quantize)
        se_indices = []
        if len(se_data) > 0:
            se_tensor = torch.tensor(se_data, dtype=torch.float32).to(self.device)
            t.trace("se_tensor", se_tensor, "VQ-VAE input tensor on device")
            with torch.no_grad():
                h = self.se_vqvae_model.encoder(se_tensor)
                h = self.se_vqvae_model.quant_conv(h)
                _, _, indices = self.se_vqvae_model.quantize(h)
                token_indices = (
                    indices[2] if isinstance(indices, tuple) and len(indices) > 2
                    else indices[0] if isinstance(indices, tuple)
                    else indices
                )
                t.trace("se_token_indices_raw", token_indices, "codebook indices before reshape")
                se_indices = token_indices.cpu().reshape(len(se_data), self.se_tokens_per_element).tolist()
                t.trace("se_indices", se_indices, f"geometry tokens ({self.se_tokens_per_element} per element)")

        surface_indices = se_indices[:num_face] if se_indices else []
        edge_indices = se_indices[num_face:num_face + num_edge] if se_indices else []
        ordered_edge_indices = [edge_indices[i] for i in range(len(edge_indices))] if edge_indices else []
        if edge_indices and len(edge_order) == len(edge_indices):
            ordered_edge_indices = [edge_indices[i] for i in edge_order]
        ordered_edge_bbox_indices = [edge_bbox_indices[i] for i in edge_order] if edge_bbox_indices and len(edge_order) > 0 else []

        t.trace("surface_indices", surface_indices, f"face geometry tokens ({num_face} faces)")
        t.trace("ordered_edge_indices", ordered_edge_indices, f"edge geometry tokens ({num_edge} edges, sorted)")
        t.trace("ordered_edge_bbox_indices", ordered_edge_bbox_indices, f"edge position tokens (sorted)")

        # Concatenate tokens
        tokens, attention_mask = [], []
        tokens.append(self.START_TOKEN); attention_mask.append(1)

        # Faces
        for i in range(num_face):
            if i < len(surf_bbox_indices):
                for bbox_idx in surf_bbox_indices[i]:  # 6
                    tokens.append(self.bbox_token_offset + int(bbox_idx)); attention_mask.append(1)
            if i < len(surface_indices):
                for surf_idx in surface_indices[i]:
                    tokens.append(self.se_token_offset + int(surf_idx)); attention_mask.append(1)
            tokens.append(self.face_index_offset + face_index_map[i]); attention_mask.append(1)

        tokens.append(self.SEP_TOKEN); attention_mask.append(1)

        # Edges
        for k, (face_pair) in enumerate(ordered_edge_face_pairs):
            src, dst = face_pair
            tokens.append(self.face_index_offset + face_index_map[src]); attention_mask.append(1)
            tokens.append(self.face_index_offset + face_index_map[dst]); attention_mask.append(1)

            if k < len(ordered_edge_bbox_indices):
                for bbox_idx in ordered_edge_bbox_indices[k]:  # 6
                    tokens.append(self.bbox_token_offset + int(bbox_idx)); attention_mask.append(1)

            if k < len(ordered_edge_indices):
                for eidx in ordered_edge_indices[k]:
                    tokens.append(self.se_token_offset + int(eidx)); attention_mask.append(1)

        tokens.append(self.END_TOKEN); attention_mask.append(1)

        t.trace("final_tokens", tokens, f"complete sequence (vocab offsets: face_idx=0, geo={self.se_token_offset}, pos={self.bbox_token_offset}, special={self.START_TOKEN}+)")
        t.trace("attention_mask", attention_mask, "attention mask (all 1s)")

        # ===== Detail: show actual token values for face[0], edge[0], blocks, and full sequence =====
        if t.enabled and len(t.details) == 0:
            sep_line = "-" * 80
            lines = []
            lines.append("=" * 80)
            lines.append("  Data Flow Trace  --  Token Details (1st sample, face[0] & edge[0])")
            lines.append("=" * 80)

            # --- Vocab layout ---
            lines.append("")
            lines.append(f"[Vocab Layout]")
            lines.append(f"  Face Index tokens : offset={self.face_index_offset}, range [0, {self.face_index_size - 1}]")
            lines.append(f"  Geometry tokens   : offset={self.se_token_offset}, codebook size={self.se_codebook_size}")
            lines.append(f"  Position tokens   : offset={self.bbox_token_offset}, quantization levels={self.bbox_index_size}")
            lines.append(f"  Special tokens    : START={self.START_TOKEN}, SEP={self.SEP_TOKEN}, END={self.END_TOKEN}, PAD={self.PAD_TOKEN}")
            lines.append(f"  Total vocab size  : {self.vocab_size}")

            # --- Face[0] three token types (raw, before offset) ---
            lines.append("")
            lines.append(sep_line)
            lines.append(f"[Face 0] -- Three Token Types (raw codebook/quantized indices, before vocab offset)")
            lines.append(sep_line)
            if surface_indices:
                lines.append(f"  Geometry tokens (codebook indices) : {surface_indices[0]}")
            if surf_bbox_indices:
                lines.append(f"  Position tokens (quantized bbox)   : {surf_bbox_indices[0]}")
            lines.append(f"  Topology token  (face index)       : {face_index_map[0]}")

            # --- Face block[0] (with offset) ---
            lines.append("")
            lines.append(f"[Face Block 0] -- Assembled block (with vocab offsets applied)")
            lines.append(f"  Structure: [6 position tokens] + [4 geometry tokens] + [1 face index token] = 11 tokens")
            fb = []
            if surf_bbox_indices:
                for idx in surf_bbox_indices[0]:
                    fb.append(self.bbox_token_offset + int(idx))
            if surface_indices:
                for idx in surface_indices[0]:
                    fb.append(self.se_token_offset + int(idx))
            fb.append(self.face_index_offset + face_index_map[0])
            pos_part = fb[:6] if len(fb) >= 6 else fb
            geo_part = fb[6:10] if len(fb) >= 10 else []
            topo_part = fb[10:11] if len(fb) >= 11 else []
            lines.append(f"  Position (6) : {pos_part}")
            lines.append(f"  Geometry (4) : {geo_part}")
            lines.append(f"  Topology (1) : {topo_part}")
            lines.append(f"  Full block   : {fb}")

            # --- Edge[0] three token types (raw, before offset) ---
            lines.append("")
            lines.append(sep_line)
            lines.append(f"[Edge 0] -- Three Token Types (raw codebook/quantized indices, before vocab offset)")
            lines.append(sep_line)
            if ordered_edge_indices:
                lines.append(f"  Geometry tokens (codebook indices)     : {ordered_edge_indices[0]}")
            if ordered_edge_bbox_indices:
                lines.append(f"  Position tokens (quantized bbox)       : {ordered_edge_bbox_indices[0]}")
            if ordered_edge_face_pairs:
                src0, dst0 = ordered_edge_face_pairs[0]
                lines.append(f"  Topology tokens (adjacent face indices): [{face_index_map[src0]}, {face_index_map[dst0]}]")

            # --- Edge block[0] (with offset) ---
            lines.append("")
            lines.append(f"[Edge Block 0] -- Assembled block (with vocab offsets applied)")
            lines.append(f"  Structure: [2 face index tokens] + [6 position tokens] + [4 geometry tokens] = 12 tokens")
            eb = []
            if ordered_edge_face_pairs:
                src0, dst0 = ordered_edge_face_pairs[0]
                eb.append(self.face_index_offset + face_index_map[src0])
                eb.append(self.face_index_offset + face_index_map[dst0])
            if ordered_edge_bbox_indices:
                for idx in ordered_edge_bbox_indices[0]:
                    eb.append(self.bbox_token_offset + int(idx))
            if ordered_edge_indices:
                for idx in ordered_edge_indices[0]:
                    eb.append(self.se_token_offset + int(idx))
            topo_e = eb[:2] if len(eb) >= 2 else eb
            pos_e = eb[2:8] if len(eb) >= 8 else []
            geo_e = eb[8:12] if len(eb) >= 12 else []
            lines.append(f"  Topology (2) : {topo_e}")
            lines.append(f"  Position (6) : {pos_e}")
            lines.append(f"  Geometry (4) : {geo_e}")
            lines.append(f"  Full block   : {eb}")

            # --- Final sequence ---
            lines.append("")
            lines.append(sep_line)
            lines.append(f"[Final Holistic Token Sequence]")
            lines.append(sep_line)
            lines.append(f"  Length: {len(tokens)} tokens")
            lines.append(f"  Format: [START] + {num_face} face blocks + [SEP] + {num_edge} edge blocks + [END]")
            lines.append(f"         = 1 + {num_face}*11 + 1 + {num_edge}*12 + 1 = {1 + num_face*11 + 1 + num_edge*12 + 1}")
            lines.append(f"  Tokens: {tokens}")

            lines.append("=" * 80)
            t.details.append("\n".join(lines))

        return tokens, attention_mask

    # ---------- Process single CAD ----------
    def _process_single_cad(self, path, split='train'):
        try:
            with open(path, 'rb') as f:
                cad = pickle.load(f)

            # Basic fields
            surf_ncs = np.array(cad.get('surf_ncs', []), dtype=np.float32)       # (F,32,32,3)
            edge_ncs = np.array(cad.get('edge_ncs', []), dtype=np.float32)       # (E,32,3)
            edge_bbox_wcs = np.array(cad.get('edge_bbox_wcs', []), dtype=np.float32)  # (E,6)
            surf_bbox_wcs = np.array(cad.get('surf_bbox_wcs', []), dtype=np.float32)  # (F,6)
            edgeFace_adj = cad.get('edgeFace_adj', [])
            faceEdge_adj = cad.get('faceEdge_adj', None)  # list[list[edge_idx]]

            # -- Trace: raw input data --
            t = self.tracer
            t.trace("surf_ncs", surf_ncs, "face UV samples (raw)")
            t.trace("edge_ncs", edge_ncs, "edge U samples (raw)")
            t.trace("surf_bbox_wcs", surf_bbox_wcs, "face bboxes [xmin,ymin,zmin,xmax,ymax,zmax]")
            t.trace("edge_bbox_wcs", edge_bbox_wcs, "edge bboxes [xmin,ymin,zmin,xmax,ymax,zmax]")
            t.trace("edgeFace_adj", edgeFace_adj, "edge-face adjacency list")
            t.trace("faceEdge_adj", faceEdge_adj, "face-edge adjacency list")

            # 1) Empty check
            if len(surf_ncs) == 0 or len(edge_ncs) == 0:
                return None

            # 2) Upper limit filtering
            if len(surf_ncs) > int(self.args.max_face):
                return None
            if len(edge_ncs) > int(self.args.max_edge):
                return None


            # Filter out faces that are too close to each other
            threshold_value = 0.05
            scaled_value = 3
            
            surf_bbox = surf_bbox_wcs * scaled_value

            _surf_bbox_ = surf_bbox.reshape(len(surf_bbox), 2, 3)
            non_repeat = _surf_bbox_[:1]
            for bbox in _surf_bbox_:
                diff = np.max(np.max(np.abs(non_repeat - bbox), -1), -1)
                same = diff < threshold_value
                if same.sum() >= 1:
                    continue # Duplicate value
                else:
                    non_repeat = np.concatenate([non_repeat, bbox[np.newaxis,:,:]], 0)
            if len(non_repeat) != len(_surf_bbox_):
                return

            # Filter out edges that are too close to each other
            se_bbox = []
            for adj in faceEdge_adj:
                if len(edge_bbox_wcs[adj]) == 0: 
                    return
                se_bbox.append(edge_bbox_wcs[adj] * scaled_value)

            for bbb in se_bbox:
                _edge_bbox_ = bbb.reshape(len(bbb), 2, 3)
                non_repeat = _edge_bbox_[:1]
                for bbox in _edge_bbox_:
                    diff = np.max(np.max(np.abs(non_repeat - bbox), -1), -1)
                    same = diff < threshold_value
                    if same.sum() >= 1:
                        continue # Duplicate value
                    else:
                        non_repeat = np.concatenate([non_repeat, bbox[np.newaxis,:,:]], 0)
                if len(non_repeat) != len(_edge_bbox_):
                    return None

            # Rotation angle set: only train uses data augmentation, val and test only use original data
            if split == 'train' and bool(self.args.aug):
                rotation_angles = [0, 90, 180, 270]
            else:
                rotation_angles = [0]

            # Determine save format based on split
            if split == 'train':
                # train saves group format (original + augmented)
                group = {
                    'original': None,
                    'augmented': []
                }
                
                for rot in rotation_angles:
                    tokens, attn = self._encode_single_rotation(
                        surf_ncs, edge_ncs,
                        surf_bbox_wcs, edge_bbox_wcs,
                        edgeFace_adj,
                        rotation_angle=rot
                    )
                    item = {'input_ids': tokens, 'attention_mask': attn}
                    if rot == 0:
                        group['original'] = item
                        if self.tracer.enabled:
                            self.tracer.enabled = False
                            self._trace_done_pending = True
                    else:
                        group['augmented'].append(item)
                
                # If exception causes original to be empty, don't return
                if group['original'] is None:
                    return None
                
                return group
            else:
                # val and test only save original data
                tokens, attn = self._encode_single_rotation(
                    surf_ncs, edge_ncs,
                    surf_bbox_wcs, edge_bbox_wcs,
                    edgeFace_adj,
                    rotation_angle=0
                )
                if self.tracer.enabled:
                    self.tracer.enabled = False
                    self._trace_done_pending = True
                item = {'input_ids': tokens, 'attention_mask': attn}
                group = {'original': item}
                return group

        except Exception as e:
            print(f"[WARN] Error processing {path}: {e}")
            return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_list', type=str, default='data/abc_data_split_6bit.pkl', help='Path to pkl with train/val/test paths')
    parser.add_argument('--output_file', type=str, default='data/abc_sequences.pkl', help='Output pickle file (group format)')
    parser.add_argument('--vqvae_se_weight', type=str, default='checkpoint/se/abc_se_vqvae_epoch.pt', help='Pre-trained face/edge VQ-VAE model weight path')
    parser.add_argument('--max_face', type=int, default=50)
    parser.add_argument('--max_edge', type=int, default=150)
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--aug', default=True, type=bool, help='Whether to save rotation augmentation (90/180/270)')
    parser.add_argument("--gpu", type=int, nargs='+', default=[0], help="GPU IDs to use")
    parser.add_argument('--trace', action='store_true', default=False, help='Trace data flow of the 1st sample (print + save report)')
    parser.add_argument('--trace_output', type=str, default='data/trace_report.txt', help='Path to save trace report')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.gpu))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    se_vqvae_model = load_se_vqvae_model(args.vqvae_se_weight, False, 'abc', device)

    tracer = DataTracer(enabled=args.trace)
    processor = ARDataPreprocessor(args.data_list, se_vqvae_model, args, tracer=tracer)

    # Split cache to each split
    train_groups, val_groups, test_groups = [], [], []
    for split, group in processor.group_cache:
        if split == 'train': train_groups.append(group)
        elif split == 'val': val_groups.append(group)
        elif split == 'test': test_groups.append(group)

    # Package output (including metadata, ARData will use)
    output_data = {
        'train': train_groups,
        'val': val_groups,
        'test': test_groups,
        'vocab_size': processor.vocab_size,
        'special_token_size': processor.special_token_size,
        'face_index_size': processor.face_index_size,
        'se_codebook_size': processor.se_codebook_size,
        'bbox_index_size': processor.bbox_index_size,
        'face_index_offset': processor.face_index_offset,
        'se_token_offset': processor.se_token_offset,
        'bbox_token_offset': processor.bbox_token_offset,
        'se_tokens_per_element': processor.se_tokens_per_element,
        'bbox_tokens_per_element': processor.bbox_tokens_per_element,
        'special_tokens': {
            'START_TOKEN': processor.START_TOKEN,
            'SEP_TOKEN': processor.SEP_TOKEN,
            'END_TOKEN': processor.END_TOKEN,
            'PAD_TOKEN': processor.PAD_TOKEN,
        }
    }

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'wb') as f:
        pickle.dump(output_data, f)

    print(f"[DONE] Saved groups -> {args.output_file}")
    print(f"  train: {len(train_groups)} | val: {len(val_groups)} | test: {len(test_groups)}")
    print(f"  aug enabled: {bool(args.aug)}")


if __name__ == "__main__":
    main()
