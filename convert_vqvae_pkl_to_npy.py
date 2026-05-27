import argparse
import gc
import os
import pickle

import numpy as np


def load_pickle_data(path, field_name=""):
    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        if field_name and field_name in data:
            data = data[field_name]
        elif "data" in data:
            data = data["data"]
        else:
            keys = ", ".join(str(k) for k in data.keys())
            raise KeyError(f"{path} does not contain '{field_name}' or 'data'. Keys: {keys}")
    return data


def describe_array_like(data):
    if hasattr(data, "shape") and hasattr(data, "dtype"):
        return data.shape, data.dtype, "ndarray"

    if isinstance(data, (list, tuple)):
        if len(data) == 0:
            raise ValueError("Cannot infer shape/dtype from an empty list")
        sample = np.asarray(data[0])
        if sample.dtype == object:
            raise TypeError("List item converted to object dtype; data is not a uniform numeric array")
        return (len(data),) + sample.shape, sample.dtype, type(data).__name__

    raise TypeError(f"Unsupported pickle payload type: {type(data).__name__}")


def write_npy(data, output_path, dtype=None, chunk_size=4096):
    source_shape, source_dtype, source_kind = describe_array_like(data)
    target_dtype = np.dtype(dtype) if dtype is not None else np.dtype(source_dtype)
    tmp_path = output_path + ".tmp.npy"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    if hasattr(data, "shape") and hasattr(data, "dtype"):
        arr = data.astype(target_dtype, copy=False) if data.dtype != target_dtype else data
        np.save(tmp_path, arr)
        expected_shape = arr.shape
    else:
        expected_shape = source_shape
        mmap = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=target_dtype, shape=expected_shape)
        total = len(data)
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            chunk = np.asarray(data[start:end], dtype=target_dtype)
            expected_chunk_shape = (end - start,) + expected_shape[1:]
            if chunk.shape != expected_chunk_shape:
                raise ValueError(
                    f"Non-uniform item shape near [{start}:{end}]: got {chunk.shape}, "
                    f"expected {expected_chunk_shape}"
                )
            mmap[start:end] = chunk
            if start == 0 or end == total or end % (chunk_size * 20) == 0:
                print(f"[write] {output_path}: {end}/{total}", flush=True)
        mmap.flush()
        del mmap

    os.replace(tmp_path, output_path)
    return expected_shape, target_dtype, source_shape, source_dtype, source_kind


def verify_npy(data, output_path, expected_shape, expected_dtype, verify_samples):
    saved = np.load(output_path, mmap_mode="r")
    print(
        f"[done] {output_path}: shape={saved.shape}, dtype={saved.dtype}, "
        f"size={saved.nbytes / 1024**3:.2f} GiB",
        flush=True,
    )

    if saved.shape != expected_shape or saved.dtype != expected_dtype:
        raise RuntimeError(
            f"Verification failed: saved shape/dtype {saved.shape}/{saved.dtype}, "
            f"expected {expected_shape}/{expected_dtype}"
        )

    sample_count = min(verify_samples, len(saved))
    if sample_count > 0:
        sample_indices = np.linspace(0, len(saved) - 1, sample_count, dtype=np.int64)
        for idx in sample_indices:
            original = np.asarray(data[idx], dtype=expected_dtype)
            if not np.array_equal(saved[idx], original):
                raise RuntimeError(f"Verification failed at sample index {idx}")
        print(f"[verify] {output_path}: {sample_count} samples match", flush=True)
    del saved


def convert_one(name, source_path, output_path, field_name, dtype, skip_existing, verify_samples, chunk_size):
    if not source_path:
        print(f"[skip] {name}: no source path provided")
        return
    if skip_existing and os.path.exists(output_path):
        print(f"[skip] {name}: output exists: {output_path}")
        return

    print(f"[load] {name}: {source_path}", flush=True)
    data = load_pickle_data(source_path, field_name=field_name)
    source_shape, source_dtype, source_kind = describe_array_like(data)
    print(f"[source] {name}: kind={source_kind}, shape={source_shape}, dtype={source_dtype}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    print(f"[save] {name}: {output_path}", flush=True)
    expected_shape, expected_dtype, _, _, _ = write_npy(data, output_path, dtype=dtype, chunk_size=chunk_size)
    verify_npy(data, output_path, expected_shape, expected_dtype, verify_samples)

    del data
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Convert VQ-VAE pkl arrays/lists to .npy files for NumPy mmap loading.")
    parser.add_argument("--surface_pkl", default="/workspace/data/deduplicate/abc_data_faces.pkl")
    parser.add_argument("--edge_pkl", default="/workspace/data/deduplicate/abc_data_edges.pkl")
    parser.add_argument("--val_surface_pkl", default="/workspace/data/deduplicate/abc_val_surfaces_cache.pkl")
    parser.add_argument("--val_edge_pkl", default="/workspace/data/deduplicate/abc_val_edges_cache.pkl")
    parser.add_argument("--output_dir", default="/workspace/data1_ly/vqvae_mmap")
    parser.add_argument("--dtype", default="keep", choices=["keep", "float32", "float64"])
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--verify_samples", type=int, default=5)
    parser.add_argument("--chunk_size", type=int, default=4096)
    args = parser.parse_args()

    dtype = None if args.dtype == "keep" else np.dtype(args.dtype)
    outputs = {
        "surface": os.path.join(args.output_dir, "abc_data_faces.npy"),
        "edge": os.path.join(args.output_dir, "abc_data_edges.npy"),
        "val_surface": os.path.join(args.output_dir, "abc_val_surfaces.npy"),
        "val_edge": os.path.join(args.output_dir, "abc_val_edges.npy"),
    }

    convert_one("surface", args.surface_pkl, outputs["surface"], "surf_ncs", dtype, args.skip_existing, args.verify_samples, args.chunk_size)
    convert_one("edge", args.edge_pkl, outputs["edge"], "edge_ncs", dtype, args.skip_existing, args.verify_samples, args.chunk_size)
    convert_one("val_surface", args.val_surface_pkl, outputs["val_surface"], "surf_ncs", dtype, args.skip_existing, args.verify_samples, args.chunk_size)
    convert_one("val_edge", args.val_edge_pkl, outputs["val_edge"], "edge_ncs", dtype, args.skip_existing, args.verify_samples, args.chunk_size)

    print("\nUse these training arguments inside the container:")
    print(f"  --surface_mmap {outputs['surface']}")
    print(f"  --edge_mmap {outputs['edge']}")
    print(f"  --val_surface_mmap {outputs['val_surface']}")
    print(f"  --val_edge_mmap {outputs['val_edge']}")


if __name__ == "__main__":
    main()
