import argparse
import gc
import os
import pickle

import numpy as np
from tqdm import tqdm


def load_split_paths(data_list, split):
    with open(data_list, "rb") as f:
        split_data = pickle.load(f)
    if split not in split_data:
        raise KeyError(f"Split '{split}' not found in {data_list}")
    return split_data[split]


def build_cache(data_paths, field_name, empty_shape, output_path, dtype):
    arrays = []
    for path in tqdm(data_paths, desc=f"Loading {field_name}"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        if field_name in data:
            arr = data[field_name]
            if dtype:
                arr = arr.astype(dtype, copy=False)
            arrays.append(arr)

    if arrays:
        cached = np.vstack(arrays)
    else:
        cached = np.array([], dtype=dtype or np.float32).reshape(empty_shape)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(cached, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved {field_name} cache to {output_path}")
    print(f"  shape={cached.shape}, dtype={cached.dtype}, size={cached.nbytes / 1024**3:.2f} GiB")

    del arrays
    del cached
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Build VQ-VAE validation caches from split pkl files.")
    parser.add_argument("--data_list", required=True, help="Path to split pkl containing train/val/test file lists.")
    parser.add_argument("--output_surface", required=True, help="Output pkl for stacked val surf_ncs arrays.")
    parser.add_argument("--output_edge", required=True, help="Output pkl for stacked val edge_ncs arrays.")
    parser.add_argument("--split", default="val", help="Split name to cache.")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64", "keep"],
                        help="Dtype for cached arrays. Use 'keep' to preserve source dtype.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip outputs that already exist.")
    args = parser.parse_args()

    dtype = None if args.dtype == "keep" else np.dtype(args.dtype)
    data_paths = load_split_paths(args.data_list, args.split)
    print(f"Found {len(data_paths)} files in split '{args.split}'")

    if args.skip_existing and os.path.exists(args.output_surface):
        print(f"Surface cache exists, skipping: {args.output_surface}")
    else:
        build_cache(data_paths, "surf_ncs", (0, 32, 32, 3), args.output_surface, dtype)

    if args.skip_existing and os.path.exists(args.output_edge):
        print(f"Edge cache exists, skipping: {args.output_edge}")
    else:
        build_cache(data_paths, "edge_ncs", (0, 32, 3), args.output_edge, dtype)


if __name__ == "__main__":
    main()
