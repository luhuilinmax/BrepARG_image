import os
import pickle
import argparse
from tqdm import tqdm
from hashlib import sha256
from convert_utils import *


def resolve_output_path(args):
    if args.output:
        return args.output
    if args.option == "deepcad":
        return f"data/deepcad_data_split_{args.bit}bit.pkl"
    if args.option == "abc":
        return f"data/abc_data_split_{args.bit}bit.pkl"
    return f"data/furniture_data_split_{args.bit}bit.pkl"


def load_input_paths(input_pkl_path):
    if os.path.isdir(input_pkl_path):
        pkl_paths = []
        for root, _, files in os.walk(input_pkl_path):
            for name in files:
                if name.endswith(".pkl"):
                    pkl_paths.append(os.path.join(root, name))
        pkl_paths = sorted(pkl_paths)
        print(f"检测到目录输入，递归找到 {len(pkl_paths)} 个 pkl 文件")
        return pkl_paths

    with open(input_pkl_path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict) and "train" in data and isinstance(data["train"], list):
        print(f"检测到 split pkl，使用 train 列表，共 {len(data['train'])} 条")
        return data["train"]

    if isinstance(data, list):
        print(f"检测到路径列表 pkl，共 {len(data)} 条")
        return data

    if isinstance(data, dict) and "surf_wcs" in data:
        print("检测到单样本 pkl，按单文件处理")
        return [input_pkl_path]

    raise ValueError(
        "无法识别输入 pkl 结构。请提供单样本 pkl、包含 train 的 split pkl，或路径列表 pkl。"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pkl", type=str, required=True, help="Input pkl path")
    parser.add_argument("--bit", type=int, default=6, help="Deduplication precision (bit)")
    parser.add_argument(
        "--option",
        type=str,
        choices=["abc", "deepcad", "furniture"],
        default="abc",
        help="Select dataset type [abc/deepcad/furniture] (default: abc)",
    )
    parser.add_argument("--output", type=str, default=None, help="Output split pkl path")
    args = parser.parse_args()

    output_path = resolve_output_path(args)
    input_paths = load_input_paths(args.input_pkl)

    print("\nStart deduplicating CAD files...")
    train_path = []
    unique_hash = set()
    total = 0

    for path_idx, pkl_path in tqdm(
        enumerate(input_paths), total=len(input_paths), desc="Deduplicating train set"
    ):
        total += 1
        try:
            with open(pkl_path, "rb") as file:
                data = pickle.load(file)
        except Exception as e:
            print(f"Failed to read {pkl_path}: {e}")
            continue

        if "surf_wcs" not in data:
            print(f"Missing key 'surf_wcs' in {pkl_path}, skipped")
            continue

        surfs_wcs = data["surf_wcs"]
        surf_hash_total = []
        for surf in surfs_wcs:
            np_bit = real2bit(surf, n_bits=args.bit).reshape(-1, 3)
            surf_hash_total.append(sha256(np_bit.tobytes()).hexdigest())

        data_hash = "_".join(sorted(surf_hash_total))
        prev_len = len(unique_hash)
        unique_hash.add(data_hash)
        if prev_len < len(unique_hash):
            train_path.append(pkl_path)

        if path_idx % 2000 == 0:
            print(f"Deduplication rate: {len(unique_hash) / total:.2%}")

    val_path = list(train_path)

    print("\nTraining set deduplication finished:")
    print(f"  Before: {len(input_paths)} files")
    print(f"  After: {len(train_path)} files")
    print(f"  Duplicates removed: {len(input_paths) - len(train_path)} files")
    if len(input_paths) > 0:
        print(f"  Retention rate: {len(train_path) / len(input_paths):.2%}")

    data_path = {
        "train": train_path,
        "val": val_path,
    }

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "wb") as tf:
        pickle.dump(data_path, tf)

    print("\nFinal dataset statistics:")
    print(f"  Train: {len(train_path)} files")
    print(f"  Val:   {len(val_path)} files (same as train)")
    print(f"  Total: {len(train_path) + len(val_path)} file references")
    print(f"\nResult saved to: {output_path}")


if __name__ == "__main__":
    main()

