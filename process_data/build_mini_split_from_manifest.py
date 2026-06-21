import argparse
import os
import pickle
import random
from pathlib import Path


def load_manifest(path):
    with open(path, 'r', encoding='utf-8') as f:
        items = [line.strip() for line in f if line.strip()]
    return items


def main():
    parser = argparse.ArgumentParser(description='Build a mini split pickle from a STEP manifest.')
    parser.add_argument('--manifest', type=str, required=True, help='Text file with one STEP path per line')
    parser.add_argument('--output', type=str, required=True, help='Output split pickle path')
    parser.add_argument('--train_ratio', type=float, default=0.9, help='Train ratio, default 0.9')
    parser.add_argument('--seed', type=int, default=42, help='Shuffle seed')
    args = parser.parse_args()

    paths = load_manifest(args.manifest)
    if not paths:
        raise FileNotFoundError(f'No STEP paths found in manifest: {args.manifest}')

    rng = random.Random(args.seed)
    rng.shuffle(paths)

    train_count = int(len(paths) * args.train_ratio)
    train_paths = paths[:train_count]
    val_paths = paths[train_count:]

    split_data = {
        'train': train_paths,
        'val': val_paths,
        'test': [],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(split_data, f)

    print(f'Saved: {output_path}')
    print(f"train: {len(train_paths)} | val: {len(val_paths)} | test: 0")


if __name__ == '__main__':
    main()
