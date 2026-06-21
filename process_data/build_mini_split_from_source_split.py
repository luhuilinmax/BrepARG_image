import argparse
import pickle
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Build a mini split directly from the original source split pkl.')
    parser.add_argument('--source_split', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--train_count', type=int, default=90)
    parser.add_argument('--val_count', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--source_subset', type=str, choices=['train', 'val', 'test'], default='train')
    args = parser.parse_args()

    with open(args.source_split, 'rb') as f:
        split_data = pickle.load(f)

    pool = list(split_data.get(args.source_subset, []))
    if len(pool) < args.train_count + args.val_count:
        raise ValueError(
            f"Not enough samples in source subset '{args.source_subset}': "
            f"need {args.train_count + args.val_count}, got {len(pool)}"
        )

    rng = random.Random(args.seed)
    rng.shuffle(pool)

    train_paths = pool[:args.train_count]
    val_paths = pool[args.train_count:args.train_count + args.val_count]

    output_data = {
        'train': train_paths,
        'val': val_paths,
        'test': [],
        'metadata': {
            'source_split': args.source_split,
            'source_subset': args.source_subset,
            'train_count': len(train_paths),
            'val_count': len(val_paths),
            'seed': args.seed,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)

    print(f'Saved: {output_path}')
    print(f"source_subset: {args.source_subset}")
    print(f"train: {len(train_paths)} | val: {len(val_paths)} | test: 0")
    print('first_train:', train_paths[0] if train_paths else None)
    print('first_val:', val_paths[0] if val_paths else None)


if __name__ == '__main__':
    main()
