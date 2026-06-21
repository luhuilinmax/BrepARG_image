import argparse
import pickle
import random
from pathlib import Path


def stem(path):
    return Path(path).stem


def main():
    parser = argparse.ArgumentParser(description='Build a mini split from rendered sample stems and original CAD split pkl.')
    parser.add_argument('--render_index', type=str, required=True)
    parser.add_argument('--source_split', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--train_ratio', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    with open(args.render_index, 'rb') as f:
        render_data = pickle.load(f)
    render_stems = set(render_data['renders'].keys())

    with open(args.source_split, 'rb') as f:
        split_data = pickle.load(f)

    matched = []
    for split_name in ('train', 'val', 'test'):
        for path in split_data.get(split_name, []):
            if stem(path) in render_stems:
                matched.append(path)

    matched = sorted(set(matched))
    matched_stems = {stem(p) for p in matched}
    missing = sorted(render_stems - matched_stems)

    rng = random.Random(args.seed)
    rng.shuffle(matched)
    train_count = int(len(matched) * args.train_ratio)
    train_paths = matched[:train_count]
    val_paths = matched[train_count:]

    output_data = {
        'train': train_paths,
        'val': val_paths,
        'test': [],
        'metadata': {
            'render_index': args.render_index,
            'source_split': args.source_split,
            'matched_count': len(matched),
            'missing_render_stems': missing,
        }
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)

    print(f'Saved: {output_path}')
    print(f'matched: {len(matched)}')
    print(f'train: {len(train_paths)} | val: {len(val_paths)} | test: 0')
    print(f'missing_render_stems: {len(missing)}')
    if missing:
        print('first_missing:', missing[0])


if __name__ == '__main__':
    main()
