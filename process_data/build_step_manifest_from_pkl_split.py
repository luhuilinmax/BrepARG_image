import argparse
import pickle
from pathlib import Path


def step_path_from_pkl(pkl_path, step_root):
    stem = Path(pkl_path).stem
    sample_id = stem.split('_')[0]
    step_path = Path(step_root) / sample_id / f'{stem}.step'
    stp_path = Path(step_root) / sample_id / f'{stem}.stp'
    if step_path.exists():
        return step_path
    if stp_path.exists():
        return stp_path
    return None


def main():
    parser = argparse.ArgumentParser(description='Build STEP manifest from a CAD pkl split file.')
    parser.add_argument('--split_pkl', type=str, required=True)
    parser.add_argument('--step_root', type=str, required=True)
    parser.add_argument('--output_manifest', type=str, required=True)
    args = parser.parse_args()

    with open(args.split_pkl, 'rb') as f:
        data = pickle.load(f)

    pkl_paths = list(data.get('train', [])) + list(data.get('val', [])) + list(data.get('test', []))
    manifest_paths = []
    missing = []
    for pkl_path in pkl_paths:
        step_path = step_path_from_pkl(pkl_path, args.step_root)
        if step_path is None:
            missing.append(pkl_path)
        else:
            manifest_paths.append(str(step_path))

    output_path = Path(args.output_manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for path in manifest_paths:
            f.write(path + '\n')

    print(f'Saved manifest: {output_path}')
    print(f'manifest_count: {len(manifest_paths)}')
    print(f'missing_count: {len(missing)}')
    if manifest_paths:
        print('first_manifest:', manifest_paths[0])
    if missing:
        print('first_missing_pkl:', missing[0])


if __name__ == '__main__':
    main()
