import argparse
import glob
import pickle
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Merge multiple render_index.pkl files.')
    parser.add_argument('--input_glob', type=str, required=True, help='Glob for render_index.pkl files')
    parser.add_argument('--output', type=str, required=True, help='Merged output pickle path')
    args = parser.parse_args()

    paths = sorted(glob.glob(args.input_glob))
    if not paths:
        raise FileNotFoundError(f'No files matched: {args.input_glob}')

    merged_renders = {}
    merged_failures = []
    configs = []

    for path in paths:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        merged_renders.update(data.get('renders', {}))
        merged_failures.extend(data.get('failures', []))
        configs.append({'source': path, 'config': data.get('config', {})})

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(
            {
                'renders': merged_renders,
                'failures': merged_failures,
                'config': {
                    'merged_from': paths,
                    'batch_configs': configs,
                },
            },
            f,
        )

    print('Merged files:', len(paths))
    print('Merged renders:', len(merged_renders))
    print('Merged failures:', len(merged_failures))
    print('Saved:', output_path)


if __name__ == '__main__':
    main()
