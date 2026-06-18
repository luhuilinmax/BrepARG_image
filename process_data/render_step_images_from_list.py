import argparse
import os
import pickle
import tempfile
from pathlib import Path

from render_step_images import load_step_shape, mesh_shape, collect_mesh, write_obj, render_with_blender
from tqdm import tqdm


def load_step_list(path):
    with open(path, 'r', encoding='utf-8') as f:
        items = [Path(line.strip()) for line in f if line.strip()]
    return items


def main():
    parser = argparse.ArgumentParser(description='Render STEP files listed in a manifest into single-view PNG images.')
    parser.add_argument('--input_list', type=str, required=True, help='Text file containing one STEP path per line')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save rendered PNGs')
    parser.add_argument('--index_file', type=str, default='', help='Optional pickle file to save render index')
    parser.add_argument('--mesh_dir', type=str, default='', help='Optional directory to save OBJ mesh cache; defaults to <output_dir>/obj_cache')
    parser.add_argument('--blender_bin', type=str, default='/tmp/blender-3.6.22/blender', help='Blender executable')
    parser.add_argument('--image_size', type=int, default=768, help='Output PNG size')
    parser.add_argument('--samples', type=int, default=128, help='Cycles samples per image')
    parser.add_argument('--elevation', type=float, default=35.0, help='Camera elevation in degrees')
    parser.add_argument('--azimuth_start', type=float, default=45.0, help='Base camera azimuth in degrees')
    parser.add_argument('--azimuth_step', type=float, default=17.0, help='Deterministic azimuth step per sample')
    parser.add_argument('--camera_distance', type=float, default=2.5, help='Camera distance after normalization')
    parser.add_argument('--background', type=float, nargs=3, default=(0.98, 0.98, 0.98), help='Background RGB in 0-1')
    parser.add_argument('--object_color', type=float, nargs=3, default=(0.52, 0.58, 0.64), help='Object RGB in 0-1')
    parser.add_argument('--roughness', type=float, default=0.68, help='Principled BSDF roughness')
    parser.add_argument('--key_light', type=float, default=1200.0, help='Key light energy')
    parser.add_argument('--fill_light', type=float, default=350.0, help='Fill light energy')
    parser.add_argument('--linear_deflection', type=float, default=0.05, help='Meshing linear deflection')
    parser.add_argument('--angular_deflection', type=float, default=0.3, help='Meshing angular deflection')
    parser.add_argument('--max_faces', type=int, default=1000000, help='Skip samples whose triangulated mesh exceeds this many faces; <=0 disables the guard')
    parser.add_argument('--render_timeout_sec', type=int, default=120, help='Per-sample Blender render timeout in seconds; <=0 disables the timeout')
    parser.add_argument('--reuse_obj', action='store_true', help='Reuse existing OBJ cache if present')
    args = parser.parse_args()

    step_files = load_step_list(args.input_list)
    if not step_files:
        raise FileNotFoundError(f'No STEP files listed in: {args.input_list}')

    output_dir = Path(args.output_dir)
    mesh_dir = Path(args.mesh_dir) if args.mesh_dir else output_dir / 'obj_cache'
    render_index = {}
    failures = []

    for idx, step_path in enumerate(tqdm(step_files, desc='Rendering STEP images')):
        stem = step_path.stem
        image_path = output_dir / f'{stem}.png'
        obj_path = mesh_dir / f'{stem}.obj'
        azimuth = (args.azimuth_start + idx * args.azimuth_step) % 360.0
        try:
            if not (args.reuse_obj and obj_path.exists()):
                shape = load_step_shape(step_path)
                mesh_shape(shape, linear_deflection=args.linear_deflection, angular_deflection=args.angular_deflection)
                vertices, faces = collect_mesh(shape)
                if args.max_faces > 0 and len(faces) > args.max_faces:
                    raise RuntimeError(f'Mesh face count {len(faces)} exceeds max_faces={args.max_faces}')
                write_obj(vertices, faces, obj_path)
            render_with_blender(
                obj_path=obj_path,
                output_path=image_path,
                blender_bin=args.blender_bin,
                image_size=args.image_size,
                samples=args.samples,
                elevation_deg=args.elevation,
                azimuth_deg=azimuth,
                camera_distance=args.camera_distance,
                background=args.background,
                object_color=args.object_color,
                roughness=args.roughness,
                key_light=args.key_light,
                fill_light=args.fill_light,
                render_timeout_sec=args.render_timeout_sec,
            )
            render_index[stem] = {
                'image_path': str(image_path),
                'step_path': str(step_path),
                'obj_path': str(obj_path),
                'view': {
                    'elevation': float(args.elevation),
                    'azimuth': float(azimuth),
                    'distance': float(args.camera_distance),
                },
                'render': {
                    'backend': 'blender_cycles',
                    'image_size': int(args.image_size),
                    'samples': int(args.samples),
                },
                'mesh': {
                    'linear_deflection': float(args.linear_deflection),
                    'angular_deflection': float(args.angular_deflection),
                },
            }
        except Exception as exc:
            failures.append({'step_path': str(step_path), 'error': str(exc)})

    if args.index_file:
        index_path = Path(args.index_file)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(index_path, 'wb') as f:
            pickle.dump(
                {
                    'renders': render_index,
                    'failures': failures,
                    'config': {
                        'input_list': args.input_list,
                        'output_dir': args.output_dir,
                        'mesh_dir': str(mesh_dir),
                        'image_size': args.image_size,
                        'samples': args.samples,
                        'elevation': args.elevation,
                        'azimuth_start': args.azimuth_start,
                        'azimuth_step': args.azimuth_step,
                        'camera_distance': args.camera_distance,
                        'background': list(args.background),
                        'object_color': list(args.object_color),
                        'roughness': args.roughness,
                        'key_light': args.key_light,
                        'fill_light': args.fill_light,
                        'linear_deflection': args.linear_deflection,
                        'angular_deflection': args.angular_deflection,
                        'max_faces': args.max_faces,
                        'render_timeout_sec': args.render_timeout_sec,
                        'blender_bin': args.blender_bin,
                    },
                },
                f,
            )

    print(f'Rendered: {len(render_index)}')
    print(f'Failed:   {len(failures)}')
    if failures:
        print('First failure:', failures[0])


if __name__ == '__main__':
    main()
