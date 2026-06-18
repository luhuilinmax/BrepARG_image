import argparse
import os
import pickle
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import topods
from tqdm import tqdm


BLENDER_SCRIPT = r'''
import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--image_size", type=int, default=768)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--elevation", type=float, default=35.0)
    parser.add_argument("--azimuth", type=float, default=45.0)
    parser.add_argument("--distance", type=float, default=2.5)
    parser.add_argument("--background", type=float, nargs=3, default=(0.98, 0.98, 0.98))
    parser.add_argument("--object_color", type=float, nargs=3, default=(0.52, 0.58, 0.64))
    parser.add_argument("--roughness", type=float, default=0.68)
    parser.add_argument("--key_light", type=float, default=1200.0)
    parser.add_argument("--fill_light", type=float, default=350.0)
    return parser.parse_args(argv)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.outliner.orphans_purge(do_recursive=True)


def set_render_settings(output_path, image_size, samples):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.preview_samples = min(samples, 16)
    scene.render.resolution_x = image_size
    scene.render.resolution_y = image_size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.render.filepath = str(output_path)
    scene.display_settings.display_device = "sRGB"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0


def setup_world(background):
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    bpy.context.scene.world = world
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    for node in list(nodes):
        nodes.remove(node)
    bg = nodes.new(type="ShaderNodeBackground")
    out = nodes.new(type="ShaderNodeOutputWorld")
    bg.inputs["Color"].default_value = (background[0], background[1], background[2], 1.0)
    bg.inputs["Strength"].default_value = 0.35
    links.new(bg.outputs["Background"], out.inputs["Surface"])


def create_material(object_color, roughness):
    mat = bpy.data.materials.new(name="Clay")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for node in list(nodes):
        nodes.remove(node)
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    out = nodes.new(type="ShaderNodeOutputMaterial")
    bsdf.inputs["Base Color"].default_value = (object_color[0], object_color[1], object_color[2], 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = 0.0
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def load_obj(obj_path):
    bpy.ops.import_scene.obj(filepath=str(obj_path), axis_forward="-Z", axis_up="Y")
    mesh_objs = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
    if not mesh_objs:
        raise RuntimeError(f"No mesh objects imported from {obj_path}")
    return mesh_objs


def normalize_meshes(mesh_objs):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in mesh_objs:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objs[0]
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    coords = []
    for obj in mesh_objs:
        for v in obj.data.vertices:
            coords.append(v.co.copy())
    if not coords:
        raise RuntimeError("Imported mesh has no vertices")

    mins = Vector((min(v.x for v in coords), min(v.y for v in coords), min(v.z for v in coords)))
    maxs = Vector((max(v.x for v in coords), max(v.y for v in coords), max(v.z for v in coords)))
    center = (mins + maxs) * 0.5
    scale = max(maxs.x - mins.x, maxs.y - mins.y, maxs.z - mins.z)
    if scale <= 0:
        raise RuntimeError("Degenerate mesh")
    inv_scale = 1.0 / scale

    for obj in mesh_objs:
        mesh = obj.data
        for v in mesh.vertices:
            v.co = (v.co - center) * inv_scale
        mesh.update()


def add_lighting(key_light, fill_light):
    key_data = bpy.data.lights.new(name="KeyLight", type="AREA")
    key_data.energy = key_light
    key = bpy.data.objects.new(name="KeyLight", object_data=key_data)
    bpy.context.collection.objects.link(key)
    key.location = (2.5, -2.2, 3.2)
    key.data.shape = "RECTANGLE"
    key.data.size = 4.0
    key.data.size_y = 4.0

    fill_data = bpy.data.lights.new(name="FillLight", type="AREA")
    fill_data.energy = fill_light
    fill = bpy.data.objects.new(name="FillLight", object_data=fill_data)
    bpy.context.collection.objects.link(fill)
    fill.location = (-2.0, 1.8, 2.3)
    fill.data.shape = "RECTANGLE"
    fill.data.size = 5.0
    fill.data.size_y = 5.0


def setup_camera(distance, azimuth_deg, elevation_deg):
    cam_data = bpy.data.cameras.new(name="Camera")
    cam_data.lens = 55.0
    cam_data.sensor_width = 36.0
    cam = bpy.data.objects.new(name="Camera", object_data=cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam

    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    cam.location = (
        distance * math.cos(el) * math.cos(az),
        distance * math.cos(el) * math.sin(az),
        distance * math.sin(el),
    )
    direction = Vector((0.0, 0.0, 0.0)) - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    return cam


def assign_material(mesh_objs, material):
    for obj in mesh_objs:
        if obj.data.materials:
            obj.data.materials[0] = material
        else:
            obj.data.materials.append(material)


def main():
    args = parse_args()
    clear_scene()
    setup_world(args.background)
    mesh_objs = load_obj(args.obj_path)
    normalize_meshes(mesh_objs)
    assign_material(mesh_objs, create_material(args.object_color, args.roughness))
    add_lighting(args.key_light, args.fill_light)
    setup_camera(args.distance, args.azimuth, args.elevation)
    set_render_settings(Path(args.output_path), args.image_size, args.samples)
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
'''


def load_step_shape(step_path):
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"Failed to read STEP: {step_path}")
    reader.TransferRoots()
    return reader.OneShape()


def mesh_shape(shape, linear_deflection=0.1, angular_deflection=0.5):
    mesher = BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)
    mesher.Perform()
    if not mesher.IsDone():
        raise RuntimeError("Meshing failed")


def collect_mesh(shape):
    vertices = []
    faces = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    loc = TopLoc_Location()

    while explorer.More():
        face = topods.Face(explorer.Current())
        triangulation = BRep_Tool.Triangulation(face, loc)
        if triangulation is None:
            explorer.Next()
            continue

        transform = loc.Transformation()
        base_index = len(vertices)
        face_vertices = []
        for i in range(1, triangulation.NbNodes() + 1):
            point = triangulation.Node(i).Transformed(transform)
            face_vertices.append((float(point.X()), float(point.Y()), float(point.Z())))
        vertices.extend(face_vertices)

        orientation = face.Orientation()
        for i in range(1, triangulation.NbTriangles() + 1):
            n1, n2, n3 = triangulation.Triangle(i).Get()
            if orientation == 1:
                n2, n3 = n3, n2
            faces.append((base_index + n1, base_index + n2, base_index + n3))
        explorer.Next()

    if not vertices or not faces:
        raise RuntimeError("No triangulated mesh extracted from shape")
    return np.asarray(vertices, dtype=np.float32), faces


def write_obj(vertices, faces, obj_path):
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    with open(obj_path, "w", encoding="ascii") as f:
        for x, y, z in vertices:
            f.write(f"v {x:.9f} {y:.9f} {z:.9f}\n")
        for i, j, k in faces:
            f.write(f"f {i} {j} {k}\n")


def render_with_blender(
    obj_path,
    output_path,
    blender_bin,
    image_size,
    samples,
    elevation_deg,
    azimuth_deg,
    camera_distance,
    background,
    object_color,
    roughness,
    key_light,
    fill_light,
    render_timeout_sec,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix="_blender_render.py", delete=False) as tmp:
        tmp.write(BLENDER_SCRIPT)
        script_path = Path(tmp.name)
    try:
        cmd = [
            blender_bin,
            "-b",
            "--python",
            str(script_path),
            "--",
            "--obj_path",
            str(obj_path),
            "--output_path",
            str(output_path),
            "--image_size",
            str(image_size),
            "--samples",
            str(samples),
            "--elevation",
            str(elevation_deg),
            "--azimuth",
            str(azimuth_deg),
            "--distance",
            str(camera_distance),
            "--background",
            str(background[0]),
            str(background[1]),
            str(background[2]),
            "--object_color",
            str(object_color[0]),
            str(object_color[1]),
            str(object_color[2]),
            "--roughness",
            str(roughness),
            "--key_light",
            str(key_light),
            "--fill_light",
            str(fill_light),
        ]
        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=render_timeout_sec if render_timeout_sec > 0 else None,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Blender render timed out after {render_timeout_sec}s: {obj_path}")
        if result.returncode != 0:
            raise RuntimeError(f"Blender render failed: {result.stderr.strip() or result.stdout.strip()}")
    finally:
        script_path.unlink(missing_ok=True)


def collect_step_files(input_path, limit=0):
    input_path = Path(input_path)
    if input_path.is_file() and input_path.suffix.lower() in {".step", ".stp"}:
        return [input_path]

    step_files = []
    for root, _, files in os.walk(input_path):
        for name in sorted(files):
            if not name.lower().endswith((".step", ".stp")):
                continue
            step_files.append(Path(root) / name)
            if limit and len(step_files) >= limit:
                return step_files
    return step_files


def main():
    parser = argparse.ArgumentParser(description="Render STEP files into single-view PNG images with OBJ cache and Blender Cycles.")
    parser.add_argument("--input", type=str, required=True, help="STEP file or directory containing STEP files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save rendered PNGs")
    parser.add_argument("--index_file", type=str, default="", help="Optional pickle file to save render index")
    parser.add_argument("--mesh_dir", type=str, default="", help="Optional directory to save OBJ mesh cache; defaults to <output_dir>/obj_cache")
    parser.add_argument("--blender_bin", type=str, default="/tmp/blender-3.6.22/blender", help="Blender executable")
    parser.add_argument("--limit", type=int, default=0, help="Render only the first N STEP files")
    parser.add_argument("--image_size", type=int, default=768, help="Output PNG size")
    parser.add_argument("--samples", type=int, default=128, help="Cycles samples per image")
    parser.add_argument("--elevation", type=float, default=35.0, help="Camera elevation in degrees")
    parser.add_argument("--azimuth_start", type=float, default=45.0, help="Base camera azimuth in degrees")
    parser.add_argument("--azimuth_step", type=float, default=17.0, help="Deterministic azimuth step per sample")
    parser.add_argument("--camera_distance", type=float, default=2.5, help="Camera distance after normalization")
    parser.add_argument("--background", type=float, nargs=3, default=(0.98, 0.98, 0.98), help="Background RGB in 0-1")
    parser.add_argument("--object_color", type=float, nargs=3, default=(0.52, 0.58, 0.64), help="Object RGB in 0-1")
    parser.add_argument("--roughness", type=float, default=0.68, help="Principled BSDF roughness")
    parser.add_argument("--key_light", type=float, default=1200.0, help="Key light energy")
    parser.add_argument("--fill_light", type=float, default=350.0, help="Fill light energy")
    parser.add_argument("--linear_deflection", type=float, default=0.05, help="Meshing linear deflection")
    parser.add_argument("--angular_deflection", type=float, default=0.3, help="Meshing angular deflection")
    parser.add_argument("--max_faces", type=int, default=1000000, help="Skip samples whose triangulated mesh exceeds this many faces; <=0 disables the guard")
    parser.add_argument("--render_timeout_sec", type=int, default=120, help="Per-sample Blender render timeout in seconds; <=0 disables the timeout")
    parser.add_argument("--reuse_obj", action="store_true", help="Reuse existing OBJ cache if present")
    args = parser.parse_args()

    step_files = collect_step_files(args.input, limit=args.limit)
    if not step_files:
        raise FileNotFoundError(f"No STEP files found under: {args.input}")

    output_dir = Path(args.output_dir)
    mesh_dir = Path(args.mesh_dir) if args.mesh_dir else output_dir / "obj_cache"
    render_index = {}
    failures = []

    for idx, step_path in enumerate(tqdm(step_files, desc="Rendering STEP images")):
        stem = step_path.stem
        image_path = output_dir / f"{stem}.png"
        obj_path = mesh_dir / f"{stem}.obj"
        azimuth = (args.azimuth_start + idx * args.azimuth_step) % 360.0
        try:
            if not (args.reuse_obj and obj_path.exists()):
                shape = load_step_shape(step_path)
                mesh_shape(shape, linear_deflection=args.linear_deflection, angular_deflection=args.angular_deflection)
                vertices, faces = collect_mesh(shape)
                if args.max_faces > 0 and len(faces) > args.max_faces:
                    raise RuntimeError(
                        f"Mesh face count {len(faces)} exceeds max_faces={args.max_faces}"
                    )
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
                "image_path": str(image_path),
                "step_path": str(step_path),
                "obj_path": str(obj_path),
                "view": {
                    "elevation": float(args.elevation),
                    "azimuth": float(azimuth),
                    "distance": float(args.camera_distance),
                },
                "render": {
                    "backend": "blender_cycles",
                    "image_size": int(args.image_size),
                    "samples": int(args.samples),
                },
                "mesh": {
                    "linear_deflection": float(args.linear_deflection),
                    "angular_deflection": float(args.angular_deflection),
                },
            }
        except Exception as exc:
            failures.append({"step_path": str(step_path), "error": str(exc)})

    if args.index_file:
        index_path = Path(args.index_file)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(index_path, "wb") as f:
            pickle.dump(
                {
                    "renders": render_index,
                    "failures": failures,
                    "config": {
                        "input": args.input,
                        "output_dir": args.output_dir,
                        "mesh_dir": str(mesh_dir),
                        "image_size": args.image_size,
                        "samples": args.samples,
                        "elevation": args.elevation,
                        "azimuth_start": args.azimuth_start,
                        "azimuth_step": args.azimuth_step,
                        "camera_distance": args.camera_distance,
                        "background": list(args.background),
                        "object_color": list(args.object_color),
                        "roughness": args.roughness,
                        "key_light": args.key_light,
                        "fill_light": args.fill_light,
                        "linear_deflection": args.linear_deflection,
                        "angular_deflection": args.angular_deflection,
                        "max_faces": args.max_faces,
                        "render_timeout_sec": args.render_timeout_sec,
                        "blender_bin": args.blender_bin,
                    },
                },
                f,
            )

    print(f"Rendered: {len(render_index)}")
    print(f"Failed:   {len(failures)}")
    if failures:
        print("First failure:", failures[0])


if __name__ == "__main__":
    main()
