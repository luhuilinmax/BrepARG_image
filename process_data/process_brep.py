import os 
import pickle 
import argparse
import numpy as np
from tqdm import tqdm
from multiprocessing.pool import Pool
from convert_utils import *
from occwl.io import load_step as load_solids_from_step
import shutup; shutup.please()


# To speed up processing, define maximum threshold
MAX_FACE = 200


def dump_data_dict_inspection(data, report_path, max_preview_elements=128):
    """
    将组装的 data 字典写成文本：每个 key 的名称、类型、维度/长度、dtype，
    以及数值预览（小数组全文，大数组统计量 + 前若干元素）。
    """
    lines = []

    def append_array(name, arr):
        lines.append(f"  shape: {arr.shape}")
        lines.append(f"  dtype: {arr.dtype}")
        flat = np.ravel(arr)
        n = flat.size
        if n == 0:
            lines.append("  values: (empty)")
            return
        if np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.integer):
            fi = flat.astype(np.float64) if np.issubdtype(arr.dtype, np.floating) else flat.astype(np.int64)
            lines.append(f"  min: {fi.min()}, max: {fi.max()}")
            if np.issubdtype(arr.dtype, np.floating):
                lines.append(f"  mean: {float(fi.mean())}")
        if n <= max_preview_elements:
            lines.append(f"  values (full):\n{np.array2string(arr, threshold=np.inf, max_line_width=120)}")
        else:
            prev = flat[:max_preview_elements]
            lines.append(
                f"  values (preview, first {max_preview_elements} elements raveled): {prev}"
            )

    for key, val in data.items():
        lines.append("=" * 72)
        lines.append(f"key: {key}")
        lines.append(f"type: {type(val).__name__}")
        if isinstance(val, np.ndarray):
            append_array(key, val)
        elif isinstance(val, (list, tuple)):
            lines.append(f"  len: {len(val)}")
            if len(val) == 0:
                lines.append("  values: (empty sequence)")
            else:
                first = val[0]
                lines.append(f"  elem[0] type: {type(first).__name__}")
                if len(val) <= 32:
                    lines.append(f"  values (full): {val}")
                else:
                    lines.append(f"  values (first 32): {list(val[:32])} ...")
        else:
            s = repr(val)
            lines.append(f"  repr: {s if len(s) <= 2000 else s[:2000] + '...<truncated>'}")
        lines.append("")

    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[inspect] wrote {report_path}")


def normalize(surf_pnts, edge_pnts, corner_pnts):
    """
    Various levels of normalization 
    """
    # Global normalization to -1~1
    total_points = np.array(surf_pnts).reshape(-1, 3)
    min_vals = np.min(total_points, axis=0)
    max_vals = np.max(total_points, axis=0)
    global_offset = min_vals + (max_vals - min_vals)/2 
    global_scale = max(max_vals - min_vals)
    assert global_scale != 0, 'scale is zero'

    surfs_wcs, edges_wcs, surfs_ncs, edges_ncs = [],[],[],[]

    # Normalize corner 
    corner_wcs = (corner_pnts - global_offset[np.newaxis,:]) / (global_scale * 0.5)

    # Normalize surface
    for surf_pnt in surf_pnts:    
        # Normalize CAD to WCS
        surf_pnt_wcs = (surf_pnt - global_offset[np.newaxis,np.newaxis,:]) / (global_scale * 0.5)
        surfs_wcs.append(surf_pnt_wcs)
        # Normalize Surface to NCS
        min_vals = np.min(surf_pnt_wcs.reshape(-1,3), axis=0)
        max_vals = np.max(surf_pnt_wcs.reshape(-1,3), axis=0)
        local_offset = min_vals + (max_vals - min_vals)/2 
        local_scale = max(max_vals - min_vals)
        pnt_ncs = (surf_pnt_wcs - local_offset[np.newaxis,np.newaxis,:]) / (local_scale * 0.5)
        surfs_ncs.append(pnt_ncs)
       
    # Normalize edge
    for edge_pnt in edge_pnts:    
        # Normalize CAD to WCS
        edge_pnt_wcs = (edge_pnt - global_offset[np.newaxis,:]) / (global_scale * 0.5)
        edges_wcs.append(edge_pnt_wcs)
        # Normalize Edge to NCS
        min_vals = np.min(edge_pnt_wcs.reshape(-1,3), axis=0)
        max_vals = np.max(edge_pnt_wcs.reshape(-1,3), axis=0)
        local_offset = min_vals + (max_vals - min_vals)/2 
        local_scale = max(max_vals - min_vals)
        pnt_ncs = (edge_pnt_wcs - local_offset) / (local_scale * 0.5)
        edges_ncs.append(pnt_ncs)
        assert local_scale != 0, 'scale is zero'

    surfs_wcs = np.stack(surfs_wcs)
    surfs_ncs = np.stack(surfs_ncs)
    edges_wcs = np.stack(edges_wcs)
    edges_ncs = np.stack(edges_ncs)

    return surfs_wcs, edges_wcs, surfs_ncs, edges_ncs, corner_wcs


def parse_solid(solid):
    """
    Parse the surface, curve, face, edge, vertex in a CAD solid.
   
    Args:
    - solid (occwl.solid): A single brep solid in occwl data format.

    Returns:
    - data: A dictionary containing all parsed data
    """
    assert isinstance(solid, Solid)

    # Split closed surface and closed curve to halve
    solid = solid.split_all_closed_faces(num_splits=0)
    solid = solid.split_all_closed_edges(num_splits=0)

    if len(list(solid.faces())) > MAX_FACE:
        return None
        
    # Extract all B-rep primitives and their adjacency information
    #import pdb; pdb.set_trace()
    face_pnts, edge_pnts, edge_corner_pnts, edgeFace_IncM, faceEdge_IncM = extract_primitive(solid)
    #import pdb; pdb.set_trace()
    
    # Normalize the CAD model
    surfs_wcs, edges_wcs, surfs_ncs, edges_ncs, corner_wcs = normalize(face_pnts, edge_pnts, edge_corner_pnts)
    #import pdb; pdb.set_trace()

    # Remove duplicate and merge corners 
    corner_wcs = np.round(corner_wcs,4) 
    corner_unique = []
    for corner_pnt in corner_wcs.reshape(-1,3):
        if len(corner_unique) == 0:
            corner_unique = corner_pnt.reshape(1,3)
        else:
            # Check if it exist or not 
            exists = np.any(np.all(corner_unique == corner_pnt, axis=1))
            if exists:
                continue 
            else:
                corner_unique = np.concatenate([corner_unique, corner_pnt.reshape(1,3)], 0)

    # Edge-corner adjacency  
    edgeCorner_IncM = []
    for edge_corner in corner_wcs:
        start_corner_idx = np.where((corner_unique == edge_corner[0]).all(axis=1))[0].item()
        end_corner_idx = np.where((corner_unique == edge_corner[1]).all(axis=1))[0].item()
        edgeCorner_IncM.append([start_corner_idx, end_corner_idx])
    edgeCorner_IncM = np.array(edgeCorner_IncM)

    # Surface global bbox
    surf_bboxes = []
    for pnts in surfs_wcs:
        min_point, max_point = get_bbox(pnts.reshape(-1,3))
        surf_bboxes.append( np.concatenate([min_point, max_point]))
    surf_bboxes = np.vstack(surf_bboxes)

    # Edge global bbox
    edge_bboxes = []
    for pnts in edges_wcs:
        min_point, max_point = get_bbox(pnts.reshape(-1,3))
        edge_bboxes.append(np.concatenate([min_point, max_point]))
    edge_bboxes = np.vstack(edge_bboxes)

    # Convert to float32 to save space
    data = {
        'surf_wcs':surfs_wcs.astype(np.float32),
        'edge_wcs':edges_wcs.astype(np.float32),
        'surf_ncs':surfs_ncs.astype(np.float32),
        'edge_ncs':edges_ncs.astype(np.float32),
        'corner_wcs':corner_wcs.astype(np.float32),
        'edgeFace_adj': edgeFace_IncM,
        'edgeCorner_adj':edgeCorner_IncM,
        'faceEdge_adj':faceEdge_IncM,
        'surf_bbox_wcs':surf_bboxes.astype(np.float32),
        'edge_bbox_wcs':edge_bboxes.astype(np.float32),
        'corner_unique':corner_unique.astype(np.float32),
    }

    return data


def process(args):
    step_folder, OUTPUT, INPUT_ROOT, inspect = args
    try:
        # Load cad data
        if step_folder.endswith('.step'):
            step_path = step_folder 
        else:
            for _, _, files in os.walk(step_folder):
                assert len(files) == 1 
                step_path = os.path.join(step_folder, files[0])

        # Check single solid
        #print("ABOUT TO PDB", flush=True)
        #import pdb; pdb.set_trace()
        cad_solid = load_solids_from_step(step_path)
        if len(cad_solid)!=1: 
            return 0 
        # Start data parsing
        #import pdb; pdb.set_trace()
        data = parse_solid(cad_solid[0])
        if data is None: 
            return 0

        # Preserve directory structure consistent with original STEP files (commented out)
        # rel_path = os.path.relpath(step_path, INPUT_ROOT)
        # rel_dir = os.path.dirname(rel_path)
        # base_name = os.path.splitext(os.path.basename(step_path))[0]
        # save_folder = os.path.join(OUTPUT, rel_dir)
        # os.makedirs(save_folder, exist_ok=True)
        # save_path = os.path.join(save_folder, base_name + '.pkl')
        # with open(save_path, "wb") as tf:
        #     pickle.dump(data, tf)
        
        # Save directly under the OUTPUT folder
        base_name = os.path.splitext(os.path.basename(step_path))[0]
        os.makedirs(OUTPUT, exist_ok=True)
        save_path = os.path.join(OUTPUT, base_name + '.pkl')
        with open(save_path, "wb") as tf:
            pickle.dump(data, tf)
        if inspect:
            inspect_path = os.path.join(OUTPUT, base_name + '_data_inspect.txt')
            dump_data_dict_inspection(data, inspect_path)
        return 1 
    except Exception as e:
        return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Data folder path", default='')
    parser.add_argument("--output", type=str, help="Output folder path", default='data/deepcad_parsed')
    parser.add_argument("--interval", type=int, default=0, help="Data range index, only required for abc/deepcad")
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Also write <stem>_data_inspect.txt (keys, shapes, dtypes, stats, value preview) next to each .pkl",
    )
    args = parser.parse_args()
    

    OUTPUT = args.output
    # Load all STEP files
    step_dirs = load_step(args.input)


    # # Process B-reps in parallel
    # valid = 0
    # # Pass (file_path, OUTPUT, INPUT_ROOT) tuples to each process
    # convert_iter = Pool(os.cpu_count()).imap(process, [(step_dir, OUTPUT, args.input, args.inspect) for step_dir in step_dirs]) 
    # for status in tqdm(convert_iter, total=len(step_dirs)):
    #     valid += status 
    # print(f'Done... Data Converted Ratio {100.0*valid/len(step_dirs)}%')

    # Process B-reps in single process (debug-friendly)
    valid = 0
    for step_dir in tqdm(step_dirs, total=len(step_dirs)):
        status = process((step_dir, OUTPUT, args.input, args.inspect))
        valid += status
    print(f'Done... Data Converted Ratio {100.0*valid/len(step_dirs)}%')