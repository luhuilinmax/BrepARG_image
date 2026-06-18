# Render Stage 1 Handoff

## Goal

Continue the first-stage image-input integration work for BrepARG:

- render a small subset of ABC `STEP` files into single-view images
- attach image paths back to split/sequence data
- keep the scope minimal before changing AR training code

This stage is only about:

- small-scale rendering
- data pairing
- stable file naming/indexing

This stage is **not** about:

- multi-view rendering
- modifying VQ-VAE
- modifying AR model conditioning yet


## Current Decisions

These decisions were already made and should be preserved unless there is a strong reason to change them:

- use **single-view** rendering for MVP
- use **white background**
- use **clay-like neutral appearance**
- keep **one rendered image per CAD sample** for stage 1
- keep image integration limited to the **AR stage** later
- do **not** modify VQ-VAE for image input


## Data Paths

Known dataset path from the previous environment:

- STEP root: `/workspace/dataset/uz`

Expected render outputs should be stored under a new subdirectory inside:

- `/workspace/dataset`


## Files Added In This Repo

Two helper scripts were added during the previous session:

- [process_data/render_step_images.py](/workspace/BrepARG_image/process_data/render_step_images.py)
- [process_data/attach_image_paths.py](/workspace/BrepARG_image/process_data/attach_image_paths.py)

Purpose:

1. `render_step_images.py`
- intended to render `STEP -> PNG`
- produces a render index keyed by CAD stem

2. `attach_image_paths.py`
- attaches rendered image paths to split or sequence pickle files
- uses the CAD file stem for alignment


## Important Data Alignment Assumption

The current preprocessing flow preserves the CAD sample stem as the practical identity key:

- `STEP filename stem`
- `parsed .pkl filename stem`

The sequence generation stage does **not** currently preserve a richer sample ID.

Therefore, stage 1 pairing is based on:

- `cad_stem = os.path.splitext(os.path.basename(path))[0]`

This is the intended key for:

- render output naming
- render index lookup
- split/sequence attachment


## What Was Learned About The Old Environment

The previous container/environment was not suitable for rendering work.

### pythonOCC path

Attempted path:

- `pythonOCC/OCC -> STEP read -> mesh -> matplotlib -> PNG`

Observed result:

- `matplotlib` import worked
- `OCC` import/initialization did not behave reliably
- render jobs stalled before producing output

Conclusion:

- not a safe path in that environment

### Other rendering/tooling checks

Observed in the old environment:

- `blender` not preinstalled
- `freecadcmd` not preinstalled
- `gmsh` not preinstalled
- `pyvista` not installed
- `trimesh` import worked
- `vtk` import failed with a shared-library/symbol issue

Known `vtk` failure:

- `libnetcdf.so.19: undefined symbol: H5Pset_fapl_ros3`

Conclusion:

- the old environment had broader CAD/graphics runtime problems


## What Was Learned About Blender In The Old Environment

Blender `4.1.1` was downloaded and unpacked to:

- `/workspace/tools/blender-4.1.1-linux-x64`
- symlink: `/workspace/tools/blender`

However, Blender could not run in that environment.

Known direct runtime error:

- `libXrender.so.1: cannot open shared object file`

Later `ldd /workspace/tools/blender` showed many missing shared libraries, including:

- `libXrender.so.1`
- `libXxf86vm.so.1`
- `libXfixes.so.3`
- `libXi.so.6`
- `libxkbcommon.so.0`
- `libembree4.so.4`
- `libOpenImageIO.so.2.5`
- `libOpenColorIO.so.2.3`
- `libOpenImageDenoise.so.2`
- `libopenvdb.so.11.0`
- multiple `boost 1.82` libraries
- several OpenEXR / Imath / OSL / USD related libraries

Environment variables in that old environment also showed library-path pollution:

- `LD_LIBRARY_PATH=/workspace/offline_pkgs/libs:/workspace/envs/brepgen_env/lib:...`

Conclusion:

- the old issue looked like a **container/userspace runtime problem**
- not primarily a host-kernel problem
- rendering should move to a **clean render container**


## Recommendation For The New Render Container

Do **not** reuse the old training container runtime assumptions.

Recommended properties of the new render container:

- clean Ubuntu userspace
- no training conda env auto-activated
- no polluted `LD_LIBRARY_PATH`
- Blender installed and runnable via:
  - `blender --version`
  - `blender -b --python ...`

Blender runtime dependencies should be available in the new container.

Also note:

- Blender is a strong rendering backend
- but direct `STEP` import may still require:
  - a STEP import add-on, or
  - a separate `STEP -> mesh` preprocessing tool


## Recommended Next Steps In The New Container

When resuming in the new container, do this first:

1. verify Blender runs:
- `blender --version`

2. verify how STEP will enter the rendering pipeline:
- direct Blender STEP import, or
- external `STEP -> mesh` conversion first

3. inspect the existing helper scripts in this repo:
- `process_data/render_step_images.py`
- `process_data/attach_image_paths.py`

4. adapt or replace the renderer implementation so it uses the new container's working backend

5. render a very small subset first:
- `1` file
- then `10`
- then `100`

6. keep naming stable:
- `<cad_stem>.png`

7. save an index file:
- `cad_stem -> image_path`

8. attach image paths back to split or sequence pickle data


## Suggested Resume Prompt

In the new container, resume with a prompt like this:

```text
We are continuing the BrepARG image-input stage-1 work.

Please first read:
- render_stage1_handoff.md
- process_data/render_step_images.py
- process_data/attach_image_paths.py

Context:
- goal is small-scale STEP rendering + data pairing
- STEP root is /workspace/dataset/uz
- render outputs should go under /workspace/dataset
- keep the previous decisions: single-view, white background, one image per sample, no VQ-VAE changes

First check the new container's Blender/rendering status, then continue from there.
```


## Minimal Success Criteria For Stage 1

Stage 1 is successful when all of the following are true:

- a small subset of STEP files is rendered successfully
- output images are named by CAD stem
- a render index is created
- split or sequence data can be enriched with `image_path`
- the pipeline is ready for later AR-side image conditioning experiments
