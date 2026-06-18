# Stage 1 分批渲染方案

这份方案用于避免重复渲染，并把前 100 个 STEP 文件稳定地拆成多个小批次执行。

## 目标

这一步的目标是：
- 固定前 100 个 STEP 样本
- 分成 5 批，每批 20 个
- 每批单独渲染，互不重复
- 每批单独生成 `render_index.pkl`
- 最后把所有批次的 `render_index.pkl` 合并成一个总索引

## 第 1 步：导出前 100 个 STEP 文件清单

```bash
mkdir -p /workspace/dataset/render_stage1_batches
find /workspace/dataset/uz -type f \( -iname '*.step' -o -iname '*.stp' \) | sort | head -n 100 > /workspace/dataset/render_stage1_batches/step_manifest_100.txt
```

这一步会生成：
- `/workspace/dataset/render_stage1_batches/step_manifest_100.txt`

这个文件里每一行是一个 STEP 文件路径。

## 第 2 步：拆成 5 个 batch，每批 20 个

```bash
mkdir -p /workspace/dataset/render_stage1_batches/manifests
split -d -l 20 /workspace/dataset/render_stage1_batches/step_manifest_100.txt /workspace/dataset/render_stage1_batches/manifests/batch_
```

这一步会生成 5 个 manifest：
- `batch_00`
- `batch_01`
- `batch_02`
- `batch_03`
- `batch_04`

每个文件包含 20 个 STEP 路径。

## 第 3 步：逐批渲染

使用脚本：
- [process_data/render_step_images_from_list.py](/workspace/BrepARG_image/process_data/render_step_images_from_list.py)

示例：渲染第 1 批 `batch_00`

```bash
/tmp/miniconda3/envs/brepocc/bin/python process_data/render_step_images_from_list.py \
  --input_list /workspace/dataset/render_stage1_batches/manifests/batch_00 \
  --output_dir /workspace/dataset/render_stage1_batches/batch_00 \
  --index_file /workspace/dataset/render_stage1_batches/batch_00/render_index.pkl \
  --image_size 768 \
  --samples 64 \
  --background 0.92 0.92 0.92 \
  --object_color 0.42 0.52 0.64 \
  --roughness 0.70 \
  --key_light 950 \
  --fill_light 260 \
  --max_faces 1000000 \
  --render_timeout_sec 120
```

然后依次把 `batch_00` 改成：
- `batch_01`
- `batch_02`
- `batch_03`
- `batch_04`

分别执行一遍。

## 第 4 步：合并所有 render index

使用脚本：
- [process_data/merge_render_indexes.py](/workspace/BrepARG_image/process_data/merge_render_indexes.py)

执行：

```bash
/tmp/miniconda3/envs/brepocc/bin/python process_data/merge_render_indexes.py \
  --input_glob '/workspace/dataset/render_stage1_batches/batch_*/render_index.pkl' \
  --output /workspace/dataset/render_stage1_batches/render_index_100.pkl
```

这一步会生成总索引：
- `/workspace/dataset/render_stage1_batches/render_index_100.pkl`

## 第 5 步：检查合并结果

```bash
/tmp/miniconda3/envs/brepocc/bin/python - <<'PY'
import pickle
p = '/workspace/dataset/render_stage1_batches/render_index_100.pkl'
data = pickle.load(open(p, 'rb'))
print('renders', len(data['renders']))
print('failures', len(data['failures']))
if data['failures']:
    print('first_failure', data['failures'][0])
PY
```

重点看：
- `renders` 是否接近 100
- `failures` 是否在可接受范围内
- 如果有失败样本，先看第一条错误信息

## 说明

这套分批方案相对直接整批跑有两个优点：
- 不会重复渲染前面的样本
- 即使某一批出问题，也不会影响前面已经完成的批次

当前脚本里已经有两个保护机制：
- `--max_faces 1000000`
- `--render_timeout_sec 120`

它们可以避免单个超大 mesh 或超慢 Blender 渲染把整批任务卡死。
