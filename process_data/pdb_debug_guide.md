# process_brep.py 的 pdb 全流程调试手册

本文目标：让你用一次单样本调试，把 `process_brep.py` 从 STEP 读取到 PKL 保存的全过程看清楚。

## 1. 调试前准备

### 1.1 建议输入
- 使用**单个 STEP 文件**，避免被批量进度条干扰。
- `process_brep.py` 已支持直接传 `.step` 文件路径（`load_step` 里有文件分支）。

### 1.2 建议启动命令
```bash
cd /root/autodl-tmp/AR/BrepARG_m/process_data
python process_brep.py \
  --input /你的/样本.step \
  --output /tmp/brep_debug_out
```

## 2. 调用链总览（先建立脑图）

主链路：
`__main__` -> `load_step(args.input)` -> 循环 `process((step_dir, OUTPUT, args.input))`
-> `load_solids_from_step(step_path)` -> `parse_solid(cad_solid[0])`
-> `extract_primitive(solid)` -> `normalize(...)` -> 组装 `data` -> `pickle.dump(...)`

你当前代码里已有断点（`pdb.set_trace()`）：
- `process()` 中：加载 STEP 前、调用 `parse_solid` 前
- `parse_solid()` 中：`extract_primitive` 后、`normalize` 后、`data` 组装后

## 3. 按断点推进：每站看什么

下面每一站都给了“建议命令 + 判定标准”。

---

### 断点 A：`process()`，加载 STEP 前

目的：确认输入路径解析正确，避免后面都在调假问题。

建议命令：
```text
l
p step_folder
p step_path
p OUTPUT
p INPUT_ROOT
!import os; print(os.path.exists(step_path), step_path.endswith(".step"))
```

判定标准：
- `step_path` 指向真实存在的 `.step` 文件。
- `OUTPUT` 是你预期的输出目录。

---

### 断点 B：`process()`，`load_solids_from_step` 之后

目的：确认样本是否为单体 solid（代码要求 `len(cad_solid)==1`）。

建议命令：
```text
p type(cad_solid)
p len(cad_solid)
p [type(x).__name__ for x in cad_solid[:3]]
```

判定标准：
- `len(cad_solid) == 1` 才会继续。
- 如果不是 1，这个样本会直接 `return 0`（不是脚本坏了，是数据不符合当前约束）。

---

### 断点 C1：`parse_solid()`，`extract_primitive` 前

目的：确认进入的是 `Solid`，且 face 数量不过阈值。

建议命令：
```text
p type(solid)
p len(list(solid.faces()))
p MAX_FACE
```

判定标准：
- `type(solid)` 应对应 occwl 的 `Solid`。
- 面数 `<= MAX_FACE`（默认 200），否则 `parse_solid` 返回 `None`。

---

### 断点 C2：`parse_solid()`，`extract_primitive` 后

目的：看清几何原语和邻接关系的“原始形态”。

建议命令：
```text
p type(face_pnts), face_pnts.shape
p type(edge_pnts), edge_pnts.shape
p type(edge_corner_pnts), edge_corner_pnts.shape
p type(edgeFace_IncM), edgeFace_IncM.shape
p type(faceEdge_IncM), len(faceEdge_IncM)
!import numpy as np
!print("nan-face", np.isnan(face_pnts).any(), "nan-edge", np.isnan(edge_pnts).any())
```

判定标准：
- 典型 shape：
  - `face_pnts`: `(N, 32, 32, 3)`
  - `edge_pnts`: `(M, 32, 3)`
  - `edge_corner_pnts`: `(M, 2, 3)`
  - `edgeFace_IncM`: `(M, 2)`
  - `faceEdge_IncM`: 长度为 `N` 的列表（每项是某 face 的相邻 edge 索引）
- 不应出现 NaN/Inf。

---

### 断点 D：`parse_solid()`，`normalize` 后

目的：验证全局/局部归一化是否正常。

建议命令：
```text
p surfs_wcs.shape, edges_wcs.shape
p surfs_ncs.shape, edges_ncs.shape
p corner_wcs.shape
!import numpy as np
!print("surfs_wcs range", float(np.min(surfs_wcs)), float(np.max(surfs_wcs)))
!print("edges_wcs range", float(np.min(edges_wcs)), float(np.max(edges_wcs)))
!print("surfs_ncs range", float(np.min(surfs_ncs)), float(np.max(surfs_ncs)))
!print("edges_ncs range", float(np.min(edges_ncs)), float(np.max(edges_ncs)))
!print("has_nan", np.isnan(surfs_wcs).any() or np.isnan(edges_wcs).any())
```

判定标准：
- shape 数量关系与 C2 保持一致（`N/M` 对齐）。
- 数值通常在 `[-1,1]` 附近，少量越界可接受（浮点误差/局部尺度导致），但大幅异常要警惕。
- 不应出现 NaN/Inf。

---

### 断点 E：`parse_solid()`，`data` 字典组装后

目的：确认最终训练输入结构完整，索引关系不越界。

建议命令：
```text
p data.keys()
p data["surf_wcs"].shape
p data["edge_wcs"].shape
p data["edgeFace_adj"].shape
p data["edgeCorner_adj"].shape
p len(data["faceEdge_adj"])
p data["corner_unique"].shape
!import numpy as np
!print("edge idx max", int(np.max(data["edgeFace_adj"])), "face count", data["surf_wcs"].shape[0])
!print("corner idx max", int(np.max(data["edgeCorner_adj"])), "corner_unique", data["corner_unique"].shape[0])
```

判定标准：
- `data` 必须至少包含：
  `surf_wcs`, `edge_wcs`, `surf_ncs`, `edge_ncs`, `corner_wcs`,
  `edgeFace_adj`, `edgeCorner_adj`, `faceEdge_adj`,
  `surf_bbox_wcs`, `edge_bbox_wcs`, `corner_unique`。
- 邻接索引最大值应小于对应实体数量（不越界）。

## 4. 保存后验证（脱离 pdb 再验一次）

脚本结束后，验证输出文件：
```bash
python - <<'PY'
import pickle, glob, os
files = sorted(glob.glob('/tmp/brep_debug_out/*.pkl'))
print('pkl_count:', len(files))
if files:
    p = files[0]
    with open(p, 'rb') as f:
        d = pickle.load(f)
    print('file:', p)
    print('keys:', sorted(d.keys()))
    for k in ['surf_wcs','edge_wcs','surf_ncs','edge_ncs','corner_wcs','edgeFace_adj','edgeCorner_adj']:
        v = d[k]
        print(k, getattr(v, 'shape', type(v)))
PY
```

## 5. 高频 pdb 命令（只记这几个就够）

- `n`：下一行（不进函数）
- `s`：进入函数
- `c`：继续到下一个断点
- `l`：看当前代码上下文
- `p 变量`：打印变量
- `pp 变量`：结构化打印
- `q`：退出调试
- `!python语句`：临时执行 Python（很适合做 shape/range 快检）

## 6. 常见问题与快速定位

1. `len(cad_solid) != 1`
- 含多个实体，当前逻辑会跳过。
- 先换单实体样本，确保流程通，再考虑扩展多实体支持。

2. `parse_solid` 返回 `None`
- 常见原因：face 数量超过 `MAX_FACE`。
- 先打印 `len(list(solid.faces()))` 确认。

3. 直接 `return 0` 看不到错误
- `process()` 用了 `try/except` 吞异常。
- 调试时在异常分支观察 `e`，或临时把 `except` 改为打印后再 `return 0`（仅本地调试用）。

4. 归一化报 `scale is zero`
- 几何退化导致全体点重合或维度塌缩。
- 记录样本路径并跳过，后续专门做异常样本清洗。

## 7. 一次完整调试的推荐节奏

1) `c` 到断点 A，做路径检查  
2) `c` 到断点 B，确认单 solid  
3) `c` 到断点 C2，确认原语 shape/邻接  
4) `c` 到断点 D，确认归一化范围  
5) `c` 到断点 E，确认 `data` 全量结构  
6) `c` 跑完，执行“保存后验证”脚本核对产物

按这个节奏走一遍，你就能把预处理链路从“能跑”提升到“每步都知道在做什么”。
