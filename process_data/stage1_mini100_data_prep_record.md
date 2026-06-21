# 第一阶段 mini100 图像条件数据准备记录

这份文档记录当前已经完成的第一阶段工作：为图像条件 AR 模型准备一份可联调的 `mini100` 小数据集。

## 目标

这一阶段只做数据侧闭环，不改模型结构。

目标是打通下面三件事：

1. 从原始 source split 中抽取 100 个样本
2. 为这 100 个样本渲染图片并提取 DINOv2 特征
3. 生成能被 `ARData` 正常读取的 `sequence_mini_100.pkl` 和图像特征索引

## 当前产物

### 1. source 对齐的小 split

- [`data/abc_split_mini_100_from_source.pkl`](/workspace/BrepARG_image/data/abc_split_mini_100_from_source.pkl)

来源：

- `/workspace/data/deduplicate/abc_data_split_6bit.pkl`

统计：

- `train=90`
- `val=10`
- `test=0`

### 2. 渲染结果与总索引

渲染输出根目录：

- `/workspace/dataset/render_stage1_mini100`

总索引：

- `/workspace/dataset/render_stage1_mini100/render_index_100.pkl`

统计：

- `png=100`
- `obj=100`
- `renders=100`
- `failures=0`

说明：

- 这 100 个渲染样本已经和 source split 对齐，不是之前那批随机 STEP 样本
- 当前渲染风格采用最初确认的方案：蓝灰色 CAD，灰色背景

### 3. mini sequence 数据

- [`data/sequence_mini_100.pkl`](/workspace/BrepARG_image/data/sequence_mini_100.pkl)

生成后实际保留下来的样本数：

- `train=51`
- `val=7`
- `test=0`

说明：

- 不是 100 个都成功进入 `sequence_mini_100.pkl`
- 这是 `2sequence.py` 的结果，不是图像特征阶段丢失

### 4. 图像特征

特征目录：

- [`data/image_features_mini_100`](/workspace/BrepARG_image/data/image_features_mini_100)

特征索引：

- [`data/image_feature_index_100.pkl`](/workspace/BrepARG_image/data/image_feature_index_100.pkl)

当前使用的视觉 backbone：

- `DINOv2 ViT-L`

当前保存格式：

- 每个样本一个 `.pt`
- 内容为 `{'patch_tokens': tensor}`

单样本张量形状：

- `patch_tokens: (1369, 1024)`

## 本阶段新增或修改的脚本

新增脚本：

- [process_data/build_mini_split_from_source_split.py](/workspace/BrepARG_image/process_data/build_mini_split_from_source_split.py)
- [process_data/build_step_manifest_from_pkl_split.py](/workspace/BrepARG_image/process_data/build_step_manifest_from_pkl_split.py)
- [process_data/extract_dinov2_features.py](/workspace/BrepARG_image/process_data/extract_dinov2_features.py)

修改脚本：

- [2sequence.py](/workspace/BrepARG_image/2sequence.py)
  - 给每个 group 补充了 `cad_path` 和 `cad_stem`
- [dataset.py](/workspace/BrepARG_image/dataset.py)
  - `ARData` 支持 `image_feature_index_file`
  - `__getitem__` 会返回 `image_features` 和 `cad_stem`
  - `collate_fn` 会堆叠 `image_features`

## 关键执行命令

### 1. 从原始 split 构建 mini100

```bash
source /workspace/scripts/setup_offline_env.sh
python process_data/build_mini_split_from_source_split.py \
  --source_split /workspace/data/deduplicate/abc_data_split_6bit.pkl \
  --output /workspace/BrepARG_image/data/abc_split_mini_100_from_source.pkl \
  --train_count 90 \
  --val_count 10
```

### 2. 由 mini split 导出 STEP manifest

```bash
source /workspace/scripts/setup_offline_env.sh
python process_data/build_step_manifest_from_pkl_split.py \
  --split_file /workspace/BrepARG_image/data/abc_split_mini_100_from_source.pkl \
  --step_root /workspace/dataset/uz \
  --output_manifest /workspace/dataset/render_stage1_mini100/step_manifest_100.txt
```

### 3. 重新生成 sequence mini 数据

```bash
source /workspace/scripts/setup_offline_env.sh
python 2sequence.py \
  --data_list /workspace/BrepARG_image/data/abc_split_mini_100_from_source.pkl \
  --output_file /workspace/BrepARG_image/data/sequence_mini_100.pkl \
  --vqvae_se_weight /workspace/BrepARG_m_ddp_fix/checkpoints_bs768/vqvae_mmap_bs768/abc_se_vqvae_epoch_600.pt \
  --aug False
```

### 4. 提取 DINOv2 特征

说明：

- 这一步不是在 `setup_offline_env.sh` 环境里跑的
- 使用的是 `/tmp/miniconda3/envs/brepocc`
- 因为当时 offline env 的 Python 版本不适合直接走当前的 DINOv2 `torch.hub` 代码

命令：

```bash
/tmp/miniconda3/envs/brepocc/bin/python process_data/extract_dinov2_features.py \
  --render_index /workspace/dataset/render_stage1_mini100/render_index_100.pkl \
  --output_dir /workspace/BrepARG_image/data/image_features_mini_100 \
  --index_output /workspace/BrepARG_image/data/image_feature_index_100.pkl \
  --image_size 518 \
  --device cpu
```

## 验证结果

已经完成一次 `ARData` 冒烟测试，结果正常。

加载方式：

- `sequence_file=/workspace/BrepARG_image/data/sequence_mini_100.pkl`
- `image_feature_index_file=/workspace/BrepARG_image/data/image_feature_index_100.pkl`

测试结论：

- `ARData` 能成功读取样本
- `cad_stem` 能正确对上图像特征
- `collate_fn` 能正常组成 batch

一个 batch 的关键形状如下：

- `input_ids: (2, 541)`
- `attention_mask: (2, 541)`
- `image_features: (2, 1369, 1024)`

## 当前结论

第一阶段的数据侧闭环已经打通，可以进入下一步模型联调。

更具体地说，现在已经具备：

1. source 对齐的小规模 AR 样本
2. 与这些样本一一对应的渲染图
3. 与这些渲染图一一对应的 DINOv2 patch tokens
4. 能把序列和图像特征同时喂给 `ARData` 的数据接口

## 下一步

下一步进入模型侧：

1. 训练入口把 `image_features` 从 dataloader 传给模型
2. AR 模型加入视觉投影层
3. 在 decoder 中接入 cross-attention
4. 先做 `batch_size=1` 的 forward/backward 冒烟测试
5. 再进入 5 样本极限过拟合实验
