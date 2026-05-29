# VQVAE 与 AR 生成流程详解

本文档用于梳理 BrepARG 论文逻辑、当前代码实现，以及在“只想验证自己训练的 VQVAE 效果”这个目标下应该采用的生成式验证流程。

对应代码库：

- `/data/project/ly/BrepARG_m_ddp_fix`
- 论文：`/data/project/ly/BrepARG_m_root/BrepARG-AutoRegressive Generation with B-rep Holistic Token Sequence Representation.pdf`

## 1. 论文整体思路

BrepARG 的核心目标是把 B-rep 的几何、位置和拓扑统一表示成一个 token 序列，然后用 decoder-only Transformer 做自回归生成。

完整流程可以理解为：

```text
真实 B-rep CAD
  -> 采样 face / edge 几何
  -> VQVAE 把 face / edge 几何离散成 geometry tokens
  -> bbox 均匀量化成 position tokens
  -> face index 表示 topology tokens
  -> 拼成 holistic token sequence
  -> AR 模型学习 next-token prediction
  -> 推理时 AR 从 START 生成完整 token sequence
  -> detokenize + VQVAE decoder + bbox 反量化
  -> joint optimization
  -> OpenCascade 构造并 sew 成 B-rep solid
```

论文把 token 分成三类：

```text
Geometry Tokens: face / edge 的几何 token，来自 VQVAE codebook
Position Tokens: face / edge 的 bbox token，来自均匀标量量化
Topology Tokens: face index token，用于表示 face 和 edge 的连接关系
```

最终序列形式是：

```text
[START] + face blocks + [SEP] + edge blocks + [END]
```

其中：

```text
face block = 6 个 bbox tokens + 4 个 geometry tokens + 1 个 face index token
edge block = 2 个 face index tokens + 6 个 bbox tokens + 4 个 geometry tokens
```

## 2. VQVAE 在论文中的作用

VQVAE 不是完整 CAD 生成器。它的作用是把连续 face / edge 几何压缩成离散 geometry tokens，并能从这些 tokens 重建几何。

### 2.1 输入是什么

论文中，每个 face 会在 UV 参数域采样成：

```text
face: 32 x 32 x 3
```

每条 edge 会沿 U 方向采样成：

```text
edge: 32 x 3
```

由于当前 VQVAE 使用 2D 卷积结构，edge 会被 broadcast 成：

```text
edge: 32 x 32 x 3
```

当前代码里，edge 扩展逻辑可见：

- `utils.py` 的 `prepare_vqvae_input()`：`edge_data.unsqueeze(2).repeat(1, 1, 32, 1)`
- `2sequence.py` 的 `prepare_surface_edge_batch_for_vqvae()`：`np.tile(edge_data[:, :, np.newaxis, :], (1, 1, 32, 1))`

然后 VQVAE 做：

```
NCS geometry
  -> encoder
  -> 2 x 2 latent map
  -> nearest codebook lookup
  -> 4 个 geometry token
  -> decoder
  -> reconstructed geometry
```

所以论文里每个 face 或 edge 都会得到 4 个 geometry tokens。你的代码里也对应```tokens_per_element = 4```。

VQVAE 训练目标是重建几何：

```
loss = reconstruction loss + VQ/codebook loss
```
它只学习：

- face/edge 的局部几何形状如何压缩成 codebook token
- codebook token 如何还原成 face/edge NCS

它不学习：

- CAD 有几个 face
- 有几条 edge
- bbox 在哪里
- 哪条 edge 连接哪两个 face
- 这些不是 VQVAE 的职责。

## 3. 训练 VQVAE 时 encoder、decoder、reconstructed geometry 分别做什么

对应代码主要在：

- `trainer.py`
- `quantise.py`

### 3.1 Encoder 做什么

encoder 接收 face / edge 的 NCS 几何输入：

```text
输入 x: (B, 3, 32, 32)
```

然后把它压缩成一个低分辨率 latent feature map。论文中说它会下采样 16 倍，所以：

```text
32 x 32 -> 2 x 2
```

代码位置：

```text
trainer.py:269-271
```

```python
h = model.encoder(batch_data)
h = model.quant_conv(h)
```

含义是：

- `model.encoder(batch_data)`：把原始几何变成连续 latent feature。
- `model.quant_conv(h)`：用 1x1 卷积把 latent channel 投影到 codebook 使用的维度。

论文里对应为：

```text
x -> encoder -> ze
ze -> 1x1 conv -> z'e
```

### 3.2 Quantizer / codebook 做什么

quantizer 把连续 latent vector 映射到离散 codebook index。

论文中每个 face / edge 最后得到 `2 x 2 = 4` 个 latent vectors，所以每个 face / edge 得到：

```text
4 个 geometry tokens
```

代码位置：

```text
trainer.py:273-274
quantise.py:55-71
```

关键代码：

```python
quant_out, vq_loss, indices = model.quantize(h)
```

`indices` 就是 codebook index，也就是后续 AR 序列里的 geometry tokens。

在 `quantise.py` 中，当前项目使用 cosine similarity 查找最近 codebook：

```python
normed_z_flattened = F.normalize(z_flattened, dim=1).detach()
normed_codebook = F.normalize(self.embedding.weight, dim=1)
d = torch.einsum('bd,dn->bn', normed_z_flattened, rearrange(normed_codebook, 'n d -> d n'))
encoding_indices = indices[:,-1]
```

注意这里 `d` 越大表示 cosine similarity 越高，所以取排序后的最后一个 index。

### 3.3 Decoder 做什么

decoder 接收量化后的 latent，也就是 `quant_out`，把它还原成几何网格：

```text
quantized latent -> decoder -> reconstructed geometry
```

代码位置：

```text
trainer.py:284-285
```

```python
recon = model.decoder(model.post_quant_conv(quant_out))
```

这里的 `recon` 就是 reconstructed geometry。

它的形状仍然是：

```text
(B, 3, 32, 32)
```

对 face 来说，它对应重建后的 `32 x 32 x 3` face NCS。

对 edge 来说，因为训练时 edge 被扩展成 `32 x 32 x 3`，解码后再通过当前工具函数转回 `32 x 3` edge 曲线。当前代码里 `convert_vqvae_output_to_ncs()` 对 edge 的做法是对一个维度求平均：

```text
utils.py:1131-1138
```

## 4. Reconstruction loss 和 VQ/codebook loss 怎么算

### 4.1 Reconstruction loss

reconstruction loss 衡量：

```text
VQVAE decoder 重建出的 geometry 与原始输入 geometry 有多接近
```

代码位置：

```text
trainer.py:287-294
trainer.py:413-415
```

训练时关键代码：

```python
recon_loss = nn.functional.mse_loss(recon, batch_data)
```

如果启用了 `use_type_flag`，输入有 4 个通道，但只对前 3 个坐标通道算 loss：

```python
coords_input = batch_data[:, :3, :, :]
recon_loss = nn.functional.mse_loss(recon, coords_input)
```

简单说：

```text
reconstruction loss = mean((reconstructed_geometry - original_geometry)^2)
```

这个 loss 越低，说明 VQVAE 对 face / edge 几何的还原越好。

### 4.2 VQ/codebook loss

VQ/codebook loss 在 `quantise.py` 的 `VectorQuantiser.forward()` 中计算。

代码位置：

```text
quantise.py:68-73
```

关键代码：

```python
z_q = torch.matmul(encodings, self.embedding.weight).view(z.shape)
loss = self.beta * torch.mean((z_q.detach()-z)**2) + torch.mean((z_q - z.detach()) ** 2)
z_q = z + (z_q - z).detach()
```

可以拆成两部分理解：

```text
commitment loss = beta * mean((stop_grad(z_q) - z)^2)
codebook loss   = mean((z_q - stop_grad(z))^2)
```

含义是：

- `commitment loss`：约束 encoder 输出 `z` 不要随意漂移，要靠近选中的 codebook vector。
- `codebook loss`：让 codebook vector 向 encoder 输出靠近。
- `z_q = z + (z_q - z).detach()`：straight-through estimator，让前向使用量化后的 `z_q`，反向梯度能传回 encoder。

当前项目还启用了自定义 `VectorQuantiser` 的 codebook restart 和 contrastive loss：

```text
quantise.py:81-111
```

其中：

- `embed_prob` 统计 codebook 使用率。
- 低使用率 code 会从 feature pool 中重新初始化，缓解 codebook collapse。
- 如果 `contras_loss=True`，还会额外加一个 contrastive loss。

最终 VQVAE 训练总 loss 在：

```text
trainer.py:296-297
```

```python
total_loss = recon_loss + vq_loss
```

## 5. AR 模型和 VQVAE 如何合作

### 5.1 训练 AR 前，VQVAE 先把真实 CAD 几何变成 token

`2sequence.py` 会读取真实 CAD pkl。

对每个 face / edge：

```text
surf_ncs / edge_ncs
  -> VQVAE encoder
  -> quant_conv
  -> quantize
  -> geometry token indices
```

代码位置：

```text
2sequence.py:522-538
```

也就是说，AR 训练数据中的 geometry tokens 来自已经训练好的 VQVAE。

### 5.2 bbox 和 topology 不经过 VQVAE

bbox token 来自均匀标量量化：

```text
utils.py:507-521
```

topology token 来自 face index：

```text
每个 face block 末尾有 1 个 face index
每个 edge block 开头有 2 个 face index
```

edge block 开头的两个 face index 表示：

```text
这条 edge 连接哪两个 face
```

### 5.3 AR 学什么

AR 是 GPT2LMHeadModel 风格的 decoder-only Transformer。

代码位置：

```text
model.py:6-39
model.py:52-69
trainer.py:913-928
```

训练时输入完整 token sequence，labels 等于 input_ids：

```python
labels = input_ids.clone()
outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
ce_loss = outputs.loss
```

所以 AR 学的是：

```text
p(next token | previous tokens)
```

它同时学习：

- face block 数量
- edge block 数量
- geometry token 分布
- bbox token 分布
- topology token 分布
- START / SEP / END 的结构规律

## 6. 有 AR 权重时的生成流程

有 AR 权重时，才是论文意义上的“生成新 CAD”。

流程是：

```text
START token
  -> AR 自回归采样完整 sequence
  -> parse sequence
  -> VQVAE decoder 解 geometry tokens
  -> bbox tokens 反量化
  -> face index tokens 还原 topology
  -> joint optimization
  -> construct_brep
  -> OpenCascade sew
  -> STEP/STL
```

代码入口：

```text
generate_brep.py:451-573
```

其中：

```text
generate_brep.py:456-466
```

调用 `generate_sequence()` 生成完整 token sequence。

```text
generate_brep.py:497-505
```

调用 `reconstruct_cad_from_sequence()` 把 sequence 重建为 B-rep solid。

```text
generate_brep.py:539-548
```

写出 STEP/STL。

这时各类信息来源是：

```text
face 数量: AR 生成了多少个 face block
edge 数量: AR 生成了多少个 edge block
bbox: AR 生成 position tokens
topology: AR 生成 edge block 开头的两个 face index tokens
geometry: AR 生成 geometry tokens，再由 VQVAE decoder 解码
```

所以有 AR 权重时，AR 是真正决定 CAD 结构和 token 内容的生成器；VQVAE 是 geometry tokens 的解码器。

## 7. bbox tokens 反量化是什么

论文里 bbox 是：

```text
b = [xmin, ymin, zmin, xmax, ymax, zmax] ∈ [-1, 1]^6
```

量化时，把每个连续坐标映射到 `0 ~ L-1` 的整数 token。当前代码中 `L=2048`。

量化代码：

```text
utils.py:507-521
```

```python
normalized_coords = (bbox_coords + 1) / 2.0
normalized_coords = np.clip(normalized_coords, 0, 1)
scaled_coords = normalized_coords * (num_tokens - 1)
quantized_indices = np.round(scaled_coords).astype(int)
```

反量化就是把 token index 还原回 `[-1, 1]` 范围内的连续 bbox 坐标。

代码位置：

```text
utils.py:523-533
```

```python
float_indices = indices.astype(float)
normalized_coords = float_indices / (num_tokens - 1)
bbox_coords = normalized_coords * 2.0 - 1.0
```

在解析 AR 生成序列时调用：

```text
utils.py:767-770
```

```python
surf_bbox_wcs = dequantize_bbox(np.array(face_bboxes), num_tokens=bbox_index_size).tolist()
edge_bbox_wcs = dequantize_bbox(np.array(edge_bboxes), num_tokens=bbox_index_size).tolist()
```

直观理解：

```text
AR 生成的是整数 token，例如 1234
反量化后变成连续坐标，例如 0.205...
6 个坐标组合成一个 bbox
```

这一步不是学习出来的，是确定性的数学映射。

## 8. joint optimization 是什么

VQVAE decoder 解出来的是 NCS，即 normalized coordinate system 下的 face / edge 几何。bbox 反量化得到的是 WCS 中的位置范围。

但是仅仅把 NCS 按 bbox 放回 WCS，通常还不够，因为：

- edge 的两个端点可能和推断出的 vertex 不完全对齐。
- face 曲面边界和 edge 曲线可能不完全贴合。
- VQVAE quantization 会带来几何误差。

因此需要 joint optimization，让 face、edge、vertex 在拓扑约束下更一致。

### 8.1 reconstruct_cad_from_sequence 的前置工作

代码位置：

```text
utils.py:785-1110
```

主要做：

1. `parse_sequence_to_cad_data()` 解析 token。
2. 得到 `surf_ncs_vqvae`、`edge_ncs_vqvae`、`surf_bbox_vqvae`、`edge_bbox_vqvae`、`graph_edges`。
3. 根据 edge 的两个 face index 建立 `FaceEdgeAdj`。
4. 根据 edge 端点推断 candidate vertices。
5. 用 union-find 合并同一个 face 内、不同 face 间应该相同的 vertex。
6. 调用 `joint_optimize()`。
7. 调用 `construct_brep()`。

### 8.2 joint_optimize 具体做什么

代码位置：

```text
utils.py:1294-1398
```

它分两部分：

### 第一部分：调整 edge

代码位置：

```text
utils.py:1310-1347
```

它根据 `EdgeVertexAdj` 找到每条 edge 的两个目标 vertex，然后：

- 计算 edge 当前 NCS 起终点距离。
- 计算目标 vertex 起终点距离。
- 用两者比例缩放 edge。
- 判断 edge 是否需要反向。
- 平移 edge，使起终点靠近目标 vertex。
- 用线性权重把起点和终点误差沿整条曲线传播。

直观理解：

```text
让每条解码出的 edge 曲线尽量接到推断出的两个 vertex 上
```

### 第二部分：调整 face

代码位置：

```text
utils.py:1349-1398
```

它先根据 face bbox 初始化 face WCS：

```python
wcs = ncs * (surf_scale/2) + surf_center
```

然后优化一个很小的变换参数 `surf_st`，让 face 曲面更贴近它周围的 edge。

关键代码：

```text
utils.py:1376-1392
```

```python
surf_updated = surf + surf_offset
surf_loss += ChamferDistance(surf_pnt, edge_pnts)
surf_loss.backward()
optimizer.step()
```

直观理解：

```text
让 face 曲面边界附近尽量贴住它关联的 edge 曲线
```

所以 joint optimization 不是重新生成 CAD，而是在已有 topology 下，把 VQVAE 解出来的几何和 bbox/topology 对齐。

## 9. OpenCascade sew 是什么

`construct_brep()` 负责把优化后的点云几何变成真正的 OpenCascade B-rep。

代码位置：

```text
utils.py:1453-1583
```

### 9.1 拟合 face surface

代码位置：

```text
utils.py:1458-1469
```

它把每个 `32 x 32` face 点网格拟合成 B-spline surface：

```python
approx_face = GeomAPI_PointsToBSplineSurface(...).Surface()
```

### 9.2 拟合 edge curve

代码位置：

```text
utils.py:1471-1488
```

它把每条 edge 的 32 个点拟合成 B-spline curve：

```python
approx_edge = GeomAPI_PointsToBSpline(...).Curve()
```

再通过：

```text
utils.py:1490-1494
```

把 curve 变成 OCC edge：

```python
edge = BRepBuilderAPI_MakeEdge(curve).Edge()
```

### 9.3 用 wire 裁剪 face

代码位置：

```text
utils.py:1496-1565
```

它根据 `FaceEdgeAdj` 和 `EdgeVertexAdj` 排出每个 face 的边界 loop，然后：

```python
wire_builder = BRepBuilderAPI_MakeWire()
face_builder = BRepBuilderAPI_MakeFace(surface, outer_wire)
```

这一步相当于：

```text
用 edge loop 在 surface 上裁剪出真正的 B-rep face
```

### 9.4 sew 成 shell，再 make solid

代码位置：

```text
utils.py:1567-1581
```

关键代码：

```python
sewing = BRepBuilderAPI_Sewing()
sewing.SetTolerance(1e-3)
for face in post_faces:
    sewing.Add(face)
sewing.Perform()
sewn_shell = sewing.SewedShape()

maker = BRepBuilderAPI_MakeSolid()
maker.Add(sewn_shell)
maker.Build()
solid = maker.Solid()
```

OpenCascade sew 的作用是：

```text
把多个已经裁剪好的 face 按边界缝合成一个 shell
```

然后 `BRepBuilderAPI_MakeSolid()` 把 shell 转成 solid。

如果 face/edge 对不齐、wire 不闭合、边界自交、缝隙太大，sew 或 valid check 就可能失败。

## 10. 没有 AR 权重时能做什么

没有 AR 权重时，不能从 `START` 自动生成新 CAD。

原因是 VQVAE 不负责生成：

- face 数量
- edge 数量
- bbox
- topology
- SEP / END 结构

没有 AR 时，合理流程是：

```text
真实 ABC 样本
  -> 读取真实 face/edge 数量
  -> 读取真实 bbox
  -> 读取真实 topology
  -> 用你的 VQVAE encoder 编 geometry tokens
  -> 用你的 VQVAE decoder 解回 NCS
  -> 借用真实 bbox/topology 走 reconstruct_cad_from_sequence
  -> 输出 STEP/STL + MSE + brep_valid
```

这叫 VQVAE round-trip reconstruction，验证的是 VQVAE 的几何编码/解码效果。

## 11. 你现在验证自己 VQVAE 效果应该采用的流程

你的目标是：

```text
通过生成 CAD 来验证自己训练的 VQVAE 是否有效
```

我建议采用下面这个流程：

```text
真实 ABC pkl 样本
  -> 按论文/2sequence.py 的方式排序 face 和 edge
  -> 用你的 VQVAE encode surf_ncs / edge_ncs
  -> 得到每个 face/edge 的 4 个 geometry tokens
  -> 用真实 bbox 量化成 position tokens
  -> 用真实 topology 生成 face index tokens
  -> 组装 holistic token sequence
  -> 调用 reconstruct_cad_from_sequence
  -> VQVAE decoder 解 geometry tokens
  -> bbox 反量化
  -> joint optimization
  -> OpenCascade construct_brep + sew
  -> 输出 STEP/STL、MSE、brep_valid
```

这个流程的优点：

- 最大程度复用论文和原代码的 detokenize / reconstruct 后半段。
- 不需要 AR 权重。
- 可以隔离 AR 的影响，专门验证 VQVAE 几何 token 是否可靠。
- 可以同时看两类指标：
  - 几何指标：face MSE、edge MSE、token 范围、codebook 使用情况。
  - CAD 指标：STEP/STL 是否成功、`brep_valid` 是否为 true。

需要注意：

```text
如果这个流程失败，问题可能来自 VQVAE 几何误差，也可能来自后处理对 edge/face 边界对齐很敏感。
如果这个流程成功，只能说明 VQVAE 几何重建和后处理可用，不能说明 AR 生成能力已经可用。
```

## 12. 有 AR 权重和没有 AR 权重的区别

| 信息来源 | 有 AR 权重 | 没有 AR 权重 |
| --- | --- | --- |
| face 数量 | AR 生成多少个 face block | 来自真实样本 |
| edge 数量 | AR 生成多少个 edge block | 来自真实样本 |
| bbox | AR 生成 bbox tokens，再反量化 | 来自真实样本，再量化/反量化 |
| topology | AR 生成 face index tokens | 来自真实样本 |
| geometry tokens | AR 生成 | VQVAE encode 真实几何得到 |
| geometry reconstruction | VQVAE decoder | VQVAE decoder |
| CAD 构造 | joint optimization + OpenCascade sew | joint optimization + OpenCascade sew |
| 验证对象 | AR + VQVAE + 后处理整体生成能力 | 主要验证 VQVAE 几何编码/解码能力 |


## 13. 什么是 VQVAE 往返重建

VQVAE 的往返重建，也就是 round-trip reconstruction，指的是让真实几何完整走一遍：

```text
连续几何 -> VQVAE encoder -> codebook tokens -> VQVAE decoder -> 连续几何
```

在当前任务里更具体是：

```text
真实 ABC CAD pkl
  -> 读取 surf_ncs / edge_ncs
  -> 用 VQVAE encoder 编成 geometry tokens
  -> 每个 face/edge 通常得到 4 个 tokens
  -> 用 VQVAE decoder 解回 surf_ncs_recon / edge_ncs_recon
  -> 与原始 surf_ncs / edge_ncs 计算 MSE
  -> 借用真实 bbox/topology 尝试重建 STEP/STL
```

它验证的是 VQVAE 对几何的压缩和还原能力。它不是从无到有生成新 CAD，因为 face 数量、edge 数量、bbox、topology 仍然来自真实样本。

所以：

```text
VQVAE 往返重建成功
  = VQVAE geometry tokenization / detokenization 可用
  != AR 生成能力已经可用
```

## 14. joint optimization 的定位和大致流程

### 14.1 定位
`joint optimization` 不是一个复杂神经网络生成器，而是生成管线中的几何后处理步骤。它位于：

```text
utils.py:1294-1398
```

它的输入包括：

```text
surf_ncs          VQVAE decoder 得到的 face NCS
edge_ncs          VQVAE decoder 得到的 edge NCS
surfPos           bbox 反量化得到的 face bbox
unique_vertices   推断出的唯一顶点坐标
EdgeVertexAdj     每条 edge 连接哪两个 vertex
FaceEdgeAdj       每个 face 由哪些 edge 围成
```

它解决的问题是：VQVAE 解码出的 face/edge 几何不一定天然严丝合缝。比如 edge 起终点可能没有正好落到共同 vertex 上，face 曲面边界也可能没有正好贴住周围 edge。OpenCascade 对这些几何闭合关系比较敏感，所以在构造 B-rep 前要先做一次局部对齐。

这个步骤不是 AR 生成器，也不是 VQVAE 训练的一部分。它更像 detokenization 之后的 B-rep 几何修复与对齐模块。

从论文角度看，BrepARG 的主要创新是 holistic token sequence、VQVAE geometry tokenization、topology-aware sequentialization 和 AR 生成框架。joint optimization / vertex merge / OpenCascade sew 是使 token 序列能落成有效 B-rep 的必要后处理，但不是最核心的生成模型创新。

### 14.2 大致流程
先处理 edge，代码里先根据推断出的 unique_vertices 和 EdgeVertexAdj，知道每条 edge 应该连接哪两个 vertex。

然后对每条 edge 做：

```
edge NCS 曲线
  -> 根据目标 vertex 距离缩放
  -> 判断是否需要反向
  -> 平移到目标 vertex 附近
  -> 把起点/终点误差沿整条曲线线性传播
```

直观理解：

让每条 edge 的起点和终点尽量贴到对应 vertex 上。

再处理 face，face 先根据 bbox 从 NCS 放回 WCS：

surf_wcs = surf_ncs * scale + center

然后优化一个很小的 surface offset，使 face 点云更靠近它周围的 edge 点云。

代码里用的是 ChamferDistance：

face surface points 和 adjacent edge points 之间的 Chamfer 距离

优化目标大概是：

让 face 曲面边界附近尽量贴住它关联的 edge。
所以它是一个局部几何对齐过程。


## 15. unique_vertices 是怎么获得的

`unique_vertices` 在 `reconstruct_cad_from_sequence()` 中推断得到，代码位置：

```text
utils.py:852-1080
```

### 15.1 先把每条 edge 的 NCS 放回 WCS

代码先对每条 edge 读取反量化后的 bbox：

```text
utils.py:856-868
```

核心逻辑是：

```python
bcenter, bsize = compute_bbox_center_and_size(min_point, max_point)
wcs_curve = ncs_curve * (bsize / 2) + bcenter
```

也就是：

```text
edge_ncs_vqvae 是 [-1, 1] 范围内的归一化曲线
bbox 给出这条 edge 在世界坐标中的中心和尺度
将 NCS 曲线按 bbox 缩放和平移到 WCS
```

然后取每条 edge 的第一个点和最后一个点作为候选 vertex：

```text
utils.py:871-873
```

```python
bbox_start_end = wcs_curve[[0, -1]]
edgeV_bbox.append(bbox_start_end)
```

如果有 `E` 条 edge，那么一开始会有：

```text
2 * E 个候选端点
```

每个候选点有一个 global id：

```text
edge_idx * 2 + vertex_pos_idx
```

代码位置：

```text
utils.py:887-888
```

### 15.2 face 内合并候选顶点

对于每个 face，代码先根据 `FaceEdgeAdj` 找到围成这个 face 的 edge，再收集这些 edge 的所有端点：

```text
utils.py:908-918
```

然后计算这个 face 内所有候选端点之间的欧氏距离：

```text
utils.py:920-929
```

接着用 greedy 方式不断找最近的一对端点合并，但会跳过同一条 edge 的两个端点：

```text
utils.py:931-969
```

这一步的直观含义是：

```text
一个 face 边界 loop 上，不同 edge 的相邻端点应该是同一个 vertex。
因此把距离最近、且不来自同一条 edge 的端点合并。
```

合并用的是 union-find：

```text
utils.py:890-901
```

### 15.3 face 间继续合并

同一个真实 vertex 可能出现在多个 face 的边界上。代码会检查不同 face 的合并组，如果两个组合并过共同端点，就继续把它们 union 到一起：

```text
utils.py:993-1012
```

这一步的作用是：

```text
把不同 face 中代表同一个空间顶点的候选端点统一成一个 vertex group。
```

### 15.4 每个 group 求平均，得到 unique_vertices

最后，代码遍历 union-find 的最终 groups，对每个 group 内所有候选端点坐标求平均：

```text
utils.py:1026-1049
```

关键代码：

```python
avg_position = np.mean(group_positions, axis=0)
unique_vertices.append(avg_position)
```

因此：

```text
unique_vertices 中的每个点
  = 一组被认为属于同一个真实 vertex 的 edge endpoint 候选点的平均位置
```

同时代码会建立 `vertex_mapping`，再得到 `EdgeVertexAdj`：

```text
utils.py:1051-1065
```

`EdgeVertexAdj[edge_idx] = [start_vertex_idx, end_vertex_idx]`，表示每条 edge 连接哪两个唯一顶点。

## 16. edge NCS 曲线是否需要反向是怎么判断的

edge 曲线本身是一串有方向的 32 个采样点：

```text
edge_ncs[0]   起点
edge_ncs[-1]  终点
```

但经过 VQVAE 解码、bbox 映射和 topology 推断后，这条曲线的采样方向可能和 `EdgeVertexAdj` 中的 vertex 顺序相反。所以代码会判断是否需要把 edge 反过来。

代码位置：

```text
utils.py:1316-1334
```

先取 edge 的 NCS 起终点：

```python
edge_ncs_se = edge_ncs[:, [0, -1]]
```

再取这条 edge 应该连接的两个目标 vertex：

```python
edge_vertex_se = unique_vertices[EdgeVertexAdj]
```

然后比较两种情况。

### 16.1 正向匹配

假设：

```text
edge 起点 -> vertex 0
edge 终点 -> vertex 1
```

对应代码：

```python
offset = (vertex_se - edge_se)
offset_error = np.abs(offset[0] - offset[1]).mean()
```

这里不是直接比较距离，而是比较两个端点所需的平移 offset 是否一致。

如果一条 edge 的方向正确，那么起点到 vertex 0、终点到 vertex 1 所需的平移量应该比较接近。也就是说，用一个整体平移就能同时让两个端点贴近目标 vertex。

### 16.2 反向匹配

反向情况是假设：

```text
edge 起点 -> vertex 1
edge 终点 -> vertex 0
```

对应代码：

```python
offset_rev = (vertex_se - edge_se[::-1])
offset_rev_error = np.abs(offset_rev[0] - offset_rev[1]).mean()
```

### 16.3 判断规则

代码判断：

```python
if offset_rev_error < offset_error:
    edge_updated = edge_updated[::-1]
    offset = offset_rev
```

也就是：

```text
如果反向后两个端点所需的平移更一致，就认为 edge 方向反了，需要 reverse。
```

直观理解：

```text
哪种方向能让 edge 的两个端点通过同一个平移更自然地对齐目标 vertex，就采用哪种方向。
```

反向之后，代码再用 `offset.mean(0)` 给整条 edge 做平移：

```text
utils.py:1336
```

然后把起点和终点剩余误差沿 32 个采样点线性传播：

```text
utils.py:1341-1347
```

这样可以强制 edge 两端贴住目标 vertex，同时尽量平滑地调整整条曲线。

## 17. 优化一个很小的 surface offset 是什么意思

`STModel` 中定义了一个可学习参数：

```text
utils.py:1235-1239
```

```python
self.surf_st = nn.Parameter(torch.FloatTensor([1, 0, 0, 0]).unsqueeze(0).repeat(num_surf, 1))
```

每个 face 有 4 个参数：

```text
[scale_like_param, offset_x, offset_y, offset_z]
```

但当前 `joint_optimize()` 实际使用的是：

```text
utils.py:1379-1381
```

```python
surf_scale = model.surf_st[:,0].reshape(-1,1,1,1)
surf_offset = model.surf_st[:,1:].reshape(-1,1,1,3)
surf_updated = surf + surf_offset
```

注意：`surf_scale` 被取出来了，但当前代码没有真正乘到 `surf_updated` 上。因此目前实际优化的是：

```text
每个 face 一个三维平移 offset
```
注意它们在同一个优化循环里一起更新，但参数是按face分开的。

所谓“很小的 surface offset”，就是让每个 face 曲面整体在 XYZ 方向上轻微移动，而不是改变其局部形状。

更准确地说，当前实现不是复杂形变优化，而是：

```text
固定 face 的采样点形状
只优化每个 face 的整体平移量
```

## 18. 怎么通过优化 surface offset 让 face 点云靠近周围 edge 点云

在 joint optimization 里，先把 face 从 NCS 放回 WCS：

```text
utils.py:1359-1374
```

核心公式是：

```python
wcs = ncs * (surf_scale / 2) + surf_center
```

这得到初始 face 点云 `surf_wcs_init`。

然后转成 torch tensor：

```text
utils.py:1377
```

```python
surf = torch.FloatTensor(surf_wcs_init).cuda()
```

每轮优化时，对每个 face 加上它自己的 offset：

```text
utils.py:1378-1381
```

```python
surf_offset = model.surf_st[:,1:].reshape(-1,1,1,3)
surf_updated = surf + surf_offset
```

接着，对每个 face，取它关联的 edge 点：

```text
utils.py:1350-1353
```

```python
for adj in FaceEdgeAdj:
    all_pnts = edge_wcs[adj]
    face_edges.append(torch.FloatTensor(all_pnts).cuda())
```

优化目标是 ChamferDistance：

```text
utils.py:1383-1388
```

```python
surf_loss += loss_func(
    surf_pnt.unsqueeze(0),
    edge_pnts.unsqueeze(0),
    bidirectional=False,
    reverse=True
)
```

直观理解：

```text
对于一个 face，把它的 32x32 点云整体平移一点；
希望周围 edge 点云能够更贴近这个 face 点云；
ChamferDistance 越小，说明 face 和 edge 的几何位置越一致。
```

然后正常反向传播：

```text
utils.py:1390-1392
```

```python
optimizer.zero_grad()
surf_loss.backward()
optimizer.step()
```

因为唯一真正参与的可学习变量是 `surf_offset`，所以优化器会更新每个 face 的三维平移量，使 ChamferDistance 下降。

这一步的结果是：

```text
surf_wcs = surf_updated.detach().cpu().numpy()
```

返回给 `construct_brep()`，用于后续拟合 B-spline surface 和裁剪 B-rep face。

## 19. 对当前实现的一个重要判断

当前 `joint_optimize()` 名字里有 optimize，但它的优化范围其实比较有限：

- edge 部分主要是确定性缩放、方向判断、平移和端点误差传播。
- face 部分主要优化每个 face 的整体平移 offset。
- 不会重新学习 VQVAE。
- 不会重新生成 topology。
- 不会大幅改变 face / edge 的局部形状。

所以它适合做：

```text
把 VQVAE 解码几何、bbox 和 topology 对齐到足以构造 B-rep
```

但如果 VQVAE 解码几何本身已经严重失真，或者 topology / bbox 不合理，joint optimization 通常无法根本修复。

对你当前目标来说，它的作用是辅助判断：

```text
如果 VQVAE round-trip 的 MSE 低，并且经过 joint optimization 后能输出 valid STEP/STL，
说明你训练的 VQVAE 至少具备可用于后续 AR 生成管线的几何 tokenization 能力。
```


## 20. 当前脚本的定位

当前新增脚本：

```text
reconstruct_vqvae_sample.py
```

它的定位应该是：

```text
VQVAE 单样本 round-trip CAD 重建验证脚本
```

它不是完整 AR 生成脚本。它适合回答：

```text
我训练出来的 VQVAE 能不能把真实 ABC 样本的 face/edge 几何编码成 token，再解码回可构造 CAD 的几何？
```

它不适合回答：

```text
我的模型能不能从无到有生成新的 CAD？
```

如果要让它更严格对齐论文逻辑，后续应补齐：

- `2sequence.py` 中的 face DFS 排序。
- edge MAX-IDX-A 排序。
- face index cyclic re-index。
- 与 `generate_brep.py` 更一致的 STEP/STL 写文件超时保护。
