# AR 实验分层与 Teacher-Forced 说明

本文档用于统一梳理当前 `BrepARG_m_ddp_fix` 中几类容易混淆的实验：

- VQVAE round-trip
- AR teacher-forced evaluation
- AR free-running generation
- strict teacher forcing
- partial / half teacher forcing
- scheduled / mixed forcing

重点是明确每类实验：

- 输入给模型的真实 token 有多少
- 模型输出是什么
- 最终是否得到一条“自洽的完整生成序列”
- 是否适合反解并后处理成 CAD
- 对当前项目的实验作用是什么

## 1. 为什么要分层做实验

当前生成链路不是单一模型，而是多阶段串联：

```text
真实 CAD / 或 AR 生成的 sequence
  -> token 解析
  -> VQVAE decoder 解 geometry tokens
  -> bbox 反量化
  -> joint optimization / vertex merge / OpenCascade sew
  -> STEP / STL / valid B-rep
```

所以“最终随机采样出来的 CAD 效果差”并不能直接说明是哪个环节有问题。至少需要区分：

```text
1. VQVAE 几何 token 化 / 反 token 化是否可用
2. AR 在真实前缀条件下是否学会了 next-token prediction
3. AR 一旦脱离真实前缀，自由生成是否迅速误差累积
4. CAD 后处理对 token 小错误是否非常敏感
```

因此需要把实验拆分成不同层级，而不是只看最终 free-running 结果。

## 2. VQVAE round-trip

### 2.1 定义

VQVAE round-trip 指的是：

```text
真实 CAD 几何
  -> VQVAE encoder
  -> codebook geometry tokens
  -> VQVAE decoder
  -> reconstructed geometry
```

在当前仓库中，最接近这个实验的是：

```text
reconstruct_vqvae_sample.py
```

### 2.2 真实 token 提供情况

这个实验不依赖 AR 自回归模型，因此不存在“给 AR 提供多少真实 token”的问题。

它使用的真实信息包括：

- 真实 face / edge 几何
- 真实 bbox
- 真实 topology
- 真实 face / edge 数量

只有 geometry tokens 是通过 VQVAE encode 得到再 decode 回去的。

### 2.3 最终输出

典型输出包括：

- `face_mse`
- `edge_mse`
- geometry tokens 范围
- round-trip 中间数据
- holistic sequence
- STEP / STL
- `brep_valid`

### 2.4 是否需要反解生成 CAD

需要。

因为这个实验的核心目的之一就是验证：

```text
VQVAE token -> geometry -> CAD 后处理
```

这条链路是否能落成可构造的 B-rep。

### 2.5 实验作用

它回答的是：

- 你训练的 VQVAE 是否已经具备可靠的 geometry tokenization / detokenization 能力
- 在使用真实 bbox / topology 的前提下，后处理能否成功生成 valid CAD

它不能回答的是：

- AR 是否学会了生成 sequence
- 从 `START` 自由生成新 CAD 是否可行

## 3. AR teacher-forced evaluation

### 3.1 定义

AR teacher-forced evaluation 指的是：

对一条真实 sequence，在每个位置都使用“真实前缀”作为条件，评估 AR 对下一个 token 的预测质量。

例如真实序列为：

```text
a b c d e
```

则严格 teacher-forced eval 的过程是：

```text
输入真实 a        -> 预测 b
输入真实 a b      -> 预测 c
输入真实 a b c    -> 预测 d
输入真实 a b c d  -> 预测 e
```

注意：

- 预测出的 `b_hat` 不会替换真实 `b`
- 预测出的 `c_hat` 是在真实 `ab` 条件下得到，不是在 `a b_hat` 条件下得到

### 3.2 真实 token 提供情况

严格 teacher-forced eval 中，当前步之前的全部 token 都是真实 token。

也就是说，在位置 `t` 预测时，前缀 `x_<t` 全是真实值。

### 3.3 最终输出

典型输出包括：

- 每个位置的 logits
- 每个位置是否预测正确
- overall loss
- overall accuracy
- perplexity
- 按 token 类型统计的 accuracy / loss
- 按结构区段统计的表现

例如可以细分为：

- face index tokens
- bbox tokens
- geometry tokens
- `SEP`
- `END`
- face 段
- edge 段

### 3.4 是否需要反解生成 CAD

通常不作为首要输出。

原因是严格 teacher-forced eval 产生的是“局部条件预测结果”，不是一条真正 rollout 出来的完整序列。

例如：

```text
输入真实 a      -> 得到 b_hat
输入真实 a b    -> 得到 c_hat
输入真实 a b c  -> 得到 d_hat
```

这里的 `c_hat` 是基于真实 `b` 得到的，而不是基于 `b_hat` 得到的，所以

```text
a b_hat c_hat d_hat
```

并不是一条严格意义上自洽的生成序列。

因此：

- 不适合把它当成正式生成结果去评价 CAD 效果
- 可以做单点替换、局部扰动敏感性分析

### 3.5 实验作用

它回答的是：

- 这个 AR 权重在真实上下文下到底有没有学会 next-token prediction
- 模型主要在哪一类 token 上犯错
- 错误是均匀分布，还是集中在某个结构阶段

如果 teacher-forced eval 都很差，说明：

- AR 本身还没学好
- 继续只看 free-running 失败并没有太大诊断价值

如果 teacher-forced eval 还可以，但 free-running 很差，说明：

- AR 学到了一部分局部条件分布
- 但一旦吃自己生成的 token，误差会快速累积

## 4. AR free-running generation

### 4.1 定义

AR free-running generation 指的是：

```text
从 START token 开始
  -> 模型生成下一个 token
  -> 再把这个生成 token 作为下一步输入的一部分
  -> 一直滚动到 END 或 max_length
```

在当前仓库中，对应的就是 `generate_brep.py` 的主要流程。

### 4.2 真实 token 提供情况

几乎不提供真实 token。

典型情况是只给：

```text
[START]
```

然后后续全部由模型自己生成。

### 4.3 最终输出

典型输出包括：

- 一条完整生成 sequence
- 重建后的 STEP / STL
- 是否 `brep_valid`
- 成功率
- 序列长度
- 生成失败原因

### 4.4 是否需要反解生成 CAD

需要。

因为这是最终真正的生成实验，它的意义就在于：

```text
从 START 出发
  -> 生成完整 sequence
  -> 反解成 CAD
  -> 检查可构造性与质量
```

### 4.5 实验作用

它回答的是：

- 当前 AR + VQVAE + 后处理整条链路的真正生成能力
- 模型是否能独立生成复杂 CAD
- 最终可构造率、成功率、样本复杂度上限如何

## 5. strict teacher forcing

### 5.1 定义

strict teacher forcing 就是最标准的 teacher forcing。

对每一个预测位置，都使用完整真实前缀作为输入条件。

### 5.2 提供多少真实 token

在位置 `t` 处，提供全部真实前缀 `x_<t`。

因此它不是“给固定数量真实 token”，而是：

- 每一步都给到该步之前的全部真实 token

### 5.3 最终输出

输出主要是 token-level 结果：

- logits
- correctness
- loss
- accuracy
- perplexity

### 5.4 是否需要反解 CAD

不应把 strict teacher forcing 的逐位置预测硬拼成最终完整 CAD 生成结果。

可以做：

- token 分类错误分析
- 位置错误热图
- 复杂样本错误定位
- 局部替换敏感性测试

### 5.5 实验作用

它是最干净的 AR 诊断实验，用于测：

- AR 在理想真实上下文下是否学会了局部条件分布

## 6. partial / half teacher forcing

### 6.1 定义

partial teacher forcing 指的是：

- 先给一段真实前缀
- 从某个切换点开始，后面全部改为自由生成

例如真实序列 `a b c d e`，如果给前 2 个真实 token：

```text
输入真实 a b     -> 预测 c_hat
输入 a b c_hat   -> 预测 d_hat
输入 a b c_hat d_hat -> 预测 e_hat
```

这里一旦离开真实前缀区，后面就用模型自己生成的 token。

### 6.2 提供多少真实 token

有两种主要选择方式：

1. 按固定长度切

例如：

- 前 16 个 token 为真实
- 前 32 个 token 为真实
- 前 128 个 token 为真实

2. 按结构边界切

对当前 CAD holistic sequence，更推荐按结构切，例如：

- 只给 `[START]`
- 给完整第一个 face block
- 给完整 face 段直到 `[SEP]`
- 给前若干 edge block

### 6.3 为什么选择这个数量的真实 token

选择原则不是越多越好，而是看你想定位什么问题。

如果你想看：

- 模型从最开始就会不会崩
  -> 只给 `[START]`

- face 段是否能稳住
  -> 给第一个 face block 或前几个 face block

- edge / topology 段是否才是主要难点
  -> 给完整 face 段直到 `[SEP]`

- 更后段是否只是收尾差
  -> 给更长真实前缀

因此对当前项目，按结构边界选择真实前缀通常比按固定 token 数量更有解释性。

### 6.4 最终输出

输出是一条真正自洽的完整序列：

```text
真实前缀 + 生成后缀
```

这和 strict teacher forcing 不一样，因为后缀确实是在模型自己生成 token 的条件下滚动出来的。

### 6.5 是否需要反解 CAD

需要，而且非常适合反解成 CAD。

因为它最终得到的是一条自洽完整序列，可以严肃地进行：

- token 解析
- VQVAE decoder
- CAD 后处理
- STEP / STL 输出
- valid B-rep 检查

### 6.6 实验作用

它回答的是：

- 模型从哪个结构边界开始自由生成时会崩
- 问题主要在 face 段、edge 段，还是更后面的收尾
- 真实前缀能否显著提高 CAD 成功率

这是连接 strict teacher-forced eval 和 free-running generation 之间最重要的桥梁实验。

## 7. scheduled / mixed forcing

### 7.1 定义

scheduled / mixed forcing 指的是：

- 有时给真实 token
- 有时给模型自己生成的 token

可以按位置随机切换，也可以按预设概率切换。

一个常见形式是：

```text
以概率 p 使用真实 token
以概率 1-p 使用模型 token
```

### 7.2 提供多少真实 token

不是固定长度，而是由策略决定：

- 固定概率混合
- 随训练轮数逐渐减少真实 token 比例
- 指定某些结构区段强制真实、某些区段自由生成

### 7.3 最终输出

如果作为训练策略，输出主要仍然是 loss。

如果作为推理实验，输出是一条“混合条件下 rollout”的序列，但定义和解释会更复杂。

### 7.4 是否需要反解 CAD

可以，但通常不是当前阶段的首选。

原因是：

- 实验解释成本较高
- 一旦结果变好或变差，较难直接判断原因
- 目前更适合先做 strict 和 partial 两类边界更清晰的实验

### 7.5 实验作用

它更偏向研究：

- 如何缓解 exposure bias
- 模型在训练和推理条件不一致时如何过渡

对于你当前项目，它不是第一优先级。

## 8. 各类实验的对比总结

| 实验 | 提供多少真实 token | 最终输出 | 是否得到完整自洽序列 | 是否建议反解 CAD | 主要作用 |
| --- | --- | --- | --- | --- | --- |
| VQVAE round-trip | 不涉及 AR；使用真实几何/拓扑/bbox | MSE、tokens、STEP/STL、brep_valid | 是，但不是 AR 生成序列 | 建议 | 验证 VQVAE 与后处理链路 |
| AR strict teacher-forced eval | 每一步都提供该步前全部真实前缀 | logits、逐位置正确率、loss、accuracy、perplexity | 否 | 一般不建议作为正式生成结果 | 验证 AR 是否学会 next-token prediction |
| AR partial teacher forcing | 前缀真实，后缀自由生成 | 完整 sequence、STEP/STL、brep_valid | 是 | 建议 | 定位模型从哪里开始自由生成会崩 |
| AR free-running generation | 通常只给 `[START]` | 完整 sequence、STEP/STL、brep_valid | 是 | 建议 | 评估真正无条件生成能力 |
| scheduled / mixed forcing | 由混合策略决定 | loss 或混合 rollout 序列 | 视实验定义而定 | 可做但非首选 | 研究 exposure bias 与过渡机制 |

## 9. 对当前项目的建议实验顺序

建议按下面顺序推进，而不是继续只盯着最终随机采样成功率：

### 9.1 第一步：VQVAE round-trip

确认：

- geometry tokenization 是否可靠
- geometry detokenization 是否可靠
- 后处理在真实 topology / bbox 条件下能否得到 valid CAD

这个阶段你已经基本做过。

### 9.2 第二步：AR strict teacher-forced eval

确认：

- 你当前小数据 AR 权重是否已经学到 next-token prediction
- 哪类 token 最难
- 复杂样本是否在某段突然崩

这个阶段重点看 token-level 指标，不以最终 CAD 为主要目标。

### 9.3 第三步：AR partial teacher forcing

建议至少做以下几种切法：

- 只给 `[START]`
- 给真实第一个 face block
- 给完整真实 face 段直到 `[SEP]`
- 给前若干 edge block

然后对这些完整序列进行 CAD 反解，比较成功率和失败模式。

### 9.4 第四步：AR free-running generation

最后再用：

- 纯 `[START]` 开始
- 随机采样

去评估真正无条件生成能力。

这样你就能明确知道：

- 是 AR 根本没学会
- 还是 AR 学到了一些，但一自由滚动就崩
- 还是 CAD 后处理对少量结构错误过于敏感

## 10. 最终输出与是否需要反解 CAD 总结

最简总结如下：

### 10.1 VQVAE round-trip

最终输出：

- MSE
- round-trip 中间数据
- holistic sequence
- STEP / STL
- `brep_valid`

是否需要反解 CAD：

- 需要

### 10.2 AR strict teacher-forced eval

最终输出：

- logits
- 每个位置是否预测正确
- loss / accuracy / perplexity
- 按 token 类型、结构区段统计的诊断结果

是否需要反解 CAD：

- 通常不需要作为主输出
- 不建议把逐位置预测硬拼成正式生成 CAD 结果

### 10.3 AR partial teacher forcing

最终输出：

- 真实前缀 + 生成后缀组成的完整 sequence
- STEP / STL
- `brep_valid`
- 不同切换点下的成功率与失败模式

是否需要反解 CAD：

- 需要

### 10.4 AR free-running generation

最终输出：

- 从 `[START]` 开始生成的完整 sequence
- STEP / STL
- `brep_valid`
- 最终成功率、复杂度、失败原因

是否需要反解 CAD：

- 需要

### 10.5 scheduled / mixed forcing

最终输出：

- 依赖实验定义，可能是 loss，也可能是混合 rollout 的 sequence

是否需要反解 CAD：

- 可以做，但不是当前首要任务
