# 第二阶段记录：5样本 Prefix 过拟合验证与 Cross-Attention 决策

这份文档记录第二阶段的工作：

1. 完成 5 样本图像条件 prefix 版 AR 过拟合
2. 用 teacher-forced、free-running、partial teacher forcing、CAD 重建对比验证链路
3. 基于结果判断是否进入 cross-attention 阶段

## 这一阶段做了什么

### 1. 实现了 prefix 条件版 AR

核心思路：

- 冻结 DINOv2 图像特征
- 将图像 patch tokens 投影成固定数量的 prefix embeddings
- 将 prefix embeddings 拼到 AR 输入前面

涉及代码：

- [model.py](/workspace/BrepARG_image/model.py)
- [train_ar.py](/workspace/BrepARG_image/train_ar.py)
- [trainer.py](/workspace/BrepARG_image/trainer.py)
- [utils.py](/workspace/BrepARG_image/utils.py)

### 2. 构建了 5 样本过拟合集

数据文件：

- [data/sequence_overfit_5.pkl](/workspace/BrepARG_image/data/sequence_overfit_5.pkl)
- [data/image_feature_index_overfit_5.pkl](/workspace/BrepARG_image/data/image_feature_index_overfit_5.pkl)

样本选择原则：

- 从 `sequence_mini_100.pkl` 的 train split 中选序列较长、相对复杂的样本
- 这 5 个样本同时放进 train 和 val，用于极限过拟合验证

### 3. 完成了 one-batch smoke test

目标：

- 单卡 `batch_size=1` 跑通一次 `forward/backward`
- 检查 `shape / dtype / mask / device`

结果：

- prefix 条件链路成功接入训练
- `image_features` 正常进入模型
- 训练步成功完成，没有 shape 或 device 错误

### 4. 完成了 1000 epoch 的 5 样本正式过拟合训练

训练目录：

- [checkpoints/ar_prefix_overfit5](/workspace/BrepARG_image/checkpoints/ar_prefix_overfit5)

关键产物：

- [checkpoints/ar_prefix_overfit5/abc_ar_vqvae_best_model.pt](/workspace/BrepARG_image/checkpoints/ar_prefix_overfit5/abc_ar_vqvae_best_model.pt)
- [checkpoints/ar_prefix_overfit5/epoch_1000.pt](/workspace/BrepARG_image/checkpoints/ar_prefix_overfit5/epoch_1000.pt)

## 评估结果

### 1. Strict Teacher-Forced

结果文件：

- [checks/teacher_forced_overfit5_val.json](/workspace/BrepARG_image/checks/teacher_forced_overfit5_val.json)

关键结果：

- `Accuracy = 1.0`
- `Top-5 Accuracy = 1.0`
- `Face Accuracy = 1.0`
- `Edge Accuracy = 1.0`
- `Perplexity ≈ 1.0`

结论：

- 在真实图片条件和真实前缀下，5 个样本已经被完整记住
- prefix 版图像条件 AR 在这个小集合上已经完全过拟合成功

### 2. Free-Running

结果文件：

- [checks/overfit5_sample0_generation.json](/workspace/BrepARG_image/checks/overfit5_sample0_generation.json)

后续判断：

- 单独看最早 free-running 结果容易误判
- 因为必须结合 prefix 条件和 continuation 对齐方式来解释
- 因此后面重点依赖 partial teacher forcing 来定位自回归起步是否稳定

### 3. Partial Teacher Forcing

结果文件：

- [checks/partial_teacher_forcing_sample0.json](/workspace/BrepARG_image/checks/partial_teacher_forcing_sample0.json)

样本：

- `00387821_fe7376ac8dcc8bbc715c8909_step_000`

三种前缀条件下的 continuation 匹配长度：

- `start_only`: `1007`
- `first_face_block`: `996`
- `through_sep`: `588`

结论：

- 模型并不是“只要 free-running 就立刻乱掉”
- 即使只给 `image prefix + [START]`，也已经能正确续写很长一段
- 图像条件确实在起作用，而且 AR 主干已经学会根据图像条件起步生成

### 4. Target / Generated CAD 重建对比

结果目录：

- [checks/reconstruct_compare_sample0](/workspace/BrepARG_image/checks/reconstruct_compare_sample0)

关键文件：

- [checks/reconstruct_compare_sample0/compare_metrics.json](/workspace/BrepARG_image/checks/reconstruct_compare_sample0/compare_metrics.json)
- [checks/reconstruct_compare_sample0/target.step](/workspace/BrepARG_image/checks/reconstruct_compare_sample0/target.step)
- [checks/reconstruct_compare_sample0/generated.step](/workspace/BrepARG_image/checks/reconstruct_compare_sample0/generated.step)

关键结果：

- `target.brep_valid = true`
- `generated.brep_valid = true`

结论：

- 不只是 token 层面过拟合成功
- 生成序列已经可以成功重建出有效 CAD
- 这说明“图像 -> prefix 条件 -> AR 序列 -> CAD 重建”这一条小闭环已经跑通

## 这一阶段的整体结论

这阶段已经证明：

1. prefix 条件注入路线可行
2. 5 样本图像条件 AR 可以完整过拟合
3. strict teacher-forced、partial teacher forcing、CAD 重建三条证据彼此一致
4. 当前 prefix 方案已经足够作为后续图像条件 AR 的稳定基线

## 是否还要推进到 100 样本 / 10000 样本

当前判断：

- 不需要再花太多时间做 prefix 版的 100 样本系统实验
- 也不建议现在直接上 10000 样本正式大训练

原因：

- 5 样本已经证明链路通了
- 现在的主要问题不再是“prefix 能不能工作”
- 而是“cross-attention 是否能比 prefix 更强、更稳”

## 下一阶段建议

推荐顺序：

1. 开始实现 cross-attention 版模型
2. 先在 5 样本上做 one-batch smoke test 和极限过拟合
3. 再在 100 样本上比较 prefix 与 cross-attention
4. 如果 cross-attention 在 100 样本上明显更好，再考虑 10000 样本正式训练

因此，下一阶段的重点不再是重复 prefix 基线，而是进入 cross-attention 结构验证。
