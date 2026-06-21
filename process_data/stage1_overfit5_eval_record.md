# 5样本过拟合后评估记录

这份文档记录 5 样本图像条件 AR 过拟合完成后的三类核心实验：

1. strict teacher-forced
2. free-running
3. partial teacher forcing

以及一个补充验证：target / generated 序列的 CAD 重建对比。

## 实验背景

训练数据：

- [data/sequence_overfit_5.pkl](/workspace/BrepARG_image/data/sequence_overfit_5.pkl)
- [data/image_feature_index_overfit_5.pkl](/workspace/BrepARG_image/data/image_feature_index_overfit_5.pkl)

AR checkpoint：

- [/workspace/BrepARG_image/checkpoints/ar_prefix_overfit5/abc_ar_vqvae_best_model.pt](/workspace/BrepARG_image/checkpoints/ar_prefix_overfit5/abc_ar_vqvae_best_model.pt)

VQ-VAE checkpoint：

- `/workspace/BrepARG_m_ddp_fix/checkpoints_bs768/vqvae_mmap_bs768/abc_se_vqvae_epoch_600.pt`

条件注入方式：

- 冻结 DINOv2 特征
- 将图像 patch tokens 投影成 `16` 个 prefix embeddings
- 拼到 AR 输入前面

## 1. Strict Teacher-Forced

脚本：

- [teacher_forced_eval_ar.py](/workspace/BrepARG_image/teacher_forced_eval_ar.py)

执行命令：

```bash
source /workspace/scripts/setup_offline_env.sh
python teacher_forced_eval_ar.py   --sequence_file /workspace/BrepARG_image/data/sequence_overfit_5.pkl   --image_feature_index_file /workspace/BrepARG_image/data/image_feature_index_overfit_5.pkl   --weight /workspace/BrepARG_image/checkpoints/ar_prefix_overfit5/abc_ar_vqvae_best_model.pt   --split val   --batch_size 1   --max_seq_len 4096   --device cuda   --output_json /workspace/BrepARG_image/checks/teacher_forced_overfit5_val.json   --save_worst_k 5   --error_top_k 20
```

结果文件：

- [checks/teacher_forced_overfit5_val.json](/workspace/BrepARG_image/checks/teacher_forced_overfit5_val.json)

关键结果：

- `Samples: 5 / 5`
- `Tokens: 8638`
- `Loss: 0.001637`
- `Perplexity: 1.001639`
- `Accuracy: 1.000000`
- `Top-5 Accuracy: 1.000000`
- `Face Accuracy: 1.0`
- `Edge Accuracy: 1.0`

结论：

- 在“真实图片条件 + 真实历史 token”下，模型已经完整记住这 5 个样本
- 数据、prefix 条件、AR 主干、训练和评估链路都已经打通

## 2. Free-Running

脚本：

- [check_single_image_conditioned_generation.py](/workspace/BrepARG_image/check_single_image_conditioned_generation.py)

执行命令：

```bash
source /workspace/scripts/setup_offline_env.sh
python check_single_image_conditioned_generation.py   --sequence_file /workspace/BrepARG_image/data/sequence_overfit_5.pkl   --image_feature_index_file /workspace/BrepARG_image/data/image_feature_index_overfit_5.pkl   --weight /workspace/BrepARG_image/checkpoints/ar_prefix_overfit5/abc_ar_vqvae_best_model.pt   --sample_index 0   --split val   --max_seq_len 4096   --max_new_tokens 256   --device cuda   --output_json /workspace/BrepARG_image/checks/overfit5_sample0_generation.json
```

结果文件：

- [checks/overfit5_sample0_generation.json](/workspace/BrepARG_image/checks/overfit5_sample0_generation.json)

说明：

- 这里的 free-running 并不是“不喂图片”
- 实际输入是：`image prefix + [START]`
- 也就是给了图像条件，但文本起点只给 `START_TOKEN`

结论：

- 单看最早版本的 free-running 对比结果容易误判，因为比较口径没有处理 prefix / continuation 对齐
- 后续 partial teacher forcing 结果表明，模型实际上已经能从图像条件起步生成长段正确 continuation
- 因此 free-running 的解释必须结合 continuation 对齐后的结果看，不能只看“第一个位置是否相同”

## 3. Partial Teacher Forcing

脚本：

- [partial_teacher_forcing_image_prefix.py](/workspace/BrepARG_image/partial_teacher_forcing_image_prefix.py)

执行命令：

```bash
source /workspace/scripts/setup_offline_env.sh
python partial_teacher_forcing_image_prefix.py   --sequence_file /workspace/BrepARG_image/data/sequence_overfit_5.pkl   --image_feature_index_file /workspace/BrepARG_image/data/image_feature_index_overfit_5.pkl   --weight /workspace/BrepARG_image/checkpoints/ar_prefix_overfit5/abc_ar_vqvae_best_model.pt   --sample_index 0   --split val   --max_seq_len 4096   --device cuda   --max_generate_len 1024   --output_json /workspace/BrepARG_image/checks/partial_teacher_forcing_sample0.json
```

结果文件：

- [checks/partial_teacher_forcing_sample0.json](/workspace/BrepARG_image/checks/partial_teacher_forcing_sample0.json)

评估口径：

- 只比较“真实 prefix 之后”的 continuation
- 不再混淆真实前缀与新生成段

样本：

- `cad_stem = 00387821_fe7376ac8dcc8bbc715c8909_step_000`
- `target_len = 1681`
- `sep_pos = 419`

三种前缀条件结果：

1. `start_only`
- `prefix_len = 1`
- `continuation_match_len = 1007`

2. `first_face_block`
- `prefix_len = 12`
- `continuation_match_len = 996`

3. `through_sep`
- `prefix_len = 420`
- `continuation_match_len = 588`

结论：

- 模型不是“只要 free-running 就立刻乱掉”
- 在 `image prefix + [START]` 条件下，已经可以正确续写非常长的一段
- 说明图像条件确实起作用，而且 AR 主干已经在这个 5 样本任务上学会了从图像条件起步生成

## 4. Target / Generated 重建对比

脚本：

- [compare_target_vs_generated_reconstruction.py](/workspace/BrepARG_image/compare_target_vs_generated_reconstruction.py)

执行命令：

```bash
source /workspace/scripts/setup_offline_env.sh
CUDA_VISIBLE_DEVICES=0 python compare_target_vs_generated_reconstruction.py   --sequence_file /workspace/BrepARG_image/data/sequence_overfit_5.pkl   --image_feature_index_file /workspace/BrepARG_image/data/image_feature_index_overfit_5.pkl   --weight /workspace/BrepARG_image/checkpoints/ar_prefix_overfit5/abc_ar_vqvae_best_model.pt   --se_vqvae /workspace/BrepARG_m_ddp_fix/checkpoints_bs768/vqvae_mmap_bs768/abc_se_vqvae_epoch_600.pt   --sample_index 0   --split val   --max_seq_len 4096   --device cuda   --dataset_type abc   --max_generate_len 2048   --output_dir /workspace/BrepARG_image/checks/reconstruct_compare_sample0
```

结果目录：

- [checks/reconstruct_compare_sample0](/workspace/BrepARG_image/checks/reconstruct_compare_sample0)

关键结果：

- `target_len = 1681`
- `generated_len = 1680`
- `target.brep_valid = true`
- `generated.brep_valid = true`

关键文件：

- [checks/reconstruct_compare_sample0/compare_metrics.json](/workspace/BrepARG_image/checks/reconstruct_compare_sample0/compare_metrics.json)
- [checks/reconstruct_compare_sample0/target.step](/workspace/BrepARG_image/checks/reconstruct_compare_sample0/target.step)
- [checks/reconstruct_compare_sample0/generated.step](/workspace/BrepARG_image/checks/reconstruct_compare_sample0/generated.step)

结论：

- 不只是 token 层面过拟合成功
- 在这个样本上，图像条件 AR 生成出的序列已经足以成功重建出有效 STEP / STL
- 因此“图像 -> prefix 条件 -> AR 序列 -> CAD 重建”的小闭环已经跑通

## 总结

这一阶段的 5 样本实验已经支持下面几个结论：

1. prefix 条件注入路线可行
2. 5 样本上 strict teacher-forced 已经达到 100% 准确率
3. partial teacher forcing 表明模型已经可以从图像条件起步生成长段正确 continuation
4. 至少在样本 0 上，generated sequence 已经能成功重建出有效 CAD

因此，当前工作已经足以支持进入下一阶段：

- 扩到 100 样本做正式图像条件 AR 训练 / 评估
- 当前 prefix 方案可以作为第一版稳定基线
- 后续如果需要，再评估是否升级到 cross-attention 路线
