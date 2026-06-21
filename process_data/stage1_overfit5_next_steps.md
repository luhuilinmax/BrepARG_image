# 5样本过拟合与单样本生成检查

## 正式过拟合训练命令

```bash
source /workspace/scripts/setup_offline_env.sh
python train_ar.py   --sequence_file /workspace/BrepARG_image/data/sequence_overfit_5.pkl   --image_feature_index_file /workspace/BrepARG_image/data/image_feature_index_overfit_5.pkl   --dataset_type abc   --batch_size 1   --train_epoch 1000   --test_epoch 20   --save_epoch 100   --max_seq_len 4096   --learning_rate 1e-4   --weight_decay 0.0   --dropout 0.0   --label_smoothing 0.0   --d_model 256   --nhead 8   --num_layers 8   --dim_feedforward 1024   --use_image_prefix   --image_feature_dim 1024   --num_image_prefix_tokens 16   --env ar_prefix_overfit5   --tb_log_dir /workspace/BrepARG_image/logs/ar_prefix_overfit5   --dir_name /workspace/BrepARG_image/checkpoints
```

## 更激进的对照配置

```bash
source /workspace/scripts/setup_offline_env.sh
python train_ar.py   --sequence_file /workspace/BrepARG_image/data/sequence_overfit_5.pkl   --image_feature_index_file /workspace/BrepARG_image/data/image_feature_index_overfit_5.pkl   --dataset_type abc   --batch_size 1   --train_epoch 1000   --test_epoch 20   --save_epoch 100   --max_seq_len 4096   --learning_rate 3e-4   --weight_decay 0.0   --dropout 0.0   --label_smoothing 0.0   --d_model 512   --nhead 8   --num_layers 8   --dim_feedforward 2048   --use_image_prefix   --image_feature_dim 1024   --num_image_prefix_tokens 16   --env ar_prefix_overfit5_fast   --tb_log_dir /workspace/BrepARG_image/logs/ar_prefix_overfit5_fast   --dir_name /workspace/BrepARG_image/checkpoints
```

## 单样本 free-running 检查

使用脚本：

- [check_single_image_conditioned_generation.py](/workspace/BrepARG_image/check_single_image_conditioned_generation.py)

示例：

```bash
source /workspace/scripts/setup_offline_env.sh
python check_single_image_conditioned_generation.py   --sequence_file /workspace/BrepARG_image/data/sequence_overfit_5.pkl   --image_feature_index_file /workspace/BrepARG_image/data/image_feature_index_overfit_5.pkl   --weight /workspace/BrepARG_image/checkpoints/ar_prefix_overfit5_smoke50/abc_ar_vqvae_best_model.pt   --sample_index 0   --split val   --max_seq_len 4096   --max_new_tokens 256   --device cuda   --output_json /workspace/BrepARG_image/checks/overfit5_sample0_generation.json
```
