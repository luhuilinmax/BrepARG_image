# AutoRegressive Generation with B-rep Holistic Token Sequence Representation（CVPR 2026）
[CVPR 2026] Official PyTorch Implementation of "AutoRegressive Generation with B-rep Holistic Token Sequence Representation".
<img width="1476" height="708" alt="image" src="https://github.com/user-attachments/assets/0c2bec0e-fbd3-43ec-b2bf-f7533cc76d8c" />

# 其他
START_TOKEN = 10290，
SEP_TOKEN = 10291，
END_TOKEN = 10292，
PAD_TOKEN = 10293

# environment
We will provide a Conda environment package later.
```python
conda create --name breparg python=3.10
conda activate breparg

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

# process data
```python
python process_brep.py
python deduplicate_cad.py
python deduplicate_se_data.py

# 1) CAD 去重（输入你的单样本 pkl，输出 train/val 一致的 split pkl）
python /root/autodl-tmp/AR/BrepARG_m/process_data/deduplicate_cad.py \
  --input_pkl /root/autodl-fs/preprocessed_data/abc_0071_step_v00_qian100_breparg/00710004_32879ef7e9e47ed44bd9e64f_step_000.pkl \
  --bit 6 \
  --option abc \
  --output /root/autodl-tmp/AR/BrepARG_m/process_data/data/abc_single_data_split_6bit.pkl

# 2) SE 去重（face）
python /root/autodl-tmp/AR/BrepARG_m/process_data/deduplicate_se_data.py \
  --data_list /root/autodl-tmp/AR/BrepARG_m/process_data/data/abc_single_data_split_6bit.pkl \
  --mode face \
  --bit 6 \
  --output /root/autodl-tmp/AR/BrepARG_m/process_data/data/abc_single_unique_surfaces.pkl

# 3) SE 去重（edge）
python /root/autodl-tmp/AR/BrepARG_m/process_data/deduplicate_se_data.py \
  --data_list /root/autodl-tmp/AR/BrepARG_m/process_data/data/abc_single_data_split_6bit.pkl \
  --mode edge \
  --bit 6 \
  --output /root/autodl-tmp/AR/BrepARG_m/process_data/data/abc_single_unique_edges.pkl
```

# training
**VQVAE:** --batch_size (Bigger is better) --train_epoch (Adjust according to the data volume)
```python
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_vqvae.py --data_list 'your own data paths' --surface_list 'deduplicated surface source data' --edge_list 'deduplicated edge source data' --batch_size 512 --train_epoch 3000

# 我改过代码后用的命令
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
cd /root/autodl-tmp/AR/BrepARG_m

python train_vqvae.py \
  --data_list /root/autodl-tmp/AR/BrepARG_m/process_data/data/abc_single_data_split_6bit.pkl \
  --surface_list /root/autodl-tmp/AR/BrepARG_m/process_data/data/abc_single_unique_surfaces.pkl \
  --edge_list /root/autodl-tmp/AR/BrepARG_m/process_data/data/abc_single_unique_edges.pkl \
  --dataset_type abc \
  --batch_size 8 \
  --train_epoch 20 \
  --test_epoch 1 \
  --save_epoch 5 \
  --env vqvae_single_gpu_debug \
  --dir_name checkpoints \
  --tb_log_dir logs/vqvae_single_gpu_debug
```

**AR:**
1. Prepare the AR data:

max_face：限制每个 CAD 的总 face 数，超过就直接 return None 丢掉。
max_edge：限制每个 CAD 的总 edge 数，超过就丢掉。

```python
python 2sequence.py

# 我改过代码后用的命令
python 2sequence.py \
  --data_list /root/autodl-tmp/AR/BrepARG_m/process_data/data/abc_single_data_split_6bit.pkl \
  --vqvae_se_weight /root/autodl-tmp/AR/BrepARG_m/checkpoints/vqvae_single_gpu_debug/abc_se_vqvae_best.pt \
  --output_file /root/autodl-tmp/AR/BrepARG_m/data/abc_single_sequences.pkl \
  --max_face 20 \
  --max_edge 60 \
  --scale 1.0 \
  --gpu 0

```
2. Train the autoregressive model:
```python
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_ar.py --sequence_file 'your own sequences path' --batch_size 32 --train_epoch 500 --learning_rate 1e-3

# 我改过代码后用的命令
python train_ar.py \
  --sequence_file /root/autodl-tmp/AR/BrepARG_m/data/abc_single_sequences.pkl \
  --dataset_type abc \
  --batch_size 4 \
  --train_epoch 50 \
  --test_epoch 1 \
  --save_epoch 10 \
  --learning_rate 1e-3 \
  --max_seq_len 2048 \
  --env ar_single_gpu_debug \
  --dir_name checkpoints \
  --tb_log_dir logs/ar_single_gpu_debug1
```

# generating brep
```python
python generate_brep.py
```

## 单卡：基于 abc_multi_data_split_6bit.pkl 的完整命令
```python
# 0) 环境变量
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
cd /root/autodl-tmp/AR/BrepARG_m

# 1) VQ-VAE 训练（多样本）
python train_vqvae.py \
  --data_list /root/autodl-tmp/AR/BrepARG_m/process_data/data/abc_multi_data_split_6bit.pkl \
  --surface_list /root/autodl-tmp/AR/BrepARG_m/process_data/data/abc_multi_unique_surfaces.pkl \
  --edge_list /root/autodl-tmp/AR/BrepARG_m/process_data/data/abc_multi_unique_edges.pkl \
  --dataset_type abc \
  --batch_size 16 \
  --train_epoch 300 \
  --test_epoch 1 \
  --save_epoch 100 \
  --env vqvae_single_gpu_multi \
  --dir_name checkpoints \
  --tb_log_dir logs/vqvae_single_gpu_multi

# 2) 准备 AR 序列数据
python 2sequence.py \
  --data_list /root/autodl-tmp/AR/BrepARG_m/process_data/data/abc_multi_data_split_6bit.pkl \
  --vqvae_se_weight /root/autodl-tmp/AR/BrepARG_m/checkpoints/vqvae_single_gpu_multi/abc_se_vqvae_best.pt \
  --output_file /root/autodl-tmp/AR/BrepARG_m/data/abc_multi_sequences.pkl \
  --max_face 50 \
  --max_edge 150 \
  --scale 1.0 \
  --gpu 0

# 3) 训练 AR（单卡）
python train_ar.py \
  --sequence_file /root/autodl-tmp/AR/BrepARG_m/data/abc_multi_sequences.pkl \
  --dataset_type abc \
  --batch_size 8 \
  --train_epoch 500 \
  --test_epoch 1 \
  --save_epoch 100 \
  --learning_rate 1e-3 \
  --max_seq_len 2048 \
  --env ar_single_gpu_multi \
  --dir_name checkpoints \
  --tb_log_dir logs/ar_single_gpu_multi

# 4) 生成（单样本先验证）
python generate_brep.py \
  --config /root/autodl-tmp/AR/BrepARG_m/config.json \
  --dataset_type abc \
  --ar_model /root/autodl-tmp/AR/BrepARG_m/checkpoints/ar_single_gpu_multi/abc_ar_vqvae_best_model.pt \
  --se_vqvae /root/autodl-tmp/AR/BrepARG_m/checkpoints/vqvae_single_gpu_multi/abc_se_vqvae_best.pt \
  --mode single \
  --max_length 2048 \
  --top_p 0.9 \
  --temperature 1.0 \
  --gpu 0 \
  --output_dir /root/autodl-tmp/AR/BrepARG_m/result/generated_brep/abc_multi_single \
  --filename_prefix abc_multi_single
```

# evaluation

**Valid = success rate * watertight rate**

- **Success rate:** Generated B-reps / Total attempts
- **Watertight Rate:** Watertight models / Generated B-reps

**other Metric:** Follwing BrepGen https://github.com/samxuxiang/BrepGen?tab=readme-ov-file


# Citation
We would like to acknowledge the foundational contributions of the following works:
```bibtex
@article{xu2024brepgen,
  title={BrepGen: A B-rep Generative Diffusion Model with Structured Latent Geometry},
  author={Xu, Xiang and Lambourne, Joseph G and Jayaraman, Pradeep Kumar and Wang, Zhengqing and Willis, Karl DD and Furukawa, Yasutaka},
  journal={arXiv preprint arXiv:2401.15563},
  year={2024}
}
@inproceedings{li2025dtgbrepgen,
  title={Dtgbrepgen: A novel b-rep generative model through decoupling topology and geometry},
  author={Li, Jing and Fu, Yihang and Chen, Falai},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={21438--21447},
  year={2025}
}
```
If you find our work or this paper helpful to your research, please consider citing:
```bibtex
@article{li2026autoregressive,
  title={AutoRegressive Generation with B-rep Holistic Token Sequence Representation},
  author={Li, Jiahao and Bai, Yunpeng and Dai, Yongkang and Guo, Hao and Gan, Hongping and Shi, Yilei},
  journal={arXiv preprint arXiv:2601.16771},
  year={2026}
}
```
