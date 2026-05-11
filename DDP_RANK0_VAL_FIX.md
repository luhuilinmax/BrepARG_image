# DDP 验证集 Rank 0 加载说明

## 背景

原始 `train_vqvae.py` 在 DDP 多卡训练时，每个 rank 都会完整执行数据集初始化：

```text
train_dataset = CombinedData(...)
val_dataset = CombinedData(... validate=True ...)
```

这意味着如果用 5 个 rank 训练，就会同时加载 5 份训练集和 5 份验证集。当前验证集会读取大量 pkl 文件并在内存中堆叠，容易造成 CPU 内存占用过高，最终进程可能被系统 `SIGKILL` 杀掉。

## 本次修改目标

本次修改只针对 VQ-VAE 训练入口，目标是：

```text
所有 rank 正常加载训练集
只有 rank 0 加载验证集
非 rank 0 跳过验证集加载
```

这样可以显著减少 DDP 启动阶段的验证集重复内存占用。

## 修改文件

### `train_vqvae.py`

修改前，所有 rank 都会创建 `val_dataset`。

修改后：

```python
if rank == 0:
    val_dataset = CombinedData(
        args.data_list,
        args.surface_list,
        args.edge_list,
        validate=True,
        aug=False,
        use_type_flag=args.use_type_flag,
    )
else:
    val_dataset = None
```

因此只有 `rank 0` 会进入 `CombinedData(val)`，其他 rank 不再加载验证集。

### `trainer.py`

`VQVAETrainer` 现在支持 `val_dataset=None`：

- 只有 `val_dataset is not None` 时才创建验证集 `DistributedSampler`。
- 只有 `val_dataset is not None` 时才创建 `self.val_dataloader`。
- 否则设置 `self.val_dataloader = None`。
- `test_val()` 增加保护逻辑，如果没有验证 DataLoader 就直接返回。

## 预期日志变化

修改前，多卡训练时可能看到每个 rank 都加载验证集：

```text
[rank 0] CombinedData(val): ...
[rank 1] CombinedData(val): ...
[rank 2] CombinedData(val): ...
[rank 3] CombinedData(val): ...
[rank 4] CombinedData(val): ...
```

修改后，应该只看到：

```text
[rank 0] CombinedData(val): ...
```

不应该再看到非 `rank 0` 的 `CombinedData(val)` 日志。

## 验证方式

在本项目目录下运行：

```bash
cd /data/project/ly/BrepARG_m_ddp_fix
python -m py_compile train_vqvae.py trainer.py
```

本次修改后该语法检查已通过。

训练时建议同时监控 CPU 内存：

```bash
watch -n 5 free -h
```

也可以查看内存占用最高的进程：

```bash
watch -n 5 'ps -eo user,pid,comm,rss,%mem --sort=-rss | head -30'
```

## 注意事项

这次修改只解决验证集在 DDP 下被每个 rank 重复加载的问题。

训练集目前仍然是每个 rank 各加载一份，因此如果 rank 数过多，CPU 内存仍然可能被训练集撑满。若 5 卡训练仍然出现 `SIGKILL` 或内存峰值过高，下一步应继续改造数据加载方式，例如：

- 将训练集也改为 lazy loading。
- 将大数组改为分片读取或 memory-mapped 格式。
- 降低 `nproc_per_node`，先用 2 卡或 3 卡验证稳定性。
- 降低 DataLoader 的 `num_workers`，减少额外进程带来的内存和 IO 压力。

## 2026-05-11 补充修复：DDP 初始化顺序

第一次修改后，验证集只在 `rank 0` 加载，但加载发生在 `VQVAETrainer` 初始化之前。由于 `VQVAETrainer` 内部才会初始化 DDP process group，非 `rank 0` 会先进入 DDP 初始化并等待，而 `rank 0` 仍在加载验证集，最终可能导致其他 rank 超时退出。

```
补充修复前的执行流程
rank 0:
  加载 train_dataset
  加载 val_dataset，很慢，日志显示 21 分钟才到 12%
  还没进入 VQVAETrainer
  还没进入 DDP 初始化

rank 1/2/3/4:
  加载 train_dataset
  跳过 val_dataset
  进入 VQVAETrainer
  初始化 DDP / 包装 DDP 模型
  等 rank 0 一起参与通信
```

本次补充修复调整为：

```text
所有 rank 先加载训练集
所有 rank 先初始化 VQVAETrainer / DDP process group
rank 0 再加载验证集并挂到 trainer 上
其他 rank 在 barrier 等待 rank 0 完成验证集加载
```

同时将 `dist.init_process_group` 的 timeout 调整为 4 小时，避免验证集加载和 rank 0 验证耗时较长时，其他 rank 在 barrier 中过早超时。

训练循环中也增加了验证和保存后的同步：

```text
rank 0 执行验证后，所有 rank barrier
rank 0 保存模型后，所有 rank barrier
```

这样可以避免非 `rank 0` 在 `rank 0` 验证或保存期间提前进入下一轮 DDP 训练。
