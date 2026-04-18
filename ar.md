AR 模型内部没有做任何特殊的数据处理。 它就是一个标准的 GPT-2，把整个 token 序列当作纯粹的"语言"来做 next-token prediction。

下面按顺序说清楚 token 进入 AR 后经历的每一步变换：

训练阶段（trainer.py ARTrainer.train_one_epoch）：

步骤	操作	维度	值范围	说明
1
ARData.__getitem__
[L] int64
[0, 10293]
原样取出 token 序列，没有任何预处理
2
collate_fn padding
[B, L_max] int64
[0, 10293]
短序列用 PAD_TOKEN(10293) 右填充到 batch 内最长
3
labels = input_ids.clone()
[B, L_max] int64
[-100, 10293]
PAD 位置设为 -100（CrossEntropy 忽略位）
4
GPT2 wte (token embedding lookup)
[B, L, 256] float32
连续浮点
10294 个 embedding 向量，每个 256 维
5
GPT2 wpe (position embedding)
[B, L, 256] float32
连续浮点
按位置索引查表，加到 token embedding 上
6
Embedding dropout
[B, L, 256] float32
同上
dropout=0.1
7
8 层 Transformer Block (每层)
[B, L, 256] float32
连续浮点
LayerNorm -> Causal Attention -> Residual -> LayerNorm -> MLP(256->1024->256) -> Residual
8
Final LayerNorm
[B, L, 256] float32
连续浮点
9
lm_head (线性投影)
[B, L, 10294] float32
连续浮点
256 维映射到 vocab_size=10294 维的 logits
10
CrossEntropyLoss
scalar float32
logits vs labels，忽略 -100 位置
推理阶段（generate_brep.py generate_sequence）：

步骤	操作	维度	说明
1
prompt = [START_TOKEN]
[1, 1] int64
只给一个起始 token
2
同上 4-9 步
[1, t, 10294]
每步只算最新 token 的 logits（用 KV cache 加速）
3
Temperature + Top-p 采样
[1] int64
从 logits 采样一个 token（temperature=0.6, top_p=0.6）
4
循环 2-3
逐步增长
直到生成 END_TOKEN 或达到 max_length
关键结论：

模型内部不区分 token 类型。面索引 token、几何 token、位置 token、特殊 token 对模型来说没有任何区别，全部通过同一个 embedding table 查表。模型需要从数据中自己"学会"不同位置应该生成什么范围的 token。

没有自定义的中间处理层。从 model.py 第 52-69 行可以看到，forward 直接调用 GPT2LMHeadModel，没有添加任何额外逻辑：


model.py
Lines 52-69
    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
数据集侧也没有预处理。dataset.py 的 __getitem__ 里注释掉了 _reindex（cyclic offset 已在 2sequence.py 阶段完成），直接返回原始 token：

dataset.py
Lines 289-309
    def __getitem__(self, idx):
        group = self.groups[idx]
        // ... select original or augmented ...
        input_ids = torch.tensor(sample["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(sample["attention_mask"], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}
所以，token 从 2sequence.py 输出到 AR 训练/推理，中间没有任何额外的数据变换。维度变化只发生在 GPT-2 标准的 embedding -> transformer -> logits 这条线上。