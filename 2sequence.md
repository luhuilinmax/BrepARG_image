# 新增内容:

## DataTracer 类（约 140 行，文件顶部）

trace(name, data, note) -- 自动识别 ndarray/Tensor/list/dict/int/float 类型，记录 shape、dtype、min/max/mean
print_report() / save_report(path) -- 格式化表格输出
register_vqvae_hooks(model) -- 为 VQ-VAE 的 encoder/quant_conv/quantize 注册 forward hook

## ARDataPreprocessor 修改:

__init__ 接收 tracer 参数，注册 hook
_process_single_cad 中追踪 6 个原始输入数据
_encode_single_rotation 中追踪约 24 个关键步骤（数据准备、排序、bbox 量化、VQ-VAE 编码、token 拼接）
只追踪第一个成功样本的 rotation=0 编码，之后自动关闭
main() 修改:

新增 --trace 和 --trace_output 参数