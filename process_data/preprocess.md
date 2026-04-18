python -B process_brep.py --input /root/autodl-fs/00710004_32879ef7e9e47ed44bd9e64f_step_000.step --output /root/autodl-fs/preprocessed_data/abc_0071_step_v00_qian100_breparg


已经按你确认的方向改好了，两个文件都已更新：

autodl-tmp/AR/BrepARG_m/process_data/deduplicate_cad.py

改为支持 --input_pkl。
输入可兼容三种：
单样本 pkl（你这个 process_brep.py 产物）
旧的 split pkl（含 train）
路径列表 pkl（list[str]）
输出只保留 train 和 val，并且 val == train。
不再生成 test。
autodl-tmp/AR/BrepARG_m/process_data/deduplicate_se_data.py

--data_list 现在同样兼容三种输入（单样本 / split / 路径列表）。
单样本 pkl 可以直接跑，不再依赖必须有 train/test 结构。
保存输出时处理了无目录路径的情况，避免 os.makedirs('') 报错。
另外我已经做过语法检查，两份脚本都通过。

你可以直接用下面命令：

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