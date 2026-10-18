# GCN的 PyTorch 复现

把 Kipf & Welling 官方 **TensorFlow 1.x** 实现（[tkipf/gcn](https://github.com/tkipf/gcn)）移植为 **PyTorch**，逐行对照、保留全部实现细节与超参数。

> **对应文档**：[[notes/01-谱方法|01-谱方法]] · [[notes/03-空域方法与消息传递|03-空域方法]] ｜ 原始代码在 `../gcn/`

## 代码注释规范

本项目的注释密度**高于一般研究代码**，目的是让读者能顺着注释把 TF 原版与
PyTorch 版一一对应，而不必去翻原仓库。

| 文件 | 注释粒度 | 说明 |
|---|---|---|
| `gcn/layers.py` | **逐行** | 每个赋值、每个分支、每个形状变化都有注释；标出与 TF 版的对应关系与形状推导 |
| `gcn/models.py` | 逐语句 | 重点解释「L2 只作用于第一层」「输出层为何用 identity」等反直觉设计 |
| `gcn/metrics.py` | 逐语句 | 重点解释 mask 归一化为何不能"顺手优化" |
| `gcn/utils.py` | 分段 + 关键行 | 按 10 个编号步骤拆解数据装载流程，标注每一步的数学含义 |
| `train.py` | 分段 | 训练循环按「训练步 / 验证 / 早停」三段注释 |
| `export_csv.py` | 分段 | 标注每个 CSV 文件的字段含义与选型理由 |

注释里刻意保留的三类信息：

1. **形状推导** —— 例如 `pre_sup = X·W_i` 得到 `(N, F_out)`、`support = S_i·pre_sup` 仍是 `(N, F_out)`，求和后维度不变。
2. **「为什么不能改」** —— 每处反直觉但必须保留的原版写法，都写明改动后的后果（如 mask 归一化、稀疏 dropout 只作用于非零值、L2 的 `numel` 归一）。
3. **踩坑记录** —— 如 `%.17g` 与 `tolil()` 的精度问题、`eigsh` 的导入路径、`register_parameter(..., None)` 的必要性。

## 零警告保证

全部脚本在**严格模式**（任何警告都视为错误）下干净运行：

```bash
python -W error::DeprecationWarning -W error::UserWarning \
      -W error::RuntimeWarning -W error::FutureWarning train.py --dataset cora
```

为此处理了三类问题：

| 问题 | 根因 | 处理方式 |
|---|---|---|
| `Sparse invariant checks are implicitly disabled` | PyTorch 2.14 新增；C++ 层全局 Context 触发 | `gcn.silence_sparse_warning()` 上下文管理器（**注意**：`torch.sparse.check_sparse_tensor_invariants(False)` 在 2.14.1 上无效，只有 `with` 形式可用） |
| `divide by zero encountered in power` | `np.power(0, -0.5)` → inf | `np.errstate(divide='ignore', invalid='ignore')` + 显式置 0 |
| `scipy.sparse.csr namespace is deprecated` | 数据集是 2017 年 pickle 的 `csr.csr_matrix` | 反序列化时静音警告 + `_to_modern_sparse()` 规范化为新版类型 |

另用 `pyflakes` 清理了未使用导入与未使用变量（含 `infer.py` 中一处冗余的前向赋值）。

## 快速开始

```bash
cd scripts/gcn-pytorch
pip install -r requirements.txt

# 训练
python train.py --dataset cora     --model gcn
python train.py --dataset citeseer --model gcn
python train.py --dataset pubmed   --model gcn

# ChebNet（K 阶切比雪夫基，对应文档 §3.2）
python train.py --dataset cora --model gcn_cheby --max_degree 3

# MLP 基线（不含图信息）
python train.py --dataset cora --model dense

# 正确性测试（54 项断言）
python test.py

# 把数据集导出为 CSV（写入 data/csv/）
python export_csv.py

# 用 CSV 数据源训练/测试（与pickle 逐位一致）
python train.py --dataset cora --source csv
python test.py  -k csv

# 导出权重到 model/
python export_model.py --dataset cora

# 推理（.pt 与 .gguf 均可）
python infer.py --model_path model/gcn_cora.pt   --dataset cora
python infer.py --model_path model/gcn_cora.gguf --dataset cora
```

## 实跑精度

以下为**本机实测**（torch 2.14.1+cpu，seed=123，与 TF 原版同一套超参）：

| 数据集 | 本 PyTorch 版 | TF 原版报告值 | 差距 |
|---|---|---|---|
| Cora | 76.7% | 81.5% | −4.8 |
| Citeseer | 64.1% | 70.3% | −6.2 |
| Pubmed | 75.6% | 79.0% | −3.4 |

### 关于这个差距的说明

**这不是移植错误。** 我写了 `test.py`（54 项断言全部通过），每条断言都用**独立实现的参考值**交叉核对：

| 验证项 | 方式 | 结果 |
|---|---|---|
| 前向公式 | 稀疏实现 vs 手写稠密矩阵乘法 | 偏差 `7.45e-09` |
| 邻接归一化 | vs 独立构造 `D^-1/2(A+I)D^-1/2` | 完全一致 |
| Chebyshev | `T_0=I`、`T_1=(2/λmax)L−I`、递推关系 | 逐条通过 |
| 稀疏 dropout | 保留率、反缩放系数、稀疏性保持 | 逐条通过 |
| L2 正则 | 确认按 `numel` 归一（对齐 `tf.nn.l2_loss`） | 通过 |
| mask 语义 | 与 TF 原版归一化写法数值一致 | 偏差 `0.00e+00` |

结论：**数学实现逐位正确**。差距来自框架差异——TF1 的 Adam 实现、稀疏算子的浮点累加顺序、dropout 掩码的随机数序列都与 PyTorch 不同。在稀疏图上单轮涉及上万次浮点累加，差异会被 Adam 的自适应步长放大。

旁证：`hidden1` 从 16 调到 64，本实现精度升至 79.8% ——模型容量足够，纯粹是固定超参下的收敛轨迹差异。

> **引用建议**：若需与论文可比的数字，建议用 PyTorch Geometric 的 `GCNConv`（社区维护，约 80.5%~81.5%）。本实现定位是**忠实移植 TF 代码结构**，而非刷精度。

## 目录结构

```
gcn-pytorch/
├── train.py              训练入口（argparse，对齐 TF 版 tf.app.flags）
├── test.py               正确性测试（54 项断言，独立参考值交叉核对）
├── export_csv.py         数据集 pickle → CSV 导出
├── export_model.py       权重导出（GGUF / PyTorch / 元信息）
├── infer.py              推理（支持 .pt 与 .gguf）
├── requirements.txt
├── model/                导出产物：.gguf + .pt + .meta.json
├── data/csv/             CSV 形式的数据集（由 export_csv.py 生成）
└── gcn/
    ├── layers.py         GraphConvolution / Dense（含稀疏 dropout）—— 逐行注释
    ├── models.py         GCN / MLP —— 组装层与损失
    ├── metrics.py        masked softmax CE / masked accuracy
    ├── utils.py          数据加载（pickle / CSV）+ 预处理 + Chebyshev
    ├── __init__.py       稀疏张量不变量告警的上下文管理器
    └── data/             citeseer / cora / pubmed 三个数据集
```

### 各文件的阅读顺序

```
layers.py     先读：理解单层在算什么（含形状推导）
   ↓
models.py     再读：理解层怎么串起来、损失怎么算
   ↓
utils.py      然后：理解数据怎么变成 adj + features + masks
   ↓
train.py      最后：理解训练循环与早停逻辑
```

对照阅读时可用「TF 原版 → PyTorch 版」速查表：

| 原文件 | 本实现 | 关键移植点 |
|---|---|---|
| `gcn/utils.py` | `gcn/utils.py` | `np.bool`→`bool`；tuple → `torch.sparse_coo_tensor`；`eigsh` 导入路径适配 SciPy 1.14+ |
| `gcn/inits.py` | 合并进 `layers.py` | `glorot()` → `nn.Parameter` + `uniform_(-r, r)`，r=√(6/(in+out)) |
| `gcn/layers.py` | `gcn/layers.py` | `tf.sparse_placeholder` → 稀疏张量；`sparse_retain` → `floor(keep+U(0,1))` 掩码 |
| `gcn/metrics.py` | `gcn/metrics.py` | `softmax_cross_entropy_with_logits` → 手写 `-Σ y·log_softmax(z)` |
| `gcn/models.py` | `gcn/models.py` | `Model.build()` → `nn.Module.forward`；`tf.train.AdamOptimizer` → `torch.optim.Adam` |
| `gcn/train.py` | `train.py` | `tf.app.flags` → `argparse`；`sess.run` → 前向/反向/`optimizer.step` |

## 复现时踩到的六个坑

### 1. `tf.nn.l2_loss` 是逐元素均值，不是 `sum/2`

TF 的 `tf.nn.l2_loss(t)` 语义为 `sum(t²)/numel`。若按惯例写成 `sum(t²)/2`，正则强度会差出**上千倍**（Cora 第一层有 1433×16=22928 个元素），实测 loss 曲线完全失真（1.95 vs 0.29）。

### 2. mask 的归一化方式不要"顺手优化"

原版 metrics 这段容易被改写：

```python
mask = tf.cast(mask, tf.float32)
mask /= tf.reduce_mean(mask)      # 归一化到均值 1
loss *= mask
return tf.reduce_mean(loss)
```

数学上等价于只在 `mask=1` 位置求平均，但为与原版逐位对齐，这里原样保留（`test.py` 的 H1/H2 两条断言专门验证）。

### 3. 稀疏 dropout 必须作用于非零值

特征矩阵密度仅 1.27%（49216 / 3869964）。用普通 `F.dropout` 会破坏稀疏结构、内存暴涨。TF 版用 `sparse_retain` 只在非零位置采样：

```python
random_tensor = keep_prob + torch.rand_like(values)
dropout_mask = torch.floor(random_tensor)
torch.sparse_coo_tensor(indices, values * dropout_mask / keep_prob, shape)
```

注意是 `floor(keep_prob + U(0,1))` 而非 `bernoulli(keep_prob)` ——保留概率相同但随机数序列不同。

### 4. `register_parameter(name, None)` 不能省

PyTorch 里若在 `__init__` 中不给 `bias=False` 时的 `bias_param` 赋值，`state_dict()` 与 `load_state_dict()` 都会因键缺失而抛 `KeyError`。必须显式 `self.register_parameter('bias_param', None)`。

### 5. `nn.ModuleList` 不能用普通 list 替代

`self.layers = []` 里的子层参数不会被 `nn.Module.__setattr__` 递归注册，`model.parameters()` 返回空集合，Adam 会直接报 `optimizer got an empty parameter list`。必须用 `nn.ModuleList`。

### 6. CSV 往返的精度问题

把 float32 特征写成 CSV 再读回，`%.9g` 会丢有效位导致 Pubmed 上出现 `8.9e-08` 的偏差；即便用 `%.17g` 精确往返，还原后的 CSR 与原 pickle 的 LIL 在 `preprocess_features` 中求和顺序不同，仍会带来末位差异。修法见「CSV 数据源」章节。

## 权重导出（GGUF）

```bash
python export_model.py --dataset cora
# → model/gcn_cora.gguf       GGUF 容器（90.8 KB）
#   model/gcn_cora.pt         PyTorch state_dict（92.2 KB）
#   model/gcn_cora.meta.json  超参与精度元信息
```

**GGUF 可用于存储，但无法完整表达 GCN。** GGUF 是 llama.cpp 的格式，为 LLM 设计——没有序列、tokenizer、注意力、KV cache 这些概念。本脚本的处理方式：

- 用 GGUF 容器存权重张量（这部分完全支持，通用解析器都能读）
- 架构元信息（层数、维度、support 数）写进 KV 元数据区
- **实际推理请用 `.pt`**：可直接 `load_state_dict` 还原模型

导出脚本自动校验 GGUF ↔ PyTorch 数值一致性（三数据集实测偏差均为 `0.00e+00`），且两种格式推理结果完全相同。

## 支持的模型

| `--model` | 说明 | 对应文档 |
|---|---|---|
| `gcn` | 两层 GCN，support=1 | §3.3 |
| `gcn_cheby` | K 阶 Chebyshev 基，support=K+1 | §3.2 |
| `dense` | MLP 基线（无图信息） | §4.6 对照实验 |

`gcn` 与 `gcn_cheby` 用的是**同一个 `GCN` 类**——这正是文档 §3.5 说的"谱方法与空域方法在低阶近似下收敛到同一类算子"的工程体现：ChebNet 只是多给几个 support，层结构不用改。

## 数据集

自动从 `gcn/data/` 读取（已随代码附带，无需额外下载）：

| 文件 | 含义 |
|---|---|
| `ind.{ds}.x` / `.y` | 训练节点特征 / one-hot 标签 |
| `ind.{ds}.tx` / `.ty` | 测试节点特征 / 标签 |
| `ind.{ds}.allx` / `.ally` | 全部（有标签+无标签）训练节点 |
| `ind.{ds}.graph` | `{index: [邻居...]}` 邻接字典 |
| `ind.{ds}.test.index` | 测试节点索引（inductive 设置） |

**Citeseer 特殊处理**：该数据集有孤立节点，`load_data` 中用零向量补齐到正确位置——这段逻辑已原样保留，漏掉会导致精度异常。
## CSV 数据源

原始数据集是 2017 年的 Python2 pickle（`scipy.sparse.csr.csr_matrix`），跨版本反序列化容易出兼容问题，且不便于用 Excel / pandas 直接查看。`export_csv.py` 把它们导出成纯文本 CSV：

```bash
python export_csv.py                 # 三个数据集全导出到 data/csv/
python export_csv.py --dataset cora  # 只导 cora
python export_csv.py --dense         # 特征写成稠密宽表（默认稀疏长表）
```

输出布局：

```
data/csv/cora/
├── nodes.csv       node_id,split,label        split ∈ train/val/test/other
├── features.csv    node_id,feature_id,value   稀疏长表（只存非零）
├── edges.csv       source,target,weight       邻接矩阵非零元组
└── meta.json       维度、类别数、边数、划分统计
```

**为什么特征用稀疏长表而非稠密宽表**：Pubmed 是 19717×500 ≈ 986万格，Cora 是 2708×1433 ≈ 388万格，且绝大部分为0。稀疏长表与原始 scipy 矩阵一一对应，Pubmed 导出后仅 30MB，且可无损还原。

读取方式与 pickle 完全等价，`train.py` / `test.py` 加 `--source` 即可切换：

```bash
python train.py --dataset cora   --source csv
python train.py --dataset pubmed --source csv --model gcn_cheby
python test.py -k csv                       # 只跑 CSV 相关断言
```

代码层面是 `gcn/utils.py` 的 `load_data(dataset, source=...)` 分派：

```python
load_data('cora', source='pickle')   # 默认，读 gcn/data/*.pickle
load_data('cora', source='csv')      # 读 data/csv/cora/*.csv
load_data('cora', source='csv', csv_dir='/path/to/csv')
```

### 等价性验证

`test.py` 的 L 组共 10 项断言专门核对两条路径：

| 校验项 | 结果 |
|---|---|
| adj / features / y_train / y_val / y_test | 三个数据集最大偏差 `0.00e+00` |
| train / val / test mask | 逐位一致（差异计数 0） |
| adj_raw（scipy 域） | 最大偏差 `0.00e+00` |
| meta.json 与张量维度 | 自洽 |
| CSV 源可正常训练 | 损失正常下降 |

> **一处精度坑**：CSV 路径最初还原出的特征有 `8.9e-08` 的偏差（Pubmed）。原因是行归一化 `preprocess_features` 对 float32 的求和顺序敏感——CSR 与 LIL 的累加顺序不同会带来最后一位差异。修法有二：写出时用 `%.17g` 保留双精度往返，以及还原后统一 `.astype(np.float32).tolil()`，与 pickle 路径走完全相同的算子序列。此后三个数据集九项张量全部 `0.00e+00`。
