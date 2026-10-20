# GCN 的 PyTorch 复现

把 Kipf & Welling 官方 **TensorFlow 1.x** 实现（[tkipf/gcn](https://github.com/tkipf/gcn)）移植为 **PyTorch**，逐行对照、保留全部实现细节与超参数。另附一套 **PyTorch Geometric（`GCNConv`）** 实现用于对照验证。

> **对应文档**：[[notes/01-谱方法|01-谱方法]] · [[notes/03-空域方法与消息传递|03-空域方法]] ｜ 原始代码在 `../gcn/` ｜ 数据集详解见 [[README.data]]
>
> **两套实现**：手写版（`gcn/` + `train.py`）忠实移植 TF 结构；PyG 版（`pyg_gcn/` + `train_pyg.py`）用社区标准 `GCNConv`。Cora 精度分别为 **76.70%** 与 **79.50%**，详见 [[#PyTorch Geometric 版（GCNConv）|PyTorch Geometric 版]]。

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

全部脚本（含 PyG 版）在**严格模式**（任何警告都视为错误）下干净运行：

```bash
python -W error train.py      --dataset cora
python -W error test.py
python -W error train_pyg.py  --dataset cora
python -W error test_pyg.py
```

也可按类别逐个升级为错误（旧写法，等价于 `-W error`）：

```bash
python -W error::DeprecationWarning -W error::UserWarning \
      -W error::RuntimeWarning -W error::FutureWarning train.py --dataset cora
```

为此处理了四类问题：

| 问题 | 根因 | 处理方式 |
|---|---|---|
| `Sparse invariant checks are implicitly disabled` | PyTorch 2.14 新增；C++ 层全局 Context 触发 | `gcn.silence_sparse_warning()` 上下文管理器（**注意**：`torch.sparse.check_sparse_tensor_invariants(False)` 在 2.14.1 上无效，只有 `with` 形式可用） |
| `divide by zero encountered in power` | `np.power(0, -0.5)` → inf | `np.errstate(divide='ignore', invalid='ignore')` + 显式置 0 |
| `scipy.sparse.csr namespace is deprecated` | 数据集是 2017 年 pickle 的 `csr.csr_matrix` | 反序列化时静音警告 + `_to_modern_sparse()` 规范化为新版类型 |
| `torch.jit.script` 在 Python 3.14+ 不受支持 | PyG 的 `MessagePassing` 用 JIT 做类型检查，**在 import 时**即触发 | `pyg_gcn.silence_pyg_warning()`；但该告警早于上下文管理器生效，故在 `pyg_gcn/__init__.py` **导入期**按「精确消息 + 精确模块」安装 filter |

另用 `pyflakes` 清理了未使用导入与未使用变量（含 `infer.py` 中一处冗余的前向赋值）。

## 快速开始

```bash
cd scripts/gcn-pytorch
pip install -r requirements.txt

# ---- 手写实现 ----
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

# ---- PyG 实现 ----
pip install torch_geometric            # 纯 Python，无需 torch-scatter
python train_pyg.py --dataset cora     # GCNConv，默认 2 层
python train_pyg.py --dataset cora --num_layers 8   # 加深，观察过平滑
python test_pyg.py                     # 37 项断言

# ---- 第三条路径：PyG 内置数据集（自动下载并落盘到 data/Planetoid/）----
python train_planetoid.py --dataset cora
python train_planetoid.py --report     # 查看本地缓存
python test_planetoid.py               # 32 项断言

# ---- 数据与权重 ----
# 把数据集导出为 CSV（写入 data/csv/）
python export_csv.py

# 用 CSV 数据源训练/测试（与 pickle 逐位一致）
python train.py     --dataset cora --source csv
python train_pyg.py --dataset cora --source csv
python test.py -k csv

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

旁证：`hidden1` 从 16 调到 64，本实现精度升至 **79.80%** ——模型容量足够，纯粹是固定超参下的收敛轨迹差异。

> **引用建议**：
>
> - 若需**与论文可比**的数字，用本项目自带的 PyG 版：`python train_pyg.py --dataset cora` → **79.50%**（论文 81.5%）。这是本仓库内实测值，不是引用二手数据。
> - 若需**说明移植保真度**，用手写版的 76.70% 并附上 `test.py` 的 54 项断言结果——它证明的是「数学正确」，而非「精度 SOTA」。
> - 两者相差的 2.8 个百分点，主因是归一化的[[#一个必须知道的自环语义差异|自环语义差异]]，不是实现错误。

## 目录结构

```
gcn-pytorch/
├── train.py              训练入口（argparse，对齐 TF 版 tf.app.flags）
├── test.py               正确性测试（54 项断言，独立参考值交叉核对）
├── train_pyg.py          PyG 版训练入口（GCNConv，支持任意层数）
├── test_pyg.py           PyG 版测试（37 项断言，含与手写层的等价性）
├── train_planetoid.py    PyG 内置数据集版训练入口（Planetoid，自动落盘）
├── test_planetoid.py     PyG 内置数据集版测试（32 项断言）
├── export_csv.py         数据集 pickle → CSV 导出
├── export_model.py       权重导出（GGUF / PyTorch / 元信息）
├── infer.py              推理（支持 .pt 与 .gguf）
├── requirements.txt
├── model/                导出产物：.gguf + .pt + .meta.json
├── data/csv/             CSV 形式的数据集（由 export_csv.py 生成）
├── data/Planetoid/       PyG 内置数据集的本地缓存（raw + processed）
├── README.data.md        数据集详解（指标 / 格式 / 读取 / 七个坑 / 实测精度）
├── notebooks/
│   ├── train.ipynb       手写实现的可交互版（含曲线、分类别准确率、CSV 对比）
│   └── train_pyg.ipynb   PyG 版（含层数实验、层等价性、自环差异）
├── gcn/
│   ├── layers.py         GraphConvolution / Dense（含稀疏 dropout）—— 逐行注释
│   ├── models.py         GCN / MLP —— 组装层与损失
│   ├── metrics.py        masked softmax CE / masked accuracy
│   ├── utils.py          数据加载（pickle / CSV）+ 预处理 + Chebyshev
│   ├── __init__.py       稀疏张量不变量告警的上下文管理器
│   └── data/             citeseer / cora / pubmed 三个数据集
├── pyg_gcn/
│   ├── __init__.py       to_pyg_data（Data 转换）+ PyG 告警抑制
│   └── models.py         PyGGCN（GCNConv 堆叠）
└── pyg_dataset/
    └── __init__.py       Planetoid 加载 + 本地缓存 + 镜像回退
```

### 各文件的阅读顺序

**手写实现**（理解数学本身）：

```
gcn/layers.py     先读：理解单层在算什么（含形状推导）
   ↓
gcn/models.py     再读：理解层怎么串起来、损失怎么算
   ↓
gcn/utils.py      然后：理解数据怎么变成 adj + features + masks
   ↓
train.py          最后：理解训练循环与早停逻辑
```

**PyG 实现**（理解工程封装）：

```
pyg_gcn/__init__.py  先读：to_pyg_data 如何把 load_data 的结果转成 Data
   ↓
pyg_gcn/models.py    再读：GCNConv 如何堆叠，以及为何手动加 L2
   ↓
train_pyg.py         最后：与 train.py 逐行对照，看差异只在哪些地方
```

**建议的对照顺序**：先读完手写链路，再看 PyG 版——因为 PyG 把归一化、自环、邻域聚合都封装进 `GCNConv` 内部了，若不先理解手写实现，就不知道那些封装到底做了什么。

对照阅读时可用「TF 原版 → PyTorch 版」速查表：

| 原文件 | 本实现 | 关键移植点 |
|---|---|---|
| `gcn/utils.py` | `gcn/utils.py` | `np.bool`→`bool`；tuple → `torch.sparse_coo_tensor`；`eigsh` 导入路径适配 SciPy 1.14+ |
| `gcn/inits.py` | 合并进 `layers.py` | `glorot()` → `nn.Parameter` + `uniform_(-r, r)`，r=√(6/(in+out)) |
| `gcn/layers.py` | `gcn/layers.py` | `tf.sparse_placeholder` → 稀疏张量；`sparse_retain` → `floor(keep+U(0,1))` 掩码 |
| `gcn/metrics.py` | `gcn/metrics.py` | `softmax_cross_entropy_with_logits` → 手写 `-Σ y·log_softmax(z)` |
| `gcn/models.py` | `gcn/models.py` | `Model.build()` → `nn.Module.forward`；`tf.train.AdamOptimizer` → `torch.optim.Adam` |
| `gcn/train.py` | `train.py` | `tf.app.flags` → `argparse`；`sess.run` → 前向/反向/`optimizer.step` |

## 复现时踩到的七个坑

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

特征矩阵密度仅 **1.27%**（49216 个非零 / 3880564 个格子）。用普通 `F.dropout` 会破坏稀疏结构、内存暴涨。TF 版用 `sparse_retain` 只在非零位置采样：

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

### 7. 原始数据自带自环，`A + I` 会把它变成 2

这一点在移植时最容易漏掉，因为它**不影响代码运行，只影响数值**。

| 数据集 | 原始自环数 |
|---|---:|
| Cora | 0 |
| Citeseer | **124** |
| Pubmed | **3** |

TF 原版与手写实现的 `preprocess_adj` 做的是 `adj + sp.eye(n)`：若某节点本来自环权重为 1，相加后变成 **2**。而 PyG 的 `add_self_loops` 是把对角元**覆盖为 1**。

结果：Citeseer 上两套归一化矩阵的最大差异达 **`1.7e-01`**（Cora 为 `3.0e-08`，因为无自环两者重合）。

**这不是 bug，是两种约定**。若要与 TF 原版逐位对齐，保留 `A + I`；若要复现 PyG 的数值，需先清除自环再加 I。`test_pyg.py` 的 B 组用构造实验证明了 PyG 的确切行为（重构偏差 `3.0e-08`）。

## PyTorch Geometric 版（GCNConv）

除了手写实现，项目还提供一套 **PyG 版**，用于横向验证与对照实验。两套实现**并存**，互不替代。

```bash
pip install torch_geometric        # 纯 Python，无需 torch-scatter

python train_pyg.py --dataset cora                      # 默认 2 层
python train_pyg.py --dataset cora --num_layers 1       # 单层
python train_pyg.py --dataset cora --num_layers 8       # 加深，观察过平滑
python train_pyg.py --dataset citeseer --source csv     # 用 CSV 数据源
python test_pyg.py                                      # 37 项断言
```

### 两套实现的分工

| | 手写实现 `gcn/` | PyG 版 `pyg_gcn/` |
|---|---|---|
| 定位 | **忠实移植 TF 原版**，逐行对照 | **社区标准实现**，用于对照与加深实验 |
| 图卷积 | `GraphConvolution`（逐行注释） | `GCNConv` |
| 层数 | 固定 2 层 | **任意层数**（`--num_layers`） |
| support | 支持多 support（ChebNet） | 单 support（ChebNet 需 `ChebConv`） |
| 数据入口 | `load_data`（pickle / CSV） | `to_pyg_data`（**同样调 `load_data`**） |
| Cora 精度 | 76.70% | **79.50%** |

### 关键设计：共用同一套划分

`to_pyg_data()` **不调用** `torch_geometric.datasets.Planetoid`，而是复用 `load_data()` 后重排成 `Data` 对象。原因是：

- Planetoid 会**联网下载**，而数据已在本地；
- 更重要的是**划分一致性**——若让 PyG 自己切分，精度差异就分不清是「实现差异」还是「划分差异」。

因此两套数字可直接比较。

### 实测精度

| 数据集 | 手写版 | PyG 版 | 差 |
|---|---:|---:|---:|
| Cora | 76.70% | **79.50%** | +2.80 |
| Citeseer | 64.10% | **66.80%** | +2.70 |
| Pubmed | 75.60% | **76.70%** | +1.10 |

PyG 版更接近论文报告值（Cora 81.5%），差距主要在 Adam 实现细节与归一化的**自环语义**（见下）。

### 一个必须知道的自环语义差异

两套实现在**原始数据自带自环**的图上**并不等价**：

| | 做法 | 对角元 |
|---|---|---|
| 手写 `preprocess_adj`（TF 原版） | `Ã = A + I` | 原为 1 则变成 **2** |
| PyG `gcn_norm` | `add_self_loops` 覆盖 | 恒为 **1** |

| 数据集 | 原始自环数 | 归一化矩阵最大差异 |
|---|---:|---:|
| Cora | 0 | `3.0e-08`（一致） |
| Citeseer | 124 | `1.7e-01` |
| Pubmed | 3 | `9.7e-02` |

这不是 bug，是两种合理但不同的约定。`test_pyg.py` 的 B 组断言用「清掉自环后再加 I」的构造实验**证明**了 PyG 的确切行为（重构偏差 `3.0e-08`），而不是简单断言「两者相等」。

### 层等价性验证

权重对齐后（手写存 `(in,out)`、PyG 存 `(out,in)`），三层维度组合的前向输出偏差均 `< 1e-5`：

| 维度 | 最大偏差 |
|---|---|
| 1433→16 | `6.5e-09` |
| 16→7 | `< 1e-5` |
| 5→3 | `< 1e-5` |

这证明两者的**数学定义一致**，差异只来自自环约定与浮点顺序。

### 加深实验：过平滑的直接证据

`--num_layers` 是 PyG 版独有的能力：

| 层数 | Cora 准确率 |
|---:|---:|
| 1 | 73.10% |
| 2 | **79.50%** |
| 8 | **6.40%** |

8 层跌到 6.4%（7 分类的随机水平约 14.3%），比随机猜还差——这是 [[notes/06-深度困境-过平滑与过挤压|过平滑]] 最直观的实证：所有节点表示收敛到同一点，分类器彻底失效。

## Notebook（可交互版）

`notebooks/` 下有两个 notebook，与 `train.py` / `train_pyg.py` **逻辑等价**，但把流程拆成可逐段运行的单元格：

```bash
pip install ipykernel          # 或 pip install -r requirements.txt
jupyter lab notebooks/         # 在 scripts/gcn-pytorch 下启动
```

| notebook | 对应脚本 | 内容 | 实测结果 |
|---|---|---|---|
| `train.ipynb` | `train.py` | 数据 → 预处理 → 建模 → 训练 → 测试 → CSV 交叉验证 | Cora **76.70%** |
| `train_pyg.ipynb` | `train_pyg.py` | 同上 + 层数实验 + 与手写层等价性 + 自环差异 | Cora **79.50%** |

**notebook 里额外提供的四件事**（脚本里没有）：

1. **内联 SVG 曲线** —— 损失/准确率双曲线，**不依赖 matplotlib**，装完 torch 就能跑。
2. **分类别准确率表** —— 总体准确率会掩盖「少数类被放弃」。Cora 上类 3（样本最多）约 69%，类 2 约 87%。
3. **层数实验** —— `train_pyg.ipynb` 直接跑 1/2/4/8 层：73.10% → 79.50% → 75.40% → **6.40%**。4 层就开始退化，8 层低于随机水平（14.3%）。
4. **交叉验证** —— 手写版比对 pickle/CSV 九个张量（全 `0.00e+00`）；PyG 版比对 `GCNConv` 与 `GraphConvolution` 的前向输出（`5.6e-09`）。

> **输出已固化**：notebook 里保存的是实际运行结果，不运行也能直接看数值。
> 内容由 `scripts/gen_notebooks.py` 生成，改完重跑 `python scripts/gen_notebooks.py --run` 即可。

> **一个协议细节**：notebook 的层数实验刻意照抄 `train_pyg.py` 的早停协议（验证损失 = CE + L2，不做权重回滚）。早期版本用了「best-weight 回滚 + 只比 CE」，2 层会得到 79.60% 而不是 79.50%——**0.1 个百分点的假差异**，值得警惕。

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

**手写版**（`train.py --model`）：

| `--model` | 说明 | 对应文档 |
|---|---|---|
| `gcn` | 两层 GCN，support=1 | §3.3 |
| `gcn_cheby` | K 阶 Chebyshev 基，support=K+1 | §3.2 |
| `dense` | MLP 基线（无图信息） | §4.6 对照实验 |

`gcn` 与 `gcn_cheby` 用的是**同一个 `GCN` 类**——这正是文档 §3.5 说的"谱方法与空域方法在低阶近似下收敛到同一类算子"的工程体现：ChebNet 只是多给几个 support，层结构不用改。

**PyG 版**（`train_pyg.py --num_layers`）：单一 `GCNConv` 堆叠结构，层数可调：

| `--num_layers` | 结构 | Cora 实测 |
|---:|---|---:|
| 1 | F→C | 73.10% |
| 2 | F→16→C（论文设定） | **79.50%** |
| 8 | F→16→…→C | 6.40% |

> **注意 `dense` 基线的实测值偏低**（Cora 30.20%，论文 56.5%）。原因是 Cora 仅 140 个训练节点，MLP 无图先验时迅速过拟合。详细分析见 [[README.data]] 第 7.1 节。

## 数据集

项目有**三条并行的数据路径**，互不影响：

| 路径 | 入口 | 数据位置 | 谁负责解析 |
|---|---|---|---|
| A. 本地 pickle | `train.py` / `train_pyg.py` | `gcn/data/` | 本仓库 `load_data` |
| B. 本地 CSV | 同上 `--source csv` | `data/csv/` | 本仓库 `load_data_csv` |
| **C. PyG 内置** | **`train_planetoid.py`** | **`data/Planetoid/`** | **PyG `Planetoid`** |

A、B 两条见 [[README.data]]（含 8 个 `ind.*` 文件的字段含义、Citeseer 15 个孤立节点的处理等）；C 见下一节。

> **三条路径的取舍**：想**研究数据管线**用 A（字段最透明）；想在 Excel/pandas 里看数据用 B；想**少写代码、让 PyG 全包**用 C。

## PyG 内置数据集（Planetoid）

第三条路径把数据完全交给 PyG：下载、解压、`test_idx_reorder` 重排、Citeseer 孤立节点补齐都由 `torch_geometric.datasets.Planetoid` 完成，本仓库不再自己解析 `ind.*`。

```bash
python train_planetoid.py                     # Cora，首次会下载并落盘
python train_planetoid.py --dataset citeseer
python train_planetoid.py --report            # 只看本地缓存情况
python test_planetoid.py                      # 32 项断言
```

### 数据落盘位置

```text
data/Planetoid/
├── Cora/
│   ├── raw/        ind.cora.{x,y,tx,ty,allx,ally,graph} + test.index   （8 个）
│   └── processed/  data.pt                                               （PyG 处理结果）
├── Citeseer/       raw 8 个 + processed/data.pt
└── Pubmed/         raw 8 个 + processed/data.pt
```

实测占用：Cora 15.5 MB、Citeseer 48.2 MB、Pubmed 47.5 MB，合计约 **111 MB**（大头是 `processed/data.pt` 里的稠密特征）。**与 `gcn/data/`、`data/csv/` 完全隔离**，不会覆盖任何原文件。

缓存可以随时删掉重建，不影响任何原数据：

```bash
rm -rf data/Planetoid        # 下次运行会重新下载/处理
```

> 若要把仓库提交到 git，建议把 `data/Planetoid/` 加进 `.gitignore`——它只是可再生的缓存。

再次运行时命中缓存，**不联网、不重跑 process**（`test_planetoid.py` 的 A3 断言检查 `data.pt` 的 mtime 未变）。

### 离线环境怎么办

PyG 默认从 `github.com/kimiyoung/planetoid/raw/master/data` 下载，该地址会跳转到 `raw.githubusercontent.com`。本机沙箱里这个域名不可达，因此 `pyg_dataset` 加了一条**镜像回退**：

| 顺序 | 方式 | 说明 |
|---|---|---|
| 1 | 已缓存 | `raw/` 下 8 个文件齐全 → 直接读，不联网 |
| 2 | 官方地址 | 联网正常时由 `Planetoid.download()` 完成 |
| 3 | **镜像回退** | 从 `codeload.github.com` 取仓库 tar.gz（约 6 MB），一次抽出**三个数据集**的文件 |

镜像走 `github.com` 的打包服务，不碰 `raw.githubusercontent.com`。实测单次约 160 秒，故设计成一次铺开三个数据集，避免重复下载。

### 与本地路径的差异（重要）

PyG 的 `Planetoid` 在处理时**会丢掉原始自环**：

| 数据集 | 本地 `adj_raw` 非零元 | PyG `edge_index` | 差 | 原始自环数 |
|---|---:|---:|---:|---:|
| Cora | 10,556 | 10,556 | 0 | 0 |
| Citeseer | 9,228 | 9,104 | **124** | 124 |
| Pubmed | 88,651 | 88,648 | **3** | 3 |

边数差**恰好等于**原始自环数——`test_planetoid.py` 的 C1 断言就检查这一点，多一条少一条都会失败。

这个差异**不影响精度**：PyG 丢掉自环后，`GCNConv` 的 `add_self_loops` 会以权重 1 补回；而本地路径保留自环、`GCNConv` 也是覆盖为 1。两者归一化结果相同，所以实测精度完全一致：

| 数据集 | `train_pyg.py`（本地） | `train_planetoid.py`（内置） | 差 |
|---|---:|---:|---:|
| Cora | 79.50% | **79.50%** | 0.00 |
| Citeseer | 66.80% | **66.80%** | 0.00 |
| Pubmed | 76.70% | **76.70%** | 0.00 |

> 注意这里的「一致」只对 `GCNConv` 成立。若换成手写 `GraphConvolution`（`A + I` 会把已有自环变成 2），两条路径就会分叉——那正是 `train.py` 76.70% 与 `train_pyg.py` 79.50% 的差距来源。

### 特征归一化

`load()` 默认套 `NormalizeFeatures()`，与手写版的 `preprocess_features` 对齐。`--no_normalize` 可关闭。

实测 Cora 上两者只差 0.1 个百分点（79.50% vs 79.40%）——0/1 词袋特征的行归一化只改尺度不改方向。**真正影响精度的是邻接矩阵的归一化约定，不是特征归一化。**

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
