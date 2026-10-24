# 对 GCN 的理解（Understanding of GCN）

> 本节为补充理解笔记：在通读原文与官方实现（[tkipf/gcn](https://github.com/tkipf/gcn)，TensorFlow 1.x）之后，对 GCN 的系统性梳理。按「思考 → 任务 → 算法 → 损失函数 → 优化器 → 数据集 → 训练循环 → 评估指标 → 防过拟合 → 初始化 → 超参数调优 → 框架实现」十二个一级小标题组织；公式只列核心结论式，谱域推导过程不复述。

## Thoughts（思考）

### 1. 把图结构从「损失函数」搬进「网络本体」

**传统图半监督范式**把图信息写成损失里的显式正则项：

$$
\mathcal{L} = \mathcal{L}_0 + \lambda \mathcal{L}_{\mathrm{reg}}, \qquad \mathcal{L}_{\mathrm{reg}} = \sum_{i,j} A_{ij} \| f(X_i) - f(X_j) \|^2
$$

这一范式的隐含假设是「相连的节点很可能共享同一标签」。**GCN 的根本转向是**：不再把图当作损失里的先验，而是让网络本身以邻接矩阵为条件，即建模 $f(X, A)$。带来两个直接影响——(a) 不需要手工设计正则项及其权重 $\lambda$；(b) 有标签节点的梯度会**沿传播路径自动流到无标签节点**，因为所有节点共享同一套权重、同一张图。这是「半监督」在 GCN 里真正生效的机制。

### 2. 三级近似换来线性复杂度

谱域卷积 $g_\theta \star x = U g_\theta U^\top x$ 要乘特征向量矩阵，复杂度 $O(N^2)$，且需要对拉普拉斯矩阵做特征分解（eigendecomposition）。论文用一连串近似把它压成一次稀疏矩阵乘法（sparse matrix multiplication）：

| 步骤                               | 表达式                                           | 换来什么                                 |
| -------------------------------- | --------------------------------------------- | ------------------------------------ |
| 谱图卷积（spectral graph convolution） | $U g_\theta U^\top x$                         | 表达力最强，但 $O(N^2)$ + 特征分解              |
| 切比雪夫（Chebyshev）$K$ 阶展开           | $\sum_{k=0}^{K}\theta_k T_k(\tilde{L}) x$     | $K$-局部化，$O(\rvert\mathcal{E}\lvert)$ |
| 取 $K=1$（逐层线性，layer-wise linear）  | $\theta_0' x - \theta_1' D^{-1/2}A D^{-1/2}x$ | 每层只有一个邻域聚合算子                         |
| 约束为单参数（single parameter）         | $\theta(I_N + D^{-1/2}A D^{-1/2})x$           | 参数量从两个 $\theta$ 减到一个 $\theta$        |
| 重归一化技巧（renormalization trick）    | $\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$   | 数值稳定 + 性能最优                          |

**关键判断**：论文并未主张「一阶近似无损」。它主张的是——一阶算子 + 堆叠多层，已经足以恢复一大类滤波器函数，同时换来了可扩展性与深度。Table 3 的消融正好支持这一点：重归一化版本（式 8）在 Citeseer/Cora/Pubmed 上同时优于切比雪夫 $K=2,3$、一阶模型、单参数模型与纯 MLP，即**简化不但没掉点，还更快更准**。

### 3. 重归一化不是「数值小trick」，而是滤波器的频域设计

$I_N + D^{-1/2}A D^{-1/2}$ 的特征值落在 $[0, 2]$，反复施加同一算子会放大信号、导致梯度爆炸/消失（exploding/vanishing gradients）。换成 $\hat{A} = \tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$ 后，特征值被压回 $[-1, 1]$ 附近，谱上等价于一个**稳定的低通滤波器**。这一改动同时做了三件事：(a) 加自环 $\tilde{A} = A + I_N$ 让节点保留自身信息；(b) 对称度归一化（symmetric degree normalization）抑制高度数节点的支配作用；(c) 稳定深层堆叠。所以「重归一化」在论文里既是工程稳定性措施，也是模型表达力的一部分。

### 4. GCN 是 Weisfeiler-Lehman 检验的可微、参数化推广

附录 A 给出了本文最有解释力的视角。把 WL-1 算法里的 hash（哈希）函数换成可微的神经网络层：

$$
h_i^{(l+1)} = \sigma \left( \sum_{j \in \mathcal{N}_i} \frac{1}{c_{ij}} h_j^{(l)} W^{(l)} \right)
$$

取 $c_{ij} = \sqrt{d_i d_j}$ 就精确回到 GCN 的传播规则。由此可推出两条反直觉但重要的结论：

- **未训练的随机权重 GCN 也是强特征提取器**：附录 A.1 用三层随机权重 GCN 在 karate club 上得到的嵌入，与需要昂贵无监督训练的 DeepWalk 可比。图结构本身携带了大量可用的信息。
- **最少监督即可线性分离社区**：附录 A.2 用每类 1 个标签、300 次迭代就把 karate club 的四个社区线性分开。

这解释了为什么 GCN 在标签率极低时仍然有效——它学的主要是「怎么聚合邻居」，而不是「记住标签」。

### 5. 模型隐含了什么假设（论文自己承认的部分）

通过第 2 节的两级近似，GCN 隐含两个假设：(a) **局部性**——$K$ 层 GCN 只依赖 $K$ 阶邻域；(b) **自连接与邻居边等权**。论文在 7.2 节提出引入权衡参数 $\tilde{A} = A + \lambda I_N$（式 11），并指出它扮演的角色类似传统半监督中监督/无监督损失的权衡系数，但此处**可由梯度下降学出**。本质上，GCN 仍然站在同配性（homophily）假设上——它没有消除这个假设，只是把它从损失层面挪到了架构层面。

### 6. 放到后见之明里看：GCN 是消息传递的一个特例

用今天的消息传递神经网络（Message Passing Neural Network, MPNN）视角回看，GCN 的每一层恰好是三件事的组合：聚合（aggregate）邻居 $\hat{A} H^{(l)}$、变换（transform）共享权重 $W^{(l)}$、非线性（activate）$\sigma(\cdot)$。后续的 GraphSAGE、GAT、GraphSAGE+、Gine 等都是在「聚合」这一步上换花样——GAT 加注意力权重，GraphSAGE 换邻居采样与聚合函数，而 GCN 保留了最朴素的加权求和。它的影响力在于确立了「逐层传播 + 共享权重 + 局部邻域」这条路线的基本范式。

## Task（任务）

**任务定义**：图结构数据上的**半监督节点分类**（semi-supervised node classification），直推式（transductive）设定。

- **输入**：图 $\mathcal{G} = (\mathcal{V}, \mathcal{E})$，节点特征矩阵 $X \in \mathbb{R}^{N \times C}$，以及少量带标签节点集合 $\mathcal{Y}_L \subseteq \mathcal{V}$。
- **输出**：每个/所有节点上的类别分布 $Z \in \mathbb{R}^{N \times F}$（$F$ 为类别数），取 $\arg\max$ 得预测标签。
- **学习信号**：**只有** $\mathcal{Y}_L$ 上的标签参与损失计算；无标签节点只贡献特征与图结构。


**为什么这个任务值得研究**：真实图场景的标签极稀缺——Cora 只有 5.2% 的节点带标签、Pubmed 仅 0.3%、NELL 仅 0.1%。这与本项目关注的教育场景高度同构：学生作答记录同样是「图 + 稀疏标签」的形态，知识点是节点、作答/引用是边、掌握/未掌握是标签。

**与相邻任务的边界**：与节点嵌入（node embedding，如 DeepWalk）的区别在于本文是**端到端的、面向下游分类目标**的半监督学习，而非无监督预训练；与图分类（graph classification）的区别在于预测单位是节点而非整图；与链接预测（link prediction）无直接关系。

## Algorithm（算法）

### 核心：逐层传播规则（式 2）

$$
H^{(l+1)} = \sigma \left( \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)} \right)
$$

- $H^{(l)} \in \mathbb{R}^{N \times D}$：第 $l$ 层激活矩阵，$H^{(0)} = X$；
- $\sigma(\cdot)$：逐点非线性（如 ReLU）；
- $W^{(l)}$：**层特定且全图共享**的可训练权重（不是每节点一份，也不是每边一份）。

### 矩阵形式与计算顺序（式 8）

把信号推广到多通道、多个滤波器：

$$
Z = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} X \Theta, \qquad \Theta \in \mathbb{R}^{C \times F}
$$

官方实现里实际执行的顺序是 **先变换后聚合**：

$$
Z = \hat{A}\, (X \Theta)
$$

这个顺序不是随意的：$X$ 是极度稀疏的词袋矩阵（bag-of-words），先做稀疏×稠密的 $X\Theta$ 得到稠密中间结果，再做一次稀疏邻接矩阵×稠密矩阵的乘法，比反过来 $(A X)\Theta$ 省一个稀疏矩阵乘法，复杂度 $\tilde{O}(|\hat{\mathcal{E}}|CHF)$，与边数线性相关。

### 与 WL-1 的对应关系

$$
h_i^{(l+1)} = \sigma \left( \sum_{j \in \mathcal{N}_i} \frac{1}{c_{ij}} h_j^{(l)} W^{(l)} \right), \qquad c_{ij} = \sqrt{d_i d_j}
$$

代入 $c_{ij} = \sqrt{d_i d_j}$ 后，与式 2 的向量形式完全一致。**理解要点**：GCN 的每层就是「邻居状态按 $\frac{1}{\sqrt{d_i d_j}}$ 加权求和 → 共享线性变换 → 非线性」，深度 $L$ 层等价于在 $L$ 阶邻域上做 $L$ 次加权平滑。

### 推理阶段的算法流程

```
输入：归一化邻接矩阵 Â，节点特征 X，权重 {W^(l)}
1. H = X
2. for l = 0 .. L-2:
       H = σ( Â · (H · W^(l)) )        # 隐层，ReLU
3. Z = softmax( Â · (H · W^(L-1)) )     # 输出层
输出：Z
```

### 复杂度与可扩展性

| 项目 | 复杂度 | 说明 |
| --- | --- | --- |
| 单层前向 | $O(\rvert\mathcal{E}\lvert \cdot F C)$ | 稀疏邻接 × 稠密特征（sparse–dense） |
| 两层整体 | $\tilde{O}(\rvert\hat{\mathcal{E}}\lvert C H F)$ | 与边数线性 |
| 邻接矩阵内存 | $O(\rvert\mathcal{E}\lvert)$ | 稀疏存储（sparse storage） |
| 特征内存 | $O(N C)$ | 稀疏存储（sparse storage） |

论文 6.3 节在随机图上实测：GPU 上 100 万条边约 0.3 秒/epoch，CPU 上约 1 秒/epoch；1000 万条边时 GPU 显存溢出，CPU 约 8 秒/epoch。

## Loss Function（损失函数）

**带掩码的交叉熵（cross-entropy）**，仅在有标签节点上计算（式 10）：

$$
\mathcal{L} = - \sum_{l \in \mathcal{Y}_L} \sum_{f=1}^{F} Y_{lf} \ln Z_{lf}
$$

论文式 10 是**求和**形式；官方实现 `masked_softmax_cross_entropy` 实际等价于在有标签节点上取**均值**：

$$
\mathcal{L} = \frac{1}{|\mathcal{Y}_L|} \sum_{l \in \mathcal{Y}_L} \left( -\sum_{f=1}^{F} Y_{lf} \ln Z_{lf} \right)
$$

两者的梯度方向完全一致，只差一个与标签数成正比的缩放因子——在固定标签集的前提下不影响最优解。

**权重衰减（weight decay / L2 正则化 regularization）**：只加在**第一层**权重上（这是论文中一个容易被忽略、但对复现很关键的细节）。完整目标：

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda \| W^{(0)} \|_F^2
$$

官方代码对应 `for var in self.layers[0].vars.values(): self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)`——**只遍历 `layers[0]`**。为什么只正则第一层：第一层参数最多（$C \times H$），最容易过拟合；第二层只有 $H \times F$，正则化收益小、副作用大。

**与式 1 的对照（这是理解 GCN 的关键）**：式 1 的 $\mathcal{L}_{\text{reg}}$ 是「图结构上的标签平滑」，本质上是把标签信息沿图扩散的**手工先验**；GCN 把这件事交给了 $\hat{A}$ 这个可学习权重矩阵上的**架构**。代价是失去了式 1 中「显式权衡监督/无监督」的旋钮（论文用可学习的 $\lambda I_N$ 部分补偿），收益是模型容量大幅提升——这正是 Table 2 中 GCN 全面超过 ManiReg / SemiEmb / LP 的结构性原因。

**数值稳定**：官方实现用 `tf.nn.softmax_cross_entropy_with_logits`，内部用 log-sum-exp 稳定技巧，不要自己先算 softmax 再取 log。

## Optimizer（优化器）

| 项目 | 设置 | 出处 |
| --- | --- | --- |
| 优化器 | Adam（Kingma & Ba, 2015） | 5.2 节 / `train.py` |
| 学习率（learning rate） | 0.01（恒定，无衰减） | 5.2 节 |
| 梯度模式 | **全批量梯度下降**（full-batch gradient descent） | 3.1 节 |
| 最大轮数 | 200 epochs | 5.2 节 |
| 早停（early stopping） | 验证损失连续 10 轮不下降则停（窗口 10） | 5.2 节 |
| 随机性来源 | Dropout（Srivastava et al., 2014） | 3.1 节 |

**为什么是全批量**：因为一次前向要算全图，而稀疏邻接矩阵下全图前向本身已经是 $O(|\mathcal{E}|)$，再切小批带来的稀疏矩阵切分开销会抵消收益。论文明确把「内存高效的小批量随机梯度下降」列为未来工作，并指出其难点：生成小批量时必须考虑层数，因为 $K$ 层 GCN 的 $K$ 阶邻域必须被完整存储才能保证梯度精确。

**早停（early stopping）判据的一个实现细节**：论文表述为「验证损失连续 10 个 epoch 没有下降就停止」，而代码实现是

```python
if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
    break
```

即「当前验证损失 > 前 10 轮验证损失的均值」时停止。这是一个**滑动窗口均值**判据，而非「逐轮严格下降」判据——两者行为相近但不完全等价，复现时按代码实现更稳妥。

## Datasets（数据集）

原文 Table 1（统计量取自 Yang et al., 2016）：

| 数据集 | 类型 | 节点数 | 边数 | 类别数 | 特征维度 | 标签率 |
| --- | --- | --- | --- | --- | --- | --- |
| Citeseer | 引文网络（citation network） | 3,327 | 4,732 | 6 | 3,703 | 0.036 |
| Cora | 引文网络 | 2,708 | 5,429 | 7 | 1,433 | 0.052 |
| Pubmed | 引文网络 | 19,717 | 44,338 | 3 | 500 | 0.003 |
| NELL | 知识图谱（knowledge graph） | 65,755 | 266,144 | 210 | 5,414 | 0.001 |

**四个数据集的设计意图各不相同**：

- **引文网络三件套**：节点=文档，边=引用（视为无向边），特征=稀疏词袋，标签=文档类别。用于考察标准半监督节点分类。
- **NELL**：从知识图谱抽出的**二部图**（bipartite graph）。原始知识图谱是有向带标签边的实体集合，论文把每个实体对 $(e_1, r, e_2)$ 拆成 $(e_1, r_1)$ 与 $(e_2, r_2)$，为每个关系节点分配唯一独热表示，特征维数扩到 61,278。标签极端稀缺：**每类仅 1 个标注样本**。
- **随机图**：$N$ 个节点、$2N$ 条均匀随机边，特征取 $X = I_N$，标签全置 1。仅用于测量每轮训练时间，不做精度评估。

**数据划分**（沿用 Planetoid 的划分，这是与基线可比的前提）：

| 集合 | 规模 | 用途 |
| --- | --- | --- |
| 训练集（training set） | 引文网络每类 20 个标签；NELL 每类 1 个 | 梯度更新（gradient update） |
| 验证集（validation set） | 500 个带标签节点 | 超参数优化与早停，**标签不参与训练** |
| 测试集（test set） | 1,000 个带标签节点 | 最终精度评估 |

**预处理**：输入特征按行归一化（row-normalize），即 $x_i \leftarrow x_i / \sum_j x_{ij}$，使文档长度不影响尺度；邻接矩阵对称化并加自环后做对称度归一化。注意 5.1 节说明特征向量的预处理是在 `preprocess_features` 中做的，与论文正文「相应地（按行）归一化输入特征向量」对应。

## Training Loop（训练循环）

对应官方 `train.py` 的完整一轮：

```
# ---- 一次性预处理（循环外）----
features = preprocess_features(features)      # 稀疏、按行归一化
support  = [preprocess_adj(adj)]             # Â = D̃^{-1/2} Ã D̃^{-1/2}，稀疏化
model    = GCN(placeholders, input_dim=features.shape[1])
sess.run(tf.global_variables_initializer())

for epoch in range(FLAGS.epochs):            # 最多 200
    # 1) 构造 feed dict：特征、Â、训练标签、训练掩码、dropout 比例
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})   # 0.5

    # 2) 训练步：前向 + 交叉熵 + 反向 + Adam 更新（一次 sess.run 完成）
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

    # 3) 在验证集上评估（dropout 关闭）
    cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)

    # 4) 早停判定
    if epoch > FLAGS.early_stopping and \
       cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        break

# 5) 测试集最终评估（dropout 关闭）
test_cost, test_acc, _ = evaluate(features, support, y_test, test_mask, placeholders)
```

**几个容易踩的点**：

1. **Dropout 的作用位置**：官方 `GraphConvolution._call` 里，dropout 施加在**每层的输入端**（`x = tf.nn.dropout(x, 1-self.dropout)` 在矩阵乘法之前）。因此论文所说的「0.5 dropout rate, first and last layer」，精确含义是：第一层卷积的输入（原始特征）与最后一层卷积的输入（隐层激活）各做一次 dropout。
2. **稀疏特征的 dropout 要专门实现**：`sparse_dropout` 按非零元素数采样掩码（`tf.sparse_retain`），而不是把稀疏矩阵转稠密再 dropout——后者会摧毁稀疏性。
3. **验证/测试阶段 dropout 必须关闭**：`placeholders['dropout']` 的默认值是 0，`evaluate()` 不写入该键，因此自动关闭。
4. **无偏置项**：官方 `GraphConvolution` 的 `bias` 参数默认 `False`，两层都不加偏置。复现时若自行加入偏置，结果会与论文有微小差异。
5. **随机种子**：官方设定 `np.random.seed(123)` 与 `tf.set_random_seed(123)`。

## Evaluation Metrics（评估指标）

**主指标：分类准确率（accuracy，百分比）**，在 1,000 个带标签节点的测试集上评估。

**报告口径**：

- 主表（Table 2）在 Planetoid 固定划分上，报告 **100 次随机权重初始化**的平均准确率；
- 另设 `GCN (rand. splits)` 一行：在 10 个随机抽取的、同规模的数据集划分上评估，报告**均值 ± 标准误**（如 Cora $80.1 \pm 0.5$），用于说明结果不依赖某一次幸运的划分；
- ICA 基线报告 100 次随机节点顺序的平均准确率；其余基线数字直接引自 Planetoid 论文。

**主结果**：

| 方法 | Citeseer | Cora | Pubmed | NELL |
| --- | --- | --- | --- | --- |
| ManiReg | 60.1 | 59.5 | 70.7 | 21.8 |
| LP | 45.3 | 68.0 | 63.0 | 26.5 |
| DeepWalk | 43.2 | 67.2 | 65.3 | 58.1 |
| ICA | 69.1 | 75.1 | 73.9 | 23.1 |
| Planetoid* | 64.7 | 75.7 | 77.2 | 61.9 |
| **GCN（本文）** | **70.3** | **81.5** | **79.0** | **66.0** |
| GCN（随机划分） | 67.9 ± 0.5 | 80.1 ± 0.5 | 78.9 ± 0.7 | 58.4 ± 1.7 |

括号内另报**收敛墙钟时间**（秒），GCN 在四个数据集上均快于 Planetoid。

**辅助指标**：

- **传播模型消融**（Table 3）：固定数据划分，比较 7 种传播模型的平均准确率；
- **每轮训练时间**（Figure 2）：随机图上 100 轮的平均墙钟时间，GPU 与 CPU 对比；
- **模型深度**（附录 B）：2~10 层，5 折交叉验证，比较标准 GCN 与加残差连接（residual connection）的变体。

**指标选择的一个隐含判断**：论文只用准确率，不用 Macro-F1 / AUC。理由是这些任务类别相对均衡且是单标签多分类；但在 NELL（210 类、每类 1 个标签）这类极端不均衡设定下，准确率的信息量有限，评估时需自行补充其他指标。

## Avoid overfitting（防过拟合）

论文实际使用的手段，按作用机制分四类：

**1. 结构性正则**

- **小隐藏维度**：默认 $H = 16$（NELL 为 64），相对输入维度（最高 61,278）极小，天然限制了模型容量；总参数量仅 $C \times H + H \times F$。
- **层数受限**：附录 B 显示 2~3 层最优；**超过 7 层且无残差连接时训练开始困难**，原因有二——(a) 每个节点的有效上下文随深度按 $K$ 阶邻域膨胀；(b) 参数量随深度增加。
- **残差连接（residual connection，仅在深度实验中作为变体引入）**：$H^{(l+1)} = \sigma(\hat{A}H^{(l)}W^{(l)}) + H^{(l)}$，缓解深层退化。
- **重归一化技巧**：把特征值范围控制在 $[-1,1]$ 附近，避免深层堆叠时的数值不稳定（属于「防训练崩溃」而非「防过拟合」，但同样必要）。

**2. 显式正则项**

- **Dropout 0.5**：施加在每层输入端（见「训练循环」）。
- **L2 正则 $5 \times 10^{-4}$**：只作用于第一层权重。

**3. 训练过程控制**

- **早停**：验证损失窗口 10。
- **最大 200 epochs**，远少于深度实验的 400 epochs。

**4. 隐式正则：参数共享与结构先验**

- 权重 $W^{(l)}$ **全图共享**，模型没有为每个节点学一份参数，参数量与 $N$ 无关；
- $K$ 阶局部性先验限制了每个节点能「看到」的信息量；
- 论文 2.2 节的论证是：逐层线性形式能**缓解在宽度差异很大的图上对局部邻域结构的过拟合**。

**原文未涉及、但现代 GCN 实现中常用的补充手段**（复现或扩展时值得考虑）：DropEdge / GraphDrop 等结构扰动、BatchNorm / GraphNorm、邻居采样（GraphSAGE 式）、自适应感受野（如 APPNP、GCNII）、以及 GAT 式的注意力权重（让模型自己学「哪些邻居重要」，从而缓解对等权聚合的过拟合）。

## Initialization strategy（初始化策略）

**权重初始化（initialization）：Glorot & Bengio（2010）的 Xavier 均匀初始化（Xavier uniform initialization）**。官方 `inits.py`：

```python
def glorot(shape, name=None):
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)
```

即 $W \sim U\!\left(-\sqrt{\frac{6}{n_{in}+n_{out}}},\, \sqrt{\frac{6}{n_{in}+n_{out}}}\right)$。

**为什么是 Xavier 而不是 He/Kaiming**：这是 GCN 特有的一个细节——GCN 的每一层前面挂着 $\hat{A}$ 这个显式的度归一化矩阵，它本身就在做一次「方差缩放」。在此前提下，Xavier 那种保持前向/反向方差守恒（variance preserving）的初始化更合适；He 初始化是为「卷积层后紧跟 ReLU、前面无归一化」设计的，直接套用会与 $\hat{A}$ 的缩放效应叠加。

**其他初始化相关约定**：

- **无偏置项**：`bias=False`，两层皆然。
- **特征归一化即一种初始化**：输入特征按行归一化 $x_i / \sum_j x_{ij}$，等价于让每个节点的初始状态落在单位尺度上，避免长文档主导初始信号。
- **邻接矩阵不是「初始化」但同样关键**：$\hat{A}$ 是一次性的、确定性的、无参数的预处理；它是模型唯一注入图结构的地方。
- **随机初始化的价值**：论文反复用「100 次随机权重初始化的平均准确率」来报告结果，说明**初始化方差是需要被报告的量**，而不是可以忽略的噪声。附录 A.1 更进一步把「随机权重 + 未训练」当作一个合法的基线——它揭示了图结构自身的信息含量。

## Hyperparameter tuning（超参数调优）

**调参协议**（5.2 节）：

- 用**验证集**（500 个带标签节点）做超参数优化，验证集标签**不用于训练**；
- 优化对象三个：**所有层的 dropout 率（dropout rate）**、**第一层 GCN 的 L2 正则化因子（regularization factor）**、**隐藏单元数（number of hidden units）**；
- 学习率固定 0.01、最大轮数固定 200、早停窗口固定 10 —— 这三项**不参与调参**；
- **引文网络只调 Cora 一家**，Citeseer 与 Pubmed 直接沿用同一组参数（这是一个偏保守但高效的做法，也意味着 Pubmed 的表现可能未达其上限）。

**最终选定的超参数（hyperparameters）**：

| 数据集 | Dropout 率 | L2 正则（第一层） | 隐藏单元数 | 学习率 |
| --- | --- | --- | --- | --- |
| Citeseer / Cora / Pubmed | 0.5 | $5 \times 10^{-4}$ | 16 | 0.01 |
| NELL | 0.1 | $1 \times 10^{-5}$ | 64 | 0.01 |
| 随机图（仅测速度） | 0（不启用） | 0（不启用） | 32 | 0.01 |

**深度实验（附录 B）的独立配置**：400 epochs、**不使用早停**、Adam 学习率 0.01、dropout 0.5（首末层）、L2 $5\times10^{-4}$（第一层）、16 个隐藏单元。

**调参范围的合理性判断**：$H=16$、dropout 0.5、$L2=5\times10^{-4}$ 是深度学习社区的常规缺省值量级，说明作者在引文网络上基本没有过度搜索；真正起决定作用的是**验证集的早停**与**验证集上的模型选择**，而非精细的网格。NELL 的配置差异（dropout 0.1、$L2$ 降到 $10^{-5}$、隐藏单元提到 64）反映了它特征维数极高（61,278）、类别极多（210）的特点——正则化力度反而要放松、容量反而要加大。

**复现建议**：本文的调参协议简单但可复现性良好。若需在自己的数据上扩展，优先扫这三个量即可；此外建议补上原文没有的敏感性分析——标签率（label rate）从 0.5% 到 5% 的扫描、以及边数/节点数变化下模型 scaling 行为的分析。

## Framework implementation（框架实现）

**官方实现**：Python 2/3 + TensorFlow 1.x，代码在 [github.com/tkipf/gcn](https://github.com/tkipf/gcn)，目录结构与职责如下：

| 文件 | 职责 | 关键内容 |
| --- | --- | --- |
| `gcn/inits.py` | 初始化 | `glorot`、`uniform`、`zeros`、`ones` |
| `gcn/layers.py` | 层定义 | `GraphConvolution`、`Dense`、`sparse_dropout` |
| `gcn/models.py` | 模型与损失 | `GCN`、`MLP`；`_loss()` 拼装权重衰减（weight decay）+ 交叉熵（cross-entropy）；`AdamOptimizer` |
| `gcn/metrics.py` | 指标 | `masked_softmax_cross_entropy`、`masked_accuracy` |
| `gcn/utils.py` | 数据与预处理 | `load_data`、`preprocess_features`、`normalize_adj`、`preprocess_adj`、`chebyshev_polynomials` |
| `gcn/train.py` | 训练循环 | 超参 flags、主循环、早停 |

**核心工程手法**：

1. **稀疏张量占位符**：`tf.sparse_placeholder(tf.float32)` 承载特征与 $\hat{A}$，配合 `tf.sparse_tensor_dense_matmul`；`sparse_to_tuple` 把 scipy 稀疏矩阵转成 (coords, values, shape) 三元组喂给 TF。
2. **邻接矩阵归一化的标准写法**：

```python
def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.       # 孤立节点保护
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
```

`np.isinf` 那一行不可省——Citeseer 存在孤立节点（官方 `load_data` 里专门有一段为孤立节点补零向量的补丁）。

3. **多支持集（multi-support）抽象**：$\hat{A}$ 被抽象为 `support` 列表，切比雪夫变体 `gcn_cheby` 只需把 `chebyshev_polynomials(adj, K)` 的结果整组塞进去，每层对每个 support 各算一次再 `tf.add_n` 相加。这是 GCN 家族后续扩展（多尺度、多视图）的通用接口。
4. **无偏置 + 第一层专属 L2**（见前两节）。
5. **可复现性**：`seed = 123` 同时设定 numpy 与 TF 随机种子。

**现代等价实现对照**（若要在 PyTorch / PyG / DGL 中重写）：

| 论文概念 | 官方 TF 1.x | 现代等价 |
| --- | --- | --- |
| $\hat{A}$ 归一化 | `preprocess_adj` | `torch_geometric.utils.add_self_loops` + `gcn_norm` |
| 图卷积层 | `GraphConvolution` | `torch_geometric.nn.GCNConv`（默认 `add_self_loops=True, normalize=True`） |
| 掩码交叉熵（masked cross-entropy） | `masked_softmax_cross_entropy` | `F.cross_entropy(logits, y, reduction='none')` + 掩码平均 |
| 稀疏 dropout（sparse dropout） | `sparse_dropout` | 稀疏算子上的 scatter-dropout，或先稀疏化再 `F.dropout` |
| Adam + 早停 | `train.py` | `torch.optim.Adam` + `EarlyStopping` / `ReduceLROnPlateau` |

**迁移到本项目时的两点注意**：其一，PyG 的 `GCNConv` 默认已含自环与对称归一化，若自行再加一次会重复归一化；其二，教育场景的图常常带**时间方向**（先学 A 再学 B）与**多关系**（知识点—知识点、知识点—题目），而原论文明确指出其框架「不自然地支持边特征、且限于无向图」，NELL 上的处理办法是把有向图改造成无向二部图（bipartite graph）并加关系节点——这是引入边类型时可直接借用的思路。
