"""PyTorch 版模型定义。

对照 TensorFlow 原版 ``gcn/models.py`` 移植，提供两个模型：
  - MLP  ：不含图信息的基线（``--model dense``）
  - GCN  ：两组图卷积（``--model gcn``）

严格保留原版的损失构造方式：**只对第一层的权重做 L2 正则**
（TF 版是 ``for var in self.layers[0].vars.values()``），
这一点对复现论文数值很关键 —— 若改成全层正则，准确率会偏离原文。

模块职责划分
------------
    layers.py   ：单层计算（图卷积 / 全连接），只关心前向传播
    metrics.py  ：损失与准确率，只关心数值计算
    models.py   ：本文件 —— 把层串起来，并负责 loss / accuracy / predict 的组装
"""

import torch                         # 张量与 softmax
import torch.nn as nn                # 继承 Module/ModuleList

from .layers import Dense, GraphConvolution       # 两种层
from .metrics import masked_accuracy, masked_softmax_cross_entropy  # 两个指标


def identity(x):
    """恒等激活，对应 TF 原版的 ``act=lambda x: x``（输出层不激活，保留 logits）。

    输出层必须保留原始 logits：softmax 交叉熵内部自己会做 log_softmax，
    若这里先 relu 会丢失负半轴信息，且与TF 版数值不一致。
    """
    return x                         # 原样返回，不做任何变换


class Model(nn.Module):
    """模型基类，负责损失、准确率与预测。

    TF 版这里是 ``class Model(object)`` 且用 ``self.layers[i].vars`` 访问变量；
    PyTorch 版改为继承 ``nn.Module``，参数由 ``nn.Module.__setattr__``
    自动注册进 ``model.parameters()``，无需手动收集 vars。
    """

    def __init__(self, **kwargs):
        """初始化基类的公共状态。子类会覆盖大部分字段。"""
        super(Model, self).__init__()  # 初始化 nn.Module 内部状态

        # ---- 元信息（不参与梯度计算）----
        # 类名小写作为默认名字，GCN -> 'gcn'，MLP -> 'mlp'
        self.name = kwargs.get('name', self.__class__.__name__.lower())
        self.logging = kwargs.get('logging', False)   # 调试开关，对齐 TF 版

        # ---- 网络结构 ----
        self.layers = nn.ModuleList()  # 用 ModuleList 而非普通 list，
                                       # 否则 nn.Module 不会遍历到子层的参数，
                                       # optimizer 会报 "no parameters" 错误
        self.input_dim = None           # 输入特征维度 F_in
        self.output_dim = None          # 输出类别数 C

        # ---- 超参默认值（子类的 __init__ 会用实参覆盖）----
        # 这几个值与 TF 原版默认值保持一致：dropout=0.5、L2=5e-4、lr=0.01
        self.dropout_rate = 0.5         # 丢弃概率 p
        self.weight_decay = 5e-4        # L2 正则系数
        self.learning_rate = 0.01       # 学习率（仅记录，optimizer 在 train.py 建）

        # ---- 前向传播的输出缓存 ----
        # 每次 forward 都会覆盖，供 _loss / _accuracy / predict 复用，
        # 避免为了算损失再前向一次
        self.outputs = None

        # ---- 本次损失/准确率使用的标签与 mask ----
        # 由 train.py 在每个阶段直接赋值（训练用 y_train+train_mask，
        # 验证用 y_val+val_mask，测试用 y_test+test_mask）
        # 不设为 nn.Parameter，因为它们是数据而非可学习参数
        self.labels = None
        self.labels_mask = None

    def _build(self):
        """构造网络结构，由子类实现（多态入口）。"""
        raise NotImplementedError       # 基类不提供实现

    def forward(self, features):
        """按顺序把 features 送过所有层。

        :param features: (N, F_in) 稀疏或稠密的节点特征
        :return: (N, C) 的 logits（最后一层用 identity 激活）
        """
        h = features                   # h 表示当前的隐藏表示，逐层覆盖
        for layer in self.layers:      # 顺序遍历 ModuleList
            h = layer(h)                # 交给该层的 forward，返回下一层的输入
        self.outputs = h               # 缓存输出，供 _loss/_accuracy/predict 使用
        return h

    def _loss(self):
        """计算总损失 = L2 正则 + 带 mask 的交叉熵。

        :return: 标量张量
        """
        # ---- 权重衰减：只作用于第一层（与 TF 原版一致）----
        # 关键：TF 的 tf.nn.l2_loss(t) 语义是 sum(t^2) / numel（逐元素取均值），
        # 不是 sum(t^2)/2，也不是 sum(t^2)。这里必须用 numel 归一，
        # 否则正则强度会差出上千倍，精度会明显偏低。
        loss = 0                        # 累加器起点（int 0 可与 Tensor 相加）
        for var in self.layers[0].parameters():
            # 只遍历 layers[0]：layers[0] 是第一层（GCN 中是 graphconvolution_1）
            # 注意第二层的权重**不做**正则，这是 TF 原版的刻意选择
            # var*var 逐元素平方 -> sum 求和 -> 除以元素个数 numel 得均值
            loss += self.weight_decay * torch.sum(var * var) / var.numel()

        # ---- 交叉熵 ----
        # masked_softmax_cross_entropy 内部会用 self.outputs / labels / labels_mask
        # 与上面 forward 缓存的输出保持一致，避免重复前向
        loss = loss + masked_softmax_cross_entropy(
            self.outputs, self.labels, self.labels_mask)
        return loss

    def _accuracy(self):
        """在 ``labels_mask`` 指定的子集上计算准确率。"""
        return masked_accuracy(
            self.outputs, self.labels, self.labels_mask)

    def predict(self):
        """返回各节点的类别概率（对 logits 做 softmax）。

        注意：推理路径必须先调 ``model.eval()`` 关掉 dropout，
        否则预测结果会带随机噪声。
        """
        # dim=1 是类别维，(N,C) -> (N,C) 每行归一化为概率分布
        return torch.softmax(self.outputs, dim=1)


class MLP(Model):
    """不含图信息的 MLP 基线（``--model dense``）。

    用两层全连接代替图卷积：结构和超参与 GCN 完全一致，
    唯一差别是没有 support 传播，因此对比实验只差「有没有图信息」这一个变量。
    """

    def __init__(self, input_dim, output_dim, supports, hidden1=16,
                 dropout=0.5, weight_decay=5e-4, learning_rate=0.01, **kwargs):
        """构造两层 MLP。

        :param input_dim:  输入特征维度 F_in
        :param output_dim: 输出类别数 C
        :param supports:   **未使用**，仅为与 GCN 保持统一构造签名（可传 None）
        :param hidden1:    隐藏层宽度，论文与原版均为 16
        :param dropout:    丢弃概率
        :param weight_decay: L2 正则系数
        :param learning_rate: 学习率（记录用）
        """
        super(MLP, self).__init__(**kwargs)   # 先初始化基类公共状态

        # ---- 保存超参，供 _build 使用 ----
        self.input_dim = input_dim           # F_in
        self.output_dim = output_dim         # C
        self.dropout_rate = dropout           # 丢弃概率
        self.weight_decay = weight_decay       # L2 系数
        self.learning_rate = learning_rate     # 学习率

        self._build(hidden1)                # 建层
        self._init_parameters()             # 初始化参数（Glorot 已在层内完成）

    def _build(self, hidden1):
        """搭建 F_in -> hidden1 -> C 两层全连接。

        :param hidden1: 隐藏层宽度
        """
        # ---- 第 1 层：稀疏输入 -> hidden1 维，ReLU 激活 ----
        # sparse_inputs=True：输入是稀疏词袋特征，必须走 sparse_dropout
        #（否则零位置被噪声污染，稀疏结构被破坏）
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=hidden1,
                                 dropout=self.dropout_rate,
                                 act=torch.relu,          # 隐藏层用 ReLU
                                 sparse_inputs=True,
                                 logging=self.logging,
                                 name='dense_1'))
        # ---- 第 2 层：hidden1 -> C，identity 激活 ----
        # sparse_inputs=False：第 1 层输出已是稠密张量
        # act=identity：输出层保留 logits，交给 softmax 交叉熵处理
        self.layers.append(Dense(input_dim=hidden1,
                                 output_dim=self.output_dim,
                                 dropout=self.dropout_rate,
                                 act=identity,
                                 logging=self.logging,
                                 name='dense_2'))

    def _init_parameters(self):
        """无需额外初始化：Dense 构造函数内已做 Glorot 初始化。"""
        pass


class GCN(Model):
    """两层图卷积网络（Kipf & Welling 2017）。

        H^(1) = σ( D̃^-1/2 Ã D̃^-1/2 X W^(0) )     维度 (N, hidden1)
        H^(2) = σ( D̃^-1/2 Ã D̃^-1/2 H^(1) W^(1) ) 维度 (N, C)

    两层都用同一组 support，因此权重张量总数为 2×len(supports)×in×out。
    """

    def __init__(self, input_dim, output_dim, supports, hidden1=16,
                 dropout=0.5, weight_decay=5e-4, learning_rate=0.01, **kwargs):
        """构造两层图卷积网络。

        :param input_dim:  输入特征维度 F_in（Cora 为 1433）
        :param output_dim: 输出类别数 C
        :param supports:   support 稀疏矩阵列表
                           - ``--model gcn``     : [D̃^-1/2 Ã D̃^-1/2]，长度 1
                           - ``--model gcn_cheby``: [T_0, ..., T_K]，长度 K+1
        :param hidden1:    隐藏层宽度，论文与原版均为 16
        :param dropout:    丢弃概率
        :param weight_decay: L2 正则系数
        :param learning_rate: 学习率
        """
        super(GCN, self).__init__(**kwargs)   # 先初始化基类公共状态

        # ---- 保存结构与超参，供 _build 使用 ----
        self.supports = supports            # support 列表，两层共享同一组
        self.input_dim = input_dim           # F_in
        self.output_dim = output_dim         # C
        self.dropout_rate = dropout           # 丢弃概率
        self.weight_decay = weight_decay       # L2 系数
        self.learning_rate = learning_rate     # 学习率

        self._build(hidden1)                # 建层
        self._init_parameters()             # 初始化参数

    def _build(self, hidden1):
        """搭建 F_in -> hidden1 -> C 两层图卷积。

        :param hidden1: 隐藏层宽度
        """
        # ---- 第 1 层：稀疏输入 -> hidden1 维，ReLU 激活 ----
        # sparse_inputs=True：X 是稀疏词袋/ TF-IDF 特征
        # supports=self.supports：每层为每个 support 各建一个权重矩阵
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=hidden1,
                                            supports=self.supports,
                                            dropout=self.dropout_rate,
                                            act=torch.relu,       # 隐藏层 ReLU
                                            sparse_inputs=True,
                                            logging=self.logging,
                                            name='graphconvolution_1'))
        # ---- 第 2 层：hidden1 -> C，identity 激活 ----
        # sparse_inputs=False：第 1 层输出是稠密张量
        # act=identity：保留 logits
        self.layers.append(GraphConvolution(input_dim=hidden1,
                                            output_dim=self.output_dim,
                                            supports=self.supports,
                                            dropout=self.dropout_rate,
                                            act=identity,
                                            logging=self.logging,
                                            name='graphconvolution_2'))

    def _init_parameters(self):
        """无需额外初始化：Glorot 初始化已在 GraphConvolution.reset_parameters 完成。

        PyTorch 会通过 ``nn.Module.__setattr__`` 自动把 layers 里的
        ``nn.Parameter`` 注册进 ``self._parameters``，optimizer 直接
        ``model.parameters()`` 即可拿到，无需像 TF 版那样手动遍历 vars。
        """
        pass
