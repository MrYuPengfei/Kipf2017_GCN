"""PyTorch 版层定义。

对照 TensorFlow 原版 ``gcn/layers.py`` 移植，保持相同的计算图语义：
  - Dense：dropout -> 线性变换 -> (+bias) -> 激活
  - GraphConvolution：dropout -> 按 support 个数分别线性变换 -> support 前向 -> 求和 -> (+bias) -> 激活

关键差异：TF 用 ``tf.sparse_placeholder`` 接收稀疏输入，这里改用
``torch.sparse_coo_tensor``；稀疏 dropout 通过对稀疏张量的 values 加噪实现，
与 TF 的 ``sparse_retain`` + 缩放等价。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def sparse_dropout(x, keep_prob):
    """稀疏张量的 dropout。

    TF 原版（严格等价移植）：
        random_tensor = keep_prob + tf.random_uniform(noise_shape)
        mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        out = tf.sparse_retain(x, mask) * (1 / keep_prob)

    注意：``floor(keep_prob + U(0,1))`` 的保留概率**恰好是 keep_prob**
    （当 keep_prob<1 时 floor 结果为 0 或 1），但用的是均匀分布而非伯努利分布，
    必须照搬才能与 TF 版逐位对齐。
    """
    if keep_prob >= 1.0:
        return x
    values = x._values()
    indices = x._indices()
    shape = x.shape

    random_tensor = keep_prob + torch.rand_like(values)
    dropout_mask = torch.floor(random_tensor)

    return torch.sparse_coo_tensor(
        indices, values * dropout_mask / keep_prob, shape).coalesce()


class GraphConvolution(nn.Module):
    """图卷积层。

    对应文档 §3.3 的传播规则：
        H^(l+1) = σ( D̃^-1/2 Ã D̃^-1/2 H^(l) W^(l) )

    支持多个 support（如 Chebyshev 基），每个 support 拥有独立权重矩阵，
    最终输出求和 —— 这正是 ChebNet 到 GCN 的推广形式。
    """

    def __init__(self, input_dim, output_dim, supports, dropout=0.,
                 act=F.relu, bias=False, sparse_inputs=False,
                 featureless=False, logging=False, name=None):
        super(GraphConvolution, self).__init__()

        self.supports = supports
        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.dropout = dropout
        self.logging = logging
        self.name = name

        # 每个 support 一个权重矩阵（对应 TF 的 weights_0, weights_1, ...）
        self.weights = nn.ParameterList()
        for i in range(len(self.supports)):
            self.weights.append(nn.Parameter(torch.empty(input_dim, output_dim)))
            self.reset_parameters(self.weights[i], input_dim, output_dim)

        if self.bias:
            self.bias_param = nn.Parameter(torch.zeros(output_dim))
        else:
            self.register_parameter('bias_param', None)

    @staticmethod
    def reset_parameters(param, input_dim, output_dim):
        """Glorot & Bengio (AISTATS 2010) 初始化，与 TF 原版 glorot() 一致。"""
        init_range = np.sqrt(6.0 / (input_dim + output_dim))
        with torch.no_grad():
            param.uniform_(-init_range, init_range)

    def forward(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout)
        else:
            x = F.dropout(x, p=self.dropout, training=self.training)

        # convolve：对每个 support 做 (S_i · X · W_i) 再求和
        supports = []
        for i in range(len(self.supports)):
            if not self.featureless:
                pre_sup = torch.sparse.mm(x, self.weights[i]) if self.sparse_inputs \
                    else torch.mm(x, self.weights[i])
            else:
                pre_sup = self.weights[i]
            # support 前向：稀疏矩阵 × 稠密矩阵
            support = torch.sparse.mm(self.supports[i], pre_sup)
            supports.append(support)
        output = torch.sum(torch.stack(supports), dim=0)

        if self.bias:
            output = output + self.bias_param

        return self.act(output)


class Dense(nn.Module):
    """全连接层（MLP 基线用）。"""

    def __init__(self, input_dim, output_dim, dropout=0.,
                 act=F.relu, bias=False, sparse_inputs=False,
                 featureless=False, logging=False, name=None):
        super(Dense, self).__init__()

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.dropout = dropout
        self.logging = logging
        self.name = name

        self.weights = nn.Parameter(torch.empty(input_dim, output_dim))
        self.reset_parameters(self.weights, input_dim, output_dim)

        if self.bias:
            self.bias_param = nn.Parameter(torch.zeros(output_dim))
        else:
            self.register_parameter('bias_param', None)

    @staticmethod
    def reset_parameters(param, input_dim, output_dim):
        init_range = np.sqrt(6.0 / (input_dim + output_dim))
        with torch.no_grad():
            param.uniform_(-init_range, init_range)

    def forward(self, inputs):
        x = inputs

        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout)
        else:
            x = F.dropout(x, p=self.dropout, training=self.training)

        output = torch.sparse.mm(x, self.weights) if self.sparse_inputs \
            else torch.mm(x, self.weights)

        if self.bias:
            output = output + self.bias_param

        return self.act(output)