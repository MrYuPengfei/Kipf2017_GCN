"""PyTorch 版模型定义。

对照 TensorFlow 原版 ``gcn/models.py`` 移植，提供两个模型：
  - MLP  ：不含图信息的基线（``--model dense``）
  - GCN  ：两组图卷积（``--model gcn``）

严格保留原版的损失构造方式：**只对第一层的权重做 L2 正则**
（TF 版是 ``for var in self.layers[0].vars.values()``），
这一点对复现论文数值很关键 —— 若改成全层正则，准确率会偏离原文。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import Dense, GraphConvolution
from .metrics import masked_accuracy, masked_softmax_cross_entropy


def identity(x):
    """恒等激活，对应 TF 原版的 ``act=lambda x: x``（输出层不激活，保留 logits）。"""
    return x


class Model(nn.Module):
    """模型基类，负责损失、准确率与预测。"""

    def __init__(self, **kwargs):
        super(Model, self).__init__()
        self.name = kwargs.get('name', self.__class__.__name__.lower())
        self.logging = kwargs.get('logging', False)

        self.layers = nn.ModuleList()
        self.input_dim = None
        self.output_dim = None
        self.dropout_rate = 0.5
        self.weight_decay = 5e-4
        self.learning_rate = 0.01

        self.outputs = None

    def _build(self):
        raise NotImplementedError

    def forward(self, features):
        h = features
        for layer in self.layers:
            h = layer(h)
        self.outputs = h
        return h

    def _loss(self):
        # ---- 权重衰减：只作用于第一层（与 TF 原版一致）----
        # 关键：TF 的 tf.nn.l2_loss(t) 语义是 sum(t^2) / numel（逐元素取均值），
        # 不是 sum(t^2)/2，也不是 sum(t^2)。这里必须用 numel 归一，
        # 否则正则强度会差出上千倍，精度会明显偏低。
        loss = 0
        for var in self.layers[0].parameters():
            loss += self.weight_decay * torch.sum(var * var) / var.numel()

        # ---- 交叉熵 ----
        loss = loss + masked_softmax_cross_entropy(
            self.outputs, self.labels, self.labels_mask)
        return loss

    def _accuracy(self):
        return masked_accuracy(
            self.outputs, self.labels, self.labels_mask)

    def predict(self):
        return torch.softmax(self.outputs, dim=1)


class MLP(Model):
    """不含图信息的MLP 基线。"""

    def __init__(self, input_dim, output_dim, supports, hidden1=16,
                 dropout=0.5, weight_decay=5e-4, learning_rate=0.01, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate

        self._build(hidden1)
        self._init_parameters()

    def _build(self, hidden1):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=hidden1,
                                 dropout=self.dropout_rate,
                                 act=torch.relu,
                                 sparse_inputs=True,
                                 logging=self.logging,
                                 name='dense_1'))
        self.layers.append(Dense(input_dim=hidden1,
                                 output_dim=self.output_dim,
                                 dropout=self.dropout_rate,
                                 act=identity,
                                 logging=self.logging,
                                 name='dense_2'))

    def _init_parameters(self):
        pass


class GCN(Model):
    """两层图卷积网络（Kipf & Welling 2017）。"""

    def __init__(self, input_dim, output_dim, supports, hidden1=16,
                 dropout=0.5, weight_decay=5e-4, learning_rate=0.01, **kwargs):
        super(GCN, self).__init__(**kwargs)
        self.supports = supports
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate

        self._build(hidden1)
        self._init_parameters()

    def _build(self, hidden1):
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=hidden1,
                                            supports=self.supports,
                                            dropout=self.dropout_rate,
                                            act=torch.relu,
                                            sparse_inputs=True,
                                            logging=self.logging,
                                            name='graphconvolution_1'))
        self.layers.append(GraphConvolution(input_dim=hidden1,
                                            output_dim=self.output_dim,
                                            supports=self.supports,
                                            dropout=self.dropout_rate,
                                            act=identity,
                                            logging=self.logging,
                                            name='graphconvolution_2'))

    def _init_parameters(self):
        # PyTorch 会遍历 module 自动注册参数，这里无需额外处理；
        # Glorot 初始化已在 GraphConvolution.reset_parameters 中完成。
        pass