"""PyG 版 GCN 模型定义。

与手写实现 ``gcn/models.py`` 的 ``GCN`` 结构**一一对应**，只是把
``GraphConvolution`` 换成了 ``torch_geometric.nn.GCNConv``：

    手写版                              PyG 版
    ----------------------------------  ----------------------------------
    GraphConvolution(in, out, [S], ...) GCNConv(in, out, bias=False)
    手动 preprocess_adj 得到 S           GCNConv 内部 add_self_loops + gcn_norm
    F.dropout(稠密) / sparse_dropout    F.dropout(稠密)
    L2 由 Model._loss 手动加            同样手动加（不用 weight_decay 参数）

**为什么不用 GCNConv 的 ``weight_decay`` 参数**：那个参数作用于
``lin.weight`` 的**原始梯度**（等价于 L2），但作用范围是**每一层**；
而论文与 TF 原版只对**第一层**做 L2。为保持与手写实现可比，
这里统一在 loss 里手动加正则，且只加第一层。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class PyGGCN(nn.Module):
    """基于 ``GCNConv`` 的多层图卷积网络。

    默认 ``num_layers=2``，即论文的经典配置。支持加深（num_layers>2）用于
    观察 [[notes/06-深度困境-过平滑与过挤压|过平滑]] 现象——这是 PyG 版相对
    手写版的一个便利之处：手写版固定两层，PyG 版可一键加深。

    结构::

        x -> [GCNConv -> ReLU -> Dropout] × (L-1) -> GCNConv -> logits

    注意最后一层不加 ReLU、不加 dropout，输出原始 logits 供交叉熵使用。
    """

    def __init__(self, input_dim, output_dim, hidden1=16, num_layers=2,
                 dropout=0.5, weight_decay=5e-4, cached=False):
        """构造网络。

        :param input_dim:  输入特征维度 F
        :param output_dim: 类别数 C
        :param hidden1:    隐藏层宽度
        :param num_layers: 图卷积层数，2 为论文设定
        :param dropout:    丢弃概率
        :param weight_decay: L2 系数（在 loss 中手动使用，不传给 GCNConv）
        :param cached:     是否缓存归一化邻接矩阵。全图训练可设 True 提速，
                           但每轮会复用同一份归一化结果，仅在静态图上正确
        """
        super(PyGGCN, self).__init__()

        if num_layers < 1:
            raise ValueError('num_layers 至少为 1')

        self.num_layers = num_layers
        self.dropout = dropout
        self.weight_decay = weight_decay

        # ---- 逐层构造 ----
        # bias=False 与论文一致（GCN 原版两层均无偏置）
        self.convs = nn.ModuleList()
        if num_layers == 1:
            # 单层：直接 F -> C，等价于逻辑回归 + 图传播
            self.convs.append(GCNConv(input_dim, output_dim,
                                      bias=False, cached=cached))
        else:
            # 首层：F -> hidden1
            self.convs.append(GCNConv(input_dim, hidden1,
                                      bias=False, cached=cached))
            # 中间层：hidden1 -> hidden1
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden1, hidden1,
                                          bias=False, cached=cached))
            # 末层：hidden1 -> C
            self.convs.append(GCNConv(hidden1, output_dim,
                                      bias=False, cached=cached))

    def forward(self, data):
        """前向传播。

        :param data: PyG ``Data`` 对象（需含 ``x`` 与 ``edge_index``）
        :return: (N, C) 的 logits
        """
        x, edge_index = data.x, data.edge_index

        # 前 L-1 层：卷积 + 激活 + dropout
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            # 用 self.training 判断而非外部标志，与手写实现一致
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 最后一层：只卷积，不加激活（保留 logits）
        x = self.convs[-1](x, edge_index)
        return x

    def l2_loss(self):
        """第一层权重的 L2 正则，语义对齐 ``tf.nn.l2_loss``。

        TF 的 ``l2_loss(t)`` 是 ``sum(t²)/numel``（逐元素均值），
        不是 ``sum(t²)/2``。写错会让正则强度差出上千倍，
        这是本项目在 [[README|手写实现]] 中记录过的第一个坑。

        只遍历 ``convs[0]`` 的原因见模块 docstring：论文只正则第一层。
        """
        loss = 0
        for var in self.convs[0].parameters():
            loss = loss + self.weight_decay * torch.sum(var * var) / var.numel()
        return loss


def count_parameters(model):
    """统计可训练参数总数（含偏置，便于与手写版对照）。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
