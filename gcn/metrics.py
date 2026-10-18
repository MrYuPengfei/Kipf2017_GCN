"""PyTorch 版评估指标。

对照 TensorFlow 原版 ``gcn/metrics.py`` 移植。

注意 mask 的处理方式与 TF 版完全一致：
    mask = cast(mask, float32)
    mask /= reduce_mean(mask)      # 归一化到均值 1
    loss *= mask
    return reduce_mean(loss)
这样最终结果等价于「只在 mask=1 的位置上求平均」。

为什么不能"顺手优化"成 sum(loss[mask]) / mask.sum()
----------------------------------------------------
数学上两者等价，但浮点累加顺序不同：TF 版是先在**全N 行**上算loss、
再乘 mask、最后对全 N 行求均值；而 sum 写法只累加被选中的行。
在 mask 稀疏（如 Cora 测试集 1000/2708）时两者末位会差约 1e-8。
本项目的定位是忠实移植，故保留 TF 原写法，test.py 的 H1/H2 断言专门核对这一点。
"""

import torch                         # 张量运算
import torch.nn.functional as F      # log_softmax


def masked_softmax_cross_entropy(preds, labels, mask):
    """带 mask 的 softmax 交叉熵。

    对应 TF 的 ``softmax_cross_entropy_with_logits(logits=preds, labels=labels)``
    在 one-hot 标签下的展开形式：``-Σ_c y_c · log_softmax(z)_c``。

    :param preds:  (N, C) 未归一化的 logits（模型最后一层的 identity 输出）
    :param labels: (N, C) one-hot 标签
    :param mask:   (N,) 布尔张量，True 表示该节点参与本次损失计算
    :return: 标量张量，损失值
    """
    # ---- 1. 逐样本交叉熵 ----
    # log_softmax(preds, dim=1) 对每个样本的 C 个类别做 log-sum-exp 归一化
    # 与 -labels 相乘并沿类别维求和：one-hot 标签下等价于取真实类别那一项的负对数概率
    # shapes: labels (N,C) * log_softmax (N,C) -> (N,C)，sum(dim=1) -> (N,)
    loss = torch.sum(-labels * F.log_softmax(preds, dim=1), dim=1)

    # ---- 2. mask 归一化到均值 1 ----
    # 这是 TF 原版的写法，不能改成 sum(mask) 归一化，否则浮点结果会有末位差异
    mask = mask.type(torch.float32)         # bool -> float32 才能参与乘法
    mask = mask / torch.mean(mask)           # 归一化到均值 1（全 N 行的平均）

    # ---- 3. 应用 mask 并对全 N 行求均值 ----
    # 未被选中的行 loss 置 0，但仍参与最后的 mean，
    # 由于 mask 已归一化到均值 1，这与「只在选中行上求平均」数学等价
    loss = loss * mask                       # (N,) * (N,) -> (N,)

    return torch.mean(loss)                  # 对全 N 行求均值 -> 标量


def masked_accuracy(preds, labels, mask):
    """带 mask 的准确率。

    对应 TF 原版的 ``masked_accuracy``。

    :param preds:  (N, C) logits
    :param labels: (N, C) one-hot 标签
    :param mask:   (N,) 布尔张量，True 表示该节点参与准确率统计
    :return: 标量张量，准确率
    """
    # ---- 1. 逐样本是否预测正确 ----
    # argmax 对 logits 取最大类；标签是 one-hot，argmax 等价于类别编号
    # torch.eq 做逐行比较，(N,) 与 (N,) -> (N,) 布尔张量
    correct_prediction = torch.eq(torch.argmax(preds, dim=1),
                                  torch.argmax(labels, dim=1))

    # ---- 2. 布尔转 float32，作为 0/1 得分 ----
    accuracy_all = correct_prediction.type(torch.float32)

    # ---- 3. 与交叉熵同样的 mask 归一化逻辑（保持两条路径一致）----
    mask = mask.type(torch.float32)         # bool -> float32
    mask = mask / torch.mean(mask)           # 归一化到均值 1

    # ---- 4. 未选中位置置 0 后求均值 ----
    accuracy_all = accuracy_all * mask       # (N,) * (N,) -> (N,)

    return torch.mean(accuracy_all)          # 对全 N 行求均值 -> 标量
