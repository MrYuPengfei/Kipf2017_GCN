"""PyTorch 版评估指标。

对照 TensorFlow 原版 ``gcn/metrics.py`` 移植。

注意 mask 的处理方式与 TF 版完全一致：
    mask = cast(mask, float32)
    mask /= reduce_mean(mask)      # 归一化到均值 1
    loss *= mask
    return reduce_mean(loss)
这样最终结果等价于「只在 mask=1 的位置上求平均」。
"""

import torch
import torch.nn.functional as F


def masked_softmax_cross_entropy(preds, labels, mask):
    """带 mask 的 softmax 交叉熵。"""
    # PyTorch的 cross_entropy 内部已做 log_softmax，等价于 TF 的
    # softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    # labels 为 one-hot 形式，故用 soft labels 版本
    loss = torch.sum(-labels * F.log_softmax(preds, dim=1), dim=1)

    mask = mask.type(torch.float32)
    mask = mask / torch.mean(mask)
    loss = loss * mask

    return torch.mean(loss)


def masked_accuracy(preds, labels, mask):
    """带 mask 的准确率。"""
    correct_prediction = torch.eq(torch.argmax(preds, dim=1),
                                  torch.argmax(labels, dim=1))
    accuracy_all = correct_prediction.type(torch.float32)
    mask = mask.type(torch.float32)
    mask = mask / torch.mean(mask)
    accuracy_all = accuracy_all * mask
    return torch.mean(accuracy_all)