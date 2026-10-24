"""PyTorch Geometric 版 GCN 训练入口（使用 ``GCNConv``）。

与手写实现 ``train.py`` 的关系：**同一份数据、同一套划分、同一组超参数**，
只把图卷积层换成 PyG 的 ``GCNConv``。因此两者的精度差异可归因于
「实现差异」而非「数据或划分差异」。

用法：
    python train_pyg.py --dataset cora
    python train_pyg.py --dataset citeseer --source csv
    python train_pyg.py --dataset pubmed --num_layers 3     # 加深，看满平滑
    python train_pyg.py --dataset cora --num_layers 1       # 单层，看传播本身贡献

输出格式与 ``train.py`` 完全一致，便于逐行对照。

对应文档：[[GNN-EDU-Research#3. 谱域方法|§3 谱域方法]]、§4 空域方法
"""

from __future__ import division, print_function

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from gcn import silence_sparse_warning
from pyg_gcn import describe, silence_pyg_warning, to_pyg_data
from pyg_gcn.models import PyGGCN, count_parameters

# 固定随机种子，与 train.py 保持一致
seed = 123
np.random.seed(seed)
torch.manual_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description='GCN (PyTorch Geometric / GCNConv)')
    parser.add_argument('--dataset', type=str, default='cora',
                        choices=['cora', 'citeseer', 'pubmed'],
                        help='Dataset string.')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='图卷积层数，2 为论文设定，>2 可观察过平滑。')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--hidden1', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--early_stopping', type=int, default=10)
    parser.add_argument('--source', type=str, default='pickle',
                        choices=['pickle', 'csv'],
                        help='数据源：pickle 或 csv（export_csv.py 生成）。')
    parser.add_argument('--csv_dir', type=str, default=None,
                        help='CSV 数据目录，默认为 <项目>/data/csv。')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA.')
    parser.add_argument('--seed', type=int, default=123,
                        help='随机种子（影响初始化与 dropout 掩码）。')
    return parser.parse_args()


def masked_loss(logits, y, mask, model, normalize=True):
    """带 mask 的交叉熵 + 第一层 L2，语义与手写版 ``Model._loss`` 一致。

    手写版用「mask 归一化到均值 1」的 TF 写法；这里用等价的
    ``F.cross_entropy(..., reduction='none')`` 后按 mask 求均值，
    两者数学等价，差异只在浮点累加顺序（约 1e-8 量级）。

    :param normalize: True 时按 mask 内样本数求平均（标准做法）
    """
    ce = F.cross_entropy(logits[mask], y[mask], reduction='mean' if normalize else 'sum')
    return model.l2_loss() + ce


def accuracy(logits, y, mask):
    """mask 内的分类准确率。"""
    pred = logits[mask].argmax(dim=1)
    return (pred == y[mask]).float().mean()


def main():
    FLAGS = parse_args()

    # 重新设种子：命令行可覆盖默认，保证可复现
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)

    # ---- 数据 ----
    data = to_pyg_data(FLAGS.dataset, source=FLAGS.source,
                       csv_dir=FLAGS.csv_dir)
    print('Dataset: {}, Model: gcn (PyG GCNConv, {} layers)'.format(
        FLAGS.dataset, FLAGS.num_layers))
    print(describe(data))
    print('Num supports: 1 (GCNConv 内部归一化), Hidden units: {}'.format(
        FLAGS.hidden1))

    device = torch.device('cuda' if torch.cuda.is_available() and
                          not FLAGS.no_cuda else 'cpu')

    # ---- 模型 ----
    model = PyGGCN(input_dim=data.x.shape[1],
                   output_dim=data.num_classes,
                   hidden1=FLAGS.hidden1,
                   num_layers=FLAGS.num_layers,
                   dropout=FLAGS.dropout,
                   weight_decay=FLAGS.weight_decay)
    model = model.to(device)
    print('Trainable parameters: {}'.format(count_parameters(model)))

    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate)

    cost_val = []

    # ---- 训练 ----
    for epoch in range(FLAGS.epochs):
        t = time.time()

        model.train()
        optimizer.zero_grad()
        logits = model(data)
        loss = masked_loss(logits, data.y, data.train_mask, model)
        loss.backward()
        optimizer.step()
        train_acc = accuracy(logits, data.y, data.train_mask).item()

        # ---- 验证 ----
        model.eval()
        with torch.no_grad():
            val_logits = model(data)
            val_loss = masked_loss(val_logits, data.y, data.val_mask, model)
            val_acc = accuracy(val_logits, data.y, data.val_mask)
        cost_val.append(val_loss.item())

        # 输出格式与 train.py 逐字段对齐
        print("Epoch:", '%04d' % (epoch + 1),
              "train_loss=", "{:.5f}".format(loss.item()),
              "train_acc=", "{:.5f}".format(train_acc),
              "val_loss=", "{:.5f}".format(val_loss.item()),
              "val_acc=", "{:.5f}".format(val_acc.item()),
              "time=", "{:.5f}".format(time.time() - t))

        # 早停判据与 train.py 完全一致
        if epoch > FLAGS.early_stopping and \
                cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping + 1):-1]):
            print("Early stopping...")
            break

    print("Optimization Finished!")

    # ---- 测试 ----
    t_test = time.time()
    model.eval()
    with torch.no_grad():
        test_logits = model(data)
        test_cost = masked_loss(test_logits, data.y, data.test_mask, model)
        test_acc = accuracy(test_logits, data.y, data.test_mask)
    print("Test set results:", "cost=", "{:.5f}".format(test_cost.item()),
          "accuracy=", "{:.5f}".format(test_acc.item()),
          "time=", "{:.5f}".format(time.time() - t_test))


if __name__ == '__main__':
    # 两层上下文：稀疏不变量（本项目）+ PyG 的 typing 弃用告警
    with silence_sparse_warning(), silence_pyg_warning():
        main()
