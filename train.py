"""PyTorch 版训练入口。

逐行对照 TensorFlow 原版 ``gcn/train.py`` 移植，**超参数默认值完全一致**，
输出格式也对齐原版，便于直接比对复现精度。

用法：
    python train.py --dataset cora --model gcn
    python train.py --dataset citeseer --model gcn_cheby --max_degree 3
    python train.py --dataset pubmed --model dense      # MLP 基线

对应文档：[[GNN-EDU-Research#3. 谱域方法|§3 谱域方法]]、§4 空域方法
"""

from __future__ import division, print_function

import argparse
import time

import numpy as np
import torch

from gcn import silence_sparse_warning
from gcn.models import GCN, MLP
from gcn.utils import (chebyshev_polynomials, load_data, preprocess_adj)

# 固定随机种子，与 TF 原版一致（seed=123）
# 必须放在任何网络初始化之前，否则权重初始化会因随机数消耗顺序不同而不同
seed = 123
np.random.seed(seed)     # 控制 numpy 随机（如 dropout 掩码的底层实现）
torch.manual_seed(seed)  # 控制 torch 权重初始化


def parse_args():
    parser = argparse.ArgumentParser(description='GCN (PyTorch)')
    parser.add_argument('--dataset', type=str, default='cora',
                        choices=['cora', 'citeseer', 'pubmed'],
                        help='Dataset string.')
    parser.add_argument('--model', type=str, default='gcn',
                        choices=['gcn', 'gcn_cheby', 'dense'],
                        help='Model string.')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--hidden1', type=int, default=16,
                        help='Number of units in hidden layer 1.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight for L2 loss on embedding matrix.')
    parser.add_argument('--early_stopping', type=int, default=10,
                        help='Tolerance for early stopping (# of epochs).')
    parser.add_argument('--max_degree', type=int, default=3,
                        help='Maximum Chebyshev polynomial degree.')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA.')
    parser.add_argument('--source', type=str, default='pickle',
                        choices=['pickle', 'csv'],
                        help='数据源：pickle（gcn/data 原始文件）或 csv（export_csv.py 生成）。')
    parser.add_argument('--csv_dir', type=str, default=None,
                        help='CSV 数据目录，默认为 <项目>/data/csv。')
    return parser.parse_args()


def evaluate(model, features, labels, mask):
    """在给定标签子集上评估，返回 (loss, accuracy, duration)。

    评估时必须关掉 dropout：``model.eval()`` 会让 F.dropout 走恒等分支，
    否则预测结果带随机噪声，准确率会偏低且不可复现。

    :param model:  待评估的模型
    :param features: 节点特征，(N, F)
    :param labels:   one-hot 标签，(N, C)
    :param mask:     布尔 mask，(N,)
    :return: (loss张量, acc张量, 耗时秒数)
    """
    t_test = time.time()      # 计时（评估应无梯度，比训练快）
    model.eval()              # 切换到推理模式：关闭 dropout
    with torch.no_grad():     # 关闭自动求导，省显存且更快
        model(features)       # 前向一次，结果缓存在 model.outputs
        loss = model._loss()  # 用当前 labels/labels_mask 算损失
        acc = model._accuracy()
    return loss, acc, (time.time() - t_test)


def main():
    FLAGS = parse_args()

    # Load data
    # load_data 统一返回 9 元组：adj(features 已 torch 化) / features /
    # y_train,y_val,y_test / train_mask,val_mask,test_mask / adj_raw(scipy 域)
    (adj, features, y_train, y_val, y_test,
     train_mask, val_mask, test_mask, adj_raw) = load_data(
        FLAGS.dataset, source=FLAGS.source, csv_dir=FLAGS.csv_dir)

    # Some preprocessing
    if FLAGS.model == 'gcn':
        # 标准 GCN：单个 support = D̃^-1/2 Ã D̃^-1/2（加自环 + 对称归一化）
        support = [preprocess_adj(adj_raw)]
        num_supports = 1                   # 只有 1 个权重矩阵
        model_func = GCN
    elif FLAGS.model == 'gcn_cheby':
        # ChebNet：K 阶 Chebyshev 基作为 K+1 个 support
        support = chebyshev_polynomials(adj_raw, FLAGS.max_degree)
        num_supports = 1 + FLAGS.max_degree
        model_func = GCN                    # 同一个类，只是 support 多了几个
    elif FLAGS.model == 'dense':
        # MLP 基线：虽然不用图，但仍传一个 support 以保持构造签名统一
        support = [preprocess_adj(adj_raw)]  # Not used
        num_supports = 1
        model_func = MLP
    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

    # ---- 从数据推断网络维度 ----
    input_dim = features.shape[1]      # F：输入特征维度
    output_dim = y_train.shape[1]      # C：类别数
    print('Dataset: {}, Model: {}'.format(FLAGS.dataset, FLAGS.model))
    print('Num nodes: {}, Num features: {}, Num classes: {}'.format(
        adj.shape[0], input_dim, output_dim))
    print('Num supports: {}, Hidden units: {}'.format(num_supports, FLAGS.hidden1))

    # Create model
    model = model_func(input_dim=input_dim,
                       output_dim=output_dim,
                       supports=support,
                       hidden1=FLAGS.hidden1,
                       dropout=FLAGS.dropout,
                       weight_decay=FLAGS.weight_decay,
                       learning_rate=FLAGS.learning_rate,
                       logging=True)

    device = torch.device('cuda' if torch.cuda.is_available() and
                          not FLAGS.no_cuda else 'cpu')
    model = model.to(device)                        # 参数搬到 GPU
    features = features.to(device)                  # 特征搬到 GPU
    y_train, y_val, y_test = y_train.to(device), y_val.to(device), y_test.to(device)
    train_mask, val_mask, test_mask = (train_mask.to(device),
                                       val_mask.to(device), test_mask.to(device))

    # Adam 优化器；超参与 TF 版一致（lr=0.01，b1/b2 用 PyTorch 默认 0.9/0.999）
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate)

    cost_val = []          # 记录每轮验证损失，供早停判断

    # Train model
    for epoch in range(FLAGS.epochs):
        t = time.time()   # 计时本轮耗时

        # ---- Training step ----
        model.train()# 切换到训练模式（启用 dropout）
        model.labels = y_train        # 损失用训练集标签
        model.labels_mask = train_mask  # 损失只在训练集节点上计算
        optimizer.zero_grad()# 清空上一轮梯度
        model(features)        # 前向，输出缓存在 model.outputs
        loss = model._loss()   # L2 + masked交叉熵
        loss.backward()# 反向传播，自动求导
        optimizer.step()       # 更新参数
        train_acc = model._accuracy().item()   # 训练集准确率（仅作监控）

        # ---- Validation ----
        model.labels = y_val            # 换成验证集标签
        model.labels_mask = val_mask    # 换成验证集 mask
        val_loss, val_acc, _ = evaluate(model, features, y_val, val_mask)
        cost_val.append(val_loss.item())  # 记入历史，供早停比较

        # Print results（格式与 TF 版一致）
        print("Epoch:", '%04d' % (epoch + 1),
              "train_loss=", "{:.5f}".format(loss.item()),
              "train_acc=", "{:.5f}".format(train_acc),
              "val_loss=", "{:.5f}".format(val_loss.item()),
              "val_acc=", "{:.5f}".format(val_acc.item()),
              "time=", "{:.5f}".format(time.time() - t))

        # ---- Early stopping ----
        # 判据：最近一轮验证损失高于前 early_stopping 轮的平均值，
        # 说明已经过拟合，继续训练只会更差
        if epoch > FLAGS.early_stopping and \
                cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping + 1):-1]):
            print("Early stopping...")
            break

    print("Optimization Finished!")

    # ---- Testing：整个测试集一次性评估 ----
    model.labels = y_test              # 换成测试集标签
    model.labels_mask = test_mask      # 换成测试集 mask
    test_cost, test_acc, test_duration = evaluate(model, features, y_test, test_mask)
    print("Test set results:", "cost=", "{:.5f}".format(test_cost.item()),
          "accuracy=", "{:.5f}".format(test_acc.item()),
          "time=", "{:.5f}".format(test_duration))


if __name__ == '__main__':
    # 用上下文管理器显式声明稀疏不变量检查策略，消除 PyTorch 2.14 的 UserWarning
    with silence_sparse_warning():
        main()