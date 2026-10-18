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

from gcn.models import GCN, MLP
from gcn.utils import (chebyshev_polynomials, load_data, preprocess_adj)

# 固定随机种子，与 TF 原版一致（seed=123）
seed = 123
np.random.seed(seed)
torch.manual_seed(seed)


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
    return parser.parse_args()


def evaluate(model, features, labels, mask):
    """在给定标签子集上评估，返回 (loss, accuracy, duration)。"""
    t_test = time.time()
    model.eval()
    with torch.no_grad():
        model(features)
        loss = model._loss()
        acc = model._accuracy()
    return loss, acc, (time.time() - t_test)


def main():
    FLAGS = parse_args()

    # Load data
    (adj, features, y_train, y_val, y_test,
     train_mask, val_mask, test_mask, adj_raw) = load_data(FLAGS.dataset)

    # Some preprocessing
    if FLAGS.model == 'gcn':
        support = [preprocess_adj(adj_raw)]
        num_supports = 1
        model_func = GCN
    elif FLAGS.model == 'gcn_cheby':
        support = chebyshev_polynomials(adj_raw, FLAGS.max_degree)
        num_supports = 1 + FLAGS.max_degree
        model_func = GCN
    elif FLAGS.model == 'dense':
        support = [preprocess_adj(adj_raw)]  # Not used
        num_supports = 1
        model_func = MLP
    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

    input_dim = features.shape[1]
    output_dim = y_train.shape[1]
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
    model = model.to(device)
    features = features.to(device)
    y_train, y_val, y_test = y_train.to(device), y_val.to(device), y_test.to(device)
    train_mask, val_mask, test_mask = (train_mask.to(device),
                                      val_mask.to(device), test_mask.to(device))

    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate)

    cost_val = []

    # Train model
    for epoch in range(FLAGS.epochs):
        t = time.time()

        # Training step
        model.train()
        model.labels = y_train
        model.labels_mask = train_mask
        optimizer.zero_grad()
        model(features)
        loss = model._loss()
        loss.backward()
        optimizer.step()
        train_acc = model._accuracy().item()

        # Validation
        model.labels = y_val
        model.labels_mask = val_mask
        val_loss, val_acc, _ = evaluate(model, features, y_val, val_mask)
        cost_val.append(val_loss.item())

        # Print results（格式与 TF 版一致）
        print("Epoch:", '%04d' % (epoch + 1),
              "train_loss=", "{:.5f}".format(loss.item()),
              "train_acc=", "{:.5f}".format(train_acc),
              "val_loss=", "{:.5f}".format(val_loss.item()),
              "val_acc=", "{:.5f}".format(val_acc.item()),
              "time=", "{:.5f}".format(time.time() - t))

        if epoch > FLAGS.early_stopping and \
                cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping + 1):-1]):
            print("Early stopping...")
            break

    print("Optimization Finished!")

    # Testing
    model.labels = y_test
    model.labels_mask = test_mask
    test_cost, test_acc, test_duration = evaluate(model, features, y_test, test_mask)
    print("Test set results:", "cost=", "{:.5f}".format(test_cost.item()),
          "accuracy=", "{:.5f}".format(test_acc.item()),
          "time=", "{:.5f}".format(test_duration))


if __name__ == '__main__':
    main()