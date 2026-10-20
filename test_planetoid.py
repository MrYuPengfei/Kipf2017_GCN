"""``train_planetoid.py`` 的最小验证集。

只测三件最容易出错、且 PyG 悄悄替你决定的事：
    1. 数据划分与规模是否和论文 Table 1 一致
    2. Planetoid 在处理时对原始自环做了什么
    3. GCNConv 的归一化是不是论文里的 D^-1/2 (A+I) D^-1/2

用法：
    python test_planetoid.py            # 全部数据集
    python test_planetoid.py cora       # 只跑一个
"""

import sys

import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.transforms import NormalizeFeatures

# 论文 Table 1 的规模：(节点数, 特征维度, 类别数, 原始边数)
TABLE1 = {
    'cora':    (2708, 1433, 7, 10556),
    'citeseer': (3327, 3703, 6, 9104),
    'pubmed':  (19717, 500, 3, 88648),
}

ROOT = 'data/Planetoid'
ok = fail = 0


def check(cond, msg):
    global ok, fail
    if cond:
        ok += 1
        print(f'  [ok] {msg}')
    else:
        fail += 1
        print(f'  [FAIL] {msg}')


def test_scale(name):
    """A. 数据规模与论文 Table 1 逐项对齐。"""
    print(f'\nA. 数据规模 · {name}')
    data = Planetoid(ROOT, name)[0]
    n, f, c, e = TABLE1[name]

    check(data.num_nodes == n, f'节点数 {data.num_nodes} == {n}')
    check(data.num_features == f, f'特征维度 {data.num_features} == {f}')
    check(int(data.y.max()) + 1 == c, f'类别数 {int(data.y.max()) + 1} == {c}')
    # PyG 的 edge_index 已按无向图双向存储，故直接等于论文的边数
    check(data.num_edges == e, f'边数 {data.num_edges} == {e}')

    # 论文的公开划分是固定的那一份：每类 20 训练 / 500 验证 / 1000 测试
    check(int(data.val_mask.sum()) == 500, '验证集 500 个节点')
    check(int(data.test_mask.sum()) == 1000, '测试集 1000 个节点')
    check(int(data.train_mask.sum()) == 20 * c,
          f'训练集 {int(data.train_mask.sum())} == 20 × {c} 类')
    check(bool((data.train_mask & data.val_mask).sum() == 0),
          'train 与 val mask 不重叠')


def test_self_loop(name):
    """B. Planetoid 会丢弃原始自环，Cora 无自环、Citeseer 有 124 个。"""
    print(f'\nB. 自环 · {name}')
    data = Planetoid(ROOT, name)[0]
    loop = (data.edge_index[0] == data.edge_index[1]).sum().item()

    # Citeseer 原始 pickle 里有 124 个自环，Pubmed 有 3 个，Cora 没有；
    # Planetoid 读取时全部剔除，所以这里三个数据集都为 0
    check(loop == 0,
          f'edge_index 中的自环数 {loop} == 0（Planetoid 已剔除原始自环）')


def test_normalization(name):
    """C. GCNConv 的归一化 = D^-1/2 (A+I) D^-1/2，且对角元恒为 1。

    走 ``gcn_norm(edge_index, edge_weight)``——这正是 GCNConv 内部实际
    调用的签名，而不是自己拼 scipy 矩阵。
    """
    import numpy as np
    import scipy.sparse as sp
    from torch_geometric.nn.conv.gcn_conv import gcn_norm
    from torch_geometric.utils import add_self_loops

    print(f'\nC. 归一化 · {name}')
    data = Planetoid(ROOT, name)[0]
    n = data.num_nodes
    idx = data.edge_index

    edge_weight = torch.ones(idx.shape[1])
    # PyG 2.x 的 gcn_norm 返回 (edge_index, edge_weight) 元组
    norm_idx, norm_w = gcn_norm(idx, edge_weight, num_nodes=n)
    adj_n = sp.coo_matrix((norm_w.numpy(), (norm_idx[0].numpy(),
                                            norm_idx[1].numpy())),
                          shape=(n, n)).toarray()

    # 独立重建：先按多值编码累加成矩阵，加 I，再做度归一化
    a = sp.coo_matrix((np.ones(idx.shape[1]), (idx[0].numpy(), idx[1].numpy())),
                      shape=(n, n)).toarray() + np.eye(n)
    d = a.sum(1)
    d_inv_sqrt = np.zeros_like(d)
    nz = d > 0
    d_inv_sqrt[nz] = np.power(d[nz], -0.5)
    ref = d_inv_sqrt[:, None] * a * d_inv_sqrt[None, :]

    err = np.abs(adj_n - ref).max()
    check(err < 1e-5, f'与独立构造的 D^-1/2(A+I)D^-1/2 最大偏差 {err:.2e}')

    # 自环语义：归一化后的对角元应是 1/deg（自环权重为 1，再被度归一化除一次）
    deg = a.sum(1)
    diag = np.diag(adj_n)
    check(np.allclose(diag, 1.0 / deg, atol=1e-6),
          '归一化后对角元 == 1/deg')

    # 「覆盖而非累加」：要查归一化之前的 add_self_loops。
    # 若原始数据自带自环，累加式会得到 2，而 PyG 一律置1。
    looped_idx, _ = add_self_loops(idx, num_nodes=n)
    looped = sp.coo_matrix(
        (np.ones(looped_idx.shape[1]),
         (looped_idx[0].numpy(), looped_idx[1].numpy())),
        shape=(n, n)).toarray()
    check(np.allclose(np.diag(looped), 1.0),
          'add_self_loops 后对角元恒为 1（覆盖语义，非累加）')


def test_model_forward(name):
    """D. 模型能跑，且2 层输出形状正确。"""
    print(f'\nD. 前向 · {name}')
    from train_planetoid import GCN
    ds = Planetoid(ROOT, name, transform=NormalizeFeatures())
    data = ds[0]
    out = GCN(data.num_features, 16, ds.num_classes, 2, 0.5)(data.x, data.edge_index)
    check(out.shape == (data.num_nodes, ds.num_classes),
          f'输出形状 {tuple(out.shape)} == ({data.num_nodes}, {ds.num_classes})')
    check(torch.isfinite(out).all(), '输出无 NaN/Inf')


if __name__ == '__main__':
    names = sys.argv[1:] or list(TABLE1)
    for nm in names:
        test_scale(nm)
        test_self_loop(nm)
        test_normalization(nm)
        test_model_forward(nm)
    print(f'\n{ok} passed, {fail} failed')
    sys.exit(1 if fail else 0)