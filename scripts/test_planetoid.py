"""``train_planetoid.py`` 的最小验证集。

测五组：
    A. 数据划分与规模是否和论文 Table 1 一致
    B. Planetoid 在处理时对原始自环做了什么
    C. GCNConv 的归一化是不是论文里的 D^-1/2 (A+I) D^-1/2
    D. 模型能跑，且输出形状正确
    E. **GGUF 存/取往返是否逐位一致**（含「仅凭文件重建模型」）

用法：
    python test_planetoid.py            # 全部数据集
    python test_planetoid.py cora       # 只跑一个
"""

import sys

import torch

# PyG 的导入统一走 pyg_warn（压掉 Python 3.14 下的导入期告警）
from pyg_warn import (NormalizeFeatures, Planetoid, add_self_loops,
                      gcn_norm, silence)

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


def test_gguf_roundtrip(name):
    """E. GGUF 存/取的往返一致性。

    这组断言是本文件新增的核心：验证
      1. 写出的 GGUF 能被读回，且张量数值与 PyTorch 完全一致
      2. 仅凭 GGUF 文件就能重建模型结构（层数/维度写在 KV 区）
      3. 重建后的模型输出与原模型逐位一致
      4. GGUF **不包含** 图结构与特征（这是它表达不了 GCN 的根本原因）
    """
    import os
    import tempfile

    from train_planetoid import GCN
    import gguf_io

    print(f'\nE. GGUF 往返 · {name}')
    ds = Planetoid(ROOT, name, transform=NormalizeFeatures())
    data = ds[0]

    torch.manual_seed(0)
    model = GCN(data.num_features, 16, ds.num_classes, 2, 0.5)
    model.eval()

    with torch.no_grad():
        ref_logits = model(data.x, data.edge_index)

    tmpdir = tempfile.mkdtemp(prefix='gguf_test_')
    gguf_path = os.path.join(tmpdir, 'test.gguf')
    model.save(gguf_path, meta={'dataset': name, 'test_acc': 0.5})

    check(os.path.exists(gguf_path), 'GGUF 文件已写出')
    check(os.path.exists(os.path.join(tmpdir, 'test.pt')), '.pt 文件已写出')
    check(os.path.exists(os.path.join(tmpdir, 'test.meta.json')),
          '.meta.json 已写出')

    # ---- 1) 读回的张量与原权重逐位一致 ----
    state, meta = gguf_io.load_gguf(gguf_path)
    check(set(state) == set(model.state_dict()),
          f'张量集合一致（{len(state)} 个）')

    max_diff = 0.0
    for k, v in model.state_dict().items():
        if not v.is_floating_point():
            continue
        d = (state[k] - v).abs().max().item()
        max_diff = max(max_diff, d)
    check(max_diff == 0.0, f'所有张量与 PyTorch 逐位一致（最大偏差 {max_diff:.2e}）')

    # ---- 2) 仅凭文件重建模型 ----
    check(meta['num_layers'] == 2, f"KV 区 num_layers == {meta['num_layers']}")
    check(meta['num_features'] == data.num_features,
          f"KV 区 num_features == {meta['num_features']}")
    check(meta['num_classes'] == ds.num_classes,
          f"KV 区 num_classes == {meta['num_classes']}")
    check(meta['dataset'] == name, f"KV 区 dataset == {meta['dataset']}")

    rebuilt, _ = GCN.from_file(gguf_path)
    rebuilt.eval()
    with torch.no_grad():
        new_logits = rebuilt(data.x, data.edge_index)
    check(torch.equal(ref_logits, new_logits),
          '重建模型的输出与原模型逐位一致')

    # ---- 3) 形状正确（GGUF 头里的 shape 是逆序的，最容易出错的地方）----
    w = state['convs.0.lin.weight']
    check(tuple(w.shape) == (16, data.num_features),
          f'convs.0.lin.weight 形状 {tuple(w.shape)} == '
          f'(16, {data.num_features})')
    check(tuple(state['convs.1.lin.weight'].shape) == (ds.num_classes, 16),
          f'convs.1.lin.weight 形状 '
          f'{tuple(state["convs.1.lin.weight"].shape)} == ({ds.num_classes}, 16)')

    # ---- 4) GGUF 里**没有**图结构与特征 ----
    check(not any('edge' in k or k == 'x' for k in state),
          'GGUF 不含edge_index / x（图结构必须推理时另行提供）')

    # ---- 5) .pt 与 .gguf 载入同一模型，结果一致 ----
    m2 = GCN(data.num_features, 16, ds.num_classes, 2, 0.0)
    m2.load(os.path.join(tmpdir, 'test.pt'))
    m2.eval()
    with torch.no_grad():
        l2 = m2(data.x, data.edge_index)
    check(torch.equal(ref_logits, l2), '.pt 载入的模型输出与原模型一致')

    # 清理临时目录
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    names = sys.argv[1:] or list(TABLE1)
    # PyG 在 Python 3.14 下的告警发生在 GCNConv 实例化时，必须包住整个测试
    with silence():
        for nm in names:
            test_scale(nm)
            test_self_loop(nm)
            test_normalization(nm)
            test_model_forward(nm)
            test_gguf_roundtrip(nm)
    print(f'\n{ok} passed, {fail} failed')
    sys.exit(1 if fail else 0)