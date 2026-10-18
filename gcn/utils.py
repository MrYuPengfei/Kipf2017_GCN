"""PyTorch 版数据加载与预处理。

逐行对照 TensorFlow 原版 ``gcn/utils.py`` 移植，保留全部实现细节：
  - Citeseer 孤立节点修复
  - 测试集索引重排（test_idx_reorder）
  - 特征行归一化
  - 对称归一化 + 自环（renormalization trick）
  - Chebyshev 多项式递推

与原版的差异仅在于：
  - ``np.bool`` -> ``bool``（NumPy 1.24+ 已移除 np.bool）
  - 邻接矩阵用 ``torch.sparse_coo_tensor`` 直接给出，而非TF 的 sparse placeholder
  - 特征不再转成 tuple 表示，直接给稀疏张量
"""

import os
import pickle as pkl
import sys

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch

# 原版写作 ``from scipy.sparse.linalg.eigen.arpack import eigsh``，
# 但 SciPy 1.14+ 移除了 eigen 子包，这里用跨版本可用的导入路径。
try:                                        # SciPy >= 1.14
    from scipy.sparse.linalg import eigsh
except ImportError:                         # 兼容旧版
    from scipy.sparse.linalg.eigen.arpack import eigsh

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def parse_index_file(filename):
    """Parse index file."""
    index = []
    with open(filename) as f:
        for line in f:
            index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=bool)


def _to_torch_sparse(mx):
    """scipy稀疏矩阵 -> torch 稀疏张量（coalesce 以便高效 matmul）。"""
    mx = sp.coo_matrix(mx)
    indices = torch.from_numpy(np.vstack((mx.row, mx.col)).astype(np.int64))
    values = torch.from_numpy(mx.data.astype(np.float32))
    return torch.sparse_coo_tensor(indices, values, torch.Size(mx.shape)).coalesce()


def load_data(dataset_str, data_dir=DATA_DIR):
    """加载引文网络数据集。

    文件含义（与原版一致）：
        ind.{ds}.x        训练节点特征 (scipy.sparse.csr)
        ind.{ds}.y        训练节点 one-hot 标签 (np.ndarray)
        ind.{ds}.tx       测试节点特征 (scipy.sparse.csr)
        ind.{ds}.ty       测试节点 one-hot 标签 (np.ndarray)
        ind.{ds}.allx     全部有标签+无标签训练节点特征（x 的超集）
        ind.{ds}.ally     allx 对应的标签
        ind.{ds}.graph    {index: [邻居index,...]} 的 dict
        ind.{ds}.test.index  测试节点在图中的索引（inductive 设置用）

    :param dataset_str: 'cora' / 'citeseer' / 'pubmed'
    :return: adj_t, features, y_train, y_val, y_test,
             train_mask, val_mask, test_mask, adj_raw
             —— 前 8 项与 TF 原版一一对应（adj/features 已转为 torch 稀疏张量），
             最后一项 adj_raw 是 scipy 域的原始邻接矩阵，供 chebyshev_polynomials 使用。
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for name in names:
        path = os.path.join(data_dir, 'ind.{}.{}'.format(dataset_str, name))
        with open(path, 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        os.path.join(data_dir, 'ind.{}.test.index'.format(dataset_str)))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Citeseer 有孤立节点：用零向量补齐到正确位置
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    adj_raw = sp.coo_matrix(adj)     # scipy 域副本（Chebyshev 用）
    adj_t = _to_torch_sparse(adj)

    # ---- NumPy -> torch ----
    # 注意：原始（未加自环、未归一化）的 scipy 邻接矩阵单独返回，
    # 供 chebyshev_polynomials 使用——它内部要在 scipy 域做谱分解。
    features = _to_torch_sparse(preprocess_features(features))
    y_train = torch.from_numpy(y_train.astype(np.float32))
    y_val = torch.from_numpy(y_val.astype(np.float32))
    y_test = torch.from_numpy(y_test.astype(np.float32))
    train_mask = torch.from_numpy(train_mask)
    val_mask = torch.from_numpy(val_mask)
    test_mask = torch.from_numpy(test_mask)

    return (adj_t, features, y_train, y_val, y_test,
            train_mask, val_mask, test_mask, adj_raw)


def preprocess_features(features):
    """特征矩阵行归一化：每行除以该行之和（零行保持零向量）。"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(features)


def normalize_adj(adj):
    """对称归一化邻接矩阵 D^-1/2 A D^-1/2。"""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """简单 GCN 用的邻接矩阵预处理：加自环 + 对称归一化（renormalization trick）。

    输入 scipy 稀疏矩阵，输出 torch 稀疏张量。
    对应文档 §3.3：Ã = A + I，再做 D̃^-1/2 Ã D̃^-1/2。
    """
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return _to_torch_sparse(adj_normalized)


def chebyshev_polynomials(adj, k):
    """计算到K 阶的Chebyshev 多项式，返回 torch 稀疏张量列表。

    输入 **scipy 稀疏矩阵**（未加自环的原始邻接矩阵）。
    对应文档 §3.2：把谱滤镜用 Chebyshev 多项式近似，得到 K 跳局部性，
    且避免O(N^3) 的特征分解。
    """
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for _ in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return [_to_torch_sparse(m) for m in t_k]