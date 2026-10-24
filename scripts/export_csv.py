"""把gcn/data 下的引文网络数据集导出为 CSV。

用法：
    python export_csv.py                        # 导出 cora/citeseer/pubmed
    python export_csv.py --dataset cora         # 只导出 cora
    python export_csv.py --outdir data/csv      # 自定义输出目录
    python export_csv.py --dense                # 特征写成稠密宽表（不推荐）

输出布局（每个数据集一个子目录）
--------------------------------
    data/csv/cora/nodes.csv      节点表：node_id,split,label
    data/csv/cora/features.csv   稀疏特征长表：node_id,feature_id,value
    data/csv/cora/edges.csv      有向边表：source,target,weight
    data/cora_meta.csv? 否 —— 元信息统一写在 meta.json

    data/csv/cora/meta.json      维度/类别数/划分统计

为什么用稀疏长表而不用稠密宽表
------------------------------
Pubmed 有 19717 x 500 = 986万格，Cora 有 2708 x 1433 = 388万格。稠密 CSV
体积大、读写慢，且绝大部分是 0。稀疏长表（只存非零元组）与原始 scipy
矩阵一一对应，可无损还原，`load_data_csv()` 与 `load_data()` 数值完全一致。
"""

from __future__ import division, print_function

import argparse
import json
import os
import pickle as pkl
import sys
import warnings

import networkx as nx
import numpy as np
import scipy.sparse as sp

from gcn import silence_sparse_warning

DEFAULT_DATASETS = ('cora', 'citeseer', 'pubmed')

# 默认输出目录：本脚本同级下的 data/csv
DEFAULT_OUTDIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'data', 'csv')


def parse_args():
    parser = argparse.ArgumentParser(
        description='将 GCN 数据集从 pickle 导出为 CSV')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=list(DEFAULT_DATASETS) + ['all'],
                        help='要导出的数据集，all 表示三个都导出。')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='pickle 数据目录，默认为 gcn/data。')
    parser.add_argument('--outdir', type=str, default=DEFAULT_OUTDIR,
                        help='CSV 输出目录，默认为 data/csv。')
    parser.add_argument('--dense', action='store_true',
                        help='把节点特征写成稠密宽表（默认写稀疏长表）。')
    return parser.parse_args()


def _load_objects(data_dir, dataset_str):
    """读取并规范化 7 个 pickle 对象（旧 pickle 会触发 scipy 弃用告警）。"""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for name in names:
        path = os.path.join(data_dir, 'ind.{}.{}'.format(dataset_str, name))
        with open(path, 'rb') as f:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=DeprecationWarning)
                warnings.filterwarnings(
                    'ignore',
                    message='.*scipy\\.sparse\\..*namespace is deprecated.*')
                obj = pkl.load(f, encoding='latin1') \
                    if sys.version_info > (3, 0) else pkl.load(f)
        objects.append(_to_modern_sparse(obj))
    return tuple(objects)


def _to_modern_sparse(obj):
    """把旧命名空间的稀疏矩阵换成新版 scipy 类型（消警+ 未来兼容）。"""
    if sp.issparse(obj):
        fmt = obj.format
        new_cls = getattr(sp, fmt + '_matrix', None)
        if new_cls is not None and type(obj) is not new_cls:
            return new_cls(obj)
        return obj
    return obj


def parse_index_file(filename):
    index = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line:
                index.append(int(line))
    return index


def build_raw_dataset(data_dir, dataset_str):
    """按与 ``gcn.utils.load_data`` 完全一致的顺序拼装原始 numpy 结构。

    返回 (features_lil, labels_ndarray, adj_scipy, splits)
      features : 全部节点的原始特征（未归一化）
      labels   : 全部节点的 one-hot 标签
      adj      : scipy 邻接矩阵（CSR，无自环）
      splits   : 长度 N 的 str 数组，取值 train/val/test/other
    """
    x, y, tx, ty, allx, ally, graph = _load_objects(data_dir, dataset_str)
    test_idx_reorder = parse_index_file(
        os.path.join(data_dir, 'ind.{}.test.index'.format(dataset_str)))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Citeseer 存在孤立节点，测试区间需要补零行，与原版一致
        test_idx_range_full = range(min(test_idx_reorder),
                                    max(test_idx_reorder) + 1)
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

    n = labels.shape[0]
    splits = np.array(['other'] * n, dtype=object)
    splits[:len(y)] = 'train'
    splits[len(y):len(y) + 500] = 'val'
    splits[test_idx_range] = 'test'      # test 覆盖优先，保持与原 mask 一致

    return features, labels, sp.csr_matrix(adj), splits


def export_dataset(dataset_str, data_dir, outdir, dense=False):
    """导出单个数据集，返回 meta 字典。"""
    features, labels, adj, splits = build_raw_dataset(data_dir, dataset_str)
    n_nodes = labels.shape[0]

    ds_dir = os.path.join(outdir, dataset_str)
    os.makedirs(ds_dir, exist_ok=True)

    # ---- nodes.csv：node_id, split, label ----
    # label 为 one-hot 的 argmax；全零行（无标签节点）记为 -1
    has_label = labels.sum(1) > 0               # 标签行全 0 => 该节点无标签
    label_idx = np.where(has_label, labels.argmax(1), -1)   # 无标签记 -1
    nodes_path = os.path.join(ds_dir, 'nodes.csv')
    with open(nodes_path, 'w', newline='', encoding='utf-8') as f:
        # newline='' 防止 Windows 下额外插入 \r\n 造成空行
        f.write('node_id,split,label\n')        # 表头固定，读取端按列名取
        for i in range(n_nodes):               # 逐节点写一行
            f.write('{},{},{}\n'.format(i, splits[i], int(label_idx[i])))

    # ---- features.csv ----
    if dense:
        # 稠密宽表：每行 F 列。直观但体积巨大（Pubmed 986万格），仅供调试
        feats_path = os.path.join(ds_dir, 'features_dense.csv')
        with open(feats_path, 'w', newline='', encoding='utf-8') as f:
            f.write('node_id,' +
                    ','.join('f{}'.format(j)      # 列名 f0, f1, ..., f_{F-1}
                             for j in range(features.shape[1])) + '\n')
            for i in range(n_nodes):           # 逐行 densify
                row = features.getrow(i).toarray()[0]   # (1,F) -> (F,)
                f.write(str(i) + ',' +
                        ','.join('{:.6g}'.format(v) for v in row) + '\n')
    else:
        # 稀疏长表（默认）：每行一个非零元组，与 scipy 的 COO 一一对应
        coo = sp.coo_matrix(features)           # 转COO 以便遍历非零元组
        feats_path = os.path.join(ds_dir, 'features.csv')
        with open(feats_path, 'w', newline='', encoding='utf-8') as f:
            f.write('node_id,feature_id,value\n')
            for r, c, v in zip(coo.row, coo.col, coo.data):
                # %.17g 保证 float64 双精度往返无损，从而与 pickle 路径逐位一致
                f.write('{},{},{:.17g}\n'.format(r, c, v))

    # ---- edges.csv：邻接矩阵非零元组（对称，无向图双向各存一行）----
    acoo = sp.coo_matrix(adj)                   # 邻接矩阵转 COO
    edges_path = os.path.join(ds_dir, 'edges.csv')
    with open(edges_path, 'w', newline='', encoding='utf-8') as f:
        f.write('source,target,weight\n')       # 有向边表，每条边一行
        for r, c, v in zip(acoo.row, acoo.col, acoo.data):
            f.write('{},{},{:.17g}\n'.format(r, c, v))   # 权重恒为 1

    # ---- meta.json：维度信息，供读取端免去形状反推 ----
    meta = {
        'dataset': dataset_str,
        'num_nodes': int(n_nodes),
        'num_features': int(features.shape[1]),
        'num_classes': int(labels.shape[1]),
        'num_edges': int(acoo.nnz),                          # 有向边计数（对称图是 2×无向边数）
        'num_nonzero_features': int(sp.coo_matrix(features).nnz),
        'features_format': 'dense' if dense else 'sparse',   # 告诉读取端走哪条解析分支
        'files': {                                            # 文件名记录，文件名可改
            'nodes': os.path.basename(nodes_path),
            'features': os.path.basename(feats_path),
            'edges': os.path.basename(edges_path),
        },
        'splits': {
            'train': int(np.sum(splits == 'train')),
            'val': int(np.sum(splits == 'val')),
            'test': int(np.sum(splits == 'test')),
            'other': int(np.sum(splits == 'other')),
        },
    }
    with open(os.path.join(ds_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return meta


def main():
    FLAGS = parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = FLAGS.data_dir or os.path.join(here, 'gcn', 'data')
    datasets = (DEFAULT_DATASETS if FLAGS.dataset == 'all'
                else (FLAGS.dataset,))
    os.makedirs(FLAGS.outdir, exist_ok=True)

    print('Source : {}'.format(data_dir))
    print('Output : {}'.format(os.path.abspath(FLAGS.outdir)))
    for ds in datasets:
        meta = export_dataset(ds, data_dir, FLAGS.outdir, FLAGS.dense)
        print('[{}] nodes={} features={} classes={} edges={} | train/val/test/other='
              '{}/{}/{}/{}'.format(
                  meta['dataset'], meta['num_nodes'], meta['num_features'],
                  meta['num_classes'], meta['num_edges'],
                  meta['splits']['train'], meta['splits']['val'],
                  meta['splits']['test'], meta['splits']['other']))


if __name__ == '__main__':
    with silence_sparse_warning():
        main()
