"""PyTorch Geometric（PyG）版 GCN 的辅助模块。

本包与手写实现（``gcn/``）**并存**，用于横向对照，不替代手写实现。

设计要点
--------
**不调用 ``torch_geometric.datasets.Planetoid``。** 原因有二：

1. Planetoid 会联网下载数据，而本项目的数据已在 ``gcn/data/`` 与 ``data/csv/`` 中；
2. 更关键的是**划分一致性**——自带的 loader 与手写实现共用同一套 public split，
   才能保证两条路径的精度可以直接对比。若让 PyG 自己下数据、自己切分，
   数值差异就分不清是「实现差异」还是「划分差异」。

因此本包的数据入口是 :func:`to_pyg_data`：调用 ``gcn.utils.load_data``，
再把结果重排成 PyG 的 ``Data`` 对象。

关于告警抑制
------------
PyG 在 ``MessagePassing`` 上用 ``torch.jit.script`` 做类型检查，在 Python 3.14
下会触发 ``typing._eval_type`` 的 DeprecationWarning（PEP 695 相关）。
这是 PyG 与新版 Python 的兼容问题，本项目无法修改，故用
:func:`silence_pyg_warning` 精确抑制——只屏蔽这一条消息，不影响其他告警。
"""

import contextlib
import warnings

import numpy as np
import scipy.sparse as sp
import torch

from gcn import silence_sparse_warning
from gcn.utils import load_data


@contextlib.contextmanager
def silence_pyg_warning():
    """抑制 PyG 在 Python 3.14 下的 typing 弃用告警。

    只匹配一条具体消息（``typing._eval_type`` 的 ``type_params`` 参数），
    不做宽泛的 ``filterwarnings('ignore')``——否则会连带屏蔽掉真正需要
    关注的告警，违背本项目「零警告=真零警告」的原则。
    """
    with warnings.catch_warnings():
        _install_filters()
        yield


def _install_filters():
    """安装两条精确的告警过滤（供上下文管理器与导入期共用）。"""
    warnings.filterwarnings(
        'ignore',
        message=r'.*Failing to pass a value to the .type_params. parameter.*')
    warnings.filterwarnings(
        'ignore',
        message=r'.*torch\.jit\.script. is not supported in Python 3\.14.*')


# ---- 导入期就必须安装，不能只靠上下文管理器 ----
# 原因：`torch_geometric.nn` 在**被导入的那一刻**就调用 torch.jit.script
# 做类型检查，FutureWarning 在 models.py 的 import 语句执行时就已发出。
# 而 `__main__` 里的上下文管理器要等 import 全部完成才生效，来不及拦截。
# 这里按「精确消息 + 精确模块」安装，作用域可控。
warnings.filterwarnings(
    'ignore',
    message=r'.*torch\.jit\.script. is not supported in Python 3\.14.*',
    category=FutureWarning,
    module=r'torch\.jit\._script')


def to_pyg_data(dataset_str, source='pickle', csv_dir=None):
    """把数据集转成 PyG 的 ``Data`` 对象。

    与 ``load_data`` 共用同一套划分，因此与手写实现的精度可直接比较。

    :param dataset_str: 'cora' / 'citeseer' / 'pubmed'
    :param source: 'pickle' 或 'csv'
    :param csv_dir: CSV 目录，默认 ``<项目>/data/csv``
    :return: ``torch_geometric.data.Data``，字段见下表

    ============  ==========================  ====================================
    字段          形状                        说明
    ============  ==========================  ====================================
    ``x``         (N, F) float32              行归一化后的节点特征（稠密）
    ``edge_index`` (2, 2E) int64              无向图双向各一条，**不含自环**
    ``y``         (N,) int64                  类别下标；无标签节点填 0 并靠 mask 屏蔽
    ``train_mask`` (N,) bool                  训练集
    ``val_mask``  (N,) bool                   验证集
    ``test_mask`` (N,) bool                   测试集
    ============  ==========================  ====================================

    **为什么 edge_index 不含自环**：``GCNConv`` 默认 ``add_self_loops=True``，
    会自行加上并重新归一化，行为与论文的 ``Ã = A + I`` 一致。
    若这里再加一遍，会变成 ``A + 2I``，权重失真（这正是初学者最常见的错误）。
    """
    with silence_sparse_warning():
        # 手写 loader 返回 9 元组，前 8 项为 torch 张量，第 9 项是 scipy 邻接矩阵
        (adj, features, y_train, y_val, y_test,
         train_mask, val_mask, test_mask, adj_raw) = load_data(
            dataset_str, source=source, csv_dir=csv_dir)

    # ---- 特征：稀疏 -> 稠密 ----
    # PyG 的 GCNConv 需要稠密 x（稀疏输入需走 SparseTensor 路径，语义不同）
    x = features.to_dense().to(torch.float32)

    # ---- 邻接矩阵 -> edge_index ----
    # adj_raw 是 scipy COO，且已是无向对称（双向各存一条边）
    coo = sp.coo_matrix(adj_raw)
    edge_index = torch.from_numpy(
        np.vstack((coo.row, coo.col)).astype(np.int64))

    # ---- 标签：one-hot (N,C) -> 类别下标 (N,) ----
    # y_train / y_val / y_test 是零填充的 one-hot，只有对应 mask 的行非零。
    # 这里合并成单一 y：被任一 mask 覆盖的行取其 argmax，其余填 0（靠 mask 屏蔽）。
    covered = train_mask | val_mask | test_mask
    labels = torch.zeros(x.shape[0], dtype=torch.long)
    for y_part in (y_train, y_val, y_test):
        idx = y_part.sum(dim=1) > 0            # 该部分有标签的行
        labels[idx] = y_part[idx].argmax(dim=1)

    # 一致性检查：覆盖到的行必须有非零标签，否则说明 mask 与标签错位
    if covered.any():
        assert (labels[covered] >= 0).all()

    from torch_geometric.data import Data   # 延迟导入，避免无 PyG 时的导入错误

    return Data(x=x,
                edge_index=edge_index,
                y=labels,
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask,
                num_classes=int(y_train.shape[1]))


def describe(data):
    """返回一行数据摘要，便于日志与测试断言。"""
    n, f = data.x.shape
    return ('nodes=%d features=%d edges=%d(directed) classes=%d '
            'train/val/test=%d/%d/%d' % (
                n, f, data.edge_index.shape[1], data.num_classes,
                int(data.train_mask.sum()), int(data.val_mask.sum()),
                int(data.test_mask.sum())))
