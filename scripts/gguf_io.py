"""PyG 版 GCN 的 GGUF 权重读写。

与 ``export_model.py``（手写版）的区别
--------------------------------------
手写版的层参数叫 ``layers.0.weights.0``，形状是 ``(in, out)``；
PyG 的 ``GCNConv`` 参数叫 ``convs.0.lin.weight``，形状是 ``(out, in)``，
还多一个 ``convs.0.bias``。两者键名与布局都不同，因此**不能复用**
``export_model.py`` 的读写函数，这里单独实现一套。

GGUF 存什么、不存什么
---------------------
| 内容 | 是否存入 GGUF |
|---|---|
| 各层权重张量（``lin.weight`` / ``bias``） | ✅ 存在张量区 |
| 层数、维度、测试精度等元信息 | ✅ 存在 KV 区 |
| **图结构（edge_index）与特征 x** | ❌ **不存** |

最后一条是关键：GGUF 是 llama.cpp 的模型格式，**没有「图」的概念**。
推理时必须重新提供 ``edge_index`` 和 ``x``，否则无从计算。
这一点写进了 KV 区的 ``gnn.note``，任何解析器都能读到。

张量命名
--------
用 ``convs.{i}.lin.weight`` / ``convs.{i}.bias``——与 PyTorch 的
``state_dict`` 键名完全一致。这样加载时可以直接 ``load_state_dict``，
不需要任何键名映射表，也不会因为「转置了没有」而搞错方向。
"""

import json
import os

import numpy as np
import torch

# 张量名前缀，与 PyTorch state_dict 键名保持一致
TENSOR_PREFIX = 'convs'


def _gguf_value_type(val, GGUFValueType):
    """按 Python 值类型推断 GGUFValueType。"""
    if isinstance(val, bool):
        return GGUFValueType.BOOL
    if isinstance(val, int):
        return GGUFValueType.UINT32 if val >= 0 else GGUFValueType.INT32
    if isinstance(val, float):
        return GGUFValueType.FLOAT32
    return GGUFValueType.STRING


def _state_dict_to_tensors(model):
    """从模型的 state_dict 取出 (名称, numpy 数组) 列表。

    只取浮点张量（权重与偏置）。``GCNConv`` 还有 ``cached`` 之类的
    运行时缓存（``cached=False`` 时不存在），不是权重，不应存入。
    """
    out = []
    for name, tensor in model.state_dict().items():
        if not tensor.is_floating_point():
            continue
        out.append((name, tensor.detach().cpu().numpy().astype(np.float32)))
    return out


def save_gguf(path, model, meta=None):
    """把模型权重写入 GGUF 文件。

    :param path: 输出路径（``.gguf``）
    :param model: ``torch.nn.Module``
    :param meta: 额外元信息 dict，会写进 KV 区与同名 ``.meta.json``
    :return: 写入的张量列表 ``[(名称, 形状), ...]``
    """
    from gguf import GGUFWriter, GGUFValueType

    meta = dict(meta or {})
    # 从模型自身推出结构信息，保证元信息与权重永远自洽
    convs = model.convs
    meta.setdefault('num_layers', len(convs))
    meta.setdefault('num_features', int(convs[0].lin.weight.shape[1]))
    meta.setdefault('hidden_dim', int(convs[0].lin.weight.shape[0]))
    meta.setdefault('num_classes', int(convs[-1].lin.weight.shape[0]))
    meta.setdefault('model_type', 'gcn_pyg')

    writer = GGUFWriter(path, arch='gcn')

    def kv(key, val):
        writer.add_key_value(key, val, _gguf_value_type(val, GGUFValueType))

    # ---- 通用元信息 ----
    # 注意：GGUFWriter(path, arch='gcn') 已自动写入 general.architecture，
    # 此处不要重复写，否则 gguf 会发出 "Duplicated key name" 警告。
    kv('general.name', str(meta.get('name', 'gcn')))
    kv('general.description',
       'Graph Convolutional Network (Kipf & Welling 2017) '
       'reproduced in PyTorch / PyTorch Geometric')
    kv('general.dataset', str(meta.get('dataset', 'unknown')))
    kv('general.model_type', str(meta['model_type']))

    # ---- 架构元信息 ----
    kv('gnn.num_layers', int(meta['num_layers']))
    kv('gnn.num_features', int(meta['num_features']))
    kv('gnn.hidden_dim', int(meta['hidden_dim']))
    kv('gnn.num_classes', int(meta['num_classes']))
    if 'test_acc' in meta:
        kv('gnn.test_accuracy', float(meta['test_acc']))

    kv('gnn.weight_layout',
       'torch_pyg (out, in) for lin.weight; NOT transposed')
    kv('gnn.tensor_naming',
       'identical to torch state_dict keys, so load_state_dict works directly')
    kv('gnn.note',
       'Weights only. Graph structure (edge_index) and node features (x) '
       'are NOT embedded - they must be supplied at inference time. '
       'For real inference use the sibling .pt file.')

    # ---- 权重张量 ----
    tensors = _state_dict_to_tensors(model)
    for name, arr in tensors:
        writer.add_tensor(name, arr)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    # ---- 旁边放一份 .meta.json，方便人直接看（GGUF 的 KV 区不便于快速浏览）----
    meta_path = os.path.splitext(path)[0] + '.meta.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return [(n, a.shape) for n, a in tensors], meta_path


def save_pt(path, model, meta=None):
    """保存 PyTorch 原生 checkpoint（**实际推理请用这个**）。

    GGUF 容器能存权重，但无法表达「图结构必须随模型一起提供」这件事，
    ``.pt`` 则可以把元信息与权重完整打包。
    """
    meta = dict(meta or {})
    ckpt = {
        'state_dict': model.state_dict(),
        'num_layers': len(model.convs),
        'num_features': int(model.convs[0].lin.weight.shape[1]),
        'hidden_dim': int(model.convs[0].lin.weight.shape[0]),
        'num_classes': int(model.convs[-1].lin.weight.shape[0]),
        'meta': meta,
    }
    if 'test_acc' in meta:
        ckpt['test_acc'] = float(meta['test_acc'])
    torch.save(ckpt, path)
    return path


def _read_kv(reader):
    """读出 GGUF 的全部 KV 字段，合成一个 dict。

    gguf 0.19 的 ``GGUFReader`` 没有 ``keys()`` 方法，字段名在
    ``reader.fields``（OrderedDict）里。注意 ``fields`` 里也含
    ``GGUF.version`` 这类保留键，调用方按需过滤。
    """
    out = {}
    for key in reader.fields:
        f = reader.get_field(key)
        if f is None:
            continue
        v = f.contents
        out[key] = v() if callable(v) else v
    return out


def load_gguf(path, model=None):
    """从 GGUF 读回权重。

    :param path: ``.gguf`` 路径
    :param model: 若给出，则把权重载入该模型并返回它；
                  若为None，则只返回 ``(state_dict, meta)``
    :return: ``model`` 或 ``(state_dict, meta)``

    **GGUF 张量自带形状**（``tensor.shape``），所以即使不传 model，
    也能纯从文件重建出模型结构——这是本实现比旧 ``infer.py``
    （硬编码层数与维度）更健壮的地方。
    """
    from gguf import GGUFReader

    reader = GGUFReader(path)
    kv = _read_kv(reader)

    state = {}
    shapes = {}
    for t in reader.tensors:
        arr = np.asarray(t.data)
        # **不要用 t.shape 做 reshape。**
        #
        # GGUF 头里声明的 shape 是**逆序**的（GGML 约定 ne[0] 是最快维）：
        # 写入 numpy (2,3) 时声明为 [3,2]。所以 reshape(t.shape) 会把形状弄反。
        #
        # 而 ``t.data`` 本身（memmap）的形状**已经是正确的 numpy 形状**，
        # 元素顺序也与原始数组完全一致——直接用即可，不需要任何 reshape。
        state[t.name] = torch.from_numpy(arr.copy()).float()
        shapes[t.name] = tuple(arr.shape)

    meta = {
        'num_layers': int(kv.get('gnn.num_layers', len(model.convs)
                                  if model else 0)),
        'num_features': int(kv.get('gnn.num_features', 0)),
        'hidden_dim': int(kv.get('gnn.hidden_dim', 0)),
        'num_classes': int(kv.get('gnn.num_classes', 0)),
        'dataset': kv.get('general.dataset', 'unknown'),
        'test_acc': kv.get('gnn.test_accuracy'),
        'tensor_shapes': shapes,
    }

    if model is None:
        return state, meta

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise ValueError('GGUF 缺少权重: %s' % missing)
    if unexpected:
        raise ValueError('GGUF 含有模型不认识的张量: %s' % unexpected)
    return model


def load_pt(path, model=None):
    """从 ``.pt`` 读回权重，语义与 :func:`load_gguf` 一致。"""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    state = ckpt['state_dict']
    meta = {k: ckpt.get(k) for k in
            ('num_layers', 'num_features', 'hidden_dim', 'num_classes',
             'test_acc')}
    meta['dataset'] = ckpt.get('meta', {}).get('dataset', 'unknown')
    if model is None:
        return state, meta
    model.load_state_dict(state)
    return model


def load_any(path, model=None):
    """按扩展名自动选择 ``.gguf`` 或 ``.pt`` 载入。"""
    if path.endswith('.gguf'):
        return load_gguf(path, model)
    if path.endswith('.pt'):
        return load_pt(path, model)
    raise ValueError('不认识的模型格式: %s（支持 .gguf / .pt）' % path)
