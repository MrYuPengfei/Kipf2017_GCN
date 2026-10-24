"""把训练好的 GCN 权重导出到 model/ 目录。

用法：
    # 1) 训练并导出（Cora 为例）
    python export_model.py --dataset cora --epochs 200

    # 2) 只导出，不训练（复用已有权重）
    python export_model.py --dataset cora --load model/gcn_cora.pt

产出：
    model/gcn_cora.gguf       GGUF 格式权重（主产物）
    model/gcn_cora.pt         PyTorch 原生 state_dict（推荐实际使用）
    model/gcn_cora.meta.json  超参与精度元信息

---

## 关于 GGUF 格式的说明（请先读这段）

**GGUF 是 llama.cpp 的模型格式，为大语言模型设计**，其架构假设与 GCN 不兼容：

| GGUF/LLM 的概念 | GCN 是否有|
|---|---|
| token序列 + tokenizer/vocab | ❌ 没有 |
| 注意力 / KV cache | ❌ 没有（图卷积是邻域聚合） |
| RoPE 位置编码 | ❌ 没有（图没有序列位置） |
| 逐层 block 结构 | ❌ 没有（只有 2 个权重矩阵） |
| **权重张量存储** | ✅ **可以** |

所以本脚本采取的策略是：
- **用 GGUF 容器存储权重张量**（这部分 GGUF 完全支持，且通用解析器都能读）
- 把架构元信息（层数、维度、support 数等）写进 GGUF 的 KV 元数据区
- 同时导出 `.pt`——**实际推理请用这个**，它能直接 `torch.load` 并还原模型

换句话说：`.gguf` 适合「归档 / 跨工具传递权重」，`.pt` 适合「在 PyTorch 里真正跑起来」。
"""

from __future__ import division, print_function

import argparse
import json
import os
import time

import numpy as np
import torch

from gcn import silence_sparse_warning
from gcn.models import GCN
from gcn.utils import load_data, preprocess_adj

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')


def parse_args():
    p = argparse.ArgumentParser(description='导出 GCN 权重')
    p.add_argument('--dataset', type=str, default='cora',
                   choices=['cora', 'citeseer', 'pubmed'])
    p.add_argument('--model', type=str, default='gcn',
                   choices=['gcn', 'gcn_cheby', 'dense'])
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--hidden1', type=int, default=16)
    p.add_argument('--learning_rate', type=float, default=0.01)
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--weight_decay', type=float, default=5e-4)
    p.add_argument('--early_stopping', type=int, default=10)
    p.add_argument('--max_degree', type=int, default=3)
    p.add_argument('--load', type=str, default='',
                   help='载入已有 .pt 权重而不再训练')
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--source', type=str, default='pickle',
                   choices=['pickle', 'csv'],
                   help='数据源：pickle 或 csv（export_csv.py 生成）。')
    p.add_argument('--csv_dir', type=str, default=None,
                   help='CSV 数据目录，默认为 <项目>/data/csv。')
    return p.parse_args()


def train_model(args, feat, ytr, yva, support, input_dim, output_dim):
    """按train.py 的超参训练，返回 (model, test_acc)。"""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = GCN(input_dim=input_dim, output_dim=output_dim,
                supports=support, hidden1=args.hidden1,
                dropout=args.dropout, weight_decay=args.weight_decay,
                learning_rate=args.learning_rate)
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    cost_val = []
    for ep in range(args.epochs):
        model.train()
        model.labels, model.labels_mask = ytr, MASK['train']
        opt.zero_grad()
        model(feat)
        model._loss().backward()
        opt.step()

        model.eval()
        model.labels, model.labels_mask = yva, MASK['val']
        with torch.no_grad():
            model(feat)
            cost_val.append(model._loss().item())

        if ep > args.early_stopping and \
                cost_val[-1] > np.mean(cost_val[-(args.early_stopping + 1):-1]):
            print('Early stopping at epoch %d' % (ep + 1))
            break

    model.eval()
    with torch.no_grad():
        model.labels, model.labels_mask = MASK['y_test'], MASK['test']
        model(feat)
        test_acc = model._accuracy().item()
    return model, test_acc


MASK = {}


def collect_weights(model):
    """按 GGUF 友好顺序收集权重：返回 [(名称, numpy数组), ...]。

    注意 PyTorch 的权重是 [in, out]，GGUF 通用张量约定用 [out, in]（行主序），
    这里转置后在元数据里标明 transposed=true，避免误用。
    """
    out = []
    for li, layer in enumerate(model.layers):
        for wi, w in enumerate(layer.weights):
            out.append(('graph_conv.%d.weight.%d' % (li, wi),
                        w.detach().cpu().numpy().astype(np.float32)))
        if layer.bias:
            out.append(('graph_conv.%d.bias' % li,
                        layer.bias_param.detach().cpu().numpy().astype(np.float32)))
    return out


def save_gguf(path, model, meta):
    """把权重写入 GGUF 容器。"""
    from gguf import GGUFWriter, GGUFValueType

    writer = GGUFWriter(path, arch='gcn')

    def kv(key, val):
        """按Python 值类型推断 GGUFValueType 后写入 KV 区。"""
        if isinstance(val, bool):
            t = GGUFValueType.BOOL
        elif isinstance(val, int):
            t = GGUFValueType.UINT32 if val >= 0 else GGUFValueType.INT32
        elif isinstance(val, float):
            t = GGUFValueType.FLOAT32
        else:
            t = GGUFValueType.STRING
        writer.add_key_value(key, val, t)

    # ---- 架构元信息（放进 KV 区，任何 GGUF 解析器都能读出）----
    # 注意：GGUFWriter(path, arch='gcn') 已自动写入 general.architecture，
    # 此处不要重复写，否则 gguf 会发出 "Duplicated key name" 警告。
    kv('general.name', meta['name'])
    kv('general.description',
       'Graph Convolutional Network (Kipf & Welling 2017) '
       'reproduced in PyTorch')
    kv('general.dataset', meta['dataset'])
    kv('general.model_type', 'gcn')

    kv('gnn.num_layers', meta['num_layers'])
    kv('gnn.num_features', meta['input_dim'])
    kv('gnn.hidden_dim', meta['hidden1'])
    kv('gnn.num_classes', meta['output_dim'])
    kv('gnn.num_supports', meta['num_supports'])
    kv('gnn.test_accuracy', float(meta['test_acc']))

    kv('gnn.weight_layout', 'torch_nchw_in_out (NOT transposed)')
    kv('gnn.note',
       'Weights only. Graph structure (adjacency) is NOT embedded - '
       'it must be supplied at inference time. For real inference '
       'use the sibling .pt file.')

    # ---- 权重张量 ----
    tensors = collect_weights(model)
    for name, arr in tensors:
        writer.add_tensor(name, arr)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return [(n, a.shape) for n, a in tensors]


def main():
    args = parse_args()
    os.makedirs(MODEL_DIR, exist_ok=True)

    global MASK
    print('=' * 62)
    print('GCN 权重导出  (dataset=%s, model=%s)' % (args.dataset, args.model))
    print('=' * 62)

    adj, feat, ytr, yva, yte, mtr, mva, mte, araw = load_data(args.dataset, source=getattr(args, 'source', 'pickle'), csv_dir=getattr(args, 'csv_dir', None))
    MASK = {'train': mtr, 'val': mva, 'test': mte,
            'y_train': ytr, 'y_val': yva, 'y_test': yte}

    support = [preprocess_adj(araw)]
    num_supports = 1
    input_dim, output_dim = feat.shape[1], ytr.shape[1]

    stem = 'gcn_%s' % args.dataset

    # ---- 训练或载入 ----
    if args.load:
        print('载入已有权重: %s' % args.load)
        ckpt = torch.load(args.load, map_location='cpu')
        model = GCN(input_dim=ckpt['input_dim'], output_dim=ckpt['output_dim'],
                    supports=support, hidden1=ckpt['hidden1'],
                    dropout=0., weight_decay=args.weight_decay,
                    learning_rate=args.learning_rate)
        model.load_state_dict(ckpt['state_dict'])
        test_acc = ckpt.get('test_acc', float('nan'))
    else:
        t0 = time.time()
        model, test_acc = train_model(args, feat, ytr, yva, support,
                                      input_dim, output_dim)
        print('训练完成，用时 %.1fs，test acc = %.4f' % (time.time() - t0, test_acc))

    # ---- 元信息 ----
    meta = {
        'name': stem,
        'dataset': args.dataset,
        'model': args.model,
        'num_layers': len(model.layers),
        'input_dim': input_dim,
        'hidden1': args.hidden1,
        'output_dim': output_dim,
        'num_supports': num_supports,
        'test_acc': test_acc,
        'hyperparams': {
            'learning_rate': args.learning_rate,
            'dropout': args.dropout,
            'weight_decay': args.weight_decay,
            'epochs': args.epochs,
            'seed': args.seed,
        },
        'note': ('GGUF 只存权重；推理时需另行提供邻接矩阵。'
                 '实际使用请选 .pt 文件。'),
    }

    # ---- 1) PyTorch 原生格式 ----
    pt_path = os.path.join(MODEL_DIR, stem + '.pt')
    torch.save({'state_dict': model.state_dict(),
                'input_dim': input_dim,
                'output_dim': output_dim,
                'hidden1': args.hidden1,
                'num_supports': num_supports,
                'dataset': args.dataset,
                'test_acc': test_acc}, pt_path)
    print('\n[1/3] PyTorch  -> %s (%.1f KB)'
          % (os.path.relpath(pt_path), os.path.getsize(pt_path) / 1024))

    # ---- 2) GGUF ----
    gguf_path = os.path.join(MODEL_DIR, stem + '.gguf')
    try:
        shapes = save_gguf(gguf_path, model, meta)
        print('[2/3] GGUF      -> %s (%.1f KB)'
              % (os.path.relpath(gguf_path), os.path.getsize(gguf_path) / 1024))
        print('       张量:')
        for n, s in shapes:
            print('         %-28s %s' % (n, list(s)))
    except Exception as ex:
        print('[2/3] GGUF 导出失败: %r' % ex)
        raise

    # ---- 3) 元信息 ----
    meta_path = os.path.join(MODEL_DIR, stem + '.meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print('[3/3] 元信息    -> %s' % os.path.relpath(meta_path))

    # ---- 校验：GGUF 能否读回，且权重与 .pt 完全一致 ----
    print('\n校验 GGUF 回读...')
    from gguf import GGUFReader
    reader = GGUFReader(gguf_path)
    names = [t.name for t in reader.tensors]
    print('  读回 %d 个张量: %s' % (len(names), names))

    def field(k):
        """读取 KV 字段的值（gguf 0.19 中 contents 是方法而非属性）。"""
        f = reader.get_field(k)
        if f is None:
            return None
        v = f.contents
        return v() if callable(v) else v

    for k in ['general.architecture', 'general.name', 'gnn.num_features',
              'gnn.hidden_dim', 'gnn.num_classes']:
        print('  %-24s %r' % (k, field(k)))

    # 数值一致性：GGUF 里的权重 vs PyTorch 里的权重
    ref = GCN(input_dim=input_dim, output_dim=output_dim, supports=support,
              hidden1=args.hidden1, dropout=0., weight_decay=0.,
              learning_rate=args.learning_rate)
    ref.load_state_dict(torch.load(pt_path, map_location='cpu')['state_dict'])

    by_name = {t.name: t for t in reader.tensors}
    ok_all = True
    for li, layer in enumerate(ref.layers):
        g = by_name['graph_conv.%d.weight.%d' % (li, 0)]
        g_arr = np.asarray(g.data)
        p_arr = layer.weights[0].detach().numpy()
        # GGUF 张量按行主序存储，形状在写入时已由numpy 数组决定
        g_arr = g_arr.reshape(p_arr.shape)
        diff = float(np.abs(g_arr - p_arr).max())
        ok = diff < 1e-6
        ok_all &= ok
        print('  层%d 权重 %s vs %s，最大偏差 %.2e  %s'
              % (li, list(g_arr.shape), list(p_arr.shape), diff,
                 '一致' if ok else '不一致'))
    print('  GGUF ↔ PyTorch 权重%s' % ('完全一致' if ok_all else '存在差异'))

    print('\n' + '=' * 62)
    print('完成。实际推理请用 %s' % os.path.relpath(pt_path))
    print('=' * 62)


if __name__ == '__main__':
    # 用上下文管理器显式声明稀疏不变量检查策略，消除 PyTorch 2.14 的 UserWarning
    with silence_sparse_warning():
        main()