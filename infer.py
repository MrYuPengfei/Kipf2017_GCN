"""载入导出的权重做推理，验证模型可用。

用法：
    # 用 .pt 推理
    python infer.py --model_path model/gcn_cora.pt --dataset cora

    # 用 .gguf 推理（演示 GGUF 权重可直接回读使用）
    python infer.py --model_path model/gcn_cora.gguf --dataset cora

注意：**推理时必须提供邻接矩阵**，它不在权重文件里——
这正是 GGUF 无法完整表达 GCN 的根本原因（详见 export_model.py 顶部说明）。
"""

from __future__ import division, print_function

import argparse

import numpy as np
import torch

from gcn import silence_sparse_warning
from gcn.models import GCN
from gcn.utils import load_data, preprocess_adj


def parse_args():
    p = argparse.ArgumentParser(description='GCN 推理')
    p.add_argument('--model_path', type=str, required=True,
                   help='.pt 或 .gguf 权重路径')
    p.add_argument('--dataset', type=str, default='cora',
                   choices=['cora', 'citeseer', 'pubmed'])
    p.add_argument('--hidden1', type=int, default=16)
    p.add_argument('--source', type=str, default='pickle',
                   choices=['pickle', 'csv'],
                   help='数据源：pickle 或 csv（export_csv.py 生成）。')
    p.add_argument('--csv_dir', type=str, default=None,
                   help='CSV 数据目录，默认为 <项目>/data/csv。')
    return p.parse_args()


def load_weights(path, input_dim, output_dim, hidden1, supports):
    """从 .pt 或 .gguf 载入权重，返回 state_dict 形式的参数。"""
    if path.endswith('.gguf'):
        from gguf import GGUFReader
        reader = GGUFReader(path)
        by_name = {t.name: t for t in reader.tensors}

        def get(nm):
            t = by_name[nm]
            arr = np.asarray(t.data)
            return torch.from_numpy(arr.reshape(
                t.data.shape if hasattr(t.data, 'shape') else arr.shape).copy()
            ).float()

        state = {
            'layers.0.weights.0': get('graph_conv.0.weight.0'),
            'layers.1.weights.0': get('graph_conv.1.weight.0'),
        }
        print('已从 GGUF 载入 %d 个张量' % len(state))
    else:
        ckpt = torch.load(path, map_location='cpu')
        state = ckpt['state_dict']
        print('已从 PyTorch 载入权重（记录精度 %.4f）' % ckpt.get('test_acc', float('nan')))

    model = GCN(input_dim=input_dim, output_dim=output_dim,
                supports=supports, hidden1=hidden1,
                dropout=0., weight_decay=0., learning_rate=0.01)
    model.load_state_dict(state)
    return model


def main():
    args = parse_args()

    # 邻接矩阵必须重新计算（不在权重文件中）
    adj, feat, ytr, yva, yte, mtr, mva, mte, araw = load_data(args.dataset, source=getattr(args, 'source', 'pickle'), csv_dir=getattr(args, 'csv_dir', None))
    supports = [preprocess_adj(araw)]

    model = load_weights(args.model_path, feat.shape[1], ytr.shape[1],
                         args.hidden1, supports)
    model.eval()

    for name, labels, mask in (('训练集', ytr, mtr),
                               ('验证集', yva, mva),
                               ('测试集', yte, mte)):
        model.labels, model.labels_mask = labels, mask
        with torch.no_grad():
            model(feat)          # 前向，结果存入 model.outputs
            acc = model._accuracy().item()
        print('%-6s 准确率 %.4f  (样本 %d)' % (name, acc, int(mask.sum())))

    # 预测示例（predict() 复用上一步的 model.outputs）
    model.labels, model.labels_mask = yte, mte
    with torch.no_grad():
        prob = model.predict()
    test_idx = torch.where(mte)[0][:10]
    print('\n前 10 个测试节点的预测:')
    for i in test_idx:
        p = int(prob[i].argmax())
        print('  node %-5d 真实=%d 预测=%d  置信度 %.3f'
              % (int(i), int(yte[i].argmax()), p, float(prob[i][p])))


if __name__ == '__main__':
    # 用上下文管理器显式声明稀疏不变量检查策略，消除 PyTorch 2.14 的 UserWarning
    with silence_sparse_warning():
        main()