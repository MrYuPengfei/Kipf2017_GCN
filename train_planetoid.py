"""GCN 复现：完全交给 PyG 内置数据集与模块的最简版本。

设计原则：**不自己解析数据、不自己算归一化、不自己写图层**。
    数据集  torch_geometric.datasets.Planetoid      （下载/解压/划分/补孤立节点）
    特征    NormalizeFeatures()                     （行归一化）
    图卷积  torch_geometric.nn.GCNConv             （自环 + D^-1/2 A D^-1/2）
    训练    全图 batch，Adam + 早停

对应论文：Kipf & Welling, *Semi-Supervised Classification with Graph
Convolutional Networks* (ICLR 2017)。原始代码是 TF 1.x，本文件是 PyG 版。

用法：
    python train_planetoid.py                       # Cora
    python train_planetoid.py --dataset citeseer
    python train_planetoid.py --num_layers 8        # 加深，观察过平滑
    python train_planetoid.py --report              # 只看本地缓存，不训练
"""

import argparse

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import degree

DATA_ROOT = 'data/Planetoid'


class GCN(torch.nn.Module):
    """论文的两层结构：F -> 16 -> C。

    ``GCNConv`` 内部已经完成了 ``Â = D^-1/2 (A + I) D^-1/2`` 的归一化，
    所以这里不需要（也不应该）再手动处理邻接矩阵。

    ``nn.Sequential`` 用不了——每层要收两个参数 (x, edge_index)，
    而 ``Sequential.forward`` 只透传一个。所以自己写一个几行的 forward，
    这也是 PyG 官方 example 的写法。
    """

    def __init__(self, in_dim, hidden, out_dim, num_layers, dropout):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            last = i == num_layers - 1
            self.convs.append(GCNConv(
                in_dim if i == 0 else hidden,
                out_dim if last else hidden))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:            # 末层不加激活，保留 logits
            x = F.dropout(F.relu(conv(x, edge_index)), p=self.dropout,
                          training=self.training)
        return self.convs[-1](x, edge_index)


def masked_accuracy(logits, y, mask):
    """只在 ``mask`` 为True 的节点上算准确率。"""
    return (logits[mask].argmax(1) == y[mask]).float().mean().item()


def report(name, data):
    """打印数据规模与类别分布，用来和论文的 Table 1 对照。"""
    n = data.num_nodes
    deg = degree(data.edge_index[0], n)
    # 注意：不同PyG 版本里 data.num_classes 不一定存在，从 y 推断最稳
    num_classes = int(data.y.max()) + 1
    print(f'{name}: {n} 节点, {data.num_edges} 条边, '
          f'{data.num_features} 维特征, {num_classes} 类')
    print(f'  划分  train {int(data.train_mask.sum())} / '
          f'val {int(data.val_mask.sum())} / test {int(data.test_mask.sum())}')
    print(f'  平均度 {deg.mean():.2f}, 孤立节点 {int((deg == 0).sum())}')
    print(f'  特征已行归一化: {bool(data.x.sum() > 0)}')


def main():
    p = argparse.ArgumentParser(description='GCN (PyG Planetoid + GCNConv)')
    p.add_argument('--dataset', default='cora',
                   choices=['cora', 'citeseer', 'pubmed'])
    p.add_argument('--num_layers', type=int, default=2, help='2 为论文设定')
    p.add_argument('--hidden', type=int, default=16, help='论文用16')
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--weight_decay', type=float, default=5e-4)
    p.add_argument('--patience', type=int, default=100,
                   help='验证损失连续多少轮不下降就停')
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--report', action='store_true', help='只打印缓存信息')
    args = p.parse_args()

    torch.manual_seed(args.seed)

    # ---- 数据：唯一需要自己写的两行 ----
    dataset = Planetoid(DATA_ROOT, args.dataset,
                        transform=NormalizeFeatures())
    data = dataset[0]

    report(args.dataset, data)
    if args.report:
        return

    # ---- 模型：论文超参 F -> 16 -> C，两层 ----
    model = GCN(data.num_features, args.hidden, dataset.num_classes,
                args.num_layers, args.dropout)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)

    best_val, best_test, best_epoch, wait = float('inf'), 0.0, 0, 0
    for epoch in range(1, 201):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(data.x, data.edge_index)[data.train_mask],
                               data.y[data.train_mask])
        loss.backward()
        opt.step()

        # 验证：早停判据只看验证损失
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            val_loss = F.cross_entropy(logits[data.val_mask],
                                       data.y[data.val_mask]).item()
            val_acc = masked_accuracy(logits, data.y, data.val_mask)

        if val_loss < best_val:
            best_val, best_epoch, wait = val_loss, epoch, 0
            # 论文按验证损失选模型，所以记录的是那一轮的测试精度
            best_test = masked_accuracy(logits, data.y, data.test_mask)
        else:
            wait += 1
            if wait >= args.patience:
                break

        if epoch % 20 == 0 or epoch == 1:
            print(f'epoch {epoch:3d}  loss {loss.item():.4f}  '
                  f'val_loss {val_loss:.4f}  val_acc {val_acc:.4f}')

    print(f'\nbest epoch {best_epoch}  val_loss {best_val:.4f}  '
          f'test_acc {best_test:.4f}')


if __name__ == '__main__':
    main()