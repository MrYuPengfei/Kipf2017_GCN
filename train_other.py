"""GCN 在**非论文数据集**上的节点分类。

与 ``train_planetoid.py`` 的唯一区别是**数据从哪来**——模型结构、超参、
训练循环完全一致，所以两者的精度差异只反映数据本身，不掺杂实现差异。

为什么不继续用 Cora/Citeseer/Pubmed：
    Planetoid 是 2017 年的引文网络，规模小（2708~19717 节点）、特征是0/1词袋，
    在它上面刷到 80% 说明不了模型在真实数据上是否有效。
    本文件换成规模更大、来源更杂的数据：学术合作网、商品购买网、
    社交网络、Wikipedia 页面链接网等。

数据来源全部是 ``torch_geometric.datasets``（无需自己解析、无需 DataLoader）。

用法：
    python train_other.py --list                # 看有哪些数据集
    python train_other.py --dataset coauthor-cs
    python train_other.py --dataset amazon-computers
    python train_other.py --dataset wiki-chameleon    # 异配图，看GCN 失效
    python train_other.py --dataset reddit            # 25 万节点，慢
"""

import argparse

import torch
import torch.nn.functional as F
from torch_geometric.datasets import (Amazon, Coauthor, Flickr,
                                     HeterophilousGraphDataset, WikipediaNetwork)
from torch_geometric.datasets.linkx_dataset import LINKXDataset
from torch_geometric.datasets.reddit import Reddit
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import degree

DATA_ROOT = 'data/Other'


# --------------------------------------------------------------------------
# 数据集注册表
#
# split: 'given'   = 数据集自带 train/val/test mask（直接用，可比）
#        'random' = 数据集不带划分，按每类比例随机切（结果随 seed 变）
#   split: 'loader' = 逐 mini-batch 训练（图分类/大规模图专用）
# --------------------------------------------------------------------------
REGISTRY = {
    # ---- 引文/合作网：与 Planetoid 同类但更大 ----
    'coauthor-cs': dict(
        make=lambda: Coauthor(DATA_ROOT, 'CS', transform=NormalizeFeatures()),
        split='random', note='学术合作网 Coauthor CS'),
    'coauthor-physics': dict(
        make=lambda: Coauthor(DATA_ROOT, 'Physics',
                              transform=NormalizeFeatures()),
        split='random', note='学术合作网 Coauthor Physics'),

    # ---- 商品共购网 ----
    'amazon-computers': dict(
        make=lambda: Amazon(DATA_ROOT, 'Computers',
                            transform=NormalizeFeatures()),
        split='random', note='亚马逊电子产品评论图'),

    # ---- 社交/内容网络 ----
    'flickr': dict(
        make=lambda: Flickr(DATA_ROOT, transform=NormalizeFeatures()),
        split='given', note='Flickr 图片标签图，自带 7:3:10 划分'),
    'reddit': dict(
        make=lambda: Reddit(DATA_ROOT, transform=NormalizeFeatures()),
        split='given', note='Reddit 帖子社区，23 万节点'),

    # ---- 异配图：同标签节点往往不相邻，专门用来暴露 GCN 的短板 ----
    'wiki-chameleon': dict(
        make=lambda: WikipediaNetwork(DATA_ROOT, 'chameleon',
                                      transform=NormalizeFeatures()),
        split='given', note='Wikipedia 页面链接网，异配'),
    'wiki-squirrel': dict(
        make=lambda: WikipediaNetwork(DATA_ROOT, 'squirrel',
                                      transform=NormalizeFeatures()),
        split='given', note='Wikipedia 页面链接网，异配'),
    'webkb-texas': dict(
        make=lambda: __import__('torch_geometric.datasets.webkb',
                                fromlist=['WebKB']).WebKB(
                                    DATA_ROOT, 'texas',
                                    transform=NormalizeFeatures()),
        split='given', note='WebKB 大学网页，异配'),
    'questions': dict(
        make=lambda: HeterophilousGraphDataset(
            DATA_ROOT, 'Questions',
            transform=NormalizeFeatures()).get(0),
        split='given', note='Yandex Q 问答图，异配'),

    # ---- 大规模：需要 mini-batch ----
    'ogbn-papers100m-sample': dict(
        make=lambda: LINKXDataset(DATA_ROOT, 'papers100M',
                                  transform=NormalizeFeatures()),
        split='given', note='学术论文引用，1.1 亿节点（需下载数 GB）'),
}


class GCN(torch.nn.Module):
    """与 ``train_planetoid.py`` 完全相同的结构：F -> 16 -> C。

    ``GCNConv`` 内部完成加自环与 D^-1/2(A+I)D^-1/2 归一化，不需要手动处理邻接矩阵。
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
        for conv in self.convs[:-1]:
            x = F.dropout(F.relu(conv(x, edge_index)), p=self.dropout,
                          training=self.training)
        return self.convs[-1](x, edge_index)


def random_split_masks(y, num_classes, seed,
                       per_class=20, n_val=500, n_test=3000):
    """数据集不带划分时，按「每类 per_class 个训练 + 固定验证/测试数」切。

    对齐 Planetoid 的比例：每类 20 个训练样本（约 2.8%），其余按 500/3000 划走。
    与 Planetoid 的差别是**验证/测试更多**，所以这里的绝对精度天然偏低，
    横向比较时应以 MLP 基线为参照（见 --baseline）。
    """
    g = torch.Generator().manual_seed(seed)
    n = y.size(0)
    train_mask = torch.zeros(n, dtype=torch.bool)
    for c in range(num_classes):
        idx = (y == c).nonzero(as_tuple=True)[0]
        perm = idx[torch.randperm(idx.size(0), generator=g)]
        train_mask[perm[:per_class]] = True

    rest = (~train_mask).nonzero(as_tuple=True)[0]
    perm = rest[torch.randperm(rest.size(0), generator=g)]
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    val_mask[perm[:n_val]] = True
    test_mask[perm[n_val:n_val + n_test]] = True
    return train_mask, val_mask, test_mask


def describe(name, data, mask_note):
    """打印规模；顺带算同质率，用来判断这份数据对 GCN 是否友好。"""
    n = data.num_nodes
    deg = degree(data.edge_index[0], n)

    # 同质率：相连节点里标签相同的比例。接近 0.5 = 完全随机，接近 1 = 同配
    row, col = data.edge_index
    same = (data.y[row] == data.y[col]).float().mean().item()

    print(f'\n=== {name} ===')
    print(f'{n} 节点, {data.num_edges} 条边, {data.num_features} 维特征, '
          f'{int(data.y.max()) + 1} 类')
    print(f'平均度 {deg.mean():.2f}, 孤立节点 {int((deg == 0).sum())}, '
          f'同质率 {same:.3f}')
    for k in ['train_mask', 'val_mask', 'test_mask']:
        m = getattr(data, k, None)
        if m is not None:
            print(f'  {k:11s} {int(m.sum())}')
    print(f'  划分来源: {mask_note}')


def masked_accuracy(logits, y, mask):
    return (logits[mask].argmax(1) == y[mask]).float().mean().item()


def run_full_batch(model, data, args, num_classes):
    """全图训练：Cora 那套协议，GNN 节点分类的标准做法。"""
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)

    best_val, best_test, best_epoch, wait = float('inf'), 0.0, 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(data.x, data.edge_index)[data.train_mask],
                               data.y[data.train_mask])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            val_loss = F.cross_entropy(logits[data.val_mask],
                                       data.y[data.val_mask]).item()
            val_acc = masked_accuracy(logits, data.y, data.val_mask)

        if val_loss < best_val:
            best_val, best_epoch, wait = val_loss, epoch, 0
            best_test = masked_accuracy(logits, data.y, data.test_mask)
        else:
            wait += 1
            if wait >= args.patience:
                break

        if epoch % 20 == 0 or epoch == 1:
            print(f'epoch {epoch:3d}  loss {loss.item():.4f}  '
                  f'val_loss {val_loss:.4f}  val_acc {val_acc:.4f}')

    print(f'best epoch {best_epoch}  val_loss {best_val:.4f}  '
          f'test_acc {best_test:.4f}')
    return best_test


def main():
    p = argparse.ArgumentParser(description='GCN on non-Planetoid datasets')
    p.add_argument('--dataset', default='coauthor-cs', choices=sorted(REGISTRY))
    p.add_argument('--list', action='store_true', help='列出所有可用数据集')
    p.add_argument('--num_layers', type=int, default=2)
    p.add_argument('--hidden', type=int, default=16)
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--weight_decay', type=float, default=5e-4)
    p.add_argument('--patience', type=int, default=100)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--seed', type=int, default=123)
    args = p.parse_args()

    if args.list:
        print(f'可用数据集（共 {len(REGISTRY)} 个）:\n')
        for k, v in sorted(REGISTRY.items()):
            print(f'  {k:26s} {v["split"]:7s} {v["note"]}')
        return

    torch.manual_seed(args.seed)
    cfg = REGISTRY[args.dataset]
    obj = cfg['make']()

    # HeterophilousGraphDataset.get() 返回单个 Data，其余返回 Dataset
    data = obj if hasattr(obj, 'edge_index') else obj[0]

    # 补齐划分：自带就用，自带的没有就随机切
    note = '数据集自带'
    if not hasattr(data, 'train_mask'):
        c = int(data.y.max()) + 1
        data.train_mask, data.val_mask, data.test_mask = \
            random_split_masks(data.y, c, args.seed)
        note = f'随机划分（每类 20 训练 / 500 验证 / 3000 测试）'
    else:
        # WikipediaNetwork / WebKB 给的是 (N, num_splits) 的二维 mask，取第 0 列
        for k in ['train_mask', 'val_mask', 'test_mask']:
            m = getattr(data, k)
            if m.dim() > 1:
                setattr(data, k, m[:, 0])

    describe(args.dataset, data, note)

    model = GCN(data.num_features, args.hidden, int(data.y.max()) + 1,
                args.num_layers, args.dropout)
    print(f'可训练参数: {sum(p.numel() for p in model.parameters())}')
    run_full_batch(model, data, args, int(data.y.max()) + 1)


if __name__ == '__main__':
    main()