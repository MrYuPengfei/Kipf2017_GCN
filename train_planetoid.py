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

    # 保存 / 加载 GGUF 权重
    python train_planetoid.py --save                # 训练后存 model/gcn_pyg_cora.gguf
    python train_planetoid.py --load model/gcn_pyg_cora.gguf   # 跳过训练，直接推理
"""

import argparse
import os

import torch
import torch.nn.functional as F

import gguf_io
# PyG 的导入统一走 pyg_warn：它在 catch_warnings 块内完成导入，
# 从而压掉 Python 3.14 下的导入期告警（详见 pyg_warn 模块 docstring）
from pyg_warn import GCNConv, NormalizeFeatures, Planetoid, degree, silence

DATA_ROOT = 'data/Planetoid'
MODEL_DIR = 'model'


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

    # ------------------------------------------------------------------
    # 权重存取
    #
    # GGUF 存什么、不存什么，详见 gguf_io 模块 docstring。
    # 最重要的一条：**图结构（edge_index）与特征 x 不在权重文件里**，
    # 推理时必须重新提供——这是 GGUF 无法完整表达 GCN 的根本原因。
    # ------------------------------------------------------------------

    def save(self, path, meta=None):
        """把权重存成 ``.gguf``（主产物）与 ``.pt``（推荐实际使用）。

        :param path: ``.gguf`` 路径；``.pt`` 与 ``.meta.json`` 写在同目录
        :param meta: 额外元信息（数据集名、测试精度等）
        :return: ``(gguf_path, pt_path, meta_path)``
        """
        meta = dict(meta or {})
        meta.setdefault('name', os.path.splitext(os.path.basename(path))[0])
        meta.setdefault('dataset', 'unknown')

        tensors, meta_path = gguf_io.save_gguf(path, self, meta)
        pt_path = gguf_io.save_pt(os.path.splitext(path)[0] + '.pt', self, meta)

        print(f'\n已保存权重 -> {path}  ({os.path.getsize(path) / 1024:.1f} KB)')
        for name, shape in tensors:
            print(f'    {name:28s} {list(shape)}')
        print(f'    -> {pt_path}')
        print(f'    -> {meta_path}')
        return path, pt_path, meta_path

    def load(self, path, strict=True):
        """从 ``.gguf`` 或 ``.pt`` 载入权重到当前模型。"""
        gguf_io.load_any(path, self)
        print(f'已从 {path} 载入权重')
        return self

    @classmethod
    def from_file(cls, path, dropout=0.0):
        """**只凭权重文件**重建模型。

        GGUF 的 KV 区存了层数与各层维度，张量本身也带形状，
        所以不需要外部传 in_dim/hidden/out_dim——这比硬编码参数更健壮。

        注意：重建出来的模型**只有权重**，仍需外部提供 x 与 edge_index。
        """
        state, meta = gguf_io.load_any(path)
        model = cls(meta['num_features'], meta['hidden_dim'],
                    meta['num_classes'], meta['num_layers'], dropout)
        model.load_state_dict(state)
        print(f'已从 {path} 重建模型: {meta["num_layers"]} 层, '
              f'{meta["num_features"]}->{meta["hidden_dim"]}->'
              f'{meta["num_classes"]}, 数据集 {meta.get("dataset")}')
        if meta.get('test_acc') is not None:
            print(f'    文件内记录的测试精度: {meta["test_acc"]:.4f}')
        return model, meta


def masked_accuracy(logits, y, mask):
    """只在 ``mask`` 为True 的节点上算准确率。"""
    return (logits[mask].argmax(1) == y[mask]).float().mean().item()


def report(name, data):
    """打印数据规模与类别分布，用来和论文的 Table 1 对照。"""
    n = data.num_nodes
    deg = degree(data.edge_index[0], n)
    # 注意：不同PyG 版本里 data.num_classes 不一定存在，从 y 推断最稳
    num_classes = int(data.y.max()) + 1
    print(f'{name}: {n} 节点, {data.num_edges // 2} 条边, '
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
    p.add_argument('--save', action='store_true',
                   help='训练后把权重存成 GGUF + .pt')
    p.add_argument('--load', type=str, default='',
                   help='从 .gguf / .pt 载入权重并直接推理（跳过训练）')
    p.add_argument('--model_dir', type=str, default=MODEL_DIR,
                   help='权重保存目录，默认 model/')
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

    # ---- 载入已有权重：跳过训练，直接推理 ----
    if args.load:
        model.load(args.load)
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
        acc = masked_accuracy(logits, data.y, data.test_mask)
        print(f'\n从 {args.load} 载入权重，test_acc {acc:.4f}')

        if args.save:
            stem = f'gcn_pyg_{args.dataset}'
            os.makedirs(args.model_dir, exist_ok=True)
            model.save(os.path.join(args.model_dir, stem + '.gguf'),
                       meta={'dataset': args.dataset, 'test_acc': acc})
        return

    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)

    best_val, best_test, best_epoch, wait = float('inf'), 0.0, 0, 0
    # 验证损失最优那一轮的权重快照。
    # 为什么需要它：训练结束时 model 停在**最后一轮**，而 best_test 记的是
    # **验证最优轮**的测试精度——两者本就不是同一个模型。若直接保存 model，
    # 就会出现「训练报 72.3%、重新加载却是 71.5%」的困惑。
    # 存最优轮的权重，才能让 --load 复现出与训练时相同的精度。
    best_state = None

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
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                break

        if epoch % 20 == 0 or epoch == 1:
            print(f'epoch {epoch:3d}    loss {loss.item():.4f}  '
                  f'val_loss {val_loss:.4f}  val_acc {val_acc:.4f}')

    print(f'\nbest epoch {best_epoch}  val_loss {best_val:.4f}  '
          f'test_acc {best_test:.4f}')

    # ---- 保存权重 ----
    if args.save:
        os.makedirs(args.model_dir, exist_ok=True)
        stem = f'gcn_pyg_{args.dataset}'
        # 回滚到验证最优轮的权重，保证 --load 能复现出同样的 test_acc
        if best_state is not None:
            model.load_state_dict(best_state)
        model.save(os.path.join(args.model_dir, stem + '.gguf'),
                   meta={'dataset': args.dataset, 'test_acc': best_test,
                         'best_epoch': best_epoch})


if __name__ == '__main__':
    # PyG 在 Python 3.14 下的告警发生在 GCNConv 实例化时（inspect_signature），
    # 必须包住 main()，详见 pyg_warn 模块 docstring
    with silence():
        main()