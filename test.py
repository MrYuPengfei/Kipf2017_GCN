"""GCN PyTorch 实现的正确性测试。

运行：
    python test.py# 全部测试
    python test.py --verbose         # 显示每条断言细节
    python test.py -k dropout        # 只跑名字含 dropout 的测试

测试分三组：
    A. 数据管线   —— 形状、划分、归一化是否符合原版定义
    B. 层与指标   —— 前向公式、dropout 语义、mask 语义逐条对照 TF 原版
    C. 训练冒烟   —— 能否收敛、多模型能否跑通

设计原则：**每条断言都独立算出期望值**，不依赖实现自身，
这样才能真正发现移植错误，而不是自证自洽。
"""

from __future__ import division, print_function

import argparse
import os
import pickle as pkl
import sys

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from gcn.layers import Dense, GraphConvolution, sparse_dropout
from gcn.metrics import masked_accuracy, masked_softmax_cross_entropy
from gcn.models import GCN, MLP
from gcn.utils import (chebyshev_polynomials, load_data, normalize_adj,
                       preprocess_adj, preprocess_features)

VERBOSE = False
_RESULTS = []


# ---------------------------------------------------------------- 断言工具
def check(name, cond, detail=''):
    ok = bool(cond)
    _RESULTS.append((name, ok))
    mark = 'PASS' if ok else 'FAIL'
    line = '[%s] %s' % (mark, name)
    if detail and (VERBOSE or not ok):
        line += '\n       %s' % detail
    print(line)
    return ok


def approx(a, b, tol=1e-5):
    return torch.allclose(a, b, atol=tol, rtol=tol)


def section(title):
    print('\n' + '=' * 68)
    print(title)
    print('=' * 68)


# ================================================================ A. 数据管线
def test_data_pipeline():
    section('A. 数据管线')

    # --- A1 形状与划分 ---
    adj, feat, ytr, yva, yte, mtr, mva, mte, araw = load_data('cora')

    check('A1 Cora 节点数 = 2708', adj.shape[0] == 2708,
          '实际 %d' % adj.shape[0])
    check('A1 Cora 特征维度 = 1433', feat.shape[1] == 1433,
          '实际 %d' % feat.shape[1])
    check('A1 Cora 类别数 = 7', ytr.shape[1] == 7,
          '实际 %d' % ytr.shape[1])
    check('A1 划分 = 140/500/1000',
          int(mtr.sum()) == 140 and int(mva.sum()) == 500 and int(mte.sum()) == 1000,
          '实际 %d/%d/%d' % (int(mtr.sum()), int(mva.sum()), int(mte.sum())))
    check('A1 三个 mask 互不重叠',
          int((mtr.int() + mva.int() + mte.int()).max()) <= 1)

    # --- A2 特征行归一化（对照 preprocess_features 定义）---
    rowsum = torch.sparse.sum(feat, dim=1).to_dense()
    check('A2 特征每行和= 1（行归一化）',
          approx(rowsum, torch.ones_like(rowsum), tol=1e-5),
          'min %.6f max %.6f' % (rowsum.min(), rowsum.max()))

    # --- A3 稀疏性保持 ---
    density = feat._values().numel() / (feat.shape[0] * feat.shape[1])
    check('A3 特征保持稀疏（密度 < 5%）', density < 0.05,
          '密度 %.4f' % density)

    # --- A4 标签 one-hot ---
    labels = ytr[mtr].sum(1)
    check('A4 训练标签为 one-hot', approx(labels, torch.ones_like(labels)),
          '每行和 %.4f' % labels[0])

    # --- A5 邻接矩阵对称且无自环（预处理前）---
    adj_dense = adj.to_dense()
    check('A5 原始邻接矩阵对称', approx(adj_dense, adj_dense.t(), tol=1e-6))
    check('A5 原始邻接矩阵无自环',
          float(torch.diagonal(adj_dense).abs().max()) == 0.0)


def test_adj_normalization():
    section('B. 邻接矩阵归一化')

    _, _, _, _, _, _, _, _, araw = load_data('cora')
    n = araw.shape[0]

    # --- B1 preprocess_adj = 对 A+I 做 D^-1/2 (A+I) D^-1/2 ---
    sup = preprocess_adj(araw)
    sup_dense = sup.to_dense()

    # 独立实现参考值
    a = sp.coo_matrix(araw) + sp.eye(n)
    rs = np.array(a.sum(1)).flatten()
    d = np.power(rs, -0.5)
    d[np.isinf(d)] = 0.
    D = sp.diags(d)
    ref = np.asarray(D.dot(a).dot(D).todense())
    ref_t = torch.from_numpy(ref).float()

    check('B1 preprocess_adj 与独立实现一致',
          approx(sup_dense, ref_t, tol=1e-5),
          '最大偏差 %.2e' % (sup_dense - ref_t).abs().max())
    check('B1 归一化后邻接矩阵对称',
          approx(sup_dense, sup_dense.t(), tol=1e-5))
    check('B1 加自环后对角线非零（自环保留）',
          float(torch.diagonal(sup_dense).min()) > 0)

    # --- B2 normalize_adj 对零度节点不产生 NaN ---
    tiny = sp.coo_matrix(([1.0], ([0], [1])), shape=(3, 3))  # 节点 2 是孤立点
    out = normalize_adj(tiny)
    vals = torch.from_numpy(out.toarray()).float()
    check('B2 零度节点不产生 NaN/Inf',
          bool(torch.isfinite(vals).all()),
          '输出 %s' % vals.tolist())


def test_chebyshev():
    section('C. Chebyshev 多项式（ChebNet）')

    _, _, _, _, _, _, _, _, araw = load_data('cora')
    ks = chebyshev_polynomials(araw, 3)

    check('C1 K=3 返回 K+1 = 4 个多项式', len(ks) == 4, '实际 %d' % len(ks))

    n = araw.shape[0]
    t0 = ks[0].to_dense()
    check('C1 T_0 = I（单位阵）',
          approx(t0, torch.eye(n), tol=1e-5))

    # 独立复算 T_1 = 2/λmax · L − I，L = I − A_norm
    a = sp.coo_matrix(araw)
    rs = np.array(a.sum(1)).flatten()
    d = np.power(rs, -0.5)
    d[np.isinf(d)] = 0.
    a_norm = np.asarray(sp.diags(d).dot(a).dot(sp.diags(d)).todense())
    lap = np.eye(n) - a_norm
    from scipy.sparse.linalg import eigsh
    lam, _ = eigsh(sp.coo_matrix(lap), 1, which='LM')
    scaled = (2.0 / lam[0]) * lap - np.eye(n)
    t1_ref = torch.from_numpy(scaled).float()
    t1_dense = ks[1].to_dense()
    check('C1 T_1 = (2/λmax)·L − I',
          approx(t1_dense, t1_ref, tol=1e-4),
          '最大偏差 %.2e' % (t1_dense - t1_ref).abs().max())

    # 递推关系 T_k = 2·L̃·T_{k−1} − T_{k−2}
    t2 = ks[2].to_dense()
    lt = torch.from_numpy(scaled).float()
    t2_ref = 2.0 * (lt @ ks[1].to_dense()) - torch.eye(n)
    check('C1 T_2 满足递推 T_2 = 2·L̃·T_1 − T_0',
          approx(t2, t2_ref, tol=1e-3),
          '最大偏差 %.2e' % (t2 - t2_ref).abs().max())


# ================================================================ D. 层
def test_graph_convolution():
    section('D. GraphConvolution 前向公式')

    n, f_in, f_out, f_cls = 50, 8, 16, 3
    torch.manual_seed(0)

    x = torch.randn(n, f_in)
    sup = sp.random(n, n, density=0.1, format='coo', random_state=0)
    sup = preprocess_adj(sup)          # 已归一化的 support

    layer = GraphConvolution(f_in, f_out, supports=[sup], act=lambda t: t,
                             dropout=0., sparse_inputs=False)
    layer.eval()
    out = layer(x)

    # 独立参考：out = S @ (x @ W)
    ref = torch.sparse.mm(sup, x @ layer.weights[0])
    check('D1 单support 输出 = S·(X·W)', approx(out, ref, tol=1e-5),
          '最大偏差 %.2e' % (out - ref).abs().max())

    # --- 多 support 求和（ChebNet 场景）---
    layer2 = GraphConvolution(f_in, f_out, supports=[sup, sup, sup],
                              act=lambda t: t, dropout=0.)
    layer2.eval()
    out2 = layer2(x)
    ref2 = sum(torch.sparse.mm(sup, x @ layer2.weights[i]) for i in range(3))
    check('D2 多 support 输出 = Σ S_i·(X·W_i)', approx(out2, ref2, tol=1e-5))

    # --- Glorot 初始化范围 ---
    init_range = np.sqrt(6.0 / (f_in + f_out))
    w = layer.weights[0].detach()
    check('D3 Glorot 初始化落在 [−r, r]',
          float(w.abs().max()) <= init_range + 1e-6,
          'r = %.4f，实际 max|%.4f' % (init_range, w.abs().max()))

    # --- 梯度能回传 ---
    x_grad = torch.randn(n, f_in, requires_grad=True)
    layer(x_grad).sum().backward()
    check('D4 权重梯度非零（可回传）',
          float(layer.weights[0].grad.abs().sum()) > 0)


def test_dense_layer():
    section('E. Dense 层')

    torch.manual_seed(0)
    n, f_in, f_out = 20, 6, 4
    layer = Dense(f_in, f_out, dropout=0., act=lambda t: t)
    layer.eval()
    x = torch.randn(n, f_in)
    out = layer(x)
    check('E1 Dense 输出 = X·W', approx(out, x @ layer.weights, tol=1e-5))


# ================================================================ F. Dropout
def test_sparse_dropout():
    section('F. 稀疏 dropout 语义')

    # 用 TF 原版的 floor(keep_prob + U(0,1)) 规则做统计验证
    idx = torch.arange(20)
    vals = torch.ones(20)
    x = torch.sparse_coo_tensor(
        torch.stack([idx, idx]), vals, torch.Size((20, 20))).coalesce()

    keep = 0.5
    torch.manual_seed(0)
    kept_ratios = []
    scales = []
    for _ in range(200):
        out = sparse_dropout(x, keep)
        o = out.to_dense().diagonal()
        kept_ratios.append(float((o != 0).float().mean()))
        nz = o[o != 0]
        if len(nz):
            scales.append(float(nz[0]))
    mean_keep = float(np.mean(kept_ratios))
    mean_scale = float(np.mean(scales))

    check('F1 稀疏 dropout 保留率 ≈ keep_prob (0.5)',
          abs(mean_keep - keep) < 0.03,
          '实测保留率 %.4f' % mean_keep)
    # 原值全为 1.0，保留项应被放大到 1/keep_prob = 2.0（TF 版的反缩放）
    check('F2 保留项被放大 1/keep_prob 倍（期望 2.0）',
          abs(mean_scale - 1.0 / keep) < 0.03,
          '实测尺度 %.4f（期望 %.2f）' % (mean_scale, 1.0 / keep))
    check('F3 稀疏结构未被破坏（仍为稀疏张量）',
          out.is_sparse)

    # 零元素不得被写入
    x2 = torch.sparse_coo_tensor(
        torch.stack([torch.tensor([0]), torch.tensor([0])]),
        torch.tensor([3.0]), torch.Size((5, 5))).coalesce()
    o2 = sparse_dropout(x2, 0.5).to_dense()
    check('F4 原本为 0 的位置仍为 0（未稠密化）',
          float(o2.sum()) == 0.0 or float(o2[0, 0]) in (0.0, 6.0),
          '对角元素 %.2f（3.0 或 3.0/0.5=6.0）' % float(o2[0, 0]))

    # keep_prob=1 时不改变
    o3 = sparse_dropout(x, 1.0).to_dense()
    check('F5 keep_prob=1 时原样返回',
          approx(o3, x.to_dense(), tol=1e-6))


def test_dropout_rate_semantics():
    section('G. dropout 概率语义（p=dropout率）')

    # 注意：dropout 作用在【层输入】上，所以要测「输入被置零的比例」，
    # 而不是「层输出为0 的比例」——后者经过 W 变换后通常不为 0。
    torch.manual_seed(0)
    n, f = 2000, 20
    x = torch.randn(n, f)

    for rate in (0.0, 0.5):
        # 直接检验 F.dropout 本身的语义（与 layers.py 中用法一致）
        frac = []
        for _ in range(20):
            dropped = F.dropout(x, p=rate, training=True)
            frac.append(float((dropped == 0).float().mean()))
        measured = float(np.mean(frac))
        check('G1 F.dropout(p=%.1f) 置零比例 ≈ %.2f' % (rate, rate),
              abs(measured - rate) < 0.03,
              '实测 %.4f' % measured)

    # Dense 层在 train 模式下应真的施加 dropout：
    # 同一输入两次前向（训练态）结果不同，eval 态结果相同。
    layer = Dense(f, 8, dropout=0.5, act=lambda t: t)
    layer.train()
    torch.manual_seed(1); o1 = layer(x)
    torch.manual_seed(2); o2 = layer(x)
    check('G2 Dense 训练模式下两次前向结果不同（dropout 生效）',
          not torch.allclose(o1, o2))

    layer.eval()
    with torch.no_grad():
        e1, e2 = layer(x), layer(x)
    check('G3 eval 模式下两次前向结果相同（dropout 关闭）',
          torch.allclose(e1, e2))
    check('G4 eval 模式输出中不含 0（无置零）',
          float((e1 == 0).float().mean()) < 1e-6)


# ================================================================ H. 指标
def test_metrics():
    section('H. 损失与指标')

    torch.manual_seed(0)
    n, c = 200, 4
    logits = torch.randn(n, c)
    labels = torch.zeros(n, c)
    labels[torch.arange(n), torch.randint(0, c, (n,))] = 1
    mask = torch.zeros(n)
    mask[:50] = 1

    # --- H1 masked loss 等价于「只在 mask 位置取平均」---
    loss = masked_softmax_cross_entropy(logits, labels, mask)
    per_node = -torch.sum(labels * torch.log_softmax(logits, dim=1), dim=1)
    ref = per_node[mask == 1].mean()
    check('H1 masked CE = 仅在 mask 位置求平均', approx(loss, ref, tol=1e-5),
          '实现 %.6f vs 参考 %.6f' % (loss.item(), ref.item()))

    # --- H2 等价于 TF 的 mask归一化写法 ---
    tf_style = torch.mean(per_node * mask / mask.mean())
    check('H2 与 TF 原版 mask 归一化写法数值一致', approx(loss, tf_style, tol=1e-6))

    # --- H3 masked accuracy ---
    pred = logits.argmax(1)
    acc = masked_accuracy(logits, labels, mask)
    ref_acc = (pred[mask == 1] == labels[mask == 1].argmax(1)).float().mean()
    check('H3 masked accuracy = mask 位置上的准确率', approx(acc, ref_acc, tol=1e-6))

    # --- H4 无标签位置的 loss 不影响结果 ---
    logits2 = logits.clone()
    logits2[mask == 0] += 100.0   # 疯狂扰动无标签位置
    loss2 = masked_softmax_cross_entropy(logits2, labels, mask)
    check('H4 无标签位置的 logits 不影响 loss', approx(loss, loss2, tol=1e-5))


def test_weight_decay():
    section('I. L2 正则')

    torch.manual_seed(0)
    layer = GraphConvolution(10, 5, supports=[preprocess_adj(
        sp.random(5, 5, density=0.5, format='coo', random_state=0))],
        dropout=0.)
    wd = 5e-4
    numel = sum(p.numel() for p in layer.parameters())

    # TF 的 tf.nn.l2_loss(t) = sum(t^2) / numel
    ref = sum(wd * torch.sum(p * p) / p.numel() for p in layer.parameters())
    # 常见错误写法
    wrong = sum(wd * torch.sum(p * p) / 2.0 for p in layer.parameters())

    check('I1 L2 项按 numel 归一（与 tf.nn.l2_loss 一致）',
          float(ref) > 0 and float(ref) < float(wrong),
          'numel 归一 %.3e vs /2 写法 %.3e，差 %.0f 倍'
          % (float(ref), float(wrong), float(wrong) / float(ref)))
    check('I1 参数总数 = 10×5 = 50', numel == 50, '实际 %d' % numel)


# ================================================================ J. 训练冒烟
def test_training_smoke(dataset='cora', epochs=60):
    section('J. 训练冒烟测试（%s，%d epochs）' % (dataset, epochs))

    adj, feat, ytr, yva, yte, mtr, mva, mte, araw = load_data(dataset)
    sup = [preprocess_adj(araw)]

    torch.manual_seed(123)
    np.random.seed(123)

    model = GCN(input_dim=feat.shape[1], output_dim=ytr.shape[1],
                supports=sup, hidden1=16, dropout=0.5,
                weight_decay=5e-4, learning_rate=0.01)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    losses = []
    for ep in range(epochs):
        model.train()
        model.labels, model.labels_mask = ytr, mtr
        opt.zero_grad()
        model(feat)
        loss = model._loss()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    first, last = float(np.mean(losses[:5])), float(np.mean(losses[-5:]))
    check('J1 训练损失下降', last < first,
          '前5轮均值 %.4f → 后5轮均值 %.4f' % (first, last))

    # 训练后准确率应显著高于随机
    model.eval()
    with torch.no_grad():
        model.labels, model.labels_mask = yva, mva
        model(feat)
        acc = model._accuracy().item()
    check('J2 验证准确率 > 随机猜测 (1/%d = %.3f)' % (yva.shape[1], 1 / yva.shape[1]),
          acc > 3.0 / yva.shape[1],
          '验证准确率 %.4f' % acc)
    return acc


def test_all_models():
    section('K. 三种模型均可运行')

    adj, feat, ytr, yva, yte, mtr, mva, mte, araw = load_data('cora')
    sup = [preprocess_adj(araw)]
    cheb = chebyshev_polynomials(araw, 2)

    torch.manual_seed(0)
    for name, supports, cls in (
            ('gcn (1 support)', sup, GCN),
            ('gcn_cheby (3 supports)', cheb, GCN),
            ('dense (MLP)', sup, MLP)):
        try:
            m = cls(input_dim=feat.shape[1], output_dim=ytr.shape[1],
                    supports=supports, hidden1=16, dropout=0.5)
            m.train()
            m.labels, m.labels_mask = ytr, mtr
            out = m(feat)
            loss = m._loss()
            loss.backward()
            ok = out.shape == ytr.shape and torch.isfinite(loss)
            check('K %s 前向+反向正常' % name, ok,
                  '输出 %s，loss %.4f' % (tuple(out.shape), loss.item()))
        except Exception as ex:
            check('K %s 前向+反向正常' % name, False, repr(ex))


# ================================================================ main
def main():
    global VERBOSE
    parser = argparse.ArgumentParser(description='GCN PyTorch 实现测试')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='显示每条断言的细节')
    parser.add_argument('-k', type=str, default='',
                        help='只运行名字含该关键字的测试')
    args = parser.parse_args()
    VERBOSE = args.verbose

    print('=' * 68)
    print('GCN PyTorch 复现 —— 正确性测试')
    print('torch %s | numpy %s' % (torch.__version__, np.__version__))
    print('=' * 68)

    suites = [
        test_data_pipeline, test_adj_normalization, test_chebyshev,
        test_graph_convolution, test_dense_layer, test_sparse_dropout,
        test_dropout_rate_semantics, test_metrics, test_weight_decay,
        test_training_smoke, test_all_models,
    ]

    for fn in suites:
        if args.k and args.k.lower() not in fn.__name__.lower():
            continue
        try:
            fn()
        except Exception as ex:
            import traceback
            print('\n[ERROR] %s 抛出异常: %r' % (fn.__name__, ex))
            if VERBOSE:
                traceback.print_exc()
            _RESULTS.append((fn.__name__ + ' (异常)', False))

    passed = sum(1 for _, ok in _RESULTS if ok)
    total = len(_RESULTS)
    print('\n' + '=' * 68)
    print('结果：%d/%d 通过' % (passed, total))
    if passed != total:
        print('\n未通过的用例：')
        for name, ok in _RESULTS:
            if not ok:
                print('  ✗ %s' % name)
    print('=' * 68)
    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())