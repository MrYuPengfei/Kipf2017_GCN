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
import warnings

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
CSV_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'data', 'csv')


def parse_index_file(filename):
    """逐行读取索引文件（每行一个整数），返回 int 列表。

    对应原版的 ``ind.{ds}.test.index``：存放测试节点的原始索引。
    """
    index = []                       # 结果列表
    with open(filename) as f:        # 文本模式，行分隔
        for line in f:               # 逐行遍历（文件末尾通常有换行）
            index.append(int(line.strip()))   # 去空白后转 int
    return index


def sample_mask(idx, l):
    """把索引集合转成布尔 mask。

    对应 TF 版的 ``tf.sequence_mask`` 写法。之所以不用 TF 原版的
    ``np.array(mask)[idx]`` 循环方式，是因为 NumPy 的花式索引赋值
    对重复索引只生效一次，而这里 idx 由 ``range()`` 构成故无歧义。

    :param idx:参与计算的节点索引（列表/ range）
    :param l:   mask 总长度，必须等于全体节点数 N
    :return:长度 l 的布尔 ndarray
    """
    mask = np.zeros(l)              # 先全 0 的 float64 数组
    mask[idx] = 1                    # 花式索引赋值，把选中的位置置 1
    return np.array(mask, dtype=bool) # 转布尔类型，供后续 torch.from_numpy 使用


def _to_torch_sparse(mx):
    """scipy稀疏矩阵 -> torch 稀疏张量（coalesce 以便高效 matmul）。

    :param mx:任意 scipy 稀疏矩阵/数组
    :return:  coalesce 过的 ``torch.sparse_coo_tensor``（float32 values）
    """
    mx = sp.coo_matrix(mx)                    # 统一到 COO 格式
    # mx.row / mx.col 是 (nnz,) 的行/列索引，vstack 后变成 (2, nnz)
    # torch 稀疏张量的 indices 要求 int64，故显式 astype
    indices = torch.from_numpy(np.vstack((mx.row, mx.col)).astype(np.int64))
    values = torch.from_numpy(mx.data.astype(np.float32))   # 统一 float32
    # coalesce() 会排序并合并重复坐标，这是 torch.sparse.mm 高效执行的前提
    return torch.sparse_coo_tensor(indices, values, torch.Size(mx.shape)).coalesce()


def _to_modern_sparse(obj):
    """把旧版 pickle 里的稀疏矩阵规范化为新版 scipy 命名空间的对象。

    数据集文件是 2017 年序列化的，其中存储的是 `scipy.sparse.csr.csr_matrix`。
    SciPy 1.14 起该子模块被弃用（SciPy 2.0 将移除），访问其属性会触发
    DeprecationWarning。这里显式转成 `sp.csr_matrix` 等新版类型，
    既消除警告，也保证未来 SciPy 2.0 下仍可运行。
    """
    if sp.issparse(obj):
        # 用 __class__ 保持具体类型（csr_matrix/csr_array/coo_matrix…），
        # 只把命名空间从废弃的 scipy.sparse.csr.* 换到新版 scipy.sparse.*
        fmt = obj.format
        new_cls = getattr(sp, fmt + '_matrix', None)
        if new_cls is not None and type(obj) is not new_cls:
            return new_cls(obj)
        return obj
    return obj


def load_data(dataset_str, data_dir=DATA_DIR, source='pickle', csv_dir=None):
    """加载引文网络数据集。

    :param dataset_str: 'cora' / 'citeseer' / 'pubmed'
    :param data_dir:pickle 数据目录
    :param source: 'pickle'（原始 TF 格式）或 'csv'（由 export_csv.py 生成）
    :param csv_dir: CSV 数据目录，默认 ``<项目>/data/csv``
    :return: adj_t, features, y_train, y_val, y_test,
             train_mask, val_mask, test_mask, adj_raw
             —— 前 8 项与 TF 原版一一对应（adj/features 已转为 torch 稀疏张量），
             最后一项 adj_raw 是 scipy 域的原始邻接矩阵，供 chebyshev_polynomials 使用。
    """
    if source == 'csv':
        return load_data_csv(dataset_str, csv_dir or CSV_DIR)

    # ---- 1. 读取 7 个 pickle 文件 ----
    # 各文件含义：
    #   x/y      训练节点的特征与 one-hot 标签
    #   tx/ty    测试节点的特征与标签
    #   allx/ally 全部「训练侧」节点（含有标签与无标签）的特征与标签
    #   graph    {节点索引: [邻居索引,...]} 邻接字典
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []                      # 按names 顺序存放解析结果
    for name in names:                # 逐个读取
        path = os.path.join(data_dir, 'ind.{}.{}'.format(dataset_str, name))
        with open(path, 'rb') as f:  # 二进制模式，pickle 要求
            # 反序列化旧 pickle 时，scipy 内部会访问已弃用的 csr/lil 等
            # 子模块而发出 DeprecationWarning；此处静音并随后规范化类型。
            with warnings.catch_warnings():   # 局部静音，不污染全局
                warnings.filterwarnings(
                    'ignore',
                    message='.*scipy\\.sparse\\..*namespace is deprecated.*')
                warnings.filterwarnings(
                    'ignore',
                    category=DeprecationWarning,
                    module='scipy')
                if sys.version_info > (3, 0):
                    # Python3 反序列化 Python2 的 pickle 必须用 latin1，
                    # 否则非 ASCII 字符串会直接抛 UnicodeDecodeError
                    obj = pkl.load(f, encoding='latin1')
                else:
                    obj = pkl.load(f)        # Python2 分支（本项目不会走到）
        # 规范化稀疏矩阵类型，消除后续所有对废弃子模块的访问
        objects.append(_to_modern_sparse(obj))

    # 解包成具名变量，后面的公式才读得懂
    x, y, tx, ty, allx, ally, graph = tuple(objects)

    # ---- 2. 测试索引与 Citeseer 孤立节点修复 ----
    test_idx_reorder = parse_index_file(
        os.path.join(data_dir, 'ind.{}.test.index'.format(dataset_str)))
    # test_idx_reorder 是**乱序**的原始索引；排序后才是连续的测试区间。
    # 后面要先按sorted 顺序拼接到allx 后面，再按reorder 顺序换回原位。
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Citeseer 有孤立节点：test.index 里缺少这些节点，
        # 直接拼接会让测试集整体错位。用零向量补齐到正确位置。
        # 补齐区间 = [min(test_idx_reorder), max(test_idx_reorder)] 的整数闭区间
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        # 先建一个全零的 LIL 矩阵（按行赋值效率高于 CSR）
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        # 把 tx 的每一行放到它真正该在的位置（相对 min 偏移）
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        # 标签同理补零；孤立节点的标签本就是未知的
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    # ---- 3. 拼装全体节点的特征矩阵 ----
    # vstack：训练侧特征在上、测试侧特征在下 -> (N, F)
    # 转LIL 是因为下面要用花式索引赋值（CSR 不支持），代价是后续 matmul 稍慢，
    # 但预处理只有一次，可以接受
    features = sp.vstack((allx, tx)).tolil()
    # 把按sorted 顺序拼接的测试行，换回 test_idx_reorder 指定的原始顺序
    # 左值是「期望位置」，右值是「当前所在位置」，numpy 会自动拷贝
    features[test_idx_reorder, :] = features[test_idx_range, :]

    # ---- 4. 邻接矩阵 ----
    # from_dict_of_lists 把 {idx: [邻居...]} 转成 networkx 图
    # adjacency_matrix 再转成 scipy 稀疏矩阵（无向图会生成对称矩阵）
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    # ---- 5. 拼装全体节点标签 ----
    # ally 是训练侧（含有无标签节点，标签全 0），ty 是测试侧
    labels = np.vstack((ally, ty))                       # (N, C)
    labels[test_idx_reorder, :] = labels[test_idx_range, :]  # 同样换回原顺序

    # ---- 6. 划分 train / val / test 三个索引集合 ----
    idx_test = test_idx_range.tolist()       # 测试集：排好序的测试索引
    idx_train = range(len(y))                # 训练集：前 len(y) 个节点
    idx_val = range(len(y), len(y) + 500)    # 验证集：紧接训练集的 500 个节点
    # 这是 Planetoid 论文的标准划分（public splits），验证集固定 500 个

    # ---- 7. 生成布尔 mask ----
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    # ---- 8. 三个 one-hot 标签张量（非对应位置的行全为 0）----
    # 这样 masked 交叉熵只需乘上 mask 即可，避免逐子集切片
    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]   # 只填训练集的行
    y_val[val_mask, :] = labels[val_mask, :]         # 只填验证集的行
    y_test[test_mask, :] = labels[test_mask, :]       # 只填测试集的行

    # ---- 9. scipy 域 / torch 域双份邻接矩阵 ----
    adj_raw = sp.coo_matrix(adj)     # scipy 域副本（Chebyshev 用，要做谱分解）
    adj_t = _to_torch_sparse(adj)    # torch 域副本（GPU 计算用）

    # ---- 10. NumPy -> torch ----
    # 注意：原始（未加自环、未归一化）的 scipy 邻接矩阵单独返回，
    # 供 chebyshev_polynomials 使用——它内部要在 scipy 域做谱分解。
    # 特征在这里做行归一化（每行除以行和），后续层不再重复处理
    features = _to_torch_sparse(preprocess_features(features))
    y_train = torch.from_numpy(y_train.astype(np.float32))   # 标签转 float32
    y_val = torch.from_numpy(y_val.astype(np.float32))
    y_test = torch.from_numpy(y_test.astype(np.float32))
    train_mask = torch.from_numpy(train_mask)               # 布尔 mask 保持 bool
    val_mask = torch.from_numpy(val_mask)
    test_mask = torch.from_numpy(test_mask)

    # 返回 9 元组：前 8 项已转 torch，最后一项保留在 scipy 域
    return (adj_t, features, y_train, y_val, y_test,
            train_mask, val_mask, test_mask, adj_raw)


def preprocess_features(features):
    """特征矩阵行归一化：每行除以该行之和（零行保持零向量）。

    对论文而言这是 §3.4 的式(1) 里的 R 部分：
        X̂ = R^-1 X,  R = diag(Σ_j X_ij)

    作用是消除节点间的「论文篇幅差异」—— 特征总和等价于一篇文档的词数，
    归一化后每篇文档被当作等长输入处理。

    与 normalize_adj 同理：零特征行会让 0^-1 = inf，需显式静音并置 0
    （孤立节点 / 无词节点因此保持零向量，不影响传播）。
    """
    rowsum = np.array(features.sum(1))# (N,1) 每行的非零元素之和
    # 0^-1 = inf 会触发 divide-by-zero 警告；我们要的正是「inf → 0」，
    # 故先用 errstate 静音，再手动替换，避免污染 -W error 严格模式
    with np.errstate(divide='ignore', invalid='ignore'):
        r_inv = np.power(rowsum, -1).flatten()             # 倒数，展平成 (N,)
    r_inv[np.isinf(r_inv)] = 0.                # inf → 0（零行保持零向量）
    r_mat_inv = sp.diags(r_inv)                         # 对应对角矩阵 R^-1
    return r_mat_inv.dot(features)                      # (N,N)·(N,F) -> (N,F)


def normalize_adj(adj):
    """对称归一化邻接矩阵 D^-1/2 A D^-1/2。

    为什么要对称归一化（而不是行归一化 D^-1 A）：
    行归一化会让节点的更新幅度依赖其度数，高度数节点被放大；
    对称归一化保证「传播」在无向图上是可逆的、且保持谱对称性，
    这是 GCN 能在引文网络上稳定训练的前提。

    零度节点（孤立节点）会导致 0^-0.5 = inf。这里用 np.errstate 显式
    抑制除零警告，并把结果中的 inf 置 0（对应「零度节点不影响传播」）。
    """
    adj = sp.coo_matrix(adj)               # 统一格式，便于后续点乘
    rowsum = np.array(adj.sum(1))          # 每个节点的度数，形状 (N,1)
    # np.power(0, -0.5) 会产生 inf 并触发 divide-by-zero 警告；
    # 语义上我们要的正是「inf → 0」，故先静音再替换。
    with np.errstate(divide='ignore', invalid='ignore'):
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()   # 度数的 -1/2 次方
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.               # 孤立节点 → 0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)                # 对角矩阵 D^-1/2
    # A·D^-1/2 再转置得D^-1/2·A^T（无向图A 对称，但转置写法更通用）
    # 最后再右乘 D^-1/2，完整得到 D^-1/2 A D^-1/2
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """简单 GCN 用的邻接矩阵预处理：加自环 + 对称归一化（renormalization trick）。

    对应文档 §3.3：Ã = A + I，再做 D̃^-1/2 Ã D̃^-1/2。

    **加自环的意义**：没有自环时，孤立节点的行全是 0，传播后其表示恒为零，
    无法学习到任何信息；加上自环后 Ã 的每行至少有一个非零对角元，
    节点至少能保留自身特征。这就是论文里那个 "+I" 的作用。
    """
    n = adj.shape[0]                                      # 节点数
    adj_normalized = normalize_adj(adj + sp.eye(n))       # (A + I) 先加自环再归一化
    return _to_torch_sparse(adj_normalized)               # 转 torch 稀疏张量


def chebyshev_polynomials(adj, k):
    """计算到 K 阶的 Chebyshev 多项式，返回 torch 稀疏张量列表。

    输入 **scipy 稀疏矩阵**（未加自环的原始邻接矩阵）。
    对应文档 §3.2：把谱滤镜用 Chebyshev 多项式近似，得到 K 跳局部性，
    且避免 O(N^3) 的完整特征分解（这里只用 Lanczos 算法求最大特征值）。

    推导链条：
        L̃ = I − D^-1/2 A D^-1/2                归一化拉普拉斯
        λ̃ = 2L̃ / λ_max − I把谱域映射到 [-1, 1]，Chebyshev 多项式在该区间最良态
        T_0 = I,  T_1 = λ̃
        T_k = 2 · λ̃ · T_{k-1} − T_{k-2}Chebyshev 二阶递推

    :param adj: 原始（无自环）邻接矩阵，scipy 稀疏格式
    :param k:   最高阶数，返回列表长度 k+1
    :return:    长度 k+1 的 torch 稀疏张量列表
    """
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    # ---- 1. 构造归一化拉普拉斯 ----
    adj_normalized = normalize_adj(adj)                        # D^-1/2 A D^-1/2
    laplacian = sp.eye(adj.shape[0]) - adj_normalized# L̃ = I − Ã
    n = adj.shape[0]                                           # 节点数

    # ---- 2. 求最大特征值λ_max（只求最大的一个）----
    # eigsh 是 Lanczos 迭代，稀疏矩阵下复杂度 O(k·nnz)，
    # 远优于稠密特征分解的 O(N^3)（Pubmed 有19717 个节点，稠密分解不可行）
    # which='LM' 表示 Largest Magnitude，取模最大的特征值
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')

    # 缩放到 [-1, 1]：λ̃ = 2L̃/λ_max − I
    # 这一步是 Chebyshev 展开数值稳定的前提，否则高阶项会数值爆炸
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(n)

    # ---- 3. 按递推式逐阶生成 ----
    t_k = list()                    # 结果列表
    t_k.append(sp.eye(n))                       # T_0 = I
    t_k.append(scaled_laplacian)                # T_1 = λ̃

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        """T_k = 2·λ̃·T_{k-1} − T_{k-2}

        显式转成 CSR 再乘，避免 LIL/COO 混合格式下的隐式转换开销。
        """
        s_lap = sp.csr_matrix(scaled_lap, copy=True)   # 矩阵乘法要求高效格式
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    # 从第 2 阶开始递推到第 k 阶
    for _ in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    # 全部转成 torch 稀疏张量，供 GraphConvolution 直接使用
    return [_to_torch_sparse(m) for m in t_k]


# ---------------------------------------------------------------------------
# CSV 数据源（由 export_csv.py 生成）
# ---------------------------------------------------------------------------

def _read_csv_rows(path):
    """读取带表头的 CSV，返回 (表头, 行列表)。不依赖 pandas。

    刻意不引入 pandas：本项目只做三段式纯数字 CSV 的解析，
    标准库足够，也能避免为一个导出功能增加一层重依赖。

    :param path: CSV 文件路径
    :return: (header列表, rows列表)，rows 中每项是字符串列表
    """
    with open(path, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split(',')# 第一行是表头
        rows = [line.strip().split(',') for line in f if line.strip()]  # 跳过空行
    return header, rows


def load_data_csv(dataset_str, csv_dir=CSV_DIR):
    """从 CSV 目录加载数据集，返回值与 :func:`load_data` 完全一致。

    目录布局（由 ``export_csv.py`` 生成）：
        {csv_dir}/{dataset_str}/nodes.csv      node_id,split,label
        {csv_dir}/{dataset_str}/features.csv   node_id,feature_id,value
        {csv_dir}/{dataset_str}/edges.csv      source,target,weight
        {csv_dir}/{dataset_str}/meta.json      维度信息

    还原步骤与 pickle 路径一一对应：拼特征 -> 邻接矩阵 -> one-hot 标签 ->
    mask -> NumPy 转 torch，因此两条路径的数值结果完全相同。
    """
    ds_dir = os.path.join(csv_dir, dataset_str)          # 数据集子目录
    if not os.path.isdir(ds_dir):                        # 没导出过就直接报错并给出修复命令
        raise FileNotFoundError(
            'CSV 数据集不存在: {}。请先运行 `python export_csv.py`。'.format(ds_dir))

    # ---- 1. meta.json：拿到维度，避免靠矩阵形状反推 ----
    import json                                           # 局部导入，仅此处用到
    with open(os.path.join(ds_dir, 'meta.json'), 'r', encoding='utf-8') as f:
        meta = json.load(f)                                # 解析成 dict
    n_nodes = int(meta['num_nodes'])                      # 节点数 N
    n_features = int(meta['num_features'])                # 特征维度 F
    n_classes = int(meta['num_classes'])                  # 类别数 C

    # ---- 2. features.csv -> scipy 稀疏矩阵 (N, F) ----
    feats_path = os.path.join(ds_dir, meta['files']['features'])
    if meta.get('features_format') == 'dense':
        # 稠密宽表路径：每行 F 个数字，需逐行压缩成 CSR 的三个数组
        _, rows = _read_csv_rows(feats_path)
        indptr = [0]                                      # CSR 行指针，len = N+1
        indices, data = [], []                # 列索引数组 / 非零值数组
        for r in rows:                          # 逐行处理
            for j, v in enumerate(r[1:]):# 跳过第0 列的 node_id
                if float(v) != 0.0:            # 只保留非零
                    indices.append(j)
                    data.append(float(v))
            indptr.append(len(indices))         # 记录该行结束时累计的非零个数
        features = sp.csr_matrix((np.array(data), np.array(indices), indptr),
                                 shape=(n_nodes, n_features))
    else:
        # 稀疏长表路径（默认）：每行是 (node_id, feature_id, value)
        _, rows = _read_csv_rows(feats_path)
        rr = np.array([int(r[0]) for r in rows], dtype=np.int64)   # 行索引
        cc = np.array([int(r[1]) for r in rows], dtype=np.int64)   # 列索引
        vv = np.array([float(r[2]) for r in rows], dtype=np.float64)  # 值
        features = sp.coo_matrix((vv, (rr, cc)),              #三元组构造 COO
                                 shape=(n_nodes, n_features)).tocsr()

    # 原始 pickle 中的特征是 float32，且行归一化的结果对 float32 的求和
    # 顺序敏感（CSR 与 LIL 的累加顺序不同会带来最后一位差异）。
    # 这里同样转成 LIL 再归一化，使 CSV 路径与 pickle 路径逐位一致。
    features = features.astype(np.float32).tolil()

    # ---- 3. edges.csv -> 邻接矩阵 (N, N) ----
    # 无向图在导出时已双向各存一行，这里直接按三元组还原即为对称矩阵
    _, erows = _read_csv_rows(os.path.join(ds_dir, meta['files']['edges']))
    er = np.array([int(r[0]) for r in erows], dtype=np.int64)   # 源节点
    ec = np.array([int(r[1]) for r in erows], dtype=np.int64)   # 目标节点
    ev = np.array([float(r[2]) for r in erows], dtype=np.float64)  # 权重（全1）
    adj = sp.coo_matrix((ev, (er, ec)), shape=(n_nodes, n_nodes))

    # ---- 4. nodes.csv -> 标签与三个 mask ----
    _, nrows = _read_csv_rows(os.path.join(ds_dir, meta['files']['nodes']))
    splits = np.array([r[1] for r in nrows], dtype=object)  # train/val/test/other
    label_idx = np.array([int(r[2]) for r in nrows], dtype=np.int64)  # -1 表示无标签

    # 由类别下标还原 one-hot 标签矩阵 (N, C)
    labels = np.zeros((n_nodes, n_classes))
    labeled = label_idx >= 0                # 布尔掩码，标记有标签的节点
    labels[np.where(label_idx >= 0)[0], label_idx[labeled]] = 1.0

    # 由 split 列直接生成三个布尔 mask，与 pickle 路径的 sample_mask 等价
    train_mask = np.array(splits == 'train', dtype=bool)
    val_mask = np.array(splits == 'val', dtype=bool)
    test_mask = np.array(splits == 'test', dtype=bool)

    # 三个 one-hot 标签张量，非对应位置的行保持全 0
    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    # ---- 5. 与 pickle 路径收尾方式完全一致 ----
    adj_raw = sp.coo_matrix(adj)                        # scipy 域副本（Chebyshev 用）
    adj_t = _to_torch_sparse(adj)                       # torch 域副本
    features = _to_torch_sparse(preprocess_features(features))   # 行归一化后再转 torch
    y_train = torch.from_numpy(y_train.astype(np.float32))       # 标签转 float32
    y_val = torch.from_numpy(y_val.astype(np.float32))
    y_test = torch.from_numpy(y_test.astype(np.float32))
    train_mask = torch.from_numpy(train_mask)                    # 布尔 mask 保持 bool
    val_mask = torch.from_numpy(val_mask)
    test_mask = torch.from_numpy(test_mask)

    # 返回结构与 load_data(pickle) 完全相同的 9 元组
    return (adj_t, features, y_train, y_val, y_test,
            train_mask, val_mask, test_mask, adj_raw)