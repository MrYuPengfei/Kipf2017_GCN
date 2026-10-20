"""PyG 内置数据集（`Planetoid`）的加载与本地缓存。

与项目里另一条数据路径的关系
----------------------------
本项目原本有两条「自己管数据」的路径，本包**都不使用、也不修改**：

    gcn/utils.py  load_data(source='pickle')   读 gcn/data/*.pickle
    gcn/utils.py  load_data(source='csv')      读 data/csv/*.csv

本包走第三条路：**交给 PyG 的 `torch_geometric.datasets.Planetoid` 处理**。
好处是省事——下载、解压、`test_idx_reorder` 重排、Citeseer 孤立节点补齐
全部由 PyG 负责，且它的 `processed/data.pt` 可以直接复用。

代价是**必须能拿到原始 `ind.*` 文件**。PyG 默认从
`github.com/kimiyoung/planetoid/raw/master/data` 下载，该地址实际会跳转到
`raw.githubusercontent.com`。在受限网络（如本机沙箱）里这个域名不可达，
因此本包提供一个镜像回退：从 `codeload.github.com` 取仓库 tar.gz
（同一个源头，只是走 github.com 的打包服务），再从中抽出 `ind.*`。

三种获取方式的优先级
--------------------
1. **已缓存**：`<root>/Planetoid/<Name>/raw/` 下 8 个 `ind.*` 齐全 → 直接用，不联网
2. **PyG 官方下载**：联网正常时由 `Planetoid.download()` 自己完成
3. **镜像回退**：官方域名不可达时，从 codeload 取 tar.gz 并抽出所需文件

无论走哪条，最终产物都落在 `<root>/Planetoid/<Name>/` 下：

    raw/ind.cora.{x,y,tx,ty,allx,ally,graph}   原始数据（8 文件）
    raw/ind.cora.test.index                   测试索引
    processed/data.pt                         PyG 处理后的 Data 对象

默认 root 为 `scripts/gcn-pytorch/data/Planetoid`，实际落盘为
`data/Planetoid/Cora/`、`data/Planetoid/Citeseer/`、`data/Planetoid/Pubmed/`，
与既有的 `gcn/data/`、`data/csv/` 并列，**互不影响**。
"""

import io
import os
import shutil
import tarfile
import urllib.request

# ------------------------------------------------------------------ 路径约定
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 与既有的 gcn/data/、data/csv/ 并列，互不干扰
DEFAULT_ROOT = os.path.join(PROJ_DIR, 'data', 'Planetoid')

# PyG 2.x 的目录布局是 <root>/<Name>/{raw,processed}（root 里不再自动加 'Planetoid'）
DATASET_DIRNAME = ''

# 原始文件清单（缺一不可）
RAW_SUFFIXES = ('x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph', 'test.index')

# PyG 官方下载地址（会 302 到 raw.githubusercontent.com）
OFFICIAL_BASE = 'https://github.com/kimiyoung/planetoid/raw/master/data'

# 镜像：github.com 的打包服务，不需要 raw.githubusercontent.com
MIRROR_TARBALL = ('https://codeload.github.com/kimiyoung/planetoid/'
                  'tar.gz/refs/heads/master')

# Planetoid 的名字映射（PyG 用首字母大写）
NAME_MAP = {'cora': 'Cora', 'citeseer': 'Citeseer', 'pubmed': 'Pubmed'}


def dataset_dir(name, root=DEFAULT_ROOT):
    """返回某数据集的目录 `<root>/<Name>/`（PyG 2.x 的实际布局）。"""
    return os.path.join(root, DATASET_DIRNAME, _pyg_name(name))


def raw_dir(name, root=DEFAULT_ROOT):
    """返回 raw 目录（原始 `ind.*` 文件所在处）。"""
    return os.path.join(dataset_dir(name, root), 'raw')


def _pyg_name(name):
    """'cora' -> 'Cora'，已经是首字母大写则原样返回。"""
    low = name.lower()
    if low not in NAME_MAP:
        raise ValueError('未知数据集 %r，可选: %s' % (name, list(NAME_MAP)))
    return NAME_MAP[low]


def _raw_paths(name, root):
    """8 个原始文件的完整路径列表。"""
    d = raw_dir(name, root)
    short = _pyg_name(name).lower()
    return [os.path.join(d, 'ind.%s.%s' % (short, s)) for s in RAW_SUFFIXES]


def raw_ready(name, root=DEFAULT_ROOT):
    """原始文件是否已全部就位（就位则 PyG 不会触发下载）。"""
    return all(os.path.exists(p) for p in _raw_paths(name, root))


def processed_ready(name, root=DEFAULT_ROOT):
    """PyG 处理后的 data.pt 是否已生成。"""
    return os.path.exists(os.path.join(dataset_dir(name, root),
                                       'processed', 'data.pt'))


# ------------------------------------------------------------ 镜像回退下载
def _fetch_from_mirror(name, root=DEFAULT_ROOT, timeout=900):
    """从 codeload 取 planetoid 仓库 tar.gz，抽出所需的 `ind.*` 文件。

    只用标准库（urllib + tarfile），不引入额外依赖。

    **一次下载抽出三个数据集**：tar.gz 里本来就有 cora/citeseer/pubmed，
    如果每次只抽一个数据集，就要为三个数据集重复下载同一个 6 MB 包
    （实测单次约 160 秒，三次就是 8 分钟）。这里一次性铺开，
    后续再加载别的数据集就直接命中缓存。

    :return: 本次为**请求的数据集**抽出的文件数
    """
    dest_of = {n.lower(): raw_dir(n.lower(), root)
               for n in NAME_MAP.values()}
    for d in dest_of.values():
        os.makedirs(d, exist_ok=True)

    req = urllib.request.Request(
        MIRROR_TARBALL, headers={'User-Agent': 'gcn-pytorch/1.0'})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        blob = resp.read()

    # 直接在内存里解 tar，避免把 6 MB 压缩包和解压目录写进项目
    want = {'ind.%s.%s' % (ds, suf)
            for ds in dest_of for suf in RAW_SUFFIXES}
    written = []
    with tarfile.open(fileobj=io.BytesIO(blob), mode='r:gz') as tar:
        for member in tar.getmembers():
            base = os.path.basename(member.name)
            if base not in want or not member.isfile():
                continue
            ds = base.split('.')[1]          # ind.<ds>.<suffix>
            src = tar.extractfile(member)
            if src is None:
                continue
            with open(os.path.join(dest_of[ds], base), 'wb') as out:
                shutil.copyfileobj(src, out)
            written.append(base)

    short = _pyg_name(name).lower()
    missing = ['ind.%s.%s' % (short, suf) for suf in RAW_SUFFIXES
               if not os.path.exists(os.path.join(dest_of[short],
                                                  'ind.%s.%s' % (short, suf)))]
    if missing:
        raise RuntimeError('镜像中缺少文件: %s' % missing)

    return sum(1 for b in written if b.startswith('ind.%s.' % short))


# ---------------------------------------------------------------- 对外入口
def cache_report(name=None, root=DEFAULT_ROOT):
    """列出本地缓存情况，用于证明「数据确实落在本地」。

    :param name: 指定数据集；None 表示三个都查
    :return: 行列表，每行 [数据集, raw 文件数, processed, 占用]
    """
    names = [_pyg_name(name)] if name else ['Cora', 'Citeseer', 'Pubmed']
    rows = []
    for n in names:
        low = n.lower()
        d = os.path.join(root, DATASET_DIRNAME, n)
        n_raw = sum(1 for p in _raw_paths(low, root) if os.path.exists(p))
        proc = '✓' if processed_ready(low, root) else '—'
        size = 0
        for dirpath, _dirnames, filenames in os.walk(d):
            for f in filenames:
                size += os.path.getsize(os.path.join(dirpath, f))
        rows.append([n, '%d/8' % n_raw, proc,
                     '%.1f MB' % (size / 1024 / 1024) if size else '0'])
    return rows


def ensure_raw(name, root=DEFAULT_ROOT, allow_network=True, verbose=True):
    """保证 `<root>/Planetoid/<Name>/raw/` 下的 8 个原始文件存在。

    :return: 'cached' | 'mirror'
    :raises RuntimeError: 三种方式都失败时，附带手工放置文件的说明
    """
    if raw_ready(name, root):
        if verbose:
            print('[cache] %s 原始文件已就位，跳过下载' % _pyg_name(name))
        return 'cached'

    os.makedirs(raw_dir(name, root), exist_ok=True)

    if allow_network:
        # 先试官方地址：只要有一个文件下下来就说明域名可达，
        # 剩下的交给 PyG 自己的 download() 去补
        try:
            if verbose:
                print('[net] 尝试官方地址 %s ...' % OFFICIAL_BASE)
            _probe_official(name, root)
            if raw_ready(name, root):
                return 'cached'
        except Exception as ex:                       # noqa: BLE001
            if verbose:
                print('[net] 官方地址不可用（%s），改用镜像' % type(ex).__name__)

        if verbose:
            print('[net] 从 codeload 镜像获取 ...')
        _fetch_from_mirror(name, root)
        if verbose:
            print('[mirror] 已抽出 %d 个文件到 %s' % (len(RAW_SUFFIXES),
                                                     raw_dir(name, root)))
        return 'mirror'

    raise RuntimeError(
        '缺少原始文件且不允许联网。请手工把 ind.%s.* 共 8 个文件放到:\n  %s\n'
        '来源: https://github.com/kimiyoung/planetoid/tree/master/data'
        % (_pyg_name(name).lower(), raw_dir(name, root)))


def _probe_official(name, root):
    """用官方地址试下第一个文件；成功就逐个补齐。

    单独写而不用 PyG 的 download()，是为了在失败时能立刻回退到镜像，
    不被 aiohttp 的长超时拖住。
    """
    short = _pyg_name(name).lower()
    dest = raw_dir(name, root)
    for suffix in RAW_SUFFIXES:
        path = os.path.join(dest, 'ind.%s.%s' % (short, suffix))
        if os.path.exists(path):
            continue
        url = '%s/ind.%s.%s' % (OFFICIAL_BASE, short, suffix)
        req = urllib.request.Request(
            url, headers={'User-Agent': 'gcn-pytorch/1.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
        if not data:
            raise RuntimeError('空响应: %s' % url)
        with open(path, 'wb') as f:
            f.write(data)


def load(name, root=DEFAULT_ROOT, normalize=True, allow_network=True,
         verbose=True):
    """加载 PyG 内置的 Planetoid 数据集，必要时先落盘到本地。

    :param name: 'cora' / 'citeseer' / 'pubmed'
    :param root: 缓存根目录，默认 `data/`
    :param normalize: 是否套 `NormalizeFeatures()`（行归一化）。
        项目里既有的 `preprocess_features` 也是行归一化，置 True 才能与之可比。
    :param allow_network: 允许联网获取原始文件
    :return: (dataset, data, info)
        ``dataset`` 是 PyG 的 Dataset（含 num_classes 等）
        ``data``    是单个图（单图数据集，`dataset[0]`）
        ``info``    记录来源方式、缓存路径等，便于日志与测试
    """
    from torch_geometric.datasets import Planetoid
    from torch_geometric.transforms import NormalizeFeatures

    mode = ensure_raw(name, root, allow_network=allow_network, verbose=verbose)

    transform = NormalizeFeatures() if normalize else None
    ds = Planetoid(root=root, name=_pyg_name(name), transform=transform)
    data = ds[0]

    info = {
        'name': _pyg_name(name),
        'mode': mode,
        'root': dataset_dir(name, root),
        'raw_dir': raw_dir(name, root),
        'processed': os.path.join(dataset_dir(name, root),
                                  'processed', 'data.pt'),
        'normalized': normalize,
        'num_nodes': int(data.num_nodes),
        'num_features': int(ds.num_features),
        'num_classes': int(ds.num_classes),
        'num_edges': int(data.edge_index.shape[1]),
        'splits': (int(data.train_mask.sum()), int(data.val_mask.sum()),
                   int(data.test_mask.sum())),
    }
    return ds, data, info


def describe(data, num_classes=None):
    """一行摘要，日志与断言共用。

    ``num_classes`` 是 **Dataset** 的属性，不在 Data 上，需要显式传入；
    不传时退化为从 ``y`` 推断。
    """
    if num_classes is None:
        num_classes = (int(data.y.max()) + 1 if data.y.numel() else 0)
    return ('nodes=%d features=%d edges=%d classes=%d '
            'train/val/test=%d/%d/%d' % (
                data.num_nodes, data.num_features,
                data.edge_index.shape[1], num_classes,
                int(data.train_mask.sum()), int(data.val_mask.sum()),
                int(data.test_mask.sum())))
