"""PyG 在 Python 3.14 下的告警抑制。

**告警从哪来**
PyG 的 ``MessagePassing.__init__`` 会调用 ``inspect_signature`` 去解析类型注解，
最终走到 ``typing._eval_type``。在 Python 3.14 上这会抛
``DeprecationWarning``（Python 3.15 将彻底禁止）。
另外 ``torch.jit.script`` 在 Python 3.14+ 不受支持，PyG 导入时会抛
``FutureWarning``。

所以告警有**两个时机**：

============  ==========================================  ==========
时机          触发点                                        能否用 import 期拦截
============  ==========================================  ==========
导入期        ``import torch_geometric.nn``                   能
**运行期**    **``GCNConv(...)`` 实例化时**（inspect_signature）  **不能**
============  ==========================================  ==========

运行期那条才是主要麻烦——它发生在 ``main()`` 里的模型构造处。

**难点：``-W error`` 会压过 filterwarnings**
``filterwarnings('ignore', ...)`` 只是往过滤器列表**追加**一条。若用户用
``python -W error`` 运行，``error`` 过滤器位于列表最前，告警直接抛异常，
追加的 ignore 永远轮不到。

而 ``catch_warnings()`` 会**保存并重置**过滤器列表，块内用
``filterwarnings('ignore', ...)` 安装的 ignore 会排到最前、优先生效。

**本模块的方案**
:func:`silence` 是一个上下文管理器，用 ``catch_warnings`` 重置过滤器列表并
精确安装两条 ignore。进入它即可让运行期的告警也被压掉；退出后列表恢复，
其余告警照常上报——「零警告 = 真零警告」依然成立。

用法::

    from pyg_warn import silence

    with silence():
        model = GCNConv(1433, 16)     # 这里的 DeprecationWarning 被压掉
"""

import contextlib
import warnings

# 两条精确的匹配串（全项目仅此两条，不会误伤其它告警）
_TYPING_EVAL_MSG = r'.*Failing to pass a value to the .type_params. parameter.*'
_JIT_SCRIPT_MSG = r'.*torch\.jit\.script. is not supported in Python 3\.14.*'


def _install_precise():
    """在当前过滤器列表**最前**装上两条精确的 ignore。

    依赖调用方已处于 ``catch_warnings()`` 块内（列表已被重置，
    此时追加的过滤器会排在默认的 ``default`` 之前）。
    """
    warnings.filterwarnings('ignore', message=_TYPING_EVAL_MSG,
                            category=DeprecationWarning)
    warnings.filterwarnings('ignore', message=_JIT_SCRIPT_MSG,
                            category=FutureWarning)


@contextlib.contextmanager
def silence():
    """抑制 PyG 在 Python 3.14 下的两条已知告警。

    用法::

        with silence():
            model = GCNConv(...)
    """
    with warnings.catch_warnings():
        _install_precise()
        yield


# ---- 导入期：先装好 filter，再导入 PyG ----
# 顺序很重要——告警在 ``import torch_geometric.nn`` 执行的那一行就发出。
with warnings.catch_warnings():
    _install_precise()
    from torch_geometric.datasets import Planetoid            # noqa: F401
    from torch_geometric.loader import DataLoader              # noqa: F401
    from torch_geometric.nn import GCNConv                    # noqa: F401
    from torch_geometric.nn.conv.gcn_conv import gcn_norm     # noqa: F401
    from torch_geometric.transforms import NormalizeFeatures  # noqa: F401
    from torch_geometric.utils import add_self_loops           # noqa: F401
    from torch_geometric.utils import degree                  # noqa: F401

__all__ = ['Planetoid', 'DataLoader', 'GCNConv', 'NormalizeFeatures',
           'degree', 'gcn_norm', 'add_self_loops', 'silence']
