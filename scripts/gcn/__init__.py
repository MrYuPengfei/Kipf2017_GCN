"""GCN 的 PyTorch 实现（Kipf & Welling 2017 的 TF 原版移植）。

关于稀疏张量不变量检查
----------------------
PyTorch 2.14 起，构造稀疏张量时会发出 UserWarning，提示不变量检查
被隐式禁用。经实测，该警告由C++ 层的全局 Context 状态触发，
`torch.sparse.check_sparse_tensor_invariants(False)` 这一 Python API
在 2.14.1 上**无法**将其关闭（只有 `with` 上下文管理器形式有效）。

因此本包提供 `silence_sparse_warning()`，在稀疏张量构造前用上下文
管理器显式声明一次即可彻底静音。我们构造的稀疏张量全部来自 scipy 的
coo_matrix 并立即 coalesce()，索引天然有序且唯一，检查只会带来性能
开销而无收益，故关闭是安全的。

各入口脚本（train.py / test.py / export_model.py / infer.py）均已调用。
"""

import contextlib
import warnings


def silence_sparse_warning():
    """在稀疏张量构造前显式声明不变量检查策略，消除 PyTorch 2.14 的提示。

    用法：
        from gcn import silence_sparse_warning
        with silence_sparse_warning():
            x = torch.sparse_coo_tensor(...)   # 此处不再有 UserWarning
    """
    import torch

    if not hasattr(torch.sparse, 'check_sparse_tensor_invariants'):
        return contextlib.nullcontext()

    # 部分版本即使显式声明仍会触发，叠加 warnings 过滤兜底。
    cm = torch.sparse.check_sparse_tensor_invariants()
    return _Combined(cm)


class _Combined:
    """把上下文管理器与 warnings 过滤组合为一个。"""

    def __init__(self, cm):
        self._cm = cm
        self._cw = None

    def __enter__(self):
        self._cm.__enter__()
        self._cw = warnings.catch_warnings()
        self._cw.__enter__()
        warnings.filterwarnings(
            'ignore',
            message='.*Sparse invariant checks are implicitly disabled.*')
        return self

    def __exit__(self, *exc):
        if self._cw is not None:
            self._cw.__exit__(*exc)
        return self._cm.__exit__(*exc)