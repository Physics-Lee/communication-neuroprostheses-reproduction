# -*- coding: utf-8 -*-
"""forcedAlignmentHMM 的 numpy 2.x 兼容垫片（同名模块，供 spawn 的多进程 worker 也能命中）。

背景：官方 forcedAlignmentHMM.py 在 numpy 1.x 时代写成；numpy 2.x 起 np.arange 收到
size-1 非 0 维数组直接 TypeError（makeTimeWindowMask 的 timeFraction 计算）。
Windows 多进程为 spawn，worker 重新 import 模块，主进程 monkeypatch 不生效，
因此本文件以同名模块方式垫在 code/（调用方 sys.path 里 code/ 在官方仓库之前）。

做法：exec 官方源码到本模块命名空间，再替换 makeTimeWindowMask 为 squeeze 版。
官方源码本身一字未改（只读引用）。
"""
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_OFFICIAL = os.path.join(_HERE, '..', 'open_source_code_for_reference',
                         '20210512_willett_handwritingBCI', 'forcedAlignmentHMM.py')

with open(_OFFICIAL, encoding='utf-8') as _f:
    exec(compile(_f.read(), _OFFICIAL, 'exec'), globals())


class _NpArangeProxy:
    """numpy 2.x 兼容：官方代码多处给 np.arange 传 size-1 数组（numpy 1.x 隐式 float()
    通过，2.x 抛 TypeError）。代理仅对 arange 参数做 float() 强转，其余属性透传。
    官方代码只会传标量或 size-1 数组，强转语义与 numpy 1.x 完全一致。

    Parameters
    ----------
    np_mod : module
        The real ``numpy`` module to wrap; every attribute except ``arange``
        is forwarded to it unchanged.
    """

    def __init__(self, np_mod):
        """Store the wrapped numpy module, bypassing ``__getattr__`` recursion.

        Parameters
        ----------
        np_mod : module
            The real ``numpy`` module to proxy.
        """
        object.__setattr__(self, '_np', np_mod)

    def __getattr__(self, name):
        """Forward any attribute access to the wrapped numpy module.

        Parameters
        ----------
        name : str
            Attribute name being looked up on the proxy.

        Returns
        -------
        Any
            The value of ``getattr(numpy, name)`` from the wrapped module.
        """
        return getattr(object.__getattribute__(self, '_np'), name)

    def arange(self, *args, **kwargs):
        """numpy.arange with numpy-1.x-style coercion of size-1 array arguments.

        Positional arguments that are numpy scalars or size-1 arrays are
        converted with ``float()`` before being forwarded to the real
        ``numpy.arange``; everything else (and all keyword arguments) is
        passed through unchanged.

        Parameters
        ----------
        *args : Any
            Arguments to ``numpy.arange`` (start, stop, step, dtype...).
            Each ndarray argument is flattened and its first element taken
            (official code only ever passes scalars or size-1 arrays).
        **kwargs : Any
            Keyword arguments forwarded verbatim to ``numpy.arange``.

        Returns
        -------
        numpy.ndarray
            The result of the real ``numpy.arange`` on the coerced arguments.
        """
        real = object.__getattribute__(self, '_np')
        coerced = []
        for a in args:
            if isinstance(a, np.ndarray):
                a = float(real.asarray(a).reshape(-1)[0])
            elif isinstance(a, np.generic):
                a = float(a)
            coerced.append(a)
        return real.arange(*coerced, **kwargs)


globals()['np'] = _NpArangeProxy(np)
globals()['numpy'] = globals()['np']  # 官方文件里 `import numpy as np` 之外若有直接引用 numpy 的也一并代理
