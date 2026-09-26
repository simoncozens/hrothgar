"""Stub for the uharfbuzz C extension.

uharfbuzz ships as a compiled ``_harfbuzz`` module re-exported through
``uharfbuzz/__init__.py``, with no ``py.typed`` marker or stubs. Without this
stub basedpyright cannot see the re-exported names (``Font``, ``Face``,
``Blob``, ...) and reports every use as an unknown attribute.

The extension is not worth modelling precisely here; the module-level
``__getattr__`` tells the type checker that any attribute exists and is
untyped.
"""

from typing import Any

def __getattr__(name: str) -> Any: ...
