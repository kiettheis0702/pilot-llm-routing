"""Read only hash-pinned RouterBench pickles; this is not a general pickle loader."""
from __future__ import annotations

import pickle
from pathlib import Path

try:
    from scripts.fetch_data import verify
except ModuleNotFoundError:
    from fetch_data import verify


ALLOWED_GLOBALS = {
    ("pandas.core.frame", "DataFrame"),
    ("pandas.core.internals.managers", "BlockManager"),
    ("pandas._libs.internals", "_unpickle_block"),
    ("pandas.core.indexes.base", "_new_Index"),
    ("pandas.core.indexes.base", "Index"),
    ("pandas.core.indexes.range", "RangeIndex"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy.core.numeric", "_frombuffer"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy._core.numeric", "_frombuffer"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("builtins", "slice"),
}


class RouterBenchUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if (module, name) not in ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(f"Disallowed pickle global: {module}.{name}")
        return super().find_class(module, name)


def load_routerbench(path: Path, source: dict):
    verify(path, source)
    with path.open("rb") as stream:
        return RouterBenchUnpickler(stream).load()
