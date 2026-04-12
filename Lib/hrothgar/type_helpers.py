from collections.abc import Iterator, Iterable
from typing import Generic, TypeVar, overload

from torch import nn

T = TypeVar("T", bound=nn.Module)


class TypedModuleList(Generic[T], nn.ModuleList):
    """
    A type-annotated version of nn.ModuleList that preserves the type of its elements.
    """

    def __init__(self, modules: Iterable[T] | None = None) -> None:
        super().__init__(modules)

    def __iter__(self) -> Iterator[T]:
        return super().__iter__()  # type: ignore[no-any-return]

    def append(self, module: T) -> "TypedModuleList[T]":  # type: ignore[override]
        return super().append(module)  # type: ignore[return-value]

    @overload
    def __getitem__(self, idx: slice) -> "TypedModuleList[T]": ...

    @overload
    def __getitem__(self, idx: int) -> T: ...

    def __getitem__(self, idx):  # type: ignore[no-untyped-def]
        return super().__getitem__(idx)

    def __setitem__(self, idx: int, module: T) -> None:  # type: ignore[override]
        super().__setitem__(idx, module)
