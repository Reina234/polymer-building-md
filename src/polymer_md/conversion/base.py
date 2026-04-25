from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Callable, overload

from polymer_md.conversion.file_formats import FileFormats
from polymer_md.core.molecule_input import MoleculeInput

ConversionKey = tuple[type | FileFormats, type | FileFormats]


class _Handler:
    def __init__(
        self,
        func: Callable,
        input_type: type | FileFormats,
        output_type: type | FileFormats,
    ) -> None:
        self.func = func
        self.input_type = input_type
        self.output_type = output_type

    def __set_name__(self, owner: type, _name: str) -> None:
        if "_own_handlers" not in owner.__dict__:
            owner._own_handlers = {}
        owner._own_handlers[(self.input_type, self.output_type)] = self

    @overload
    def __get__(self, obj: None, _owner_class: type | None = None) -> _Handler: ...

    @overload
    def __get__(self, obj: Any, _owner_class: type | None = None) -> Callable: ...

    def __get__(
        self, obj: Any, _owner_class: type | None = None
    ) -> _Handler | Callable:
        if obj is None:
            return self
        return functools.partial(self.func, obj)


def handles(
    input_type: type | FileFormats,
    output_type: type | FileFormats,
) -> Callable[[Callable], _Handler]:
    def decorator(func: Callable) -> _Handler:
        return _Handler(func, input_type, output_type)

    return decorator


class Converter:
    def convert(
        self,
        source: Path | MoleculeInput,
        output_type: type | FileFormats,
        output_dir: Path,
        output_name: str,
        overwrite: bool = True,
    ) -> Any:
        handler = self._resolve_handler(source, output_type)
        return handler(source, output_dir, output_name, overwrite)

    @classmethod
    def supports(cls) -> list[ConversionKey]:
        return list(cls._handler_map().keys())

    def _resolve_handler(
        self,
        source: Path | MoleculeInput,
        output_type: type | FileFormats,
    ) -> Callable:
        key = self._make_key(source, output_type)
        handler = self._handler_map().get(key)
        if handler is None:
            raise ValueError(
                f"No handler for {self._describe_source(source)} → {output_type}"
            )
        return handler.__get__(self, type(self))

    @classmethod
    def _handler_map(cls) -> dict[ConversionKey, _Handler]:
        result: dict[ConversionKey, _Handler] = {}
        for base_class in reversed(cls.__mro__):
            result.update(getattr(base_class, "_own_handlers", {}))
        return result

    @staticmethod
    def _make_key(
        source: Path | MoleculeInput,
        output_type: type | FileFormats,
    ) -> ConversionKey:
        if isinstance(source, Path):
            input_key: type | FileFormats = FileFormats(source.suffix.lstrip("."))
        else:
            input_key = type(source)
        return (input_key, output_type)

    @staticmethod
    def _describe_source(source: Path | MoleculeInput) -> str:
        if isinstance(source, Path):
            return f"Path[{source.suffix}]"
        return type(source).__name__
