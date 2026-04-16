from __future__ import annotations

from pathlib import Path
from typing import Any

from polymer_md.conversion.base import Converter, ConversionKey
from polymer_md.conversion.file_formats import FileFormats
from polymer_md.core.molecule_input import MoleculeInput


class ConversionRegistry:
    def __init__(self) -> None:
        self._registry: dict[ConversionKey, type[Converter]] = {}

    def register(self, converter_class: type[Converter]) -> None:
        for key in converter_class.supports():
            self._registry[key] = converter_class

    def find(self, source: Path | MoleculeInput, output_type: type | FileFormats) -> type[Converter]:
        key = self._make_key(source, output_type)
        if key not in self._registry:
            raise KeyError(
                f"No converter registered for {self._describe_source(source)} → {output_type}"
            )
        return self._registry[key]

    def can_convert(self, source: Path | MoleculeInput, output_type: type | FileFormats) -> bool:
        return self._make_key(source, output_type) in self._registry

    def supported_conversions(self) -> list[ConversionKey]:
        return list(self._registry.keys())

    @staticmethod
    def _make_key(source: Path | MoleculeInput, output_type: type | FileFormats) -> ConversionKey:
        if isinstance(source, Path):
            input_key: type | FileFormats = FileFormats(source.suffix.lstrip("."))
        else:
            input_key = type(source)
        return (input_key, output_type)

    @staticmethod
    def _describe_source(source: Any) -> str:
        if isinstance(source, Path):
            return f"Path[{source.suffix}]"
        return type(source).__name__


REGISTRY = ConversionRegistry()


def register_converter(cls: type[Converter]) -> type[Converter]:
    REGISTRY.register(cls)
    return cls
