from pathlib import Path
from typing import Set, Union

PathType = Union[str, Path]


class FileHelper:
    @staticmethod
    def assert_suffix_type(path: PathType, suffix: str) -> None:
        path = Path(path)
        suffix = FileHelper.normalise_suffix_with_period(suffix=suffix)
        assert path.suffix == suffix

    @staticmethod
    def overwrite_suffix(path: PathType, suffix: str) -> Path:
        path = Path(path)
        return path.with_suffix(FileHelper.normalise_suffix_with_period(suffix=suffix))

    @staticmethod
    def normalise_suffix_with_period(suffix: str) -> str:
        if "." in suffix:
            return suffix
        return f".{suffix}"

    @staticmethod
    def assert_suffix_is_supported(
        path: PathType, supported_suffixes: Set[str]
    ) -> None:
        path = Path(path)
        normalised_suffixes = set(
            [
                FileHelper.normalise_suffix_with_period(suffix=suffix)
                for suffix in supported_suffixes
            ]
        )
        if path.stem not in normalised_suffixes:
            raise TypeError(
                f"[UNSUPPORTED_FILE_FORMAT]: Path {path} is of unsupported type {path.stem}. Supported formats: {', '.join(normalised_suffixes)} "
            )

    @staticmethod
    def safe_get_suffix_type(path: PathType, supported_suffixes: Set[str]) -> str:
        FileHelper.assert_suffix_is_supported(
            path=path, supported_suffixes=supported_suffixes
        )
        return FileHelper.get_suffix_type(path=path)

    @staticmethod
    def get_suffix_type(path: PathType) -> str:
        path = Path(path)
        suffix_with_period = path.suffix
        return suffix_with_period.split(".")[-1]

    @staticmethod
    def construct_path(
        path_dir: PathType,
        stem: str,
        suffix: str,
    ) -> Path:
        path_dir = Path(path_dir)
        path_dir.mkdir(exist_ok=True)
        return path_dir / f"{stem}.{suffix}"
