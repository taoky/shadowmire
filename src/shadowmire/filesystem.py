import logging
import os
import re
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any, Literal

logger = logging.getLogger(__name__)


@contextmanager
def overwrite(
    file_path: Path, mode: str = "w", tmp_suffix: str = ".tmp"
) -> Generator[IO[Any], None, None]:
    tmp_path = file_path.parent / (file_path.name + tmp_suffix)
    with open(tmp_path, mode) as tmp_file:
        yield tmp_file
    tmp_path.rename(file_path)


def fast_readall(file_path: Path) -> bytes:
    """
    Save some extra read(), lseek() and ioctl().
    """
    fd = os.open(file_path, os.O_RDONLY)
    if fd < 0:
        raise FileNotFoundError(file_path)
    try:
        contents = os.read(fd, file_path.stat().st_size)
        return contents
    finally:
        os.close(fd)


def normalize(name: str) -> str:
    """
    See https://peps.python.org/pep-0503/#normalized-names
    """
    return re.sub(r"[-_.]+", "-", name).lower()


def remove_dir_with_files(directory: Path) -> None:
    """
    Remove dir in a safer (non-recursive) way, which means that the directory should have no child directories.
    """
    if not directory.exists():
        return
    assert directory.is_dir()
    for item in directory.iterdir():
        item.unlink()
    directory.rmdir()
    logger.info("Removed dir %s", directory)


def fast_iterdir(
    directory: Path | str, filter_type: Literal["dir", "file"]
) -> Generator[os.DirEntry[str], Any, None]:
    """
    iterdir() in pathlib would ignore file type information from getdents64(),
    which is not acceptable when you have millions of files in one directory,
    and you need to filter out all files/directories.
    """
    assert filter_type in ["dir", "file"]
    for item in os.scandir(directory):
        if (
            filter_type == "dir"
            and item.is_dir()
            or filter_type == "file"
            and item.is_file()
        ):
            yield item
