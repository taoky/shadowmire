#!/usr/bin/env python3
# This script is used to create EMPTY package files USED FOR DEBUG ONLY!
# It requires a full simple/ and db (genlocal-ed)
# Call like: python -m utils.create_package_stubs /path/to/pypi/

from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import unquote
from pathlib import Path
import sys
import os
from os.path import (
    normpath,
)  # fast path computation, instead of accessing real files like pathlib

from tqdm import tqdm
from shadowmire import LocalVersionKV, get_package_urls_size_from_index_json

IOWORKERS = int(os.environ.get("SHADOWMIRE_IOWORKERS", "2"))

if __name__ == "__main__":
    try:
        repo = sys.argv[1]
    except IndexError:
        print("Please give repo basedir as argv[1].")
        sys.exit(-1)
    basedir = Path(repo).resolve()
    local_db = LocalVersionKV(basedir / "local.db", basedir / "local.json")
    local_names = set(local_db.keys())
    simple_dir = basedir / "simple"
    with ThreadPoolExecutor(max_workers=IOWORKERS) as executor:

        def handle(package_name: str) -> None:
            package_simple_path = simple_dir / package_name
            json_simple = package_simple_path / "index.v1_json"
            hrefsize_json = get_package_urls_size_from_index_json(json_simple)
            for href, _ in hrefsize_json:
                relative = unquote(href)
                dest = Path(normpath(package_simple_path / relative))
                if not dest.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    fd = os.open(dest, os.O_CREAT, 0o664)
                    os.close(fd)

        futures = {
            executor.submit(handle, package_name): package_name
            for package_name in local_names
        }
        for future in tqdm(
            as_completed(futures),
            total=len(local_names),
            desc="Creating stub",
        ):
            future.result()
