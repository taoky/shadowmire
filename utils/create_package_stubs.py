#!/usr/bin/env python3
# This script is used to create EMPTY package files USED FOR DEBUG ONLY!
# It requires a full simple/ and db (genlocal-ed)
# Call like: python -m utils.create_package_stubs /path/to/pypi/

from urllib.parse import unquote
from shadowmire import LocalVersionKV, get_package_urls_size_from_index_json
from pathlib import Path
import sys
from tqdm import tqdm
from os.path import (
    normpath,
)  # fast path computation, instead of accessing real files like pathlib


if __name__ == "__main__":
    pass
    try:
        repo = sys.argv[1]
    except IndexError:
        print("Please give repo basedir as argv[1].")
        sys.exit(-1)
    basedir = Path(repo).resolve()
    local_db = LocalVersionKV(basedir / "local.db", basedir / "local.json")
    local_names = set(local_db.keys())
    simple_dir = basedir / "simple"
    for package_name in tqdm(local_names, desc="Creating stub"):
        package_simple_path = simple_dir / package_name
        json_simple = package_simple_path / "index.v1_json"
        hrefsize_json = get_package_urls_size_from_index_json(json_simple)
        for href, _ in hrefsize_json:
            relative = unquote(href)
            dest = Path(normpath(package_simple_path / relative))
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                dest.touch()
