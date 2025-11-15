#!/usr/bin/env python3

# Generate a full list from a curated package list (resolving all possible dependencies)
# Currently from PyPI

# Call with:
# python3 -m utils.generate_full_package_list curated_list.txt full_list.txt

import argparse
from pathlib import Path
import sys
import tempfile
import queue

from tqdm import tqdm
from packaging.requirements import Requirement

import shadowmire


def main(args):
    stub_local_db = shadowmire.LocalVersionKV(
        dbpath=args.tempdir / "local_db.sqlite", jsonpath=args.tempdir / "local_db.json"
    )
    if args.upstream:
        sync = shadowmire.SyncPlainHTTP(
            upstream=args.upstream, basedir=args.tempdir, local_db=stub_local_db
        )
    else:
        sync = shadowmire.SyncPyPI(basedir=args.tempdir, local_db=stub_local_db)
    full_package_list = set()

    input_queue = queue.Queue()
    with open(args.input_file, "r") as infile:
        for line in infile:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            input_queue.put(line)

    with tqdm(unit="packages") as pbar:
        while not input_queue.empty():
            package_name = input_queue.get()
            if package_name in full_package_list:
                continue
            full_package_list.add(package_name)
            pbar.update(1)
            pbar.set_description(f"Processing {package_name}")

            try:
                meta = sync.get_package_metadata(package_name)
                if meta["info"].get("requires_dist"):
                    for dep in meta["info"]["requires_dist"]:
                        req = Requirement(dep)
                        dep_name = req.name
                        if dep_name not in full_package_list:
                            input_queue.put(dep_name)
            except shadowmire.PackageNotFoundError:
                print(f"Warning: Package {package_name} not found. Skipping.", file=sys.stderr)
    
    with open(args.output_file, "w") as outfile:
        for pkg in sorted(full_package_list):
            outfile.write(f"{pkg}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a full package list from a curated list by resolving dependencies."
    )
    parser.add_argument("input_file", help="Path to the curated package list file")
    parser.add_argument("output_file", help="Path to save the full package list file")
    parser.add_argument(
        "--upstream",
        default="",
        help="A custom shadowmire upstream URL. PyPI if not set.",
    )
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tempdir:
        args.tempdir = Path(tempdir)
        main(args)
