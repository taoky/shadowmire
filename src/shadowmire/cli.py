import functools
import json
import logging
import os
import signal
import sys
import tomllib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from os.path import normpath
from pathlib import Path
from types import FrameType
from typing import Any
from urllib.parse import unquote

import click
from tqdm import tqdm

from . import __version__
from .constants import IOWORKERS, LOCAL_DB_NAME, LOCAL_JSON_NAME, WORKERS
from .database import LocalVersionKV
from .errors import ExitProgramException, exit_with_futures
from .filesystem import fast_iterdir, fast_readall, overwrite
from .filters import (
    PACKAGE_FILTER,
    FileInclusionChecker,
    PackageInclusionChecker,
)
from .simple import get_existing_hrefs
from .sync.base import SyncBase
from .sync.plain_http import SyncPlainHTTP
from .sync.pypi import SyncPyPI

LOG_FORMAT = "%(asctime)s %(levelname)s: %(message)s (%(filename)s:%(lineno)d)"
logger = logging.getLogger("shadowmire")


def exit_handler(signum: int, frame: FrameType | None) -> None:
    raise ExitProgramException


def get_local_serial(package_meta_direntry: os.DirEntry[str]) -> int | None:
    """
    Accepts /json/<package_name> as package_meta_path
    """
    package_name = package_meta_direntry.name
    try:
        contents = fast_readall(Path(package_meta_direntry.path))
    except FileNotFoundError:
        logger.warning("%s does not have JSON metadata, skipping", package_name)
        return None
    try:
        meta = json.loads(contents)
        return meta["last_serial"]
    except Exception:
        logger.warning("cannot parse %s's JSON metadata", package_name, exc_info=True)
        return None


def sync_shared_args(func: Callable[..., Any]) -> Callable[..., Any]:
    shared_options = [
        click.option(
            "--sync-packages/--no-sync-packages",
            default=False,
            help="Sync packages instead of just indexes, by default it's --no-sync-packages",
        ),
        click.option(
            "--shadowmire-upstream",
            required=False,
            type=str,
            help="Use another upstream using shadowmire instead of PyPI",
        ),
        click.option(
            "--use-pypi-index/--no-use-pypi-index",
            default=False,
            help="Always use PyPI index metadata (via XMLRPC). It's a no-op without --shadowmire-upstream. Some packages might not be downloaded successfully. Defaults to false.",
        ),
        click.option(
            "--exclude",
            multiple=True,
            help="Remote package names to exclude (regex patterns).",
        ),
        click.option(
            "--include",
            multiple=True,
            help="Only include these remote package names (regex patterns). --include has higher priority than --exclude.",
        ),
        click.option(
            "--package-filter",
            "package_filters",
            multiple=True,
            type=PACKAGE_FILTER,
            help="Apply an ordered package filter in ACTION:PATTERN format. ACTION is include, metadata-only, or exclude. The first match wins; unmatched packages are included.",
        ),
        click.option(
            "--prerelease-exclude",
            multiple=True,
            help="Package names of which prereleases will be excluded (regex patterns).",
        ),
        click.option(
            "--excluded-wheel-filename",
            multiple=True,
            help="Specify patterns to exclude wheel files (applies to all packages, regex patterns).",
        ),
        click.option(
            "--filter-metadata/--no-filter-metadata",
            default=True,
            help="Whether to modify each package's metadata according to release and file filtering rules. Defaults to true.",
        ),
        click.option(
            "--skip-yanked/--no-skip-yanked",
            default=False,
            help="Whether to skip yanked release files when syncing packages. Defaults to false.",
        ),
        click.option(
            "--skip-old-packages-days",
            default=None,
            type=int,
            help="Skip files whose upload time is earlier than the specified number of days. Defaults to None (do not skip any).",
        ),
        click.option(
            "--least-releases-to-keep",
            default=0,
            type=int,
            help="If --skip-old-packages-days ignores too many releases, at least keep this many latest releases while respecting other rules. Defaults to 0 (do not enforce).",
        ),
    ]

    @functools.wraps(func)
    @click.pass_context
    def wrapper(ctx: click.Context, *args, **kwargs):
        package_filters = kwargs.pop("package_filters")
        exclude = kwargs.pop("exclude")
        include = kwargs.pop("include")
        if package_filters and (exclude or include):
            raise click.UsageError(
                "--package-filter/package_filters cannot be used together with --include/include or --exclude/exclude"
            )
        package_inclusion_checker = PackageInclusionChecker(
            exclude=exclude,
            include=include,
            package_filters=package_filters,
        )
        file_inclusion_checker = FileInclusionChecker(
            prerelease_exclude=kwargs.pop("prerelease_exclude"),
            excluded_wheel_filename=kwargs.pop("excluded_wheel_filename"),
            filter_meta=kwargs.pop("filter_metadata"),
            skip_yanked=kwargs.pop("skip_yanked"),
            skip_old_packages_days=kwargs.pop("skip_old_packages_days"),
            least_releases_to_keep=kwargs.pop("least_releases_to_keep"),
        )
        kwargs["package_inclusion_checker"] = package_inclusion_checker
        kwargs["file_inclusion_checker"] = file_inclusion_checker
        return ctx.invoke(func, *args, **kwargs)

    decorated = wrapper
    for opt in reversed(shared_options):
        decorated = opt(decorated)
    return decorated


def read_config(ctx: click.Context, param: click.Option, filename: str | None) -> None:
    # Set default repo as cwd
    ctx.default_map = {}
    ctx.default_map["repo"] = "."

    if filename is None:
        return
    with open(filename, "rb") as f:
        data = tomllib.load(f)
    try:
        options = dict(data["options"])
    except KeyError:
        options = {}
    if options.get("repo"):
        ctx.default_map["repo"] = options["repo"]
        del options["repo"]

    logger.info("Read options from %s: %s", filename, options)

    ctx.default_map["sync"] = options
    ctx.default_map["verify"] = options
    ctx.default_map["do-update"] = options
    ctx.default_map["do-remove"] = options


@click.group()
@click.version_option(version=__version__)
@click.option(
    "--config",
    type=click.Path(dir_okay=False),
    help="Read option defaults from specified TOML file",
    callback=read_config,
    expose_value=False,
)
@click.option("--repo", type=click.Path(file_okay=False), help="Repo (basedir) path")
@click.pass_context
def cli(ctx: click.Context, repo: str) -> None:
    log_level = logging.DEBUG if os.environ.get("DEBUG") else logging.INFO
    logging.basicConfig(level=log_level, format=LOG_FORMAT)
    ctx.ensure_object(dict)

    if WORKERS > 10:
        logger.warning(
            "You have set a worker value larger than 10, which is forbidden by PyPI maintainers."
        )
        logger.warning("Don't blame me if you were banned!")

    # Make sure basedir is absolute
    basedir = Path(repo).resolve()
    basedir.mkdir(parents=True, exist_ok=True)
    local_db = LocalVersionKV(basedir / LOCAL_DB_NAME, basedir / LOCAL_JSON_NAME)
    ctx.obj["basedir"] = basedir
    ctx.obj["local_db"] = local_db


def get_syncer(
    basedir: Path,
    local_db: LocalVersionKV,
    sync_packages: bool,
    shadowmire_upstream: str | None,
    use_pypi_index: bool,
) -> SyncBase:
    syncer: SyncBase
    if shadowmire_upstream:
        syncer = SyncPlainHTTP(
            upstream=shadowmire_upstream,
            basedir=basedir,
            local_db=local_db,
            sync_packages=sync_packages,
            use_pypi_index=use_pypi_index,
        )
    else:
        syncer = SyncPyPI(
            basedir=basedir, local_db=local_db, sync_packages=sync_packages
        )
    return syncer


@cli.command(help="Sync from upstream")
@click.pass_context
@sync_shared_args
@click.option(
    "--reconcile-package-files/--no-reconcile-package-files",
    default=False,
    help="Scan existing projects to apply include/metadata-only filter changes to package files. Defaults to false.",
)
@click.option(
    "--dry-run/--no-dry-run",
    default=False,
    help="Print the synchronization plan as JSON without applying it.",
)
def sync(
    ctx: click.Context,
    sync_packages: bool,
    shadowmire_upstream: str | None,
    package_inclusion_checker: PackageInclusionChecker,
    file_inclusion_checker: FileInclusionChecker,
    use_pypi_index: bool,
    reconcile_package_files: bool,
    dry_run: bool,
) -> None:
    basedir: Path = ctx.obj["basedir"]
    local_db: LocalVersionKV = ctx.obj["local_db"]
    syncer = get_syncer(
        basedir, local_db, sync_packages, shadowmire_upstream, use_pypi_index
    )
    local = local_db.dump(skip_invalid=False)
    local_file_serials = local_db.dump_file_serials(skip_invalid=False)
    plan = syncer.determine_sync_plan(
        local,
        package_inclusion_checker,
        reconcile_package_files=reconcile_package_files,
        local_file_serials=local_file_serials,
    )
    plan_json = json.dumps(plan, default=vars, indent=2)
    if dry_run:
        click.echo(plan_json)
        return

    # save plan for debugging
    with overwrite(basedir / "plan.json") as f:
        f.write(plan_json)
    success = syncer.do_sync_plan(
        plan, package_inclusion_checker, file_inclusion_checker
    )
    syncer.finalize(plan.remote_last_serial)

    logger.info("Synchronization finished. Success: %s", success)

    if not success:
        sys.exit(1)


@cli.command(help="(Re)generate local db and json from json/")
@click.pass_context
def genlocal(ctx: click.Context) -> None:
    basedir: Path = ctx.obj["basedir"]
    local_db: LocalVersionKV = ctx.obj["local_db"]
    local = {}
    json_dir = basedir / "json"
    logger.info("Iterating all items under %s", json_dir)
    dir_items = [d for d in fast_iterdir(json_dir, "file")]
    logger.info("Detected %s packages in %s in total", len(dir_items), json_dir)
    with ThreadPoolExecutor(max_workers=IOWORKERS) as executor:
        futures = {
            executor.submit(get_local_serial, package_metapath): package_metapath
            for package_metapath in dir_items
        }
        try:
            for future in tqdm(
                as_completed(futures),
                total=len(dir_items),
                desc="Reading packages from json/",
            ):
                package_name = futures[future].name
                try:
                    serial = future.result()
                    if serial:
                        local[package_name] = serial
                except Exception as e:
                    if isinstance(e, (KeyboardInterrupt)):
                        raise
                    logger.warning(
                        "%s generated an exception", package_name, exc_info=True
                    )
        except (ExitProgramException, KeyboardInterrupt):
            exit_with_futures(futures)
    logger.info(
        "%d out of %d packages have valid serial number", len(local), len(dir_items)
    )
    local_db.nuke(commit=False)
    local_db.batch_set(local)
    local_db.dump_json()


@cli.command(
    help="Verify existing sync from local db, download missing things, remove unreferenced packages"
)
@click.pass_context
@sync_shared_args
@click.option(
    "--remove-not-in-local", is_flag=True, help="Do step 1 instead of skipping"
)
@click.option(
    "--compare-size",
    is_flag=True,
    help="Instead of just check if it exists, also compare local package size when possible, to decide if local package file is valid",
)
def verify(
    ctx: click.Context,
    sync_packages: bool,
    shadowmire_upstream: str | None,
    package_inclusion_checker: PackageInclusionChecker,
    file_inclusion_checker: FileInclusionChecker,
    remove_not_in_local: bool,
    compare_size: bool,
    use_pypi_index: bool,
) -> None:
    basedir: Path = ctx.obj["basedir"]
    local_db: LocalVersionKV = ctx.obj["local_db"]
    syncer = get_syncer(
        basedir, local_db, sync_packages, shadowmire_upstream, use_pypi_index
    )

    logger.info("====== Step 1. Remove packages NOT in local db ======")
    local_names = set(local_db.keys())
    simple_dirs = {i.name for i in fast_iterdir((basedir / "simple"), "dir")}
    json_files = {i.name for i in fast_iterdir((basedir / "json"), "file")}
    not_in_local = (simple_dirs | json_files) - local_names
    logger.info(
        "%d out of %d local packages NOT in local db",
        len(not_in_local),
        len(local_names),
    )
    for package_name in not_in_local:
        logger.info("package %s not in local db", package_name)
        if remove_not_in_local:
            # Old bandersnatch would download packages without normalization,
            # in which case one package file could have multiple "packages"
            # with different names, but normalized to the same one.
            # So, when in verify, we always set remove_packages=False
            # In step 4 unreferenced files would be removed, anyway.
            syncer.do_remove(package_name, remove_packages=False)

    logger.info("====== Step 2. Remove packages NOT in remote index ======")
    local = local_db.dump(skip_invalid=False)
    plan = syncer.determine_sync_plan(local, package_inclusion_checker)
    logger.info(
        "%s packages NOT in remote index -- this might contain packages that also do not exist locally",
        len(plan.remove),
    )
    for package_name in plan.remove:
        # We only take the plan.remove part here
        logger.info("package %s not in remote index", package_name)
        syncer.do_remove(package_name, remove_packages=False)

    # After some removal, local_names is changed.
    local_names = set(local_db.keys())

    logger.info("====== Step 3. Caching packages/ dirtree in memory for Step 4 & 5.")
    packages_pathcache: set[str] = set()
    with ThreadPoolExecutor(max_workers=IOWORKERS) as executor:

        def packages_iterate(first_dirname: str, position: int) -> list[str]:
            with tqdm(
                desc=f"Iterating packages/{first_dirname}/*/*/*", position=position
            ) as pb:
                res = []
                for d1 in fast_iterdir(basedir / "packages" / first_dirname, "dir"):
                    for d2 in fast_iterdir(d1.path, "dir"):
                        for file in fast_iterdir(d2.path, "file"):
                            pb.update(1)
                            res.append(file.path)
                return res

        futures = {
            executor.submit(
                packages_iterate, first_dir.name, idx % IOWORKERS
            ): first_dir.name
            for idx, first_dir in enumerate(fast_iterdir((basedir / "packages"), "dir"))
        }
        try:
            for future in as_completed(futures):
                sname = futures[future]
                try:
                    for p in future.result():
                        packages_pathcache.add(p)
                except Exception as e:
                    if isinstance(e, (KeyboardInterrupt)):
                        raise
                    logger.warning("%s generated an exception", sname, exc_info=True)
                    success = False
        except (ExitProgramException, KeyboardInterrupt):
            exit_with_futures(futures)

    logger.info(
        "====== Step 4. Make sure all local indexes are valid, and (if --sync-packages) have valid local package files ======"
    )
    success = syncer.check_and_update(
        list(local_names),
        package_inclusion_checker,
        file_inclusion_checker,
        json_files,
        packages_pathcache,
        compare_size,
    )
    syncer.finalize(plan.remote_last_serial)

    logger.info(
        "====== Step 5. Remove any unreferenced files in `packages` folder ======"
    )
    ref_set: set[str] = set()
    with ThreadPoolExecutor(max_workers=IOWORKERS) as executor:
        # Part 1: iterate simple/
        def iterate_simple(sname: str) -> list[str]:
            sd = basedir / "simple" / sname
            hrefs = get_existing_hrefs(sd)
            hrefs = [] if hrefs is None else hrefs
            nps = []
            for href, has_metadata in hrefs:
                i = unquote(href)
                # use normpath, which is much faster than pathlib resolve(), as it does not need to access fs
                # we could make sure no symlinks could affect this here
                np = normpath(sd / i)
                logger.debug("add to ref_set: %s", np)
                nps.append(np)
                # also add metadata file to reference set if it exists
                metadata_path = Path(np + ".metadata")
                if has_metadata:
                    metadata_np = str(metadata_path)
                    logger.debug("add to ref_set: %s", metadata_np)
                    nps.append(metadata_np)
            return nps

        futures = {
            executor.submit(iterate_simple, sname): sname for sname in simple_dirs
        }
        try:
            for future in tqdm(
                as_completed(futures),
                total=len(simple_dirs),
                desc="Iterating simple/ directory",
            ):
                sname = futures[future]
                try:
                    nps = future.result()
                    for np in nps:
                        ref_set.add(np)
                except Exception as e:
                    if isinstance(e, (KeyboardInterrupt)):
                        raise
                    logger.warning("%s generated an exception", sname, exc_info=True)
                    success = False
        except (ExitProgramException, KeyboardInterrupt):
            exit_with_futures(futures)

        # Part 2: handling packages
        for path in tqdm(packages_pathcache, desc="Iterating path cache"):
            if path not in ref_set:
                logger.info("removing unreferenced file %s", path)
                Path(path).unlink(missing_ok=True)

    logger.info("Verification finished. Success: %s", success)

    if not success:
        sys.exit(1)


@cli.command(help="Manual update given package for debugging purpose")
@click.pass_context
@sync_shared_args
@click.argument("package_name")
def do_update(
    ctx: click.Context,
    sync_packages: bool,
    shadowmire_upstream: str | None,
    package_inclusion_checker: PackageInclusionChecker,
    file_inclusion_checker: FileInclusionChecker,
    use_pypi_index: bool,
    package_name: str,
) -> None:
    basedir: Path = ctx.obj["basedir"]
    local_db: LocalVersionKV = ctx.obj["local_db"]
    if package_inclusion_checker.has_rules():
        logger.warning("package filter rules are ignored in do_update()")
    syncer = get_syncer(
        basedir, local_db, sync_packages, shadowmire_upstream, use_pypi_index
    )
    syncer.do_update(package_name, file_inclusion_checker, package_files_included=True)


@cli.command(help="Manual remove given package for debugging purpose")
@click.pass_context
@sync_shared_args
@click.argument("package_name")
def do_remove(
    ctx: click.Context,
    sync_packages: bool,
    shadowmire_upstream: str | None,
    package_inclusion_checker: PackageInclusionChecker,
    file_inclusion_checker: FileInclusionChecker,
    use_pypi_index: bool,
    package_name: str,
) -> None:
    basedir = ctx.obj["basedir"]
    local_db = ctx.obj["local_db"]
    if package_inclusion_checker.has_rules() or file_inclusion_checker.has_rules():
        logger.warning("package or file filter rules are ignored in do_remove()")
    syncer = get_syncer(
        basedir, local_db, sync_packages, shadowmire_upstream, use_pypi_index
    )
    syncer.do_remove(package_name)


@cli.command(help="Call pypi list_packages_with_serial() for debugging")
@click.pass_context
def list_packages_with_serial(ctx: click.Context) -> None:
    basedir = ctx.obj["basedir"]
    local_db = ctx.obj["local_db"]
    syncer = SyncPyPI(basedir, local_db)
    syncer.fetch_remote_versions()


@cli.command(help="Clear invalid package status in local database")
@click.pass_context
def clear_invalid_packages(ctx: click.Context) -> None:
    local_db: LocalVersionKV = ctx.obj["local_db"]
    total = local_db.remove_invalid()
    logger.info("Removed %s invalid status in local database", total)


def main() -> None:
    signal.signal(signal.SIGTERM, exit_handler)
    cli()
