# Shadowmire

Shadowmire syncs PyPI (or plain HTTP(S) PyPI mirrors using Shadowmire) with a lightweight and easy approach.

Requires Python 3.11+.

## Docs

### Background

Bandersnatch is the recommended solution to sync from PyPI. However, it has these 2 issues that haven't been solved for a long time:

- Bandersnatch does not support removing packages that have been removed from upstream, making it easier to be the target of supply chain attack.
- The upstream must implement [XML-RPC APIs](https://warehouse.pypa.io/api-reference/xml-rpc.html#mirroring-support), which is not acceptable for most mirror sites.

Shadowmire is a lightweight solution to these issues.

### Syncing Protocol

#### From PyPI

PyPI's XML-RPC APIs have `list_packages_with_serial()` method to list ALL packages with "serial" (you could consider it as a version integer that just increases every few moments). `changelog_last_serial()` and `changelog_since_serial()` are NOT used as they could not handle package deletion. Local packages not in the list result are removed.

Results from `list_packages_with_serial()` are stored in `remote.json`. `local.db` is a sqlite database which just stores every local package name and its local serial. `local.json` is dumped from `local.db` for downstream cosumption.

#### From upstream using shadowmire

Obviously, `list_packages_with_serial()`'s alternative is the `local.json`, which could be easily served by any HTTP server. Don't use `local.db`, as it could have consistency issues when shadowmire upstream is syncing.

### How to use

> [!IMPORTANT]
> Shadowmire is still in experimental state. Please consider take a snapshot before using (if you're using ZFS/BtrFS), to avoid Shadowmire eating all you packages in accident.

#### Synchronization

If you just need to fetch all indexes (and then use a cache solution for packages):

```shell
./shadowmire.py --repo /path/to/pypi sync
```

If `--repo` argument is not set, it defaults to current working directory.

If you need to download all packages, add `--sync-packages`.

```shell
./shadowmire.py sync --sync-packages
```

> [!IMPORTANT]
> If you sync with indexes only first, `--sync-packages` would NOT update packages which have been the latest versions. Use `verify` command for this.

Sync command also supports `--exclude` -- you could give multiple regexes like this:

```shell
./shadowmire.py sync --exclude package1 --exclude ^0
```

Also it supports prerelease filtering like [this](https://bandersnatch.readthedocs.io/en/latest/filtering_configuration.html#prerelease-filtering):

```shell
./shadowmire.py sync --sync-packages --prerelease-exclude '^duckdb$'
```

And `--shadowmire-upstream`, if you don't want to sync from PyPI directly.

```shell
./shadowmire.py sync --shadowmire-upstream http://example.com/pypi/
```

> [!NOTE]
> Upstream must also use shadowmire to serve `local.json`. Otherwise, you must specify `--use-pypi-index`.

If you already have a PyPI repo, use `genlocal` first to generate a local db:

```shell
./shadowmire.py genlocal
```

> [!IMPORTANT]
> You shall have file `json/<package_name>` before `genlocal`.

#### Verification

`verify` command could be used if you believe that something is wrong (inconsistent). It would:

1. remove packages NOT in local db (skip by default, it would only print package names without `--remove-not-in-local`)
2. remove packages NOT in remote (with consideration of `--exclude`)
3. make sure all local indexes are valid, and (if --sync-packages) have valid local package files
   (`--prerelease-exclude` would used only for packages that requires updating)
4. delete unreferenced files (i.e. blobs) in `packages` folder

```shell
./shadowmire.py verify --sync-packages
```

> [!TIP]
> users are recommended to run `verify` regularly (e.g. every half a year) to make sure everything is in order, thanks to the unpredictable nature of PyPI.

`verify` command accepts same arguments as sync, and accepts some new arguments. Please check `./shadowmire.py verify --help` for more information.

> [!TIP]
> You could set `SHADOWMIRE_IOWORKERS` environment variable to a number to set threads to do local I/O. Defaults to 2.

> [!IMPORTANT]
> For users switching from Bandersnatch to Shadowmire, you **MUST** run the following commands (with exclusion, of course) before regular syncing:
>
> 1. `./shadowmire.py genlocal`: generate database from local packages.
> 1. `./shadowmire.py verify --sync-packages --remove-not-in-local --compare-size`: remove any local packages that were missing from upstream index (normally removed from PyPI), then download any missing metadata and packages. **This step is likely to take very long time, depending on your network and disk speed.**
>     * Q: Why will there be packages that are in neither local db nor remote index?
>     * A: They are packages without valid local metadata, and do not exist on PyPI anymore. These packages are typically downloaded a long time ago and removed from upstream, but they may still share some blobs with currently available packages. E.g. after name normalization of `Foo` to `foo`, they share all existings blobs, but `Foo` does not change any more.
>     * Q: My HDD disk (array) is too, too, too slooooow, any method to speed up?
>     * A: You could try remove `--compare-size` argument, at the cost of having a very small possible part of package file inconsistencies locally.
> 1. `./shadowmire.py genlocal`: generate local database again.
> 1. `./shadowmire.py sync --sync-packages`: synchronize new changes after verification.

### Config file

If you don't like appending a long argument list, you could use `--config` ([example](./config.example.toml)):

```shell
./shadowmire.py --config config.toml sync
```

Also, if you need debugging, you could use `do-update` and `do-remove` command to operate on a single package.

## License

Apache-2.0/AFL-3.0 dual license

## Acknowledgements

This project uses some code from PyPI's official mirroring tools, [bandersnatch](https://github.com/pypa/bandersnatch). It uses Academic Free License v3, and you could read its license contents [here](./LICENSE.AFL).

Also special thanks to TUNA for testing shadowmire (it's also available in [tunasync-scripts](https://github.com/tuna/tunasync-scripts/)).

## Naming

Suggested by LLM.

> Sure, to capture the mysterious, fantastical, and intriguing nature of "Bandersnatch," here are some similar-style project name suggestions:
>
> 1. **Shadowmire**:
>    - Meaning: A mysterious shadowy swamp, implying the unknown and exploration.
