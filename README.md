# Shadowmire

Shadowmire syncs PyPI (or plain HTTP(S) PyPI mirrors using Shadowmire) with a lightweight and easy approach.

## Docs

### Background

Bandersnatch is the recommended solution to sync from PyPI. However, it has these 2 issues that haven't been solved for a long time:

- Bandersnatch does not support removing packages that have been removed from upstream, making it easier to be the target of supply chain attack.
- The upstream must implement [XML-RPC APIs](https://warehouse.pypa.io/api-reference/xml-rpc.html#mirroring-support), which is not acceptable for most mirror sites.

Shadowmire is a light solution to these issues.

### Syncing Protocol

#### From PyPI

PyPI's XML-RPC APIs have `list_packages_with_serial()` method to list ALL packages with "serial" (you could consider it as a version integer that just increases every few moments). `changelog_last_serial()` and `changelog_since_serial()` are NOT used as they could not handle package deletion. Local packages not in the list result are removed.

Results from `list_packages_with_serial()` are stored in `remote.json`. `local.db` is a sqlite database which just stores every local package name and its local serial. `local.json` is dumped from `local.db` for downstream cosumption.

#### From upstream using shadowmire

Obviously, `list_packages_with_serial()`'s alternative is the `local.json`, which could be easily served by any HTTP server. Don't use `local.db`, as it could have consistency issues when shadowmire upstream is syncing.

### How to use

If you just need to fetch all indexes (and then use a cache solution for packages):

```shell
REPO=/path/to/pypi ./shadowmire.py sync
```

If `REPO` env is not set, it defaults to current working directory.

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

If you already have a pypi repo, use `genlocal` first to generate a local db:

```shell
./shadowmire.py genlocal
```

Verify command could be used if you believe that something is wrong. It would remove packages NOT in local db, update all local packages, and delete unreferenced files in `packages` folder:

```shell
./shadowmire.py verify --sync-packages
```

Verify command accepts same arguments as sync.

Also, if you need debugging, you could use `do-update` and `do-remove` command to operate on a single package.

## Acknowledgements

This project uses some code from PyPI's official mirroring tools, [bandersnatch](https://github.com/pypa/bandersnatch).

## Naming

Suggested by LLM.

> Sure, to capture the mysterious, fantastical, and intriguing nature of "Bandersnatch," here are some similar-style project name suggestions:
>
> 1. **Shadowmire**:
>    - Meaning: A mysterious shadowy swamp, implying the unknown and exploration.
