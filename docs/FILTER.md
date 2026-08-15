# Package Filtering

Shadowmire supports scoped, ordered package filters and the legacy `include` and `exclude` options. Both forms use Python regular expressions with `re.search()` against normalized package names. Use anchors such as `^` and `$` when an exact match is required.

The two forms cannot be used together. Shadowmire exits with an error if `package_filters`/`--package-filter` is combined with a legacy `include` or `exclude` rule.

## Ordered package filters

Each `package_filters` rule has an action, pattern, and optional target:

- `metadata` controls whether the project exists in the root simple indexes, `simple/<project>/`, `json/`, `local.db`, and `local.json`.
- `package` controls distribution files and their PEP 658 metadata sidecars.
- `both` applies to both targets and is the default when `target` is omitted.

Rules are evaluated independently for each target, from top to bottom. Rules for another target are skipped. The first applicable matching rule decides whether that target is included or excluded, and a target that matches no rule is included. Excluding project metadata also prevents its package files from being synchronized, regardless of package-target rules.

Package-excluded projects retain their project metadata and simple pages. File links in those pages still point to the mirror's local `packages/` paths; a Web or CDN layer may provide fallback behavior for files that are not mirrored. When a project becomes package-excluded, Shadowmire removes existing distribution and sidecar files during the next sync, including when using `--no-sync-packages`.

For example, this always removes a problematic project, keeps metadata for all other projects, and mirrors package files only for `requests` and `flask`:

```toml
[options]
package_filters = [
    { action = "exclude", pattern = "^problematic-project$", target = "both" },
    { action = "include", pattern = "^requests$", target = "package" },
    { action = "include", pattern = "^flask$", target = "package" },
    { action = "exclude", pattern = ".*", target = "package" },
]
```

The equivalent CLI arguments are:

```shell
./shadowmire.py sync \
    --sync-packages \
    --package-filter 'exclude@both:^problematic-project$' \
    --package-filter 'include@package:^requests$' \
    --package-filter 'include@package:^flask$' \
    --package-filter 'exclude@package:.*'
```

The CLI value is `ACTION@TARGET:PATTERN`, where `ACTION` is `include` or `exclude`, and `TARGET` is `package`, `metadata`, or `both`. Omitting `@TARGET` defaults to `both`, for example `exclude:^broken-project$`. Only the first colon is treated as the separator, so the pattern may contain additional colons or `@` characters.

For an air-gapped mirror, target both metadata and package files and finish with a catch-all exclusion:

```toml
[options]
package_filters = [
    { action = "include", pattern = "^requests$", target = "both" },
    { action = "include", pattern = "^flask$", target = "both" },
    { action = "exclude", pattern = ".*", target = "both" },
]
```

Rule order matters. In the following incorrect configuration, the second rule can never apply because the first rule matches every package:

```toml
package_filters = [
    { action = "exclude", pattern = ".*", target = "package" },
    { action = "include", pattern = "^requests$", target = "package" },
]
```

The package target is unrelated to `--filter-metadata`: the former selects projects whose files are stored locally, while `--filter-metadata` controls whether release/file filtering is reflected in each project's JSON metadata.

## Legacy `include` and `exclude`

The legacy options retain their existing behavior:

1. A package matching any `include` pattern is included.
2. Otherwise, a package matching any `exclude` pattern is excluded.
3. Otherwise, the default depends on which lists are configured.

| `include` | `exclude` | Package matching neither |
| --- | --- | --- |
| empty | empty | included |
| non-empty | empty | excluded |
| empty | non-empty | included |
| non-empty | non-empty | included |

Consequently, `include` alone is a whitelist, `exclude` alone is a denylist, and an `include` match takes priority when a package matches both lists. When both lists are present, `include` acts as an exception to `exclude`; packages matching neither list are still included.

Legacy TOML example:

```toml
[options]
include = ["^django-ninja$"]
exclude = ["^django"]
```

Legacy CLI example:

```shell
./shadowmire.py sync \
    --include '^django-ninja$' \
    --exclude '^django'
```
