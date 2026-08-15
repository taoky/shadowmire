# Package Filtering

Shadowmire supports ordered package filters and the legacy `include` and `exclude` options. Both forms use Python regular expressions with `re.search()` against normalized package names. Use anchors such as `^` and `$` when an exact match is required.

The two forms cannot be used together. Shadowmire exits with an error if `package_filters`/`--package-filter` is combined with a legacy `include` or `exclude` rule.

## Ordered package filters

Each `package_filters` rule has an action and pattern:

- `include` includes both project metadata and package files.
- `metadata-only` includes project metadata but not distribution files or their PEP 658 metadata sidecars.
- `exclude` excludes both project metadata and package files.

Rules are evaluated from top to bottom. The first matching rule selects the project's state, and a project that matches no rule is included. Project metadata consists of the root simple indexes, `simple/<project>/`, `json/`, `local.db`, and `local.json`.

Metadata-only projects retain their project metadata and simple pages. File links in those pages still point to the mirror's local `packages/` paths; a Web or CDN layer may provide fallback behavior for files that are not mirrored.

Normal sync is incremental and does not scan package files for projects whose upstream serial is unchanged. When an updated project is metadata-only, Shadowmire removes its existing distribution and sidecar files even with `--no-sync-packages`; when it is included, `--sync-packages` downloads its files normally.

### When reconciliation is required

Shadowmire records applied package-file state in `local.db`. Once a project has explicit state, normal sync applies `include`/`metadata-only` changes without scanning unrelated projects. See [Partial mirrors](PARTIAL.md) for the state model and generated-list workflows.

| Change | Reconciliation required? | Reason |
| --- | --- | --- |
| `include` → `metadata-only` with explicit state | No | The recorded state selects only the changed project for cleanup. |
| `metadata-only` → `include` with explicit state | No, with `--sync-packages` | The recorded metadata-only state schedules that project for download. |
| First application of package-file rules to an existing mirror | Yes | Legacy `NULL` state is trusted for upgrade compatibility; reconciliation establishes actual policy state. |
| Changing release/file filters without an upstream serial change | Yes | Project-level state does not fingerprint release/file filter configuration. |
| `include` or `metadata-only` → `exclude` | No | The project disappears from the filtered remote index, so normal sync removes the whole project even when its serial is unchanged. |
| `exclude` → `include` or `metadata-only` | No | The project appears as new in the filtered remote index, so normal sync schedules a full project update. |
| A fresh, empty mirror | No | Every selected project is already scheduled for its initial update. |
| No filter change, and affected projects received new upstream serials | No | Their normal incremental updates apply the current package-file decision. |

Reconciliation scans all metadata-visible projects. It removes existing distribution files and PEP 658 sidecars from metadata-only projects; when both `--sync-packages` and `--reconcile-package-files` are given, it also schedules included projects whose referenced files or sidecars are missing.

```shell
shadowmire sync --sync-packages --reconcile-package-files
```

The flag only adds this scan for ordered `package_filters`; legacy `include`/`exclude` rules select whole projects and do not need package-file reconciliation. Using the flag when it is not required is safe, but adds an O(number of metadata-visible projects) filesystem scan.

`verify` already performs a full consistency scan, so it applies the same package-filter decisions without this flag. Use `verify --sync-packages` when missing included files must be downloaded. If a mirror was initially created with `--no-sync-packages` and has no ordered `package_filters`, use `verify --sync-packages`; `sync --reconcile-package-files` has nothing to reconcile without ordered rules.

For example, this always removes a problematic project, keeps metadata for all other projects, and mirrors package files only for `requests` and `flask`:

```toml
[options]
package_filters = [
    { action = "exclude", pattern = "^problematic-project$" },
    { action = "include", pattern = "^requests$" },
    { action = "include", pattern = "^flask$" },
    { action = "metadata-only", pattern = ".*" },
]
```

For an existing mirror that previously included every project's package files, apply the equivalent CLI rules with reconciliation:

```shell
shadowmire sync \
    --sync-packages \
    --reconcile-package-files \
    --package-filter 'exclude:^problematic-project$' \
    --package-filter 'include:^requests$' \
    --package-filter 'include:^flask$' \
    --package-filter 'metadata-only:.*'
```

The CLI value is `ACTION:PATTERN`, where `ACTION` is `include`, `metadata-only`, or `exclude`. Only the first colon is treated as the separator, so the pattern may contain additional colons or `@` characters.

For an air-gapped mirror, include lock-derived projects and finish with a catch-all exclusion:

```toml
[options]
package_filters = [
    { action = "include", pattern = "^requests$" },
    { action = "include", pattern = "^flask$" },
    { action = "exclude", pattern = ".*" },
]
```

In this air-gapped configuration, changing the lock-derived list moves projects between `include` and `exclude`. Normal sync detects both directions, so `--reconcile-package-files` is not required.

Rule order matters. In the following incorrect configuration, the second rule can never apply because the first rule matches every package:

```toml
package_filters = [
    { action = "metadata-only", pattern = ".*" },
    { action = "include", pattern = "^requests$" },
]
```

The `metadata-only` action is unrelated to `--filter-metadata`: the former selects projects whose files are stored locally, while `--filter-metadata` controls whether release/file filtering is reflected in each project's JSON metadata.

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
shadowmire sync \
    --include '^django-ninja$' \
    --exclude '^django'
```
