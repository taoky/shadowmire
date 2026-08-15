# Partial mirrors

Shadowmire can keep complete project metadata while storing distribution files
for only selected projects, or produce a self-contained allowlisted mirror for
an air-gapped environment. Package selection is project-level: an included
project still uses the existing release and filename filters.

## Popular-project mirror

Put permanent exclusions first, generated popular-project rules next, and a
catch-all `metadata-only` rule last:

```toml
[options]
sync_packages = true
package_filters = [
    { action = "exclude", pattern = "^.{201,}$" },

    # Generated entries are inserted here.
    { action = "include", pattern = '^(?:flask|requests|urllib3)$' },

    { action = "metadata-only", pattern = ".*" },
]
```

Metadata-only projects remain in the root simple indexes, `simple/<project>/`,
`json/`, `local.db`, and `local.json`. Their simple pages still contain the
upstream file links rewritten to local `packages/` paths. A public partial
mirror therefore needs a Web/CDN fallback for files it does not store.

Shadowmire records the applied package-file state in the existing `local`
SQLite table. A change between `include` and `metadata-only` only examines the
projects whose recorded state changed; it does not scan every simple project.

## Air-gapped mirror

Extract the resolved project names from the lock file, insert their generated
rules, and finish with `exclude`:

```toml
[options]
sync_packages = true
package_filters = [
    { action = "include", pattern = '^(?:certifi|requests|urllib3)$' },
    { action = "exclude", pattern = ".*" },
]
```

Only included projects appear in simple and JSON metadata. A lock extractor
must use the dependency closure already present in the lock; it must not resolve
dependencies again. Non-index dependencies need separate handling by the
caller.

Moving an existing full mirror to a small allowlist can legitimately delete a
large number of projects. `SHADOWMIRE_MAX_DELETION` continues to guard that
operation; inspect `plan.json` before raising the limit.

## Generating filter entries

Data acquisition belongs in `utils/`: BigQuery, nginx-log, database, and lock
parsers should emit one project name per line. Convert that canonical list into
a TOML fragment with:

```shell
python -m utils.generate_package_filters packages.txt package_filters.toml
```

The `utils` modules are source-checkout tooling and are deliberately not
included in the installed wheel or exposed as console commands.

The output contains entries for the inside of `package_filters`, not a complete
configuration. The caller is responsible for assembling it, for example with
concatenation or Jinja2. The generator:

- validates and PEP 503-normalizes names;
- ignores blank lines and whole-line comments;
- sorts and deduplicates names;
- groups names into bounded exact-match regexes instead of producing one rule
  per project; and
- atomically replaces its output, so a failed refresh preserves the previous
  fragment.

Do not mix generated ordered rules with the legacy `include` or `exclude`
options. Always retain the final catch-all rule: unmatched ordered rules default
to `include`.

## Package-file state

`local.value` is the project metadata serial. The nullable
`local.file_serial` column records package-file state:

| `file_serial` | Meaning |
| --- | --- |
| `NULL` | Compatibility state; for included projects it is treated as equal to `value`, and it never implies a package-policy transition by itself. |
| `-2` | The project is included, but its package files need synchronization. |
| `-1` | Metadata-only has been applied and referenced local files removed. |
| non-negative | Package files were synchronized at this upstream serial. |

The nullable default makes database upgrades lazy. Adding the column does not
rewrite existing rows, and unchanged legacy projects do not suddenly scan
simple or download files. In particular, a legacy `NULL` row currently matched
by `metadata-only` is not interpreted as an `include` to `metadata-only`
transition. State becomes explicit only when a project is updated or it is
verified/reconciled.
`local.json` keeps its existing `{project: metadata_serial}` format.

An included project updated with `--no-sync-packages` is marked `-2`, even
though its metadata serial advances. A later `--sync-packages` run therefore
fills its files without requiring a full reconcile.

## Reconciliation and verification

Ordinary generated-list changes do not require
`--reconcile-package-files` once explicit states exist. Reconciliation remains
useful when first applying a package-file policy to an existing mirror, after
changing release/file filters, or when local state cannot be trusted:

```shell
shadowmire sync --sync-packages --reconcile-package-files
```

`verify` performs a full consistency pass and refreshes explicit state for the
projects it verifies. `genlocal` cannot infer file completeness from JSON
metadata; its new rows retain the backwards-compatible `NULL` state.
