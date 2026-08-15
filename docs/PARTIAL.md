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

### nginx combined-log traffic

`utils.analyze_nginx_log` expands a full-path glob for nginx combined access
logs, sorts matching regular files by modification time, and analyzes the
newest `k` files (`7` by default). It counts `/simple/<project>/` accesses by
normalized project name and selects the projects needed to reach a cumulative
traffic ratio. For example, to retain projects responsible for 99% of simple
index requests from the seven most recent matching logs:

```shell
uv run --locked --no-dev --group utils python -m utils.analyze_nginx_log \
    --glob '/var/log/nginx/pypi.access.log*' \
    --recent 7 \
    --coverage 0.99 \
    --metric requests \
    --output popular-projects.txt

uv run --locked --no-dev --group utils python -m utils.generate_package_filters \
    popular-projects.txt package_filters.toml
```

`--glob` is required and accepts absolute or relative paths. Quote it so the
Python tool, rather than the shell, performs the expansion before selecting the
newest files. Plain files are opened directly. Compressed files are streamed
by invoking system binaries based on their suffix: `gzip -dc` for `.gz`,
`xz -dc` for `.xz`, and `zstd -dc` for `.zst` or `.zstd`. Install the
corresponding binary for every compression format present among the selected
files. A missing or failing decompressor aborts the run before the output file
is opened for writing.

Only successful GET responses (HTTP 2xx and 3xx) are counted. `requests` means
HTTP requests, not unique users or installations; `bytes` is the nginx
`$body_bytes_sent` value for simple index responses. Package-file URLs are not
processed.

To reduce vote stuffing, accesses to the same normalized project from the same
IPv4 `/24` or IPv6 `/48` network are counted only once within five minutes.
IPv4-mapped IPv6 addresses use the IPv4 `/24` rule. The window is shared across
all selected log files, which are processed oldest first, and a suppressed
duplicate does not extend the window. Different projects from the same network
are counted independently.

The output is sorted and contains one project name per line. A malformed line
is skipped and reported unless `--strict` is used. If no attributable traffic
is found, the command fails without writing `--output`; use `--allow-empty`
only when an empty generated list is intentional. Run `--help` for the complete
options.

### PyPI BigQuery simple-index traffic

`utils.analyze_bigquery` selects popular projects from PyPI's public
`bigquery-public-data.pypi.simple_requests` table. It measures requests to the
global PyPI service's `/simple/<project>/` pages, so its metric is analogous to
the nginx analyzer's request metric. The public table does not contain client
IP addresses, so the IPv4 `/24` and IPv6 `/48` vote deduplication cannot be
applied to this source. The query counts all recorded simple-index requests,
including automation; it does not filter by installer or CI metadata.

Create or choose a Google Cloud project, enable the BigQuery API, and make
Application Default Credentials available. For local development with the
Google Cloud CLI:

```shell
gcloud auth application-default login
```

The authenticated identity needs permission to create BigQuery jobs in the
billing project, normally through `bigquery.jobs.create`. In containers and
other unattended environments, use the platform's Application Default
Credentials mechanism, such as an attached service account or workload
identity, rather than embedding credentials in the image.

Estimate a seven-day query without incurring query-processing charges or
writing an output file:

```shell
uv run --locked --no-dev --group utils python -m utils.analyze_bigquery \
    --project my-billing-project \
    --days 7 \
    --coverage 0.99 \
    --dry-run
```

Remove `--dry-run` and add `--output` to generate the project list:

```shell
uv run --locked --no-dev --group utils python -m utils.analyze_bigquery \
    --project my-billing-project \
    --days 7 \
    --coverage 0.99 \
    --output popular-projects.txt
```

By default, the date range is the latest complete UTC day ending at the current
UTC date. Use `--days` to expand the window and `--end-date YYYY-MM-DD` to make
the exclusive end date explicit.
The SQL filters the partitioned `timestamp` column and reads only the timestamp
and project-name fields.

Every normal run first performs a free BigQuery dry run and reports the
estimated logical bytes scanned, its percentage of the 1 TiB monthly free query
allowance, and the cost at the US on-demand rate if no free allowance remains.
It then executes only when that estimate is within
`--maximum-bytes-billed`, which defaults to 1 TiB; the same limit is also sent
with the actual query. This is a per-query limit, not an account budget: other
queries can already have consumed some or all of the shared monthly allowance.
At the current US rate of US$6.25/TiB, 100 GiB uses 9.77% of the allowance and
would cost about US$0.61 after the allowance is exhausted. Query prices can
change, so consult the current BigQuery pricing page before relying on the
displayed rate.

The exact cost is not fixed because each date partition changes with PyPI
traffic. A failed dry run, an estimate over the configured limit, a query
failure, or an empty result does not overwrite `--output`. Use `--allow-empty`
only when an empty generated list is intentional.

As a concrete reference, a dry run performed on 2026-08-16 for the seven-day
range from 2026-08-08 through 2026-08-15 (exclusive) estimated 770.60 GiB. That
was 75.25% of the 1 TiB monthly free query allowance, or about US$4.70 at
US$6.25/TiB if no free allowance remained. A dry run itself does not incur
query-processing charges or consume that allowance.

The output contains entries for the inside of `package_filters`, not a complete
configuration. The caller is responsible for assembling it, for example with
concatenation or Jinja2. The generator:

- validates and PEP 503-normalizes names;
- ignores blank lines and whole-line comments;
- sorts and deduplicates names;
- groups names into bounded exact-match regexes instead of producing one rule
  per project.

Do not mix generated ordered rules with the legacy `include` or `exclude`
options. Always retain the final catch-all rule: unmatched ordered rules default
to `include`.

### Lock-file package lists

For an air-gapped mirror, extract the package closure already recorded in a uv
project lock file:

```shell
uv run --locked --no-dev --group utils python -m utils.extract_uv_lock_packages \
    /path/to/uv.lock \
    --output locked-projects.txt
```

PEP 751 `pylock.toml` and named `pylock.<name>.toml` files use a separate
extractor:

```shell
uv run --locked --no-dev --group utils \
    python -m utils.extract_pep751_lock_packages \
    /path/to/pylock.toml \
    --output locked-projects.txt
```

Both extractors validate and normalize names, sort and deduplicate the output,
and read every package entry in the universal lock rather than resolving
dependencies or evaluating markers for the current machine. Registry/index
packages from every recorded index are included by name. Ensure the lock was
generated against PyPI, or that the configured Shadowmire upstream contains
packages which came from another recorded index.

Sources which a Python package index mirror cannot reproduce are omitted:
editable, virtual, local path/directory, VCS, and direct archive/URL packages.
They are listed on stderr and must be transported separately. Use
`--strict-non-index` to fail instead of producing a partial list. An empty index
package list is also an error unless `--allow-empty` is explicitly supplied.

Convert the extracted names to ordered filter entries in the same way as other
generated lists:

```shell
uv run --locked --no-dev --group utils python -m utils.generate_package_filters \
    locked-projects.txt package_filters.toml
```

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
