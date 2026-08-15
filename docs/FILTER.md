# Package Filtering

Shadowmire supports ordered package filters and the legacy `include` and
`exclude` options. Both forms use Python regular expressions with `re.search()`
against normalized package names. Use anchors such as `^` and `$` when an exact
match is required.

The two forms cannot be used together. Shadowmire exits with an error if
`package_filters`/`--package-filter` is combined with a legacy `include` or
`exclude` rule.

## Ordered package filters

`package_filters` evaluates rules from top to bottom. The first matching rule
decides whether the package is included or excluded, and later rules are
ignored. A package that matches no rule is included.

For example, this includes `django-ninja` as an exception and excludes other
package names beginning with `django`:

```toml
[options]
package_filters = [
    { action = "include", pattern = "^django-ninja$" },
    { action = "exclude", pattern = "^django" },
]
```

The equivalent CLI arguments are:

```shell
./shadowmire.py sync \
    --package-filter 'include:^django-ninja$' \
    --package-filter 'exclude:^django'
```

The CLI value is `ACTION:PATTERN`, where `ACTION` is `include` or `exclude`.
Only the first colon is treated as the separator, so the pattern may contain
additional colons.

Because the default is include, a whitelist needs a final catch-all exclusion:

```toml
[options]
package_filters = [
    { action = "include", pattern = "^requests$" },
    { action = "include", pattern = "^flask$" },
    { action = "exclude", pattern = ".*" },
]
```

Rule order matters. In the following incorrect configuration, the second rule
can never apply because the first rule matches every package:

```toml
package_filters = [
    { action = "exclude", pattern = ".*" },
    { action = "include", pattern = "^requests$" },
]
```

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

Consequently, `include` alone is a whitelist, `exclude` alone is a denylist,
and an `include` match takes priority when a package matches both lists. When
both lists are present, `include` acts as an exception to `exclude`; packages
matching neither list are still included.

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
