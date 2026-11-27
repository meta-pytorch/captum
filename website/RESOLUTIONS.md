# `package.json#resolutions`: rationale

Why each `resolutions` entry exists. Re-check every entry on the next
Docusaurus or Node upgrade, and drop pins the dependency tree has outgrown.

## Current pins

- `webpack` `5.104.1`: compatibility. Docusaurus 3.10 breaks with
  `webpack@5.105+` (see https://github.com/meta-pytorch/botorch/pull/3268), so `@docusaurus/*` stays on
  `~3.9.0` and webpack on 5.104.1. Re-check when Docusaurus ships a fix.
- `serialize-javascript` `>=7.0.3`: npm advisory 1113686 (XSS), reached via
  `copy-webpack-plugin` under `@docusaurus/core`.
- `tinypool` `>=2.1.2`: GHSA-5gmw-xhrv-c9v3 and GHSA-85c8-ppgw-ccpr
  (critical). `@docusaurus/core` 3.9 asks for `^1.0.2`; the build output with
  2.2.0 is byte-identical to the build with 1.1.1.
- `uuid` `^11.1.1`: GHSA-w5hq-g745-h8pq, reached via
  `webpack-dev-server > sockjs` (dev server only). `sockjs` only calls
  `uuid.v4`, which 11.x keeps.

## Dropped v1 pins

The Docusaurus 1 site pinned `url-parse`, `tar`, `send`, `immer`,
`loader-utils`, `fast-xml-parser`, `toml`, `form-data`, `got`, `qs`, `lodash`,
`minimatch`, `lodash.template` and `uuid@>=14`. None is needed by the
Docusaurus 3 tree: the npm bulk-advisory check of the new `yarn.lock` reports
no advisories.
