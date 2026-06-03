# Internal Legacy Code Omitted

The original copied internal tree contained one-off scripts, vendored runtime code,
legacy names, local absolute paths, and research artifacts that are not suitable
for a clean public release.

The restructuring keeps the public implementation in `terrabench/` and
`terra_agent/`. Legacy directories were inventoried in `docs/cleanup_audit.md`
and omitted from this release candidate to avoid path, credential, and stale-name
leakage.

If a specific legacy tool or evaluator is needed, port it into the public package
interfaces instead of restoring this internal tree wholesale.
