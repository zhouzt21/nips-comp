# Original baseline provenance

- Upstream repository: https://github.com/aims-foundations/paiec_baseline
- Frozen upstream commit: `cc88a959c1e5e7462f31a9215baba066ce6d6e2b`
- Upstream commit subject: `add citation`
- Original upstream tree: `1012011207974d13440ff71c5a912a07df5f4471`
- Import date: 2026-10-22
- Parent-repository snapshot commit: `7261997ff771237700a1a35c8608aea00a376df6`
- Parent-repository snapshot tag: `paiec-original-cc88a95`
- License and citation: [LICENSE](LICENSE), [CITATION.cff](CITATION.cff), preserved from upstream.

The tagged parent commit contains the unmodified upstream source under
`paiec_baseline/`. Its subtree Git tree ID was verified equal to the upstream
tree above, including tracked contents and executable modes. The parent commit
ID differs from the upstream commit because the files are nested inside the
research workspace.

Subsequent parent-repository commits contain this project's modifications. The
directory is vendored as ordinary tracked files, not a submodule. No continuing
upstream synchronization is configured. The original nested repository history
and all pre-migration working files were backed up locally before migration.

Compare modifications from the workspace root:

```powershell
git diff paiec-original-cc88a95 HEAD -- paiec_baseline
```
