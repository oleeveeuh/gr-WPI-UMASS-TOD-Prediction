# Data directory layout

Full provenance, ethics, checksums, and licensing:
[`docs/DATA.md`](../docs/DATA.md).  Reproduction:
[`docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md).

| Path | Contents | Tracked? |
|---|---|---|
| `raw/GSE71620_Phenotype_GEO.csv` / `.xlsx` | donor phenotype table (GEO GSE71620): numeric donor ID, Age, PMI, pH, RIN, Sex, Race, TOD | yes |
| `raw/pnas.1508249112.sd01.csv` | Chen et al. 2016 PNAS supplementary Table S1 (circadian-gene statistics; source of the 235-gene set) | yes |
| `raw/cause_of_death.csv` | cause/manner of death for 145 donors — **provenance not yet documented; not used by any script** | yes (flagged) |
| `raw/gene_names.csv` / `.xlsx` | 33k-row array annotation (platform GPL11532) — 58 MB, re-downloadable | **no** (checksum in docs/DATA.md; `scripts/download_data.py`) |
| `raw/upstream/` | anything fetched by `scripts/download_data.py` | no (git-ignored) |
| `processed/BA11_data_6_17_2024.csv` | 146 rows × (Age, Sex, TOD hours, 235 genes) | yes |
| `processed/BA47_data_6_17_2024.csv` | 146 rows × (Age, Sex, TOD hours, 235 genes) | yes |
| `processed/full_data_6_17_2024.csv` | 292 rows (both regions; each donor twice) | yes |
| (regenerable intermediates) | `train_test_split_data/`, `window/`, `encoded/`, `reduced_data/`, `reduced_encoded/`, `reduced_CNN*/`, `w{1,2,3}_conv_dense/`, `flatten_w{1,2,3}_conv/`, `all_dfs.RData` — produced by the archived pipeline, ~330 MB of CSVs; untracked since 2026 cleanup, regenerable via `research_archive/`, and fully recoverable from git history | no |
| `program_output/` | scratch output of the archived pipeline | no (git-ignored) |
