# Data Provenance, Ethics, and Governance

## Sources

| File | Origin | Status in repo |
|---|---|---|
| `data/raw/GSE71620_Phenotype_GEO.csv` / `.xlsx` | Donor phenotype table for [GEO accession GSE71620](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE71620) ("The effects of aging on circadian patterns of gene expression in the human prefrontal cortex"; Chen et al. 2016, PNAS 113(1):206-211, PMID 26699485) | tracked |
| `data/raw/pnas.1508249112.sd01.csv` | Supplementary Table S1 of Chen et al. 2016 (DOI [10.1073/pnas.1508249112](https://doi.org/10.1073/pnas.1508249112)); source of the circadian-gene selection (20,236 genes with BA11/BA47 rhythm statistics; the 235-gene analysis set is derived from it) | tracked |
| `data/raw/cause_of_death.csv` | Cause/manner-of-death annotations for 145 donors. **Provenance is not documented in this repository** — it does not appear in the GEO phenotype export and is read by no pipeline script. Kept for completeness pending confirmation from the collaboration; do not redistribute it further until its source is confirmed. | tracked, flagged |
| `data/processed/*.csv` | Derived in-house by `research_archive/data_combining.R` from GEO GSE71620 expression data + the phenotype table: one row per donor sample; Age, Sex, TOD (hours) + 235 circadian gene expression values (BA11: 146 rows; BA47: 146 rows; combined: 292) | tracked |
| gene annotation snapshot (`gene_names.csv`, 48.6 MB / `gene_names.xlsx`, 9.7 MB) | Array annotation for the study's platform (GPL11532). **No longer tracked** (large, re-downloadable); SHA-256 recorded below; fetch via `scripts/download_data.py` | untracked, re-downloadable |
| full expression matrix (`all_sample.csv`, ~33,000 genes × 292 samples) | Derived from GEO GSE71620 CEL files. **Never committed** to this repository. Obtain via the GEO download endpoint (`https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE71620`) | untracked |

`scripts/download_data.py` re-fetches upstream artifacts (series SOFT metadata,
platform annotation, optional supplementary archive) and verifies the checksums
of local snapshots.

## Checksums (SHA-256, as last tracked)

```
a711c7428fcbd028230df3d16888c036ae22e04a1bfde20a6b2bb058fbe37e1e  data/raw/GSE71620_Phenotype_GEO.csv
e89c02cf13780eaa31c228f242c2770bb1772d55f1946440c5f560ee3066ef74  data/raw/GSE71620_Phenotype_GEO.xlsx
def5b8571cab3f99a446098c2e11713cd227538db5381a86c9d437c4d6ef4211  data/raw/pnas.1508249112.sd01.csv
ecfe7a6228435398d03440da2de0936fc2dfc79597713d373b7f660679309a07  data/raw/cause_of_death.csv
40f63d7958c415f519266b3df368f11cccf689ba294d80767443b3096e3bc2dc  gene_names.csv (no longer tracked)
0df305426a67fea189ba5f34612bb3955cbb488f4841919547558a2e02199db8  gene_names.xlsx (no longer tracked)
```

## Human-subject data and deidentification

* The donor data in this repository is **secondary use of a public,
  deidentified dataset** (GEO GSE71620).  No new human subjects were enrolled
  and no new samples were collected for this project.
* Per the GEO series description: samples were obtained through the University
  of Pittsburgh's Brain Tissue Donation Program, with consent from next-of-kin
  during autopsies conducted at the Allegheny County Medical Examiner's
  Office; procedures were approved by the University of Pittsburgh's
  Institutional Review Board for Biomedical Research and Committee for
  Research Involving the Dead.  Questions about the *original* study's ethics
  approvals belong to the original authors (Chen et al. 2016).
* Donors are identified only by numeric IDs.  The phenotype table does include
  age, sex, race, post-mortem interval, tissue pH, RNA integrity, and TOD —
  standard, published biobank metadata for this cohort.  Even so, treat these
  files respectfully: do not attempt re-identification, and do not combine
  them with other datasets to that end.
* `data/processed/` inherits Age/Sex/TOD per row; all deeper intermediates
  (split/reduced/encoded CSVs, git-ignored and regenerable) contain either the
  same metadata or numeric features plus TOD only.

## Permitted use and licensing (three distinct layers)

1. **Code** in this repository: MIT (see [`LICENSE`](../LICENSE)).
2. **The published paper**: © the authors, distributed by SciTePress under
   [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)
   ([paper page](https://www.scitepress.org/Papers/2026/146360/)).  The paper's
   license covers the paper, not this code or the data.
3. **Data**: GSE71620 and the PNAS supplementary file follow the terms of their
   original publications/providers.  Cite Chen et al. 2016 when using the data,
   and review GEO/PNAS terms before redistribution.  `cause_of_death.csv`
   additionally has unresolved provenance (see above) — cite and share it only
   after that is clarified with the collaboration.

## Citation

If you use this work, please cite the paper (see
[`CITATION.cff`](../CITATION.cff) / README) **and** Chen et al. 2016 for the
underlying data.

## What is intentionally NOT in git

* The ~2,100 regenerable intermediate CSVs (windowed / encoded /
  reduced / CNN-feature splits) — recoverable from git history and regenerable
  with the archived pipeline; see `data/README.md` and
  `docs/REPRODUCIBILITY.md`.
* `gene_names.csv/.xlsx` (58 MB combined) — re-downloadable via
  `scripts/download_data.py` (checksums above).
* `data/raw/upstream/` — anything fetched by `scripts/download_data.py`.
* `data/program_output/` — scratch output of the archived pipeline.
