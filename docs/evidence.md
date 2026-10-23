# Evidence, provenance, and interpretation

## Sources

The implementation source is the downloaded `CSE 7850 - Team 20 - Hackathon Codebase-20261023T033752Z-1-001.zip`. The final pipeline is preserved in `src/pipeline.py`, with output path handling updated to write new artifacts to `results/generated/` (or `Config.output_dir`). Model and acquisition logic are unchanged. Data, archived queries, returned labels, predictions, and top-10 recommendations come from the same archive. The notebook is stored in `notebooks/`; its setup cells resolve the repository root and import from `src/`. Outputs and execution counts are cleared. Its two metric outputs are extracted into `results/saved_notebook_metrics.json` before clearing.

The report is `Team20_Hackathon_Report.pdf`. Instruction PDFs are the downloaded detailed S26 guide, overview slides, and latest logistics copy. Their source filenames and SHA-256 hashes appear in `source-manifest.json`. The manifest records the copied source versions before the two documented text changes below; it is a provenance record, not a checksum assertion for every final file.

The original `requirements.txt` is retained with `jupyterlab` appended so the documented notebook command is available. The original data README's `sequences.fasta` typo is corrected to the actual `sequence.fasta`. The team's original root README is archived as `archive/original-README.md`. New material consists of the expanded root README, this evidence note, source manifest, extracted metrics, artifact audit, `.gitignore`, and CI configuration. No pretrained weights, caches, or oracle API key are committed.

The earlier downloaded `cse_7850_hackathon-main.zip` is a different modular project. Its `plm_finetune.py` is not evidence that the supplied final pipeline fine-tuned a PLM. It is intentionally excluded. `Team20.pdf` concerns a separate microbiome project and is unrelated.

## Claims and their evidence

| Claim | Evidence | Qualification |
|---|---|---|
| Final implementation uses frozen protein model features | `compute_esm_embeddings`, `compute_prott5_embeddings`, `compute_zero_shot_scores` | `.eval()` plus `torch.no_grad()`; no PLM optimizer/training path |
| Reported external trajectory ~0.05 → ~0.19 → ~0.39 | Report, Model Performance Analysis | Approximate team-reported values; no leaderboard export supplied |
| Local stack holdout 0.6585 | Saved notebook output | Historical output, not a new run |
| Local repeated-holdout stack 0.7072 ± 0.0537 | Saved notebook output | Five random outer holdouts; not position-disjoint outer tests |
| 1,140 initial + 300 queried = 1,440 unique labels | CSVs and query files | Recomputed by artifact audit |
| 11,324 predictions and ten recommendations | `results/submissions/` files | Membership/coverage checked; hidden accuracy cannot be recomputed |
| Query means 0.7780, 0.8013, 0.8261 | Returned-label CSVs | Selection statistics, not evidence of an improved engineered protein |

## Differences between the narrative and the final artifacts

1. **Fine-tuning:** the report's round-3 wording means refining acquisition/model choices. The representation models are frozen. PEFT/LoRA appears under potential future improvements.
2. **Model roster:** one architecture passage emphasizes XGBoost, RF, ElasticNet, and optional MLP. The training passage and final code include Ridge and kNN too; all six are enabled by default.
3. **Traditional feature width:** narrative estimates vary (~45/~50). The default code creates 41 compact indicators/position columns + 4 mutation descriptors + 5 ProtParam columns + 1 BLOSUM score = **51**. The default full width is 51 + 320 ESM + 1 zero-shot = **372** before PCA, with no structural CSV.
4. **No-PLM view:** `FeatureLayout.no_plm()` concatenates traditional and trailing features. The latter includes the default zero-shot PLM score; this view excludes embeddings, not all PLM-derived information.
5. **Round-2 position overlap:** the report says round-2 positions do not overlap round 1. The supplied CSVs show **five shared residue positions**. They contain no repeated mutant identities across rounds. Different substitutions at the same residue are distinct mutants.
6. **Saved holdout:** the report describes a fixed holdout. The code reuses it only for matching sample count and ordered mutant hash, so acquisition updates can change membership.
7. **Validation grouping:** base OOF folds use mutation-position groups. Outer 12% holdouts, repeated holdouts, and the Ridge alpha search are random by sample. The code does not implement nested grouped validation at every level.
8. **Scaling and model selection:** the meta alpha search scales the full OOF matrix before its inner folds. The MLP uses the base validation fold for early stopping. Both limit claims of fully independent OOF evaluation. Optional Optuna runs on all labels before the outer holdout; it is disabled in the supplied notebook.
9. **Tail oversampling:** an enabled `use_smote` flag attempts `SMOTER` and catches exceptions, then duplicates upper-tail points with Gaussian noise. The exact branch used in the historical run is not recorded.
10. **Round-3 allocation:** the five-neighbor prefill and nominal budgets sum to 105. The 100-query cap truncates later selections, and destination balancing may change identities. The nominal fractions are not the exact final composition.
11. **Hyperparameter optimization:** support for Optuna does not mean it ran. The saved notebook calls `run_pipeline(cfg, run_optuna=False)`.
12. **Ensemble benefit:** repeated local means for stack and RF are 0.707150 and 0.706949. This difference alone is not persuasive evidence of a better stack.
13. **External versus local metrics:** ~0.39 from the report and ~0.707 from saved local evaluation are different quantities. No hidden labels or original leaderboard logs are supplied to reconcile them.
14. **Biological interpretation:** normalized DMS scores and a recommendation list do not establish a measured fitness improvement relative to wild type or a wet-lab validated engineered variant.

## Verification scope

The repository audit validates the supplied artifact relationships using the standard library. It checks the wild-type sequence for every mutation, exact reconstructed sequences in returned-label files, single-site universe coverage, disjoint queries, score bounds, full prediction coverage, and top-10 consistency. It recomputes acquisition summary statistics. CI also compiles the Python sources.

Preparing this repository does not rerun the full PLM embedding/training workflow: pretrained weights, embedding caches, the original environment lockfile, and hidden test labels were not supplied. Saved metrics remain historical evidence, explicitly labeled as such.

## Attribution and ownership

This is Team 20's course project. The report describes approximately equal contributions after initial independent exploration and does not provide component-level authorship. Course PDFs retain their original authorship. No broad software or dataset license has been added without evidence that the source owners granted it. `APIKey.txt` is omitted entirely, including from Git history.

## Repository organization

Source code lives in `src/`, the notebook in `notebooks/`, and archived submission files in `results/submissions/`. New predictions and query files default to the ignored `results/generated/` directory. The source manifest paths describe the initial imported layout; original files can be matched to their current locations by filename and hash, except for the documented edits.
