# Instructions

Put `sequence.fasta`, `train.csv`, and `test.csv` in `data/`. If you want previously measured active-learning labels merged into training, add `data/query_round_*_results.csv`. If you want prior submitted queries excluded from later rounds, move each submitted `query.txt` into `data/query_round_N.txt` before generating the next round.

Structural features are optional. To use them, set `structure_csv=ROOT / "data" / "structural.csv"` in the notebook `Config(...)` cell and provide a CSV with a `mutant` column, exactly one SASA-like numeric column, and exactly one B-factor-like numeric column. Column names are normalized so names like `SASA`, `b_factor`, and `B-factor` work, but extra numeric columns are rejected. Any missing mutants are filled with zeros.

Install:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run:

1. Open `pipeline.ipynb` from this folder.
2. Run all cells.

The notebook imports `Config` from `pipeline.py`, trains the pipeline, and writes `predictions.csv` and `top10.txt` in the project root. The last cell shows the archived `data/query_round_*.txt` files that will be excluded. If you set `query_round` to `"round1"`, `"round2"`, or `"round3"` in that cell, it also writes the current round to `query.txt` in the project root.
