#!/usr/bin/env python3
"""Verify historical hackathon artifacts without ML dependencies or model downloads."""
from __future__ import annotations

import csv
import json
import math
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AA = set("ACDEFGHIKLMNPQRSTVWY")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_csv(relative: str, expected_columns: set[str]) -> list[dict[str, str]]:
    with (ROOT / relative).open(newline="") as stream:
        reader = csv.DictReader(stream)
        require(set(reader.fieldnames or []) == expected_columns, f"Unexpected columns: {relative}")
        rows = list(reader)
    require(bool(rows), f"Empty CSV: {relative}")
    mutants = [row["mutant"] for row in rows]
    require(len(mutants) == len(set(mutants)), f"Duplicate mutant: {relative}")
    return rows


def main() -> None:
    wt = "".join(line.strip() for line in (ROOT / "data/sequence.fasta").read_text().splitlines()
                 if line.strip() and not line.startswith(">"))
    require(len(wt) == 656 and set(wt) <= AA, "Unexpected wild-type sequence")

    def sequence(mutant: str) -> str:
        match = re.fullmatch(r"([A-Z])(\d+)([A-Z])", mutant)
        require(match is not None, f"Invalid mutant: {mutant}")
        assert match is not None
        source, position, destination = match.groups()
        pos = int(position)
        require(source in AA and destination in AA and source != destination,
                f"Invalid substitution: {mutant}")
        require(0 <= pos < len(wt) and wt[pos] == source, f"WT mismatch: {mutant}")
        return wt[:pos] + destination + wt[pos + 1:]

    def scores(rows: list[dict[str, str]], key: str) -> list[float]:
        values = [float(row[key]) for row in rows]
        require(all(math.isfinite(v) and 0 <= v <= 1 for v in values), f"Invalid {key}")
        return values

    train = read_csv("data/train.csv", {"mutant", "DMS_score"})
    test = read_csv("data/test.csv", {"mutant"})
    train_ids = {row["mutant"] for row in train}
    test_ids = {row["mutant"] for row in test}
    require(len(train) == 1140 and len(test) == 11324, "Unexpected input row counts")
    require(not train_ids & test_ids, "Original training/test overlap")
    expected = {f"{source}{pos}{destination}" for pos, source in enumerate(wt)
                for destination in AA if source != destination}
    require(train_ids | test_ids == expected, "Incomplete single-substitution universe")
    scores(train, "DMS_score")
    for mutant in train_ids | test_ids:
        sequence(mutant)

    queried: set[str] = set()
    prior_positions: set[int] = set()
    summaries = []
    for round_number in range(1, 4):
        requested = (ROOT / f"data/query_round_{round_number}.txt").read_text().splitlines()
        requested = [m.strip() for m in requested if m.strip()]
        returned = read_csv(f"data/query_round_{round_number}_results.csv",
                            {"mutant", "DMS_score", "sequence"})
        ids = {row["mutant"] for row in returned}
        require(len(requested) == len(set(requested)) == len(returned) == 100,
                f"Round {round_number} query count/duplicates")
        require(set(requested) == ids and ids <= test_ids, f"Round {round_number} identity mismatch")
        require(not ids & (queried | train_ids), f"Round {round_number} repeated query")
        for row in returned:
            require(row["sequence"] == sequence(row["mutant"]),
                    f"Incorrect returned sequence: {row['mutant']}")
        values = scores(returned, "DMS_score")
        positions = {int(re.search(r"\d+", m).group()) for m in ids}
        summaries.append({"round": round_number, "count": len(ids),
                          "mean_dms": statistics.mean(values),
                          "median_dms": statistics.median(values),
                          "unique_positions": len(positions),
                          "positions_shared_with_prior_rounds": len(positions & prior_positions)})
        queried.update(ids)
        prior_positions.update(positions)
    require(len(train_ids | queried) == 1440, "Unexpected final label count")

    predictions = read_csv("predictions.csv", {"mutant", "DMS_score_predicted"})
    require([r["mutant"] for r in predictions] == [r["mutant"] for r in test],
            "Prediction coverage/order differs from test.csv")
    values = scores(predictions, "DMS_score_predicted")
    score_map = dict(zip((r["mutant"] for r in predictions), values))
    top10 = (ROOT / "top10.txt").read_text().splitlines()
    require(len(top10) == len(set(top10)) == 10 and set(top10) <= test_ids,
            "Invalid top-10 membership")
    top_values = [score_map[m] for m in top10]
    require(top_values == sorted(top_values, reverse=True), "Top-10 scores are not descending")
    others = [v for m, v in score_map.items() if m not in set(top10)]
    require(min(top_values) >= max(others), "Top-10 omits a higher-scoring candidate")
    require(not list(ROOT.rglob("APIKey.txt")), "Credential file must not be in repository")
    report = {"status": "passed", "wild_type_length": len(wt), "initial_labels": len(train),
              "final_labels": len(train_ids | queried), "prediction_rows": len(predictions),
              "top10": top10, "query_rounds": summaries}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
