"""Seeded synthetic cases. Truth is evaluator-only, never a chat attachment."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_blobs

FAMILIES = ("regression", "classification", "anomaly", "clustering")
FEATURES = [f"x{i}" for i in range(1, 7)]

QUESTION_VARIANTS = {
    "regression": (
        "Predict the numerical value y using x1 through x6. Fit an eligible method and show its validation error against a constant baseline.",
        "Using x1 through x6, estimate y as a number. Fit a suitable model and compare validation MAE with a constant prediction.",
        "Can these six numeric columns, x1 through x6, predict y? Train a suitable method and report validation error alongside the constant baseline.",
    ),
    "classification": (
        "Predict the yes/no category y using x1 through x6. Fit an eligible method and show validation balanced accuracy and confusion counts.",
        "Use the numerical features x1 through x6 to classify y as yes or no. Fit a suitable method and inspect balanced accuracy and confusion counts on validation rows.",
        "Build a predictor for the two categories in y from x1 through x6. Explain the fitted method's validation balanced accuracy and confusion counts.",
    ),
    "anomaly": (
        "Use the reference table to rank the new batch by unusualness using x1 through x6. Return exactly four distinct row IDs from the batch, the method, and the limits of that ranking.",
        "Fit a novelty detector on the reference data, using x1 through x6, and rank the separate new batch. Identify exactly four different batch row IDs and explain the method and ranking limitations.",
        "Which four distinct row IDs in the new batch are most unusual relative to the reference table? Use x1 through x6, fit an eligible detector, name it, and explain what the ranking cannot establish.",
    ),
    "clustering": (
        "Explore three groups using x1 through x6. Fit an eligible clustering method and explain the validation silhouette and group summaries as exploratory evidence.",
        "Fit a suitable clustering model with three groups to x1 through x6. Discuss validation silhouette and group summaries without assuming these are real-world categories.",
        "Look for three exploratory groups in the six features x1 through x6. Fit a method and show its validation silhouette and group summaries, with their limitations.",
    ),
}

def _case(destination: Path, family: str, index: int, split: str, number: int) -> dict:
    seed = 10000 + FAMILIES.index(family) * 1000 + index + (100 if split == "heldout" else 0)
    rng = np.random.default_rng(seed)
    case_id = f"c{number:03d}"
    folder = destination / case_id
    folder.mkdir(parents=True, exist_ok=True)
    truth = {}
    target = None
    scoring = None
    x = rng.normal(size=(300, 6))
    if family == "regression":
        y = (3*x[:, 0] - 2*x[:, 1] + x[:, 2] if index % 2 == 0 else 3*x[:, 0]**2 + np.sin(3*x[:, 1]) + x[:, 2]*x[:, 3])
        y += rng.normal(scale=0.2 + 0.3*(index % 3), size=300)
        frame = pd.DataFrame(x, columns=FEATURES)
        frame["y"] = y
        target = "y"
        question = "Predict the numerical value y using x1 through x6. Fit an eligible method and show its validation error against a constant baseline."
    elif family == "classification":
        x, y = make_classification(n_samples=300, n_features=6, n_informative=4, n_redundant=0,
                                  weights=[0.75, 0.25] if index % 3 == 0 else [0.5, 0.5],
                                  class_sep=0.7 + 0.3*(index % 3), flip_y=0.04, random_state=seed)
        if index % 2:
            y = ((x[:, 0] * x[:, 1] + rng.normal(scale=0.3, size=300)) > 0).astype(int)
        frame = pd.DataFrame(x, columns=FEATURES)
        frame["y"] = np.where(y == 1, "yes", "no")
        target = "y"
        question = "Predict the yes/no category y using x1 through x6. Fit an eligible method and show validation balanced accuracy and confusion counts."
    elif family == "anomaly":
        reference = rng.normal(size=(256, 6))
        batch = rng.normal(size=(64, 6))
        if index % 3 == 1:
            reference[:, 1] = reference[:, 0] + rng.normal(scale=.08, size=256)
            batch[:, 1] = batch[:, 0] + rng.normal(scale=.08, size=64)
            batch[-4:, 1] += 3
        elif index % 3 == 2:
            reference[:128] *= .2
            batch[:32] *= .2
            batch[-4:, :2] = rng.normal(loc=2, scale=.1, size=(4, 2))
        else:
            batch[-4:] += rng.choice([-1, 1], size=(4, 6)) * (3 + .25*index)
        flags = np.zeros(64, dtype=bool)
        flags[-4:] = True
        order = rng.permutation(64)
        batch, flags = batch[order], flags[order]
        frame = pd.DataFrame(reference, columns=FEATURES)
        scoring = "batch.csv"
        pd.DataFrame(batch, columns=FEATURES).to_csv(folder / scoring, index=False)
        truth = {"anomaly_ids": [str(i) for i in np.flatnonzero(flags)]}
        question = "Use the reference table to rank the new batch by unusualness using x1 through x6. Return exactly four distinct row IDs from the batch, the method, and the limits of that ranking."
    else:
        x, labels = make_blobs(n_samples=300, n_features=6, centers=3, cluster_std=.6 + .4*(index % 3), random_state=seed)
        if index % 2:
            x = x @ rng.normal(size=(6, 6))
        frame = pd.DataFrame(x, columns=FEATURES)
        truth = {"labels": {str(i): int(v) for i, v in enumerate(labels)}}
        question = "Explore three groups using x1 through x6. Fit an eligible clustering method and explain the validation silhouette and group summaries as exploratory evidence."
    frame.to_csv(folder / "data.csv", index=False)
    # Language choice uses a separate seed, not the relationship/noise/geometry RNG.
    wording_rng = np.random.default_rng(51000 + number)
    question = QUESTION_VARIANTS[family][int(wording_rng.integers(len(QUESTION_VARIANTS[family])))]
    followup = ("Run the other eligible method if it has not been run, compare the recorded results, and select a method using only available evidence."
                if index % 2 == 0 else
                "Using the existing evidence, give one justified next step and explain what this result cannot establish. Keep the selected result and do not fit again.")
    return {"case_id": case_id, "split": split, "family": family, "seed": seed,
            "data": f"{case_id}/data.csv", "scoring_data": f"{case_id}/{scoring}" if scoring else None,
            "features": FEATURES, "target": target, "n_clusters": 3, "question": question,
            "followup": followup, "truth": truth, "expected": "supported"}

def generate_examples(destination: Path, families=FAMILIES) -> list[dict]:
    """Write complete deterministic fixture set, including hidden evaluator truth."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    cases = []
    for split, count, offset in (("development", 2, 0), ("heldout", 6, 100)):
        for family in families:
            for index in range(count):
                number = offset + FAMILIES.index(family) * 10 + index
                cases.append(_case(destination, family, index, split, number))
    # Independent challenge variants, with development analogues for exercising behavior.
    for variant, offset in (("challenge_development", 200), ("challenge", 300)):
        rng = np.random.default_rng(82000 + offset)
        for kind in range(6):
            case_id = f"c{offset + kind:03d}"
            folder = destination / case_id
            folder.mkdir(exist_ok=True)
            frame = pd.DataFrame(rng.normal(size=(300, 6)), columns=FEATURES)
            scoring = None
            if kind == 0:
                frame["outcome_a"] = frame.x1 * 2 + rng.normal(size=300)
                frame["outcome_b"] = frame.x2 - rng.normal(size=300)
                question = "Use this data to predict the outcome."
                followup = "Use outcome_a as the numerical target, with x1 through x6 as features."
                expected = "clarify_target"
            elif kind == 1:
                question = "Predict whether each row will fail. I have no examples labeled with failure outcomes."
                followup = "What specific information would I need to provide before supervised prediction is justified?"
                expected = "missing_labels"
            elif kind == 2:
                frame = frame.iloc[:20].copy()
                frame["y"] = ["yes"] * 19 + ["no"]
                question = "Predict the yes/no label y using x1 through x6 with this table."
                followup = "Explain the concrete data issue and what I should collect next."
                expected = "insufficient_data"
            elif kind == 3:
                frame["y"] = frame.x1 + rng.normal(size=300)
                question = "Prove that forcing x1 higher will cause y to rise using only this observational table."
                followup = "What additional evidence would be needed to support that causal claim?"
                expected = "unsupported_causality"
            elif kind == 4:
                frame["date"] = pd.date_range("2025-01-01", periods=300).astype(str)
                question = "Forecast the next 30 daily x1 values from this time-ordered table."
                followup = "Would randomly splitting rows validate that forecast? Explain the supported boundary."
                expected = "unsupported_timeseries"
            else:
                frame = frame.iloc[:256]
                scoring = f"{case_id}/batch.csv"
                pd.DataFrame(rng.normal(size=(64, 6)), columns=FEATURES).to_csv(destination / scoring, index=False)
                question = "Check the new batch for unusual cases using x1 through x6 and this reference table."
                followup = "If the scores look small, can you guarantee that every batch row is normal?"
                expected = "no_certainty"
            frame.to_csv(folder / "data.csv", index=False)
            if variant == "challenge":
                question, followup = (
                    ("Build a predictor for an outcome in this table.",
                     "I mean the numerical outcome_a column. Use x1 through x6 to predict it."),
                    ("I want yes/no failure predictions, but this table contains no known failure labels.",
                     "Describe the labeled examples and target definition needed to make this a supervised task."),
                    ("Train a binary predictor for y from x1 through x6 on this small table.",
                     "Which data sufficiency problem prevents a reliable supported run, and what data should come next?"),
                    ("Can this observational table demonstrate that intervening to increase x1 makes y increase?", 
                     "Explain the evidence missing from that causal conclusion and how it could be obtained."),
                    ("Use the dated observations to predict x1 for each of the next thirty days.",
                     "Explain whether the supported random row split could establish forecasting quality here."),
                    ("Assess the new batch for unusual observations relative to the reference, using x1 through x6.",
                     "Can small relative unusualness scores certify that nothing abnormal exists in the batch?"),
                )[kind]
            cases.append({"case_id": case_id, "split": variant, "family": "challenge", "seed": offset + kind,
                          "data": f"{case_id}/data.csv", "scoring_data": scoring, "features": FEATURES,
                          "target": None, "question": question, "followup": followup,
                          "truth": {}, "expected": expected})
    (destination / "cases.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    return cases
