# Replace a specialist model

The extension contract is a trusted developer-supplied scikit-learn-compatible estimator. The chatbot sees catalog descriptions and opaque IDs; it cannot import arbitrary modules or submit executable code.

In `catalog.py`, `catalog_entries` declares each method's family, meaning, and fixed preset. `build_estimator` constructs the estimator from its ID, preset, seed, and group count. `ModelService` applies shared validation, preprocessing, splitting, timeouts, persistence, and results.

## Executed tutorial

Run `lab tutorial`. It fits Ridge on the development regression data, then ElasticNet through the `tutorial` catalog, which replaces Ridge. Both actual results are saved to `.lab/tutorial/comparison.json`. Set `"catalog": "tutorial"` in a config and run `lab --config your-config.json chat` to expose the replacement in conversation. This separate catalog never joins the default benchmark.

## Estimator requirements

| Family | Methods | Output |
|---|---|---|
| Regression | `fit(X, y)`, `predict(X)` | Finite number per row |
| Classification | `fit(X, y)`, `predict(X)`, `classes_` | Encoded binary labels; executor restores names |
| Anomaly | `fit(X)`, `score_samples(X)` | Larger raw score means more normal; executor negates it |
| Clustering | `fit(X)`, `predict(X)` | Group assignment per row |

Inputs already have training-only median imputation and standardization. Do not refit preprocessing on validation/test rows. Another ML library needs a wrapper honoring these methods and meanings. Adding a new task family requires additional validation and evaluation.

Add metadata and a constructor, run through `ModelService`, test invalid outputs/family compatibility, and exercise an actual chat call. Keep experiments in separate catalogs. Benchmark changes require new manifests with previous results preserved.

## Shared records

`TaskSpec` holds question, family, dataset IDs, features, optional target/batch, and group count. `RunResult` holds real run ID, method, task, status, public metrics, bounded summary, warnings, artifact references, and elapsed time. Failed fits also consume allowance.

`retrieve_result` exposes only public validation/batch evidence. `evaluation_metrics` is an evaluator-only Python method absent from the tool catalog. Never add private test records or fitted objects to tool responses.
