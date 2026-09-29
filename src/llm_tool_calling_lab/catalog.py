"""The bounded, inspectable catalog of specialist models."""

from copy import deepcopy


METHODS = {
    "ridge": {"family": "regression", "name": "Ridge regression", "description": "Regularized linear prediction. A simple reference for additive relationships.", "parameters": {"alpha": 1.0}},
    "random_forest_regressor": {"family": "regression", "name": "Random Forest regression", "description": "Small tree ensemble for nonlinear numerical relationships.", "parameters": {"n_estimators": 64, "max_depth": 8, "min_samples_leaf": 3, "n_jobs": 1}},
    "logistic_regression": {"family": "classification", "name": "Logistic regression", "description": "Regularized linear binary classifier with balanced class weights.", "parameters": {"C": 1.0, "max_iter": 500, "class_weight": "balanced"}},
    "random_forest_classifier": {"family": "classification", "name": "Random Forest classification", "description": "Small tree ensemble for nonlinear binary classification.", "parameters": {"n_estimators": 64, "max_depth": 8, "min_samples_leaf": 3, "class_weight": "balanced", "n_jobs": 1}},
    "isolation_forest": {"family": "anomaly", "name": "Isolation Forest", "description": "Ranks new cases by global isolation relative to reference rows. Scores are not probabilities or proof of wrongdoing.", "parameters": {"n_estimators": 100, "contamination": "auto", "n_jobs": 1}},
    "local_outlier_factor": {"family": "anomaly", "name": "Local Outlier Factor", "description": "Ranks new cases by local density relative to reference rows. Requires a separate scoring batch; novelty mode never scores training rows.", "parameters": {"n_neighbors": 20, "novelty": True, "contamination": "auto", "n_jobs": 1}},
    "kmeans": {"family": "clustering", "name": "K-Means", "description": "Explores a requested number of compact groups. Validation silhouette measures geometric separation, not real-world meaning.", "parameters": {"n_init": 10, "max_iter": 300}},
    "gaussian_mixture": {"family": "clustering", "name": "Gaussian Mixture", "description": "Explores groups with different covariance shapes, using the same validation silhouette as K-Means.", "parameters": {"covariance_type": "full", "reg_covar": 1e-6, "max_iter": 200, "n_init": 1}},
}

ELASTIC_NET = {"family": "regression", "name": "ElasticNet tutorial replacement", "description": "An exercised extension replacing Ridge in the tutorial catalog; excluded from the default benchmark.", "parameters": {"alpha": 0.1, "l1_ratio": 0.5, "max_iter": 2000}}


def catalog_entries(catalog: str = "default") -> dict:
    if catalog not in {"default", "tutorial"}:
        raise ValueError("Unknown catalog; choose default or tutorial.")
    entries = deepcopy(METHODS)
    if catalog == "tutorial":
        del entries["ridge"]
        entries["elastic_net"] = deepcopy(ELASTIC_NET)
    return entries


def describe_methods(catalog: str = "default", family: str | None = None) -> list[dict]:
    return [dict(method_id=method_id, **entry, input_requirements={"numeric_features": "2–30", "minimum_reference_rows": 100, "binary_target_required": entry["family"] == "classification", "separate_scoring_batch_required": entry["family"] == "anomaly"})
            for method_id, entry in catalog_entries(catalog).items()
            if family is None or entry["family"] == family]


def build_estimator(method_id: str, catalog: str, seed: int, n_clusters: int = 3):
    """Single extension point: declare catalog metadata and return an estimator."""
    from sklearn.cluster import KMeans
    from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
    from sklearn.mixture import GaussianMixture
    from sklearn.neighbors import LocalOutlierFactor

    entries = catalog_entries(catalog)
    if method_id not in entries:
        raise ValueError("Method is not available in this catalog.")
    parameters = entries[method_id]["parameters"].copy()
    constructors = {
        "ridge": Ridge,
        "elastic_net": ElasticNet,
        "random_forest_regressor": RandomForestRegressor,
        "logistic_regression": LogisticRegression,
        "random_forest_classifier": RandomForestClassifier,
        "isolation_forest": IsolationForest,
        "local_outlier_factor": LocalOutlierFactor,
        "kmeans": KMeans,
        "gaussian_mixture": GaussianMixture,
    }
    if method_id != "local_outlier_factor":
        parameters["random_state"] = seed
    if method_id == "kmeans":
        parameters["n_clusters"] = n_clusters
    if method_id == "gaussian_mixture":
        parameters["n_components"] = n_clusters
    return constructors[method_id](**parameters)
