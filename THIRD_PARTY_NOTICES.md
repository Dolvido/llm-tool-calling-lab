# Third-party components

The MIT license applies to this repository's original code, documentation, and generated fixtures. Dependencies are installed separately under their own licenses; tested versions are in `requirements.lock`. No dependency binaries or model weights are bundled.

Direct dependencies: NumPy, pandas, scikit-learn, Pydantic, Streamlit, Ollama's Python client, joblib, and threadpoolctl. Development tools include pytest and build. Each package distributes its own license and notices; preserve them if redistributing those packages.

The default model is `qwen3:14b-q4_K_M`. Locally installed metadata reports Apache License 2.0. Evaluation manifests record its exact digest. The project's MIT license does not relicense weights; consult upstream model notices before redistributing them.

No third-party dataset is included. Examples are generated using NumPy and scikit-learn.
