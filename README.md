\# AeroSentinel

\### Aircraft Engine Remaining Useful Life Prediction

\*\*End-to-End Machine Learning Engineering | Predictive Maintenance | Deep Learning | Model Serving\*\*



AeroSentinel is an end-to-end machine learning engineering project focused on estimating aircraft engine Remaining Useful Life (RUL) using NASA's C-MAPSS FD001 turbofan degradation dataset.



The project spans the complete ML lifecycle: data validation, leakage-aware experimental design, classical regression benchmarking, temporal feature engineering, recurrent neural network development, model evaluation, explainability, artifact persistence, automated testing, and containerized inference.



The final system uses a \*\*cycle-aware Long Short-Term Memory (LSTM)\*\* architecture deployed through a \*\*FastAPI inference service\*\*, with persisted preprocessing artifacts and a reproducible Docker runtime.



\## Performance Highlights



| Metric | Verified Result |

|---|---:|

| Official FD001 test MAE | \*\*16.75 cycles\*\* |

| Official FD001 test RMSE | \*\*23.28 cycles\*\* |

| Development-validation MAE | 17.46 cycles |

| Development-validation RMSE | 25.05 cycles |

| Official test engines | 100 |

| Automated tests | \*\*30 passed\*\* |

| Inference sequence length | 30 cycles |

| Selected sensor features | 14 |



The final model was evaluated on the official FD001 test set using a locked inference configuration. Development comparisons and official test results are reported separately to preserve their evaluation context.



\---



\## 1. Problem Definition



Remaining Useful Life prediction aims to estimate how many operational cycles remain before an engine reaches its end-of-life condition.



The FD001 dataset consists of multivariate degradation trajectories recorded across aircraft engine operating cycles.



The project addresses several technical challenges:



\- Modeling nonlinear degradation from multivariate sensor observations.

\- Capturing sequential dependencies across engine operating cycles.

\- Preventing data leakage between engine trajectories.

\- Establishing comparable classical ML and neural-network benchmarks.

\- Evaluating generalization across unseen engines.

\- Maintaining consistency between model development and deployed inference.



Rather than treating prediction as an isolated notebook experiment, AeroSentinel separates offline experimentation from production-oriented inference engineering.



\## 2. System Architecture



AeroSentinel is organized into two interconnected workflows.



\*\*Offline Model Development\*\*



Data ingestion → Engine-aware splitting → Feature engineering → Classical ML benchmarking → Sequence modeling → Evaluation → Final artifact persistence



\*\*Online Inference\*\*



HTTP request → Pydantic validation → Engine-history validation → Saved sensor scaling → Sequence construction → Saved cycle scaling → Cycle-aware LSTM → RUL response



The final inference environment is packaged using Docker, with the model initialized during FastAPI application startup.



\## 3. Data Processing and Experimental Protocol



\### Engine-Aware Dataset Partitioning



The development protocol separates engines rather than randomly distributing temporally correlated observations across training and validation sets.



The established development split contains:



\- \*\*80 development-training engines\*\*

\- \*\*20 development-validation engines\*\*

\- \*\*100 engines in the final development training protocol\*\*

\- \*\*100 engines in the official test evaluation\*\*



This design prevents engine trajectories from being shared across development-training and validation partitions.



\### RUL Target Construction



RUL targets are constructed from engine lifecycle information using the project's established uncapped RUL formulation.



The data-processing workflow preserves engine identity and operational cycle ordering for subsequent temporal modeling.



\### Sensor Selection



The locked inference configuration uses 14 sensors:



`2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21`



The final 30-cycle sensor sequence has dimensions:



\*\*30 time steps × 14 sensor features\*\*



The model additionally receives a standardized ending-cycle feature.



\## 4. Model Development



\### Classical Machine Learning Benchmarks



Classical regression experiments establish comparative reference points for the temporal modeling approach.



The repository includes evaluation artifacts for:



\- Classical model benchmarking.

\- Temporal feature engineering and refinement.

\- Engine-level prediction diagnostics.

\- Error analysis across RUL regions.

\- Feature-family and sensor importance.

\- Random Forest versus LSTM comparisons.



These outputs preserve the experimental evidence behind the final modeling decisions.



\### Cycle-Aware LSTM



The cycle-aware LSTM combines a multivariate sensor sequence with an additional engine-cycle feature.



\*\*Model inputs\*\*



| Component | Representation |

|---|---|

| Sensor history | Latest 30 observations |

| Sensor features | 14 canonical sensors |

| Sensor tensor | `(1, 30, 14)` |

| Cycle tensor | `(1, 1)` |

| Cycle representation | Standardized ending cycle |

| Inference device | CPU |



The temporal branch processes sequential sensor behavior, while the cycle feature provides explicit operational-age information.



The final inference pipeline reuses the saved transformations rather than fitting new scalers during prediction.



\## 5. Comparative Model Evaluation



The following results were obtained under the development-validation comparison protocol.



| Model | MAE ↓ | RMSE ↓ |

|---|---:|---:|

| \*\*Cycle-Aware LSTM\*\* | \*\*17.46\*\* | \*\*25.05\*\* |

| Sensor-Only LSTM | 18.01 | 26.67 |

| Random Forest (Depth 12) | 21.36 | 29.39 |



The cycle-aware LSTM achieved approximately:



\- \*\*21.36% lower MAE\*\* than the Random Forest benchmark.

\- \*\*14.78% lower RMSE\*\* than the Random Forest benchmark.

\- \*\*3.03% lower MAE\*\* than the sensor-only LSTM.



These relative improvements refer to the development-validation results, not to a separate official-test comparison of all three models.



\### Lifecycle-Specific Error Analysis



For the cycle-aware LSTM, the development evaluation reported:



| Evaluation slice | MAE |

|---|---:|

| Near failure | 4.17 |

| Medium RUL | 12.90 |

| High RUL | 27.41 |



The model performed better near failure than in the high-RUL region, where prediction errors were substantially larger.



The high-RUL region also showed a mean signed error of approximately \*\*-18.01 cycles\*\*, indicating a tendency toward underprediction in that evaluation slice.



This distinction matters because aggregate metrics alone can conceal operationally relevant differences across the degradation lifecycle.



\## 6. Official NASA Test Evaluation



Following the locked development methodology, the final cycle-aware LSTM was evaluated on the official FD001 test set.



| Metric | Result |

|---|---:|

| Test engines | 100 |

| MAE | \*\*16.7479 cycles\*\* |

| RMSE | \*\*23.2781 cycles\*\* |

| Median absolute error | 10.6517 cycles |

| 95th percentile absolute error | 48.9542 cycles |

| Mean signed error | +12.6185 cycles |



The positive overall mean signed error indicates a tendency toward RUL overprediction on the official test set.



These results represent the final recorded evaluation, not an additional model-selection experiment.



The repository retains the relevant prediction and metric artifacts under `outputs/`.



\## 7. Explainability and Error Diagnostics



The experimental workflow includes model diagnostics designed to examine more than aggregate prediction error.



Available analysis artifacts include:



\- Sensor importance estimates.

\- Lifecycle-oriented sensor importance.

\- Random Forest feature-family importance.

\- Permutation-based importance analysis.

\- Training and validation histories.

\- Engine-level error comparisons.

\- RUL-region-specific diagnostics.



These analyses support inspection of model behavior and evaluation limitations.



They should be interpreted as model-level diagnostic evidence rather than causal explanations of engine degradation.



\## 8. Production Inference Engineering



The prediction system is implemented through a reusable `RULInferenceService`, separating preprocessing and model execution from the HTTP transport layer.



\### Persisted Artifacts



The final inference service loads:



```text

models/saved\_artifacts/

├── fd001\_final\_cycle\_aware\_lstm.pt

├── fd001\_final\_cycle\_feature\_scaler.joblib

└── fd001\_final\_sequence\_sensor\_scaler.joblib

```



\### Inference Validation



Before prediction, the service validates:



1\. Required engine-history columns.

2\. Numeric and finite sensor values.

3\. Single-engine request consistency.

4\. Positive engine IDs and cycle numbers.

5\. Duplicate cycle detection.

6\. Minimum 30-observation requirement.

7\. Model checkpoint metadata.

8\. Sensor feature order and scaler compatibility.



The service sorts observations by cycle, selects the latest 30-cycle sequence, applies the persisted transformations, and executes the locked LSTM using PyTorch.



Model and preprocessing artifacts are initialized during application startup rather than loaded independently for every prediction request.



\## 9. REST API



AeroSentinel provides a REST interface built using FastAPI and Pydantic.



| Endpoint | Method | Responsibility |

|---|---|---|

| `/health` | GET | Application health response |

| `/predict` | POST | Engine-level RUL inference |

| `/docs` | GET | Interactive OpenAPI documentation |



\### Prediction Contract



The prediction endpoint accepts an engine identifier and a history of observations containing cycle values and all 14 required sensor measurements.



A successful prediction returns:



\- `engine\_id`

\- `last\_observed\_cycle`

\- `predicted\_rul`



The application distinguishes request-schema failures, inference-domain validation errors, and unexpected internal failures.



\## 10. Automated Testing



The test suite includes unit and API integration tests covering:



\- Engine-history validation.

\- Sensor and cycle constraints.

\- Invalid and insufficient observations.

\- Model and scaler loading.

\- Saved artifact compatibility.

\- Inference-service initialization.

\- Valid prediction responses.

\- FastAPI request and response behavior.

\- HTTP error handling.



\*\*Latest verified test result:\*\*



```text

30 passed, 4 warnings in 4.75s

```



Run the complete suite:



```bash

uv run python -m pytest tests/ -q

```



The passing suite includes non-failing dependency and runtime warnings. These include a scikit-learn artifact-version mismatch, a NumPy/PyTorch array-writability warning, and a Starlette TestClient deprecation warning.



The scikit-learn version mismatch should be resolved or verified against the artifact-generation environment before claiming strict cross-environment model reproducibility.



\## 11. Docker Containerization



AeroSentinel's inference backend is packaged into a Linux Docker image.



\*\*Runtime components\*\*



\- Python 3.12

\- FastAPI and Uvicorn

\- PyTorch CPU inference

\- NumPy, pandas, and scikit-learn

\- Dependency resolution using uv

\- Final model and preprocessing artifacts



\### Build



From the project root:



```bash

docker build -t nasa-cmapss-rul-api .

```



\### Run



```bash

docker run --rm -p 127.0.0.1:8000:8000 nasa-cmapss-rul-api

```



\### Verify



```bash

curl http://localhost:8000/health

```



Open the interactive documentation:



`http://localhost:8000/docs`



The local Docker runtime has been verified through successful startup, health checks, inference requests, and container recreation.



\## 12. Repository Structure



```text

AeroSentinel/

├── src/

│   ├── api/             # REST API and schemas

│   ├── data/            # Loading and partitioning

│   ├── modeling/        # Model and sequence components

│   ├── evaluation/      # Performance and diagnostics

│   ├── inference/       # Production inference service

│   └── config.py

├── models/

│   └── saved\_artifacts/

├── tests/

├── outputs/

├── notebooks/

├── Dockerfile

├── .dockerignore

├── pyproject.toml

├── uv.lock

└── README.md

```



The repository preserves experimental artifacts alongside the locked inference implementation.



\## 13. Reproducibility and Scope



The project uses `pyproject.toml` and `uv.lock` for dependency management and includes Docker configuration for reproducing the inference environment.



The repository contains historical development outputs and the artifacts required by the finalized model. Experiment artifacts and production inference artifacts serve different purposes and should not be treated interchangeably.



The deployment is an engineering demonstration for the C-MAPSS FD001 benchmark. It is not an operational aircraft maintenance or safety certification system.



\## 14. Technology Stack



| Area | Technologies |

|---|---|

| Language | Python |

| Machine Learning | PyTorch, scikit-learn |

| Data Processing | NumPy, pandas |

| API | FastAPI, Pydantic, Uvicorn |

| Testing | pytest, HTTPX |

| Dependencies | uv |

| Containerization | Docker |

| Version Control | Git, GitHub |



\---



\*\*AeroSentinel — From engine degradation modeling to reproducible ML inference.\*\*



