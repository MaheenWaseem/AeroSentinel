

\# AeroSentinel



\### Aircraft Engine Remaining Useful Life Prediction



AeroSentinel is an end-to-end machine learning engineering project built on NASA's C-MAPSS FD001 turbofan degradation dataset.



The system combines leakage-aware model development, temporal feature engineering, classical ML benchmarking, and a cycle-aware LSTM for aircraft engine Remaining Useful Life (RUL) prediction.



\## Technical Stack



Python · PyTorch · Scikit-learn · Pandas · FastAPI · Pydantic · pytest · Docker · uv



\## Key Capabilities



\- Engine-aware training and validation

\- Classical ML and temporal sequence modeling

\- Cycle-aware LSTM using 30-cycle sensor windows

\- Saved preprocessing and model artifacts

\- Reusable production inference service

\- REST API with input validation

\- Unit and integration testing

\- Containerized CPU inference



\## API Endpoints



\- `GET /health` — Health response

\- `POST /predict` — Remaining Useful Life prediction

\- `GET /docs` — Interactive API documentation



\## Local Docker Execution



Build the image:



```bash

docker build -t nasa-cmapss-rul-api .

```



Run the container:



```bash

docker run --rm -p 127.0.0.1:8000:8000 nasa-cmapss-rul-api

```



Open `http://localhost:8000/docs`.



\## Project Status



Local Docker inference has been verified. Public deployment and final performance documentation are being prepared.



\## Dataset



NASA C-MAPSS FD001.



