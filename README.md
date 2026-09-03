# Spotter Freight Rate Prediction Pipeline

An end-to-end machine learning pipeline that predicts freight shipping rates using a Gradient Boosting Regressor. The model explains ~87% of pricing variance, identifying distance and market conditions as the primary pricing drivers.

## Project Structure
* **`src/data/`**: Contains raw inputs, cleaned datasets, and final CSV deliverables.
* **`src/`**: Contains all data processing, feature engineering, and model training scripts.
* **`gb_model.pkl` & `scaler.pkl`**: Serialized artifacts for deployment and reproducibility.
* **`scorer_results/`**: Output directory for the automated validation chart.

## Setup & Dependencies
This project uses standard Python data science libraries (`pandas`, `scikit-learn`, `matplotlib`, `fpdf`, `structlog`).

If you are using `uv`:
```bash
uv sync