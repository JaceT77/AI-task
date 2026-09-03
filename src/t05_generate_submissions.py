import joblib
import pandas as pd
import structlog
from pandas.core.frame import DataFrame
from structlog.stdlib import BoundLogger

log: BoundLogger = structlog.getLogger(__name__)

equipment_dict: dict[str, int] = {"Dry Van": 0, "Flatbed": 1, "Reefer": 2}

# File Paths
target_model: str = "./src/gb_model.pkl"
target_scaler: str = "./src/scaler.pkl"
candidate_source: str = "./src/data/validation.csv"
predictions_target: str = "./src/data/validation-predictions.csv"
december_target: str = "./src/data/december-chart-inputs.csv"


def engineer_features(df: DataFrame) -> DataFrame:
    """Applies the exact mathematical transformations used in training."""
    df_features = df.copy()

    # 1. Clean missing data FIRST
    df_features["weight"] = df_features["weight"].fillna(df_features["weight"].median())
    df_features["market_index"] = df_features["market_index"].fillna(1.056)

    # 2. THEN do dates and math
    df_features["date"] = pd.to_datetime(arg=df_features["date"])
    df_features["month"] = df_features["date"].dt.month
    df_features["day_of_month"] = df_features["date"].dt.day
    df_features["day_of_week"] = df_features["date"].dt.dayofweek
    df_features["quarter"] = df_features["date"].dt.quarter
    df_features["is_weekend"] = df_features["day_of_week"].isin([5, 6]).astype(int)
    df_features["is_high_season"] = df_features["month"].isin([6, 7, 8, 9, 10]).astype(int)

    df_features["is_high_market"] = (df_features["market_index"] > 1.056).astype(int)

    # Because weight is now clean, these calculations will succeed
    df_features["distance_x_weight"] = (df_features["distance"] * df_features["weight"]).astype(float)
    df_features["distance_per_100lbs"] = df_features["distance"] / (df_features["weight"] / 100)

    df_features["equipment_encoded"] = df_features["equipment"].map(equipment_dict)
    df_features["route_encoded"] = 42

    feature_columns: list[str] = [
        "distance",
        "weight",
        "distance_per_100lbs",
        "distance_x_weight",
        "market_index",
        "quote_signal",
        "route_encoded",
        "equipment_encoded",
        "month",
        "day_of_week",
        "day_of_month",
        "quarter",
        "is_weekend",
        "is_high_season",
        "is_high_market",
    ]

    return df_features[feature_columns]


if __name__ == "__main__":
    log.info(event="loading model artifacts")
    scaler = joblib.load(filename=target_scaler)
    model = joblib.load(filename=target_model)

    # ==========================================
    # 1. Generate 12,000 Predictions
    # ==========================================
    log.info(event="processing main candidate dataset", source=candidate_source)
    df_candidate: DataFrame = pd.read_csv(filepath_or_buffer=candidate_source)

    X_candidate = engineer_features(df=df_candidate)
    X_candidate_scaled = scaler.transform(X_candidate)
    df_candidate["predicted_rate"] = model.predict(X_candidate_scaled)

    # Scorer strictly requires only two columns in this exact order
    df_predictions: DataFrame = df_candidate[["load_id", "predicted_rate"]]
    df_predictions.to_csv(path_or_buf=predictions_target, index=False)
    log.info(event="saved primary predictions", target=predictions_target, rows=len(df_predictions))

    # ==========================================
    # 2. Generate 31-Day December Trend
    # ==========================================
    log.info(event="generating december trend data")
    december_dates = pd.date_range(start="2025-12-01", end="2025-12-31", freq="D")

    # Locked inputs required by the validation script[cite: 1]
    df_december: DataFrame = pd.DataFrame(
        {
            "pickup": "Lexington",
            "delivery": "Fort Wayne",
            "distance": 360.0,
            "equipment": "Dry Van",
            "weight": 32000.0,
            "date": december_dates,
            "market_index": 1.056,  # Baseline
            "quote_signal": 1,  # Baseline
        }
    )

    X_december = engineer_features(df=df_december)
    X_december_scaled = scaler.transform(X_december)
    df_december["predicted_rate"] = model.predict(X_december_scaled)

    # Scorer strictly requires seven columns in this exact order[cite: 1]
    december_columns: list[str] = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]
    df_december_final: DataFrame = df_december[december_columns]

    # Convert dates back to strings for clean CSV output
    df_december_final["date"] = df_december_final["date"].dt.strftime("%Y-%m-%d")

    df_december_final.to_csv(path_or_buf=december_target, index=False)
    log.info(event="saved december trend predictions", target=december_target, rows=len(df_december_final))

    print("\n======== BATCH GENERATION COMPLETE ===========")
    print(
        "Run `python score.py --predictions ./src/data/predictions.csv --december-predictions ./src/data/december_chart_inputs.csv`"
    )
