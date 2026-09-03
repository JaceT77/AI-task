import joblib
import pandas as pd
import structlog
from pandas.core.frame import DataFrame
from sklearn.ensemble import GradientBoostingRegressor
from structlog.stdlib import BoundLogger

log: BoundLogger = structlog.getLogger(__name__)

model_source: str = "./src/gb_model.pkl"
feature_source: str = "./src/data/train-test-features.csv"


def generate_summary() -> None:
    log.info(event="loading model for summary statistics")
    model: GradientBoostingRegressor = joblib.load(filename=model_source)

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

    importances = model.feature_importances_
    feature_importance: list[tuple[str, float]] = sorted(
        zip(feature_columns, importances), key=lambda x: x[1], reverse=True
    )

    print("\n======== FEATURE IMPORTANCE RANKING ===========")
    for name, score in feature_importance:
        print(f"{name:<20}: {score * 100:.2f}%")

    log.info(event="summary statistics generated")


if __name__ == "__main__":
    generate_summary()
