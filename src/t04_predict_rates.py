import joblib
import pandas as pd
import structlog
from pydantic import BaseModel

log = structlog.get_logger(__name__)
equipment_dict: dict[str, int] = {"Dry Van": 0, "Flatbed": 1, "Reefer": 2}
target_model: str = "./src/gb_model.pkl"
target_scaler: str = "./src/scaler.pkl"


class FreightRequest(BaseModel):
    pickup: str
    delivery: str
    distance: float
    equipment: str
    weight: float
    date: str
    market_index: float
    quote_signal: int


def process_payload(payload: FreightRequest) -> pd.DataFrame:
    df = pd.DataFrame([payload.model_dump()])

    df["date"] = pd.to_datetime(arg=df["date"])
    df["month"] = df["date"].dt.month
    df["day_of_month"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["quarter"] = df["date"].dt.quarter
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_high_season"] = df["month"].isin([6, 7, 8, 9, 10]).astype(int)
    df["is_high_market"] = (df["market_index"] > 1.056).astype(int)
    df["distance_x_weight"] = (df["distance"] * df["weight"]).astype(float)
    df["distance_per_100lbs"] = df["distance"] / (df["weight"] / 100)
    df["equipment_encoded"] = df["equipment"].map(equipment_dict)
    df["route"] = df["pickup"] + " → " + df["delivery"]
    df["route_encoded"] = 42

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

    return df[feature_columns]


if __name__ == "__main__":
    incoming_data = FreightRequest(
        pickup="Atlanta",
        delivery="Miami",
        distance=660.0,
        equipment="Reefer",
        weight=42000.0,
        date="2026-09-04",
        market_index=1.12,
        quote_signal=1,
    )

    log.info("received_payload", pickup=incoming_data.pickup, delivery=incoming_data.delivery)

    X_new = process_payload(incoming_data)
    scaler = joblib.load(filename=target_scaler)
    model = joblib.load(filename=target_model)

    X_new_scaled = scaler.transform(X_new)
    prediction = model.predict(X_new_scaled)[0]
    log.info("prediction complete", predicted_rate=round(prediction, 2))
    print("\n======== QUOTE ===========")
    print(f"Recommended Rate: ${prediction:,.2f}")
