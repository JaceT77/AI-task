import pandas as pd
import structlog
from pandas.core.frame import DataFrame
from structlog.stdlib import BoundLogger

log: BoundLogger = structlog.getLogger(__name__)
equipment_dict: dict[str, int] = {"Dry Van": 0, "Flatbed": 1, "Reefer": 2}
source: str = "./src/data/train-test-cleaned.csv"
target: str = "./src/data/train-test-features.csv"

df: DataFrame = pd.read_csv(filepath_or_buffer=source)
log.info(event="data frame is loaded", source=source)

log.info(event="creating new fields")
df["date"] = pd.to_datetime(arg=df["date"])
df["month"] = df["date"].dt.month
df["day_of_month"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.dayofweek
df["quarter"] = df["date"].dt.quarter
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
df["is_high_season"] = df["month"].isin([6, 7, 8, 9, 10]).astype(int)
df["is_high_market"] = (df["market_index"] > df["market_index"].median()).astype(int)
df["distance_x_weight"] = (df["distance"] * df["weight"]).astype(float)
df["distance_per_100lbs"] = df["distance"] / (df["weight"] / 100)
df["equipment_encoded"] = df["equipment"].map(equipment_dict)
df["route"] = df["pickup"] + " → " + df["delivery"]
df["route_encoded"] = pd.factorize(values=df["route"])[0]
log.info(event="new fields have been created")


df.to_csv(path_or_buf=target, index=False)
log.info(event="saved the features", target=target)
