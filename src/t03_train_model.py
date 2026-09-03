import joblib
import pandas as pd
import structlog
from pandas.core.frame import DataFrame
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from structlog.stdlib import BoundLogger

log: BoundLogger = structlog.getLogger(__name__)
model = GradientBoostingRegressor(random_state=42)
scaler = StandardScaler()
source: str = "./src/data/train-test-features.csv"
target_model: str = "./src/gb_model.pkl"
target_scaler: str = "./src/scaler.pkl"


df: DataFrame = pd.read_csv(filepath_or_buffer=source)
log.info(event="source is loaded", source=source)

y = df["posted_rate"]
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
X = df[feature_columns]
log.info(event="features and target are defined")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
log.info(event="data is split into training and test data, and scaled")


log.info(event="Training model...")
model.fit(X_train_scaled, y_train)
log.info(event="Training complete!")

joblib.dump(value=model, filename=target_model)
joblib.dump(value=scaler, filename=target_scaler)
log.info(event="model saved!", model=target_model, scaler=target_scaler)


predictions = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, predictions)
rmse = root_mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n======== Model Evaluation ===========")
log.info(event="model_metrics", mae=round(mae, 2), rmse=round(rmse, 2), r2=round(r2, 4))
