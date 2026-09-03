import os

import pandas as pd
import structlog
from pandas.core.frame import DataFrame
from structlog.stdlib import BoundLogger

log: BoundLogger = structlog.getLogger(__name__)
source: str = "./src/data/train-test.csv"
target: str = "./src/data/train_test_cleaned.csv"

df: DataFrame = pd.read_csv(filepath_or_buffer=source)
log.info(event="source is loaded")

print()
print("=" * os.get_terminal_size().columns)

print("DF shape")
print(df.shape)
print()

print("Missing values")
print(df.isna().sum())
print()

print(df[["weight", "posted_rate", "market_index"]].describe())
print()

print("=" * os.get_terminal_size().columns)


log.info(event="removing negative weights")
df: DataFrame = df[df["weight"] >= 0]

log.info(event="replacing market_index null values")
weight_median = df["weight"].median()
market_index_median = df["market_index"].median()
df["weight"] = df["weight"].fillna(weight_median)
df["market_index"] = df["market_index"].fillna(market_index_median)


print("=" * os.get_terminal_size().columns)
log.info(event="values after clean up")

print("DF Shape")
print(df.shape)
print()

print("Missing values")
print(df.isna().sum())
print()

print(df[["weight", "posted_rate", "market_index"]].describe())
print()


df.to_csv(path_or_buf=target)
log.info(event="Saved df", target=target)
