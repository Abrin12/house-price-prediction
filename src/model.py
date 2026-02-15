import pandas as pd
from pathlib import Path

csv_path = Path(__file__).resolve().parent.parent / "data" / "hp.csv"
df = pd.read_csv(csv_path)

print(df.head())
print(df.columns)
