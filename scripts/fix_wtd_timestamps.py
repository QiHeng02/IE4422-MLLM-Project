from pathlib import Path
import pandas as pd

path = Path("data/processed/wtd/transcripts.csv")
audit = Path("data/processed/wtd/invalid_timestamps.csv")

df = pd.read_csv(path)

invalid = (
    df["start"].isna()
    | df["end"].isna()
    | (df["start"] >= df["end"])
)

bad = df.loc[invalid].copy()
bad.to_csv(audit, index=False)

clean = df.loc[~invalid].copy()
clean.to_csv(path, index=False)

print(f"Original: {len(df)} rows")
print(f"Excluded: {len(bad)} rows")
print(f"Remaining: {len(clean)} rows")
print(f"Audit saved: {audit}")
