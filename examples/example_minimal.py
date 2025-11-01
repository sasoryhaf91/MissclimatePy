
import pandas as pd
from missclimatepy import MissClimateImputer

df = pd.DataFrame({
    "station": ["S001"]*3 + ["S002"]*3,
    "date": pd.to_datetime(["1991-01-01","1991-01-02","1991-01-03"]*2),
    "latitude": [19.5]*6, "longitude": [-99.1]*6, "elevation": [2300]*6,
    "tmin": [8.0, None, 7.5, 9.0, None, 8.2]
})
imp = MissClimateImputer(model="rf", target="tmin", n_estimators=50, n_jobs=-1)
out = imp.fit_transform(df)
print(imp.report(out))

try:
    out.to_parquet("tmin_imputed.parquet", index=False)
    print("Saved to tmin_imputed.parquet")
except Exception as e:
    print(f"Parquet not available ({e}); saving CSV instead.")
    out.to_csv("tmin_imputed.csv", index=False)

