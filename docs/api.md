
# API

```python
from missclimatepy import MissClimateImputer
imp = MissClimateImputer(model="rf", target="tmin")
imp.fit(df)
df_out = imp.transform(df)
imp.report(df_out)
```

**Parameters**
- `model`: `"rf"` or `"idw"`.
- `target`: variable name to impute.
- Column requirements: `station, date, latitude, longitude, elevation, <target>`.
