# GPM DPR India 2024 – Dataset & System Summary

## 1. Loading Performance

- NSR load time: 1.09 sec
- LAT load time: 2.28 sec
- LON load time: 2.23 sec
- Timestamp load time: 1.25 sec
- Total load time: 6.85 sec

Observations:
- Float arrays (lat/lon) dominate load time.
- Decompression is the main cost.
- Total raw load ~7 seconds for ~15.6M observations.
- Parallel chunk decompression is justified for further speedup.

---

## 2. Dataset Size

- Total observations: 15,643,524
- Raw float memory footprint (one variable): ~59.7 MB
- 3 float arrays (nsr, lat, lon) ≈ 180 MB in memory
- Timestamps add additional memory overhead

Conclusion:
Dataset is large but manageable in RAM.

---

## 3. Spatial Structure

- Latitude range: 5° to 40°
- Longitude range: 65° to 100°
- Observations are irregular (swath-based, not grid-based)

Conclusion:
We must bin into a spatial grid before building a datacube.

---

## 4. Temporal Structure

- Unique hourly bins: 586

Conclusion:
Hourly aggregation is reasonable.
Raw timestamp dimension (~243k) is too large for cube.

---

## 5. Sparsity

- Non-zero rainfall ratio: 8.75%

Conclusion:
Dataset is highly sparse (~91% zero).
Aggregation must use mean/sum with count tracking.

---

## 6. Default Cube Design (Justified)

Chosen:
- Spatial resolution: 0.25°
- Time resolution: Hourly

Estimated dimensions:
- Lat bins ≈ 140
- Lon bins ≈ 140
- Time bins = 586

Estimated cells:
586 × 140 × 140 ≈ 11.5 million

Estimated memory:
~46 MB (float)

Conclusion:
Default cube is feasible for in-memory OLAP.
