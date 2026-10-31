# Motorcycle Safety Advisor — Streamlit Prototype (V2.0 working copy)

Uses **frozen Version 1.1 artifacts without retraining**. The application requires these two files, which are not bundled here:

- `assets/catboost_final_v1_1.joblib` — from Google Drive `MyDrive/Thesis/V1.1_Working/catboost_final_v1_1.joblib`, or the corresponding frozen snapshot.
- `assets/highway_hotspots_clean.csv` — from `MyDrive/Thesis/Processed/highway_hotspots_clean.csv`, or the corresponding frozen snapshot.

Do **not** load untrusted `.joblib` model bundles. Keep data/model files in private storage if they contain confidential information. The app displays 291-source-record/290-unique-coordinate GIS data as provided by the user's research; it deduplicates exact latitude/longitude pairs.

## Local setup

1. Install Python 3.11 or 3.12.
2. Copy the two files above into `assets/`.
3. From the project folder, run:

   ```bash
   python -m pip install -r requirements.txt
   streamlit run app.py
   ```

4. A browser opens with the app. Enter rider features and latitude/longitude, then select **Analyze rider and routes**.

## Google Colab / deployment

Colab can be used to prepare these files, but Streamlit is a server application. For permanent hosting, push this project to a GitHub repository and deploy with Streamlit Community Cloud. Set `app.py` as entrypoint. Include trusted model/data artifacts only if their licensing/privacy policies permit; consider external private storage for bigger files.

## Model and GIS methods

- Model: **CatBoost**, 8 rider/road/environment input features, selected by validation Macro-F1; final test accuracy ~0.679965 and Macro-F1 ~0.681286.
- Class labels map from `No Accident → Low`, `Moderate Accident → Medium`, `Severe Accident → High` **only for interface presentation**.
- SHAP: native CatBoost multiclass raw-score contributions. Summation verified against raw class score.
- Route geometry: public OSRM `driving` service, native alternatives and optionally six exploratory via-point offsets; candidates are not guaranteed exhaustive and do not prove motorcycle-specific accessibility.
- Blackspots: exact coordinates deduplicated; Haversine geodesic distance with Earth radius 6,371,008.8m, Gaussian kernel sigma 250m, cutoff 1,500m, sample spacing approximately 25m, trapezoidal integration over route.
- Balanced route weights: normalized exposure 0.5, duration 0.3, distance 0.2. These are experimental, not validated safety utility weights.

## Critical scientific limitations

- **NOT** a real-world crash probability model. The classifier predicts dataset-defined severity classes, without verified denominators for real-world ride-level crash chances.
- Class scores are **not calibrated** probabilities of an accident occurring.
- SHAP describes what changed the model's raw output; it does **not** prove causal safety impact.
- The model's `Road_Type` is a hypothetical rider-supplied value and is not matched to road attributes on every route segment.
- GIS blackspot proximity is **not a crash probability** and 0 exposure means no known blackspots within the chosen 1.5km radius, not that a route is safe.
- Route candidates might duplicate corridors or miss substantially safer alternatives; OSRM driving has no guaranteed motorcycle constraints.
- Public routing services have availability and usage restrictions; use an appropriately hosted routing service for production.

The original frozen V1.0/V1.1 artifacts should remain unchanged. Treat this as a new V2.0 *working prototype*, not as another completed frozen release.
