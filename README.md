# MEROLEMEZ Azimuth Crosstalk Analyzer

Streamlit app for analysing a stereo recording of the MEROLEMEZ azimuth/crosstalk test record
(22 × 1 kHz bands, IEC 98A Clause 5.7). Upload a WAV and get crosstalk and phase vs. azimuth,
the bands 6/17 symmetry check with quadratic-refined minima, a PNG figure and a CSV table.

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

## Deploy (Streamlit Community Cloud)
Push this folder to GitHub, then at share.streamlit.io choose the repo, branch `main`, main file `app.py`.

## Files
- `app.py` – Streamlit UI
- `azimuth_core.py` – DSP (burst detection, FFT tone analysis, symmetry check, plots)

Licensed under PolyForm Noncommercial 1.0.0.
