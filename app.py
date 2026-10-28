"""
MEROLEMEZ azimuth / crosstalk analyzer -- Streamlit front-end.
DSP lives in azimuth_core.py.
"""

import gc
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import azimuth_core as core
from azimuth_core import Params

st.set_page_config(page_title="MEROLEMEZ Azimuth Analyzer", page_icon="🎚️", layout="wide")

st.title("MEROLEMEZ Azimuth Crosstalk Analyzer")
st.caption("Upload a stereo recording of the MEROLEMEZ azimuth/crosstalk test record "
           "(22 × 1 kHz bands, IEC 98A Clause 5.7).")

# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("Run")
    description = st.text_input("Description (shown on plot)", value="",
                                placeholder="cartridge / setup notes")
    st.header("Analysis parameters")
    with st.expander("Tone & FFT", expanded=False):
        tone = st.number_input("Tone frequency (Hz)", 100.0, 20000.0, 1000.0, 10.0)
        half_w = st.number_input("Peak search half-width (Hz)", 1.0, 200.0, 25.0, 1.0)
        zpad = st.select_slider("FFT zero-pad factor", [1, 2, 4, 8, 16, 32], value=8)
        win_s = st.number_input("Analysis window (s)", 0.5, 20.0, 4.0, 0.5)
    with st.expander("Burst detection", expanded=False):
        min_dur = st.number_input("Min burst duration (s)", 0.1, 20.0, 2.0, 0.5)
        env_win = st.number_input("Envelope window (s)", 0.005, 0.5, 0.05, 0.005, format="%.3f")
        thresh = st.number_input("Threshold (fraction of peak)", 0.001, 0.5, 0.05, 0.005, format="%.3f")
        trim = st.slider("Trim each end of burst", 0.0, 0.45, 0.20, 0.05)

params = Params(
    tone_freq_hz=tone, burst_min_duration_s=min_dur, envelope_win_s=env_win,
    envelope_thresh_frac=thresh, steady_state_trim_frac=trim,
    analysis_window_s=win_s, zero_pad_factor=zpad, search_half_width_hz=half_w,
)


@st.cache_data(show_spinner="Analysing…", max_entries=3)
def run(raw: bytes, p: Params):
    L, R, sr = core.read_stereo_wav(raw)
    out = core.analyze(L, R, sr, p)
    out["sr"] = sr
    out["duration_s"] = len(L) / sr
    del L, R
    gc.collect()
    return out


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
up = st.file_uploader("Stereo WAV file", type=["wav"])
if up is None:
    st.info("Upload a WAV to begin. Parameters in the sidebar match the constants "
            "at the top of the original script.")
    st.stop()

try:
    out = run(up.getvalue(), params)
except Exception as exc:  # unreadable/mono/no-signal files
    st.error(f"Could not analyse this file: {exc}")
    st.stop()

results = out["results"]
n_det = out["n_detected"]

c1, c2, c3 = st.columns(3)
c1.metric("Sample rate", f"{out['sr']/1000:g} kHz")
c2.metric("Duration", f"{out['duration_s']:.1f} s")
c3.metric("Bursts detected", f"{n_det} / {core.N_BANDS}")

if n_det != core.N_BANDS:
    st.warning("Burst count mismatch — adjust the threshold / min duration in the "
               "sidebar, or inspect the envelope below."
               + (f" Bands beyond {core.N_BANDS} were ignored." if n_det > core.N_BANDS else ""))

with st.expander("Burst detection envelope", expanded=(n_det != core.N_BANDS)):
    fig_env = core.make_envelope_figure(out["env"], out["bursts"], out["sr"])
    st.pyplot(fig_env)
    plt.close(fig_env)

if not results:
    st.stop()

sym = core.symmetry_check(results)

# Headline numbers
if sym is not None:
    st.subheader("Bands 6 / 17 symmetry check")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Part 1 (L→R) minimum", f"{sym['az_min_1']:.2f}°", f"{sym['offset1']:+.2f}° from 90")
    m2.metric("Part 2 (R→L) minimum", f"{sym['az_min_2']:.2f}°", f"{sym['offset2']:+.2f}° from 90")
    m3.metric("Mean offset", f"{round(sym['mean_offset'], 2) + 0.0:+.2f}°")
    m4.metric("Sweep spread", f"{sym['spread']:.2f}°")
    (st.warning if sym["spread"] > 1.0 else st.success)(sym["verdict"])
else:
    st.info("Bands 6 and 17 were not both found, so the symmetry check was skipped.")

# Figure
fig = core.make_figure(results, sym, title_name=up.name, description=description)
st.pyplot(fig)
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=150, facecolor=fig.get_facecolor())
plt.close(fig)
st.download_button("Download figure (PNG)", buf.getvalue(),
                   file_name="merolemez_azimuth_crosstalk_result.png", mime="image/png")

# Table
st.subheader("Per-band results")
df = pd.DataFrame(results).rename(columns={
    "band": "Band", "direction": "Dir (deg)", "azimuth": "Azim (deg)",
    "crosstalk_db": "Xtalk (dB)", "phase_diff_deg": "Phase (deg)",
    "t_start": "Window start (s)", "t_end": "Window end (s)"})
df.insert(3, "Path", df.pop("main") + "→" + df.pop("leak"))
st.dataframe(df.round(2), hide_index=True, width="stretch")
st.download_button("Download table (CSV)", df.round(4).to_csv(index=False),
                   file_name="merolemez_azimuth_results.csv", mime="text/csv")

gc.collect()
