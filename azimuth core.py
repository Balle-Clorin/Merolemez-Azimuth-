"""
azimuth_core.py

Analysis core for the MEROLEMEZ azimuth / crosstalk test record.
(Refactored from merolemez_azimuth_crosstalk_analyzer.py -- the DSP is
unchanged; module-level constants became a Params dataclass so the
Streamlit front-end can expose them, and plotting/printing was split out.)

Record structure (IEC Publication 98A Clause 5.7):
  - 22 single-tone (1 kHz) bands, each cut at a different "direction of
    modulation" angle.
  - Part 1: bands 1-11,  direction 50 -> 40 deg (descending)
  - Part 2: bands 12-22, direction 40 -> 50 deg (ascending)
  - Plotted azimuth = direction + 45 deg  (so 85..95 deg, 90 = nominal)
"""

from __future__ import annotations

import io
import textwrap
from dataclasses import dataclass

import numpy as np
from scipy.io import wavfile

# Recorded "direction of modulation" table (Table 2 of the record docs)
DIRECTIONS_PART1 = list(range(50, 39, -1))   # bands 1-11: 50..40
DIRECTIONS_PART2 = list(range(40, 51))       # bands 12-22: 40..50
DIRECTIONS = DIRECTIONS_PART1 + DIRECTIONS_PART2
AZIMUTH_OFFSET_DEG = 45.0
N_BANDS = 22

COLORS = {"L->R": "#4FC3F7", "R->L": "#FF8A65"}


@dataclass(frozen=True)
class Params:
    tone_freq_hz: float = 1000.0
    burst_min_duration_s: float = 2.0     # ignore clicks/markers shorter than this
    envelope_win_s: float = 0.05          # envelope RMS window
    envelope_thresh_frac: float = 0.05    # fraction of peak envelope = "active"
    steady_state_trim_frac: float = 0.20  # trim off each end of a burst
    analysis_window_s: float = 4.0        # steady-state window length analysed
    zero_pad_factor: int = 8              # FFT zero-padding
    search_half_width_hz: float = 25.0    # search window around tone


# ----------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------

def read_stereo_wav(raw: bytes):
    """Return (L, R, sr) as float64 arrays from WAV bytes."""
    sr, data = wavfile.read(io.BytesIO(raw))
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("Expected a stereo WAV file (2 channels).")
    L = data[:, 0].astype(np.float64)
    R = data[:, 1].astype(np.float64)
    del data
    return L, R, int(sr)


# ----------------------------------------------------------------------
# Burst detection
# ----------------------------------------------------------------------

def detect_bursts(L, R, sr, p: Params):
    """Detect tone-burst regions from the stereo envelope.

    Returns (bursts, env_t, env_L, env_R, threshold) where bursts is a list
    of (start_sample, end_sample) in file order.
    """
    win = max(1, int(p.envelope_win_s * sr))
    n_win = min(len(L), len(R)) // win

    def envelope(x):
        x = x[: n_win * win].reshape(n_win, win)
        return np.sqrt(np.mean(x ** 2, axis=1))

    eL = envelope(L)
    eR = envelope(R)
    thresh = p.envelope_thresh_frac * max(eL.max(), eR.max())
    active = (eL > thresh) | (eR > thresh)

    idx = np.where(active)[0]
    if len(idx) == 0:
        raise RuntimeError("No active signal found in file -- check the WAV.")

    segments = []
    start = prev = idx[0]
    for i in idx[1:]:
        if i - prev > 1:
            segments.append((start, prev))
            start = i
        prev = i
    segments.append((start, prev))

    bursts = []
    for s, e in segments:
        s_samp, e_samp = s * win, (e + 1) * win
        if (e_samp - s_samp) / sr >= p.burst_min_duration_s:
            bursts.append((s_samp, e_samp))

    env_t = (np.arange(n_win) + 0.5) * win / sr
    return bursts, env_t, eL, eR, thresh


# ----------------------------------------------------------------------
# Per-burst tone analysis
# ----------------------------------------------------------------------

def analyze_tone(x, sr, p: Params):
    """(magnitude, phase_rad, refined_freq_hz) of the dominant tone near
    tone_freq_hz: Hann-windowed, zero-padded FFT, centroid refinement."""
    n = len(x)
    xw = x * np.hanning(n)
    n_fft = int(2 ** np.ceil(np.log2(n * p.zero_pad_factor)))
    spec = np.fft.rfft(xw, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    lo = p.tone_freq_hz - p.search_half_width_hz
    hi = p.tone_freq_hz + p.search_half_width_hz
    band_mask = (freqs >= lo) & (freqs <= hi)
    mags = np.abs(spec[band_mask])
    band_freqs = freqs[band_mask]
    band_spec = spec[band_mask]

    peak_local = int(np.argmax(mags))
    nb = 5
    i0 = max(0, peak_local - nb)
    i1 = min(len(mags), peak_local + nb + 1)
    w = mags[i0:i1] ** 2
    refined_freq = np.sum(band_freqs[i0:i1] * w) / np.sum(w)

    mag_corrected = mags[peak_local] / (0.5 * n / 2)   # Hann coherent gain 0.5
    phase = np.angle(band_spec[peak_local])
    return mag_corrected, phase, refined_freq


def wrap_deg(angle_deg):
    """Wrap an angle to (-180, 180]."""
    return ((angle_deg + 180.0) % 360.0) - 180.0


def quadratic_vertex(x, y):
    """Vertex x of a quadratic through the given points."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    a, b, _ = np.polyfit(x, y, 2)
    if abs(a) < 1e-12:
        return x[np.argmin(y)]
    return -b / (2 * a)


# ----------------------------------------------------------------------
# Full analysis
# ----------------------------------------------------------------------

def analyze(L, R, sr, p: Params):
    """Run burst detection + per-band analysis.

    Returns dict(results=[...], n_detected=int, env=(t, eL, eR, thresh),
    bursts=[(s,e),...]).  Bands beyond the 22 defined by the record have no
    azimuth and are dropped from `results` (count reported via n_detected).
    """
    bursts, env_t, eL, eR, thresh = detect_bursts(L, R, sr, p)

    results = []
    for band_num, (s, e) in enumerate(bursts, start=1):
        if band_num > N_BANDS:
            break
        direction = DIRECTIONS[band_num - 1]
        azimuth = direction + AZIMUTH_OFFSET_DEG

        dur = e - s
        trim = int(dur * p.steady_state_trim_frac)
        seg_start = s + trim
        seg_end = e - trim
        max_len = int(p.analysis_window_s * sr)
        if seg_end - seg_start > max_len:
            mid = (seg_start + seg_end) // 2
            seg_start = mid - max_len // 2
            seg_end = mid + max_len // 2

        magL, phL, _ = analyze_tone(L[seg_start:seg_end], sr, p)
        magR, phR, _ = analyze_tone(R[seg_start:seg_end], sr, p)

        if magL >= magR:
            main_ch, leak_ch = "L", "R"
            main_mag, leak_mag, main_ph, leak_ph = magL, magR, phL, phR
        else:
            main_ch, leak_ch = "R", "L"
            main_mag, leak_mag, main_ph, leak_ph = magR, magL, phR, phL

        results.append(dict(
            band=band_num, direction=direction, azimuth=azimuth,
            main=main_ch, leak=leak_ch,
            crosstalk_db=float(20 * np.log10(leak_mag / main_mag)),
            phase_diff_deg=float(wrap_deg(np.degrees(leak_ph - main_ph))),
            t_start=seg_start / sr, t_end=seg_end / sr,
        ))

    return dict(results=results, n_detected=len(bursts), bursts=bursts,
                env=(env_t, eL, eR, thresh))


def symmetry_check(results):
    """Bands 6/17 symmetry check + quadratic-refined minima.

    Returns a dict, or None if bands 6 and/or 17 are missing.
    """
    by_band = {r["band"]: r for r in results}
    b6, b17 = by_band.get(6), by_band.get(17)
    if b6 is None or b17 is None:
        return None

    part1 = [r for r in results if 1 <= r["band"] <= 11]
    part2 = [r for r in results if 12 <= r["band"] <= 22]

    def refined_min(part):
        best = min(part, key=lambda r: r["crosstalk_db"])
        nbrs = sorted(part, key=lambda r: abs(r["band"] - best["band"]))[:3]
        nbrs = sorted(nbrs, key=lambda r: r["band"])
        if len(nbrs) < 3:
            return best["azimuth"], best
        return quadratic_vertex([r["azimuth"] for r in nbrs],
                                [r["crosstalk_db"] for r in nbrs]), best

    az1, best1 = refined_min(part1)
    az2, best2 = refined_min(part2)
    off1, off2 = az1 - 90.0, az2 - 90.0
    mean_off = (off1 + off2) / 2.0
    spread = abs(off1 - off2)

    if spread > 1.0:
        verdict = ("The two sweeps disagree on the minimum's location by more "
                   "than 1 deg -- this points to something beyond a simple "
                   "azimuth offset (channel gain imbalance, asymmetric "
                   "cartridge crosstalk behaviour, or measurement noise) "
                   "rather than a clean azimuth correction.")
    else:
        verdict = (f"Both sweeps agree: residual azimuth error is "
                   f"approximately {mean_off:+.2f} deg from vertical. ")
        if abs(mean_off) > 0.05:
            d = ("clockwise (tilt left edge down)" if mean_off > 0
                 else "counter-clockwise (tilt right edge down)")
            verdict += (f"Suggested correction direction (viewed from the "
                        f"front of the headshell): {d}.")
        else:
            verdict += "Azimuth is essentially centred -- no correction needed."

    return dict(b6=b6, b17=b17, best1=best1, best2=best2,
                az_min_1=az1, az_min_2=az2, offset1=off1, offset2=off2,
                mean_offset=mean_off, spread=spread, verdict=verdict)


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

def make_figure(results, symmetry, title_name="", description=""):
    """Three-panel dark figure (crosstalk, phase, summary table)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lr = sorted((r for r in results if r["main"] == "L"), key=lambda r: r["azimuth"])
    rl = sorted((r for r in results if r["main"] == "R"), key=lambda r: r["azimuth"])
    groups = (("L->R", lr), ("R->L", rl))

    with plt.style.context("dark_background"), plt.rc_context({
        "font.size": 13, "axes.titlesize": 15, "axes.labelsize": 14,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    }):
        fig = plt.figure(figsize=(10, 12.5))
        gs = fig.add_gridspec(3, 1, height_ratios=[3, 3, 2.4], hspace=0.45)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax3 = fig.add_subplot(gs[2])
        fig.suptitle("MEROLEMEZ Azimuth Crosstalk Analysis", fontsize=18,
                     fontweight="bold", y=0.985)

        subtitle = title_name or ""
        if description:
            subtitle += f"\n{description}" if subtitle else description
        if subtitle:
            fig.text(0.5, 0.955, subtitle, ha="center", va="top",
                     fontsize=11.5, color="white")

        for label, g in groups:
            if g:
                ax1.plot([r["azimuth"] for r in g], [r["crosstalk_db"] for r in g],
                         "o-", color=COLORS[label], label=label, linewidth=2, markersize=5)
        ax1.axvline(90, color="red", linestyle="--", linewidth=1, label="reference (90 deg)")
        if symmetry is not None:
            ax1.axvline(symmetry["az_min_1"], color=COLORS["L->R"], linestyle=":", linewidth=1, alpha=0.7)
            ax1.axvline(symmetry["az_min_2"], color=COLORS["R->L"], linestyle=":", linewidth=1, alpha=0.7)
        ax1.set_ylabel("Crosstalk (dB)")
        ax1.set_title("Crosstalk vs. Recorded Azimuth")
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.3)

        for label, g in groups:
            if g:
                az = np.array([r["azimuth"] for r in g])
                ph = np.array([r["phase_diff_deg"] for r in g])
                ax2.plot(az, ph, "-", color=COLORS[label], linewidth=2)
                ax2.plot(az, ph, "o", color=COLORS[label], label=label, markersize=6)

        firsts = [g[0]["phase_diff_deg"] for _, g in groups if g]
        lasts = [g[-1]["phase_diff_deg"] for _, g in groups if g]
        if firsts and lasts:
            mid = (np.mean(firsts) + np.mean(lasts)) / 2.0
            ax2.axhline(mid, color="yellow", linestyle="--", linewidth=1.5,
                        label=f"first/last midpoint ({mid:.1f} deg)")
        ax2.axvline(90, color="red", linestyle="--", linewidth=1)
        ax2.axhline(0, color="gray", linestyle=":", linewidth=1)
        ax2.set_xlabel("Azimuth (deg)")
        ax2.set_ylabel("Channel Phase Difference (deg)")
        ax2.set_title("Leakage-to-Main Phase vs. Recorded Azimuth")
        ax2.legend(loc="best")
        ax2.grid(True, alpha=0.3)
        if results:
            lo = int(np.floor(min(r["azimuth"] for r in results)))
            hi = int(np.ceil(max(r["azimuth"] for r in results)))
            ax2.set_xticks(np.arange(lo, hi + 1, 1))

        fig.tight_layout(rect=[0, 0, 1, 0.93])

        ax3.axis("off")
        if symmetry is not None:
            s = symmetry
            wrap = lambda t, w: "\n".join(textwrap.wrap(t, width=w))
            avg_az = (s["az_min_1"] + s["az_min_2"]) / 2.0
            rows = [
                [wrap("Band 6 (azimuth 90 deg, Part 1 center)", 30),
                 wrap(f"{s['b6']['crosstalk_db']:.2f} dB, phase {s['b6']['phase_diff_deg']:.2f} deg", 34)],
                [wrap("Band 17 (azimuth 90 deg, Part 2 center)", 30),
                 wrap(f"{s['b17']['crosstalk_db']:.2f} dB, phase {s['b17']['phase_diff_deg']:.2f} deg", 34)],
                [wrap("Part 1 (L->R) discrete minimum", 30),
                 wrap(f"band {s['best1']['band']}, az {round(s['az_min_1']):.0f} deg -> "
                      f"{s['az_min_1']:.2f} deg, {s['best1']['crosstalk_db']:.2f} dB, "
                      f"phase {s['best1']['phase_diff_deg']:.2f} deg", 44)],
                [wrap("Part 2 (R->L) discrete minimum", 30),
                 wrap(f"band {s['best2']['band']}, az {round(s['az_min_2']):.0f} deg -> "
                      f"{s['az_min_2']:.2f} deg, {s['best2']['crosstalk_db']:.2f} dB, "
                      f"phase {s['best2']['phase_diff_deg']:.2f} deg", 44)],
                [wrap("Average of L->R / R->L refined minima", 30),
                 wrap(f"azimuth {avg_az:.2f} deg ({s['mean_offset']:+.2f} deg from 90)", 34)],
            ]
            table = ax3.table(cellText=rows, colLabels=["Quantity", "Value"],
                              loc="center", cellLoc="left", colWidths=[0.4, 0.6])
            table.auto_set_font_size(False)
            table.set_fontsize(11.5)
            table.scale(1, 2.7)
            for (row, col), cell in table.get_celld().items():
                cell.set_edgecolor("#555555")
                cell.PAD = 0.02
                if row == 0:
                    cell.set_text_props(fontweight="bold", color="white")
                    cell.set_facecolor("#333333")
                else:
                    cell.set_text_props(color="white", wrap=True)
                    cell.set_facecolor("#111111")
                    if row == len(rows):
                        cell.set_text_props(fontweight="bold", color="#FFD54F", wrap=True)
                        cell.set_facecolor("#222222")
    return fig


def make_envelope_figure(env, bursts, sr, n_expected=N_BANDS):
    """Envelope + detected burst regions, for sanity-checking detection."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t, eL, eR, thresh = env
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(10, 3.2))
        ax.plot(t, eL, color=COLORS["L->R"], lw=1, label="L envelope")
        ax.plot(t, eR, color=COLORS["R->L"], lw=1, label="R envelope")
        ax.axhline(thresh, color="yellow", ls="--", lw=1, label="threshold")
        for i, (s, e) in enumerate(bursts, start=1):
            ax.axvspan(s / sr, e / sr, color="white", alpha=0.08)
            if i <= n_expected:
                ax.text((s + e) / 2 / sr, ax.get_ylim()[1] * 0.95, str(i),
                        ha="center", va="top", fontsize=8, color="white")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("RMS")
        ax.set_title(f"Burst detection: {len(bursts)} found (expected {n_expected})")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
    return fig
