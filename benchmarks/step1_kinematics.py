"""Step 1, item 4: kinematics (hypothesis B) -- where the fixed spline can
beat DP at recovering velocity/acceleration; Kalman (CA + RTS) as a third
reference point.

Synthetic trajectories with known (analytical, closed-form) v and a:
acceleration/braking (constant tangential acceleration), turning
(constant speed along a circular arc -- centripetal acceleration),
stopping. Segment layout is random per seed; velocity is continuous
across segment boundaries (acceleration can jump -- realistic).
Observations: position + Gaussian noise 5m, uneven time step 1..5s.

Run: venv/bin/python benchmarks/step1_kinematics.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from traj.io import Track  # noqa: E402
from traj.simplify import simplify_with_indices  # noqa: E402
from traj.spline import derivatives as spline_derivatives  # noqa: E402
from traj.spline import fit as spline_fit  # noqa: E402
from _report_utils import upsert_section  # noqa: E402

TOL = 10.0
NOISE_STD = 5.0
DT_RANGE = (1.0, 5.0)
ACCEL_THRESHOLD = 3.0  # m/s^2
SEED = 42
N_TRACKS = 30
KALMAN_Q_CANDIDATES = [0.01, 0.1, 1.0, 10.0, 100.0]
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OUT_MD = os.path.join(RESULTS_DIR, "step1.md")
SECTION_HEADER = "## 4. Kinematics (hypothesis B): where the spline beats DP"


# ---------------------------------------------------------------- motion --

@dataclass
class Segment:
    kind: str  # "accel" | "cruise" | "turn" | "stop"
    duration: float
    accel: float = 0.0
    speed: float = 0.0
    radius: float = 0.0  # signed: + left (CCW), - right (CW)


def _segment_state(seg: Segment, s: float, x0: float, y0: float, heading0: float, v0: float):
    """State (x,y,vx,vy,ax,ay,heading,v) at local time s within the segment."""
    if seg.kind == "stop":
        return x0, y0, 0.0, 0.0, 0.0, 0.0, heading0, 0.0
    if seg.kind in ("accel", "cruise"):
        a = seg.accel if seg.kind == "accel" else 0.0
        v = v0 + a * s
        dist = v0 * s + 0.5 * a * s * s
        hx, hy = np.cos(heading0), np.sin(heading0)
        x, y = x0 + hx * dist, y0 + hy * dist
        vx, vy = v * hx, v * hy
        ax, ay = a * hx, a * hy
        return x, y, vx, vy, ax, ay, heading0, v
    if seg.kind == "turn":
        v = seg.speed
        omega = v / seg.radius
        theta = heading0 + omega * s
        x = x0 + (v / omega) * (np.sin(theta) - np.sin(heading0))
        y = y0 - (v / omega) * (np.cos(theta) - np.cos(heading0))
        vx, vy = v * np.cos(theta), v * np.sin(theta)
        ax, ay = -v * omega * np.sin(theta), v * omega * np.cos(theta)
        return x, y, vx, vy, ax, ay, theta, v
    raise ValueError(seg.kind)


def _random_segments(rng: np.random.Generator) -> list[Segment]:
    """Random layout of accel/brake/turn/stop with velocity continuous
    across boundaries (acceleration can jump)."""
    segs: list[Segment] = []
    v_cur = 0.0
    n_blocks = int(rng.integers(4, 7))
    for _ in range(n_blocks):
        choice = rng.choice(["accel_cruise", "turn", "stop"])
        if choice == "stop":
            if v_cur > 0.1:
                a_dec = float(rng.uniform(1.0, 4.0))
                segs.append(Segment("accel", duration=v_cur / a_dec, accel=-a_dec))
                v_cur = 0.0
            segs.append(Segment("stop", duration=float(rng.uniform(5, 20))))
        elif choice == "turn":
            if v_cur < 2.0:
                v_target = float(rng.uniform(5, 12))
                a_acc = float(rng.uniform(1.0, 3.0))
                segs.append(Segment("accel", duration=(v_target - v_cur) / a_acc, accel=a_acc))
                v_cur = v_target
            radius = float(rng.choice([-1.0, 1.0])) * float(rng.uniform(6, 35))
            angle = float(rng.uniform(np.pi / 4, np.pi))
            segs.append(Segment("turn", duration=abs(radius) * angle / v_cur, speed=v_cur, radius=radius))
        else:
            v_target = float(rng.uniform(5, 20))
            a_acc = float(rng.uniform(0.5, 4.0))
            if abs(v_target - v_cur) > 1e-6:
                segs.append(
                    Segment(
                        "accel", duration=abs(v_target - v_cur) / a_acc,
                        accel=np.sign(v_target - v_cur) * a_acc,
                    )
                )
                v_cur = v_target
            segs.append(Segment("cruise", duration=float(rng.uniform(5, 30)), speed=v_cur))
    if v_cur > 0.1:
        a_dec = float(rng.uniform(1.0, 4.0))
        segs.append(Segment("accel", duration=v_cur / a_dec, accel=-a_dec))
    segs.append(Segment("stop", duration=5.0))
    return [s for s in segs if s.duration > 1e-6]


def _ground_truth_fn(segments: list[Segment]):
    starts = []
    x0 = y0 = heading0 = v0 = 0.0
    t_cursor = 0.0
    for seg in segments:
        starts.append((t_cursor, x0, y0, heading0, v0))
        x0, y0, vx, vy, ax, ay, heading0, v0 = _segment_state(seg, seg.duration, x0, y0, heading0, v0)
        t_cursor += seg.duration
    total_duration = t_cursor

    def f(t: float):
        idx = 0
        for i, (t0, *_rest) in enumerate(starts):
            if t >= t0:
                idx = i
            else:
                break
        t0, sx0, sy0, sh0, sv0 = starts[idx]
        seg = segments[idx]
        s = min(max(t - t0, 0.0), seg.duration)
        x, y, vx, vy, ax, ay, _h, _v = _segment_state(seg, s, sx0, sy0, sh0, sv0)
        return x, y, vx, vy, ax, ay

    return total_duration, f


def _sample_times(total_duration: float, rng: np.random.Generator) -> np.ndarray:
    times = [0.0]
    t = 0.0
    while True:
        t += rng.uniform(*DT_RANGE)
        if t >= total_duration:
            break
        times.append(t)
    times.append(total_duration)
    return np.array(times)


def make_synthetic_track(seed: int) -> tuple[Track, np.ndarray, np.ndarray]:
    """Returns (track with noisy observations, v_true, a_true) at points track.t."""
    rng = np.random.default_rng(seed)
    segments = _random_segments(rng)
    total_duration, f = _ground_truth_fn(segments)
    t = _sample_times(total_duration, rng)
    gt = np.array([f(tt) for tt in t])
    xy_true = gt[:, 0:2]
    v_true = gt[:, 2:4]
    a_true = gt[:, 4:6]
    noise = rng.normal(0.0, NOISE_STD, xy_true.shape)
    xy_obs = xy_true + noise
    zeros = np.zeros(len(t))
    track = Track(track_id=f"synthkin{seed}", lat=zeros, lon=zeros, t=t, xy=xy_obs)
    return track, v_true, a_true


# --------------------------------------------------------------- Kalman ---

def _kalman_ca_rts_1d(t: np.ndarray, z: np.ndarray, meas_std: float, q: float) -> np.ndarray:
    """Constant-acceleration Kalman + RTS smoother along one axis.
    Returns an (n, 3) array: [pos, vel, acc]."""
    n = len(t)
    R = meas_std ** 2
    H = np.array([1.0, 0.0, 0.0])

    x_filt = np.zeros((n, 3))
    P_filt = np.zeros((n, 3, 3))
    x_pred_arr = np.zeros((n, 3))
    P_pred_arr = np.zeros((n, 3, 3))

    x = np.array([z[0], 0.0, 0.0])
    P = np.diag([R, 10.0, 10.0])
    x_filt[0], P_filt[0] = x, P
    x_pred_arr[0], P_pred_arr[0] = x, P

    for k in range(1, n):
        dt = t[k] - t[k - 1]
        F = np.array([[1.0, dt, dt * dt / 2.0], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])
        Q = q * np.array(
            [
                [dt**5 / 20.0, dt**4 / 8.0, dt**3 / 6.0],
                [dt**4 / 8.0, dt**3 / 3.0, dt**2 / 2.0],
                [dt**3 / 6.0, dt**2 / 2.0, dt],
            ]
        )
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        x_pred_arr[k], P_pred_arr[k] = x_pred, P_pred

        resid = z[k] - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H / S
        x = x_pred + K * resid
        P = P_pred - np.outer(K, H @ P_pred)
        x_filt[k], P_filt[k] = x, P

    x_smooth = x_filt.copy()
    for k in range(n - 2, -1, -1):
        dt = t[k + 1] - t[k]
        F = np.array([[1.0, dt, dt * dt / 2.0], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])
        C = P_filt[k] @ F.T @ np.linalg.inv(P_pred_arr[k + 1])
        x_smooth[k] = x_filt[k] + C @ (x_smooth[k + 1] - x_pred_arr[k + 1])

    return x_smooth


def kalman_ca_rts_2d(t: np.ndarray, xy: np.ndarray, meas_std: float, q: float):
    sx = _kalman_ca_rts_1d(t, xy[:, 0], meas_std, q)
    sy = _kalman_ca_rts_1d(t, xy[:, 1], meas_std, q)
    pos = np.column_stack([sx[:, 0], sy[:, 0]])
    vel = np.column_stack([sx[:, 1], sy[:, 1]])
    acc = np.column_stack([sx[:, 2], sy[:, 2]])
    return pos, vel, acc


# ------------------------------------------------------------- reconstr ---

def _dp_pchip_derivatives(track: Track, tol: float, t_eval: np.ndarray):
    dp_xy, dp_idx = simplify_with_indices(track, tol)
    dp_t = track.t[dp_idx]
    px = PchipInterpolator(dp_t, dp_xy[:, 0])
    py = PchipInterpolator(dp_t, dp_xy[:, 1])
    v = np.column_stack([px.derivative(1)(t_eval), py.derivative(1)(t_eval)])
    a = np.column_stack([px.derivative(2)(t_eval), py.derivative(2)(t_eval)])
    return v, a


def _spline_derivatives(track: Track, tol: float, t_eval: np.ndarray):
    sp = spline_fit(track, tol=tol, parametrization="time")
    return spline_derivatives(sp, t_eval)


# ------------------------------------------------------------------ run ---

def _pick_kalman_q(tracks_and_gt) -> float:
    best_q, best_rmse = KALMAN_Q_CANDIDATES[0], float("inf")
    for q in KALMAN_Q_CANDIDATES:
        errs = []
        for track, _v_true, a_true in tracks_and_gt:
            _pos, _v, a_kf = kalman_ca_rts_2d(track.t, track.xy, NOISE_STD, q)
            errs.append(np.linalg.norm(a_kf[1:-1] - a_true[1:-1], axis=1))
        rmse = float(np.sqrt(np.mean(np.concatenate(errs) ** 2)))
        if rmse < best_rmse:
            best_q, best_rmse = q, rmse
    return best_q


def main() -> None:
    rng_master = np.random.default_rng(SEED)
    seeds = [int(s) for s in rng_master.integers(0, 1_000_000, size=N_TRACKS)]
    tracks_and_gt = [make_synthetic_track(s) for s in seeds]

    tune_subset = tracks_and_gt[:8]
    q = _pick_kalman_q(tune_subset)
    print(f"chosen Kalman q={q} (by acceleration RMSE on {len(tune_subset)} tracks)")

    v_err = {"spline": [], "dp_pchip": [], "kalman": []}
    a_err = {"spline": [], "dp_pchip": [], "kalman": []}
    a_pred_all = {"spline": [], "dp_pchip": [], "kalman": []}
    a_true_pool = []

    for track, v_true, a_true in tracks_and_gt:
        t_eval = track.t
        v_sp, a_sp = _spline_derivatives(track, TOL, t_eval)
        v_dp, a_dp = _dp_pchip_derivatives(track, TOL, t_eval)
        _pos_kf, v_kf, a_kf = kalman_ca_rts_2d(track.t, track.xy, NOISE_STD, q)

        # exclude the first/last point -- an edge effect for all three methods
        sl = slice(1, -1)
        v_err["spline"].append(np.linalg.norm(v_sp[sl] - v_true[sl], axis=1))
        v_err["dp_pchip"].append(np.linalg.norm(v_dp[sl] - v_true[sl], axis=1))
        v_err["kalman"].append(np.linalg.norm(v_kf[sl] - v_true[sl], axis=1))

        a_err["spline"].append(np.linalg.norm(a_sp[sl] - a_true[sl], axis=1))
        a_err["dp_pchip"].append(np.linalg.norm(a_dp[sl] - a_true[sl], axis=1))
        a_err["kalman"].append(np.linalg.norm(a_kf[sl] - a_true[sl], axis=1))

        a_pred_all["spline"].append(np.linalg.norm(a_sp[sl], axis=1))
        a_pred_all["dp_pchip"].append(np.linalg.norm(a_dp[sl], axis=1))
        a_pred_all["kalman"].append(np.linalg.norm(a_kf[sl], axis=1))
        a_true_pool.append(np.linalg.norm(a_true[sl], axis=1))

    a_true_pool = np.concatenate(a_true_pool)
    true_positive_mask = a_true_pool > ACCEL_THRESHOLD

    stats = {}
    for kind in ("spline", "dp_pchip", "kalman"):
        ve = np.concatenate(v_err[kind])
        ae = np.concatenate(a_err[kind])
        pred_mask = np.concatenate(a_pred_all[kind]) > ACCEL_THRESHOLD
        tp = int((pred_mask & true_positive_mask).sum())
        fp = int((pred_mask & ~true_positive_mask).sum())
        fn = int((~pred_mask & true_positive_mask).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        stats[kind] = {
            "v_err_median": float(np.median(ve)),
            "v_err_rmse": float(np.sqrt(np.mean(ve**2))),
            "a_err_median": float(np.median(ae)),
            "a_err_rmse": float(np.sqrt(np.mean(ae**2))),
            "a_err_max": float(ae.max()),
            "precision": precision,
            "recall": recall,
        }
        print(
            f"{kind}: v_err(median/rmse)={stats[kind]['v_err_median']:.3f}/"
            f"{stats[kind]['v_err_rmse']:.3f} m/s, "
            f"a_err(median/rmse/max)={stats[kind]['a_err_median']:.3f}/"
            f"{stats[kind]['a_err_rmse']:.3f}/{stats[kind]['a_err_max']:.3f} m/s^2, "
            f"|a|>{ACCEL_THRESHOLD}: precision={precision:.3f} recall={recall:.3f}"
        )

    n_points = len(a_true_pool)
    n_positive = int(true_positive_mask.sum())

    lines = [
        f"{SECTION_HEADER}\n",
        (
            f"{N_TRACKS} synthetic trajectories (accel/brake/turn/stop, "
            f"velocity continuous across boundaries, seed={SEED}), Gaussian "
            f"noise {NOISE_STD:.0f}m on position, step {DT_RANGE[0]:.0f}..{DT_RANGE[1]:.0f}s. "
            f"Evaluated at {n_points} points (excluding the track's first/last "
            f"point -- an edge effect for all methods), of which |a_true| > "
            f"{ACCEL_THRESHOLD:.0f} m/s² for {n_positive} ({100 * n_positive / n_points:.1f}%). "
            f"Kalman: constant-acceleration + RTS, q={q:g} (chosen by "
            f"acceleration RMSE on a subsample of {len(tune_subset)} tracks).\n"
        ),
        (
            "| Method | Velocity error, m/s (median/RMSE) | Acceleration error, m/s² (median/RMSE) | "
            f"Precision \\|a\\|>{ACCEL_THRESHOLD:.0f} | Recall |"
        ),
        "|---|---|---|---|---|",
    ]
    names = {"spline": "fixed spline", "dp_pchip": "DP + PCHIP over vertices", "kalman": "Kalman (CA+RTS)"}
    for kind in ("spline", "dp_pchip", "kalman"):
        s = stats[kind]
        lines.append(
            f"| {names[kind]} | {s['v_err_median']:.3f} / {s['v_err_rmse']:.3f} | "
            f"{s['a_err_median']:.3f} / {s['a_err_rmse']:.3f} | "
            f"{s['precision']:.3f} | {s['recall']:.3f} |"
        )
    lines.append("")

    best_v = min(stats, key=lambda k: stats[k]["v_err_median"])
    best_a = min(stats, key=lambda k: stats[k]["a_err_median"])
    lines.append(
        f"Best median velocity error: {names[best_v]}. Best median "
        f"acceleration error: {names[best_a]}.\n"
    )
    lines.append(
        "Note: DP + PCHIP's acceleration RMSE is noticeably above its median "
        f"(max error {stats['dp_pchip']['a_err_max']:.0f} m/s² vs "
        f"{stats['spline']['a_err_max']:.1f} for the spline and "
        f"{stats['kalman']['a_err_max']:.1f} for Kalman) -- numerical "
        "fragility: when neighboring DP vertices end up close in time "
        "(a short segment between two DP break points), PCHIP's second "
        "derivative on that segment can blow up to thousands of m/s². "
        "Kalman shows the opposite picture: best error (RMSE and median), "
        "but noticeably lower recall for detecting |a| > "
        f"{ACCEL_THRESHOLD:.0f} -- the CA model's smoothing suppresses sharp "
        "acceleration changes (turns), not just noise.\n"
    )

    body = "\n".join(lines) + "\n"
    upsert_section(OUT_MD, SECTION_HEADER, body)
    print(f"\nwrote {OUT_MD}")


if __name__ == "__main__":
    main()
