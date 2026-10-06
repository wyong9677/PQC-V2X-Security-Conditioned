from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import qmc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

import p2b_hybrid_fallback_v1 as fb
import p2c_switching_guard_v1 as sw
import p3b0_freshness_service_audit_v1 as b0

CFG_PATH = ROOT / "01_config" / "p3b1_augmented_fixed_point_v1.json"
P1_CFG_PATH = ROOT / "01_config" / "p1_validation_v2.json"
P2B_CFG_PATH = ROOT / "01_config" / "p2b_hybrid_fallback_v1.json"
P2C_CFG_PATH = ROOT / "01_config" / "p2c_switching_guard_v1.json"
P3A_CFG_PATH = ROOT / "01_config" / "p3a_information_contract_v1.json"

RESULTS_DIR = ROOT / "04_results"
FIGURES_DIR = ROOT / "05_figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def lag_linear_motion(v0, a0, u0, slope, h, tau: float, w: float):
    """
    Exact motion over one segment for
        a' = -a/tau + u(t)/tau + w,
        u(t) = u0 + slope*t.
    """
    h = np.asarray(h, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    a0 = np.asarray(a0, dtype=float)
    u0 = np.asarray(u0, dtype=float)
    slope = np.asarray(slope, dtype=float)

    decay = np.exp(-h / tau)
    ka = tau * (1.0 - decay)
    kv = tau * h - tau**2 * (1.0 - decay)
    kr = (
        0.5 * tau * h**2
        - tau**2 * h
        + tau**3 * (1.0 - decay)
    )
    k4 = (
        tau * h**3 / 6.0
        - 0.5 * tau**2 * h**2
        + tau**3 * h
        - tau**4 * (1.0 - decay)
    )

    r0 = u0 / tau + w
    r1 = slope / tau

    p = v0 * h + a0 * kv + r0 * kr + r1 * k4
    v = v0 + a0 * ka + r0 * kv + r1 * kr
    a = a0 * decay + r0 * ka + r1 * kv
    return p, v, a


def predecessor_raw_motion(
    vp0, ap0, bar_u, age, t, slew_rate: float, p3a_cfg: dict
):
    """
    Worst-direction predecessor motion continuing the validated
    downward command-slew extremal from authenticated age `age`.
    """
    p = p3a_cfg["predecessor"]
    tau = float(p["tau"])
    u_min = float(p["command_min"])
    w = -float(p["disturbance_abs"])

    vp0 = np.asarray(vp0, dtype=float)
    ap0 = np.asarray(ap0, dtype=float)
    bar_u = np.asarray(bar_u, dtype=float)
    age = np.asarray(age, dtype=float)
    t = np.asarray(t, dtype=float)
    J = float(slew_rate)

    current_u = np.maximum(u_min, bar_u - J * age)

    if J > 0.0:
        to_sat = np.maximum(0.0, (current_u - u_min) / J)
    else:
        to_sat = np.full_like(current_u, np.inf)

    h1 = np.minimum(t, to_sat)
    p1, v1, a1 = lag_linear_motion(
        vp0, ap0, current_u, -J, h1, tau, w
    )

    h2 = np.maximum(t - h1, 0.0)
    p2, v2, a2 = lag_linear_motion(
        v1, a1, u_min, 0.0, h2, tau, w
    )

    return (
        p1 + p2,
        v2,
        a2,
        np.maximum(u_min, current_u - J * t),
    )


def predecessor_motion_with_stop(
    vp0,
    ap0,
    bar_u,
    age,
    times,
    slew_rate,
    p3a_cfg,
):
    """
    Worst-direction predecessor trajectory with explicit nonnegative-speed
    stop semantics.

    Key convention
    --------------
    * If the raw trajectory does not reach v=0 during the sampled horizon,
      t_stop = +inf.
    * If v(0)=0, the state is already in the stopped mode and t_stop = 0.
    * Only states with a finite stop time are zeroed after the stop event.

    This avoids using the sampling horizon itself as a sentinel, which can
    corrupt the endpoint at t == horizon.
    """

    vp0 = np.asarray(vp0, dtype=float)
    ap0 = np.asarray(ap0, dtype=float)
    bar_u = np.asarray(bar_u, dtype=float)
    age = np.asarray(age, dtype=float)
    times = np.asarray(times, dtype=float)

    if times.ndim != 1 or len(times) < 2:
        raise ValueError("times must be a one-dimensional grid")

    horizon = float(times[-1])

    _, raw_v_end, _, _ = predecessor_raw_motion(
        vp0,
        ap0,
        bar_u,
        age,
        horizon,
        slew_rate,
        p3a_cfg,
    )

    initially_stopped = vp0 <= 0.0
    crosses_by_end = (
        (~initially_stopped)
        &
        (raw_v_end <= 0.0)
    )

    # +inf is the only sentinel for "no stop inside this horizon".
    t_stop = np.full(
        vp0.shape,
        np.inf,
        dtype=float,
    )

    t_stop[initially_stopped] = 0.0

    if np.any(crosses_by_end):
        lo = np.zeros_like(vp0)
        hi = np.full_like(vp0, horizon)

        # Bisection is applied only to states that are known to have
        # crossed v=0 by the horizon.
        for _ in range(52):
            mid = 0.5 * (lo + hi)

            _, vm, _, _ = predecessor_raw_motion(
                vp0,
                ap0,
                bar_u,
                age,
                mid,
                slew_rate,
                p3a_cfg,
            )

            move_right = (
                crosses_by_end
                &
                (vm > 0.0)
            )

            move_left = (
                crosses_by_end
                &
                ~move_right
            )

            lo = np.where(
                move_right,
                mid,
                lo,
            )

            hi = np.where(
                move_left,
                mid,
                hi,
            )

        t_stop[crosses_by_end] = hi[crosses_by_end]

    T = np.minimum(
        times[None, :],
        t_stop[:, None],
    )

    P, V, A, U = predecessor_raw_motion(
        vp0[:, None],
        ap0[:, None],
        bar_u[:, None],
        age[:, None],
        T,
        slew_rate,
        p3a_cfg,
    )

    stopped = (
        np.isfinite(
            t_stop[:, None]
        )
        &
        (
            times[None, :]
            >=
            t_stop[:, None]
        )
    )

    V = np.where(
        stopped,
        0.0,
        V,
    )

    A = np.where(
        stopped,
        0.0,
        A,
    )

    return (
        P,
        V,
        A,
        U,
        t_stop,
    )


def endpoint_semantics_audit(
    cfg,
    p1_cfg,
    p2b_cfg,
    p3a_cfg,
    flat,
):
    """
    Independent invariant for the hybrid-stop implementation.

    For every grid state whose raw predecessor trajectory is still moving
    at t=Ts, the stop-aware routine must return exactly that same endpoint
    velocity and acceleration.  Such a state must never be spuriously
    zeroed.
    """

    (
        _vf,
        vp,
        _af,
        bar_a,
        bar_u,
        age,
    ) = flat

    J = float(
        cfg["information_contract"]["slew_rate"]
    )

    Ts = float(
        p1_cfg["plant"]["Ts"]
    )

    ap = b0.predecessor_acceleration_lower(
        age,
        bar_a,
        bar_u,
        J,
        p3a_cfg,
        p2b_cfg,
    )

    _, raw_v_end, raw_a_end, _ = predecessor_raw_motion(
        vp,
        ap,
        bar_u,
        age,
        Ts,
        J,
        p3a_cfg,
    )

    times = np.linspace(
        0.0,
        Ts,
        max(
            65,
            int(
                cfg["one_step"]["trajectory_points"]
            ),
        ),
    )

    _, V, A, _, t_stop = predecessor_motion_with_stop(
        vp,
        ap,
        bar_u,
        age,
        times,
        J,
        p3a_cfg,
    )

    returned_v_end = V[:, -1]
    returned_a_end = A[:, -1]

    moving = (
        (vp > 0.0)
        &
        (raw_v_end > 1.0e-10)
    )

    moving_count = int(
        np.count_nonzero(
            moving
        )
    )

    if moving_count == 0:
        return {
            "moving_endpoint_count": 0,
            "moving_endpoint_v_max_error": math.inf,
            "moving_endpoint_a_max_error": math.inf,
            "spurious_endpoint_zero_count": -1,
            "finite_stop_for_raw_moving_count": -1,
            "pass": False,
        }

    v_error = float(
        np.max(
            np.abs(
                returned_v_end[moving]
                -
                raw_v_end[moving]
            )
        )
    )

    a_error = float(
        np.max(
            np.abs(
                returned_a_end[moving]
                -
                raw_a_end[moving]
            )
        )
    )

    spurious_zero_count = int(
        np.count_nonzero(
            moving
            &
            (
                returned_v_end
                <=
                1.0e-12
            )
        )
    )

    finite_stop_for_raw_moving_count = int(
        np.count_nonzero(
            moving
            &
            np.isfinite(
                t_stop
            )
        )
    )

    passed = bool(
        v_error <= 1.0e-11
        and
        a_error <= 1.0e-11
        and
        spurious_zero_count == 0
        and
        finite_stop_for_raw_moving_count == 0
    )

    return {
        "moving_endpoint_count":
            moving_count,

        "moving_endpoint_v_max_error":
            v_error,

        "moving_endpoint_a_max_error":
            a_error,

        "spurious_endpoint_zero_count":
            spurious_zero_count,

        "finite_stop_for_raw_moving_count":
            finite_stop_for_raw_moving_count,

        "pass":
            passed,
    }



def build_grid(cfg):
    keys = ("v_f", "v_p", "a_f", "bar_a", "bar_u", "age")
    axes = [np.asarray(cfg["grid"][key], dtype=float) for key in keys]
    mesh = np.meshgrid(*axes, indexing="ij")
    flat = [x.reshape(-1) for x in mesh]
    shape = tuple(len(axis) for axis in axes)
    return axes, flat, shape


def cell_corner_max(nodal: np.ndarray) -> np.ndarray:
    """
    Conservative cell value: maximum over all 2^6 continuous-grid
    corners.  A final discrete-service axis, if present, is preserved.
    """
    out = nodal
    for axis in range(6):
        left = [slice(None)] * out.ndim
        right = [slice(None)] * out.ndim
        left[axis] = slice(0, -1)
        right[axis] = slice(1, None)
        out = np.maximum(out[tuple(left)], out[tuple(right)])
    return out


def locate_cells(axes, values):
    indices = []
    valid = np.ones(len(values[0]), dtype=bool)

    for axis, x in zip(axes, values):
        x = np.asarray(x, dtype=float)
        valid &= x >= axis[0] - 1e-12
        valid &= x <= axis[-1] + 1e-12

        idx = np.searchsorted(axis, x, side="right") - 1
        idx = np.clip(idx, 0, len(axis) - 2)
        indices.append(idx)

    cell_shape = tuple(len(axis) - 1 for axis in axes)
    flat_index = np.ravel_multi_index(tuple(indices), cell_shape)
    return flat_index, valid


def follower_motion(vf, af, action: float, times, p1_cfg, p2b_cfg):
    tau = float(p2b_cfg["follower"]["tau"])
    w = float(p1_cfg["uncertainty"]["follower_actuation_abs"])

    vf = np.asarray(vf, dtype=float)
    af = np.asarray(af, dtype=float)
    times = np.asarray(times, dtype=float)

    t_stop = fb.stopping_time(
        vf, af, float(action), tau, w
    )

    T = np.minimum(times[None, :], t_stop[:, None])

    P = fb.raw_position(
        vf[:, None], af[:, None], float(action), tau, w, T
    )
    V = fb.raw_velocity(
        vf[:, None], af[:, None], float(action), tau, w, T
    )
    A = fb.raw_acceleration(
        af[:, None], float(action), tau, w, T
    )

    stopped = times[None, :] >= t_stop[:, None]
    V = np.where(stopped, 0.0, V)
    A = np.where(stopped, 0.0, A)
    return P, V, A


def static_interface_audit(cfg, p1_cfg, p2b_cfg, p2c_cfg, p3a_cfg):
    failures = []

    def require(condition, message):
        if not condition:
            failures.append(message)

    require("plant" in p1_cfg, "P1 config missing plant")
    require("uncertainty" in p1_cfg, "P1 config missing uncertainty")
    require(
        "diagnostic_service_profiles" in p1_cfg,
        "P1 config missing diagnostic_service_profiles",
    )
    require("follower" in p2b_cfg, "P2B config missing follower")
    require("state_domain" in p2b_cfg, "P2B config missing state_domain")
    require("switching" in p2c_cfg, "P2C config missing switching")
    require("predecessor" in p3a_cfg, "P3A config missing predecessor")

    actions = [float(x) for x in cfg["cooperative_actions"]]
    fallback_u = float(p2b_cfg["follower"]["fallback_command"])

    require(len(actions) > 0, "empty cooperative action set")
    require(
        all(math.isfinite(x) for x in actions),
        "nonfinite cooperative action",
    )
    require(
        min(actions) > fallback_u,
        (
            "P3B1 requires cooperative protection to be "
            "strictly weaker than emergency fallback"
        ),
    )

    for key in ("v_f", "v_p", "a_f", "bar_a", "bar_u", "age"):
        axis = np.asarray(cfg["grid"][key], dtype=float)
        require(len(axis) >= 2, f"grid axis {key} too short")
        require(
            np.all(np.diff(axis) > 0.0),
            f"grid axis {key} not strictly increasing",
        )

    if failures:
        raise RuntimeError(
            "STATIC_INTERFACE_AUDIT_FAILED: " + " | ".join(failures)
        )


def build_transition_data(
    cfg,
    p1_cfg,
    p2b_cfg,
    p2c_cfg,
    p3a_cfg,
    axes,
    flat,
):
    vf, vp, af, bar_a, bar_u, age = flat
    N = len(vf)

    Ts = float(p1_cfg["plant"]["Ts"])
    J = float(cfg["information_contract"]["slew_rate"])

    # Current worst predecessor acceleration consistent with the
    # authenticated tuple and age.
    ap_lower = b0.predecessor_acceleration_lower(
        age,
        bar_a,
        bar_u,
        J,
        p3a_cfg,
        p2b_cfg,
    )

    # Certified P2-C finite-switch fallback requirement.
    _, fallback_upper, _ = sw.switching_loss_bracket(
        vf,
        af,
        vp,
        ap_lower,
        int(p2c_cfg["switching"]["N_sw"]),
        [257],
        p2c_cfg,
        p2b_cfg,
    )
    fallback_required = np.maximum(fallback_upper, 0.0)

    nt = int(cfg["one_step"]["trajectory_points"])
    times = np.linspace(0.0, Ts, nt)

    Pp, Vp, Ap, Up, _ = predecessor_motion_with_stop(
        vp,
        ap_lower,
        bar_u,
        age,
        times,
        J,
        p3a_cfg,
    )

    pp_end = Pp[:, -1]
    vp_end = Vp[:, -1]
    ap_end = Ap[:, -1]
    up_end = Up[:, -1]
    age_defer = age + Ts

    transitions = []

    # Lipschitz correction for sampled within-step closing excursion.
    sd = p2b_cfg["state_domain"]
    speed_bound = float(sd["v_f_max"]) + float(sd["v_p_max"])
    dt_sample = Ts / (nt - 1)
    lipschitz_correction = 0.5 * speed_bound * dt_sample

    for action in cfg["cooperative_actions"]:
        action = float(action)

        Pf, Vf, Af = follower_motion(
            vf,
            af,
            action,
            times,
            p1_cfg,
            p2b_cfg,
        )

        closing = Pf - Pp
        sample_max = np.max(closing, axis=1)
        step_loss_upper = np.maximum(
            sample_max + lipschitz_correction,
            0.0,
        )

        closing_end = Pf[:, -1] - pp_end
        vf_end = Vf[:, -1]
        af_end = Af[:, -1]

        defer_values = [
            vf_end,
            vp_end,
            af_end,
            bar_a,
            bar_u,
            age_defer,
        ]

        completion_values = [
            vf_end,
            vp_end,
            af_end,
            ap_end,
            up_end,
            np.zeros(N, dtype=float),
        ]

        defer_index, defer_valid = locate_cells(
            axes, defer_values
        )

        completion_index, completion_valid = locate_cells(
            axes, completion_values
        )

        transitions.append(
            {
                "action": action,
                "step_loss_upper": step_loss_upper,
                "closing_end": closing_end,
                "defer_index": defer_index,
                "defer_valid": defer_valid,
                "completion_index": completion_index,
                "completion_valid": completion_valid,
            }
        )

    return {
        "fallback_required": fallback_required,
        "ap_lower": ap_lower,
        "transitions": transitions,
        "lipschitz_correction_m": float(lipschitz_correction),
    }


def service_horizon(profile_name: str, profile: dict) -> int:
    """
    Discrete diagnostic service model:
      R=1 means completion by the next sampled update.
    This preserves ordering without silently using max_loss_burst.
    """
    bound = int(profile["eligible_bound_steps"])

    if profile_name == "ideal":
        return 1

    return max(1, bound)


def solve_profile_fixed_point(
    name: str,
    R: int,
    cfg,
    shape,
    transition_data,
):
    fallback = transition_data["fallback_required"].reshape(shape)

    # Start from fallback and monotonically enlarge the safe set by
    # decreasing required gap.
    h = np.repeat(
        fallback[..., None],
        R,
        axis=-1,
    )

    tolerance = float(cfg["fixed_point"]["tolerance_m"])
    max_iterations = int(cfg["fixed_point"]["max_iterations"])

    history = []
    monotone_violation = 0.0

    for iteration in range(1, max_iterations + 1):
        cell_max = cell_corner_max(h)
        cell_flat = [
            cell_max[..., q].reshape(-1)
            for q in range(R)
        ]

        h_new = np.repeat(
            fallback[..., None],
            R,
            axis=-1,
        )

        for r in range(1, R + 1):
            q_index = r - 1
            best = np.full(
                fallback.size,
                np.inf,
                dtype=float,
            )

            for trans in transition_data["transitions"]:
                completion_future = np.full(
                    fallback.size,
                    np.inf,
                    dtype=float,
                )

                mask = trans["completion_valid"]
                completion_future[mask] = (
                    cell_flat[R - 1][
                        trans["completion_index"][mask]
                    ]
                )

                future = completion_future

                if r > 1:
                    defer_future = np.full(
                        fallback.size,
                        np.inf,
                        dtype=float,
                    )

                    mask = trans["defer_valid"]
                    defer_future[mask] = (
                        cell_flat[r - 2][
                            trans["defer_index"][mask]
                        ]
                    )

                    # Adversary chooses completion or deferral.
                    future = np.maximum(
                        future,
                        defer_future,
                    )

                required = np.maximum(
                    trans["step_loss_upper"],
                    trans["closing_end"] + future,
                )

                best = np.minimum(
                    best,
                    required,
                )

            best = np.maximum(best, 0.0)

            # Fallback is always available.
            candidate = np.minimum(
                fallback.reshape(-1),
                best,
            )

            h_new[..., q_index] = candidate.reshape(shape)

        h_new = np.minimum(h, h_new)

        violation = float(
            np.max(h_new - h)
        )
        monotone_violation = max(
            monotone_violation,
            violation,
        )

        delta = float(
            np.max(
                np.abs(h_new - h)
            )
        )

        history.append(
            {
                "iteration": iteration,
                "sup_change_m": delta,
                "mean_required_gap_m": float(
                    np.mean(h_new[..., R - 1])
                ),
            }
        )

        h = h_new

        if delta <= tolerance:
            return {
                "name": name,
                "R": R,
                "h": h,
                "history": history,
                "converged": True,
                "iterations": iteration,
                "final_change": delta,
                "monotone_violation": monotone_violation,
            }

    return {
        "name": name,
        "R": R,
        "h": h,
        "history": history,
        "converged": False,
        "iterations": max_iterations,
        "final_change": history[-1]["sup_change_m"],
        "monotone_violation": monotone_violation,
    }


def qmc_augmented_states(power, seed, p2b_cfg, p3a_cfg):
    sampler = qmc.Sobol(
        d=7,
        scramble=True,
        seed=seed,
    )

    U = sampler.random_base2(int(power))

    sd = p2b_cfg["state_domain"]
    pc = p3a_cfg["predecessor"]

    lower = np.array(
        [
            sd["d_min"],
            sd["v_f_min"],
            sd["v_p_min"],
            sd["a_f_min"],
            pc["command_min"],
            pc["command_min"],
            0.0,
        ],
        dtype=float,
    )

    upper = np.array(
        [
            sd["d_max"],
            sd["v_f_max"],
            sd["v_p_max"],
            sd["a_f_max"],
            pc["command_max"],
            pc["command_max"],
            2.0,
        ],
        dtype=float,
    )

    return lower + U * (upper - lower)


def lookup_required_gap(h, axes, values, q_index):
    cell = cell_corner_max(h)
    flat = cell[..., q_index].reshape(-1)

    idx, valid = locate_cells(
        axes,
        values,
    )

    result = np.full(
        len(idx),
        np.inf,
        dtype=float,
    )

    result[valid] = flat[idx[valid]]
    return result


def common_domain_evaluation(
    cfg,
    p2b_cfg,
    p3a_cfg,
    axes,
    transition_data,
    profile_results,
):
    X = qmc_augmented_states(
        int(cfg["evaluation"]["qmc_power"]),
        int(cfg["random_seed"]) + 7000,
        p2b_cfg,
        p3a_cfg,
    )

    d = X[:, 0]

    values = [
        X[:, 1],
        X[:, 2],
        X[:, 3],
        X[:, 4],
        X[:, 5],
        X[:, 6],
    ]

    d_min = float(
        p2b_cfg["state_domain"]["d_min"]
    )

    shape = tuple(
        len(axis)
        for axis in axes
    )

    fallback_h = (
        transition_data["fallback_required"]
        .reshape(shape)
    )

    fallback_req = lookup_required_gap(
        fallback_h[..., None],
        axes,
        values,
        0,
    )

    verdicts = {
        "fallback":
            d >= d_min + fallback_req
    }

    required = {
        "fallback":
            fallback_req
    }

    for name, result in profile_results.items():
        R = result["R"]

        req = lookup_required_gap(
            result["h"],
            axes,
            values,
            R - 1,
        )

        required[name] = req
        verdicts[name] = (
            d >= d_min + req
        )

    fractions = {
        name: float(np.mean(verdict))
        for name, verdict
        in verdicts.items()
    }

    return {
        "X": X,
        "required": required,
        "verdicts": verdicts,
        "fractions": fractions,
    }


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def make_convergence_figure(path, profile_results):
    fig, ax = plt.subplots(
        figsize=(7.6, 4.8)
    )

    for name, result in profile_results.items():
        ax.plot(
            [
                row["iteration"]
                for row in result["history"]
            ],
            [
                row["sup_change_m"]
                for row in result["history"]
            ],
            marker="o",
            markersize=3,
            label=name,
        )

    positive_changes = [
        row["sup_change_m"]
        for result in profile_results.values()
        for row in result["history"]
        if row["sup_change_m"] > 0.0
    ]

    if positive_changes:
        ax.set_yscale("log")

    ax.set_xlabel("Predecessor iteration")
    ax.set_ylabel(
        "Sup-norm required-gap change (m)"
    )
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def make_fraction_figure(path, fractions):
    names = list(fractions.keys())
    values = [fractions[name] for name in names]

    fig, ax = plt.subplots(
        figsize=(7.5, 4.7)
    )

    ax.bar(
        np.arange(len(names)),
        values,
    )

    ax.set_xticks(
        np.arange(len(names))
    )
    ax.set_xticklabels(
        names,
        rotation=20,
    )
    ax.set_ylabel(
        "Common-domain viable fraction"
    )
    ax.grid(
        True,
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def main():
    cfg = load_json(CFG_PATH)
    p1_cfg = load_json(P1_CFG_PATH)
    p2b_cfg = load_json(P2B_CFG_PATH)
    p2c_cfg = load_json(P2C_CFG_PATH)
    p3a_cfg = load_json(P3A_CFG_PATH)

    static_interface_audit(
        cfg,
        p1_cfg,
        p2b_cfg,
        p2c_cfg,
        p3a_cfg,
    )

    axes, flat, shape = build_grid(cfg)

    stop_semantics = endpoint_semantics_audit(
        cfg,
        p1_cfg,
        p2b_cfg,
        p3a_cfg,
        flat,
    )

    print(
        "=== P3-B1-R3 STOP SEMANTICS GATE ==="
    )
    print(
        "MOVING_ENDPOINT_COUNT="
        f"{stop_semantics['moving_endpoint_count']}"
    )
    print(
        "MOVING_ENDPOINT_V_MAX_ERROR="
        f"{stop_semantics['moving_endpoint_v_max_error']:.12g}"
    )
    print(
        "MOVING_ENDPOINT_A_MAX_ERROR="
        f"{stop_semantics['moving_endpoint_a_max_error']:.12g}"
    )
    print(
        "SPURIOUS_ENDPOINT_ZERO_COUNT="
        f"{stop_semantics['spurious_endpoint_zero_count']}"
    )
    print(
        "FINITE_STOP_FOR_RAW_MOVING_COUNT="
        f"{stop_semantics['finite_stop_for_raw_moving_count']}"
    )

    if not stop_semantics["pass"]:
        raise RuntimeError(
            "P3B1_R3_STOP_SEMANTICS_GATE=FAIL"
        )

    print(
        "P3B1_R3_STOP_SEMANTICS_GATE=PASS"
    )

    transition_data = build_transition_data(
        cfg,
        p1_cfg,
        p2b_cfg,
        p2c_cfg,
        p3a_cfg,
        axes,
        flat,
    )

    profiles = p1_cfg[
        "diagnostic_service_profiles"
    ]

    profile_order = [
        "ideal",
        "diagnostic_fast",
        "diagnostic_nominal",
        "diagnostic_stressed",
    ]

    profile_results = {}

    for name in profile_order:
        R = service_horizon(
            name,
            profiles[name],
        )

        profile_results[name] = (
            solve_profile_fixed_point(
                name,
                R,
                cfg,
                shape,
                transition_data,
            )
        )

    evaluation = common_domain_evaluation(
        cfg,
        p2b_cfg,
        p3a_cfg,
        axes,
        transition_data,
        profile_results,
    )

    verdicts = evaluation["verdicts"]
    fractions = evaluation["fractions"]

    fallback = verdicts["fallback"]
    ideal = verdicts["ideal"]
    fast = verdicts["diagnostic_fast"]
    nominal = verdicts["diagnostic_nominal"]
    stressed = verdicts["diagnostic_stressed"]

    fallback_inclusion_violations = {
        name: int(
            np.count_nonzero(
                fallback
                &
                ~verdicts[name]
            )
        )
        for name in profile_order
    }

    service_order_violations = {
        "fast_not_subset_ideal":
            int(
                np.count_nonzero(
                    fast & ~ideal
                )
            ),

        "nominal_not_subset_fast":
            int(
                np.count_nonzero(
                    nominal & ~fast
                )
            ),

        "stressed_not_subset_nominal":
            int(
                np.count_nonzero(
                    stressed & ~nominal
                )
            ),
    }

    strict_gain = {
        "ideal_over_fallback":
            int(
                np.count_nonzero(
                    ideal & ~fallback
                )
            ),

        "fast_over_fallback":
            int(
                np.count_nonzero(
                    fast & ~fallback
                )
            ),

        "fast_over_stressed":
            int(
                np.count_nonzero(
                    fast & ~stressed
                )
            ),
    }

    witness_indices = np.flatnonzero(
        fast & ~stressed
    )

    witness_indices = witness_indices[
        :int(
            cfg["evaluation"][
                "witness_max_rows"
            ]
        )
    ]

    X = evaluation["X"]
    witness_rows = []

    for i in witness_indices:
        witness_rows.append(
            {
                "d": float(X[i, 0]),
                "v_f": float(X[i, 1]),
                "v_p": float(X[i, 2]),
                "a_f": float(X[i, 3]),
                "bar_a": float(X[i, 4]),
                "bar_u": float(X[i, 5]),
                "age": float(X[i, 6]),
                "fallback": int(fallback[i]),
                "ideal": int(ideal[i]),
                "fast": int(fast[i]),
                "nominal": int(nominal[i]),
                "stressed": int(stressed[i]),
                "required_gap_fast_m":
                    float(
                        evaluation[
                            "required"
                        ][
                            "diagnostic_fast"
                        ][i]
                    ),
                "required_gap_stressed_m":
                    float(
                        evaluation[
                            "required"
                        ][
                            "diagnostic_stressed"
                        ][i]
                    ),
            }
        )

    all_converged = all(
        result["converged"]
        for result
        in profile_results.values()
    )

    max_monotone_violation = max(
        result["monotone_violation"]
        for result
        in profile_results.values()
    )

    checks = {
        "STOP_SEMANTICS_GATE":
            bool(
                stop_semantics["pass"]
            ),

        "GRID_NONEMPTY":
            np.prod(shape) > 0,

        "FIXED_POINT_CONVERGED_ALL":
            all_converged,

        "MONOTONE_PREDECESSOR_ITERATION":
            max_monotone_violation
            <=
            1e-12,

        "FALLBACK_INCLUDED_ALL_PROFILES":
            sum(
                fallback_inclusion_violations.values()
            )
            ==
            0,

        "SERVICE_ORDER_NESTED":
            sum(
                service_order_violations.values()
            )
            ==
            0,

        "STRICT_COOPERATIVE_GAIN_EXISTS":
            strict_gain[
                "ideal_over_fallback"
            ]
            >
            0,

        "STRICT_SERVICE_EFFECT_EXISTS":
            strict_gain[
                "fast_over_stressed"
            ]
            >
            0,

        "COMMON_DOMAIN_FINITE":
            all(
                np.isfinite(req).all()
                for req
                in evaluation[
                    "required"
                ].values()
            ),
    }

    status = (
        "PASS"
        if all(checks.values())
        else
        "FAIL"
    )

    metrics = {
        "stop_moving_endpoint_count":
            int(
                stop_semantics["moving_endpoint_count"]
            ),

        "stop_moving_endpoint_v_max_error":
            float(
                stop_semantics["moving_endpoint_v_max_error"]
            ),

        "stop_moving_endpoint_a_max_error":
            float(
                stop_semantics["moving_endpoint_a_max_error"]
            ),

        "stop_spurious_endpoint_zero_count":
            int(
                stop_semantics["spurious_endpoint_zero_count"]
            ),

        "stop_finite_stop_for_raw_moving_count":
            int(
                stop_semantics["finite_stop_for_raw_moving_count"]
            ),

        "continuous_grid_nodes":
            int(np.prod(shape)),

        "common_qmc_samples":
            int(len(X)),

        "within_step_lipschitz_correction_m":
            float(
                transition_data[
                    "lipschitz_correction_m"
                ]
            ),

        "max_monotone_iteration_violation":
            float(
                max_monotone_violation
            ),

        **{
            f"fraction_{name}":
                float(value)
            for name, value
            in fractions.items()
        },

        **{
            f"strict_gain_{name}":
                int(value)
            for name, value
            in strict_gain.items()
        },

        **{
            f"fallback_inclusion_violation_{name}":
                int(value)
            for name, value
            in fallback_inclusion_violations.items()
        },

        **{
            f"service_order_violation_{name}":
                int(value)
            for name, value
            in service_order_violations.items()
        },
    }

    for name, result in profile_results.items():
        metrics[
            f"{name}_horizon_steps"
        ] = int(result["R"])

        metrics[
            f"{name}_iterations"
        ] = int(
            result["iterations"]
        )

        metrics[
            f"{name}_final_change_m"
        ] = float(
            result["final_change"]
        )

    stamp = (
        datetime.now(timezone.utc)
        .strftime("%Y%m%dT%H%M%SZ")
    )

    witness_csv = (
        RESULTS_DIR
        /
        f"P3B1_R3_SERVICE_WITNESSES_{stamp}.csv"
    )

    convergence_csv = (
        RESULTS_DIR
        /
        f"P3B1_R3_FIXED_POINT_CONVERGENCE_{stamp}.csv"
    )

    convergence_fig = (
        FIGURES_DIR
        /
        f"P3B1_R3_FIXED_POINT_CONVERGENCE_{stamp}.png"
    )

    fraction_fig = (
        FIGURES_DIR
        /
        f"P3B1_R3_COMMON_DOMAIN_FRACTIONS_{stamp}.png"
    )

    write_csv(
        witness_csv,
        witness_rows,
    )

    convergence_rows = []

    for name, result in profile_results.items():
        for row in result["history"]:
            convergence_rows.append(
                {
                    "profile": name,
                    **row,
                }
            )

    write_csv(
        convergence_csv,
        convergence_rows,
    )

    make_convergence_figure(
        convergence_fig,
        profile_results,
    )

    make_fraction_figure(
        fraction_fig,
        fractions,
    )

    output = {
        "schema":
            "SCV_P3B1_AUGMENTED_FIXED_POINT_RESULT_V1_R3",

        "status":
            status,

        "timestamp_utc":
            stamp,

        "classification":
            (
                "candidate pessimistic-cell finite-abstraction "
                "augmented predecessor fixed point; not yet P6 "
                "rational/Farkas certified"
            ),

        "maximal_continuous_kernel_claim":
            False,

        "p6_certified":
            False,

        "final_pqc_service_profiles":
            False,

        "checks":
            {
                key: bool(value)
                for key, value
                in checks.items()
            },

        "metrics":
            metrics,

        "fractions":
            fractions,

        "fallback_inclusion_violations":
            fallback_inclusion_violations,

        "service_order_violations":
            service_order_violations,

        "strict_gain_counts":
            strict_gain,

        "profile_summary": {
            name: {
                "horizon_steps":
                    result["R"],

                "converged":
                    result["converged"],

                "iterations":
                    result["iterations"],

                "final_change_m":
                    result["final_change"],
            }
            for name, result
            in profile_results.items()
        },

        "service_model_note":
            (
                "This diagnostic abstraction uses only "
                "eligible_bound_steps. max_loss_burst is not "
                "silently folded into the service horizon. "
                "The explicit service/loss-state automaton is "
                "deferred to E4."
            ),

        "artifacts": {
            "service_witness_csv":
                str(witness_csv),

            "convergence_csv":
                str(convergence_csv),

            "convergence_figure":
                str(convergence_fig),

            "common_domain_fraction_figure":
                str(fraction_fig),
        },

        "platform": {
            "python":
                sys.version,

            "executable":
                sys.executable,

            "system":
                platform.platform(),

            "numpy":
                np.__version__,
        },
    }

    result_path = (
        RESULTS_DIR
        /
        f"P3B1_R3_AUGMENTED_FIXED_POINT_{stamp}.json"
    )

    latest_path = (
        RESULTS_DIR
        /
        "P3B1_R3_LATEST.json"
    )

    text = json.dumps(
        output,
        indent=2,
        sort_keys=True,
    )

    atomic_write(
        result_path,
        text,
    )

    atomic_write(
        latest_path,
        text,
    )

    manifest_path = (
        RESULTS_DIR
        /
        f"P3B1_R3_MANIFEST_{stamp}.sha256"
    )

    files = [
        CFG_PATH,
        P1_CFG_PATH,
        P2B_CFG_PATH,
        P2C_CFG_PATH,
        P3A_CFG_PATH,
        Path(__file__),
        result_path,
        witness_csv,
        convergence_csv,
        convergence_fig,
        fraction_fig,
    ]

    manifest = "\n".join(
        (
            f"{sha256_file(path)}"
            f"  {path}"
        )
        for path in files
    ) + "\n"

    atomic_write(
        manifest_path,
        manifest,
    )

    print(
        "=== SCV P3-B1-R3 AUGMENTED FIXED POINT ==="
    )

    for key, value in checks.items():
        print(
            f"{key}="
            f"{'PASS' if value else 'FAIL'}"
        )

    print(
        "=== COMMON-DOMAIN FRACTIONS ==="
    )

    for key, value in fractions.items():
        print(
            f"{key.upper()}="
            f"{value:.12g}"
        )

    print(
        "=== STRICT GAINS ==="
    )

    for key, value in strict_gain.items():
        print(
            f"{key.upper()}="
            f"{value}"
        )

    print(
        "=== PROFILE FIXED POINTS ==="
    )

    for name, result in profile_results.items():
        print(
            f"{name.upper()} "
            f"R={result['R']} "
            f"ITER={result['iterations']} "
            "CONVERGED="
            f"{'YES' if result['converged'] else 'NO'} "
            "FINAL_CHANGE_M="
            f"{result['final_change']:.12g}"
        )

    print(
        f"P3B1_R3_AUGMENTED_FIXED_POINT={status}"
    )

    print(
        "MAXIMAL_CONTINUOUS_KERNEL_CLAIM=NO"
    )
    print(
        "P6_CERTIFIED=NO"
    )
    print(
        "FINAL_PQC_SERVICE_PROFILES=NO"
    )

    print(
        f"RESULT_JSON={result_path}"
    )
    print(
        f"LATEST_JSON={latest_path}"
    )
    print(
        f"MANIFEST={manifest_path}"
    )

    return (
        0
        if status == "PASS"
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
