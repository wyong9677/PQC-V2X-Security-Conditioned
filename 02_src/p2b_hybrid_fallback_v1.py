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

CFG_PATH = (
    ROOT
    / "01_config"
    / "p2b_hybrid_fallback_v1.json"
)

RESULTS_DIR = ROOT / "04_results"
FIGURES_DIR = ROOT / "05_figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def atomic_write_text(path: Path, text: str) -> None:

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_text(
        text,
        encoding="utf-8",
    )

    tmp.replace(path)


def sha256_file(path: Path) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:

        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def effective_command(
    u: float,
    tau: float,
    w: float,
) -> float:

    return u + tau * w


def raw_acceleration(
    a0,
    u: float,
    tau: float,
    w: float,
    t,
):

    ue = effective_command(
        u,
        tau,
        w,
    )

    return (
        ue
        +
        (a0 - ue)
        * np.exp(-t / tau)
    )


def raw_velocity(
    v0,
    a0,
    u: float,
    tau: float,
    w: float,
    t,
):

    ue = effective_command(
        u,
        tau,
        w,
    )

    return (
        v0
        +
        ue * t
        +
        tau
        * (a0 - ue)
        * (
            1.0
            -
            np.exp(-t / tau)
        )
    )


def raw_position(
    v0,
    a0,
    u: float,
    tau: float,
    w: float,
    t,
):

    ue = effective_command(
        u,
        tau,
        w,
    )

    return (
        v0 * t
        +
        0.5 * ue * t**2
        +
        tau
        * (a0 - ue)
        * (
            t
            -
            tau
            * (
                1.0
                -
                np.exp(-t / tau)
            )
        )
    )


def stopping_time(
    v0,
    a0,
    u: float,
    tau: float,
    w: float,
):

    v0 = np.asarray(
        v0,
        dtype=float,
    )

    a0 = np.asarray(
        a0,
        dtype=float,
    )

    ue = effective_command(
        u,
        tau,
        w,
    )

    if ue >= 0.0:
        raise ValueError(
            "stopping_time requires "
            "negative equilibrium acceleration"
        )

    def velocity(t):

        return raw_velocity(
            v0,
            a0,
            u,
            tau,
            w,
            t,
        )

    hi = (
        (
            v0
            +
            np.maximum(a0, 0.0)
            * tau
            +
            1.0
        )
        /
        (-ue)
        +
        10.0 * tau
    )

    for _ in range(16):

        mask = (
            velocity(hi) > 0.0
        )

        if not np.any(mask):
            break

        hi = np.where(
            mask,
            2.0 * hi,
            hi,
        )

    lo = np.zeros_like(
        hi
    )

    for _ in range(80):

        mid = 0.5 * (lo + hi)

        moving = (
            velocity(mid) > 0.0
        )

        lo = np.where(
            moving,
            mid,
            lo,
        )

        hi = np.where(
            moving,
            hi,
            mid,
        )

    return np.where(
        v0 <= 0.0,
        0.0,
        hi,
    )


def peak_speed(
    v0,
    a0,
    u: float,
    tau: float,
    w: float,
):

    v0 = np.asarray(
        v0,
        dtype=float,
    )

    a0 = np.asarray(
        a0,
        dtype=float,
    )

    ue = effective_command(
        u,
        tau,
        w,
    )

    if ue >= 0.0:
        raise ValueError(
            "peak_speed requires "
            "negative equilibrium acceleration"
        )

    out = np.array(
        v0,
        copy=True,
    )

    mask = a0 > 0.0

    if np.any(mask):

        t_zero = (
            tau
            *
            np.log(
                (
                    a0[mask] - ue
                )
                /
                (-ue)
            )
        )

        v_peak = raw_velocity(
            v0[mask],
            a0[mask],
            u,
            tau,
            w,
            t_zero,
        )

        out[mask] = np.maximum(
            out[mask],
            v_peak,
        )

    return out


def position_with_stop(
    v0,
    a0,
    u: float,
    tau: float,
    w: float,
    t,
    t_stop,
):

    t_eval = np.minimum(
        t,
        t_stop,
    )

    return raw_position(
        v0,
        a0,
        u,
        tau,
        w,
        t_eval,
    )


def hybrid_state_after(
    v0,
    a0,
    u: float,
    tau: float,
    w: float,
    t: float,
):

    t_stop = stopping_time(
        v0,
        a0,
        u,
        tau,
        w,
    )

    t_eval = np.minimum(
        t,
        t_stop,
    )

    p = raw_position(
        v0,
        a0,
        u,
        tau,
        w,
        t_eval,
    )

    v = raw_velocity(
        v0,
        a0,
        u,
        tau,
        w,
        t_eval,
    )

    a = raw_acceleration(
        a0,
        u,
        tau,
        w,
        t_eval,
    )

    stopped = (
        t >= t_stop
    )

    v = np.where(
        stopped,
        0.0,
        v,
    )

    a = np.where(
        stopped,
        0.0,
        a,
    )

    return p, v, a


def model_parameters(cfg):

    return {
        "uf":
            float(
                cfg["follower"][
                    "fallback_command"
                ]
            ),

        "tf":
            float(
                cfg["follower"]["tau"]
            ),

        "wf":
            float(
                cfg["follower"][
                    "worst_disturbance"
                ]
            ),

        "up":
            float(
                cfg["predecessor"][
                    "worst_command"
                ]
            ),

        "tp":
            float(
                cfg["predecessor"]["tau"]
            ),

        "wp":
            float(
                cfg["predecessor"][
                    "worst_disturbance"
                ]
            ),
    }


def relative_loss_bracket(
    vf,
    af,
    vp,
    ap,
    resolutions,
    cfg,
    chunk_size: int = 512,
):

    vf = np.asarray(
        vf,
        dtype=float,
    )

    af = np.asarray(
        af,
        dtype=float,
    )

    vp = np.asarray(
        vp,
        dtype=float,
    )

    ap = np.asarray(
        ap,
        dtype=float,
    )

    pars = model_parameters(
        cfg
    )

    tf_stop = stopping_time(
        vf,
        af,
        pars["uf"],
        pars["tf"],
        pars["wf"],
    )

    tp_stop = stopping_time(
        vp,
        ap,
        pars["up"],
        pars["tp"],
        pars["wp"],
    )

    horizon = np.maximum(
        tf_stop,
        tp_stop,
    )

    vf_peak = peak_speed(
        vf,
        af,
        pars["uf"],
        pars["tf"],
        pars["wf"],
    )

    vp_peak = peak_speed(
        vp,
        ap,
        pars["up"],
        pars["tp"],
        pars["wp"],
    )

    lipschitz = np.maximum(
        vf_peak,
        vp_peak,
    )

    n_states = len(vf)

    lower = np.full(
        n_states,
        -np.inf,
        dtype=float,
    )

    upper = np.full(
        n_states,
        np.inf,
        dtype=float,
    )

    snapshots = []

    for resolution in resolutions:

        sample_max = np.full(
            n_states,
            -np.inf,
            dtype=float,
        )

        fractions = np.linspace(
            0.0,
            1.0,
            int(resolution),
        )

        for start in range(
            0,
            n_states,
            chunk_size,
        ):

            stop = min(
                start + chunk_size,
                n_states,
            )

            sl = slice(
                start,
                stop,
            )

            t = (
                horizon[sl, None]
                *
                fractions[None, :]
            )

            sf = position_with_stop(
                vf[sl, None],
                af[sl, None],
                pars["uf"],
                pars["tf"],
                pars["wf"],
                t,
                tf_stop[sl, None],
            )

            sp = position_with_stop(
                vp[sl, None],
                ap[sl, None],
                pars["up"],
                pars["tp"],
                pars["wp"],
                t,
                tp_stop[sl, None],
            )

            sample_max[sl] = np.max(
                sf - sp,
                axis=1,
            )

        lower_now = np.maximum(
            sample_max,
            0.0,
        )

        dt = (
            horizon
            /
            (
                int(resolution)
                -
                1
            )
        )

        correction = (
            0.5
            *
            lipschitz
            *
            dt
        )

        upper_now = np.maximum(
            sample_max
            +
            correction,
            0.0,
        )

        lower = np.maximum(
            lower,
            lower_now,
        )

        upper = np.minimum(
            upper,
            upper_now,
        )

        snapshots.append(
            (
                int(resolution),
                lower.copy(),
                upper.copy(),
            )
        )

    return (
        lower,
        upper,
        snapshots,
    )


def interval_loss_bracket(
    vf,
    af,
    vp,
    ap,
    horizon: float,
    resolution: int,
    cfg,
):

    pars = model_parameters(
        cfg
    )

    vf = np.asarray(vf)
    af = np.asarray(af)
    vp = np.asarray(vp)
    ap = np.asarray(ap)

    tf_stop = stopping_time(
        vf,
        af,
        pars["uf"],
        pars["tf"],
        pars["wf"],
    )

    tp_stop = stopping_time(
        vp,
        ap,
        pars["up"],
        pars["tp"],
        pars["wp"],
    )

    fractions = np.linspace(
        0.0,
        1.0,
        int(resolution),
    )

    t = (
        horizon
        *
        fractions[None, :]
    )

    sf = position_with_stop(
        vf[:, None],
        af[:, None],
        pars["uf"],
        pars["tf"],
        pars["wf"],
        t,
        tf_stop[:, None],
    )

    sp = position_with_stop(
        vp[:, None],
        ap[:, None],
        pars["up"],
        pars["tp"],
        pars["wp"],
        t,
        tp_stop[:, None],
    )

    sample_max = np.max(
        sf - sp,
        axis=1,
    )

    vf_peak = peak_speed(
        vf,
        af,
        pars["uf"],
        pars["tf"],
        pars["wf"],
    )

    vp_peak = peak_speed(
        vp,
        ap,
        pars["up"],
        pars["tp"],
        pars["wp"],
    )

    L = np.maximum(
        vf_peak,
        vp_peak,
    )

    dt = (
        horizon
        /
        (
            int(resolution)
            -
            1
        )
    )

    lower = np.maximum(
        sample_max,
        0.0,
    )

    upper = np.maximum(
        sample_max
        +
        0.5 * L * dt,
        0.0,
    )

    return lower, upper


def sample_states(
    power: int,
    seed: int,
    cfg,
):

    sampler = qmc.Sobol(
        d=5,
        scramble=True,
        seed=seed,
    )

    U = sampler.random_base2(
        int(power)
    )

    sd = cfg[
        "state_domain"
    ]

    lower = np.array(
        [
            sd["d_min"],
            sd["delta_v_min"],
            sd["v_f_min"],
            sd["a_f_min"],
            sd["a_p_min"],
        ],
        dtype=float,
    )

    upper = np.array(
        [
            sd["d_max"],
            sd["delta_v_max"],
            sd["v_f_max"],
            sd["a_f_max"],
            sd["a_p_max"],
        ],
        dtype=float,
    )

    states = (
        lower
        +
        U
        *
        (
            upper - lower
        )
    )

    return states, lower, upper


def physical_subset(
    states,
    cfg,
):

    vf = states[:, 2]
    dv = states[:, 1]

    vp = vf + dv

    sd = cfg[
        "state_domain"
    ]

    mask = (
        (vp >= sd["v_p_min"])
        &
        (vp <= sd["v_p_max"])
    )

    return mask, vp


def evaluate_kernel_envelope(
    states,
    cfg,
    resolutions,
):

    mask, vp_all = (
        physical_subset(
            states,
            cfg,
        )
    )

    X = states[mask]

    vp = vp_all[mask]

    d = X[:, 0]
    vf = X[:, 2]
    af = X[:, 3]
    ap = X[:, 4]

    lower_loss, upper_loss, snaps = (
        relative_loss_bracket(
            vf,
            af,
            vp,
            ap,
            resolutions,
            cfg,
        )
    )

    d_min = float(
        cfg["state_domain"][
            "d_min"
        ]
    )

    inner = (
        d
        >=
        d_min
        +
        upper_loss
    )

    outer = (
        d
        >=
        d_min
        +
        lower_loss
    )

    convergence = []

    for (
        resolution,
        lower_n,
        upper_n,
    ) in snaps:

        inner_n = (
            d
            >=
            d_min
            +
            upper_n
        )

        outer_n = (
            d
            >=
            d_min
            +
            lower_n
        )

        width = (
            upper_n
            -
            lower_n
        )

        convergence.append(
            {
                "resolution":
                    int(resolution),

                "inner_fraction":
                    float(
                        np.mean(
                            inner_n
                        )
                    ),

                "outer_fraction":
                    float(
                        np.mean(
                            outer_n
                        )
                    ),

                "ambiguity_fraction":
                    float(
                        np.mean(
                            outer_n
                            &
                            ~inner_n
                        )
                    ),

                "gap_width_mean_m":
                    float(
                        np.mean(
                            width
                        )
                    ),

                "gap_width_p95_m":
                    float(
                        np.quantile(
                            width,
                            0.95,
                        )
                    ),

                "gap_width_max_m":
                    float(
                        np.max(
                            width
                        )
                    ),
            }
        )

    return {
        "mask":
            mask,

        "physical_states":
            X,

        "vp":
            vp,

        "loss_lower":
            lower_loss,

        "loss_upper":
            upper_loss,

        "inner":
            inner,

        "outer":
            outer,

        "convergence":
            convergence,
    }


def bellman_audit(
    states,
    cfg,
):

    mask, vp_all = (
        physical_subset(
            states,
            cfg,
        )
    )

    X = states[mask]

    vp = vp_all[mask]

    count = min(
        int(
            cfg["bellman"][
                "sample_count"
            ]
        ),
        len(X),
    )

    X = X[:count]
    vp = vp[:count]

    vf = X[:, 2]
    af = X[:, 3]
    ap = X[:, 4]

    resolutions = (
        cfg[
            "temporal_bracket"
        ][
            "resolutions"
        ]
    )

    lb0, ub0, _ = (
        relative_loss_bracket(
            vf,
            af,
            vp,
            ap,
            resolutions,
            cfg,
        )
    )

    pars = model_parameters(
        cfg
    )

    Ts = float(
        cfg["sampling"]["Ts"]
    )

    sf, vf1, af1 = (
        hybrid_state_after(
            vf,
            af,
            pars["uf"],
            pars["tf"],
            pars["wf"],
            Ts,
        )
    )

    sp, vp1, ap1 = (
        hybrid_state_after(
            vp,
            ap,
            pars["up"],
            pars["tp"],
            pars["wp"],
            Ts,
        )
    )

    relative_shift = (
        sf - sp
    )

    lb1, ub1, _ = (
        relative_loss_bracket(
            vf1,
            af1,
            vp1,
            ap1,
            resolutions,
            cfg,
        )
    )

    int_lb, int_ub = (
        interval_loss_bracket(
            vf,
            af,
            vp,
            ap,
            Ts,
            int(
                cfg["bellman"][
                    "interval_resolution"
                ]
            ),
            cfg,
        )
    )

    rhs_lb = np.maximum(
        int_lb,
        relative_shift + lb1,
    )

    rhs_ub = np.maximum(
        int_ub,
        relative_shift + ub1,
    )

    disjoint = np.maximum(
        np.maximum(
            lb0 - rhs_ub,
            rhs_lb - ub0,
        ),
        0.0,
    )

    return {
        "sample_count":
            int(count),

        "max_disjoint_gap":
            float(
                np.max(
                    disjoint
                )
            ),

        "mean_initial_width":
            float(
                np.mean(
                    ub0 - lb0
                )
            ),

        "mean_recursive_width":
            float(
                np.mean(
                    rhs_ub - rhs_lb
                )
            ),
    }


def predecessor_dominance_audit(
    rng,
    cfg,
):

    n = int(
        cfg["dominance"][
            "trials"
        ]
    )

    pred = cfg[
        "predecessor"
    ]

    sd = cfg[
        "state_domain"
    ]

    worst_u = float(
        pred[
            "worst_command"
        ]
    )

    worst_w = float(
        pred[
            "worst_disturbance"
        ]
    )

    tau = float(
        pred["tau"]
    )

    max_u = float(
        pred[
            "dominance_command_max"
        ]
    )

    max_w = float(
        pred[
            "disturbance_max"
        ]
    )

    violations = []

    for _ in range(n):

        v0 = float(
            rng.uniform(
                0.1,
                sd["v_p_max"],
            )
        )

        a0 = float(
            rng.uniform(
                sd["a_p_min"],
                sd["a_p_max"],
            )
        )

        u = float(
            rng.uniform(
                worst_u,
                max_u,
            )
        )

        w = float(
            rng.uniform(
                worst_w,
                max_w,
            )
        )

        ts_worst = float(
            stopping_time(
                v0,
                a0,
                worst_u,
                tau,
                worst_w,
            )
        )

        ts_test = float(
            stopping_time(
                v0,
                a0,
                u,
                tau,
                w,
            )
        )

        horizon = max(
            ts_worst,
            ts_test,
        )

        t = float(
            rng.uniform(
                0.0,
                horizon,
            )
        )

        sw = float(
            position_with_stop(
                v0,
                a0,
                worst_u,
                tau,
                worst_w,
                t,
                ts_worst,
            )
        )

        st = float(
            position_with_stop(
                v0,
                a0,
                u,
                tau,
                w,
                t,
                ts_test,
            )
        )

        violations.append(
            sw - st
        )

    return {
        "max_position_dominance_violation":
            float(
                max(
                    violations
                )
            ),

        "positive_violation_count":
            int(
                np.count_nonzero(
                    np.asarray(
                        violations
                    )
                    >
                    float(
                        cfg[
                            "dominance"
                        ][
                            "tolerance"
                        ]
                    )
                )
            ),
    }


def replicate_audit(
    cfg,
):

    reps = int(
        cfg["qmc"][
            "replicates"
        ]
    )

    power = int(
        cfg["qmc"][
            "replicate_power"
        ]
    )

    resolutions = [
        257,
        513,
    ]

    inner_values = []
    outer_values = []

    base = int(
        cfg["random_seed"]
    )

    for j in range(reps):

        states, _, _ = (
            sample_states(
                power,
                base + 1000 + j,
                cfg,
            )
        )

        result = (
            evaluate_kernel_envelope(
                states,
                cfg,
                resolutions,
            )
        )

        inner_values.append(
            float(
                np.mean(
                    result["inner"]
                )
            )
        )

        outer_values.append(
            float(
                np.mean(
                    result["outer"]
                )
            )
        )

    return {
        "inner_mean":
            float(
                np.mean(
                    inner_values
                )
            ),

        "inner_std":
            float(
                np.std(
                    inner_values,
                    ddof=1,
                )
                if reps > 1
                else 0.0
            ),

        "outer_mean":
            float(
                np.mean(
                    outer_values
                )
            ),

        "outer_std":
            float(
                np.std(
                    outer_values,
                    ddof=1,
                )
                if reps > 1
                else 0.0
            ),
    }


def write_csv(
    path,
    rows,
):

    if not rows:
        return

    with path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


def make_convergence_figure(
    path,
    convergence,
):

    x = [
        row["resolution"]
        for row
        in convergence
    ]

    inner = [
        row["inner_fraction"]
        for row
        in convergence
    ]

    outer = [
        row["outer_fraction"]
        for row
        in convergence
    ]

    ambiguity = [
        row["ambiguity_fraction"]
        for row
        in convergence
    ]

    fig, ax = plt.subplots(
        figsize=(7.2, 4.8)
    )

    ax.plot(
        x,
        inner,
        marker="o",
        label="inner envelope",
    )

    ax.plot(
        x,
        outer,
        marker="s",
        label="outer envelope",
    )

    ax.plot(
        x,
        ambiguity,
        marker="^",
        label="ambiguity",
    )

    ax.set_xscale(
        "log",
        base=2,
    )

    ax.set_xlabel(
        "Temporal resolution"
    )

    ax.set_ylabel(
        "Fraction of physical domain"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=240,
    )

    plt.close(fig)


def make_slice_figure(
    path,
    cfg,
):

    fc = cfg["figures"]
    sd = cfg["state_domain"]

    d = np.linspace(
        sd["d_min"],
        min(
            160.0,
            sd["d_max"],
        ),
        int(
            fc["slice_d_points"]
        ),
    )

    dv = np.linspace(
        sd["delta_v_min"],
        sd["delta_v_max"],
        int(
            fc[
                "slice_delta_v_points"
            ]
        ),
    )

    D, DV = np.meshgrid(
        d,
        dv,
        indexing="xy",
    )

    VF = np.full_like(
        D,
        float(
            fc["slice_v_f"]
        ),
    )

    AF = np.full_like(
        D,
        float(
            fc["slice_a_f"]
        ),
    )

    AP = np.full_like(
        D,
        float(
            fc["slice_a_p"]
        ),
    )

    states = np.column_stack(
        (
            D.ravel(),
            DV.ravel(),
            VF.ravel(),
            AF.ravel(),
            AP.ravel(),
        )
    )

    result = (
        evaluate_kernel_envelope(
            states,
            cfg,
            [
                int(
                    fc[
                        "slice_resolution"
                    ]
                )
            ],
        )
    )

    mask = result["mask"]

    classification = np.full(
        len(states),
        -1,
        dtype=int,
    )

    physical_indices = np.flatnonzero(
        mask
    )

    classification[
        physical_indices
    ] = 0

    classification[
        physical_indices[
            result["outer"]
        ]
    ] = 1

    classification[
        physical_indices[
            result["inner"]
        ]
    ] = 2

    Z = classification.reshape(
        D.shape
    )

    fig, ax = plt.subplots(
        figsize=(7.4, 5.2)
    )

    ax.contourf(
        D,
        DV,
        Z,
        levels=[
            -1.5,
            -0.5,
            0.5,
            1.5,
            2.5,
        ],
        alpha=0.85,
    )

    ax.set_xlabel(
        "Spacing d (m)"
    )

    ax.set_ylabel(
        "Relative velocity Δv = v_p - v_f (m/s)"
    )

    ax.set_title(
        "Hybrid fallback-kernel envelope"
    )

    ax.grid(
        True,
        alpha=0.20,
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=240,
    )

    plt.close(fig)


def main():

    cfg = json.loads(
        CFG_PATH.read_text(
            encoding="utf-8"
        )
    )

    rng = np.random.default_rng(
        int(
            cfg["random_seed"]
        )
    )

    resolutions = [
        int(x)
        for x
        in cfg[
            "temporal_bracket"
        ][
            "resolutions"
        ]
    ]

    states, lower_box, upper_box = (
        sample_states(
            int(
                cfg["qmc"][
                    "main_power"
                ]
            ),
            int(
                cfg["random_seed"]
            ),
            cfg,
        )
    )

    result = (
        evaluate_kernel_envelope(
            states,
            cfg,
            resolutions,
        )
    )

    convergence = (
        result[
            "convergence"
        ]
    )

    final_conv = (
        convergence[-1]
    )

    inner_fraction = float(
        np.mean(
            result["inner"]
        )
    )

    outer_fraction = float(
        np.mean(
            result["outer"]
        )
    )

    ambiguity_fraction = float(
        np.mean(
            result["outer"]
            &
            ~result["inner"]
        )
    )

    bracket_width = (
        result["loss_upper"]
        -
        result["loss_lower"]
    )

    ordered_violations = int(
        np.count_nonzero(
            result["loss_lower"]
            >
            result["loss_upper"]
            +
            1e-10
        )
    )

    subset_violations = int(
        np.count_nonzero(
            result["inner"]
            &
            ~result["outer"]
        )
    )

    bellman = bellman_audit(
        states,
        cfg,
    )

    dominance = (
        predecessor_dominance_audit(
            rng,
            cfg,
        )
    )

    replicate = (
        replicate_audit(
            cfg
        )
    )

    ambiguity_tol = float(
        cfg[
            "temporal_bracket"
        ][
            "ambiguity_fraction_tolerance"
        ]
    )

    p95_tol = float(
        cfg[
            "temporal_bracket"
        ][
            "p95_gap_width_tolerance_m"
        ]
    )

    replicate_tol = float(
        cfg["qmc"][
            "replicate_std_tolerance"
        ]
    )

    bellman_tol = float(
        cfg["bellman"][
            "overlap_tolerance"
        ]
    )

    checks = {

        "PHYSICAL_SAMPLE_NONEMPTY":
            (
                len(
                    result[
                        "physical_states"
                    ]
                )
                >
                0
            ),

        "TEMPORAL_BRACKET_ORDERED":
            (
                ordered_violations
                ==
                0
            ),

        "INNER_SUBSET_OUTER":
            (
                subset_violations
                ==
                0
            ),

        "KERNEL_INNER_NONEMPTY":
            (
                inner_fraction
                >
                0.0
            ),

        "KERNEL_OUTER_NOT_FULL":
            (
                outer_fraction
                <
                1.0
            ),

        "TEMPORAL_AMBIGUITY_SMALL":
            (
                ambiguity_fraction
                <=
                ambiguity_tol
            ),

        "TEMPORAL_P95_WIDTH_SMALL":
            (
                float(
                    np.quantile(
                        bracket_width,
                        0.95,
                    )
                )
                <=
                p95_tol
            ),

        "BELLMAN_BRACKET_CONSISTENT":
            (
                bellman[
                    "max_disjoint_gap"
                ]
                <=
                bellman_tol
            ),

        "PREDECESSOR_EXTREMAL_DOMINANCE":
            (
                dominance[
                    "positive_violation_count"
                ]
                ==
                0
            ),

        "QMC_REPLICATE_STABLE":
            (
                replicate[
                    "inner_std"
                ]
                <=
                replicate_tol
                and
                replicate[
                    "outer_std"
                ]
                <=
                replicate_tol
            ),
    }

    status = (
        "PASS"
        if all(
            checks.values()
        )
        else
        "FAIL"
    )

    physical_fraction_box = float(
        np.mean(
            result["mask"]
        )
    )

    box_volume = float(
        np.prod(
            upper_box
            -
            lower_box
        )
    )

    physical_volume = (
        box_volume
        *
        physical_fraction_box
    )

    metrics = {

        "physical_fraction_box":
            physical_fraction_box,

        "physical_volume":
            physical_volume,

        "kernel_inner_fraction_physical":
            inner_fraction,

        "kernel_outer_fraction_physical":
            outer_fraction,

        "kernel_ambiguity_fraction_physical":
            ambiguity_fraction,

        "kernel_inner_volume":
            physical_volume
            *
            inner_fraction,

        "kernel_outer_volume":
            physical_volume
            *
            outer_fraction,

        "gap_bracket_mean_m":
            float(
                np.mean(
                    bracket_width
                )
            ),

        "gap_bracket_p95_m":
            float(
                np.quantile(
                    bracket_width,
                    0.95,
                )
            ),

        "gap_bracket_max_m":
            float(
                np.max(
                    bracket_width
                )
            ),

        "ordered_bracket_violations":
            ordered_violations,

        "inner_outer_subset_violations":
            subset_violations,

        **{
            "bellman_" + k:
                v
            for k, v
            in bellman.items()
        },

        **{
            "dominance_" + k:
                v
            for k, v
            in dominance.items()
        },

        **{
            "qmc_" + k:
                v
            for k, v
            in replicate.items()
        },
    }

    stamp = (
        datetime.now(
            timezone.utc
        )
        .strftime(
            "%Y%m%dT%H%M%SZ"
        )
    )

    convergence_csv = (
        RESULTS_DIR
        /
        f"P2B_TEMPORAL_CONVERGENCE_{stamp}.csv"
    )

    convergence_fig = (
        FIGURES_DIR
        /
        f"P2B_TEMPORAL_CONVERGENCE_{stamp}.png"
    )

    slice_fig = (
        FIGURES_DIR
        /
        f"P2B_KERNEL_ENVELOPE_SLICE_{stamp}.png"
    )

    write_csv(
        convergence_csv,
        convergence,
    )

    make_convergence_figure(
        convergence_fig,
        convergence,
    )

    make_slice_figure(
        slice_fig,
        cfg,
    )

    output = {

        "schema":
            "SCV_P2B_HYBRID_FALLBACK_RESULT_V1",

        "status":
            status,

        "timestamp_utc":
            stamp,

        "classification":
            (
                "validated conservative inner/"
                "outer envelope of the prescribed-"
                "fallback hybrid safety kernel; "
                "formal rational certification "
                "deferred to P6"
            ),

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

        "checks":
            {
                key: bool(value)
                for key, value
                in checks.items()
            },

        "metrics":
            metrics,

        "temporal_convergence":
            convergence,

        "artifacts": {
            "convergence_csv":
                str(
                    convergence_csv
                ),

            "convergence_figure":
                str(
                    convergence_fig
                ),

            "kernel_slice_figure":
                str(
                    slice_fig
                ),
        },
    }

    result_path = (
        RESULTS_DIR
        /
        f"P2B_HYBRID_FALLBACK_{stamp}.json"
    )

    latest_path = (
        RESULTS_DIR
        /
        "P2B_LATEST.json"
    )

    text = json.dumps(
        output,
        indent=2,
        sort_keys=True,
    )

    atomic_write_text(
        result_path,
        text,
    )

    atomic_write_text(
        latest_path,
        text,
    )

    manifest_path = (
        RESULTS_DIR
        /
        f"P2B_MANIFEST_{stamp}.sha256"
    )

    files = [
        CFG_PATH,
        Path(__file__),
        result_path,
        convergence_csv,
        convergence_fig,
        slice_fig,
    ]

    manifest = "\n".join(
        (
            f"{sha256_file(path)}"
            f"  {path}"
        )
        for path
        in files
    ) + "\n"

    atomic_write_text(
        manifest_path,
        manifest,
    )

    print(
        "=== SCV P2-B HYBRID FALLBACK KERNEL ==="
    )

    for key, value in checks.items():

        print(
            f"{key}="
            f"{'PASS' if value else 'FAIL'}"
        )

    for key, value in metrics.items():

        if isinstance(
            value,
            (float, int),
        ):

            print(
                f"{key.upper()}="
                f"{value:.12g}"
            )

    print(
        "=== TEMPORAL CONVERGENCE ==="
    )

    for row in convergence:

        print(
            "RESOLUTION="
            f"{row['resolution']} "
            "INNER="
            f"{row['inner_fraction']:.9f} "
            "OUTER="
            f"{row['outer_fraction']:.9f} "
            "AMBIGUITY="
            f"{row['ambiguity_fraction']:.9f} "
            "P95_WIDTH_M="
            f"{row['gap_width_p95_m']:.9f}"
        )

    print(
        f"P2B_HYBRID_FALLBACK={status}"
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
