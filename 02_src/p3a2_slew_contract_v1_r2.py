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
from scipy.integrate import solve_ivp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

CFG_PATH = (
    ROOT
    / "01_config"
    / "p3a2_slew_contract_v1.json"
)

P3A_CFG_PATH = (
    ROOT
    / "01_config"
    / "p3a_information_contract_v1.json"
)

RESULTS_DIR = ROOT / "04_results"
FIGURES_DIR = ROOT / "05_figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def atomic_write(path: Path, text: str) -> None:
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


def affine_forcing_step(
    ev: float,
    ea: float,
    h: float,
    tau: float,
    r0: float,
    r1: float,
) -> tuple[float, float]:
    """
    Exact propagation of

        e_v' = e_a
        e_a' = -e_a/tau + r0 + r1*s

    over local segment time s in [0,h].
    """

    if h <= 0.0:
        return ev, ea

    decay = math.exp(
        -h / tau
    )

    ka = (
        tau
        * (1.0 - decay)
    )

    kv = (
        tau * h
        -
        tau**2
        * (1.0 - decay)
    )

    kr = (
        0.5
        * tau
        * h**2
        -
        tau**2
        * h
        +
        tau**3
        * (1.0 - decay)
    )

    ev_new = (
        ev
        +
        ea * ka
        +
        r0 * kv
        +
        r1 * kr
    )

    ea_new = (
        ea * decay
        +
        r0 * ka
        +
        r1 * kv
    )

    return ev_new, ea_new


def predecessor_parameters(
    p3a_cfg: dict,
):
    p = p3a_cfg[
        "predecessor"
    ]

    return (
        float(p["tau"]),
        float(p["command_min"]),
        float(p["command_max"]),
        float(p["disturbance_abs"]),
    )


def instantaneous_jump_bounds(
    age: float,
    bar_u: float,
    p3a_cfg: dict,
):
    tau, u_min, u_max, w_abs = (
        predecessor_parameters(
            p3a_cfg
        )
    )

    delta_minus = (
        (bar_u - u_min)
        / tau
        +
        w_abs
    )

    delta_plus = (
        (u_max - bar_u)
        / tau
        +
        w_abs
    )

    ka = (
        tau
        *
        (
            1.0
            -
            math.exp(
                -age / tau
            )
        )
    )

    kv = (
        tau * age
        -
        tau**2
        *
        (
            1.0
            -
            math.exp(
                -age / tau
            )
        )
    )

    return {
        "ev_lower":
            -delta_minus * kv,

        "ev_upper":
            delta_plus * kv,

        "ea_lower":
            -delta_minus * ka,

        "ea_upper":
            delta_plus * ka,
    }


def extremal_slew_bounds(
    age: float,
    bar_u: float,
    slew_rate: float,
    p3a_cfg: dict,
):
    """
    Exact worst directions under

        |du/dt| <= slew_rate
        u_min <= u <= u_max

    starting from u(0)=bar_u.

    The extremal command histories are

        lower: max(u_min, bar_u - J t)
        upper: min(u_max, bar_u + J t)

    with disturbance fixed at its adverse sign.
    """

    tau, u_min, u_max, w_abs = (
        predecessor_parameters(
            p3a_cfg
        )
    )

    if age <= 0.0:
        return {
            "ev_lower": 0.0,
            "ev_upper": 0.0,
            "ea_lower": 0.0,
            "ea_upper": 0.0,
        }

    if slew_rate <= 0.0:
        # command remains at authenticated value;
        # only physical disturbance evolves.
        ev_l, ea_l = affine_forcing_step(
            0.0,
            0.0,
            age,
            tau,
            -w_abs,
            0.0,
        )

        ev_u, ea_u = affine_forcing_step(
            0.0,
            0.0,
            age,
            tau,
            w_abs,
            0.0,
        )

        return {
            "ev_lower": ev_l,
            "ev_upper": ev_u,
            "ea_lower": ea_l,
            "ea_upper": ea_u,
        }

    # Lower extremal.
    t_sat_lower = max(
        0.0,
        (bar_u - u_min)
        / slew_rate,
    )

    t1 = min(
        age,
        t_sat_lower,
    )

    ev_l = 0.0
    ea_l = 0.0

    if t1 > 0.0:
        ev_l, ea_l = (
            affine_forcing_step(
                ev_l,
                ea_l,
                t1,
                tau,
                -w_abs,
                -slew_rate / tau,
            )
        )

    if age > t1:
        r_const = (
            (u_min - bar_u)
            / tau
            -
            w_abs
        )

        ev_l, ea_l = (
            affine_forcing_step(
                ev_l,
                ea_l,
                age - t1,
                tau,
                r_const,
                0.0,
            )
        )

    # Upper extremal.
    t_sat_upper = max(
        0.0,
        (u_max - bar_u)
        / slew_rate,
    )

    t1 = min(
        age,
        t_sat_upper,
    )

    ev_u = 0.0
    ea_u = 0.0

    if t1 > 0.0:
        ev_u, ea_u = (
            affine_forcing_step(
                ev_u,
                ea_u,
                t1,
                tau,
                w_abs,
                slew_rate / tau,
            )
        )

    if age > t1:
        r_const = (
            (u_max - bar_u)
            / tau
            +
            w_abs
        )

        ev_u, ea_u = (
            affine_forcing_step(
                ev_u,
                ea_u,
                age - t1,
                tau,
                r_const,
                0.0,
            )
        )

    return {
        "ev_lower": ev_l,
        "ev_upper": ev_u,
        "ea_lower": ea_l,
        "ea_upper": ea_u,
    }


def solve_ivp_extremal(
    age: float,
    bar_u: float,
    slew_rate: float,
    direction: int,
    p3a_cfg: dict,
):
    """
    Independent high-accuracy reference integration.

    Important:
    the command trajectory has a derivative discontinuity at the
    saturation time.  The IVP is therefore integrated in separate
    smooth phases rather than allowing an adaptive Runge--Kutta step
    to cross the kink.

    This routine remains independent of affine_forcing_step().
    """

    tau, u_min, u_max, w_abs = (
        predecessor_parameters(
            p3a_cfg
        )
    )

    age = float(age)
    bar_u = float(bar_u)
    slew_rate = float(slew_rate)

    if age <= 0.0:
        return 0.0, 0.0

    if slew_rate < 0.0:
        raise ValueError(
            "slew_rate must be nonnegative"
        )

    if direction not in (-1, 1):
        raise ValueError(
            "direction must be -1 or +1"
        )

    if direction < 0:
        bound = u_min
        w = -w_abs

        if slew_rate == 0.0:
            t_sat = math.inf
        else:
            t_sat = max(
                0.0,
                (bar_u - u_min)
                / slew_rate,
            )

        def ramp_command(t):
            return (
                bar_u
                -
                slew_rate * t
            )

    else:
        bound = u_max
        w = w_abs

        if slew_rate == 0.0:
            t_sat = math.inf
        else:
            t_sat = max(
                0.0,
                (u_max - bar_u)
                / slew_rate,
            )

        def ramp_command(t):
            return (
                bar_u
                +
                slew_rate * t
            )

    def integrate_phase(
        t0: float,
        t1: float,
        y0,
        command_function,
    ):
        if t1 <= t0:
            return np.asarray(
                y0,
                dtype=float,
            )

        def rhs(t, y):
            ev, ea = y

            u = float(
                command_function(t)
            )

            delta = (
                (u - bar_u)
                / tau
                +
                w
            )

            return [
                ea,
                -ea / tau
                + delta,
            ]

        duration = t1 - t0

        max_step = min(
            0.002,
            max(
                duration / 32.0,
                1.0e-8,
            ),
        )

        sol = solve_ivp(
            rhs,
            (t0, t1),
            y0,
            method="DOP853",
            rtol=2.0e-13,
            atol=2.0e-15,
            max_step=max_step,
        )

        if not sol.success:
            raise RuntimeError(
                sol.message
            )

        return sol.y[:, -1]

    y = np.array(
        [0.0, 0.0],
        dtype=float,
    )

    # Smooth slew phase.
    t_break = min(
        age,
        t_sat,
    )

    if t_break > 0.0:
        y = integrate_phase(
            0.0,
            t_break,
            y,
            ramp_command,
        )

    # Smooth saturation phase.
    if age > t_break:

        def saturated_command(_t):
            return bound

        y = integrate_phase(
            t_break,
            age,
            y,
            saturated_command,
        )

    return (
        float(y[0]),
        float(y[1]),
    )



def ivp_cross_validation(
    rng,
    cfg,
    p3a_cfg,
):
    """
    Cross-validation against a segmented DOP853 reference.

    The audit contains:
      1. random age / authenticated-command / slew-rate trials;
      2. deterministic probes immediately before and after command
         saturation kinks.

    The returned metric is the largest absolute disagreement over
    both audit families.
    """

    _, u_min, u_max, _ = (
        predecessor_parameters(
            p3a_cfg
        )
    )

    n = int(
        cfg["validation"][
            "ivp_extremal_trials"
        ]
    )

    age_max = float(
        cfg["grid"][
            "age_max_seconds"
        ]
    )

    rates = [
        float(x)
        for x
        in cfg[
            "slew_contract"
        ][
            "sensitivity_rates"
        ]
    ]

    max_error = 0.0

    def audit_case(
        age: float,
        bar_u: float,
        J: float,
    ):
        nonlocal max_error

        if age <= 0.0:
            return

        bound = (
            extremal_slew_bounds(
                age,
                bar_u,
                J,
                p3a_cfg,
            )
        )

        lo_ev, lo_ea = (
            solve_ivp_extremal(
                age,
                bar_u,
                J,
                -1,
                p3a_cfg,
            )
        )

        hi_ev, hi_ea = (
            solve_ivp_extremal(
                age,
                bar_u,
                J,
                +1,
                p3a_cfg,
            )
        )

        max_error = max(
            max_error,

            abs(
                lo_ev
                -
                bound[
                    "ev_lower"
                ]
            ),

            abs(
                lo_ea
                -
                bound[
                    "ea_lower"
                ]
            ),

            abs(
                hi_ev
                -
                bound[
                    "ev_upper"
                ]
            ),

            abs(
                hi_ea
                -
                bound[
                    "ea_upper"
                ]
            ),
        )

    # Random audit.
    for _ in range(n):

        age = float(
            rng.uniform(
                1.0e-7,
                age_max,
            )
        )

        bar_u = float(
            rng.uniform(
                u_min,
                u_max,
            )
        )

        J = float(
            rng.choice(
                rates
            )
        )

        audit_case(
            age,
            bar_u,
            J,
        )

    # Deterministic kink audit.
    probe_rates = sorted(
        set(
            [
                rates[0],
                rates[
                    len(rates) // 2
                ],
                rates[-1],
            ]
        )
    )

    bar_grid = np.linspace(
        u_min,
        u_max,
        9,
    )

    for J in probe_rates:

        if J <= 0.0:
            continue

        for bar_u in bar_grid:

            saturation_times = [
                max(
                    0.0,
                    (bar_u - u_min)
                    / J,
                ),
                max(
                    0.0,
                    (u_max - bar_u)
                    / J,
                ),
            ]

            for t_sat in saturation_times:

                candidates = [
                    0.1,
                    0.5,
                    1.0,
                    age_max,
                ]

                if (
                    t_sat > 1.0e-8
                    and
                    t_sat < age_max
                ):
                    candidates.extend(
                        [
                            max(
                                1.0e-8,
                                t_sat
                                * (
                                    1.0
                                    -
                                    1.0e-7
                                ),
                            ),
                            min(
                                age_max,
                                t_sat
                                * (
                                    1.0
                                    +
                                    1.0e-7
                                ),
                            ),
                            min(
                                age_max,
                                t_sat
                                +
                                1.0e-6,
                            ),
                        ]
                    )

                for age in candidates:

                    if (
                        age > 0.0
                        and
                        age <= age_max
                    ):
                        audit_case(
                            float(age),
                            float(bar_u),
                            float(J),
                        )

    return max_error



def random_slew_trajectory(
    rng,
    age: float,
    bar_u: float,
    slew_rate: float,
    segments: int,
    p3a_cfg: dict,
):
    tau, u_min, u_max, w_abs = (
        predecessor_parameters(
            p3a_cfg
        )
    )

    h = (
        age / segments
    )

    ev = 0.0
    ea = 0.0

    u = bar_u

    for _ in range(segments):

        lower_next = max(
            u_min,
            u
            -
            slew_rate * h,
        )

        upper_next = min(
            u_max,
            u
            +
            slew_rate * h,
        )

        u_next = float(
            rng.uniform(
                lower_next,
                upper_next,
            )
        )

        slope = (
            (u_next - u)
            / h
        )

        w = float(
            rng.uniform(
                -w_abs,
                w_abs,
            )
        )

        r0 = (
            (u - bar_u)
            / tau
            +
            w
        )

        r1 = (
            slope / tau
        )

        ev, ea = (
            affine_forcing_step(
                ev,
                ea,
                h,
                tau,
                r0,
                r1,
            )
        )

        u = u_next

    return ev, ea


def random_containment_audit(
    rng,
    cfg,
    p3a_cfg,
):
    _, u_min, u_max, _ = (
        predecessor_parameters(
            p3a_cfg
        )
    )

    trials = int(
        cfg["validation"][
            "random_piecewise_trials"
        ]
    )

    segments = int(
        cfg["validation"][
            "segments_per_trial"
        ]
    )

    age_max = float(
        cfg["grid"][
            "age_max_seconds"
        ]
    )

    rates = [
        float(x)
        for x
        in cfg[
            "slew_contract"
        ][
            "sensitivity_rates"
        ]
    ]

    max_violation = 0.0

    max_ev_util = 0.0
    max_ea_util = 0.0

    for _ in range(trials):

        age = float(
            rng.uniform(
                1e-6,
                age_max,
            )
        )

        bar_u = float(
            rng.uniform(
                u_min,
                u_max,
            )
        )

        J = float(
            rng.choice(
                rates
            )
        )

        ev, ea = (
            random_slew_trajectory(
                rng,
                age,
                bar_u,
                J,
                segments,
                p3a_cfg,
            )
        )

        b = extremal_slew_bounds(
            age,
            bar_u,
            J,
            p3a_cfg,
        )

        max_violation = max(
            max_violation,

            b["ev_lower"] - ev,
            ev - b["ev_upper"],

            b["ea_lower"] - ea,
            ea - b["ea_upper"],
        )

        if ev < 0.0:
            denominator = abs(
                b["ev_lower"]
            )
        else:
            denominator = abs(
                b["ev_upper"]
            )

        if denominator > 0.0:
            max_ev_util = max(
                max_ev_util,
                abs(ev) / denominator,
            )

        if ea < 0.0:
            denominator = abs(
                b["ea_lower"]
            )
        else:
            denominator = abs(
                b["ea_upper"]
            )

        if denominator > 0.0:
            max_ea_util = max(
                max_ea_util,
                abs(ea) / denominator,
            )

    return {
        "max_violation":
            max_violation,

        "max_ev_random_utilization":
            max_ev_util,

        "max_ea_random_utilization":
            max_ea_util,
    }


def structural_audit(
    cfg,
    p3a_cfg,
):
    _, u_min, u_max, _ = (
        predecessor_parameters(
            p3a_cfg
        )
    )

    ages = np.linspace(
        0.0,
        float(
            cfg["grid"][
                "age_max_seconds"
            ]
        ),
        int(
            cfg["grid"][
                "audit_age_points"
            ]
        ),
    )

    bar_grid = np.linspace(
        u_min,
        u_max,
        int(
            cfg["grid"][
                "bar_u_points"
            ]
        ),
    )

    rates = sorted(
        float(x)
        for x
        in cfg[
            "slew_contract"
        ][
            "sensitivity_rates"
        ]
    )

    min_ev_age_increment = math.inf
    min_ea_age_increment = math.inf

    max_jump_subset_violation = 0.0
    max_slew_nesting_violation = 0.0

    for bar_u in bar_grid:

        for J in rates:

            ev_widths = []
            ea_widths = []

            for age in ages:

                b = (
                    extremal_slew_bounds(
                        float(age),
                        float(bar_u),
                        J,
                        p3a_cfg,
                    )
                )

                jump = (
                    instantaneous_jump_bounds(
                        float(age),
                        float(bar_u),
                        p3a_cfg,
                    )
                )

                ev_widths.append(
                    b["ev_upper"]
                    -
                    b["ev_lower"]
                )

                ea_widths.append(
                    b["ea_upper"]
                    -
                    b["ea_lower"]
                )

                max_jump_subset_violation = max(
                    max_jump_subset_violation,

                    jump["ev_lower"]
                    -
                    b["ev_lower"],

                    b["ev_upper"]
                    -
                    jump["ev_upper"],

                    jump["ea_lower"]
                    -
                    b["ea_lower"],

                    b["ea_upper"]
                    -
                    jump["ea_upper"],
                )

            min_ev_age_increment = min(
                min_ev_age_increment,
                float(
                    np.min(
                        np.diff(
                            ev_widths
                        )
                    )
                ),
            )

            min_ea_age_increment = min(
                min_ea_age_increment,
                float(
                    np.min(
                        np.diff(
                            ea_widths
                        )
                    )
                ),
            )

        for age in ages:

            previous = None

            for J in rates:

                b = (
                    extremal_slew_bounds(
                        float(age),
                        float(bar_u),
                        J,
                        p3a_cfg,
                    )
                )

                if previous is not None:

                    # Larger slew rate must produce
                    # a superset or equal interval.
                    max_slew_nesting_violation = max(
                        max_slew_nesting_violation,

                        b["ev_lower"]
                        -
                        previous["ev_lower"],

                        previous["ev_upper"]
                        -
                        b["ev_upper"],

                        b["ea_lower"]
                        -
                        previous["ea_lower"],

                        previous["ea_upper"]
                        -
                        b["ea_upper"],
                    )

                previous = b

    return {
        "min_ev_age_width_increment":
            min_ev_age_increment,

        "min_ea_age_width_increment":
            min_ea_age_increment,

        "max_jump_subset_violation":
            max_jump_subset_violation,

        "max_slew_nesting_violation":
            max_slew_nesting_violation,
    }


def reduction_summary(
    cfg,
    p3a_cfg,
):
    _, u_min, u_max, _ = (
        predecessor_parameters(
            p3a_cfg
        )
    )

    bar_grid = np.linspace(
        u_min,
        u_max,
        int(
            cfg["grid"][
                "bar_u_points"
            ]
        ),
    )

    ages = [
        float(x)
        for x
        in cfg["grid"][
            "summary_ages"
        ]
    ]

    rates = [
        float(x)
        for x
        in cfg[
            "slew_contract"
        ][
            "sensitivity_rates"
        ]
    ]

    rows = []

    for J in rates:

        for age in ages:

            ev_ratios = []
            ea_ratios = []

            for bar_u in bar_grid:

                b = (
                    extremal_slew_bounds(
                        age,
                        float(bar_u),
                        J,
                        p3a_cfg,
                    )
                )

                j = (
                    instantaneous_jump_bounds(
                        age,
                        float(bar_u),
                        p3a_cfg,
                    )
                )

                slew_ev_width = (
                    b["ev_upper"]
                    -
                    b["ev_lower"]
                )

                jump_ev_width = (
                    j["ev_upper"]
                    -
                    j["ev_lower"]
                )

                slew_ea_width = (
                    b["ea_upper"]
                    -
                    b["ea_lower"]
                )

                jump_ea_width = (
                    j["ea_upper"]
                    -
                    j["ea_lower"]
                )

                ev_ratios.append(
                    (
                        slew_ev_width
                        /
                        jump_ev_width
                    )
                    if jump_ev_width > 0.0
                    else 0.0
                )

                ea_ratios.append(
                    (
                        slew_ea_width
                        /
                        jump_ea_width
                    )
                    if jump_ea_width > 0.0
                    else 0.0
                )

            rows.append(
                {
                    "slew_rate":
                        J,

                    "age_s":
                        age,

                    "ev_ratio_mean":
                        float(
                            np.mean(
                                ev_ratios
                            )
                        ),

                    "ev_ratio_max":
                        float(
                            np.max(
                                ev_ratios
                            )
                        ),

                    "ea_ratio_mean":
                        float(
                            np.mean(
                                ea_ratios
                            )
                        ),

                    "ea_ratio_max":
                        float(
                            np.max(
                                ea_ratios
                            )
                        ),
                }
            )

    return rows


def write_csv(
    path,
    rows,
):
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


def make_error_figure(
    path,
    cfg,
    p3a_cfg,
):
    ages = np.linspace(
        0.0,
        float(
            cfg["grid"][
                "age_max_seconds"
            ]
        ),
        int(
            cfg["grid"][
                "age_points"
            ]
        ),
    )

    selected = [
        float(x)
        for x
        in cfg["grid"][
            "selected_bar_u"
        ]
    ]

    J = float(
        cfg["slew_contract"][
            "candidate_slew_rate"
        ]
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.0, 4.6),
    )

    for bar_u in selected:

        ev_l = []
        ev_u = []
        ea_l = []
        ea_u = []

        for age in ages:

            b = (
                extremal_slew_bounds(
                    float(age),
                    bar_u,
                    J,
                    p3a_cfg,
                )
            )

            ev_l.append(
                b["ev_lower"]
            )

            ev_u.append(
                b["ev_upper"]
            )

            ea_l.append(
                b["ea_lower"]
            )

            ea_u.append(
                b["ea_upper"]
            )

        axes[0].plot(
            ages,
            ev_l,
            label=f"lower, bar_u={bar_u:g}",
        )

        axes[0].plot(
            ages,
            ev_u,
            linestyle="--",
            label=f"upper, bar_u={bar_u:g}",
        )

        axes[1].plot(
            ages,
            ea_l,
            label=f"lower, bar_u={bar_u:g}",
        )

        axes[1].plot(
            ages,
            ea_u,
            linestyle="--",
            label=f"upper, bar_u={bar_u:g}",
        )

    axes[0].set_xlabel(
        "Authenticated age (s)"
    )

    axes[0].set_ylabel(
        "Velocity-error bound"
    )

    axes[1].set_xlabel(
        "Authenticated age (s)"
    )

    axes[1].set_ylabel(
        "Acceleration-error bound"
    )

    for ax in axes:
        ax.grid(
            True,
            alpha=0.25,
        )

    axes[0].legend(
        fontsize=7,
        ncol=2,
    )

    axes[1].legend(
        fontsize=7,
        ncol=2,
    )

    fig.suptitle(
        f"Slew-bounded information contract, J={J:g}"
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=240,
    )

    plt.close(fig)


def make_ratio_figure(
    path,
    rows,
    cfg,
):
    J0 = float(
        cfg["slew_contract"][
            "candidate_slew_rate"
        ]
    )

    selected = [
        row
        for row in rows
        if abs(
            row["slew_rate"]
            -
            J0
        )
        < 1e-12
    ]

    ages = [
        row["age_s"]
        for row in selected
    ]

    ev_mean = [
        row["ev_ratio_mean"]
        for row in selected
    ]

    ea_mean = [
        row["ea_ratio_mean"]
        for row in selected
    ]

    fig, ax = plt.subplots(
        figsize=(7.2, 4.7)
    )

    ax.plot(
        ages,
        ev_mean,
        marker="o",
        label="velocity interval ratio",
    )

    ax.plot(
        ages,
        ea_mean,
        marker="s",
        label="acceleration interval ratio",
    )

    ax.axhline(
        1.0,
        linestyle="--",
        label="instantaneous-jump model",
    )

    ax.set_xlabel(
        "Authenticated age (s)"
    )

    ax.set_ylabel(
        "Slew-bound width / jump-bound width"
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


def main():
    cfg = load_json(
        CFG_PATH
    )

    p3a_cfg = load_json(
        P3A_CFG_PATH
    )

    rng = np.random.default_rng(
        int(
            cfg["random_seed"]
        )
    )

    ivp_error = (
        ivp_cross_validation(
            rng,
            cfg,
            p3a_cfg,
        )
    )

    random_audit = (
        random_containment_audit(
            rng,
            cfg,
            p3a_cfg,
        )
    )

    structure = structural_audit(
        cfg,
        p3a_cfg,
    )

    rows = reduction_summary(
        cfg,
        p3a_cfg,
    )

    candidate_J = float(
        cfg["slew_contract"][
            "candidate_slew_rate"
        ]
    )

    candidate_rows = [
        row
        for row in rows
        if abs(
            row["slew_rate"]
            -
            candidate_J
        )
        < 1e-12
    ]

    checks = {
        "EXTREMAL_EXACT_VS_IVP":
            (
                ivp_error
                <=
                float(
                    cfg["validation"][
                        "ivp_tolerance"
                    ]
                )
            ),

        "RANDOM_SLEW_TRAJECTORY_CONTAINMENT":
            (
                random_audit[
                    "max_violation"
                ]
                <=
                float(
                    cfg["validation"][
                        "containment_tolerance"
                    ]
                )
            ),

        "AGE_INTERVAL_WIDTH_MONOTONE":
            (
                structure[
                    "min_ev_age_width_increment"
                ]
                >=
                -1e-12
                and
                structure[
                    "min_ea_age_width_increment"
                ]
                >=
                -1e-12
            ),

        "SLEW_CONTRACT_SUBSET_OF_JUMP_CONTRACT":
            (
                structure[
                    "max_jump_subset_violation"
                ]
                <=
                float(
                    cfg["validation"][
                        "nesting_tolerance"
                    ]
                )
            ),

        "SLEW_RATE_NESTING":
            (
                structure[
                    "max_slew_nesting_violation"
                ]
                <=
                float(
                    cfg["validation"][
                        "nesting_tolerance"
                    ]
                )
            ),

        "CANDIDATE_CONTRACT_NONTRIVIAL":
            all(
                row[
                    "ev_ratio_mean"
                ]
                <
                1.0
                and
                row[
                    "ea_ratio_mean"
                ]
                <
                1.0
                for row
                in candidate_rows
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

    stamp = (
        datetime.now(
            timezone.utc
        )
        .strftime(
            "%Y%m%dT%H%M%SZ"
        )
    )

    summary_csv = (
        RESULTS_DIR
        /
        f"P3A2_SLEW_REDUCTION_{stamp}.csv"
    )

    error_fig = (
        FIGURES_DIR
        /
        f"P3A2_SLEW_ERROR_BOUNDS_{stamp}.png"
    )

    ratio_fig = (
        FIGURES_DIR
        /
        f"P3A2_SLEW_VS_JUMP_{stamp}.png"
    )

    write_csv(
        summary_csv,
        rows,
    )

    make_error_figure(
        error_fig,
        cfg,
        p3a_cfg,
    )

    make_ratio_figure(
        ratio_fig,
        rows,
        cfg,
    )

    metrics = {
        "extremal_ivp_max_error":
            float(ivp_error),

        **random_audit,

        **structure,

        "candidate_slew_rate":
            candidate_J,
    }

    for row in candidate_rows:
        age_key = (
            str(
                row["age_s"]
            )
            .replace(".", "p")
        )

        metrics[
            f"candidate_ev_ratio_mean_age_{age_key}"
        ] = (
            row["ev_ratio_mean"]
        )

        metrics[
            f"candidate_ea_ratio_mean_age_{age_key}"
        ] = (
            row["ea_ratio_mean"]
        )

    result = {
        "schema":
            "SCV_P3A2_SLEW_CONTRACT_RESULT_V1_R2",

        "status":
            status,

        "timestamp_utc":
            stamp,

        "classification":
            (
                "validated mathematical slew-bounded "
                "authenticated-information contract; "
                "candidate slew value remains provisional "
                "until physically sourced"
            ),

        "candidate_parameter_final":
            False,

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

        "sensitivity":
            rows,

        "artifacts": {
            "summary_csv":
                str(summary_csv),

            "error_bounds_figure":
                str(error_fig),

            "comparison_figure":
                str(ratio_fig),
        },
    }

    result_path = (
        RESULTS_DIR
        /
        f"P3A2_R2_SLEW_CONTRACT_{stamp}.json"
    )

    latest_path = (
        RESULTS_DIR
        /
        "P3A2_R2_LATEST.json"
    )

    text = json.dumps(
        result,
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
        f"P3A2_R2_MANIFEST_{stamp}.sha256"
    )

    files = [
        CFG_PATH,
        P3A_CFG_PATH,
        Path(__file__),
        result_path,
        summary_csv,
        error_fig,
        ratio_fig,
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
        "=== SCV P3-A2-R2 SLEW CONTRACT AUDIT ==="
    )

    for key, value in checks.items():
        print(
            f"{key}="
            f"{'PASS' if value else 'FAIL'}"
        )

    for key, value in metrics.items():
        if isinstance(
            value,
            (int, float),
        ):
            print(
                f"{key.upper()}="
                f"{value:.12g}"
            )

    print(
        "=== CANDIDATE SLEW REDUCTION ==="
    )

    for row in candidate_rows:
        print(
            "AGE="
            f"{row['age_s']:.3f} "
            "EV_MEAN_RATIO="
            f"{row['ev_ratio_mean']:.6f} "
            "EA_MEAN_RATIO="
            f"{row['ea_ratio_mean']:.6f} "
            "EV_MAX_RATIO="
            f"{row['ev_ratio_max']:.6f} "
            "EA_MAX_RATIO="
            f"{row['ea_ratio_max']:.6f}"
        )

    print(
        f"P3A2_R2_SLEW_CONTRACT={status}"
    )

    print(
        "CANDIDATE_PARAMETER_FINAL=NO"
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
