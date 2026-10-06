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
from scipy.stats import qmc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

CFG_PATH = (
    ROOT
    / "01_config"
    / "p2a_recoverability_v1.json"
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

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def stopping_distance(
    v0,
    a0,
    u_cmd: float,
    tau: float,
    disturbance: float,
):
    """
    Exact stopping distance for

        da/dt = -(a-u)/tau + w
        dv/dt = a

    under constant u and constant worst-case w.

    The effective equilibrium acceleration is

        u_eff = u + tau*w.

    The returned motion terminates at the first time v(t)=0.
    """

    v0 = np.asarray(v0, dtype=float)
    a0 = np.asarray(a0, dtype=float)

    u_eff = (
        float(u_cmd)
        +
        float(tau)
        * float(disturbance)
    )

    if u_eff >= 0.0:
        raise ValueError(
            "effective braking acceleration must be negative"
        )

    v0 = np.maximum(v0, 0.0)

    def velocity(t):
        return (
            v0
            +
            u_eff * t
            +
            tau
            * (a0 - u_eff)
            * (
                1.0
                -
                np.exp(-t / tau)
            )
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
        (-u_eff)
        +
        10.0 * tau
    )

    for _ in range(16):

        still_positive = (
            velocity(hi) > 0.0
        )

        if not np.any(still_positive):
            break

        hi = np.where(
            still_positive,
            2.0 * hi,
            hi,
        )

    lo = np.zeros_like(hi)

    for _ in range(80):

        mid = 0.5 * (lo + hi)

        positive = (
            velocity(mid) > 0.0
        )

        lo = np.where(
            positive,
            mid,
            lo,
        )

        hi = np.where(
            positive,
            hi,
            mid,
        )

    t_stop = hi

    distance = (
        v0 * t_stop
        +
        0.5
        * u_eff
        * t_stop**2
        +
        tau
        * (a0 - u_eff)
        * (
            t_stop
            -
            tau
            * (
                1.0
                -
                np.exp(
                    -t_stop / tau
                )
            )
        )
    )

    distance = np.maximum(
        distance,
        0.0,
    )

    return distance, t_stop


def stopping_distance_ivp(
    v0: float,
    a0: float,
    u_cmd: float,
    tau: float,
    disturbance: float,
) -> tuple[float, float]:

    if v0 <= 0.0:
        return 0.0, 0.0

    def rhs(_t, y):

        position, velocity, acceleration = y

        da = (
            -(acceleration - u_cmd)
            / tau
            +
            disturbance
        )

        return [
            velocity,
            acceleration,
            da,
        ]

    def stop_event(_t, y):
        return y[1]

    stop_event.terminal = True
    stop_event.direction = -1.0

    sol = solve_ivp(
        rhs,
        (0.0, 30.0),
        [
            0.0,
            v0,
            a0,
        ],
        method="DOP853",
        events=stop_event,
        rtol=1e-12,
        atol=1e-14,
        max_step=0.01,
    )

    if not sol.success:
        raise RuntimeError(
            sol.message
        )

    if len(sol.t_events[0]) == 0:
        raise RuntimeError(
            "stopping event not reached"
        )

    t_stop = float(
        sol.t_events[0][0]
    )

    d_stop = float(
        sol.y_events[0][0][0]
    )

    return d_stop, t_stop


def validate_stopping_formula(
    rng,
    cfg,
):

    trials = int(
        cfg["validation"][
            "stopping_ivp_trials"
        ]
    )

    tau = float(
        cfg["follower"]["tau"]
    )

    u = float(
        cfg["follower"][
            "emergency_command"
        ]
    )

    w = float(
        cfg["follower"][
            "disturbance_worst"
        ]
    )

    vmax = float(
        cfg["state_domain"][
            "v_f_max"
        ]
    )

    amin = float(
        cfg["state_domain"][
            "a_f_min"
        ]
    )

    amax = float(
        cfg["state_domain"][
            "a_f_max"
        ]
    )

    d_errors = []
    t_errors = []

    for _ in range(trials):

        v0 = float(
            rng.uniform(
                0.1,
                vmax,
            )
        )

        a0 = float(
            rng.uniform(
                amin,
                amax,
            )
        )

        d_exact, t_exact = (
            stopping_distance(
                v0,
                a0,
                u,
                tau,
                w,
            )
        )

        d_ivp, t_ivp = (
            stopping_distance_ivp(
                v0,
                a0,
                u,
                tau,
                w,
            )
        )

        d_errors.append(
            abs(
                float(d_exact)
                -
                d_ivp
            )
        )

        t_errors.append(
            abs(
                float(t_exact)
                -
                t_ivp
            )
        )

    return (
        np.asarray(
            d_errors,
            dtype=float,
        ),
        np.asarray(
            t_errors,
            dtype=float,
        ),
    )


def follower_stop(
    v_f,
    a_f,
    cfg,
    tau_override=None,
    brake_override=None,
):

    tau = (
        float(tau_override)
        if tau_override is not None
        else
        float(
            cfg["follower"]["tau"]
        )
    )

    u = (
        float(brake_override)
        if brake_override is not None
        else
        float(
            cfg["follower"][
                "emergency_command"
            ]
        )
    )

    w = float(
        cfg["follower"][
            "disturbance_worst"
        ]
    )

    return stopping_distance(
        v_f,
        a_f,
        u,
        tau,
        w,
    )[0]


def predecessor_shortest_stop(
    v_p,
    cfg,
):

    return stopping_distance(
        v_p,
        float(
            cfg["predecessor"][
                "initial_acceleration_shortest_stop"
            ]
        ),
        float(
            cfg["predecessor"][
                "emergency_command"
            ]
        ),
        float(
            cfg["predecessor"]["tau"]
        ),
        float(
            cfg["predecessor"][
                "disturbance_shortest_stop"
            ]
        ),
    )[0]


def fallback_required_gap(
    v_f,
    a_f,
    v_p,
    cfg,
    tau_override=None,
    brake_override=None,
):

    d_min = float(
        cfg["state_domain"]["d_min"]
    )

    d_f = follower_stop(
        v_f,
        a_f,
        cfg,
        tau_override=tau_override,
        brake_override=brake_override,
    )

    d_p = predecessor_shortest_stop(
        v_p,
        cfg,
    )

    return (
        d_min
        +
        np.maximum(
            d_f - d_p,
            0.0,
        )
    )


def switching_required_gap(
    v_f,
    v_p,
    cfg,
    switch_steps=None,
    tau_override=None,
    brake_override=None,
):

    if switch_steps is None:

        switch_steps = int(
            cfg["switching"]["N_sw"]
        )

    Ts = float(
        cfg["sampling"]["Ts"]
    )

    T_sw = (
        float(switch_steps)
        * Ts
    )

    a_sw = float(
        cfg["switching"][
            "worst_acceleration_during_switch"
        ]
    )

    s_delay = (
        v_f * T_sw
        +
        0.5
        * a_sw
        * T_sw**2
    )

    v_after = (
        v_f
        +
        a_sw
        * T_sw
    )

    d_after = follower_stop(
        v_after,
        a_sw,
        cfg,
        tau_override=tau_override,
        brake_override=brake_override,
    )

    d_p = predecessor_shortest_stop(
        v_p,
        cfg,
    )

    d_min = float(
        cfg["state_domain"]["d_min"]
    )

    return (
        d_min
        +
        np.maximum(
            s_delay
            +
            d_after
            -
            d_p,
            0.0,
        )
    )


def sobol_states(
    power: int,
    seed: int,
    cfg,
):

    sampler = qmc.Sobol(
        d=4,
        scramble=True,
        seed=seed,
    )

    unit = sampler.random_base2(
        power
    )

    sd = cfg["state_domain"]

    lower = np.array(
        [
            sd["d_min"],
            sd["delta_v_min"],
            sd["v_f_min"],
            sd["a_f_min"],
        ],
        dtype=float,
    )

    upper = np.array(
        [
            sd["d_max"],
            sd["delta_v_max"],
            sd["v_f_max"],
            sd["a_f_max"],
        ],
        dtype=float,
    )

    states = (
        lower
        +
        (upper - lower)
        * unit
    )

    return (
        states,
        lower,
        upper,
    )


def evaluate_states(
    states,
    cfg,
    switch_steps=None,
    tau_override=None,
    brake_override=None,
):

    d = states[:, 0]
    delta_v = states[:, 1]
    v_f = states[:, 2]
    a_f = states[:, 3]

    v_p = (
        v_f
        +
        delta_v
    )

    vp_min = float(
        cfg["state_domain"]["v_p_min"]
    )

    vp_max = float(
        cfg["state_domain"]["v_p_max"]
    )

    physical = (
        (v_p >= vp_min)
        &
        (v_p <= vp_max)
    )

    v_p_eval = np.clip(
        v_p,
        vp_min,
        vp_max,
    )

    fallback_gap = (
        fallback_required_gap(
            v_f,
            a_f,
            v_p_eval,
            cfg,
            tau_override=tau_override,
            brake_override=brake_override,
        )
    )

    switch_gap = (
        switching_required_gap(
            v_f,
            v_p_eval,
            cfg,
            switch_steps=switch_steps,
            tau_override=tau_override,
            brake_override=brake_override,
        )
    )

    fallback = (
        physical
        &
        (d >= fallback_gap)
    )

    guard = (
        physical
        &
        (d >= switch_gap)
    )

    return {
        "physical":
            physical,

        "fallback":
            fallback,

        "guard":
            guard,

        "fallback_gap":
            fallback_gap,

        "switch_gap":
            switch_gap,

        "v_p":
            v_p,
    }


def volume_metrics(
    states,
    lower,
    upper,
    cfg,
    **kwargs,
):

    e = evaluate_states(
        states,
        cfg,
        **kwargs,
    )

    box_volume = float(
        np.prod(
            upper - lower
        )
    )

    physical_fraction = float(
        np.mean(
            e["physical"]
        )
    )

    fallback_fraction_box = float(
        np.mean(
            e["fallback"]
        )
    )

    guard_fraction_box = float(
        np.mean(
            e["guard"]
        )
    )

    physical_volume = (
        box_volume
        * physical_fraction
    )

    fallback_volume = (
        box_volume
        * fallback_fraction_box
    )

    guard_volume = (
        box_volume
        * guard_fraction_box
    )

    n_phys = int(
        np.count_nonzero(
            e["physical"]
        )
    )

    if n_phys == 0:
        raise RuntimeError(
            "empty physical QMC domain"
        )

    fallback_conditional = float(
        np.mean(
            e["fallback"][
                e["physical"]
            ]
        )
    )

    guard_conditional = float(
        np.mean(
            e["guard"][
                e["physical"]
            ]
        )
    )

    return {
        "box_volume":
            box_volume,

        "physical_volume":
            physical_volume,

        "fallback_volume":
            fallback_volume,

        "guard_volume":
            guard_volume,

        "fallback_fraction_physical":
            fallback_conditional,

        "guard_fraction_physical":
            guard_conditional,

        "physical_fraction_box":
            physical_fraction,

        "guard_subset_violations":
            int(
                np.count_nonzero(
                    e["guard"]
                    &
                    ~e["fallback"]
                )
            ),
    }


def monotonicity_audit(cfg):

    n_v = int(
        cfg["validation"][
            "monotonicity_grid_v"
        ]
    )

    n_a = int(
        cfg["validation"][
            "monotonicity_grid_a"
        ]
    )

    v = np.linspace(
        0.0,
        float(
            cfg["state_domain"][
                "v_f_max"
            ]
        ),
        n_v,
    )

    a = np.linspace(
        float(
            cfg["state_domain"][
                "a_f_min"
            ]
        ),
        float(
            cfg["state_domain"][
                "a_f_max"
            ]
        ),
        n_a,
    )

    V, A = np.meshgrid(
        v,
        a,
        indexing="ij",
    )

    D = follower_stop(
        V,
        A,
        cfg,
    )

    min_dv = float(
        np.min(
            np.diff(
                D,
                axis=0,
            )
        )
    )

    min_da = float(
        np.min(
            np.diff(
                D,
                axis=1,
            )
        )
    )

    Dp = predecessor_shortest_stop(
        v,
        cfg,
    )

    min_dp_v = float(
        np.min(
            np.diff(
                Dp
            )
        )
    )

    return {
        "follower_min_increment_velocity":
            min_dv,

        "follower_min_increment_acceleration":
            min_da,

        "predecessor_min_increment_velocity":
            min_dp_v,
    }


def sensitivity_analysis(
    states,
    lower,
    upper,
    cfg,
):

    max_steps = int(
        cfg["switching"][
            "sensitivity_max_steps"
        ]
    )

    switch_rows = []

    for n_sw in range(
        max_steps + 1
    ):

        m = volume_metrics(
            states,
            lower,
            upper,
            cfg,
            switch_steps=n_sw,
        )

        switch_rows.append(
            {
                "switch_steps":
                    n_sw,

                "guard_fraction_physical":
                    m[
                        "guard_fraction_physical"
                    ],
            }
        )

    tau_rows = []

    for tau in cfg["sensitivity"][
        "tau_values"
    ]:

        m = volume_metrics(
            states,
            lower,
            upper,
            cfg,
            tau_override=float(tau),
        )

        tau_rows.append(
            {
                "tau":
                    float(tau),

                "fallback_fraction_physical":
                    m[
                        "fallback_fraction_physical"
                    ],

                "guard_fraction_physical":
                    m[
                        "guard_fraction_physical"
                    ],
            }
        )

    brake_rows = []

    for brake in cfg["sensitivity"][
        "brake_commands"
    ]:

        m = volume_metrics(
            states,
            lower,
            upper,
            cfg,
            brake_override=float(brake),
        )

        brake_rows.append(
            {
                "brake_command":
                    float(brake),

                "fallback_fraction_physical":
                    m[
                        "fallback_fraction_physical"
                    ],

                "guard_fraction_physical":
                    m[
                        "guard_fraction_physical"
                    ],
            }
        )

    return (
        switch_rows,
        tau_rows,
        brake_rows,
    )


def replicate_dispersion(
    cfg,
):

    fractions_f = []
    fractions_g = []

    power = int(
        cfg["validation"][
            "qmc_power_replicate"
        ]
    )

    reps = int(
        cfg["validation"][
            "qmc_replicates"
        ]
    )

    base_seed = int(
        cfg["random_seed"]
    )

    for j in range(reps):

        states, lower, upper = (
            sobol_states(
                power,
                base_seed + 1000 + j,
                cfg,
            )
        )

        m = volume_metrics(
            states,
            lower,
            upper,
            cfg,
        )

        fractions_f.append(
            m[
                "fallback_fraction_physical"
            ]
        )

        fractions_g.append(
            m[
                "guard_fraction_physical"
            ]
        )

    return {
        "fallback_mean":
            float(
                np.mean(
                    fractions_f
                )
            ),

        "fallback_std":
            float(
                np.std(
                    fractions_f,
                    ddof=1,
                )
                if reps > 1
                else 0.0
            ),

        "guard_mean":
            float(
                np.mean(
                    fractions_g
                )
            ),

        "guard_std":
            float(
                np.std(
                    fractions_g,
                    ddof=1,
                )
                if reps > 1
                else 0.0
            ),
    }


def make_slice_figure(
    path: Path,
    cfg,
):

    fig_cfg = cfg["figures"]

    d = np.linspace(
        float(
            cfg["state_domain"][
                "d_min"
            ]
        ),
        min(
            160.0,
            float(
                cfg["state_domain"][
                    "d_max"
                ]
            ),
        ),
        int(
            fig_cfg[
                "slice_d_points"
            ]
        ),
    )

    dv = np.linspace(
        float(
            cfg["state_domain"][
                "delta_v_min"
            ]
        ),
        float(
            cfg["state_domain"][
                "delta_v_max"
            ]
        ),
        int(
            fig_cfg[
                "slice_delta_v_points"
            ]
        ),
    )

    D, DV = np.meshgrid(
        d,
        dv,
        indexing="xy",
    )

    vf = np.full_like(
        D,
        float(
            fig_cfg["slice_v_f"]
        ),
    )

    af = np.full_like(
        D,
        float(
            fig_cfg["slice_a_f"]
        ),
    )

    vp = vf + DV

    valid = (
        (vp >=
         float(
             cfg["state_domain"][
                 "v_p_min"
             ]
         ))
        &
        (vp <=
         float(
             cfg["state_domain"][
                 "v_p_max"
             ]
         ))
    )

    vp_eval = np.clip(
        vp,
        float(
            cfg["state_domain"][
                "v_p_min"
            ]
        ),
        float(
            cfg["state_domain"][
                "v_p_max"
            ]
        ),
    )

    req_f = fallback_required_gap(
        vf,
        af,
        vp_eval,
        cfg,
    )

    req_g = switching_required_gap(
        vf,
        vp_eval,
        cfg,
    )

    fallback = (
        valid
        &
        (D >= req_f)
    )

    guard = (
        valid
        &
        (D >= req_g)
    )

    Z = np.zeros_like(
        D,
        dtype=int,
    )

    Z[fallback] = 1
    Z[guard] = 2

    fig, ax = plt.subplots(
        figsize=(7.2, 5.0)
    )

    ax.contourf(
        D,
        DV,
        Z,
        levels=[
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
        "Fallback recoverability and switching guard"
    )

    ax.grid(
        True,
        alpha=0.2,
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=240,
    )

    plt.close(fig)


def make_sensitivity_figure(
    path,
    switch_rows,
    tau_rows,
    brake_rows,
):

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13.0, 4.0),
    )

    axes[0].plot(
        [
            r["switch_steps"]
            for r
            in switch_rows
        ],
        [
            r[
                "guard_fraction_physical"
            ]
            for r
            in switch_rows
        ],
        marker="o",
    )

    axes[0].set_xlabel(
        "Switching steps"
    )

    axes[0].set_ylabel(
        "Guard fraction"
    )

    axes[0].grid(
        True,
        alpha=0.25,
    )

    axes[1].plot(
        [
            r["tau"]
            for r
            in tau_rows
        ],
        [
            r[
                "fallback_fraction_physical"
            ]
            for r
            in tau_rows
        ],
        marker="o",
        label="fallback",
    )

    axes[1].plot(
        [
            r["tau"]
            for r
            in tau_rows
        ],
        [
            r[
                "guard_fraction_physical"
            ]
            for r
            in tau_rows
        ],
        marker="s",
        label="guard",
    )

    axes[1].set_xlabel(
        "Follower actuator time constant (s)"
    )

    axes[1].grid(
        True,
        alpha=0.25,
    )

    axes[1].legend()

    axes[2].plot(
        [
            abs(
                r["brake_command"]
            )
            for r
            in brake_rows
        ],
        [
            r[
                "fallback_fraction_physical"
            ]
            for r
            in brake_rows
        ],
        marker="o",
        label="fallback",
    )

    axes[2].plot(
        [
            abs(
                r["brake_command"]
            )
            for r
            in brake_rows
        ],
        [
            r[
                "guard_fraction_physical"
            ]
            for r
            in brake_rows
        ],
        marker="s",
        label="guard",
    )

    axes[2].set_xlabel(
        "Emergency braking magnitude"
    )

    axes[2].grid(
        True,
        alpha=0.25,
    )

    axes[2].legend()

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=240,
    )

    plt.close(fig)


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

    d_err, t_err = (
        validate_stopping_formula(
            rng,
            cfg,
        )
    )

    mono = monotonicity_audit(
        cfg
    )

    states, lower, upper = (
        sobol_states(
            int(
                cfg["validation"][
                    "qmc_power_main"
                ]
            ),
            int(
                cfg["random_seed"]
            ),
            cfg,
        )
    )

    main_metrics = volume_metrics(
        states,
        lower,
        upper,
        cfg,
    )

    (
        switch_rows,
        tau_rows,
        brake_rows,
    ) = sensitivity_analysis(
        states,
        lower,
        upper,
        cfg,
    )

    dispersion = replicate_dispersion(
        cfg
    )

    switch_values = np.asarray(
        [
            row[
                "guard_fraction_physical"
            ]
            for row
            in switch_rows
        ],
        dtype=float,
    )

    tau_values = np.asarray(
        [
            row[
                "fallback_fraction_physical"
            ]
            for row
            in tau_rows
        ],
        dtype=float,
    )

    brake_values = np.asarray(
        [
            row[
                "fallback_fraction_physical"
            ]
            for row
            in brake_rows
        ],
        dtype=float,
    )

    tol = 1e-12

    checks = {

        "STOPPING_FORMULA_VS_IVP":
            (
                float(
                    np.max(d_err)
                )
                <=
                float(
                    cfg["validation"][
                        "stopping_ivp_tolerance"
                    ]
                )
                and
                float(
                    np.max(t_err)
                )
                <=
                float(
                    cfg["validation"][
                        "stopping_ivp_tolerance"
                    ]
                )
            ),

        "FOLLOWER_STOP_MONOTONE_V":
            (
                mono[
                    "follower_min_increment_velocity"
                ]
                >=
                -tol
            ),

        "FOLLOWER_STOP_MONOTONE_A":
            (
                mono[
                    "follower_min_increment_acceleration"
                ]
                >=
                -tol
            ),

        "PREDECESSOR_STOP_MONOTONE_V":
            (
                mono[
                    "predecessor_min_increment_velocity"
                ]
                >=
                -tol
            ),

        "FALLBACK_SET_NONEMPTY":
            (
                main_metrics[
                    "fallback_fraction_physical"
                ]
                >
                0.0
            ),

        "SWITCH_GUARD_NONEMPTY":
            (
                main_metrics[
                    "guard_fraction_physical"
                ]
                >
                0.0
            ),

        "GUARD_SUBSET_FALLBACK":
            (
                main_metrics[
                    "guard_subset_violations"
                ]
                ==
                0
            ),

        "SWITCH_DELAY_MONOTONE":
            bool(
                np.all(
                    np.diff(
                        switch_values
                    )
                    <=
                    1e-12
                )
            ),

        "ACTUATOR_LAG_MONOTONE":
            bool(
                np.all(
                    np.diff(
                        tau_values
                    )
                    <=
                    1e-12
                )
            ),

        "BRAKE_STRENGTH_MONOTONE":
            bool(
                np.all(
                    np.diff(
                        brake_values
                    )
                    >=
                    -1e-12
                )
            ),

        "QMC_REPLICATE_STABLE":
            (
                dispersion[
                    "fallback_std"
                ]
                <=
                float(
                    cfg["validation"][
                        "qmc_stability_tolerance"
                    ]
                )
                and
                dispersion[
                    "guard_std"
                ]
                <=
                float(
                    cfg["validation"][
                        "qmc_stability_tolerance"
                    ]
                )
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

    switch_csv = (
        RESULTS_DIR
        /
        f"P2A_SWITCH_SENSITIVITY_{stamp}.csv"
    )

    tau_csv = (
        RESULTS_DIR
        /
        f"P2A_TAU_SENSITIVITY_{stamp}.csv"
    )

    brake_csv = (
        RESULTS_DIR
        /
        f"P2A_BRAKE_SENSITIVITY_{stamp}.csv"
    )

    slice_fig = (
        FIGURES_DIR
        /
        f"P2A_FALLBACK_GUARD_SLICE_{stamp}.png"
    )

    sensitivity_fig = (
        FIGURES_DIR
        /
        f"P2A_SENSITIVITY_{stamp}.png"
    )

    write_csv(
        switch_csv,
        switch_rows,
    )

    write_csv(
        tau_csv,
        tau_rows,
    )

    write_csv(
        brake_csv,
        brake_rows,
    )

    make_slice_figure(
        slice_fig,
        cfg,
    )

    make_sensitivity_figure(
        sensitivity_fig,
        switch_rows,
        tau_rows,
        brake_rows,
    )

    metrics = {

        "stopping_distance_ivp_max_error":
            float(
                np.max(d_err)
            ),

        "stopping_time_ivp_max_error":
            float(
                np.max(t_err)
            ),

        **mono,

        **main_metrics,

        **{
            "qmc_" + key:
                value
            for key, value
            in dispersion.items()
        },
    }

    result = {

        "schema":
            "SCV_P2A_RECOVERABILITY_RESULT_V1",

        "status":
            status,

        "timestamp_utc":
            stamp,

        "interpretation":
            (
                "certified-physics recoverability "
                "baseline; not yet the maximal "
                "fallback viability kernel"
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

        "switch_sensitivity":
            switch_rows,

        "tau_sensitivity":
            tau_rows,

        "brake_sensitivity":
            brake_rows,

        "artifacts": {
            "switch_csv":
                str(switch_csv),

            "tau_csv":
                str(tau_csv),

            "brake_csv":
                str(brake_csv),

            "slice_figure":
                str(slice_fig),

            "sensitivity_figure":
                str(sensitivity_fig),
        },
    }

    result_path = (
        RESULTS_DIR
        /
        f"P2A_RECOVERABILITY_{stamp}.json"
    )

    latest_path = (
        RESULTS_DIR
        /
        "P2A_LATEST.json"
    )

    result_text = json.dumps(
        result,
        indent=2,
        sort_keys=True,
    )

    atomic_write_text(
        result_path,
        result_text,
    )

    atomic_write_text(
        latest_path,
        result_text,
    )

    manifest_path = (
        RESULTS_DIR
        /
        f"P2A_MANIFEST_{stamp}.sha256"
    )

    manifest_files = [
        CFG_PATH,
        Path(__file__),
        result_path,
        switch_csv,
        tau_csv,
        brake_csv,
        slice_fig,
        sensitivity_fig,
    ]

    manifest = "\n".join(
        (
            f"{sha256_file(path)}"
            f"  {path}"
        )
        for path
        in manifest_files
    ) + "\n"

    atomic_write_text(
        manifest_path,
        manifest,
    )

    print(
        "=== SCV P2-A RECOVERABILITY AUDIT ==="
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
        f"P2A_RECOVERABILITY={status}"
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
