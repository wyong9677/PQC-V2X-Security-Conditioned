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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

CFG_PATH = (
    ROOT
    / "01_config"
    / "p3a_information_contract_v1.json"
)

RESULTS_DIR = ROOT / "04_results"
FIGURES_DIR = ROOT / "05_figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
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


def directional_rates(
    bar_u,
    u_min: float,
    u_max: float,
    tau: float,
    w_abs: float,
):
    """
    Error dynamics:

        de_a/dt
        = -e_a/tau
          + (u-bar_u)/tau
          + w

    delta_minus controls the braking-critical
    negative direction.

    delta_plus controls the positive direction.
    """

    bar_u = np.asarray(bar_u, dtype=float)

    delta_minus = (
        np.maximum(
            bar_u - u_min,
            0.0,
        )
        / tau
        +
        w_abs
    )

    delta_plus = (
        np.maximum(
            u_max - bar_u,
            0.0,
        )
        / tau
        +
        w_abs
    )

    return delta_minus, delta_plus


def kernels(
    age,
    tau: float,
):
    age = np.asarray(age, dtype=float)

    ea_kernel = (
        tau
        *
        (
            1.0
            -
            np.exp(
                -age / tau
            )
        )
    )

    ev_kernel = (
        tau * age
        -
        tau**2
        *
        (
            1.0
            -
            np.exp(
                -age / tau
            )
        )
    )

    return ev_kernel, ea_kernel


def asymmetric_bounds(
    age,
    bar_u,
    cfg,
):
    p = cfg["predecessor"]

    tau = float(p["tau"])
    u_min = float(p["command_min"])
    u_max = float(p["command_max"])
    w_abs = float(p["disturbance_abs"])

    dm, dp = directional_rates(
        bar_u,
        u_min,
        u_max,
        tau,
        w_abs,
    )

    kv, ka = kernels(
        age,
        tau,
    )

    return {
        "ev_lower":
            -dm * kv,

        "ev_upper":
            dp * kv,

        "ea_lower":
            -dm * ka,

        "ea_upper":
            dp * ka,

        "delta_minus":
            dm,

        "delta_plus":
            dp,
    }


def exact_error_piecewise(
    age: float,
    tau: float,
    bar_u: float,
    commands: np.ndarray,
    disturbances: np.ndarray,
):
    """
    Exact segment-by-segment update for

        e_v_dot = e_a
        e_a_dot = -e_a/tau + delta(t)
    """

    if age <= 0.0:
        return 0.0, 0.0

    n = len(commands)

    if n == 0:
        raise ValueError(
            "empty command sequence"
        )

    h = age / n

    decay = math.exp(
        -h / tau
    )

    ka = (
        tau
        *
        (1.0 - decay)
    )

    kv = (
        tau * h
        -
        tau**2
        *
        (1.0 - decay)
    )

    ev = 0.0
    ea = 0.0

    for u, w in zip(
        commands,
        disturbances,
    ):
        delta = (
            (float(u) - bar_u)
            / tau
            +
            float(w)
        )

        ev = (
            ev
            +
            ea * ka
            +
            delta * kv
        )

        ea = (
            ea * decay
            +
            delta * ka
        )

    return ev, ea


def random_envelope_audit(
    cfg,
    rng,
):
    p = cfg["predecessor"]
    v = cfg["validation"]

    tau = float(p["tau"])
    u_min = float(p["command_min"])
    u_max = float(p["command_max"])
    w_abs = float(p["disturbance_abs"])

    age_max = float(
        cfg[
            "authenticated_information"
        ][
            "age_max_seconds"
        ]
    )

    trials = int(
        v["random_piecewise_trials"]
    )

    segments = int(
        v["segments_per_trial"]
    )

    max_ev_lower_violation = 0.0
    max_ev_upper_violation = 0.0

    max_ea_lower_violation = 0.0
    max_ea_upper_violation = 0.0

    max_ev_utilization = 0.0
    max_ea_utilization = 0.0

    for _ in range(trials):

        bar_u = float(
            rng.uniform(
                u_min,
                u_max,
            )
        )

        age = float(
            rng.uniform(
                1e-8,
                age_max,
            )
        )

        commands = rng.uniform(
            u_min,
            u_max,
            size=segments,
        )

        disturbances = rng.uniform(
            -w_abs,
            w_abs,
            size=segments,
        )

        ev, ea = exact_error_piecewise(
            age,
            tau,
            bar_u,
            commands,
            disturbances,
        )

        b = asymmetric_bounds(
            age,
            bar_u,
            cfg,
        )

        ev_l = float(b["ev_lower"])
        ev_u = float(b["ev_upper"])

        ea_l = float(b["ea_lower"])
        ea_u = float(b["ea_upper"])

        max_ev_lower_violation = max(
            max_ev_lower_violation,
            ev_l - ev,
        )

        max_ev_upper_violation = max(
            max_ev_upper_violation,
            ev - ev_u,
        )

        max_ea_lower_violation = max(
            max_ea_lower_violation,
            ea_l - ea,
        )

        max_ea_upper_violation = max(
            max_ea_upper_violation,
            ea - ea_u,
        )

        if ev < 0.0 and ev_l < 0.0:
            ev_ratio = (
                abs(ev)
                /
                abs(ev_l)
            )
        elif ev > 0.0 and ev_u > 0.0:
            ev_ratio = (
                ev / ev_u
            )
        else:
            ev_ratio = 0.0

        if ea < 0.0 and ea_l < 0.0:
            ea_ratio = (
                abs(ea)
                /
                abs(ea_l)
            )
        elif ea > 0.0 and ea_u > 0.0:
            ea_ratio = (
                ea / ea_u
            )
        else:
            ea_ratio = 0.0

        max_ev_utilization = max(
            max_ev_utilization,
            ev_ratio,
        )

        max_ea_utilization = max(
            max_ea_utilization,
            ea_ratio,
        )

    return {
        "max_ev_lower_violation":
            max_ev_lower_violation,

        "max_ev_upper_violation":
            max_ev_upper_violation,

        "max_ea_lower_violation":
            max_ea_lower_violation,

        "max_ea_upper_violation":
            max_ea_upper_violation,

        "max_ev_random_utilization":
            max_ev_utilization,

        "max_ea_random_utilization":
            max_ea_utilization,
    }


def worst_case_tightness(
    cfg,
):
    p = cfg["predecessor"]

    tau = float(p["tau"])
    u_min = float(p["command_min"])
    u_max = float(p["command_max"])
    w_abs = float(p["disturbance_abs"])

    ages = np.linspace(
        0.01,
        float(
            cfg[
                "authenticated_information"
            ][
                "age_max_seconds"
            ]
        ),
        64,
    )

    bar_values = np.linspace(
        u_min,
        u_max,
        33,
    )

    max_error = 0.0

    for bar_u in bar_values:

        for age in ages:

            b = asymmetric_bounds(
                float(age),
                float(bar_u),
                cfg,
            )

            commands = np.full(
                64,
                u_min,
            )

            disturbances = np.full(
                64,
                -w_abs,
            )

            ev, ea = exact_error_piecewise(
                float(age),
                tau,
                float(bar_u),
                commands,
                disturbances,
            )

            max_error = max(
                max_error,
                abs(
                    ev
                    -
                    float(
                        b["ev_lower"]
                    )
                ),
                abs(
                    ea
                    -
                    float(
                        b["ea_lower"]
                    )
                ),
            )

            commands[:] = u_max
            disturbances[:] = w_abs

            ev, ea = exact_error_piecewise(
                float(age),
                tau,
                float(bar_u),
                commands,
                disturbances,
            )

            max_error = max(
                max_error,
                abs(
                    ev
                    -
                    float(
                        b["ev_upper"]
                    )
                ),
                abs(
                    ea
                    -
                    float(
                        b["ea_upper"]
                    )
                ),
            )

    return max_error


def legacy_coverage_audit(
    cfg,
    bar_grid,
):
    legacy = float(
        cfg[
            "legacy_provisional_model"
        ][
            "fixed_delta_bar"
        ]
    )

    p = cfg["predecessor"]

    dm, dp = directional_rates(
        bar_grid,
        float(p["command_min"]),
        float(p["command_max"]),
        float(p["tau"]),
        float(p["disturbance_abs"]),
    )

    required = np.maximum(
        dm,
        dp,
    )

    covered = (
        legacy
        >=
        required
    )

    return {
        "legacy_delta":
            legacy,

        "required_delta_min":
            float(
                np.min(required)
            ),

        "required_delta_max":
            float(
                np.max(required)
            ),

        "coverage_fraction":
            float(
                np.mean(covered)
            ),

        "fully_valid":
            bool(
                np.all(covered)
            ),
    }


def monotonicity_audit(
    cfg,
    ages,
    bar_grid,
):
    min_ev_width_increment = math.inf
    min_ea_width_increment = math.inf

    for bar_u in bar_grid:

        b = asymmetric_bounds(
            ages,
            float(bar_u),
            cfg,
        )

        ev_width = (
            b["ev_upper"]
            -
            b["ev_lower"]
        )

        ea_width = (
            b["ea_upper"]
            -
            b["ea_lower"]
        )

        min_ev_width_increment = min(
            min_ev_width_increment,
            float(
                np.min(
                    np.diff(
                        ev_width
                    )
                )
            ),
        )

        min_ea_width_increment = min(
            min_ea_width_increment,
            float(
                np.min(
                    np.diff(
                        ea_width
                    )
                )
            ),
        )

    return {
        "min_ev_width_increment":
            min_ev_width_increment,

        "min_ea_width_increment":
            min_ea_width_increment,
    }


def write_delta_csv(
    path,
    bar_grid,
    cfg,
):
    p = cfg["predecessor"]

    dm, dp = directional_rates(
        bar_grid,
        float(p["command_min"]),
        float(p["command_max"]),
        float(p["tau"]),
        float(p["disturbance_abs"]),
    )

    legacy = float(
        cfg[
            "legacy_provisional_model"
        ][
            "fixed_delta_bar"
        ]
    )

    with path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "bar_u",
                "delta_minus",
                "delta_plus",
                "required_symmetric_delta",
                "legacy_delta",
                "legacy_valid"
            ]
        )

        for u, dmn, dpl in zip(
            bar_grid,
            dm,
            dp,
        ):

            req = max(
                float(dmn),
                float(dpl),
            )

            writer.writerow(
                [
                    f"{u:.12g}",
                    f"{dmn:.12g}",
                    f"{dpl:.12g}",
                    f"{req:.12g}",
                    f"{legacy:.12g}",
                    int(
                        legacy >= req
                    ),
                ]
            )


def make_delta_figure(
    path,
    bar_grid,
    cfg,
):
    p = cfg["predecessor"]

    dm, dp = directional_rates(
        bar_grid,
        float(p["command_min"]),
        float(p["command_max"]),
        float(p["tau"]),
        float(p["disturbance_abs"]),
    )

    legacy = float(
        cfg[
            "legacy_provisional_model"
        ][
            "fixed_delta_bar"
        ]
    )

    fig, ax = plt.subplots(
        figsize=(7.3, 4.8)
    )

    ax.plot(
        bar_grid,
        dm,
        label="braking-critical delta_minus",
    )

    ax.plot(
        bar_grid,
        dp,
        label="positive-direction delta_plus",
    )

    ax.axhline(
        legacy,
        linestyle="--",
        label="legacy provisional delta",
    )

    ax.set_xlabel(
        "Authenticated predecessor command bar_u"
    )

    ax.set_ylabel(
        "Required disturbance-rate bound"
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


def make_error_figures(
    acc_path,
    vel_path,
    ages,
    cfg,
):
    selected = (
        cfg[
            "authenticated_information"
        ][
            "selected_bar_u"
        ]
    )

    fig, ax = plt.subplots(
        figsize=(7.3, 4.8)
    )

    for bar_u in selected:

        b = asymmetric_bounds(
            ages,
            float(bar_u),
            cfg,
        )

        ax.plot(
            ages,
            b["ea_lower"],
            label=f"lower, bar_u={bar_u:g}",
        )

        ax.plot(
            ages,
            b["ea_upper"],
            linestyle="--",
            label=f"upper, bar_u={bar_u:g}",
        )

    ax.set_xlabel(
        "Authenticated information age (s)"
    )

    ax.set_ylabel(
        "Acceleration-prediction error bound"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend(
        fontsize=8,
        ncol=2,
    )

    fig.tight_layout()

    fig.savefig(
        acc_path,
        dpi=240,
    )

    plt.close(fig)

    fig, ax = plt.subplots(
        figsize=(7.3, 4.8)
    )

    for bar_u in selected:

        b = asymmetric_bounds(
            ages,
            float(bar_u),
            cfg,
        )

        ax.plot(
            ages,
            b["ev_lower"],
            label=f"lower, bar_u={bar_u:g}",
        )

        ax.plot(
            ages,
            b["ev_upper"],
            linestyle="--",
            label=f"upper, bar_u={bar_u:g}",
        )

    ax.set_xlabel(
        "Authenticated information age (s)"
    )

    ax.set_ylabel(
        "Velocity-prediction error bound"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend(
        fontsize=8,
        ncol=2,
    )

    fig.tight_layout()

    fig.savefig(
        vel_path,
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

    p = cfg["predecessor"]

    ages = np.linspace(
        0.0,
        float(
            cfg[
                "authenticated_information"
            ][
                "age_max_seconds"
            ]
        ),
        int(
            cfg[
                "authenticated_information"
            ][
                "age_grid_points"
            ]
        ),
    )

    bar_grid = np.linspace(
        float(p["command_min"]),
        float(p["command_max"]),
        int(
            cfg[
                "authenticated_information"
            ][
                "bar_u_grid_points"
            ]
        ),
    )

    random_audit = (
        random_envelope_audit(
            cfg,
            rng,
        )
    )

    tightness = (
        worst_case_tightness(
            cfg
        )
    )

    legacy = (
        legacy_coverage_audit(
            cfg,
            bar_grid,
        )
    )

    monotone = (
        monotonicity_audit(
            cfg,
            ages,
            bar_grid,
        )
    )

    tolerance = float(
        cfg["validation"][
            "envelope_tolerance"
        ]
    )

    tight_tol = float(
        cfg["validation"][
            "worst_case_tightness_tolerance"
        ]
    )

    checks = {
        "RANDOM_ENVELOPE_CONTAINMENT":
            (
                random_audit[
                    "max_ev_lower_violation"
                ]
                <= tolerance
                and
                random_audit[
                    "max_ev_upper_violation"
                ]
                <= tolerance
                and
                random_audit[
                    "max_ea_lower_violation"
                ]
                <= tolerance
                and
                random_audit[
                    "max_ea_upper_violation"
                ]
                <= tolerance
            ),

        "ADVERSARIAL_BOUNDS_TIGHT":
            (
                tightness
                <= tight_tol
            ),

        "AGE_WIDTH_MONOTONE":
            (
                monotone[
                    "min_ev_width_increment"
                ]
                >=
                -1e-12
                and
                monotone[
                    "min_ea_width_increment"
                ]
                >=
                -1e-12
            ),

        "LEGACY_FIXED_DELTA_REJECTED_FOR_P3":
            (
                not legacy[
                    "fully_valid"
                ]
            ),

        "FULL_RANGE_COMMAND_CONTRACT_VALID":
            (
                float(
                    p["command_min"]
                )
                <
                float(
                    p["command_max"]
                )
                and
                float(
                    p["tau"]
                )
                >
                0.0
                and
                float(
                    p["disturbance_abs"]
                )
                >=
                0.0
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

    delta_csv = (
        RESULTS_DIR
        /
        f"P3A_DIRECTIONAL_DELTA_{stamp}.csv"
    )

    delta_fig = (
        FIGURES_DIR
        /
        f"P3A_DIRECTIONAL_DELTA_{stamp}.png"
    )

    acc_fig = (
        FIGURES_DIR
        /
        f"P3A_ACCELERATION_ERROR_INTERVAL_{stamp}.png"
    )

    vel_fig = (
        FIGURES_DIR
        /
        f"P3A_VELOCITY_ERROR_INTERVAL_{stamp}.png"
    )

    write_delta_csv(
        delta_csv,
        bar_grid,
        cfg,
    )

    make_delta_figure(
        delta_fig,
        bar_grid,
        cfg,
    )

    make_error_figures(
        acc_fig,
        vel_fig,
        ages,
        cfg,
    )

    dm, dp = directional_rates(
        bar_grid,
        float(p["command_min"]),
        float(p["command_max"]),
        float(p["tau"]),
        float(p["disturbance_abs"]),
    )

    metrics = {
        **random_audit,

        "worst_case_tightness_error":
            float(tightness),

        **{
            "legacy_" + key:
                value
            for key, value
            in legacy.items()
            if isinstance(
                value,
                (int, float, bool),
            )
        },

        **monotone,

        "global_delta_minus_max":
            float(
                np.max(dm)
            ),

        "global_delta_plus_max":
            float(
                np.max(dp)
            ),

        "global_symmetric_delta_required":
            float(
                np.max(
                    np.maximum(
                        dm,
                        dp,
                    )
                )
            ),
    }

    result = {
        "schema":
            "SCV_P3A_INFORMATION_CONTRACT_RESULT_V1",

        "status":
            status,

        "timestamp_utc":
            stamp,

        "classification":
            (
                "state-dependent asymmetric "
                "authenticated-information uncertainty "
                "contract for P3 and later stages"
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
                k: bool(v)
                for k, v
                in checks.items()
            },

        "metrics":
            metrics,

        "contract": {
            "type":
                "asymmetric_state_dependent",

            "command_min":
                float(
                    p["command_min"]
                ),

            "command_max":
                float(
                    p["command_max"]
                ),

            "tau":
                float(
                    p["tau"]
                ),

            "disturbance_abs":
                float(
                    p["disturbance_abs"]
                ),

            "certification_status":
                "VALIDATED_FOR_P3_NUMERICAL_MODEL"
        },

        "artifacts": {
            "directional_delta_csv":
                str(delta_csv),

            "directional_delta_figure":
                str(delta_fig),

            "acceleration_error_figure":
                str(acc_fig),

            "velocity_error_figure":
                str(vel_fig),
        },
    }

    result_path = (
        RESULTS_DIR
        /
        f"P3A_INFORMATION_CONTRACT_{stamp}.json"
    )

    latest_path = (
        RESULTS_DIR
        /
        "P3A_LATEST.json"
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
        f"P3A_MANIFEST_{stamp}.sha256"
    )

    files = [
        CFG_PATH,
        Path(__file__),
        result_path,
        delta_csv,
        delta_fig,
        acc_fig,
        vel_fig,
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
        "=== SCV P3-A INFORMATION CONTRACT AUDIT ==="
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
        f"P3A_INFORMATION_CONTRACT={status}"
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
