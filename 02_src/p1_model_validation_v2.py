from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import sys

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scipy.integrate import solve_ivp
from scipy.linalg import expm

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

CFG_PATH = (
    ROOT
    / "01_config"
    / "p1_validation_v2.json"
)

RESULTS_DIR = ROOT / "04_results"
FIG_DIR = ROOT / "05_figures"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


@dataclass(frozen=True)
class ContinuousModel:
    A: np.ndarray
    B: np.ndarray
    G: np.ndarray
    E: np.ndarray


@dataclass(frozen=True)
class DiscreteModel:
    Ad: np.ndarray
    Bd: np.ndarray
    Gd: np.ndarray
    Ed: np.ndarray


def atomic_write_text(
    path: Path,
    text: str,
) -> None:

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_text(
        text,
        encoding="utf-8",
    )

    tmp.replace(path)


def sha256_file(
    path: Path,
) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:

        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def continuous_link_model(
    tau: float,
) -> ContinuousModel:

    if tau <= 0.0:
        raise ValueError(
            "tau must be positive"
        )

    A = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, -1.0 / tau],
        ],
        dtype=float,
    )

    B = np.array(
        [
            [0.0],
            [0.0],
            [0.0],
            [1.0 / tau],
        ],
        dtype=float,
    )

    G = np.array(
        [
            [0.0],
            [1.0],
            [0.0],
            [0.0],
        ],
        dtype=float,
    )

    E = np.array(
        [
            [0.0],
            [0.0],
            [0.0],
            [1.0],
        ],
        dtype=float,
    )

    return ContinuousModel(
        A=A,
        B=B,
        G=G,
        E=E,
    )


def exact_zoh(
    model: ContinuousModel,
    dt: float,
) -> DiscreteModel:

    if dt <= 0.0:
        raise ValueError(
            "sampling period must be positive"
        )

    U = np.hstack(
        (
            model.B,
            model.G,
            model.E,
        )
    )

    n = model.A.shape[0]
    m = U.shape[1]

    M = np.zeros(
        (n + m, n + m),
        dtype=float,
    )

    M[:n, :n] = model.A
    M[:n, n:] = U

    Md = expm(M * dt)

    Ud = Md[:n, n:]

    return DiscreteModel(
        Ad=Md[:n, :n],
        Bd=Ud[:, [0]],
        Gd=Ud[:, [1]],
        Ed=Ud[:, [2]],
    )


def zoh_step(
    model: DiscreteModel,
    x: np.ndarray,
    u: float,
    a_pred: float,
    w: float,
) -> np.ndarray:

    return (
        model.Ad @ x
        + model.Bd[:, 0] * u
        + model.Gd[:, 0] * a_pred
        + model.Ed[:, 0] * w
    )


def ivp_step(
    model: ContinuousModel,
    dt: float,
    x0: np.ndarray,
    u: float,
    a_pred: float,
    w: float,
) -> np.ndarray:

    def rhs(
        _t: float,
        x: np.ndarray,
    ) -> np.ndarray:

        return (
            model.A @ x
            + model.B[:, 0] * u
            + model.G[:, 0] * a_pred
            + model.E[:, 0] * w
        )

    sol = solve_ivp(
        rhs,
        (0.0, dt),
        x0,
        method="DOP853",
        rtol=1.0e-12,
        atol=1.0e-14,
    )

    if not sol.success:
        raise RuntimeError(
            sol.message
        )

    return sol.y[:, -1]


def stale_bounds(
    age: float,
    tau: float,
    delta_bar: float,
) -> tuple[float, float]:

    if age < 0.0:
        raise ValueError(
            "age must be nonnegative"
        )

    if tau <= 0.0:
        raise ValueError(
            "tau must be positive"
        )

    if delta_bar < 0.0:
        raise ValueError(
            "delta_bar must be nonnegative"
        )

    e_a = (
        delta_bar
        * tau
        * (
            1.0
            - math.exp(
                -age / tau
            )
        )
    )

    e_v = (
        delta_bar
        * (
            tau * age
            -
            tau * tau
            * (
                1.0
                -
                math.exp(
                    -age / tau
                )
            )
        )
    )

    return e_v, e_a


def stale_error_exact_piecewise(
    age: float,
    tau: float,
    deltas: np.ndarray,
) -> np.ndarray:

    if age <= 0.0:
        return np.zeros(
            2,
            dtype=float,
        )

    segments = len(deltas)

    if segments <= 0:
        raise ValueError(
            "deltas must be nonempty"
        )

    h = age / segments

    A = np.array(
        [
            [0.0, 1.0],
            [0.0, -1.0 / tau],
        ],
        dtype=float,
    )

    B = np.array(
        [
            [0.0],
            [1.0],
        ],
        dtype=float,
    )

    M = np.zeros(
        (3, 3),
        dtype=float,
    )

    M[:2, :2] = A
    M[:2, 2:] = B

    Md = expm(M * h)

    Ad = Md[:2, :2]
    Bd = Md[:2, 2]

    x = np.zeros(
        2,
        dtype=float,
    )

    for delta in deltas:

        x = (
            Ad @ x
            + Bd * float(delta)
        )

    return x


def run_zoh_ivp_validation(
    rng: np.random.Generator,
    continuous: ContinuousModel,
    discrete: DiscreteModel,
    cfg: dict,
) -> np.ndarray:

    trials = int(
        cfg["validation"][
            "zoh_ivp_trials"
        ]
    )

    errors: list[float] = []

    for _ in range(trials):

        x0 = rng.uniform(
            low=[
                -20.0,
                -10.0,
                0.0,
                -8.0,
            ],
            high=[
                120.0,
                10.0,
                40.0,
                3.0,
            ],
        ).astype(float)

        u = float(
            rng.uniform(
                cfg["plant"]["u_min"],
                cfg["plant"]["u_max"],
            )
        )

        a_pred = float(
            rng.uniform(
                cfg["plant"]["a_min"],
                cfg["plant"]["a_max"],
            )
        )

        w = float(
            rng.uniform(
                -cfg["uncertainty"][
                    "follower_actuation_abs"
                ],
                cfg["uncertainty"][
                    "follower_actuation_abs"
                ],
            )
        )

        x_zoh = zoh_step(
            discrete,
            x0,
            u,
            a_pred,
            w,
        )

        x_ivp = ivp_step(
            continuous,
            float(
                cfg["plant"]["Ts"]
            ),
            x0,
            u,
            a_pred,
            w,
        )

        errors.append(
            float(
                np.max(
                    np.abs(
                        x_zoh
                        -
                        x_ivp
                    )
                )
            )
        )

    return np.asarray(
        errors,
        dtype=float,
    )


def run_semigroup_validation(
    rng: np.random.Generator,
    continuous: ContinuousModel,
    cfg: dict,
) -> np.ndarray:

    trials = int(
        cfg["validation"][
            "semigroup_trials"
        ]
    )

    Ts = float(
        cfg["plant"]["Ts"]
    )

    Ad = exact_zoh(
        continuous,
        Ts,
    ).Ad

    errors: list[float] = []

    for _ in range(trials):

        n = int(
            rng.integers(
                1,
                21,
            )
        )

        lhs = np.linalg.matrix_power(
            Ad,
            n,
        )

        rhs = expm(
            continuous.A
            * (n * Ts)
        )

        errors.append(
            float(
                np.max(
                    np.abs(
                        lhs - rhs
                    )
                )
            )
        )

    return np.asarray(
        errors,
        dtype=float,
    )


def run_stale_random_stress(
    rng: np.random.Generator,
    cfg: dict,
) -> tuple[
    float,
    float,
    float,
]:

    trials = int(
        cfg["validation"][
            "stale_random_trials"
        ]
    )

    segments = int(
        cfg["validation"][
            "stale_segments"
        ]
    )

    tau = float(
        cfg["plant"][
            "tau_predecessor"
        ]
    )

    delta_bar = float(
        cfg["uncertainty"][
            "delta_bar_pred"
        ]
    )

    age_max = float(
        cfg["age"][
            "max_seconds"
        ]
    )

    max_ratio_v = 0.0
    max_ratio_a = 0.0
    max_violation = 0.0

    for _ in range(trials):

        age = float(
            rng.uniform(
                1.0e-8,
                age_max,
            )
        )

        deltas = rng.uniform(
            -delta_bar,
            delta_bar,
            size=segments,
        )

        e_v, e_a = (
            stale_error_exact_piecewise(
                age,
                tau,
                deltas,
            )
        )

        b_v, b_a = stale_bounds(
            age,
            tau,
            delta_bar,
        )

        max_ratio_v = max(
            max_ratio_v,
            abs(e_v)
            / (b_v + 1.0e-30),
        )

        max_ratio_a = max(
            max_ratio_a,
            abs(e_a)
            / (b_a + 1.0e-30),
        )

        max_violation = max(
            max_violation,
            abs(e_v) - b_v,
            abs(e_a) - b_a,
        )

    return (
        max_ratio_v,
        max_ratio_a,
        max_violation,
    )


def run_worst_case_tightness(
    cfg: dict,
) -> float:

    tau = float(
        cfg["plant"][
            "tau_predecessor"
        ]
    )

    delta_bar = float(
        cfg["uncertainty"][
            "delta_bar_pred"
        ]
    )

    age_max = float(
        cfg["age"][
            "max_seconds"
        ]
    )

    ages = np.linspace(
        0.01,
        age_max,
        64,
    )

    errors: list[float] = []

    for age in ages:

        for sign in (-1.0, 1.0):

            e_v, e_a = (
                stale_error_exact_piecewise(
                    float(age),
                    tau,
                    np.full(
                        32,
                        sign * delta_bar,
                    ),
                )
            )

            b_v, b_a = (
                stale_bounds(
                    float(age),
                    tau,
                    delta_bar,
                )
            )

            errors.extend(
                (
                    abs(
                        abs(e_v)
                        -
                        b_v
                    ),
                    abs(
                        abs(e_a)
                        -
                        b_a
                    ),
                )
            )

    return float(
        max(errors)
    )


def write_age_csv(
    path: Path,
    ages: np.ndarray,
    vel_bounds: np.ndarray,
    acc_bounds: np.ndarray,
) -> None:

    with path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "age_s",
                "velocity_error_bound",
                "acceleration_error_bound",
            ]
        )

        for age, e_v, e_a in zip(
            ages,
            vel_bounds,
            acc_bounds,
        ):

            writer.writerow(
                [
                    f"{age:.12g}",
                    f"{e_v:.12g}",
                    f"{e_a:.12g}",
                ]
            )


def make_figures(
    ages: np.ndarray,
    vel_bounds: np.ndarray,
    acc_bounds: np.ndarray,
    zoh_errors: np.ndarray,
    age_figure: Path,
    zoh_figure: Path,
) -> None:

    fig, ax = plt.subplots(
        figsize=(7.2, 4.6)
    )

    ax.plot(
        ages,
        vel_bounds,
        label="velocity-error bound",
    )

    ax.plot(
        ages,
        acc_bounds,
        label="acceleration-error bound",
    )

    ax.set_xlabel(
        "Authenticated information age (s)"
    )

    ax.set_ylabel(
        "Worst-case bound"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        age_figure,
        dpi=220,
    )

    plt.close(fig)

    fig, ax = plt.subplots(
        figsize=(7.2, 4.6)
    )

    ax.hist(
        zoh_errors,
        bins=min(
            30,
            max(
                5,
                len(zoh_errors) // 4,
            ),
        ),
    )

    ax.set_xlabel(
        "max abs error: exact ZOH vs DOP853"
    )

    ax.set_ylabel(
        "count"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        zoh_figure,
        dpi=220,
    )

    plt.close(fig)


def main() -> int:

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

    Ts = float(
        cfg["plant"]["Ts"]
    )

    tau = float(
        cfg["plant"][
            "tau_follower"
        ]
    )

    continuous = (
        continuous_link_model(
            tau
        )
    )

    discrete = exact_zoh(
        continuous,
        Ts,
    )

    zoh_errors = (
        run_zoh_ivp_validation(
            rng,
            continuous,
            discrete,
            cfg,
        )
    )

    semigroup_errors = (
        run_semigroup_validation(
            rng,
            continuous,
            cfg,
        )
    )

    (
        max_ratio_v,
        max_ratio_a,
        max_stale_violation,
    ) = run_stale_random_stress(
        rng,
        cfg,
    )

    worst_tightness = (
        run_worst_case_tightness(
            cfg
        )
    )

    age_points = int(
        cfg["age"][
            "grid_points"
        ]
    )

    ages = np.linspace(
        0.0,
        float(
            cfg["age"][
                "max_seconds"
            ]
        ),
        age_points,
    )

    bounds = np.asarray(
        [
            stale_bounds(
                float(age),
                float(
                    cfg["plant"][
                        "tau_predecessor"
                    ]
                ),
                float(
                    cfg["uncertainty"][
                        "delta_bar_pred"
                    ]
                ),
            )
            for age in ages
        ],
        dtype=float,
    )

    vel_bounds = bounds[:, 0]
    acc_bounds = bounds[:, 1]

    d_vel = np.diff(
        vel_bounds
    )

    d_acc = np.diff(
        acc_bounds
    )

    checks = {

        "CONFIG_PROVISIONAL_ONLY":
            (
                cfg["status"]
                ==
                "PROVISIONAL_ENGINEERING_BASELINE_NOT_FOR_MANUSCRIPT"
            ),

        "MATRICES_FINITE":
            all(
                np.isfinite(
                    matrix
                ).all()
                for matrix in (
                    discrete.Ad,
                    discrete.Bd,
                    discrete.Gd,
                    discrete.Ed,
                )
            ),

        "ZOH_DIMENSIONS":
            (
                discrete.Ad.shape
                ==
                (4, 4)
                and
                all(
                    matrix.shape
                    ==
                    (4, 1)
                    for matrix in (
                        discrete.Bd,
                        discrete.Gd,
                        discrete.Ed,
                    )
                )
            ),

        "ZOH_VS_IVP":
            (
                float(
                    np.max(
                        zoh_errors
                    )
                )
                <=
                float(
                    cfg["validation"][
                        "zoh_ivp_tolerance"
                    ]
                )
            ),

        "DISCRETE_SEMIGROUP":
            (
                float(
                    np.max(
                        semigroup_errors
                    )
                )
                <=
                float(
                    cfg["validation"][
                        "semigroup_tolerance"
                    ]
                )
            ),

        "ZERO_AGE_ZERO_ERROR":
            (
                abs(
                    vel_bounds[0]
                )
                <
                1.0e-14
                and
                abs(
                    acc_bounds[0]
                )
                <
                1.0e-14
            ),

        "STALE_VEL_BOUND_MONOTONE":
            (
                float(
                    np.min(
                        d_vel
                    )
                )
                >=
                -1.0e-12
            ),

        "STALE_ACC_BOUND_MONOTONE":
            (
                float(
                    np.min(
                        d_acc
                    )
                )
                >=
                -1.0e-12
            ),

        "STALE_RANDOM_ENVELOPE":
            (
                max_stale_violation
                <=
                float(
                    cfg["validation"][
                        "stale_tolerance"
                    ]
                )
            ),

        "STALE_WORST_CASE_TIGHT":
            (
                worst_tightness
                <=
                float(
                    cfg["validation"][
                        "worst_case_tolerance"
                    ]
                )
            ),

        "CONTROL_BOUNDS_VALID":
            (
                float(
                    cfg["plant"][
                        "u_min"
                    ]
                )
                <
                0.0
                <
                float(
                    cfg["plant"][
                        "u_max"
                    ]
                )
            ),

        "SAFETY_BOUNDS_VALID":
            (
                float(
                    cfg["safety"][
                        "d_min"
                    ]
                )
                >
                0.0
                and
                float(
                    cfg["safety"][
                        "v_max"
                    ]
                )
                >
                float(
                    cfg["safety"][
                        "v_min"
                    ]
                )
                >=
                0.0
            ),

        "FALLBACK_SWITCH_BOUND_VALID":
            (
                int(
                    cfg["fallback"][
                        "switch_steps"
                    ]
                )
                >=
                0
            ),

        "SERVICE_PROFILE_BOUNDS_VALID":
            all(
                int(
                    profile[
                        "eligible_bound_steps"
                    ]
                )
                >=
                0
                and
                int(
                    profile[
                        "max_loss_burst"
                    ]
                )
                >=
                0
                for profile
                in
                cfg[
                    "diagnostic_service_profiles"
                ].values()
            ),
    }

    checks = {
        key: bool(value)
        for key, value
        in checks.items()
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

    age_csv = (
        RESULTS_DIR
        /
        f"P1V2_AGE_BOUNDS_{stamp}.csv"
    )

    age_figure = (
        FIG_DIR
        /
        f"P1V2_AGE_BOUNDS_{stamp}.png"
    )

    zoh_figure = (
        FIG_DIR
        /
        f"P1V2_ZOH_IVP_ERROR_{stamp}.png"
    )

    write_age_csv(
        age_csv,
        ages,
        vel_bounds,
        acc_bounds,
    )

    make_figures(
        ages,
        vel_bounds,
        acc_bounds,
        zoh_errors,
        age_figure,
        zoh_figure,
    )

    result = {

        "schema":
            "SCV_P1_MODEL_VALIDATION_RESULT_V2",

        "status":
            status,

        "timestamp_utc":
            stamp,

        "provisional_parameters_only":
            True,

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
            checks,

        "metrics": {

            "zoh_ivp_max_abs_error":
                float(
                    np.max(
                        zoh_errors
                    )
                ),

            "zoh_ivp_p95_abs_error":
                float(
                    np.quantile(
                        zoh_errors,
                        0.95,
                    )
                ),

            "semigroup_max_abs_error":
                float(
                    np.max(
                        semigroup_errors
                    )
                ),

            "stale_random_max_velocity_ratio":
                float(
                    max_ratio_v
                ),

            "stale_random_max_acceleration_ratio":
                float(
                    max_ratio_a
                ),

            "stale_random_max_violation":
                float(
                    max_stale_violation
                ),

            "stale_worst_case_tightness_error":
                float(
                    worst_tightness
                ),

            "age_velocity_bound_max":
                float(
                    np.max(
                        vel_bounds
                    )
                ),

            "age_acceleration_bound_max":
                float(
                    np.max(
                        acc_bounds
                    )
                ),

            "age_velocity_increment_min":
                float(
                    np.min(
                        d_vel
                    )
                ),

            "age_acceleration_increment_min":
                float(
                    np.min(
                        d_acc
                    )
                ),

            "spectral_radius_Ad":
                float(
                    np.max(
                        np.abs(
                            np.linalg.eigvals(
                                discrete.Ad
                            )
                        )
                    )
                ),
        },

        "sampled_model": {

            "Ad":
                discrete.Ad.tolist(),

            "Bd":
                discrete.Bd.tolist(),

            "Gd":
                discrete.Gd.tolist(),

            "Ed":
                discrete.Ed.tolist(),
        },

        "artifacts": {

            "age_csv":
                str(
                    age_csv
                ),

            "age_figure":
                str(
                    age_figure
                ),

            "zoh_figure":
                str(
                    zoh_figure
                ),
        },
    }

    result_path = (
        RESULTS_DIR
        /
        f"P1V2_MODEL_VALIDATION_{stamp}.json"
    )

    latest_path = (
        RESULTS_DIR
        /
        "P1V2_LATEST.json"
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
        f"P1V2_MANIFEST_{stamp}.sha256"
    )

    manifest_files = [
        CFG_PATH,
        Path(__file__),
        result_path,
        age_csv,
        age_figure,
        zoh_figure,
    ]

    manifest_text = "\n".join(
        (
            f"{sha256_file(path)}"
            f"  {path}"
        )
        for path
        in manifest_files
    ) + "\n"

    atomic_write_text(
        manifest_path,
        manifest_text,
    )

    print(
        "=== SCV P1-V2 MODEL VALIDATION ==="
    )

    for key, value in checks.items():

        print(
            f"{key}="
            f"{'PASS' if value else 'FAIL'}"
        )

    for key, value in (
        result["metrics"]
    ).items():

        print(
            f"{key.upper()}="
            f"{value:.12g}"
        )

    print(
        f"P1V2_MODEL_VALIDATION={status}"
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
