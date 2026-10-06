from __future__ import annotations

import copy
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

sys.path.insert(
    0,
    str(SRC),
)

import p2b_hybrid_fallback_v1 as fb


CFG_PATH = (
    ROOT
    / "01_config"
    / "p3b0_freshness_service_audit_v1.json"
)

P1_CFG_PATH = (
    ROOT
    / "01_config"
    / "p1_validation_v2.json"
)

P2B_CFG_PATH = (
    ROOT
    / "01_config"
    / "p2b_hybrid_fallback_v1.json"
)

P3A_CFG_PATH = (
    ROOT
    / "01_config"
    / "p3a_information_contract_v1.json"
)

P3A2_CFG_PATH = (
    ROOT
    / "01_config"
    / "p3a2_slew_contract_v1.json"
)

RESULTS_DIR = ROOT / "04_results"
FIGURES_DIR = ROOT / "05_figures"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def load_json(path: Path) -> dict:

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def atomic_write(
    path: Path,
    text: str,
) -> None:

    tmp = path.with_suffix(
        path.suffix
        + ".tmp"
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
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def sample_augmented_states(
    power: int,
    seed: int,
    cfg: dict,
    p2b_cfg: dict,
    p3a_cfg: dict,
):

    sampler = qmc.Sobol(
        d=7,
        scramble=True,
        seed=seed,
    )

    U = sampler.random_base2(
        int(power)
    )

    sd = p2b_cfg[
        "state_domain"
    ]

    pred = p3a_cfg[
        "predecessor"
    ]

    current_age_max = float(
        cfg["freshness"][
            "current_age_max_seconds"
        ]
    )

    lower = np.array(
        [
            sd["d_min"],
            sd["delta_v_min"],
            sd["v_f_min"],
            sd["a_f_min"],
            pred["command_min"],
            pred["command_min"],
            0.0,
        ],
        dtype=float,
    )

    upper = np.array(
        [
            sd["d_max"],
            sd["delta_v_max"],
            sd["v_f_max"],
            sd["a_f_max"],
            pred["command_max"],
            pred["command_max"],
            current_age_max,
        ],
        dtype=float,
    )

    X = (
        lower
        +
        U
        *
        (
            upper
            -
            lower
        )
    )

    # Columns:
    # 0 d
    # 1 delta_v = v_p - v_f
    # 2 v_f
    # 3 a_f
    # 4 authenticated predecessor acceleration bar_a
    # 5 authenticated predecessor command bar_u
    # 6 current authenticated information age

    vf = X[:, 2]
    vp = (
        vf
        +
        X[:, 1]
    )

    physical = (
        (vp >= sd["v_p_min"])
        &
        (vp <= sd["v_p_max"])
    )

    return (
        X[physical],
        vp[physical],
        float(
            np.mean(
                physical
            )
        ),
    )


def lower_acceleration_error(
    age,
    bar_u,
    slew_rate: float,
    p3a_cfg: dict,
):
    """
    Vectorized exact lower acceleration-error extremal
    under command slew and saturation.
    """

    age = np.asarray(
        age,
        dtype=float,
    )

    bar_u = np.asarray(
        bar_u,
        dtype=float,
    )

    p = p3a_cfg[
        "predecessor"
    ]

    tau = float(
        p["tau"]
    )

    u_min = float(
        p["command_min"]
    )

    w_abs = float(
        p["disturbance_abs"]
    )

    J = float(
        slew_rate
    )

    if J <= 0.0:

        decay = np.exp(
            -age / tau
        )

        return (
            -w_abs
            *
            tau
            *
            (
                1.0
                -
                decay
            )
        )

    t_sat = np.maximum(
        0.0,
        (
            bar_u
            -
            u_min
        )
        /
        J,
    )

    t1 = np.minimum(
        age,
        t_sat,
    )

    decay1 = np.exp(
        -t1 / tau
    )

    ka1 = (
        tau
        *
        (
            1.0
            -
            decay1
        )
    )

    kv1 = (
        tau * t1
        -
        tau**2
        *
        (
            1.0
            -
            decay1
        )
    )

    r0 = -w_abs
    r1 = -J / tau

    ea1 = (
        r0 * ka1
        +
        r1 * kv1
    )

    h2 = np.maximum(
        age - t1,
        0.0,
    )

    decay2 = np.exp(
        -h2 / tau
    )

    ka2 = (
        tau
        *
        (
            1.0
            -
            decay2
        )
    )

    r_const = (
        (
            u_min
            -
            bar_u
        )
        /
        tau
        -
        w_abs
    )

    return (
        ea1 * decay2
        +
        r_const * ka2
    )


def predecessor_acceleration_lower(
    age,
    bar_a,
    bar_u,
    slew_rate: float,
    p3a_cfg: dict,
    p2b_cfg: dict,
):
    """
    Lower current predecessor acceleration consistent with
    authenticated (bar_a, bar_u), information age, and
    the validated slew contract.
    """

    age = np.asarray(
        age,
        dtype=float,
    )

    bar_a = np.asarray(
        bar_a,
        dtype=float,
    )

    bar_u = np.asarray(
        bar_u,
        dtype=float,
    )

    tau = float(
        p3a_cfg[
            "predecessor"
        ][
            "tau"
        ]
    )

    predicted = (
        bar_u
        +
        (
            bar_a
            -
            bar_u
        )
        *
        np.exp(
            -age / tau
        )
    )

    error_lower = (
        lower_acceleration_error(
            age,
            bar_u,
            slew_rate,
            p3a_cfg,
        )
    )

    lower = (
        predicted
        +
        error_lower
    )

    sd = p2b_cfg[
        "state_domain"
    ]

    return np.clip(
        lower,
        float(
            sd["a_p_min"]
        ),
        float(
            sd["a_p_max"]
        ),
    )


def control_specific_p2b_config(
    p2b_cfg: dict,
    protective_command: float,
) -> dict:

    local = copy.deepcopy(
        p2b_cfg
    )

    local[
        "follower"
    ][
        "fallback_command"
    ] = float(
        protective_command
    )

    return local


def classify_flat(
    d,
    vf,
    af,
    vp,
    bar_a,
    bar_u,
    age,
    protective_command: float,
    resolutions,
    slew_rate: float,
    p2b_cfg: dict,
    p3a_cfg: dict,
):
    """
    Robust recoverability certificate using:
      - locally sensed current relative motion;
      - authenticated acceleration/command uncertainty;
      - cooperative protective command;
      - all-time relative-loss bracket.

    This is deliberately NOT labeled K_c^star.
    """

    ap_lower = (
        predecessor_acceleration_lower(
            age,
            bar_a,
            bar_u,
            slew_rate,
            p3a_cfg,
            p2b_cfg,
        )
    )

    local_cfg = (
        control_specific_p2b_config(
            p2b_cfg,
            protective_command,
        )
    )

    (
        loss_lower,
        loss_upper,
        snapshots,
    ) = fb.relative_loss_bracket(
        np.asarray(
            vf,
            dtype=float,
        ),
        np.asarray(
            af,
            dtype=float,
        ),
        np.asarray(
            vp,
            dtype=float,
        ),
        np.asarray(
            ap_lower,
            dtype=float,
        ),
        [
            int(x)
            for x
            in resolutions
        ],
        local_cfg,
    )

    d_min = float(
        p2b_cfg[
            "state_domain"
        ][
            "d_min"
        ]
    )

    d = np.asarray(
        d,
        dtype=float,
    )

    certified = (
        d
        >=
        d_min
        +
        loss_upper
    )

    candidate = (
        d
        >=
        d_min
        +
        loss_lower
    )

    return {
        "certified":
            certified,

        "candidate":
            candidate,

        "loss_lower":
            loss_lower,

        "loss_upper":
            loss_upper,

        "ap_lower":
            ap_lower,

        "snapshots":
            snapshots,
    }


def freshness_fiber_audit(
    X,
    vp,
    cfg,
    p2b_cfg,
    p3a_cfg,
):
    ages = np.linspace(
        0.0,
        float(
            cfg["freshness"][
                "age_max_seconds"
            ]
        ),
        int(
            cfg["freshness"][
                "fiber_points"
            ]
        ),
    )

    n = len(X)
    m = len(ages)

    d = np.repeat(
        X[:, 0],
        m,
    )

    vf = np.repeat(
        X[:, 2],
        m,
    )

    af = np.repeat(
        X[:, 3],
        m,
    )

    vp_flat = np.repeat(
        vp,
        m,
    )

    bar_a = np.repeat(
        X[:, 4],
        m,
    )

    bar_u = np.repeat(
        X[:, 5],
        m,
    )

    age_flat = np.tile(
        ages,
        n,
    )

    result = classify_flat(
        d=d,
        vf=vf,
        af=af,
        vp=vp_flat,
        bar_a=bar_a,
        bar_u=bar_u,
        age=age_flat,
        protective_command=float(
            cfg[
                "cooperative_control"
            ][
                "candidate_protective_command"
            ]
        ),
        resolutions=cfg[
            "temporal_bracket"
        ][
            "resolutions"
        ],
        slew_rate=float(
            cfg[
                "information_contract"
            ][
                "candidate_slew_rate"
            ]
        ),
        p2b_cfg=p2b_cfg,
        p3a_cfg=p3a_cfg,
    )

    certified = (
        result[
            "certified"
        ]
        .reshape(
            n,
            m,
        )
    )

    candidate = (
        result[
            "candidate"
        ]
        .reshape(
            n,
            m,
        )
    )

    ambiguity = (
        candidate
        &
        ~certified
    )

    false_to_true = (
        ~certified[:, :-1]
        &
        certified[:, 1:]
    )

    violation_sample = np.any(
        false_to_true,
        axis=1,
    )

    downward_closed = (
        ~violation_sample
    )

    nonempty = np.any(
        certified,
        axis=1,
    )

    full = np.all(
        certified,
        axis=1,
    )

    threshold = np.full(
        n,
        np.nan,
        dtype=float,
    )

    for i in range(n):

        if (
            downward_closed[i]
            and
            nonempty[i]
        ):

            idx = np.flatnonzero(
                certified[i]
            )

            threshold[i] = (
                ages[
                    idx[-1]
                ]
            )

    width = (
        result["loss_upper"]
        -
        result["loss_lower"]
    )

    return {
        "ages":
            ages,

        "certified":
            certified,

        "candidate":
            candidate,

        "ambiguity":
            ambiguity,

        "downward_closed":
            downward_closed,

        "nonempty":
            nonempty,

        "full":
            full,

        "threshold":
            threshold,

        "width":
            width,
    }


def service_state_audit(
    X,
    vp,
    cfg,
    p1_cfg,
    p2b_cfg,
    p3a_cfg,
):
    profiles = (
        p1_cfg[
            "diagnostic_service_profiles"
        ]
    )

    names = list(
        profiles.keys()
    )

    n = len(X)
    p = len(names)

    Ts = float(
        p1_cfg[
            "plant"
        ][
            "Ts"
        ]
    )

    age_max = float(
        cfg["freshness"][
            "age_max_seconds"
        ]
    )

    effective_age = np.empty(
        (n, p),
        dtype=float,
    )

    within_contract = np.ones(
        (n, p),
        dtype=bool,
    )

    for j, name in enumerate(
        names
    ):

        rho = int(
            profiles[
                name
            ][
                "eligible_bound_steps"
            ]
        )

        age = (
            X[:, 6]
            +
            rho * Ts
        )

        effective_age[
            :,
            j
        ] = age

        within_contract[
            :,
            j
        ] = (
            age
            <=
            age_max
            +
            1e-12
        )

    d = np.repeat(
        X[:, 0],
        p,
    )

    vf = np.repeat(
        X[:, 2],
        p,
    )

    af = np.repeat(
        X[:, 3],
        p,
    )

    vp_flat = np.repeat(
        vp,
        p,
    )

    bar_a = np.repeat(
        X[:, 4],
        p,
    )

    bar_u = np.repeat(
        X[:, 5],
        p,
    )

    age_flat = (
        effective_age
        .reshape(-1)
    )

    result = classify_flat(
        d=d,
        vf=vf,
        af=af,
        vp=vp_flat,
        bar_a=bar_a,
        bar_u=bar_u,
        age=age_flat,
        protective_command=float(
            cfg[
                "cooperative_control"
            ][
                "candidate_protective_command"
            ]
        ),
        resolutions=cfg[
            "temporal_bracket"
        ][
            "resolutions"
        ],
        slew_rate=float(
            cfg[
                "information_contract"
            ][
                "candidate_slew_rate"
            ]
        ),
        p2b_cfg=p2b_cfg,
        p3a_cfg=p3a_cfg,
    )

    certified = (
        result[
            "certified"
        ]
        .reshape(
            n,
            p,
        )
    )

    candidate = (
        result[
            "candidate"
        ]
        .reshape(
            n,
            p,
        )
    )

    certified &= (
        within_contract
    )

    candidate &= (
        within_contract
    )

    fractions = {
        name:
            float(
                np.mean(
                    certified[
                        :,
                        j
                    ]
                )
            )
        for j, name
        in enumerate(
            names
        )
    }

    inclusion_violations = {}

    for j in range(
        p - 1
    ):

        lhs = names[j]
        rhs = names[
            j + 1
        ]

        # Slower service should not create new certified states
        # if age ordering behaves monotonically.
        count = int(
            np.count_nonzero(
                certified[
                    :,
                    j + 1
                ]
                &
                ~certified[
                    :,
                    j
                ]
            )
        )

        inclusion_violations[
            f"{rhs}_not_subset_{lhs}"
        ] = count

    return {
        "names":
            names,

        "effective_age":
            effective_age,

        "certified":
            certified,

        "candidate":
            candidate,

        "fractions":
            fractions,

        "inclusion_violations":
            inclusion_violations,
    }


def authority_sensitivity(
    X,
    vp,
    cfg,
    p1_cfg,
    p2b_cfg,
    p3a_cfg,
):
    count = min(
        len(X),
        2
        **
        int(
            cfg["qmc"][
                "sensitivity_power"
            ]
        ),
    )

    X = X[
        :count
    ]

    vp = vp[
        :count
    ]

    profiles = (
        p1_cfg[
            "diagnostic_service_profiles"
        ]
    )

    names = list(
        profiles.keys()
    )

    Ts = float(
        p1_cfg[
            "plant"
        ][
            "Ts"
        ]
    )

    p = len(
        names
    )

    rows = []

    for command in cfg[
        "cooperative_control"
    ][
        "sensitivity_commands"
    ]:

        effective_age = np.column_stack(
            [
                (
                    X[:, 6]
                    +
                    int(
                        profiles[
                            name
                        ][
                            "eligible_bound_steps"
                        ]
                    )
                    * Ts
                )
                for name
                in names
            ]
        )

        d = np.repeat(
            X[:, 0],
            p,
        )

        vf = np.repeat(
            X[:, 2],
            p,
        )

        af = np.repeat(
            X[:, 3],
            p,
        )

        vp_flat = np.repeat(
            vp,
            p,
        )

        bar_a = np.repeat(
            X[:, 4],
            p,
        )

        bar_u = np.repeat(
            X[:, 5],
            p,
        )

        ages = (
            effective_age
            .reshape(-1)
        )

        result = classify_flat(
            d=d,
            vf=vf,
            af=af,
            vp=vp_flat,
            bar_a=bar_a,
            bar_u=bar_u,
            age=ages,
            protective_command=float(
                command
            ),
            resolutions=[
                int(
                    cfg[
                        "temporal_bracket"
                    ][
                        "resolutions"
                    ][-1]
                )
            ],
            slew_rate=float(
                cfg[
                    "information_contract"
                ][
                    "candidate_slew_rate"
                ]
            ),
            p2b_cfg=p2b_cfg,
            p3a_cfg=p3a_cfg,
        )

        certified = (
            result[
                "certified"
            ]
            .reshape(
                count,
                p,
            )
        )

        age_max = float(
            cfg["freshness"][
                "age_max_seconds"
            ]
        )

        certified &= (
            effective_age
            <=
            age_max
            +
            1e-12
        )

        row = {
            "cooperative_command":
                float(
                    command
                )
        }

        for j, name in enumerate(
            names
        ):

            row[
                f"{name}_fraction"
            ] = float(
                np.mean(
                    certified[
                        :,
                        j
                    ]
                )
            )

        row[
            "ideal_vs_stressed_separation"
        ] = int(
            np.count_nonzero(
                certified[:, 0]
                &
                ~certified[:, -1]
            )
        )

        rows.append(
            row
        )

    return rows


def build_witness_rows(
    X,
    vp,
    service,
    cfg,
):
    names = (
        service[
            "names"
        ]
    )

    certified = (
        service[
            "certified"
        ]
    )

    fast_index = (
        names.index(
            "diagnostic_fast"
        )
        if "diagnostic_fast"
        in names
        else 0
    )

    stressed_index = (
        names.index(
            "diagnostic_stressed"
        )
        if "diagnostic_stressed"
        in names
        else
        len(names) - 1
    )

    mask = (
        certified[
            :,
            fast_index
        ]
        &
        ~certified[
            :,
            stressed_index
        ]
    )

    indices = np.flatnonzero(
        mask
    )

    max_rows = int(
        cfg["witness"][
            "max_rows"
        ]
    )

    indices = indices[
        :max_rows
    ]

    rows = []

    for i in indices:

        row = {
            "d":
                float(
                    X[i, 0]
                ),

            "delta_v":
                float(
                    X[i, 1]
                ),

            "v_f":
                float(
                    X[i, 2]
                ),

            "v_p":
                float(
                    vp[i]
                ),

            "a_f":
                float(
                    X[i, 3]
                ),

            "bar_a":
                float(
                    X[i, 4]
                ),

            "bar_u":
                float(
                    X[i, 5]
                ),

            "current_age":
                float(
                    X[i, 6]
                ),

            "fast_effective_age":
                float(
                    service[
                        "effective_age"
                    ][
                        i,
                        fast_index
                    ]
                ),

            "stressed_effective_age":
                float(
                    service[
                        "effective_age"
                    ][
                        i,
                        stressed_index
                    ]
                ),

            "fast_certified":
                int(
                    certified[
                        i,
                        fast_index
                    ]
                ),

            "stressed_certified":
                int(
                    certified[
                        i,
                        stressed_index
                    ]
                ),
        }

        rows.append(
            row
        )

    return rows


def write_csv(
    path: Path,
    rows: list[dict],
):
    if not rows:
        path.write_text(
            "",
            encoding="utf-8",
        )
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
        writer.writerows(
            rows
        )


def make_fiber_figure(
    path,
    fiber,
):
    certified = (
        fiber[
            "certified"
        ]
    )

    n_show = min(
        256,
        certified.shape[0],
    )

    subset = (
        certified[
            :n_show,
            :
        ]
    )

    ordering = np.argsort(
        np.sum(
            subset,
            axis=1,
        )
    )

    subset = subset[
        ordering
    ]

    fig, ax = plt.subplots(
        figsize=(8.0, 5.2)
    )

    ax.imshow(
        subset,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        extent=[
            fiber["ages"][0],
            fiber["ages"][-1],
            0,
            n_show,
        ],
    )

    ax.set_xlabel(
        "Authenticated information age (s)"
    )

    ax.set_ylabel(
        "QMC states sorted by viable-age count"
    )

    ax.set_title(
        "Diagnostic freshness-recoverability fibers"
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=240,
    )

    plt.close(fig)


def make_service_figure(
    path,
    service,
):
    names = (
        service[
            "names"
        ]
    )

    values = [
        service[
            "fractions"
        ][name]
        for name
        in names
    ]

    fig, ax = plt.subplots(
        figsize=(7.3, 4.7)
    )

    ax.bar(
        np.arange(
            len(names)
        ),
        values,
    )

    ax.set_xticks(
        np.arange(
            len(names)
        )
    )

    ax.set_xticklabels(
        [
            name.replace(
                "diagnostic_",
                "",
            )
            for name
            in names
        ],
        rotation=20,
    )

    ax.set_ylabel(
        "Certified recoverability fraction"
    )

    ax.set_title(
        "Diagnostic service-horizon dependence"
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=240,
    )

    plt.close(fig)


def make_authority_figure(
    path,
    rows,
    service_names,
):
    commands = [
        abs(
            row[
                "cooperative_command"
            ]
        )
        for row
        in rows
    ]

    fig, ax = plt.subplots(
        figsize=(7.5, 4.8)
    )

    for name in service_names:

        ax.plot(
            commands,
            [
                row[
                    f"{name}_fraction"
                ]
                for row
                in rows
            ],
            marker="o",
            label=name.replace(
                "diagnostic_",
                "",
            ),
        )

    ax.set_xlabel(
        "Protective cooperative braking magnitude"
    )

    ax.set_ylabel(
        "Certified recoverability fraction"
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

    p1_cfg = load_json(
        P1_CFG_PATH
    )

    p2b_cfg = load_json(
        P2B_CFG_PATH
    )

    p3a_cfg = load_json(
        P3A_CFG_PATH
    )

    p3a2_cfg = load_json(
        P3A2_CFG_PATH
    )

    X, vp, physical_fraction = (
        sample_augmented_states(
            int(
                cfg["qmc"][
                    "main_power"
                ]
            ),
            int(
                cfg[
                    "random_seed"
                ]
            ),
            cfg,
            p2b_cfg,
            p3a_cfg,
        )
    )

    fiber = freshness_fiber_audit(
        X,
        vp,
        cfg,
        p2b_cfg,
        p3a_cfg,
    )

    service = service_state_audit(
        X,
        vp,
        cfg,
        p1_cfg,
        p2b_cfg,
        p3a_cfg,
    )

    authority = authority_sensitivity(
        X,
        vp,
        cfg,
        p1_cfg,
        p2b_cfg,
        p3a_cfg,
    )

    witness_rows = build_witness_rows(
        X,
        vp,
        service,
        cfg,
    )

    ambiguity_fraction = float(
        np.mean(
            fiber[
                "ambiguity"
            ]
        )
    )

    width = fiber[
        "width"
    ]

    candidate_command = float(
        cfg[
            "cooperative_control"
        ][
            "candidate_protective_command"
        ]
    )

    fallback_command = float(
        p2b_cfg[
            "follower"
        ][
            "fallback_command"
        ]
    )

    downward_fraction = float(
        np.mean(
            fiber[
                "downward_closed"
            ]
        )
    )

    nonempty_fraction = float(
        np.mean(
            fiber[
                "nonempty"
            ]
        )
    )

    finite_thresholds = (
        fiber[
            "threshold"
        ][
            np.isfinite(
                fiber[
                    "threshold"
                ]
            )
        ]
    )

    if len(
        finite_thresholds
    ) > 0:

        threshold_mean = float(
            np.mean(
                finite_thresholds
            )
        )

        threshold_p10 = float(
            np.quantile(
                finite_thresholds,
                0.10,
            )
        )

        threshold_p90 = float(
            np.quantile(
                finite_thresholds,
                0.90,
            )
        )

    else:

        threshold_mean = math.nan
        threshold_p10 = math.nan
        threshold_p90 = math.nan

    names = service[
        "names"
    ]

    fast_index = (
        names.index(
            "diagnostic_fast"
        )
        if "diagnostic_fast"
        in names
        else 0
    )

    stressed_index = (
        names.index(
            "diagnostic_stressed"
        )
        if "diagnostic_stressed"
        in names
        else
        len(names) - 1
    )

    q_separation_count = int(
        np.count_nonzero(
            service[
                "certified"
            ][
                :,
                fast_index
            ]
            !=
            service[
                "certified"
            ][
                :,
                stressed_index
            ]
        )
    )

    q_fast_only_count = int(
        np.count_nonzero(
            service[
                "certified"
            ][
                :,
                fast_index
            ]
            &
            ~service[
                "certified"
            ][
                :,
                stressed_index
            ]
        )
    )

    checks = {
        "AUGMENTED_SAMPLE_NONEMPTY":
            (
                len(X) > 0
            ),

        "COOPERATIVE_AUTHORITY_DISTINCT_FROM_FALLBACK":
            (
                candidate_command
                >
                fallback_command
            ),

        "GENERAL_FRESHNESS_FIBER_NONEMPTY":
            (
                nonempty_fraction
                >
                0.0
            ),

        "TEMPORAL_BRACKET_ORDERED":
            bool(
                np.all(
                    fiber[
                        "candidate"
                    ]
                    |
                    ~fiber[
                        "certified"
                    ]
                )
            ),

        "TEMPORAL_AMBIGUITY_SMALL":
            (
                ambiguity_fraction
                <=
                float(
                    cfg[
                        "temporal_bracket"
                    ][
                        "ambiguity_fraction_tolerance"
                    ]
                )
            ),

        "TEMPORAL_P95_WIDTH_SMALL":
            (
                float(
                    np.quantile(
                        width,
                        0.95,
                    )
                )
                <=
                float(
                    cfg[
                        "temporal_bracket"
                    ][
                        "p95_width_tolerance_m"
                    ]
                )
            ),

        "SERVICE_PROFILE_EVALUATION_COMPLETE":
            (
                len(
                    service[
                        "fractions"
                    ]
                )
                ==
                len(
                    p1_cfg[
                        "diagnostic_service_profiles"
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

    diagnostics = {
        "empirical_downward_closed_fraction":
            downward_fraction,

        "empirical_age_order_counterexample_count":
            int(
                np.count_nonzero(
                    ~fiber[
                        "downward_closed"
                    ]
                )
            ),

        "q_fast_vs_stressed_different_verdict_count":
            q_separation_count,

        "q_fast_safe_stressed_unsafe_count":
            q_fast_only_count,

        "service_profile_inclusion_violations":
            service[
                "inclusion_violations"
            ],

        "architecture_service_effect_observed":
            bool(
                q_separation_count
                >
                0
            ),
    }

    metrics = {
        "physical_fraction_of_qmc_box":
            physical_fraction,

        "augmented_sample_count":
            int(
                len(X)
            ),

        "freshness_nonempty_fraction":
            nonempty_fraction,

        "freshness_full_age_range_fraction":
            float(
                np.mean(
                    fiber[
                        "full"
                    ]
                )
            ),

        "freshness_downward_closed_fraction":
            downward_fraction,

        "freshness_temporal_ambiguity_fraction":
            ambiguity_fraction,

        "freshness_gap_width_mean_m":
            float(
                np.mean(
                    width
                )
            ),

        "freshness_gap_width_p95_m":
            float(
                np.quantile(
                    width,
                    0.95,
                )
            ),

        "conditional_scalar_threshold_mean_s":
            threshold_mean,

        "conditional_scalar_threshold_p10_s":
            threshold_p10,

        "conditional_scalar_threshold_p90_s":
            threshold_p90,

        "q_fast_vs_stressed_different_verdict_count":
            q_separation_count,

        "q_fast_safe_stressed_unsafe_count":
            q_fast_only_count,

        "cooperative_candidate_command":
            candidate_command,

        "fallback_command":
            fallback_command,

        "candidate_slew_rate":
            float(
                cfg[
                    "information_contract"
                ][
                    "candidate_slew_rate"
                ]
            ),
    }

    for name, value in (
        service[
            "fractions"
        ]
        .items()
    ):

        metrics[
            f"service_{name}_fraction"
        ] = float(
            value
        )

    stamp = (
        datetime.now(
            timezone.utc
        )
        .strftime(
            "%Y%m%dT%H%M%SZ"
        )
    )

    witness_csv = (
        RESULTS_DIR
        /
        f"P3B0_Q_WITNESSES_{stamp}.csv"
    )

    authority_csv = (
        RESULTS_DIR
        /
        f"P3B0_AUTHORITY_SENSITIVITY_{stamp}.csv"
    )

    fiber_fig = (
        FIGURES_DIR
        /
        f"P3B0_FRESHNESS_FIBERS_{stamp}.png"
    )

    service_fig = (
        FIGURES_DIR
        /
        f"P3B0_SERVICE_STATE_EFFECT_{stamp}.png"
    )

    authority_fig = (
        FIGURES_DIR
        /
        f"P3B0_AUTHORITY_SENSITIVITY_{stamp}.png"
    )

    write_csv(
        witness_csv,
        witness_rows,
    )

    write_csv(
        authority_csv,
        authority,
    )

    make_fiber_figure(
        fiber_fig,
        fiber,
    )

    make_service_figure(
        service_fig,
        service,
    )

    make_authority_figure(
        authority_fig,
        authority,
        service[
            "names"
        ],
    )

    output = {
        "schema":
            "SCV_P3B0_FRESHNESS_SERVICE_AUDIT_RESULT_V1",

        "status":
            status,

        "timestamp_utc":
            stamp,

        "classification":
            (
                "diagnostic non-degeneracy audit; "
                "not the maximal augmented viability kernel "
                "and not a manuscript-final PQC service result"
            ),

        "candidate_parameters_final":
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

        "diagnostics":
            diagnostics,

        "metrics":
            metrics,

        "service_profiles":
            service[
                "fractions"
            ],

        "service_inclusion_violations":
            service[
                "inclusion_violations"
            ],

        "authority_sensitivity":
            authority,

        "witness_count_written":
            len(
                witness_rows
            ),

        "artifacts": {
            "q_witness_csv":
                str(
                    witness_csv
                ),

            "authority_csv":
                str(
                    authority_csv
                ),

            "freshness_fiber_figure":
                str(
                    fiber_fig
                ),

            "service_state_figure":
                str(
                    service_fig
                ),

            "authority_figure":
                str(
                    authority_fig
                ),
        },
    }

    result_path = (
        RESULTS_DIR
        /
        f"P3B0_FRESHNESS_SERVICE_AUDIT_{stamp}.json"
    )

    latest_path = (
        RESULTS_DIR
        /
        "P3B0_LATEST.json"
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
        f"P3B0_MANIFEST_{stamp}.sha256"
    )

    files = [
        CFG_PATH,
        P1_CFG_PATH,
        P2B_CFG_PATH,
        P3A_CFG_PATH,
        P3A2_CFG_PATH,
        Path(__file__),
        result_path,
        witness_csv,
        authority_csv,
        fiber_fig,
        service_fig,
        authority_fig,
    ]

    manifest = "\n".join(
        (
            f"{sha256_file(path)}"
            f"  {path}"
        )
        for path
        in files
    ) + "\n"

    atomic_write(
        manifest_path,
        manifest,
    )

    print(
        "=== SCV P3-B0 FRESHNESS-SERVICE AUDIT ==="
    )

    for key, value in checks.items():

        print(
            f"{key}="
            f"{'PASS' if value else 'FAIL'}"
        )

    print(
        "=== DIAGNOSTICS ==="
    )

    print(
        "EMPIRICAL_DOWNWARD_CLOSED_FRACTION="
        f"{downward_fraction:.12g}"
    )

    print(
        "AGE_ORDER_COUNTEREXAMPLES="
        f"{diagnostics['empirical_age_order_counterexample_count']}"
    )

    print(
        "Q_FAST_VS_STRESSED_DIFFERENT_VERDICTS="
        f"{q_separation_count}"
    )

    print(
        "Q_FAST_SAFE_STRESSED_UNSAFE="
        f"{q_fast_only_count}"
    )

    print(
        "ARCHITECTURE_SERVICE_EFFECT_OBSERVED="
        f"{'YES' if q_separation_count > 0 else 'NO'}"
    )

    print(
        "=== SERVICE FRACTIONS ==="
    )

    for name in service[
        "names"
    ]:

        print(
            f"{name.upper()}="
            f"{service['fractions'][name]:.12g}"
        )

    print(
        "=== CORE METRICS ==="
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
        "=== AUTHORITY SENSITIVITY ==="
    )

    for row in authority:

        print(
            "U_C="
            f"{row['cooperative_command']:.3f} "
            "IDEAL="
            f"{row['ideal_fraction']:.6f} "
            "FAST="
            f"{row['diagnostic_fast_fraction']:.6f} "
            "NOMINAL="
            f"{row['diagnostic_nominal_fraction']:.6f} "
            "STRESSED="
            f"{row['diagnostic_stressed_fraction']:.6f} "
            "IDEAL_VS_STRESSED_SEP="
            f"{row['ideal_vs_stressed_separation']}"
        )

    print(
        f"P3B0_FRESHNESS_SERVICE_AUDIT={status}"
    )

    print(
        "FINAL_KERNEL_CLAIM=NO"
    )

    print(
        "FINAL_PQC_PROFILE_CLAIM=NO"
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
