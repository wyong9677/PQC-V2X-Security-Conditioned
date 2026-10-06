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
SRC_DIR = Path(__file__).resolve().parent

sys.path.insert(
    0,
    str(SRC_DIR),
)

import p2b_hybrid_fallback_v1 as fb


CFG_PATH = (
    ROOT
    / "01_config"
    / "p2c_switching_guard_v1.json"
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

        for block in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def load_json(
    path: Path,
) -> dict:

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def switching_contract_check(
    cfg: dict,
    p1_cfg: dict,
    p2b_cfg: dict,
) -> dict:

    a_upper = float(
        cfg["switching"][
            "follower_acceleration_upper"
        ]
    )

    a_state_max = float(
        p2b_cfg["state_domain"][
            "a_f_max"
        ]
    )

    u_max = float(
        p1_cfg["plant"][
            "u_max"
        ]
    )

    tau = float(
        p2b_cfg["follower"][
            "tau"
        ]
    )

    w_max = float(
        p1_cfg["uncertainty"][
            "follower_actuation_abs"
        ]
    )

    equilibrium_upper = (
        u_max
        +
        tau * w_max
    )

    required_upper = max(
        a_state_max,
        equilibrium_upper,
    )

    return {
        "a_switch_upper":
            a_upper,

        "a_state_max":
            a_state_max,

        "a_equilibrium_upper":
            equilibrium_upper,

        "a_required_upper":
            required_upper,

        "valid":
            bool(
                a_upper
                + 1e-12
                >=
                required_upper
            ),
    }


def switch_interval_snapshots(
    vf: np.ndarray,
    vp: np.ndarray,
    ap: np.ndarray,
    T_sw: float,
    resolutions: list[int],
    a_switch_upper: float,
    p2b_cfg: dict,
    chunk_size: int = 512,
):

    n = len(vf)

    if T_sw <= 0.0:

        zeros = np.zeros(
            n,
            dtype=float,
        )

        return [
            (
                int(resolution),
                zeros.copy(),
                zeros.copy(),
            )
            for resolution
            in resolutions
        ]

    pars = fb.model_parameters(
        p2b_cfg
    )

    tp_stop = fb.stopping_time(
        vp,
        ap,
        pars["up"],
        pars["tp"],
        pars["wp"],
    )

    vp_peak = fb.peak_speed(
        vp,
        ap,
        pars["up"],
        pars["tp"],
        pars["wp"],
    )

    vf_upper_end = (
        vf
        +
        a_switch_upper
        * T_sw
    )

    lipschitz = np.maximum(
        vf_upper_end,
        vp_peak,
    )

    lower = np.full(
        n,
        -np.inf,
        dtype=float,
    )

    upper = np.full(
        n,
        np.inf,
        dtype=float,
    )

    snapshots = []

    for resolution in resolutions:

        resolution = int(
            resolution
        )

        fractions = np.linspace(
            0.0,
            1.0,
            resolution,
        )

        sample_max = np.full(
            n,
            -np.inf,
            dtype=float,
        )

        for start in range(
            0,
            n,
            chunk_size,
        ):

            stop = min(
                start + chunk_size,
                n,
            )

            sl = slice(
                start,
                stop,
            )

            t = (
                T_sw
                *
                fractions[
                    None,
                    :
                ]
            )

            sf_upper = (
                vf[sl, None]
                * t
                +
                0.5
                * a_switch_upper
                * t**2
            )

            sp = fb.position_with_stop(
                vp[sl, None],
                ap[sl, None],
                pars["up"],
                pars["tp"],
                pars["wp"],
                t,
                tp_stop[sl, None],
            )

            sample_max[sl] = (
                np.max(
                    sf_upper
                    -
                    sp,
                    axis=1,
                )
            )

        lower_now = np.maximum(
            sample_max,
            0.0,
        )

        dt = (
            T_sw
            /
            (
                resolution
                -
                1
            )
        )

        upper_now = np.maximum(
            sample_max
            +
            0.5
            * lipschitz
            * dt,
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
                resolution,
                lower.copy(),
                upper.copy(),
            )
        )

    return snapshots


def switch_end_conservative_state(
    vf: np.ndarray,
    vp: np.ndarray,
    ap: np.ndarray,
    T_sw: float,
    a_switch_upper: float,
    p2b_cfg: dict,
):

    if T_sw <= 0.0:

        return (
            np.zeros_like(vf),
            vf.copy(),
            None,
            np.zeros_like(vp),
            vp.copy(),
            ap.copy(),
        )

    pars = fb.model_parameters(
        p2b_cfg
    )

    sf_upper = (
        vf * T_sw
        +
        0.5
        * a_switch_upper
        * T_sw**2
    )

    vf_upper = (
        vf
        +
        a_switch_upper
        * T_sw
    )

    sp, vp_next, ap_next = (
        fb.hybrid_state_after(
            vp,
            ap,
            pars["up"],
            pars["tp"],
            pars["wp"],
            T_sw,
        )
    )

    return (
        sf_upper,
        vf_upper,
        np.full_like(
            vf,
            a_switch_upper,
        ),
        sp,
        vp_next,
        ap_next,
    )


def switching_loss_bracket(
    vf: np.ndarray,
    af: np.ndarray,
    vp: np.ndarray,
    ap: np.ndarray,
    N_sw: int,
    resolutions: list[int],
    cfg: dict,
    p2b_cfg: dict,
):

    Ts = float(
        p2b_cfg["sampling"][
            "Ts"
        ]
    )

    a_switch_upper = float(
        cfg["switching"][
            "follower_acceleration_upper"
        ]
    )

    if N_sw == 0:

        lower, upper, snapshots = (
            fb.relative_loss_bracket(
                vf,
                af,
                vp,
                ap,
                resolutions,
                p2b_cfg,
            )
        )

        return (
            lower,
            upper,
            snapshots,
        )

    T_sw = (
        N_sw
        *
        Ts
    )

    switch_snaps = (
        switch_interval_snapshots(
            vf,
            vp,
            ap,
            T_sw,
            resolutions,
            a_switch_upper,
            p2b_cfg,
        )
    )

    (
        sf_end,
        vf_end,
        af_end,
        sp_end,
        vp_end,
        ap_end,
    ) = switch_end_conservative_state(
        vf,
        vp,
        ap,
        T_sw,
        a_switch_upper,
        p2b_cfg,
    )

    relative_shift = (
        sf_end
        -
        sp_end
    )

    (
        _post_lower,
        _post_upper,
        post_snaps,
    ) = fb.relative_loss_bracket(
        vf_end,
        af_end,
        vp_end,
        ap_end,
        resolutions,
        p2b_cfg,
    )

    total_snaps = []

    for (
        switch_item,
        post_item,
    ) in zip(
        switch_snaps,
        post_snaps,
    ):

        r1, sw_lb, sw_ub = (
            switch_item
        )

        r2, post_lb, post_ub = (
            post_item
        )

        if r1 != r2:
            raise RuntimeError(
                "resolution mismatch"
            )

        total_lb = np.maximum(
            np.maximum(
                sw_lb,
                relative_shift
                +
                post_lb,
            ),
            0.0,
        )

        total_ub = np.maximum(
            np.maximum(
                sw_ub,
                relative_shift
                +
                post_ub,
            ),
            0.0,
        )

        total_snaps.append(
            (
                r1,
                total_lb,
                total_ub,
            )
        )

    return (
        total_snaps[-1][1],
        total_snaps[-1][2],
        total_snaps,
    )


def evaluate_guard(
    states: np.ndarray,
    N_sw: int,
    resolutions: list[int],
    cfg: dict,
    p2b_cfg: dict,
):

    physical_mask, vp_all = (
        fb.physical_subset(
            states,
            p2b_cfg,
        )
    )

    X = states[
        physical_mask
    ]

    vp = vp_all[
        physical_mask
    ]

    d = X[:, 0]
    vf = X[:, 2]
    af = X[:, 3]
    ap = X[:, 4]

    lower, upper, snapshots = (
        switching_loss_bracket(
            vf,
            af,
            vp,
            ap,
            N_sw,
            resolutions,
            cfg,
            p2b_cfg,
        )
    )

    d_min = float(
        p2b_cfg[
            "state_domain"
        ][
            "d_min"
        ]
    )

    certified = (
        d
        >=
        d_min
        +
        upper
    )

    candidate = (
        d
        >=
        d_min
        +
        lower
    )

    convergence = []

    for (
        resolution,
        lower_n,
        upper_n,
    ) in snapshots:

        certified_n = (
            d
            >=
            d_min
            +
            upper_n
        )

        candidate_n = (
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

                "certified_fraction":
                    float(
                        np.mean(
                            certified_n
                        )
                    ),

                "candidate_fraction":
                    float(
                        np.mean(
                            candidate_n
                        )
                    ),

                "ambiguity_fraction":
                    float(
                        np.mean(
                            candidate_n
                            &
                            ~certified_n
                        )
                    ),

                "width_mean_m":
                    float(
                        np.mean(
                            width
                        )
                    ),

                "width_p95_m":
                    float(
                        np.quantile(
                            width,
                            0.95,
                        )
                    ),

                "width_max_m":
                    float(
                        np.max(
                            width
                        )
                    ),
            }
        )

    return {
        "physical_mask":
            physical_mask,

        "physical_states":
            X,

        "vp":
            vp,

        "loss_lower":
            lower,

        "loss_upper":
            upper,

        "certified":
            certified,

        "candidate":
            candidate,

        "convergence":
            convergence,
    }


def zero_switch_reduction_audit(
    states,
    resolutions,
    cfg,
    p2b_cfg,
):

    guard0 = evaluate_guard(
        states,
        0,
        resolutions,
        cfg,
        p2b_cfg,
    )

    baseline = (
        fb.evaluate_kernel_envelope(
            states,
            p2b_cfg,
            resolutions,
        )
    )

    if not np.array_equal(
        guard0[
            "physical_mask"
        ],
        baseline[
            "mask"
        ],
    ):
        raise RuntimeError(
            "physical masks differ"
        )

    lower_error = float(
        np.max(
            np.abs(
                guard0[
                    "loss_lower"
                ]
                -
                baseline[
                    "loss_lower"
                ]
            )
        )
    )

    upper_error = float(
        np.max(
            np.abs(
                guard0[
                    "loss_upper"
                ]
                -
                baseline[
                    "loss_upper"
                ]
            )
        )
    )

    classification_difference = int(
        np.count_nonzero(
            guard0[
                "certified"
            ]
            !=
            baseline[
                "inner"
            ]
        )
    )

    return {
        "lower_error":
            lower_error,

        "upper_error":
            upper_error,

        "classification_difference":
            classification_difference,
    }


def immediate_fallback_subset_audit(
    states,
    guard,
    resolutions,
    p2b_cfg,
):

    baseline = (
        fb.evaluate_kernel_envelope(
            states,
            p2b_cfg,
            resolutions,
        )
    )

    if not np.array_equal(
        guard[
            "physical_mask"
        ],
        baseline[
            "mask"
        ],
    ):
        raise RuntimeError(
            "physical masks differ"
        )

    violations = int(
        np.count_nonzero(
            guard[
                "certified"
            ]
            &
            ~baseline[
                "inner"
            ]
        )
    )

    fallback_fraction = float(
        np.mean(
            baseline[
                "inner"
            ]
        )
    )

    guard_fraction = float(
        np.mean(
            guard[
                "certified"
            ]
        )
    )

    penalty_abs = (
        fallback_fraction
        -
        guard_fraction
    )

    penalty_rel = (
        penalty_abs
        /
        fallback_fraction
        if fallback_fraction > 0.0
        else math.nan
    )

    return {
        "subset_violations":
            violations,

        "fallback_fraction":
            fallback_fraction,

        "guard_fraction":
            guard_fraction,

        "switching_penalty_absolute":
            penalty_abs,

        "switching_penalty_relative":
            penalty_rel,
    }


def sensitivity_audit(
    cfg,
    p2b_cfg,
):

    power = int(
        cfg["qmc"][
            "sensitivity_power"
        ]
    )

    resolution = int(
        cfg["qmc"][
            "sensitivity_resolution"
        ]
    )

    states, _, _ = (
        fb.sample_states(
            power,
            int(
                cfg[
                    "random_seed"
                ]
            )
            +
            500,
            p2b_cfg,
        )
    )

    rows = []

    max_steps = int(
        cfg["switching"][
            "sensitivity_max_steps"
        ]
    )

    for N_sw in range(
        max_steps + 1
    ):

        result = evaluate_guard(
            states,
            N_sw,
            [resolution],
            cfg,
            p2b_cfg,
        )

        rows.append(
            {
                "N_sw":
                    N_sw,

                "certified_fraction":
                    float(
                        np.mean(
                            result[
                                "certified"
                            ]
                        )
                    ),

                "candidate_fraction":
                    float(
                        np.mean(
                            result[
                                "candidate"
                            ]
                        )
                    ),

                "ambiguity_fraction":
                    float(
                        np.mean(
                            result[
                                "candidate"
                            ]
                            &
                            ~result[
                                "certified"
                            ]
                        )
                    ),
            }
        )

    return rows


def replicate_audit(
    cfg,
    p2b_cfg,
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

    N_sw = int(
        cfg["switching"][
            "N_sw"
        ]
    )

    base_seed = int(
        cfg["random_seed"]
    )

    certified = []
    candidate = []

    for j in range(
        reps
    ):

        states, _, _ = (
            fb.sample_states(
                power,
                base_seed
                +
                1000
                +
                j,
                p2b_cfg,
            )
        )

        result = evaluate_guard(
            states,
            N_sw,
            [257, 513],
            cfg,
            p2b_cfg,
        )

        certified.append(
            float(
                np.mean(
                    result[
                        "certified"
                    ]
                )
            )
        )

        candidate.append(
            float(
                np.mean(
                    result[
                        "candidate"
                    ]
                )
            )
        )

    return {
        "certified_mean":
            float(
                np.mean(
                    certified
                )
            ),

        "certified_std":
            float(
                np.std(
                    certified,
                    ddof=1,
                )
                if reps > 1
                else 0.0
            ),

        "candidate_mean":
            float(
                np.mean(
                    candidate
                )
            ),

        "candidate_std":
            float(
                np.std(
                    candidate,
                    ddof=1,
                )
                if reps > 1
                else 0.0
            ),
    }


def boundary_replay(
    guard,
    cfg,
    p2b_cfg,
):

    ambiguous = np.flatnonzero(
        guard[
            "candidate"
        ]
        &
        ~guard[
            "certified"
        ]
    )

    if len(ambiguous) == 0:

        return {
            "states_replayed":
                0,

            "resolved_certified_fraction":
                0.0,

            "residual_ambiguity_fraction":
                0.0,
        }

    max_states = int(
        cfg[
            "boundary_replay"
        ][
            "max_states"
        ]
    )

    if len(ambiguous) > max_states:

        choice = np.linspace(
            0,
            len(ambiguous) - 1,
            max_states,
        ).astype(int)

        ambiguous = (
            ambiguous[
                choice
            ]
        )

    X = guard[
        "physical_states"
    ][
        ambiguous
    ]

    resolution = int(
        cfg[
            "boundary_replay"
        ][
            "resolution"
        ]
    )

    replay = evaluate_guard(
        X,
        int(
            cfg[
                "switching"
            ][
                "N_sw"
            ]
        ),
        [resolution],
        cfg,
        p2b_cfg,
    )

    certified_fraction = float(
        np.mean(
            replay[
                "certified"
            ]
        )
    )

    ambiguity_fraction = float(
        np.mean(
            replay[
                "candidate"
            ]
            &
            ~replay[
                "certified"
            ]
        )
    )

    return {
        "states_replayed":
            int(
                len(X)
            ),

        "resolved_certified_fraction":
            certified_fraction,

        "residual_ambiguity_fraction":
            ambiguity_fraction,
    }


def write_csv(
    path: Path,
    rows: list[dict],
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
        writer.writerows(
            rows
        )


def make_convergence_figure(
    path,
    rows,
):

    resolution = [
        r["resolution"]
        for r in rows
    ]

    certified = [
        r["certified_fraction"]
        for r in rows
    ]

    candidate = [
        r["candidate_fraction"]
        for r in rows
    ]

    ambiguity = [
        r["ambiguity_fraction"]
        for r in rows
    ]

    fig, ax = plt.subplots(
        figsize=(7.3, 4.8)
    )

    ax.plot(
        resolution,
        certified,
        marker="o",
        label="certified guard",
    )

    ax.plot(
        resolution,
        candidate,
        marker="s",
        label="candidate guard",
    )

    ax.plot(
        resolution,
        ambiguity,
        marker="^",
        label="temporal ambiguity",
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


def make_sensitivity_figure(
    path,
    rows,
):

    x = [
        r["N_sw"]
        for r in rows
    ]

    y = [
        r["certified_fraction"]
        for r in rows
    ]

    fig, ax = plt.subplots(
        figsize=(7.0, 4.6)
    )

    ax.plot(
        x,
        y,
        marker="o",
    )

    ax.set_xlabel(
        "Fallback switching delay N_sw"
    )

    ax.set_ylabel(
        "Certified switching-guard fraction"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

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

    contract = (
        switching_contract_check(
            cfg,
            p1_cfg,
            p2b_cfg,
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
        fb.sample_states(
            int(
                cfg["qmc"][
                    "main_power"
                ]
            ),
            int(
                cfg[
                    "random_seed"
                ]
            )
            +
            100,
            p2b_cfg,
        )
    )

    N_sw = int(
        cfg["switching"][
            "N_sw"
        ]
    )

    guard = evaluate_guard(
        states,
        N_sw,
        resolutions,
        cfg,
        p2b_cfg,
    )

    zero = (
        zero_switch_reduction_audit(
            states,
            resolutions,
            cfg,
            p2b_cfg,
        )
    )

    inclusion = (
        immediate_fallback_subset_audit(
            states,
            guard,
            resolutions,
            p2b_cfg,
        )
    )

    sensitivity = (
        sensitivity_audit(
            cfg,
            p2b_cfg,
        )
    )

    replicate = (
        replicate_audit(
            cfg,
            p2b_cfg,
        )
    )

    replay = boundary_replay(
        guard,
        cfg,
        p2b_cfg,
    )

    convergence = (
        guard[
            "convergence"
        ]
    )

    ambiguity = float(
        np.mean(
            guard[
                "candidate"
            ]
            &
            ~guard[
                "certified"
            ]
        )
    )

    width = (
        guard[
            "loss_upper"
        ]
        -
        guard[
            "loss_lower"
        ]
    )

    certified_fraction = float(
        np.mean(
            guard[
                "certified"
            ]
        )
    )

    candidate_fraction = float(
        np.mean(
            guard[
                "candidate"
            ]
        )
    )

    sensitivity_values = np.asarray(
        [
            row[
                "certified_fraction"
            ]
            for row
            in sensitivity
        ],
        dtype=float,
    )

    monotone = bool(
        np.all(
            np.diff(
                sensitivity_values
            )
            <=
            1e-12
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

    checks = {

        "SWITCH_CONTRACT_VALID":
            contract[
                "valid"
            ],

        "ZERO_SWITCH_REDUCES_TO_P2B":
            (
                zero[
                    "lower_error"
                ]
                <=
                1e-12
                and
                zero[
                    "upper_error"
                ]
                <=
                1e-12
                and
                zero[
                    "classification_difference"
                ]
                ==
                0
            ),

        "TEMPORAL_BRACKET_ORDERED":
            bool(
                np.all(
                    guard[
                        "loss_lower"
                    ]
                    <=
                    guard[
                        "loss_upper"
                    ]
                    +
                    1e-12
                )
            ),

        "CERTIFIED_SUBSET_CANDIDATE":
            bool(
                np.all(
                    ~guard[
                        "certified"
                    ]
                    |
                    guard[
                        "candidate"
                    ]
                )
            ),

        "CERTIFIED_GUARD_NONEMPTY":
            (
                certified_fraction
                >
                0.0
            ),

        "SWITCH_GUARD_SUBSET_IMMEDIATE_FALLBACK":
            (
                inclusion[
                    "subset_violations"
                ]
                ==
                0
            ),

        "SWITCH_DELAY_MONOTONE":
            monotone,

        "TEMPORAL_AMBIGUITY_SMALL":
            (
                ambiguity
                <=
                ambiguity_tol
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
                p95_tol
            ),

        "QMC_REPLICATE_STABLE":
            (
                replicate[
                    "certified_std"
                ]
                <=
                replicate_tol
                and
                replicate[
                    "candidate_std"
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
            guard[
                "physical_mask"
            ]
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

        "certified_guard_fraction_physical":
            certified_fraction,

        "candidate_guard_fraction_physical":
            candidate_fraction,

        "temporal_ambiguity_fraction_physical":
            ambiguity,

        "certified_guard_volume":
            physical_volume
            *
            certified_fraction,

        "candidate_guard_volume":
            physical_volume
            *
            candidate_fraction,

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

        **{
            "switch_contract_" + k:
                v
            for k, v
            in contract.items()
            if isinstance(
                v,
                (int, float, bool),
            )
        },

        **{
            "zero_switch_" + k:
                v
            for k, v
            in zero.items()
        },

        **{
            "immediate_" + k:
                v
            for k, v
            in inclusion.items()
        },

        **{
            "qmc_" + k:
                v
            for k, v
            in replicate.items()
        },

        **{
            "boundary_" + k:
                v
            for k, v
            in replay.items()
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
        f"P2C_TEMPORAL_CONVERGENCE_{stamp}.csv"
    )

    sensitivity_csv = (
        RESULTS_DIR
        /
        f"P2C_SWITCHING_SENSITIVITY_{stamp}.csv"
    )

    convergence_fig = (
        FIGURES_DIR
        /
        f"P2C_TEMPORAL_CONVERGENCE_{stamp}.png"
    )

    sensitivity_fig = (
        FIGURES_DIR
        /
        f"P2C_SWITCHING_SENSITIVITY_{stamp}.png"
    )

    write_csv(
        convergence_csv,
        convergence,
    )

    write_csv(
        sensitivity_csv,
        sensitivity,
    )

    make_convergence_figure(
        convergence_fig,
        convergence,
    )

    make_sensitivity_figure(
        sensitivity_fig,
        sensitivity,
    )

    output = {

        "schema":
            "SCV_P2C_SWITCHING_GUARD_RESULT_V1",

        "status":
            status,

        "timestamp_utc":
            stamp,

        "classification":
            (
                "certified conservative finite-switch "
                "fallback guard under explicit "
                "switching acceleration contract"
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

        "switching_sensitivity":
            sensitivity,

        "boundary_replay":
            replay,

        "artifacts": {
            "convergence_csv":
                str(
                    convergence_csv
                ),

            "sensitivity_csv":
                str(
                    sensitivity_csv
                ),

            "convergence_figure":
                str(
                    convergence_fig
                ),

            "sensitivity_figure":
                str(
                    sensitivity_fig
                ),
        },
    }

    result_path = (
        RESULTS_DIR
        /
        f"P2C_SWITCHING_GUARD_{stamp}.json"
    )

    latest_path = (
        RESULTS_DIR
        /
        "P2C_LATEST.json"
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
        f"P2C_MANIFEST_{stamp}.sha256"
    )

    files = [
        CFG_PATH,
        P1_CFG_PATH,
        P2B_CFG_PATH,
        Path(__file__),
        result_path,
        convergence_csv,
        sensitivity_csv,
        convergence_fig,
        sensitivity_fig,
    ]

    manifest = "\n".join(
        (
            f"{sha256_file(path)}"
            f"  {path}"
        )
        for path in files
    ) + "\n"

    atomic_write_text(
        manifest_path,
        manifest,
    )

    print(
        "=== SCV P2-C FINITE-SWITCH GUARD ==="
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
        "=== TEMPORAL CONVERGENCE ==="
    )

    for row in convergence:

        print(
            "RESOLUTION="
            f"{row['resolution']} "
            "CERTIFIED="
            f"{row['certified_fraction']:.9f} "
            "CANDIDATE="
            f"{row['candidate_fraction']:.9f} "
            "AMBIGUITY="
            f"{row['ambiguity_fraction']:.9f} "
            "P95_WIDTH_M="
            f"{row['width_p95_m']:.9f}"
        )

    print(
        "=== SWITCHING SENSITIVITY ==="
    )

    for row in sensitivity:

        print(
            "NSW="
            f"{row['N_sw']} "
            "CERTIFIED="
            f"{row['certified_fraction']:.9f} "
            "CANDIDATE="
            f"{row['candidate_fraction']:.9f}"
        )

    print(
        f"P2C_SWITCHING_GUARD={status}"
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
