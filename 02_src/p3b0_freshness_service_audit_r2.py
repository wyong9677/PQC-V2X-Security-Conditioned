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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent

sys.path.insert(
    0,
    str(SRC),
)

import p3b0_freshness_service_audit_v1 as base


CFG_PATH = (
    ROOT
    / "01_config"
    / "p3b0_freshness_service_audit_r2.json"
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

BASE_SOURCE_PATH = (
    ROOT
    / "02_src"
    / "p3b0_freshness_service_audit_v1.py"
)

P2B_SOURCE_PATH = (
    ROOT
    / "02_src"
    / "p2b_hybrid_fallback_v1.py"
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
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def make_base_compatible_cfg(
    cfg: dict,
) -> dict:
    """
    Convert the compact R2 configuration into the
    fields expected by the already audited P3-B0 V1
    numerical routines.
    """

    return {
        "random_seed":
            int(
                cfg["random_seed"]
            ),

        "cooperative_control": {
            "candidate_protective_command":
                float(
                    cfg[
                        "cooperative_control"
                    ][
                        "candidate_protective_command"
                    ]
                )
        },

        "information_contract": {
            "candidate_slew_rate":
                float(
                    cfg[
                        "information_contract"
                    ][
                        "candidate_slew_rate"
                    ]
                )
        },

        "freshness":
            copy.deepcopy(
                cfg["freshness"]
            ),

        "qmc":
            copy.deepcopy(
                cfg["qmc"]
            ),

        "temporal_bracket":
            copy.deepcopy(
                cfg[
                    "temporal_bracket"
                ]
            ),

        "witness":
            copy.deepcopy(
                cfg["witness"]
            ),
    }


def counterexample_indices(
    certified: np.ndarray,
) -> np.ndarray:

    false_to_true = (
        ~certified[:, :-1]
        &
        certified[:, 1:]
    )

    return np.flatnonzero(
        np.any(
            false_to_true,
            axis=1,
        )
    )


def q_witness_indices(
    service: dict,
) -> tuple[
    np.ndarray,
    int,
    int,
]:

    names = service[
        "names"
    ]

    fast_index = (
        names.index(
            "diagnostic_fast"
        )
    )

    stressed_index = (
        names.index(
            "diagnostic_stressed"
        )
    )

    mask = (
        service[
            "certified"
        ][:, fast_index]
        &
        ~service[
            "certified"
        ][:, stressed_index]
    )

    return (
        np.flatnonzero(mask),
        fast_index,
        stressed_index,
    )


def replay_age_counterexamples(
    X,
    vp,
    fiber,
    cfg,
    base_cfg,
    p2b_cfg,
    p3a_cfg,
):
    indices = counterexample_indices(
        fiber[
            "certified"
        ]
    )

    if len(indices) == 0:

        return {
            "original_count":
                0,

            "replayed_count":
                0,

            "persistent_count":
                0,

            "resolved_as_numerical_count":
                0,

            "persistent_indices":
                [],
        }

    X_sub = X[
        indices
    ]

    vp_sub = vp[
        indices
    ]

    replay_cfg = copy.deepcopy(
        base_cfg
    )

    replay_cfg[
        "temporal_bracket"
    ][
        "resolutions"
    ] = [
        int(x)
        for x
        in cfg[
            "boundary_replay"
        ][
            "resolutions"
        ]
    ]

    replay = (
        base.freshness_fiber_audit(
            X_sub,
            vp_sub,
            replay_cfg,
            p2b_cfg,
            p3a_cfg,
        )
    )

    persistent_local = (
        counterexample_indices(
            replay[
                "certified"
            ]
        )
    )

    persistent_global = (
        indices[
            persistent_local
        ]
        if len(
            persistent_local
        )
        >
        0
        else
        np.asarray(
            [],
            dtype=int,
        )
    )

    return {
        "original_count":
            int(
                len(indices)
            ),

        "replayed_count":
            int(
                len(indices)
            ),

        "persistent_count":
            int(
                len(
                    persistent_global
                )
            ),

        "resolved_as_numerical_count":
            int(
                len(indices)
                -
                len(
                    persistent_global
                )
            ),

        "persistent_indices":
            [
                int(x)
                for x
                in persistent_global
            ],
    }


def replay_q_witnesses(
    X,
    vp,
    service,
    cfg,
    base_cfg,
    p1_cfg,
    p2b_cfg,
    p3a_cfg,
):
    (
        indices,
        _fast_index,
        _stressed_index,
    ) = q_witness_indices(
        service
    )

    if len(indices) == 0:

        return {
            "original_count":
                0,

            "replayed_count":
                0,

            "persistent_count":
                0,

            "resolved_as_numerical_count":
                0,

            "persistent_indices":
                [],
        }

    X_sub = X[
        indices
    ]

    vp_sub = vp[
        indices
    ]

    replay_cfg = copy.deepcopy(
        base_cfg
    )

    replay_cfg[
        "temporal_bracket"
    ][
        "resolutions"
    ] = [
        int(x)
        for x
        in cfg[
            "boundary_replay"
        ][
            "resolutions"
        ]
    ]

    replay_service = (
        base.service_state_audit(
            X_sub,
            vp_sub,
            replay_cfg,
            p1_cfg,
            p2b_cfg,
            p3a_cfg,
        )
    )

    replay_indices, _, _ = (
        q_witness_indices(
            replay_service
        )
    )

    persistent_global = (
        indices[
            replay_indices
        ]
        if len(
            replay_indices
        )
        >
        0
        else
        np.asarray(
            [],
            dtype=int,
        )
    )

    return {
        "original_count":
            int(
                len(indices)
            ),

        "replayed_count":
            int(
                len(indices)
            ),

        "persistent_count":
            int(
                len(
                    persistent_global
                )
            ),

        "resolved_as_numerical_count":
            int(
                len(indices)
                -
                len(
                    persistent_global
                )
            ),

        "persistent_indices":
            [
                int(x)
                for x
                in persistent_global
            ],
    }


def service_replication_audit(
    cfg,
    base_cfg,
    p1_cfg,
    p2b_cfg,
    p3a_cfg,
):
    rep_cfg = cfg[
        "service_replication"
    ]

    replicates = int(
        rep_cfg[
            "replicates"
        ]
    )

    power = int(
        rep_cfg[
            "power"
        ]
    )

    resolutions = [
        int(x)
        for x
        in rep_cfg[
            "resolutions"
        ]
    ]

    rows = []

    base_seed = int(
        cfg["random_seed"]
    )

    positive_count = 0

    for rep in range(
        replicates
    ):

        seed = (
            base_seed
            +
            2000
            +
            rep
        )

        local_cfg = copy.deepcopy(
            base_cfg
        )

        local_cfg[
            "random_seed"
        ] = seed

        local_cfg[
            "qmc"
        ][
            "main_power"
        ] = power

        local_cfg[
            "temporal_bracket"
        ][
            "resolutions"
        ] = resolutions

        X, vp, physical_fraction = (
            base.sample_augmented_states(
                power,
                seed,
                local_cfg,
                p2b_cfg,
                p3a_cfg,
            )
        )

        service = (
            base.service_state_audit(
                X,
                vp,
                local_cfg,
                p1_cfg,
                p2b_cfg,
                p3a_cfg,
            )
        )

        names = service[
            "names"
        ]

        ideal = float(
            service[
                "fractions"
            ][
                "ideal"
            ]
        )

        stressed = float(
            service[
                "fractions"
            ][
                "diagnostic_stressed"
            ]
        )

        witness_indices, _, _ = (
            q_witness_indices(
                service
            )
        )

        difference = (
            ideal
            -
            stressed
        )

        if (
            difference
            >
            0.0
            and
            len(
                witness_indices
            )
            >
            0
        ):
            positive_count += 1

        row = {
            "replicate":
                rep,

            "seed":
                seed,

            "sample_count":
                int(
                    len(X)
                ),

            "physical_fraction":
                float(
                    physical_fraction
                ),

            "ideal_fraction":
                ideal,

            "fast_fraction":
                float(
                    service[
                        "fractions"
                    ][
                        "diagnostic_fast"
                    ]
                ),

            "nominal_fraction":
                float(
                    service[
                        "fractions"
                    ][
                        "diagnostic_nominal"
                    ]
                ),

            "stressed_fraction":
                stressed,

            "ideal_minus_stressed":
                difference,

            "fast_safe_stressed_unsafe_count":
                int(
                    len(
                        witness_indices
                    )
                ),
        }

        rows.append(
            row
        )

    differences = np.asarray(
        [
            row[
                "ideal_minus_stressed"
            ]
            for row
            in rows
        ],
        dtype=float,
    )

    witness_counts = np.asarray(
        [
            row[
                "fast_safe_stressed_unsafe_count"
            ]
            for row
            in rows
        ],
        dtype=float,
    )

    return {
        "rows":
            rows,

        "positive_replicates":
            int(
                positive_count
            ),

        "difference_mean":
            float(
                np.mean(
                    differences
                )
            ),

        "difference_std":
            float(
                np.std(
                    differences,
                    ddof=1,
                )
                if len(
                    differences
                )
                >
                1
                else 0.0
            ),

        "witness_count_mean":
            float(
                np.mean(
                    witness_counts
                )
            ),

        "witness_count_total":
            int(
                np.sum(
                    witness_counts
                )
            ),
    }


def persistent_witness_rows(
    X,
    vp,
    indices,
    service,
):
    names = service[
        "names"
    ]

    fast_index = names.index(
        "diagnostic_fast"
    )

    stressed_index = names.index(
        "diagnostic_stressed"
    )

    rows = []

    for index in indices:

        i = int(index)

        rows.append(
            {
                "sample_index":
                    i,

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
            }
        )

    return rows


def persistent_age_rows(
    X,
    vp,
    indices,
):
    rows = []

    for index in indices:

        i = int(index)

        rows.append(
            {
                "sample_index":
                    i,

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
            }
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


def make_service_replication_figure(
    path,
    rows,
):
    x = np.arange(
        len(rows)
    )

    ideal = np.asarray(
        [
            r["ideal_fraction"]
            for r in rows
        ]
    )

    fast = np.asarray(
        [
            r["fast_fraction"]
            for r in rows
        ]
    )

    nominal = np.asarray(
        [
            r["nominal_fraction"]
            for r in rows
        ]
    )

    stressed = np.asarray(
        [
            r["stressed_fraction"]
            for r in rows
        ]
    )

    fig, ax = plt.subplots(
        figsize=(8.0, 4.8)
    )

    ax.plot(
        x,
        ideal,
        marker="o",
        label="ideal",
    )

    ax.plot(
        x,
        fast,
        marker="s",
        label="fast",
    )

    ax.plot(
        x,
        nominal,
        marker="^",
        label="nominal",
    )

    ax.plot(
        x,
        stressed,
        marker="d",
        label="stressed",
    )

    ax.set_xlabel(
        "Independent Sobol replicate"
    )

    ax.set_ylabel(
        "Certified diagnostic fraction"
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

    base_cfg = (
        make_base_compatible_cfg(
            cfg
        )
    )

    power = int(
        cfg["qmc"][
            "main_power"
        ]
    )

    seed = int(
        cfg[
            "random_seed"
        ]
    )

    X, vp, physical_fraction = (
        base.sample_augmented_states(
            power,
            seed,
            base_cfg,
            p2b_cfg,
            p3a_cfg,
        )
    )

    fiber = (
        base.freshness_fiber_audit(
            X,
            vp,
            base_cfg,
            p2b_cfg,
            p3a_cfg,
        )
    )

    service = (
        base.service_state_audit(
            X,
            vp,
            base_cfg,
            p1_cfg,
            p2b_cfg,
            p3a_cfg,
        )
    )

    age_replay = (
        replay_age_counterexamples(
            X,
            vp,
            fiber,
            cfg,
            base_cfg,
            p2b_cfg,
            p3a_cfg,
        )
    )

    q_replay = (
        replay_q_witnesses(
            X,
            vp,
            service,
            cfg,
            base_cfg,
            p1_cfg,
            p2b_cfg,
            p3a_cfg,
        )
    )

    replication = (
        service_replication_audit(
            cfg,
            base_cfg,
            p1_cfg,
            p2b_cfg,
            p3a_cfg,
        )
    )

    width = fiber[
        "width"
    ]

    ambiguity_fraction = float(
        np.mean(
            fiber[
                "ambiguity"
            ]
        )
    )

    p95_width = float(
        np.quantile(
            width,
            0.95,
        )
    )

    service_order_violations = int(
        sum(
            service[
                "inclusion_violations"
            ].values()
        )
    )

    min_positive = int(
        cfg[
            "service_replication"
        ][
            "minimum_positive_replicates"
        ]
    )

    checks = {
        "AUGMENTED_SAMPLE_NONEMPTY":
            (
                len(X) > 0
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
                p95_width
                <=
                float(
                    cfg[
                        "temporal_bracket"
                    ][
                        "p95_width_tolerance_m"
                    ]
                )
            ),

        "SERVICE_ORDER_NO_NUMERICAL_REVERSAL":
            (
                service_order_violations
                ==
                0
            ),

        "SERVICE_WITNESS_REPLAY_CONFIRMED":
            (
                q_replay[
                    "original_count"
                ]
                >
                0
                and
                q_replay[
                    "persistent_count"
                ]
                >
                0
            ),

        "SERVICE_EFFECT_REPLICATED":
            (
                replication[
                    "positive_replicates"
                ]
                >=
                min_positive
            ),

        "AGE_COUNTEREXAMPLE_REPLAY_COMPLETE":
            (
                age_replay[
                    "replayed_count"
                ]
                ==
                age_replay[
                    "original_count"
                ]
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

    empirical_downward_fraction = float(
        np.mean(
            fiber[
                "downward_closed"
            ]
        )
    )

    metrics = {
        "physical_fraction_of_qmc_box":
            float(
                physical_fraction
            ),

        "augmented_sample_count":
            int(
                len(X)
            ),

        "temporal_ambiguity_fraction":
            ambiguity_fraction,

        "temporal_gap_width_mean_m":
            float(
                np.mean(
                    width
                )
            ),

        "temporal_gap_width_p95_m":
            p95_width,

        "temporal_gap_width_max_m":
            float(
                np.max(
                    width
                )
            ),

        "empirical_downward_closed_fraction":
            empirical_downward_fraction,

        "age_counterexample_original_count":
            int(
                age_replay[
                    "original_count"
                ]
            ),

        "age_counterexample_persistent_count":
            int(
                age_replay[
                    "persistent_count"
                ]
            ),

        "age_counterexample_numerically_resolved_count":
            int(
                age_replay[
                    "resolved_as_numerical_count"
                ]
            ),

        "q_witness_original_count":
            int(
                q_replay[
                    "original_count"
                ]
            ),

        "q_witness_persistent_count":
            int(
                q_replay[
                    "persistent_count"
                ]
            ),

        "q_witness_numerically_resolved_count":
            int(
                q_replay[
                    "resolved_as_numerical_count"
                ]
            ),

        "service_order_violation_count":
            service_order_violations,

        "service_positive_replicates":
            int(
                replication[
                    "positive_replicates"
                ]
            ),

        "service_difference_mean":
            float(
                replication[
                    "difference_mean"
                ]
            ),

        "service_difference_std":
            float(
                replication[
                    "difference_std"
                ]
            ),

        "service_witness_count_mean":
            float(
                replication[
                    "witness_count_mean"
                ]
            ),

        "service_witness_count_total":
            int(
                replication[
                    "witness_count_total"
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

    q_rows = (
        persistent_witness_rows(
            X,
            vp,
            q_replay[
                "persistent_indices"
            ],
            service,
        )
    )

    age_rows = (
        persistent_age_rows(
            X,
            vp,
            age_replay[
                "persistent_indices"
            ],
        )
    )

    q_csv = (
        RESULTS_DIR
        /
        f"P3B0_R2_Q_WITNESS_REPLAY_{stamp}.csv"
    )

    age_csv = (
        RESULTS_DIR
        /
        f"P3B0_R2_AGE_COUNTEREXAMPLE_REPLAY_{stamp}.csv"
    )

    replicate_csv = (
        RESULTS_DIR
        /
        f"P3B0_R2_SERVICE_REPLICATES_{stamp}.csv"
    )

    replicate_fig = (
        FIGURES_DIR
        /
        f"P3B0_R2_SERVICE_REPLICATES_{stamp}.png"
    )

    write_csv(
        q_csv,
        q_rows,
    )

    write_csv(
        age_csv,
        age_rows,
    )

    write_csv(
        replicate_csv,
        replication[
            "rows"
        ],
    )

    make_service_replication_figure(
        replicate_fig,
        replication[
            "rows"
        ],
    )

    interpretation = {
        "age_monotonicity_proved":
            False,

        "persistent_age_counterexamples":
            int(
                age_replay[
                    "persistent_count"
                ]
            ),

        "age_interpretation":
            (
                "Persistent replay counterexamples, if nonzero, "
                "are numerical evidence against scalar-threshold "
                "reduction for this diagnostic model; zero "
                "counterexamples remain numerical evidence only "
                "and do not replace the analytical age-simulation "
                "condition."
            ),

        "service_effect_observed":
            bool(
                q_replay[
                    "persistent_count"
                ]
                >
                0
            ),

        "service_profile_final":
            False,

        "maximal_kernel_claim":
            False,
    }

    output = {
        "schema":
            "SCV_P3B0_FRESHNESS_SERVICE_AUDIT_R2_RESULT",

        "status":
            status,

        "timestamp_utc":
            stamp,

        "classification":
            (
                "high-resolution diagnostic replay; "
                "not the maximal augmented viability kernel "
                "and not a final PQC service experiment"
            ),

        "checks":
            {
                key: bool(value)
                for key, value
                in checks.items()
            },

        "metrics":
            metrics,

        "service_fractions":
            service[
                "fractions"
            ],

        "service_inclusion_violations":
            service[
                "inclusion_violations"
            ],

        "age_replay":
            age_replay,

        "q_witness_replay":
            q_replay,

        "service_replication":
            replication,

        "interpretation":
            interpretation,

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

        "artifacts": {
            "persistent_q_witness_csv":
                str(
                    q_csv
                ),

            "persistent_age_counterexample_csv":
                str(
                    age_csv
                ),

            "service_replicate_csv":
                str(
                    replicate_csv
                ),

            "service_replicate_figure":
                str(
                    replicate_fig
                ),
        },
    }

    result_path = (
        RESULTS_DIR
        /
        f"P3B0_R2_AUDIT_{stamp}.json"
    )

    latest_path = (
        RESULTS_DIR
        /
        "P3B0_R2_LATEST.json"
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
        f"P3B0_R2_MANIFEST_{stamp}.sha256"
    )

    manifest_files = [
        CFG_PATH,
        P1_CFG_PATH,
        P2B_CFG_PATH,
        P3A_CFG_PATH,
        BASE_SOURCE_PATH,
        P2B_SOURCE_PATH,
        Path(__file__),
        result_path,
        q_csv,
        age_csv,
        replicate_csv,
        replicate_fig,
    ]

    manifest = "\n".join(
        (
            f"{sha256_file(path)}"
            f"  {path}"
        )
        for path in manifest_files
    ) + "\n"

    atomic_write(
        manifest_path,
        manifest,
    )

    print(
        "=== SCV P3-B0-R2 HIGH-RESOLUTION AUDIT ==="
    )

    for key, value in checks.items():

        print(
            f"{key}="
            f"{'PASS' if value else 'FAIL'}"
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
        "=== AGE COUNTEREXAMPLE REPLAY ==="
    )

    print(
        "ORIGINAL="
        f"{age_replay['original_count']} "
        "PERSISTENT="
        f"{age_replay['persistent_count']} "
        "NUMERICALLY_RESOLVED="
        f"{age_replay['resolved_as_numerical_count']}"
    )

    print(
        "=== Q WITNESS REPLAY ==="
    )

    print(
        "ORIGINAL="
        f"{q_replay['original_count']} "
        "PERSISTENT="
        f"{q_replay['persistent_count']} "
        "NUMERICALLY_RESOLVED="
        f"{q_replay['resolved_as_numerical_count']}"
    )

    print(
        "=== SERVICE REPLICATION ==="
    )

    for row in replication[
        "rows"
    ]:

        print(
            "REP="
            f"{row['replicate']} "
            "IDEAL="
            f"{row['ideal_fraction']:.8f} "
            "FAST="
            f"{row['fast_fraction']:.8f} "
            "NOMINAL="
            f"{row['nominal_fraction']:.8f} "
            "STRESSED="
            f"{row['stressed_fraction']:.8f} "
            "DIFF="
            f"{row['ideal_minus_stressed']:.8f} "
            "FAST_ONLY="
            f"{row['fast_safe_stressed_unsafe_count']}"
        )

    print(
        f"P3B0_R2_AUDIT={status}"
    )

    print(
        "MAXIMAL_KERNEL_CLAIM=NO"
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
