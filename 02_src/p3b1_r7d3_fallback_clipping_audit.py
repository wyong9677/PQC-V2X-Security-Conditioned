from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

import p3b1_augmented_fixed_point_v1_r3 as r3
import p3b1_r7_frozen_halo_fixed_point as r7
import p3b1_r7d1_node_level_audit as d1

P1_CFG_PATH = ROOT / "01_config" / "p1_validation_v2.json"
P2B_CFG_PATH = ROOT / "01_config" / "p2b_hybrid_fallback_v1.json"
P2C_CFG_PATH = ROOT / "01_config" / "p2c_switching_guard_v1.json"
P3A_CFG_PATH = ROOT / "01_config" / "p3a_information_contract_v1.json"
R6_PATH = ROOT / "04_results" / "P3B1_R6_LATEST.json"

RESULTS_DIR = ROOT / "04_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TOL = 1.0e-10


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


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def build_transition_data(data):
    p1_cfg = load_json(P1_CFG_PATH)
    p2b_cfg = load_json(P2B_CFG_PATH)
    p2c_cfg = load_json(P2C_CFG_PATH)
    p3a_cfg = load_json(P3A_CFG_PATH)

    return r7.build_transition_data_to_lookup(
        data["eval_cfg"],
        p1_cfg,
        p2b_cfg,
        p2c_cfg,
        p3a_cfg,
        data["eval_flat"],
        data["lookup_axes"],
    )


def raw_operator_slices(
    result: dict,
    lookup_shape,
    transition_data: dict,
    eval_fallback: np.ndarray,
):
    """
    Evaluate the converged Bellman operator on every q slice twice:

      raw_best(y,q):
          cooperative predecessor requirement before fallback clipping.

      clipped(y,q):
          min(h_F(y), raw_best(y,q)).

    At a true fixed point, clipped should reproduce the stored evaluation
    slice up to tolerance.
    """
    R = int(result["R"])
    n = len(eval_fallback)

    h = np.asarray(
        result["h_flat"],
        dtype=float,
    ).reshape(
        lookup_shape + (R,)
    )

    cell_max = r3.cell_corner_max(h)
    cell_flat = [
        cell_max[..., q].reshape(-1)
        for q in range(R)
    ]

    raw = np.full(
        (n, R),
        np.inf,
        dtype=float,
    )

    branch_codes = np.full(
        (n, R),
        -1,
        dtype=np.int8,
    )
    # 0 = completion-dominant, 1 = defer-dominant, 2 = equal/no-defer

    best_action = np.full(
        (n, R),
        -1,
        dtype=np.int16,
    )

    for r in range(1, R + 1):
        q = r - 1

        required_actions = []
        branch_actions = []

        for trans in transition_data["transitions"]:
            completion = np.full(
                n,
                np.inf,
                dtype=float,
            )

            cmask = trans["completion_valid"]
            completion[cmask] = (
                cell_flat[R - 1][
                    trans["completion_index"][cmask]
                ]
            )

            if r > 1:
                defer = np.full(
                    n,
                    np.inf,
                    dtype=float,
                )

                dmask = trans["defer_valid"]
                defer[dmask] = (
                    cell_flat[r - 2][
                        trans["defer_index"][dmask]
                    ]
                )

                future = np.maximum(
                    completion,
                    defer,
                )

                branch = np.where(
                    defer > completion + TOL,
                    1,
                    np.where(
                        completion > defer + TOL,
                        0,
                        2,
                    ),
                )
            else:
                future = completion
                branch = np.full(
                    n,
                    2,
                    dtype=np.int8,
                )

            required = np.maximum(
                trans["step_loss_upper"],
                trans["closing_end"] + future,
            )

            required_actions.append(required)
            branch_actions.append(branch)

        req_matrix = np.vstack(required_actions)
        action = np.argmin(req_matrix, axis=0)
        node = np.arange(n)

        raw[:, q] = np.maximum(
            req_matrix[action, node],
            0.0,
        )
        best_action[:, q] = action
        branch_matrix = np.vstack(branch_actions)
        branch_codes[:, q] = branch_matrix[action, node]

    clipped = np.minimum(
        eval_fallback[:, None],
        raw,
    )

    return raw, clipped, branch_codes, best_action


def q_dependence_stats(matrix: np.ndarray) -> dict:
    if matrix.shape[1] <= 1:
        return {
            "nodes": 0,
            "fraction": 0.0,
            "mean_positive_range_m": 0.0,
            "p95_positive_range_m": 0.0,
            "max_range_m": 0.0,
        }

    span = np.max(matrix, axis=1) - np.min(matrix, axis=1)
    mask = span > TOL
    positive = span[mask]

    return {
        "nodes": int(np.count_nonzero(mask)),
        "fraction": float(np.mean(mask)),
        "mean_positive_range_m":
            float(np.mean(positive)) if len(positive) else 0.0,
        "p95_positive_range_m":
            float(np.quantile(positive, 0.95))
            if len(positive) else 0.0,
        "max_range_m":
            float(np.max(span)),
    }


def pair_difference(a: np.ndarray, b: np.ndarray) -> dict:
    delta = np.abs(a - b)
    mask = delta > TOL
    positive = delta[mask]

    return {
        "different_nodes":
            int(np.count_nonzero(mask)),
        "fraction":
            float(np.mean(mask)),
        "mean_positive_m":
            float(np.mean(positive))
            if len(positive) else 0.0,
        "p95_positive_m":
            float(np.quantile(positive, 0.95))
            if len(positive) else 0.0,
        "max_abs_m":
            float(np.max(delta)),
    }


def main():
    data = d1.reconstruct()
    transition_data = build_transition_data(data)

    eval_idx = data["eval_node_indices"]
    eval_fallback = data["eval_fallback"]
    lookup_shape = data["lookup_shape"]
    results = data["results"]

    print("=== P3-B1-R7-D3 FALLBACK-CLIPPING ATTRIBUTION ===")
    print(
        "EVAL_GRID_NODES="
        f"{len(eval_fallback)}"
    )

    profile_rows = []
    raw_initial = {}
    clipped_initial = {}

    for name, result in results.items():
        raw, clipped, branch, action = raw_operator_slices(
            result,
            lookup_shape,
            transition_data,
            eval_fallback,
        )

        stored = np.asarray(
            result["h_flat"][
                eval_idx,
                :
            ],
            dtype=float,
        )

        fixed_point_error = float(
            np.max(
                np.abs(
                    stored
                    -
                    clipped
                )
            )
        )

        raw_q = q_dependence_stats(raw)
        clipped_q = q_dependence_stats(clipped)

        raw_span = (
            np.max(raw, axis=1)
            -
            np.min(raw, axis=1)
        ) if raw.shape[1] > 1 else np.zeros(len(raw))

        clipped_span = (
            np.max(clipped, axis=1)
            -
            np.min(clipped, axis=1)
        ) if clipped.shape[1] > 1 else np.zeros(len(clipped))

        eliminated = (
            (raw_span > TOL)
            &
            (clipped_span <= TOL)
        )

        gain = (
            eval_fallback
            -
            clipped[:, -1]
            >
            TOL
        )

        raw_q_dep = raw_span > TOL

        gain_raw_q_dep = int(
            np.count_nonzero(
                gain
                &
                raw_q_dep
            )
        )

        defer_dominant_initial = (
            branch[:, -1] == 1
        )

        gain_defer_initial = int(
            np.count_nonzero(
                gain
                &
                defer_dominant_initial
            )
        )

        row = {
            "profile": name,
            "R": int(result["R"]),
            "fixed_point_replay_max_error_m":
                fixed_point_error,
            "raw_q_dependence_nodes":
                raw_q["nodes"],
            "raw_q_dependence_fraction":
                raw_q["fraction"],
            "raw_q_range_mean_positive_m":
                raw_q["mean_positive_range_m"],
            "raw_q_range_p95_positive_m":
                raw_q["p95_positive_range_m"],
            "raw_q_range_max_m":
                raw_q["max_range_m"],
            "clipped_q_dependence_nodes":
                clipped_q["nodes"],
            "clipped_q_dependence_fraction":
                clipped_q["fraction"],
            "clipped_q_range_max_m":
                clipped_q["max_range_m"],
            "q_dependence_eliminated_by_fallback_nodes":
                int(np.count_nonzero(eliminated)),
            "final_gain_nodes":
                int(np.count_nonzero(gain)),
            "gain_nodes_with_raw_q_dependence":
                gain_raw_q_dep,
            "gain_nodes_defer_dominant_initial_slice":
                gain_defer_initial,
            "initial_slice_defer_dominant_nodes":
                int(np.count_nonzero(defer_dominant_initial)),
        }

        profile_rows.append(row)

        raw_initial[name] = raw[:, -1]
        clipped_initial[name] = clipped[:, -1]

        print(
            f"{name.upper()} "
            "RAW_Q_DEP="
            f"{row['raw_q_dependence_nodes']} "
            "CLIPPED_Q_DEP="
            f"{row['clipped_q_dependence_nodes']} "
            "ELIMINATED_BY_FALLBACK="
            f"{row['q_dependence_eliminated_by_fallback_nodes']} "
            "GAIN_NODES="
            f"{row['final_gain_nodes']} "
            "GAIN_RAW_Q_DEP="
            f"{row['gain_nodes_with_raw_q_dependence']} "
            "GAIN_DEFER_DOM="
            f"{row['gain_nodes_defer_dominant_initial_slice']} "
            "FP_REPLAY_ERR="
            f"{fixed_point_error:.12g}"
        )

    pairs = [
        ("ideal", "diagnostic_fast"),
        ("diagnostic_fast", "diagnostic_nominal"),
        ("diagnostic_nominal", "diagnostic_stressed"),
        ("ideal", "diagnostic_stressed"),
    ]

    pair_rows = []

    print("=== PROFILE DIFFERENCES BEFORE/AFTER FALLBACK CLIP ===")

    for left, right in pairs:
        raw_stats = pair_difference(
            raw_initial[left],
            raw_initial[right],
        )
        clipped_stats = pair_difference(
            clipped_initial[left],
            clipped_initial[right],
        )

        raw_delta = np.abs(
            raw_initial[left]
            -
            raw_initial[right]
        )

        clipped_delta = np.abs(
            clipped_initial[left]
            -
            clipped_initial[right]
        )

        eliminated = (
            (raw_delta > TOL)
            &
            (clipped_delta <= TOL)
        )

        row = {
            "left": left,
            "right": right,
            "raw_different_nodes":
                raw_stats["different_nodes"],
            "raw_difference_fraction":
                raw_stats["fraction"],
            "raw_difference_p95_positive_m":
                raw_stats["p95_positive_m"],
            "raw_difference_max_m":
                raw_stats["max_abs_m"],
            "clipped_different_nodes":
                clipped_stats["different_nodes"],
            "clipped_difference_fraction":
                clipped_stats["fraction"],
            "clipped_difference_max_m":
                clipped_stats["max_abs_m"],
            "profile_difference_eliminated_by_fallback_nodes":
                int(np.count_nonzero(eliminated)),
        }

        pair_rows.append(row)

        print(
            f"{left.upper()}_VS_{right.upper()} "
            "RAW_DIFF="
            f"{row['raw_different_nodes']} "
            "CLIPPED_DIFF="
            f"{row['clipped_different_nodes']} "
            "ELIMINATED="
            f"{row['profile_difference_eliminated_by_fallback_nodes']} "
            "RAW_MAX_M="
            f"{row['raw_difference_max_m']:.12g}"
        )

    max_replay_error = max(
        row["fixed_point_replay_max_error_m"]
        for row in profile_rows
    )

    raw_q_total = sum(
        row["raw_q_dependence_nodes"]
        for row in profile_rows
        if row["R"] > 1
    )

    clipped_q_total = sum(
        row["clipped_q_dependence_nodes"]
        for row in profile_rows
        if row["R"] > 1
    )

    eliminated_q_total = sum(
        row["q_dependence_eliminated_by_fallback_nodes"]
        for row in profile_rows
        if row["R"] > 1
    )

    raw_profile_diff_total = sum(
        row["raw_different_nodes"]
        for row in pair_rows
    )

    clipped_profile_diff_total = sum(
        row["clipped_different_nodes"]
        for row in pair_rows
    )

    gain_q_dep_total = sum(
        row["gain_nodes_with_raw_q_dependence"]
        for row in profile_rows
        if row["R"] > 1
    )

    if raw_q_total == 0:
        interpretation = (
            "SERVICE_HORIZON_HAS_NO_EFFECT_EVEN_BEFORE_FALLBACK_CLIPPING"
        )
        next_step = (
            "REDESIGN_Q_TRANSITION_AUTOMATON"
        )
    elif clipped_q_total == 0 and eliminated_q_total > 0:
        if gain_q_dep_total == 0:
            interpretation = (
                "LATENT_Q_EFFECT_EXISTS_BUT_FALLBACK_CLIPPING_REMOVES_IT_OUTSIDE_GAIN_REGION"
            )
            next_step = (
                "COUPLE_Q_TO_SUPERVISORY_FALLBACK_SWITCH_SEMANTICS_BEFORE_AUTOMATON_EXPANSION"
            )
        else:
            interpretation = (
                "LATENT_Q_EFFECT_OVERLAPS_GAIN_REGION_BUT_IS_CLIPPED_BY_FALLBACK"
            )
            next_step = (
                "REVIEW_FALLBACK_AVAILABILITY_AND_SWITCH_GUARD_BY_Q_STATE"
            )
    elif raw_profile_diff_total > 0 and clipped_profile_diff_total == 0:
        interpretation = (
            "PROFILE_DIFFERENCES_EXIST_PRECLIP_BUT_ARE_REMOVED_BY_FALLBACK"
        )
        next_step = (
            "COUPLE_SERVICE_STATE_TO_MODE_AND_FALLBACK_TRANSITION"
        )
    else:
        interpretation = (
            "SERVICE_EFFECT_SURVIVES_PRECLIP_AND_REQUIRES_DEEPER_FIXED_POINT_REVIEW"
        )
        next_step = (
            "BUILD_EXPLICIT_Q_AUTOMATON_WITH_PRESERVED_SERVICE_EFFECT"
        )

    checks = {
        "FIXED_POINT_OPERATOR_REPLAY":
            max_replay_error <= 1.0e-9,
        "PROFILE_AUDIT_COMPLETE":
            len(profile_rows) == 4,
        "PAIR_AUDIT_COMPLETE":
            len(pair_rows) == 4,
        "FINAL_CLIPPED_Q_COLLAPSE_REPRODUCED":
            clipped_q_total == 0,
        "FINAL_CLIPPED_PROFILE_COLLAPSE_REPRODUCED":
            clipped_profile_diff_total == 0,
    }

    status = (
        "PASS"
        if all(checks.values())
        else "FAIL"
    )

    stamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    profile_csv = (
        RESULTS_DIR
        /
        f"P3B1_R7D3_CLIPPING_PROFILE_AUDIT_{stamp}.csv"
    )

    pair_csv = (
        RESULTS_DIR
        /
        f"P3B1_R7D3_CLIPPING_PAIR_AUDIT_{stamp}.csv"
    )

    write_csv(profile_csv, profile_rows)
    write_csv(pair_csv, pair_rows)

    output = {
        "schema":
            "SCV_P3B1_R7D3_FALLBACK_CLIPPING_ATTRIBUTION_V1",
        "status":
            status,
        "timestamp_utc":
            stamp,
        "classification":
            (
                "diagnostic decomposition of cooperative raw Bellman "
                "requirements versus fallback clipping; not a kernel result"
            ),
        "checks":
            {k: bool(v) for k, v in checks.items()},
        "interpretation":
            interpretation,
        "recommended_next_step":
            next_step,
        "profile_stats":
            profile_rows,
        "profile_pair_stats":
            pair_rows,
        "aggregate": {
            "raw_q_dependence_nodes_sum":
                int(raw_q_total),
            "clipped_q_dependence_nodes_sum":
                int(clipped_q_total),
            "q_dependence_eliminated_by_fallback_nodes_sum":
                int(eliminated_q_total),
            "raw_profile_difference_nodes_sum":
                int(raw_profile_diff_total),
            "clipped_profile_difference_nodes_sum":
                int(clipped_profile_diff_total),
            "gain_nodes_with_raw_q_dependence_sum":
                int(gain_q_dep_total),
            "max_fixed_point_replay_error_m":
                float(max_replay_error),
        },
        "scientific_kernel_claim_authorized":
            False,
    }

    result_path = (
        RESULTS_DIR
        /
        f"P3B1_R7D3_FALLBACK_CLIPPING_AUDIT_{stamp}.json"
    )
    latest_path = (
        RESULTS_DIR
        /
        "P3B1_R7D3_LATEST.json"
    )

    text = json.dumps(
        output,
        indent=2,
        sort_keys=True,
    )

    atomic_write(result_path, text)
    atomic_write(latest_path, text)

    manifest_path = (
        RESULTS_DIR
        /
        f"P3B1_R7D3_MANIFEST_{stamp}.sha256"
    )

    manifest_files = [
        P1_CFG_PATH,
        P2B_CFG_PATH,
        P2C_CFG_PATH,
        P3A_CFG_PATH,
        R6_PATH,
        Path(__file__),
        result_path,
        profile_csv,
        pair_csv,
    ]

    manifest = "\n".join(
        f"{sha256_file(path)}  {path}"
        for path in manifest_files
    ) + "\n"

    atomic_write(
        manifest_path,
        manifest,
    )

    print("=== P3-B1-R7-D3 DECISION ===")
    print(
        "RAW_Q_DEPENDENCE_NODES_SUM="
        f"{raw_q_total}"
    )
    print(
        "CLIPPED_Q_DEPENDENCE_NODES_SUM="
        f"{clipped_q_total}"
    )
    print(
        "Q_DEP_ELIMINATED_BY_FALLBACK_SUM="
        f"{eliminated_q_total}"
    )
    print(
        "RAW_PROFILE_DIFF_NODES_SUM="
        f"{raw_profile_diff_total}"
    )
    print(
        "CLIPPED_PROFILE_DIFF_NODES_SUM="
        f"{clipped_profile_diff_total}"
    )
    print(
        "GAIN_RAW_Q_DEPENDENCE_SUM="
        f"{gain_q_dep_total}"
    )
    print(
        "INTERPRETATION="
        + interpretation
    )
    print(
        "RECOMMENDED_NEXT_STEP="
        + next_step
    )
    print(
        f"P3B1_R7D3_FALLBACK_CLIPPING_AUDIT={status}"
    )
    print(
        "SCIENTIFIC_KERNEL_CLAIM=NO"
    )
    print(
        f"RESULT_JSON={result_path}"
    )
    print(
        f"MANIFEST={manifest_path}"
    )

    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
