from __future__ import annotations

import csv
import hashlib
import json
import math
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


def q_slice_stats(result: dict, eval_idx: np.ndarray) -> dict:
    h = np.asarray(
        result["h_flat"][eval_idx, :],
        dtype=float,
    )

    R = h.shape[1]

    if R == 1:
        return {
            "R": 1,
            "nodes_with_q_dependence": 0,
            "q_dependence_fraction": 0.0,
            "q_range_mean_m": 0.0,
            "q_range_p95_positive_m": 0.0,
            "q_range_max_m": 0.0,
            "adjacent_difference_nodes_total": 0,
            "adjacent_difference_max_m": 0.0,
        }

    q_range = np.max(h, axis=1) - np.min(h, axis=1)
    positive = q_range[q_range > TOL]

    adjacent = np.abs(np.diff(h, axis=1))

    return {
        "R": int(R),
        "nodes_with_q_dependence":
            int(np.count_nonzero(q_range > TOL)),
        "q_dependence_fraction":
            float(np.mean(q_range > TOL)),
        "q_range_mean_m":
            float(np.mean(q_range)),
        "q_range_p95_positive_m":
            float(np.quantile(positive, 0.95))
            if len(positive) else 0.0,
        "q_range_max_m":
            float(np.max(q_range)),
        "adjacent_difference_nodes_total":
            int(np.count_nonzero(adjacent > TOL)),
        "adjacent_difference_max_m":
            float(np.max(adjacent)),
    }


def final_eval_slice(result: dict, eval_idx: np.ndarray) -> np.ndarray:
    return np.asarray(
        result["h_flat"][
            eval_idx,
            result["R"] - 1,
        ],
        dtype=float,
    )


def branch_audit_for_profile(
    name: str,
    result: dict,
    eval_idx: np.ndarray,
    lookup_shape,
    transition_data: dict,
    eval_fallback: np.ndarray,
) -> dict:
    R = int(result["R"])

    if R <= 1:
        return {
            "profile": name,
            "R": R,
            "gain_nodes": int(
                np.count_nonzero(
                    eval_fallback
                    -
                    final_eval_slice(result, eval_idx)
                    >
                    TOL
                )
            ),
            "defer_branch_available": False,
            "best_action_defer_dominant_nodes": 0,
            "best_action_completion_dominant_nodes": 0,
            "best_action_equal_branch_nodes": 0,
            "gain_nodes_defer_dominant": 0,
            "gain_nodes_completion_dominant": 0,
            "gain_nodes_equal_branch": 0,
            "gain_nodes_with_q_dependence": 0,
            "best_action_index_counts": {},
        }

    h = result["h_flat"].reshape(
        lookup_shape + (R,)
    )

    cell_max = r3.cell_corner_max(h)

    cell_flat = [
        cell_max[..., q].reshape(-1)
        for q in range(R)
    ]

    # Audit the initial worst-horizon slice r=R.
    completion_all = []
    defer_all = []
    required_all = []

    for trans in transition_data["transitions"]:
        completion = np.full(
            len(eval_fallback),
            np.inf,
            dtype=float,
        )
        defer = np.full(
            len(eval_fallback),
            np.inf,
            dtype=float,
        )

        cmask = trans["completion_valid"]
        dmask = trans["defer_valid"]

        completion[cmask] = (
            cell_flat[R - 1][
                trans["completion_index"][cmask]
            ]
        )

        defer[dmask] = (
            cell_flat[R - 2][
                trans["defer_index"][dmask]
            ]
        )

        future = np.maximum(
            completion,
            defer,
        )

        required = np.maximum(
            trans["step_loss_upper"],
            trans["closing_end"] + future,
        )

        completion_all.append(completion)
        defer_all.append(defer)
        required_all.append(required)

    required_matrix = np.vstack(required_all)
    best_action = np.argmin(
        required_matrix,
        axis=0,
    )

    node = np.arange(
        len(eval_fallback)
    )

    completion_best = np.vstack(
        completion_all
    )[best_action, node]

    defer_best = np.vstack(
        defer_all
    )[best_action, node]

    defer_dom = (
        defer_best
        >
        completion_best
        +
        TOL
    )

    completion_dom = (
        completion_best
        >
        defer_best
        +
        TOL
    )

    equal = ~(
        defer_dom
        |
        completion_dom
    )

    final_req = final_eval_slice(
        result,
        eval_idx,
    )

    gain = (
        eval_fallback
        -
        final_req
        >
        TOL
    )

    h_eval = np.asarray(
        result["h_flat"][eval_idx, :],
        dtype=float,
    )

    q_range = (
        np.max(h_eval, axis=1)
        -
        np.min(h_eval, axis=1)
    )

    action_counts = {
        str(i):
            int(
                np.count_nonzero(
                    best_action == i
                )
            )
        for i in range(
            required_matrix.shape[0]
        )
    }

    return {
        "profile": name,
        "R": R,
        "gain_nodes":
            int(np.count_nonzero(gain)),
        "defer_branch_available":
            True,
        "best_action_defer_dominant_nodes":
            int(np.count_nonzero(defer_dom)),
        "best_action_completion_dominant_nodes":
            int(np.count_nonzero(completion_dom)),
        "best_action_equal_branch_nodes":
            int(np.count_nonzero(equal)),
        "gain_nodes_defer_dominant":
            int(np.count_nonzero(gain & defer_dom)),
        "gain_nodes_completion_dominant":
            int(np.count_nonzero(gain & completion_dom)),
        "gain_nodes_equal_branch":
            int(np.count_nonzero(gain & equal)),
        "gain_nodes_with_q_dependence":
            int(
                np.count_nonzero(
                    gain
                    &
                    (
                        q_range
                        >
                        TOL
                    )
                )
            ),
        "best_action_index_counts":
            action_counts,
    }


def main():
    data = d1.reconstruct()

    eval_cfg = data["eval_cfg"]
    eval_flat = data["eval_flat"]
    eval_idx = data["eval_node_indices"]
    lookup_axes = data["lookup_axes"]
    lookup_shape = data["lookup_shape"]
    eval_fallback = data["eval_fallback"]
    results = data["results"]

    p1_cfg = load_json(P1_CFG_PATH)
    p2b_cfg = load_json(P2B_CFG_PATH)
    p2c_cfg = load_json(P2C_CFG_PATH)
    p3a_cfg = load_json(P3A_CFG_PATH)

    transition_data = r7.build_transition_data_to_lookup(
        eval_cfg,
        p1_cfg,
        p2b_cfg,
        p2c_cfg,
        p3a_cfg,
        eval_flat,
        lookup_axes,
    )

    completion_invalid = sum(
        int(
            np.count_nonzero(
                ~trans["completion_valid"]
            )
        )
        for trans in transition_data["transitions"]
    )

    defer_invalid = sum(
        int(
            np.count_nonzero(
                ~trans["defer_valid"]
            )
        )
        for trans in transition_data["transitions"]
    )

    if completion_invalid != 0 or defer_invalid != 0:
        raise RuntimeError(
            "P3B1_R7D2_HALO_COVERAGE_FAIL"
        )

    print(
        "=== P3-B1-R7-D2 SERVICE RECURSION AUDIT ==="
    )

    q_rows = []
    branch_rows = []

    for name, result in results.items():
        qstats = q_slice_stats(
            result,
            eval_idx,
        )

        qrow = {
            "profile": name,
            **qstats,
        }

        q_rows.append(qrow)

        print(
            f"{name.upper()} "
            f"R={qstats['R']} "
            "Q_DEP_NODES="
            f"{qstats['nodes_with_q_dependence']} "
            "Q_DEP_FRAC="
            f"{qstats['q_dependence_fraction']:.12g} "
            "Q_RANGE_MAX_M="
            f"{qstats['q_range_max_m']:.12g}"
        )

        brow = branch_audit_for_profile(
            name,
            result,
            eval_idx,
            lookup_shape,
            transition_data,
            eval_fallback,
        )

        branch_rows.append(brow)

        print(
            f"{name.upper()} "
            "GAIN_NODES="
            f"{brow['gain_nodes']} "
            "DEFER_DOM_ALL="
            f"{brow['best_action_defer_dominant_nodes']} "
            "COMPLETE_DOM_ALL="
            f"{brow['best_action_completion_dominant_nodes']} "
            "EQUAL_ALL="
            f"{brow['best_action_equal_branch_nodes']} "
            "GAIN_DEFER_DOM="
            f"{brow['gain_nodes_defer_dominant']} "
            "GAIN_COMPLETE_DOM="
            f"{brow['gain_nodes_completion_dominant']} "
            "GAIN_Q_DEP="
            f"{brow['gain_nodes_with_q_dependence']}"
        )

    initial_slices = {
        name:
            final_eval_slice(
                result,
                eval_idx,
            )
        for name, result in results.items()
    }

    profile_pairs = [
        ("ideal", "diagnostic_fast"),
        ("diagnostic_fast", "diagnostic_nominal"),
        ("diagnostic_nominal", "diagnostic_stressed"),
        ("ideal", "diagnostic_stressed"),
    ]

    pair_rows = []

    print(
        "=== FINAL PROFILE SLICE DIFFERENCES ==="
    )

    for left, right in profile_pairs:
        delta = (
            initial_slices[right]
            -
            initial_slices[left]
        )

        abs_delta = np.abs(delta)

        row = {
            "left": left,
            "right": right,
            "different_nodes":
                int(
                    np.count_nonzero(
                        abs_delta
                        >
                        TOL
                    )
                ),
            "max_abs_difference_m":
                float(np.max(abs_delta)),
            "mean_abs_difference_m":
                float(np.mean(abs_delta)),
        }

        pair_rows.append(row)

        print(
            f"{left.upper()}_VS_{right.upper()} "
            "DIFF_NODES="
            f"{row['different_nodes']} "
            "MAX_ABS_M="
            f"{row['max_abs_difference_m']:.12g}"
        )

    q_dependent_profiles = [
        row
        for row in q_rows
        if row["R"] > 1
        and
        row["nodes_with_q_dependence"] > 0
    ]

    stressed_branch = next(
        row
        for row in branch_rows
        if row["profile"]
        ==
        "diagnostic_stressed"
    )

    final_profile_difference_nodes = max(
        row["different_nodes"]
        for row in pair_rows
    )

    if not q_dependent_profiles:
        interpretation = (
            "Q_SLICES_COLLAPSE_WITHIN_EACH_PROFILE"
        )
    elif (
        stressed_branch[
            "gain_nodes_with_q_dependence"
        ]
        ==
        0
    ):
        interpretation = (
            "Q_DEPENDENCE_EXISTS_OUTSIDE_COOPERATIVE_GAIN_REGION"
        )
    elif (
        stressed_branch[
            "gain_nodes_defer_dominant"
        ]
        ==
        0
    ):
        interpretation = (
            "DEFER_BRANCH_NEVER_BINDS_ON_COOPERATIVE_GAIN_REGION"
        )
    elif final_profile_difference_nodes == 0:
        interpretation = (
            "INTERNAL_Q_DEPENDENCE_EXISTS_BUT_PROFILE_RENEWAL_COLLAPSES_INITIAL_SLICES"
        )
    else:
        interpretation = (
            "SERVICE_DIFFERENCE_EXISTS_BELOW_PREVIOUS_TOLERANCE_OR_AT_NONINITIAL_Q"
        )

    checks = {
        "HALO_COVERAGE":
            completion_invalid == 0
            and defer_invalid == 0,
        "PROFILE_AUDIT_COMPLETE":
            len(q_rows) == 4
            and len(branch_rows) == 4,
        "FINAL_PROFILE_COLLAPSE_REPRODUCED":
            final_profile_difference_nodes == 0,
        "COOPERATIVE_GAIN_REPRODUCED":
            all(
                row["gain_nodes"] == 42
                for row in branch_rows
            ),
    }

    status = (
        "PASS"
        if all(checks.values())
        else "FAIL"
    )

    stamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    q_csv = (
        RESULTS_DIR
        /
        f"P3B1_R7D2_Q_SLICE_AUDIT_{stamp}.csv"
    )

    branch_csv = (
        RESULTS_DIR
        /
        f"P3B1_R7D2_BRANCH_AUDIT_{stamp}.csv"
    )

    pair_csv = (
        RESULTS_DIR
        /
        f"P3B1_R7D2_PROFILE_DIFF_{stamp}.csv"
    )

    # Flatten the one dict-valued field for CSV.
    branch_csv_rows = []

    for row in branch_rows:
        flat_row = {
            k: v
            for k, v in row.items()
            if k != "best_action_index_counts"
        }

        for key, value in (
            row[
                "best_action_index_counts"
            ].items()
        ):
            flat_row[
                f"best_action_index_{key}_count"
            ] = value

        branch_csv_rows.append(
            flat_row
        )

    write_csv(
        q_csv,
        q_rows,
    )

    write_csv(
        branch_csv,
        branch_csv_rows,
    )

    write_csv(
        pair_csv,
        pair_rows,
    )

    output = {
        "schema":
            "SCV_P3B1_R7D2_SERVICE_RECURSION_AUDIT_V1",
        "status":
            status,
        "timestamp_utc":
            stamp,
        "classification":
            (
                "service-recursion diagnostic; "
                "no kernel certification claim"
            ),
        "checks":
            {
                k: bool(v)
                for k, v
                in checks.items()
            },
        "interpretation":
            interpretation,
        "q_slice_stats":
            q_rows,
        "branch_stats":
            branch_rows,
        "profile_pair_differences":
            pair_rows,
        "recommended_next_step":
            (
                "REPLACE_DIAGNOSTIC_PROFILE_HORIZON_WITH_EXPLICIT_Q_AUTOMATON"
                if final_profile_difference_nodes == 0
                else
                "REVIEW_SERVICE_DIFFERENCE_GEOMETRY_BEFORE_Q_REDESIGN"
            ),
        "scientific_kernel_claim_authorized":
            False,
    }

    result_path = (
        RESULTS_DIR
        /
        f"P3B1_R7D2_SERVICE_RECURSION_AUDIT_{stamp}.json"
    )

    latest_path = (
        RESULTS_DIR
        /
        "P3B1_R7D2_LATEST.json"
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
        f"P3B1_R7D2_MANIFEST_{stamp}.sha256"
    )

    manifest_files = [
        P1_CFG_PATH,
        P2B_CFG_PATH,
        P2C_CFG_PATH,
        P3A_CFG_PATH,
        R6_PATH,
        Path(__file__),
        result_path,
        q_csv,
        branch_csv,
        pair_csv,
    ]

    manifest = "\n".join(
        f"{sha256_file(path)}  {path}"
        for path
        in manifest_files
    ) + "\n"

    atomic_write(
        manifest_path,
        manifest,
    )

    print(
        "=== P3-B1-R7-D2 DECISION ==="
    )

    print(
        "INTERPRETATION="
        + interpretation
    )

    print(
        "RECOMMENDED_NEXT_STEP="
        + output[
            "recommended_next_step"
        ]
    )

    print(
        f"P3B1_R7D2_SERVICE_RECURSION_AUDIT={status}"
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
