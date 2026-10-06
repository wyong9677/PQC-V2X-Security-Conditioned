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

import p3b1_r7d1_node_level_audit as d1
import p3b1_r7d3_fallback_clipping_audit as d3
import p3b1_r7d4_supervisory_switch_semantics as d4

P1_CFG_PATH = ROOT / "01_config" / "p1_validation_v2.json"
P2B_CFG_PATH = ROOT / "01_config" / "p2b_hybrid_fallback_v1.json"
P2C_CFG_PATH = ROOT / "01_config" / "p2c_switching_guard_v1.json"
P3A_CFG_PATH = ROOT / "01_config" / "p3a_information_contract_v1.json"
D5_CFG_PATH = ROOT / "01_config" / "p3b1_r7d5_monitor_protocol_v1.json"

D2_PATH = ROOT / "04_results" / "P3B1_R7D2_LATEST.json"
D3_PATH = ROOT / "04_results" / "P3B1_R7D3_LATEST.json"
D4_PATH = ROOT / "04_results" / "P3B1_R7D4_LATEST.json"

RESULTS_DIR = ROOT / "04_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def positive_summary(x: np.ndarray, tol: float) -> dict:
    x = np.asarray(x, dtype=float)
    vals = x[x > tol]
    if len(vals) == 0:
        return {
            "count": 0,
            "min": 0.0,
            "p05": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "max": 0.0,
            "mean": 0.0,
        }
    return {
        "count": int(len(vals)),
        "min": float(np.min(vals)),
        "p05": float(np.quantile(vals, 0.05)),
        "p50": float(np.quantile(vals, 0.50)),
        "p95": float(np.quantile(vals, 0.95)),
        "max": float(np.max(vals)),
        "mean": float(np.mean(vals)),
    }


def any_q_label_change(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[1] <= 1:
        return np.zeros(matrix.shape[0], dtype=bool)
    return np.any(matrix != matrix[:, [0]], axis=1)


def neighbor_support(mask: np.ndarray, shape: tuple[int, ...]):
    """
    One-hop von Neumann support on the actual 6-D evaluation grid.

    For every node, count:
      degree  = number of existing +/- one-step axis neighbors
      support = number of those neighbors that are also switch witnesses
    """
    grid = np.asarray(mask, dtype=bool).reshape(shape)
    support = np.zeros(shape, dtype=np.int16)
    degree = np.zeros(shape, dtype=np.int16)

    for axis, size in enumerate(shape):
        if size <= 1:
            continue

        left = [slice(None)] * len(shape)
        right = [slice(None)] * len(shape)
        left[axis] = slice(0, size - 1)
        right[axis] = slice(1, size)

        lt = tuple(left)
        rt = tuple(right)

        degree[lt] += 1
        degree[rt] += 1

        support[lt] += grid[rt].astype(np.int16)
        support[rt] += grid[lt].astype(np.int16)

    support_flat = support.reshape(-1)
    degree_flat = degree.reshape(-1)
    fraction = np.zeros_like(support_flat, dtype=float)
    valid = degree_flat > 0
    fraction[valid] = support_flat[valid] / degree_flat[valid]

    return support_flat, degree_flat, fraction


def main() -> int:
    protocol = load_json(D5_CFG_PATH)
    tol = float(protocol["numeric_tolerance_m"])
    replay_tol = float(protocol["replay_tolerance_m"])
    radius_multiple_gate = float(
        protocol["robust_radius_min_multiple_of_numeric_tolerance"]
    )
    perturbations = [
        float(x) for x in protocol["boundary_perturbation_ladder_m"]
    ]
    max_isolated_fraction = float(
        protocol["geometric_support_gate"]["max_isolated_switch_fraction"]
    )
    min_median_support = float(
        protocol["geometric_support_gate"][
            "min_median_neighbor_support_fraction"
        ]
    )

    d2 = load_json(D2_PATH)
    d3_latest = load_json(D3_PATH)
    d4_latest = load_json(D4_PATH)

    if d2.get("status") != "PASS":
        raise RuntimeError("P3B1_R7D5_D2_UPSTREAM_FAIL")
    if d3_latest.get("status") != "PASS":
        raise RuntimeError("P3B1_R7D5_D3_UPSTREAM_FAIL")
    if d4_latest.get("status") != "PASS":
        raise RuntimeError("P3B1_R7D5_D4_UPSTREAM_FAIL")
    if not d4_latest["claims"].get(
        "supervisory_q_sensitivity_candidate", False
    ):
        raise RuntimeError("P3B1_R7D5_D4_SUPERVISORY_CANDIDATE_MISSING")
    if int(
        d4_latest["aggregate"]["kernel_q_component_nodes_sum"]
    ) != 0:
        raise RuntimeError("P3B1_R7D5_D4_KERNEL_COMPONENT_NONZERO")

    data = d1.reconstruct()
    transition_data = d3.build_transition_data(data)

    eval_idx = data["eval_node_indices"]
    eval_fallback = np.asarray(data["eval_fallback"], dtype=float)
    eval_shape = tuple(int(x) for x in data["eval_shape"])
    eval_flat = [np.asarray(x, dtype=float) for x in data["eval_flat"]]
    lookup_shape = data["lookup_shape"]
    results = data["results"]

    print("=== P3-B1-R7-D5 SWITCH BOUNDARY ROBUSTNESS ===")
    print(f"EVAL_GRID_NODES={len(eval_fallback)}")
    print(
        "AUDIT_SCOPE="
        "MODEL_LEVEL_NUMERICAL_AND_GRID_NEIGHBORHOOD_ROBUSTNESS_ONLY"
    )
    print("IMPLEMENTATION_REFINEMENT_Ic_SUBSET_Ac=NOT_TESTED")

    profile_rows = []
    ladder_rows = []
    node_rows = []
    switch_masks = {}
    max_replay = 0.0

    for name, result in results.items():
        raw, clipped, branch, action = d3.raw_operator_slices(
            result,
            lookup_shape,
            transition_data,
            eval_fallback,
        )

        stored = np.asarray(
            result["h_flat"][eval_idx, :],
            dtype=float,
        )
        replay_error = float(np.max(np.abs(stored - clipped)))
        max_replay = max(max_replay, replay_error)

        R = int(result["R"])
        raw_min = np.min(raw, axis=1)
        raw_max = np.max(raw, axis=1)
        raw_span = raw_max - raw_min
        clipped_span = (
            np.max(clipped, axis=1)
            -
            np.min(clipped, axis=1)
        )

        kernel_component = clipped_span
        switch_width = np.maximum(
            raw_span - kernel_component,
            0.0,
        )

        switch_mask = switch_width > (2.0 * tol)
        switch_masks[name] = switch_mask

        robust_radius = 0.5 * switch_width
        radius_stats = positive_summary(robust_radius, tol)

        support_count, degree, support_fraction = neighbor_support(
            switch_mask,
            eval_shape,
        )

        switch_idx = np.flatnonzero(switch_mask)
        if len(switch_idx):
            local_support = support_fraction[switch_idx]
            isolated = support_count[switch_idx] == 0
            isolated_count = int(np.count_nonzero(isolated))
            isolated_fraction = float(np.mean(isolated))
            support_p05 = float(np.quantile(local_support, 0.05))
            support_p50 = float(np.quantile(local_support, 0.50))
            support_p95 = float(np.quantile(local_support, 0.95))
            support_mean = float(np.mean(local_support))
        else:
            isolated_count = 0
            isolated_fraction = 0.0
            support_p05 = 0.0
            support_p50 = 0.0
            support_p95 = 0.0
            support_mean = 0.0

        action_q_change = any_q_label_change(action)
        branch_q_change = any_q_label_change(branch)

        action_change_on_switch = int(
            np.count_nonzero(action_q_change & switch_mask)
        )
        branch_change_on_switch = int(
            np.count_nonzero(branch_q_change & switch_mask)
        )

        # Optional structural diagnostic: q recursion should not need a
        # change of the selected cooperative action in D4. We do not assume
        # a direction for the raw boundary; we simply report adjacent
        # differences and sign reversals.
        if raw.shape[1] > 1:
            qdiff = np.diff(raw, axis=1)
            positive_adj = int(np.count_nonzero(qdiff > tol))
            negative_adj = int(np.count_nonzero(qdiff < -tol))
        else:
            positive_adj = 0
            negative_adj = 0

        min_radius_multiple = (
            radius_stats["min"] / tol
            if radius_stats["count"] > 0
            else 0.0
        )

        row = {
            "profile": name,
            "R": R,
            "evaluation_nodes": int(len(eval_fallback)),
            "fixed_point_replay_max_error_m": replay_error,
            "switch_witness_nodes":
                int(np.count_nonzero(switch_mask)),
            "kernel_component_nodes":
                int(np.count_nonzero(kernel_component > tol)),
            "robust_radius_min_m": radius_stats["min"],
            "robust_radius_p05_m": radius_stats["p05"],
            "robust_radius_p50_m": radius_stats["p50"],
            "robust_radius_p95_m": radius_stats["p95"],
            "robust_radius_max_m": radius_stats["max"],
            "robust_radius_mean_m": radius_stats["mean"],
            "min_radius_multiple_of_tol":
                float(min_radius_multiple),
            "isolated_switch_nodes": isolated_count,
            "isolated_switch_fraction": isolated_fraction,
            "neighbor_support_fraction_p05": support_p05,
            "neighbor_support_fraction_p50": support_p50,
            "neighbor_support_fraction_p95": support_p95,
            "neighbor_support_fraction_mean": support_mean,
            "action_q_change_nodes_on_switch":
                action_change_on_switch,
            "branch_q_change_nodes_on_switch":
                branch_change_on_switch,
            "adjacent_q_positive_differences":
                positive_adj,
            "adjacent_q_negative_differences":
                negative_adj,
        }
        profile_rows.append(row)

        for rho in perturbations:
            residual = np.maximum(
                switch_width - 2.0 * rho,
                0.0,
            )
            survivor = switch_mask & (residual > 0.0)
            denom = int(np.count_nonzero(switch_mask))
            count = int(np.count_nonzero(survivor))
            ladder_rows.append(
                {
                    "profile": name,
                    "R": R,
                    "symmetric_boundary_perturbation_m": rho,
                    "baseline_switch_nodes": denom,
                    "surviving_switch_nodes": count,
                    "survival_fraction":
                        float(count / denom) if denom else 0.0,
                    "residual_width_min_positive_m":
                        positive_summary(residual[survivor], 0.0)["min"]
                        if count else 0.0,
                    "residual_width_p50_positive_m":
                        positive_summary(residual[survivor], 0.0)["p50"]
                        if count else 0.0,
                }
            )

        vf, vp, af, bar_a, bar_u, age = eval_flat
        for i in switch_idx:
            node_rows.append(
                {
                    "profile": name,
                    "node_index": int(i),
                    "R": R,
                    "v_f": float(vf[i]),
                    "v_p": float(vp[i]),
                    "a_f": float(af[i]),
                    "bar_a": float(bar_a[i]),
                    "bar_u": float(bar_u[i]),
                    "age": float(age[i]),
                    "fallback_required_m": float(eval_fallback[i]),
                    "raw_min_m": float(raw_min[i]),
                    "raw_max_m": float(raw_max[i]),
                    "switch_interval_width_m":
                        float(switch_width[i]),
                    "symmetric_robust_radius_m":
                        float(robust_radius[i]),
                    "neighbor_degree":
                        int(degree[i]),
                    "neighbor_switch_support":
                        int(support_count[i]),
                    "neighbor_support_fraction":
                        float(support_fraction[i]),
                    "action_changes_across_q":
                        bool(action_q_change[i]),
                    "branch_changes_across_q":
                        bool(branch_q_change[i]),
                }
            )

        print(
            f"{name.upper()} "
            f"R={R} "
            f"SWITCH_NODES={row['switch_witness_nodes']} "
            f"KERNEL_COMP={row['kernel_component_nodes']} "
            f"RHO_MIN_M={row['robust_radius_min_m']:.12g} "
            f"RHO_P50_M={row['robust_radius_p50_m']:.12g} "
            f"ISOLATED={row['isolated_switch_nodes']} "
            f"NEIGHBOR_P50={row['neighbor_support_fraction_p50']:.12g} "
            f"ACTION_Q_ON_SWITCH={action_change_on_switch} "
            f"BRANCH_Q_ON_SWITCH={branch_change_on_switch} "
            f"FP_REPLAY_ERR={replay_error:.12g}"
        )

    diagnostic = (
        "diagnostic_fast",
        "diagnostic_nominal",
        "diagnostic_stressed",
    )

    masks_equal = all(
        np.array_equal(
            switch_masks[diagnostic[0]],
            switch_masks[name],
        )
        for name in diagnostic[1:]
    )

    diagnostic_rows = [
        row for row in profile_rows
        if row["profile"] in diagnostic
    ]

    switch_total = sum(
        row["switch_witness_nodes"]
        for row in diagnostic_rows
    )
    kernel_total = sum(
        row["kernel_component_nodes"]
        for row in diagnostic_rows
    )
    isolated_total = sum(
        row["isolated_switch_nodes"]
        for row in diagnostic_rows
    )
    action_confounded_total = sum(
        row["action_q_change_nodes_on_switch"]
        for row in diagnostic_rows
    )
    branch_change_total = sum(
        row["branch_q_change_nodes_on_switch"]
        for row in diagnostic_rows
    )

    min_radius_multiple_global = min(
        row["min_radius_multiple_of_tol"]
        for row in diagnostic_rows
    )
    max_isolated_observed = max(
        row["isolated_switch_fraction"]
        for row in diagnostic_rows
    )
    min_median_neighbor_support_observed = min(
        row["neighbor_support_fraction_p50"]
        for row in diagnostic_rows
    )

    numerical_margin_gate = (
        switch_total > 0
        and min_radius_multiple_global >= radius_multiple_gate
    )
    geometry_gate = (
        max_isolated_observed <= max_isolated_fraction
        and
        min_median_neighbor_support_observed >= min_median_support
    )
    action_deconfounding_gate = action_confounded_total == 0

    robustness_candidate = bool(
        kernel_total == 0
        and numerical_margin_gate
        and geometry_gate
        and action_deconfounding_gate
        and masks_equal
    )

    integrity_checks = {
        "D2_UPSTREAM_PASS":
            d2.get("status") == "PASS",
        "D3_UPSTREAM_PASS":
            d3_latest.get("status") == "PASS",
        "D4_UPSTREAM_PASS":
            d4_latest.get("status") == "PASS",
        "D4_SUPERVISORY_CANDIDATE_PRESENT":
            bool(
                d4_latest["claims"].get(
                    "supervisory_q_sensitivity_candidate",
                    False,
                )
            ),
        "D4_KERNEL_COMPONENT_ZERO":
            int(
                d4_latest["aggregate"][
                    "kernel_q_component_nodes_sum"
                ]
            ) == 0,
        "PROFILE_AUDIT_COMPLETE":
            len(profile_rows) == 4,
        "FIXED_POINT_OPERATOR_REPLAY":
            max_replay <= replay_tol,
        "D4_SWITCH_COUNT_REPRODUCED":
            switch_total
            ==
            int(
                d4_latest["aggregate"][
                    "safe_switch_q_component_nodes_sum"
                ]
            ),
    }

    status = "PASS" if all(integrity_checks.values()) else "FAIL"

    if kernel_total > 0:
        interpretation = (
            "KERNEL_COMPONENT_REAPPEARED_DURING_ROBUSTNESS_REPLAY"
        )
        next_step = (
            "STOP_AND_RECONCILE_D4_D5_KERNEL_DECOMPOSITION"
        )
    elif not numerical_margin_gate:
        interpretation = (
            "SWITCH_WITNESSES_REPRODUCED_BUT_NUMERICAL_MARGIN_IS_TOO_SMALL"
        )
        next_step = (
            "RUN_LOCAL_HIGH_PRECISION_BOUNDARY_RECOMPUTATION_BEFORE_REFINEMENT"
        )
    elif not geometry_gate:
        interpretation = (
            "SWITCH_WITNESSES_HAVE_NUMERICAL_MARGIN_BUT_WEAK_GRID_NEIGHBOR_SUPPORT"
        )
        next_step = (
            "RUN_P3B1_R7D6_LOCAL_BOUNDARY_REFINEMENT_AND_HOLDOUT_REPLAY"
        )
    elif not action_deconfounding_gate:
        interpretation = (
            "SUPERVISOR_SWITCH_EFFECT_IS_CONFOUNDED_WITH_COOPERATIVE_ACTION_CHANGE"
        )
        next_step = (
            "DECOMPOSE_ACTION_SELECTION_AND_SUPERVISOR_MODE_BEFORE_CLAIM"
        )
    elif not masks_equal:
        interpretation = (
            "SUPERVISOR_SWITCH_GEOMETRY_IS_PROFILE_SPECIFIC"
        )
        next_step = (
            "RUN_PROFILE_SPECIFIC_LOCAL_BOUNDARY_REFINEMENT"
        )
    else:
        interpretation = (
            "SUPERVISORY_Q_SWITCH_BOUNDARY_IS_NUMERICALLY_AND_GEOMETRICALLY_ROBUST_ON_CURRENT_GRID"
        )
        next_step = (
            "RUN_P3B1_R7D6_LOCAL_BOUNDARY_REFINEMENT_AND_HOLDOUT_REPLAY"
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    profile_csv = (
        RESULTS_DIR
        / f"P3B1_R7D5_PROFILE_ROBUSTNESS_{stamp}.csv"
    )
    ladder_csv = (
        RESULTS_DIR
        / f"P3B1_R7D5_PERTURBATION_LADDER_{stamp}.csv"
    )
    node_csv = (
        RESULTS_DIR
        / f"P3B1_R7D5_SWITCH_NEIGHBOR_AUDIT_{stamp}.csv"
    )

    write_csv(profile_csv, profile_rows)
    write_csv(ladder_csv, ladder_rows)
    write_csv(node_csv, node_rows)

    output = {
        "schema":
            "SCV_P3B1_R7D5_SWITCH_BOUNDARY_ROBUSTNESS_V1",
        "status":
            status,
        "timestamp_utc":
            stamp,
        "classification":
            (
                "model-level numerical and grid-neighborhood robustness "
                "audit for the certified q-dependent C/F supervisor switch "
                "boundary; not implementation refinement I_c subset A_c"
            ),
        "integrity_checks":
            {k: bool(v) for k, v in integrity_checks.items()},
        "scientific_gates": {
            "numerical_margin_gate":
                bool(numerical_margin_gate),
            "grid_neighborhood_support_gate":
                bool(geometry_gate),
            "action_deconfounding_gate":
                bool(action_deconfounding_gate),
            "diagnostic_switch_masks_identical":
                bool(masks_equal),
        },
        "interpretation":
            interpretation,
        "recommended_next_step":
            next_step,
        "profile_stats":
            profile_rows,
        "aggregate": {
            "diagnostic_switch_witness_nodes_sum":
                int(switch_total),
            "diagnostic_kernel_component_nodes_sum":
                int(kernel_total),
            "diagnostic_isolated_switch_nodes_sum":
                int(isolated_total),
            "diagnostic_action_q_change_on_switch_sum":
                int(action_confounded_total),
            "diagnostic_branch_q_change_on_switch_sum":
                int(branch_change_total),
            "min_robust_radius_multiple_of_tol":
                float(min_radius_multiple_global),
            "max_isolated_switch_fraction":
                float(max_isolated_observed),
            "min_median_neighbor_support_fraction":
                float(min_median_neighbor_support_observed),
            "max_fixed_point_replay_error_m":
                float(max_replay),
        },
        "protocol": protocol,
        "claims": {
            "scientific_kernel_claim_authorized":
                False,
            "model_level_supervisor_robustness_candidate":
                bool(robustness_candidate),
            "final_supervisory_q_sensitivity_claim_authorized":
                False,
            "implementation_refinement_claim_authorized":
                False,
            "p6_certified":
                False,
            "reason":
                (
                    "R7-D5 tests numerical margin and current-grid local "
                    "support only. R7-D6 local boundary refinement/holdout "
                    "replay remains required, and I_c subset A_c requires "
                    "real implementation traces rather than this model audit."
                ),
        },
        "artifacts": {
            "profile_csv": str(profile_csv),
            "perturbation_ladder_csv": str(ladder_csv),
            "switch_neighbor_csv": str(node_csv),
        },
    }

    result_path = (
        RESULTS_DIR
        / f"P3B1_R7D5_SWITCH_BOUNDARY_ROBUSTNESS_{stamp}.json"
    )
    latest_path = RESULTS_DIR / "P3B1_R7D5_LATEST.json"
    manifest_path = (
        RESULTS_DIR
        / f"P3B1_R7D5_MANIFEST_{stamp}.sha256"
    )

    output["artifacts"]["result_json"] = str(result_path)
    output["artifacts"]["latest_json"] = str(latest_path)
    output["artifacts"]["manifest"] = str(manifest_path)

    text = json.dumps(output, indent=2, sort_keys=True)
    atomic_write(result_path, text)
    atomic_write(latest_path, text)

    manifest_files = [
        P1_CFG_PATH,
        P2B_CFG_PATH,
        P2C_CFG_PATH,
        P3A_CFG_PATH,
        D5_CFG_PATH,
        D2_PATH,
        D3_PATH,
        D4_PATH,
        Path(d1.__file__),
        Path(d3.__file__),
        Path(d4.__file__),
        Path(d3.r7.__file__),
        Path(d3.r3.__file__),
        Path(__file__),
        result_path,
        profile_csv,
        ladder_csv,
        node_csv,
    ]

    manifest = "\n".join(
        f"{sha256_file(path)}  {path}"
        for path in manifest_files
    ) + "\n"
    atomic_write(manifest_path, manifest)

    print("=== P3-B1-R7-D5 DECISION ===")
    print(f"DIAGNOSTIC_SWITCH_WITNESS_NODES_SUM={switch_total}")
    print(f"KERNEL_COMPONENT_NODES_SUM={kernel_total}")
    print(f"ISOLATED_SWITCH_NODES_SUM={isolated_total}")
    print(f"ACTION_Q_CHANGE_ON_SWITCH_SUM={action_confounded_total}")
    print(f"BRANCH_Q_CHANGE_ON_SWITCH_SUM={branch_change_total}")
    print(
        "MIN_ROBUST_RADIUS_MULTIPLE_OF_TOL="
        f"{min_radius_multiple_global:.12g}"
    )
    print(
        "MAX_ISOLATED_SWITCH_FRACTION="
        f"{max_isolated_observed:.12g}"
    )
    print(
        "MIN_MEDIAN_NEIGHBOR_SUPPORT_FRACTION="
        f"{min_median_neighbor_support_observed:.12g}"
    )
    print(
        "DIAGNOSTIC_SWITCH_MASKS_IDENTICAL="
        + ("YES" if masks_equal else "NO")
    )
    print(f"INTERPRETATION={interpretation}")
    print(f"RECOMMENDED_NEXT_STEP={next_step}")
    print(f"P3B1_R7D5_SWITCH_BOUNDARY_ROBUSTNESS={status}")
    print(
        "MODEL_LEVEL_SUPERVISOR_ROBUSTNESS_CANDIDATE="
        + ("YES" if robustness_candidate else "NO")
    )
    print("SCIENTIFIC_KERNEL_CLAIM=NO")
    print("FINAL_SUPERVISORY_Q_SENSITIVITY_CLAIM_AUTHORIZED=NO")
    print("IMPLEMENTATION_REFINEMENT_CLAIM_AUTHORIZED=NO")
    print(f"RESULT_JSON={result_path}")
    print(f"MANIFEST={manifest_path}")

    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
