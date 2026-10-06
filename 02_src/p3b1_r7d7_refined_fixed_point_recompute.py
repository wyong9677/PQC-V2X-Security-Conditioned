from __future__ import annotations

import copy
import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

import p3b1_r7d1_node_level_audit as d1

r3 = d1.r3
r7 = d1.r7
r6 = d1.r6

P1_CFG_PATH = ROOT / "01_config" / "p1_validation_v2.json"
P2B_CFG_PATH = ROOT / "01_config" / "p2b_hybrid_fallback_v1.json"
P2C_CFG_PATH = ROOT / "01_config" / "p2c_switching_guard_v1.json"
P3A_CFG_PATH = ROOT / "01_config" / "p3a_information_contract_v1.json"
D7_CFG_PATH = (
    ROOT / "01_config" / "p3b1_r7d7_refined_fixed_point_protocol_v1.json"
)

D4_PATH = ROOT / "04_results" / "P3B1_R7D4_LATEST.json"
D5_PATH = ROOT / "04_results" / "P3B1_R7D5_LATEST.json"
D6_PATH = ROOT / "04_results" / "P3B1_R7D6_LATEST.json"
D6B_PATH = ROOT / "04_results" / "P3B1_R7D6B_LATEST.json"
D6C_PATH = ROOT / "04_results" / "P3B1_R7D6C_LATEST.json"

RESULTS_DIR = ROOT / "04_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

AXIS_NAMES = ("v_f", "v_p", "a_f", "bar_a", "bar_u", "age")


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


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def union_axis(base: np.ndarray, values) -> np.ndarray:
    merged = np.concatenate(
        [
            np.asarray(base, dtype=float),
            np.asarray(list(values), dtype=float),
        ]
    )
    merged = np.unique(merged)
    merged.sort()
    return merged


def set_cfg_axes(cfg: dict, axes) -> dict:
    local = copy.deepcopy(cfg)
    for name, axis in zip(AXIS_NAMES, axes):
        local["grid"][name] = [
            float(x) for x in np.asarray(axis, dtype=float)
        ]
    return local


def exact_axis_index(axis: np.ndarray, value: float, atol=1.0e-12) -> int:
    axis = np.asarray(axis, dtype=float)
    hits = np.flatnonzero(
        np.isclose(axis, value, rtol=0.0, atol=atol)
    )
    if len(hits) != 1:
        raise RuntimeError(
            f"EXACT_AXIS_INDEX_FAIL value={value} hits={len(hits)}"
        )
    return int(hits[0])


def point_eval_flat_index(axes, shape, row: dict) -> int:
    coords = [
        float(row["v_f"]),
        float(row["v_p"]),
        float(row["a_f"]),
        float(row["bar_a"]),
        float(row["bar_u"]),
        float(row["age"]),
    ]
    idx = tuple(
        exact_axis_index(axis, value)
        for axis, value in zip(axes, coords)
    )
    return int(np.ravel_multi_index(idx, shape))


def q_span_stats(h_eval: np.ndarray, tol: float) -> dict:
    if h_eval.shape[1] <= 1:
        span = np.zeros(h_eval.shape[0], dtype=float)
    else:
        span = np.max(h_eval, axis=1) - np.min(h_eval, axis=1)

    positive = span[span > tol]
    if len(positive):
        return {
            "count": int(len(positive)),
            "min_m": float(np.min(positive)),
            "p05_m": float(np.quantile(positive, 0.05)),
            "p50_m": float(np.quantile(positive, 0.50)),
            "p95_m": float(np.quantile(positive, 0.95)),
            "max_m": float(np.max(positive)),
            "mean_m": float(np.mean(positive)),
        }

    return {
        "count": 0,
        "min_m": 0.0,
        "p05_m": 0.0,
        "p50_m": 0.0,
        "p95_m": 0.0,
        "max_m": 0.0,
        "mean_m": 0.0,
    }


def baseline_q_control(data, tol: float) -> dict:
    out = {}
    for name, result in data["results"].items():
        h_eval = np.asarray(
            result["h_flat"][data["eval_node_indices"], :],
            dtype=float,
        )
        out[name] = q_span_stats(h_eval, tol)
    return out


def run_axis_scenario(
    axis_name: str,
    inserted_values,
    candidate_rows,
    data,
    p1_cfg,
    p2b_cfg,
    p2c_cfg,
    p3a_cfg,
    protocol,
):
    tol = float(protocol["numeric_tolerance_m"])
    min_multiple = float(
        protocol["fixed_point_candidate_min_multiple_of_tolerance"]
    )
    max_eval_nodes = int(
        protocol["max_eval_grid_nodes_per_axis_scenario"]
    )
    max_lookup_nodes = int(
        protocol["max_lookup_grid_nodes_per_axis_scenario"]
    )

    axis_index = AXIS_NAMES.index(axis_name)

    refined_eval_axes = [
        np.asarray(axis, dtype=float).copy()
        for axis in data["eval_axes"]
    ]
    refined_eval_axes[axis_index] = union_axis(
        refined_eval_axes[axis_index],
        inserted_values,
    )

    refined_lookup_axes = [
        np.asarray(axis, dtype=float).copy()
        for axis in data["lookup_axes"]
    ]
    refined_lookup_axes[axis_index] = union_axis(
        refined_lookup_axes[axis_index],
        inserted_values,
    )

    refined_eval_cfg = set_cfg_axes(
        data["eval_cfg"],
        refined_eval_axes,
    )

    eval_flat, eval_shape = r6.mesh_flat(refined_eval_axes)
    lookup_flat, lookup_shape = r6.mesh_flat(refined_lookup_axes)

    eval_nodes = int(np.prod(eval_shape))
    lookup_nodes = int(np.prod(lookup_shape))

    if eval_nodes > max_eval_nodes:
        raise RuntimeError(
            f"D7_EVAL_GRID_BUDGET_EXCEEDED axis={axis_name} "
            f"nodes={eval_nodes} max={max_eval_nodes}"
        )
    if lookup_nodes > max_lookup_nodes:
        raise RuntimeError(
            f"D7_LOOKUP_GRID_BUDGET_EXCEEDED axis={axis_name} "
            f"nodes={lookup_nodes} max={max_lookup_nodes}"
        )

    stop = r3.endpoint_semantics_audit(
        refined_eval_cfg,
        p1_cfg,
        p2b_cfg,
        p3a_cfg,
        eval_flat,
    )
    if not stop["pass"]:
        raise RuntimeError(
            f"D7_STOP_SEMANTICS_FAIL axis={axis_name}"
        )

    eval_node_indices = r7.exact_node_indices(
        refined_eval_axes,
        refined_lookup_axes,
    )

    lookup_fallback = r6.fallback_required_on_grid(
        refined_eval_cfg,
        p1_cfg,
        p2b_cfg,
        p2c_cfg,
        p3a_cfg,
        lookup_flat,
    )

    transition_data = r7.build_transition_data_to_lookup(
        refined_eval_cfg,
        p1_cfg,
        p2b_cfg,
        p2c_cfg,
        p3a_cfg,
        eval_flat,
        refined_lookup_axes,
    )

    completion_invalid = sum(
        int(np.count_nonzero(~tr["completion_valid"]))
        for tr in transition_data["transitions"]
    )
    defer_invalid = sum(
        int(np.count_nonzero(~tr["defer_valid"]))
        for tr in transition_data["transitions"]
    )

    if completion_invalid != 0 or defer_invalid != 0:
        raise RuntimeError(
            f"D7_FROZEN_HALO_COVERAGE_FAIL axis={axis_name} "
            f"completion_invalid={completion_invalid} "
            f"defer_invalid={defer_invalid}"
        )

    eval_fallback = np.asarray(
        transition_data["fallback_required"],
        dtype=float,
    )

    profiles = p1_cfg["diagnostic_service_profiles"]
    profile_order = [
        "ideal",
        "diagnostic_fast",
        "diagnostic_nominal",
        "diagnostic_stressed",
    ]

    results = {}
    profile_rows = []
    convergence_rows = []

    print(
        f"--- AXIS SCENARIO {axis_name} "
        f"INSERTED_VALUES={len(set(float(x) for x in inserted_values))} "
        f"EVAL_NODES={eval_nodes} LOOKUP_NODES={lookup_nodes} ---"
    )

    for name in profile_order:
        R = r7.service_horizon(name, profiles[name])

        result = r7.solve_frozen_halo_profile(
            name,
            R,
            refined_eval_cfg,
            eval_node_indices,
            len(eval_node_indices),
            lookup_shape,
            lookup_fallback,
            eval_fallback,
            transition_data,
        )
        results[name] = result

        h_eval = np.asarray(
            result["h_flat"][eval_node_indices, :],
            dtype=float,
        )
        global_stats = q_span_stats(h_eval, tol)

        base_lookup_indices = r7.exact_node_indices(
            data["eval_axes"],
            refined_lookup_axes,
        )
        h_base = np.asarray(
            result["h_flat"][base_lookup_indices, :],
            dtype=float,
        )
        base_stats = q_span_stats(h_base, tol)

        profile_rows.append(
            {
                "axis_scenario": axis_name,
                "profile": name,
                "R": int(R),
                "eval_grid_nodes": eval_nodes,
                "lookup_grid_nodes": lookup_nodes,
                "converged": bool(result["converged"]),
                "iterations": int(result["iterations"]),
                "final_change_m": float(result["final_change"]),
                "monotone_violation_m":
                    float(result["monotone_violation"]),
                "outside_halo_change_max_m":
                    float(result["outside_halo_change_max"]),
                "refined_eval_q_dependent_nodes":
                    int(global_stats["count"]),
                "refined_eval_q_span_min_positive_m":
                    float(global_stats["min_m"]),
                "refined_eval_q_span_p50_positive_m":
                    float(global_stats["p50_m"]),
                "refined_eval_q_span_max_m":
                    float(global_stats["max_m"]),
                "original_base_nodes_q_dependent_after_refinement":
                    int(base_stats["count"]),
                "original_base_nodes_q_span_max_m":
                    float(base_stats["max_m"]),
            }
        )

        for hist in result["history"]:
            convergence_rows.append(
                {
                    "axis_scenario": axis_name,
                    "profile": name,
                    **hist,
                }
            )

        print(
            f"{name.upper()} R={R} "
            f"ITER={result['iterations']} "
            f"CONVERGED={'YES' if result['converged'] else 'NO'} "
            f"GLOBAL_QDEP={global_stats['count']} "
            f"BASE_QDEP={base_stats['count']} "
            f"QSPAN_MAX_M={global_stats['max_m']:.12g} "
            f"FINAL_CHANGE_M={result['final_change']:.12g}"
        )

    # Structural common-domain audit.
    evaluation = r7.evaluate_common_domain(
        refined_eval_cfg,
        p2b_cfg,
        p3a_cfg,
        refined_lookup_axes,
        lookup_shape,
        lookup_fallback,
        results,
    )

    verdicts = evaluation["verdicts"]

    fallback = verdicts["fallback"]
    ideal = verdicts["ideal"]
    fast = verdicts["diagnostic_fast"]
    nominal = verdicts["diagnostic_nominal"]
    stressed = verdicts["diagnostic_stressed"]

    fallback_inclusion_violations = {
        name: int(
            np.count_nonzero(
                fallback & ~verdicts[name]
            )
        )
        for name in profile_order
    }

    service_order_violations = {
        "fast_not_subset_ideal":
            int(np.count_nonzero(fast & ~ideal)),
        "nominal_not_subset_fast":
            int(np.count_nonzero(nominal & ~fast)),
        "stressed_not_subset_nominal":
            int(np.count_nonzero(stressed & ~nominal)),
    }

    strict_gain = {
        "ideal_over_fallback":
            int(np.count_nonzero(ideal & ~fallback)),
        "fast_over_fallback":
            int(np.count_nonzero(fast & ~fallback)),
        "nominal_over_fallback":
            int(np.count_nonzero(nominal & ~fallback)),
        "stressed_over_fallback":
            int(np.count_nonzero(stressed & ~fallback)),
        "fast_over_stressed":
            int(np.count_nonzero(fast & ~stressed)),
    }

    max_monotone = max(
        float(result["monotone_violation"])
        for result in results.values()
    )
    max_halo_change = max(
        float(result["outside_halo_change_max"])
        for result in results.values()
    )
    all_converged = all(
        bool(result["converged"])
        for result in results.values()
    )

    # Exact candidate-node audit on the newly refined tensor grid.
    candidate_audit_rows = []
    all_candidate_profile_pass = True
    min_candidate_multiple = float("inf")

    for candidate in candidate_rows:
        eval_flat_idx = point_eval_flat_index(
            refined_eval_axes,
            eval_shape,
            candidate,
        )
        lookup_idx = int(eval_node_indices[eval_flat_idx])

        candidate_out = {
            "axis_scenario": axis_name,
            "unique_witness_id":
                int(candidate["unique_witness_id"]),
            "occurrence_id":
                int(candidate["occurrence_id"]),
            "edge_id":
                int(candidate["edge_id"]),
            "axis_name": axis_name,
            "t": float(candidate["t"]),
            "v_f": float(candidate["v_f"]),
            "v_p": float(candidate["v_p"]),
            "a_f": float(candidate["a_f"]),
            "bar_a": float(candidate["bar_a"]),
            "bar_u": float(candidate["bar_u"]),
            "age": float(candidate["age"]),
            "refined_eval_flat_index": int(eval_flat_idx),
            "refined_lookup_index": int(lookup_idx),
        }

        for name in (
            "diagnostic_fast",
            "diagnostic_nominal",
            "diagnostic_stressed",
        ):
            hq = np.asarray(
                results[name]["h_flat"][lookup_idx, :],
                dtype=float,
            )
            span = float(np.max(hq) - np.min(hq))
            multiple = span / tol
            positive = span > tol
            strong = multiple >= min_multiple

            candidate_out[f"{name}_q_span_m"] = span
            candidate_out[f"{name}_q_span_multiple_of_tol"] = multiple
            candidate_out[f"{name}_q_dependent"] = bool(positive)
            candidate_out[f"{name}_strong_q_dependent"] = bool(strong)

            all_candidate_profile_pass &= bool(strong)
            if positive:
                min_candidate_multiple = min(
                    min_candidate_multiple,
                    multiple,
                )

        candidate_audit_rows.append(candidate_out)

    if min_candidate_multiple == float("inf"):
        min_candidate_multiple = 0.0

    checks = {
        "STOP_SEMANTICS_GATE":
            bool(stop["pass"]),
        "FROZEN_HALO_COVERAGE":
            completion_invalid == 0 and defer_invalid == 0,
        "EVAL_LOOKUP_EMBEDDING_COMPLETE":
            len(eval_node_indices) == eval_nodes,
        "FIXED_POINT_CONVERGED_ALL":
            bool(all_converged),
        "MONOTONE_PREDECESSOR_ITERATION":
            max_monotone <= 1.0e-12,
        "HALO_OUTSIDE_EVAL_FROZEN":
            max_halo_change <= 1.0e-12,
        "COMMON_DOMAIN_LOOKUP_VALID":
            bool(evaluation["all_lookup_valid"]),
        "FALLBACK_INCLUDED_ALL_PROFILES":
            sum(fallback_inclusion_violations.values()) == 0,
        "SERVICE_ORDER_NESTED":
            sum(service_order_violations.values()) == 0,
        "ALL_AXIS_CANDIDATE_PROFILE_Q_SPANS_STRONG":
            bool(all_candidate_profile_pass),
    }

    return {
        "axis_name": axis_name,
        "inserted_values":
            sorted(set(float(x) for x in inserted_values)),
        "eval_shape": [int(x) for x in eval_shape],
        "lookup_shape": [int(x) for x in lookup_shape],
        "eval_nodes": eval_nodes,
        "lookup_nodes": lookup_nodes,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "profile_rows": profile_rows,
        "convergence_rows": convergence_rows,
        "candidate_rows": candidate_audit_rows,
        "minimum_candidate_q_span_multiple_of_tol":
            float(min_candidate_multiple),
        "fractions": {
            k: float(v)
            for k, v in evaluation["fractions"].items()
        },
        "fallback_inclusion_violations":
            fallback_inclusion_violations,
        "service_order_violations":
            service_order_violations,
        "strict_gain": strict_gain,
    }


def main() -> int:
    protocol = load_json(D7_CFG_PATH)
    tol = float(protocol["numeric_tolerance_m"])

    d4_latest = load_json(D4_PATH)
    d5_latest = load_json(D5_PATH)
    d6_latest = load_json(D6_PATH)
    d6b_latest = load_json(D6B_PATH)
    d6c_latest = load_json(D6C_PATH)

    for label, obj in (
        ("D4", d4_latest),
        ("D5", d5_latest),
        ("D6", d6_latest),
        ("D6B", d6b_latest),
        ("D6C", d6c_latest),
    ):
        if obj.get("status") != "PASS":
            raise RuntimeError(
                f"P3B1_R7D7_{label}_UPSTREAM_FAIL"
            )

    expected = (
        "Q_DEPENDENT_KERNEL_COMPONENT_HAS_ULP_STABLE_OPEN_NEIGHBORHOOD_OPERATOR_WITNESSES"
    )
    if d6c_latest.get("interpretation") != expected:
        raise RuntimeError(
            "P3B1_R7D7_UNEXPECTED_D6C_INTERPRETATION="
            + str(d6c_latest.get("interpretation"))
        )
    if not d6c_latest["claims"].get(
        "operator_level_refined_fixed_point_recompute_candidate",
        False,
    ):
        raise RuntimeError(
            "P3B1_R7D7_D6C_RECOMPUTE_CANDIDATE_MISSING"
        )

    occurrence_csv = Path(
        d6c_latest["artifacts"]["occurrence_csv"]
    )
    unique_csv = Path(
        d6c_latest["artifacts"]["unique_witness_csv"]
    )

    occurrence_rows = read_csv(occurrence_csv)
    unique_rows = read_csv(unique_csv)

    open_ids = {
        int(row["unique_witness_id"])
        for row in unique_rows
        if row["local_classification"]
        == "OPEN_NEIGHBORHOOD_KERNEL_WITNESS_CANDIDATE"
    }

    candidates = [
        row
        for row in occurrence_rows
        if int(row["unique_witness_id"]) in open_ids
    ]

    expected_open = int(
        d6c_latest["aggregate"][
            "unique_open_neighborhood_witnesses"
        ]
    )

    if len(candidates) != expected_open:
        raise RuntimeError(
            "P3B1_R7D7_OPEN_WITNESS_COUNT_MISMATCH "
            f"candidates={len(candidates)} expected={expected_open}"
        )

    by_axis = defaultdict(list)
    for row in candidates:
        by_axis[row["axis_name"]].append(row)

    data = d1.reconstruct()

    p1_cfg = load_json(P1_CFG_PATH)
    p2b_cfg = load_json(P2B_CFG_PATH)
    p2c_cfg = load_json(P2C_CFG_PATH)
    p3a_cfg = load_json(P3A_CFG_PATH)

    baseline = baseline_q_control(data, tol)

    baseline_diag_qdep = sum(
        baseline[name]["count"]
        for name in (
            "diagnostic_fast",
            "diagnostic_nominal",
            "diagnostic_stressed",
        )
    )

    print("=== P3-B1-R7-D7 REFINED FIXED-POINT KERNEL RECOMPUTATION ===")
    print(f"D6C_OPEN_WITNESSES={expected_open}")
    print(
        "WITNESS_AXES="
        + ",".join(
            f"{name}:{len(rows)}"
            for name, rows in sorted(by_axis.items())
        )
    )
    print(
        "BASELINE_DIAGNOSTIC_Q_DEPENDENT_NODES_SUM="
        f"{baseline_diag_qdep}"
    )

    scenario_results = []

    for axis_name in AXIS_NAMES:
        rows = by_axis.get(axis_name, [])
        if not rows:
            continue

        inserted_values = [
            float(row[axis_name])
            for row in rows
        ]

        scenario = run_axis_scenario(
            axis_name,
            inserted_values,
            rows,
            data,
            p1_cfg,
            p2b_cfg,
            p2c_cfg,
            p3a_cfg,
            protocol,
        )
        scenario_results.append(scenario)

        print(
            f"AXIS_{axis_name.upper()}_STATUS="
            f"{scenario['status']} "
            f"CANDIDATES={len(rows)} "
            f"MIN_CANDIDATE_MULTIPLE="
            f"{scenario['minimum_candidate_q_span_multiple_of_tol']:.12g}"
        )

    all_scenarios_pass = all(
        scenario["status"] == "PASS"
        for scenario in scenario_results
    )
    total_candidate_rows = sum(
        len(scenario["candidate_rows"])
        for scenario in scenario_results
    )
    strong_candidate_rows = sum(
        int(
            all(
                bool(row[f"{name}_strong_q_dependent"])
                for name in (
                    "diagnostic_fast",
                    "diagnostic_nominal",
                    "diagnostic_stressed",
                )
            )
        )
        for scenario in scenario_results
        for row in scenario["candidate_rows"]
    )

    global_qdep_by_profile = {
        name: 0
        for name in (
            "diagnostic_fast",
            "diagnostic_nominal",
            "diagnostic_stressed",
        )
    }
    base_qdep_after_refine_by_profile = {
        name: 0
        for name in global_qdep_by_profile
    }

    for scenario in scenario_results:
        rows_by_profile = {
            row["profile"]: row
            for row in scenario["profile_rows"]
        }
        for name in global_qdep_by_profile:
            global_qdep_by_profile[name] += int(
                rows_by_profile[name][
                    "refined_eval_q_dependent_nodes"
                ]
            )
            base_qdep_after_refine_by_profile[name] += int(
                rows_by_profile[name][
                    "original_base_nodes_q_dependent_after_refinement"
                ]
            )

    min_candidate_multiple = min(
        scenario["minimum_candidate_q_span_multiple_of_tol"]
        for scenario in scenario_results
    ) if scenario_results else 0.0

    refined_candidate_confirmed = bool(
        baseline_diag_qdep == 0
        and all_scenarios_pass
        and total_candidate_rows == expected_open
        and strong_candidate_rows == expected_open
    )

    if refined_candidate_confirmed:
        interpretation = (
            "Q_DEPENDENT_KERNEL_COMPONENT_PERSISTS_AFTER_TRUE_AXIS_REFINED_FIXED_POINT_RECOMPUTATION"
        )
        next_step = (
            "RUN_P3B1_R7D7B_SECOND_LEVEL_REFINEMENT_STABILITY_AND_CERTIFICATE_AUDIT"
        )
    elif baseline_diag_qdep != 0:
        interpretation = (
            "BASELINE_ZERO_Q_DEPENDENCE_CONTROL_NOT_REPRODUCED"
        )
        next_step = (
            "STOP_AND_RECONCILE_D3_D7_BASELINE_FIXED_POINT"
        )
    elif not all_scenarios_pass:
        interpretation = (
            "REFINED_FIXED_POINT_STRUCTURAL_GATE_FAILED"
        )
        next_step = (
            "REPAIR_FAILED_REFINEMENT_GATE_BEFORE_KERNEL_INTERPRETATION"
        )
    else:
        interpretation = (
            "OPERATOR_LEVEL_KERNEL_WITNESSES_DO_NOT_ALL_PERSIST_AFTER_FIXED_POINT_RESOLUTION"
        )
        next_step = (
            "LOCALIZE_NONPERSISTENT_WITNESSES_AND_DO_NOT_PROMOTE_KERNEL_CLAIM"
        )

    integrity_checks = {
        "D4_UPSTREAM_PASS":
            d4_latest.get("status") == "PASS",
        "D5_UPSTREAM_PASS":
            d5_latest.get("status") == "PASS",
        "D6_UPSTREAM_PASS":
            d6_latest.get("status") == "PASS",
        "D6B_UPSTREAM_PASS":
            d6b_latest.get("status") == "PASS",
        "D6C_UPSTREAM_PASS":
            d6c_latest.get("status") == "PASS",
        "D6C_OPEN_WITNESS_COUNT_REPRODUCED":
            len(candidates) == expected_open,
        "BASELINE_ZERO_Q_DEPENDENCE_CONTROL":
            baseline_diag_qdep == 0,
        "AXIS_SCENARIOS_PRESENT":
            len(scenario_results) > 0,
    }

    status = "PASS" if all(integrity_checks.values()) else "FAIL"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    profile_csv = (
        RESULTS_DIR
        / f"P3B1_R7D7_AXIS_PROFILE_AUDIT_{stamp}.csv"
    )
    candidate_csv = (
        RESULTS_DIR
        / f"P3B1_R7D7_REFINED_KERNEL_WITNESSES_{stamp}.csv"
    )
    convergence_csv = (
        RESULTS_DIR
        / f"P3B1_R7D7_FIXED_POINT_CONVERGENCE_{stamp}.csv"
    )
    scenario_csv = (
        RESULTS_DIR
        / f"P3B1_R7D7_SCENARIO_SUMMARY_{stamp}.csv"
    )

    profile_rows = [
        row
        for scenario in scenario_results
        for row in scenario["profile_rows"]
    ]
    candidate_rows = [
        row
        for scenario in scenario_results
        for row in scenario["candidate_rows"]
    ]
    convergence_rows = [
        row
        for scenario in scenario_results
        for row in scenario["convergence_rows"]
    ]

    scenario_rows = []
    for scenario in scenario_results:
        scenario_rows.append(
            {
                "axis_scenario": scenario["axis_name"],
                "status": scenario["status"],
                "inserted_values":
                    ";".join(
                        f"{x:.17g}"
                        for x in scenario["inserted_values"]
                    ),
                "eval_nodes": scenario["eval_nodes"],
                "lookup_nodes": scenario["lookup_nodes"],
                "minimum_candidate_q_span_multiple_of_tol":
                    scenario[
                        "minimum_candidate_q_span_multiple_of_tol"
                    ],
                "checks":
                    json.dumps(
                        scenario["checks"],
                        sort_keys=True,
                    ),
                "fractions":
                    json.dumps(
                        scenario["fractions"],
                        sort_keys=True,
                    ),
                "strict_gain":
                    json.dumps(
                        scenario["strict_gain"],
                        sort_keys=True,
                    ),
            }
        )

    write_csv(profile_csv, profile_rows)
    write_csv(candidate_csv, candidate_rows)
    write_csv(convergence_csv, convergence_rows)
    write_csv(scenario_csv, scenario_rows)

    output = {
        "schema":
            "SCV_P3B1_R7D7_AXIS_REFINED_FIXED_POINT_V1",
        "status":
            status,
        "timestamp_utc":
            stamp,
        "classification":
            (
                "axis-grouped tensor-grid frozen-halo Bellman fixed-point "
                "recomputation with D6C witnesses inserted as exact state "
                "nodes; finite-abstraction result only"
            ),
        "integrity_checks":
            {k: bool(v) for k, v in integrity_checks.items()},
        "interpretation":
            interpretation,
        "recommended_next_step":
            next_step,
        "protocol":
            protocol,
        "baseline_q_control":
            baseline,
        "scenario_summaries": [
            {
                "axis_name": scenario["axis_name"],
                "status": scenario["status"],
                "inserted_values":
                    scenario["inserted_values"],
                "eval_shape":
                    scenario["eval_shape"],
                "lookup_shape":
                    scenario["lookup_shape"],
                "eval_nodes":
                    scenario["eval_nodes"],
                "lookup_nodes":
                    scenario["lookup_nodes"],
                "checks":
                    scenario["checks"],
                "minimum_candidate_q_span_multiple_of_tol":
                    scenario[
                        "minimum_candidate_q_span_multiple_of_tol"
                    ],
                "fractions":
                    scenario["fractions"],
                "fallback_inclusion_violations":
                    scenario[
                        "fallback_inclusion_violations"
                    ],
                "service_order_violations":
                    scenario[
                        "service_order_violations"
                    ],
                "strict_gain":
                    scenario["strict_gain"],
            }
            for scenario in scenario_results
        ],
        "aggregate": {
            "d6c_open_witnesses":
                int(expected_open),
            "axis_scenarios":
                int(len(scenario_results)),
            "candidate_rows_recomputed":
                int(total_candidate_rows),
            "candidate_rows_strong_in_all_diagnostic_profiles":
                int(strong_candidate_rows),
            "minimum_candidate_q_span_multiple_of_tol":
                float(min_candidate_multiple),
            "baseline_diagnostic_q_dependent_nodes_sum":
                int(baseline_diag_qdep),
            "refined_eval_q_dependent_nodes_sum_by_profile":
                global_qdep_by_profile,
            "original_base_nodes_q_dependent_after_refinement_sum_by_profile":
                base_qdep_after_refine_by_profile,
        },
        "claims": {
            "refined_finite_abstraction_q_dependent_kernel_candidate":
                bool(refined_candidate_confirmed),
            "scientific_kernel_claim_authorized":
                False,
            "maximal_continuous_kernel_claim_authorized":
                False,
            "implementation_refinement_claim_authorized":
                False,
            "p6_certified":
                False,
            "reason":
                (
                    "A positive R7-D7 confirms q-dependence after an actual "
                    "refined Bellman fixed-point solve, but independent second-"
                    "level refinement stability, certificate rationalization/"
                    "outward rounding, and theorem/implementation gates remain "
                    "required before a manuscript-level kernel claim."
                ),
        },
        "artifacts": {
            "profile_csv": str(profile_csv),
            "candidate_csv": str(candidate_csv),
            "convergence_csv": str(convergence_csv),
            "scenario_csv": str(scenario_csv),
        },
    }

    result_path = (
        RESULTS_DIR
        / f"P3B1_R7D7_REFINED_FIXED_POINT_{stamp}.json"
    )
    latest_path = RESULTS_DIR / "P3B1_R7D7_LATEST.json"
    manifest_path = (
        RESULTS_DIR
        / f"P3B1_R7D7_MANIFEST_{stamp}.sha256"
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
        D7_CFG_PATH,
        D4_PATH,
        D5_PATH,
        D6_PATH,
        D6B_PATH,
        D6C_PATH,
        occurrence_csv,
        unique_csv,
        Path(d1.__file__),
        Path(r3.__file__),
        Path(r7.__file__),
        Path(r6.__file__),
        Path(__file__),
        result_path,
        profile_csv,
        candidate_csv,
        convergence_csv,
        scenario_csv,
    ]

    manifest = "\n".join(
        f"{sha256_file(path)}  {path}"
        for path in manifest_files
    ) + "\n"
    atomic_write(manifest_path, manifest)

    print("=== P3-B1-R7-D7 DECISION ===")
    print(f"AXIS_SCENARIOS={len(scenario_results)}")
    print(f"CANDIDATE_ROWS_RECOMPUTED={total_candidate_rows}")
    print(
        "CANDIDATE_ROWS_STRONG_ALL_DIAGNOSTIC_PROFILES="
        f"{strong_candidate_rows}"
    )
    print(
        "MIN_CANDIDATE_Q_SPAN_MULTIPLE_OF_TOL="
        f"{min_candidate_multiple:.12g}"
    )
    for name in (
        "diagnostic_fast",
        "diagnostic_nominal",
        "diagnostic_stressed",
    ):
        print(
            f"REFINED_{name.upper()}_Q_DEPENDENT_NODES_SUM="
            f"{global_qdep_by_profile[name]}"
        )
        print(
            f"BASE_NODES_{name.upper()}_Q_DEP_AFTER_REFINEMENT_SUM="
            f"{base_qdep_after_refine_by_profile[name]}"
        )

    print(f"INTERPRETATION={interpretation}")
    print(f"RECOMMENDED_NEXT_STEP={next_step}")
    print(f"P3B1_R7D7_REFINED_FIXED_POINT_AUDIT={status}")
    print(
        "REFINED_FINITE_ABSTRACTION_Q_DEPENDENT_KERNEL_CANDIDATE="
        + ("YES" if refined_candidate_confirmed else "NO")
    )
    print("SCIENTIFIC_KERNEL_CLAIM_AUTHORIZED=NO")
    print("MAXIMAL_CONTINUOUS_KERNEL_CLAIM=NO")
    print("IMPLEMENTATION_REFINEMENT_CLAIM=NO")
    print(f"RESULT_JSON={result_path}")
    print(f"MANIFEST={manifest_path}")

    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
