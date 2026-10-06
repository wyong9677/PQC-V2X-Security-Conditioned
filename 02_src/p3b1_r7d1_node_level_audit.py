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
import p3b1_r5_refinement_attribution as r5
import p3b1_r6_continuation_halo as r6
import p3b1_r7_frozen_halo_fixed_point as r7

CFG_PATH = ROOT / "01_config" / "p3b1_augmented_fixed_point_v1.json"
P1_CFG_PATH = ROOT / "01_config" / "p1_validation_v2.json"
P2B_CFG_PATH = ROOT / "01_config" / "p2b_hybrid_fallback_v1.json"
P2C_CFG_PATH = ROOT / "01_config" / "p2c_switching_guard_v1.json"
P3A_CFG_PATH = ROOT / "01_config" / "p3a_information_contract_v1.json"
R6_PATH = ROOT / "04_results" / "P3B1_R6_LATEST.json"

RESULTS_DIR = ROOT / "04_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

AXIS_NAMES = ("v_f", "v_p", "a_f", "bar_a", "bar_u", "age")
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


def positive_stats(delta: np.ndarray) -> dict:
    mask = delta > TOL
    values = delta[mask]

    if len(values) == 0:
        return {
            "count": 0,
            "fraction": 0.0,
            "mean_m": 0.0,
            "p50_m": 0.0,
            "p95_m": 0.0,
            "max_m": 0.0,
        }

    return {
        "count": int(len(values)),
        "fraction": float(np.mean(mask)),
        "mean_m": float(np.mean(values)),
        "p50_m": float(np.quantile(values, 0.50)),
        "p95_m": float(np.quantile(values, 0.95)),
        "max_m": float(np.max(values)),
    }


def reconstruct():
    cfg = load_json(CFG_PATH)
    p1_cfg = load_json(P1_CFG_PATH)
    p2b_cfg = load_json(P2B_CFG_PATH)
    p2c_cfg = load_json(P2C_CFG_PATH)
    p3a_cfg = load_json(P3A_CFG_PATH)
    r6_result = load_json(R6_PATH)

    if r6_result.get("status") != "PASS":
        raise RuntimeError("R6_NOT_PASS")

    selected_axes = list(r6_result["selected_refinement_axes"])

    eval_cfg = r5.refine_cfg(
        cfg,
        tuple(selected_axes),
    )

    eval_axes, eval_flat, eval_shape = r3.build_grid(eval_cfg)

    halo_spec = load_json(
        Path(r6_result["halo_spec"])
    )

    lookup_axes = [
        np.asarray(
            halo_spec["lookup_halo_grid"][name],
            dtype=float,
        )
        for name in AXIS_NAMES
    ]

    lookup_flat, lookup_shape = r6.mesh_flat(
        lookup_axes
    )

    eval_node_indices = r7.exact_node_indices(
        eval_axes,
        lookup_axes,
    )

    lookup_fallback = r6.fallback_required_on_grid(
        eval_cfg,
        p1_cfg,
        p2b_cfg,
        p2c_cfg,
        p3a_cfg,
        lookup_flat,
    )

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
        int(np.count_nonzero(~t["completion_valid"]))
        for t in transition_data["transitions"]
    )
    defer_invalid = sum(
        int(np.count_nonzero(~t["defer_valid"]))
        for t in transition_data["transitions"]
    )

    if completion_invalid != 0 or defer_invalid != 0:
        raise RuntimeError("R7D1_HALO_COVERAGE_FAIL")

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

    for name in profile_order:
        R = r7.service_horizon(
            name,
            profiles[name],
        )

        results[name] = r7.solve_frozen_halo_profile(
            name,
            R,
            eval_cfg,
            eval_node_indices,
            len(eval_node_indices),
            lookup_shape,
            lookup_fallback,
            eval_fallback,
            transition_data,
        )

    return {
        "cfg": cfg,
        "eval_cfg": eval_cfg,
        "p2b_cfg": p2b_cfg,
        "selected_axes": selected_axes,
        "eval_axes": eval_axes,
        "eval_flat": eval_flat,
        "eval_shape": eval_shape,
        "lookup_axes": lookup_axes,
        "lookup_shape": lookup_shape,
        "eval_node_indices": eval_node_indices,
        "lookup_fallback": lookup_fallback,
        "eval_fallback": eval_fallback,
        "results": results,
    }


def eval_initial_slice(result: dict, eval_node_indices: np.ndarray) -> np.ndarray:
    return np.asarray(
        result["h_flat"][
            eval_node_indices,
            result["R"] - 1,
        ],
        dtype=float,
    )


def profile_q_slice_stats(result: dict, eval_node_indices: np.ndarray) -> dict:
    h = result["h_flat"][eval_node_indices, :]
    if h.shape[1] <= 1:
        return {
            "q_slices": int(h.shape[1]),
            "nodes_with_q_dependence": 0,
            "q_dependence_fraction": 0.0,
            "max_q_range_m": 0.0,
            "p95_positive_q_range_m": 0.0,
        }

    q_range = np.max(h, axis=1) - np.min(h, axis=1)
    positive = q_range[q_range > TOL]

    return {
        "q_slices": int(h.shape[1]),
        "nodes_with_q_dependence":
            int(np.count_nonzero(q_range > TOL)),
        "q_dependence_fraction":
            float(np.mean(q_range > TOL)),
        "max_q_range_m":
            float(np.max(q_range)),
        "p95_positive_q_range_m":
            float(np.quantile(positive, 0.95))
            if len(positive) else 0.0,
    }


def make_witness_rows(
    eval_flat,
    lower_req: np.ndarray,
    upper_req: np.ndarray,
    d_min: float,
    d_max: float,
    label: str,
    max_rows: int = 64,
):
    improvement = upper_req - lower_req
    idx = np.flatnonzero(improvement > TOL)

    if len(idx) == 0:
        return []

    order = idx[
        np.argsort(
            improvement[idx]
        )[::-1]
    ]

    vf, vp, af, bar_a, bar_u, age = eval_flat

    rows = []

    for i in order:
        low_boundary = d_min + lower_req[i]
        high_boundary = d_min + upper_req[i]

        witness_d = 0.5 * (
            low_boundary + high_boundary
        )

        if not (
            witness_d >= d_min - 1e-12
            and witness_d <= d_max + 1e-12
        ):
            continue

        rows.append(
            {
                "witness_type": label,
                "node_index": int(i),
                "d_witness": float(witness_d),
                "v_f": float(vf[i]),
                "v_p": float(vp[i]),
                "a_f": float(af[i]),
                "bar_a": float(bar_a[i]),
                "bar_u": float(bar_u[i]),
                "age": float(age[i]),
                "lower_required_gap_m": float(lower_req[i]),
                "upper_required_gap_m": float(upper_req[i]),
                "gap_advantage_m": float(improvement[i]),
                "lower_boundary_d_m": float(low_boundary),
                "upper_boundary_d_m": float(high_boundary),
            }
        )

        if len(rows) >= max_rows:
            break

    return rows


def main():
    data = reconstruct()

    cfg = data["cfg"]
    p2b_cfg = data["p2b_cfg"]
    eval_flat = data["eval_flat"]
    eval_node_indices = data["eval_node_indices"]
    eval_fallback = data["eval_fallback"]
    results = data["results"]

    required = {
        name:
            eval_initial_slice(
                result,
                eval_node_indices,
            )
        for name, result in results.items()
    }

    print("=== P3-B1-R7-D1 NODE-LEVEL FIXED-POINT AUDIT ===")
    print(
        "SELECTED_REFINEMENT_AXES="
        + ",".join(data["selected_axes"])
    )
    print(
        "EVAL_GRID_NODES="
        f"{len(eval_fallback)}"
    )

    profile_stats = {}

    for name, req in required.items():
        delta = eval_fallback - req
        stats = positive_stats(delta)

        first_change = float(
            results[name]["history"][0]["sup_change_m"]
        )

        profile_stats[name] = {
            "fallback_improvement":
                stats,
            "first_iteration_sup_change_m":
                first_change,
            "iterations":
                int(results[name]["iterations"]),
            "q_slice_stats":
                profile_q_slice_stats(
                    results[name],
                    eval_node_indices,
                ),
        }

        print(
            f"{name.upper()} "
            "GAIN_NODES="
            f"{stats['count']} "
            "GAIN_FRAC="
            f"{stats['fraction']:.12g} "
            "MEAN_GAIN_M="
            f"{stats['mean_m']:.12g} "
            "P95_GAIN_M="
            f"{stats['p95_m']:.12g} "
            "MAX_GAIN_M="
            f"{stats['max_m']:.12g} "
            "FIRST_DELTA_M="
            f"{first_change:.12g}"
        )

    ideal = required["ideal"]
    fast = required["diagnostic_fast"]
    nominal = required["diagnostic_nominal"]
    stressed = required["diagnostic_stressed"]

    service_pairs = {
        "ideal_over_stressed":
            stressed - ideal,
        "fast_over_stressed":
            stressed - fast,
        "fast_over_nominal":
            nominal - fast,
        "nominal_over_stressed":
            stressed - nominal,
    }

    service_stats = {}

    print("=== NODE-LEVEL SERVICE DIFFERENCES ===")

    for name, delta in service_pairs.items():
        stats = positive_stats(delta)
        service_stats[name] = stats

        print(
            f"{name.upper()} "
            "NODES="
            f"{stats['count']} "
            "FRACTION="
            f"{stats['fraction']:.12g} "
            "MEAN_M="
            f"{stats['mean_m']:.12g} "
            "P95_M="
            f"{stats['p95_m']:.12g} "
            "MAX_M="
            f"{stats['max_m']:.12g}"
        )

    order_violations = {
        "ideal_gt_fast":
            int(np.count_nonzero(ideal > fast + TOL)),
        "fast_gt_nominal":
            int(np.count_nonzero(fast > nominal + TOL)),
        "nominal_gt_stressed":
            int(np.count_nonzero(nominal > stressed + TOL)),
        "stressed_gt_fallback":
            int(
                np.count_nonzero(
                    stressed > eval_fallback + TOL
                )
            ),
    }

    print("=== NODE-LEVEL ORDER ===")
    for name, count in order_violations.items():
        print(
            f"{name.upper()}="
            f"{count}"
        )

    d_min = float(
        p2b_cfg["state_domain"]["d_min"]
    )
    d_max = float(
        p2b_cfg["state_domain"]["d_max"]
    )

    cooperative_witnesses = make_witness_rows(
        eval_flat,
        ideal,
        eval_fallback,
        d_min,
        d_max,
        "ideal_safe_fallback_unsafe",
        max_rows=64,
    )

    service_witnesses = make_witness_rows(
        eval_flat,
        fast,
        stressed,
        d_min,
        d_max,
        "fast_safe_stressed_unsafe",
        max_rows=64,
    )

    print(
        "COOPERATIVE_EXACT_WITNESS_COUNT="
        f"{len(cooperative_witnesses)}"
    )
    print(
        "SERVICE_EXACT_WITNESS_COUNT="
        f"{len(service_witnesses)}"
    )

    ideal_gain_count = (
        profile_stats["ideal"][
            "fallback_improvement"
        ]["count"]
    )

    fast_stressed_count = (
        service_stats["fast_over_stressed"][
            "count"
        ]
    )

    checks = {
        "FIXED_POINT_MOVED_FROM_FALLBACK":
            ideal_gain_count > 0,
        "NODE_LEVEL_COOPERATIVE_GAIN_EXISTS":
            ideal_gain_count > 0,
        "NODE_LEVEL_SERVICE_EFFECT_EXISTS":
            fast_stressed_count > 0,
        "SERVICE_ORDER_NESTED_AT_NODES":
            sum(order_violations.values()) == 0,
        "COOPERATIVE_EXACT_WITNESS_EXISTS":
            len(cooperative_witnesses) > 0,
        "SERVICE_EXACT_WITNESS_EXISTS":
            len(service_witnesses) > 0,
        "ALL_PROFILES_CONVERGED":
            all(
                bool(result["converged"])
                for result in results.values()
            ),
    }

    status = (
        "PASS"
        if all(checks.values())
        else "FAIL"
    )

    if ideal_gain_count == 0:
        interpretation = (
            "RECURSION_ELIMINATES_FIRST_STEP_GAIN"
        )
    elif fast_stressed_count == 0:
        interpretation = (
            "COOPERATIVE_FIXED_POINT_GAIN_EXISTS_BUT_SERVICE_HORIZON_COLLAPSES"
        )
    else:
        interpretation = (
            "NODE_LEVEL_COOPERATIVE_AND_SERVICE_GAINS_CONFIRMED_QMC_MISSED_SMALL_SET"
        )

    stamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    coop_csv = (
        RESULTS_DIR
        /
        f"P3B1_R7D1_COOPERATIVE_WITNESSES_{stamp}.csv"
    )

    service_csv = (
        RESULTS_DIR
        /
        f"P3B1_R7D1_SERVICE_WITNESSES_{stamp}.csv"
    )

    write_csv(
        coop_csv,
        cooperative_witnesses,
    )
    write_csv(
        service_csv,
        service_witnesses,
    )

    output = {
        "schema":
            "SCV_P3B1_R7D1_NODE_LEVEL_FIXED_POINT_AUDIT_V1",
        "status":
            status,
        "timestamp_utc":
            stamp,
        "classification":
            (
                "node-level finite-abstraction fixed-point audit; "
                "not P6 certified"
            ),
        "checks":
            {k: bool(v) for k, v in checks.items()},
        "interpretation":
            interpretation,
        "selected_refinement_axes":
            data["selected_axes"],
        "evaluation_grid_nodes":
            int(len(eval_fallback)),
        "profile_stats":
            profile_stats,
        "service_difference_stats":
            service_stats,
        "service_order_violations":
            order_violations,
        "cooperative_exact_witness_rows":
            int(len(cooperative_witnesses)),
        "service_exact_witness_rows":
            int(len(service_witnesses)),
        "artifacts": {
            "cooperative_witness_csv":
                str(coop_csv),
            "service_witness_csv":
                str(service_csv),
        },
        "p6_certified":
            False,
        "maximal_continuous_kernel_claim":
            False,
    }

    result_path = (
        RESULTS_DIR
        /
        f"P3B1_R7D1_NODE_AUDIT_{stamp}.json"
    )
    latest_path = (
        RESULTS_DIR
        /
        "P3B1_R7D1_LATEST.json"
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
        f"P3B1_R7D1_MANIFEST_{stamp}.sha256"
    )

    manifest_files = [
        CFG_PATH,
        P1_CFG_PATH,
        P2B_CFG_PATH,
        P2C_CFG_PATH,
        P3A_CFG_PATH,
        R6_PATH,
        Path(__file__),
        result_path,
        coop_csv,
        service_csv,
    ]

    manifest = "\n".join(
        f"{sha256_file(path)}  {path}"
        for path in manifest_files
    ) + "\n"

    atomic_write(
        manifest_path,
        manifest,
    )

    print("=== P3-B1-R7-D1 DECISION ===")
    print(
        "INTERPRETATION="
        + interpretation
    )
    print(
        f"P3B1_R7D1_NODE_AUDIT={status}"
    )
    print(
        "P6_CERTIFIED=NO"
    )
    print(
        "MAXIMAL_CONTINUOUS_KERNEL_CLAIM=NO"
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
