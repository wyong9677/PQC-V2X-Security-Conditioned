from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

import p3b1_r7d1_node_level_audit as d1
import p3b1_r7d7_refined_fixed_point_recompute as d7

P1_CFG_PATH = ROOT / "01_config" / "p1_validation_v2.json"
P2B_CFG_PATH = ROOT / "01_config" / "p2b_hybrid_fallback_v1.json"
P2C_CFG_PATH = ROOT / "01_config" / "p2c_switching_guard_v1.json"
P3A_CFG_PATH = ROOT / "01_config" / "p3a_information_contract_v1.json"
D7B_CFG_PATH = (
    ROOT / "01_config" /
    "p3b1_r7d7b_second_level_refinement_protocol_v1.json"
)

D6C_PATH = ROOT / "04_results" / "P3B1_R7D6C_LATEST.json"
D7_PATH = ROOT / "04_results" / "P3B1_R7D7_LATEST.json"

RESULTS_DIR = ROOT / "04_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DIAGNOSTIC_NAMES = (
    "diagnostic_fast",
    "diagnostic_nominal",
    "diagnostic_stressed",
)


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


def base_bracket(axis: np.ndarray, x: float):
    axis = np.asarray(axis, dtype=float)
    lower = axis[axis < x]
    upper = axis[axis > x]
    if len(lower) == 0 or len(upper) == 0:
        raise RuntimeError(
            f"D7B_WITNESS_NOT_STRICTLY_INSIDE_BASE_AXIS x={x}"
        )
    lo = float(np.max(lower))
    hi = float(np.min(upper))
    if not (lo < x < hi):
        raise RuntimeError("D7B_INVALID_BASE_BRACKET")
    return lo, hi


def synth_row(
    center_row: dict,
    bar_a: float,
    synthetic_id: int,
    role: str,
    level: str,
    delta_bar_a: float,
):
    row = dict(center_row)
    row["bar_a"] = f"{float(bar_a):.17g}"
    row["unique_witness_id"] = str(int(synthetic_id))
    row["occurrence_id"] = str(int(synthetic_id))
    row["edge_id"] = str(int(center_row["edge_id"]))
    row["synthetic_role"] = role
    row["refinement_level"] = level
    row["source_unique_witness_id"] = str(
        int(center_row["unique_witness_id"])
    )
    row["delta_bar_a"] = f"{float(delta_bar_a):.17g}"
    return row


def enrich_center_row(center_row: dict, level: str):
    row = dict(center_row)
    row["synthetic_role"] = "center"
    row["refinement_level"] = level
    row["source_unique_witness_id"] = str(
        int(center_row["unique_witness_id"])
    )
    row["delta_bar_a"] = "0"
    return row


def strong_all_profiles(row: dict) -> bool:
    return all(
        bool(row[f"{name}_strong_q_dependent"])
        for name in DIAGNOSTIC_NAMES
    )


def spans_for(row: dict):
    return {
        name: float(row[f"{name}_q_span_m"])
        for name in DIAGNOSTIC_NAMES
    }


def structural_checks_only(scenario: dict) -> dict:
    excluded = {"ALL_AXIS_CANDIDATE_PROFILE_Q_SPANS_STRONG"}
    return {
        k: bool(v)
        for k, v in scenario["checks"].items()
        if k not in excluded
    }


def attach_input_metadata(scenario: dict, input_rows: list[dict]) -> None:
    """
    d7.run_axis_scenario() intentionally returns only the canonical candidate
    audit columns, so D7B-specific synthetic metadata must be reattached here
    using the unique_witness_id that *is* preserved by the D7 helper.
    """
    meta_by_id = {
        int(row["unique_witness_id"]): {
            "synthetic_role": row.get("synthetic_role", "center"),
            "refinement_level": row.get("refinement_level", ""),
            "source_unique_witness_id": int(
                row.get("source_unique_witness_id", row["unique_witness_id"])
            ),
            "delta_bar_a": float(row.get("delta_bar_a", 0.0)),
        }
        for row in input_rows
    }

    seen = set()
    for row in scenario["candidate_rows"]:
        uid = int(row["unique_witness_id"])
        if uid not in meta_by_id:
            raise RuntimeError(
                f"P3B1_R7D7B_METADATA_ID_MISSING uid={uid}"
            )
        if uid in seen:
            raise RuntimeError(
                f"P3B1_R7D7B_DUPLICATE_CANDIDATE_ID uid={uid}"
            )
        seen.add(uid)
        row.update(meta_by_id[uid])

    missing = sorted(set(meta_by_id) - seen)
    if missing:
        raise RuntimeError(
            "P3B1_R7D7B_METADATA_OUTPUT_MISSING_IDS="
            + ",".join(str(x) for x in missing)
        )


def map_candidate_rows(scenario: dict):
    mapping = defaultdict(list)
    for row in scenario["candidate_rows"]:
        src = int(row["source_unique_witness_id"])
        role = row["synthetic_role"]
        mapping[(src, role)].append(row)
    return mapping


def main() -> int:
    protocol = load_json(D7B_CFG_PATH)
    tol = float(protocol["numeric_tolerance_m"])
    min_multiple = float(
        protocol["fixed_point_candidate_min_multiple_of_tolerance"]
    )
    denom_a = int(protocol["level_a_cell_fraction_denominator"])
    denom_b = int(protocol["level_b_cell_fraction_denominator"])
    guard_multiple = float(
        protocol["numerical_precertificate_guard_multiple_of_tolerance"]
    )
    guard = guard_multiple * tol

    d6c_latest = load_json(D6C_PATH)
    d7_latest = load_json(D7_PATH)

    if d6c_latest.get("status") != "PASS":
        raise RuntimeError("P3B1_R7D7B_D6C_UPSTREAM_FAIL")
    if d7_latest.get("status") != "PASS":
        raise RuntimeError("P3B1_R7D7B_D7_UPSTREAM_FAIL")

    expected_d7 = (
        "Q_DEPENDENT_KERNEL_COMPONENT_PERSISTS_AFTER_TRUE_AXIS_REFINED_FIXED_POINT_RECOMPUTATION"
    )
    if d7_latest.get("interpretation") != expected_d7:
        raise RuntimeError(
            "P3B1_R7D7B_UNEXPECTED_D7_INTERPRETATION="
            + str(d7_latest.get("interpretation"))
        )
    if not d7_latest["claims"].get(
        "refined_finite_abstraction_q_dependent_kernel_candidate",
        False,
    ):
        raise RuntimeError("P3B1_R7D7B_D7_CANDIDATE_MISSING")

    d6c_occurrence_csv = Path(
        d6c_latest["artifacts"]["occurrence_csv"]
    )
    d6c_unique_csv = Path(
        d6c_latest["artifacts"]["unique_witness_csv"]
    )
    d7_candidate_csv = Path(
        d7_latest["artifacts"]["candidate_csv"]
    )

    occurrence_rows = read_csv(d6c_occurrence_csv)
    unique_rows = read_csv(d6c_unique_csv)
    d7_candidate_rows = read_csv(d7_candidate_csv)

    open_ids = {
        int(row["unique_witness_id"])
        for row in unique_rows
        if row["local_classification"]
        == "OPEN_NEIGHBORHOOD_KERNEL_WITNESS_CANDIDATE"
    }

    centers = [
        row for row in occurrence_rows
        if int(row["unique_witness_id"]) in open_ids
    ]

    if not centers:
        raise RuntimeError("P3B1_R7D7B_NO_CENTER_WITNESSES")

    if any(row["axis_name"] != "bar_a" for row in centers):
        raise RuntimeError(
            "P3B1_R7D7B_NON_BAR_A_WITNESS_REQUIRES_GENERALIZED_PROTOCOL"
        )

    expected_centers = int(
        d7_latest["aggregate"]["candidate_rows_recomputed"]
    )
    if len(centers) != expected_centers:
        raise RuntimeError(
            f"P3B1_R7D7B_CENTER_COUNT_MISMATCH "
            f"{len(centers)} != {expected_centers}"
        )

    d7_center_by_uid = {
        int(row["unique_witness_id"]): row
        for row in d7_candidate_rows
    }

    if set(d7_center_by_uid) != open_ids:
        raise RuntimeError(
            "P3B1_R7D7B_D7_CENTER_ID_SET_MISMATCH"
        )

    data = d1.reconstruct()
    base_bar_a = np.asarray(data["eval_axes"][3], dtype=float)

    p1_cfg = load_json(P1_CFG_PATH)
    p2b_cfg = load_json(P2B_CFG_PATH)
    p2c_cfg = load_json(P2C_CFG_PATH)
    p3a_cfg = load_json(P3A_CFG_PATH)

    # Build two nested, previously untested micro-refinement levels.
    level_a_rows = []
    level_b_rows = []
    level_a_values = set()
    level_b_values = set()

    geometry_rows = []

    for center in centers:
        uid = int(center["unique_witness_id"])
        x = float(center["bar_a"])
        lo, hi = base_bracket(base_bar_a, x)
        width = hi - lo

        da = width / denom_a
        db = width / denom_b

        xa_minus = x - da
        xa_plus = x + da
        xb_minus = x - db
        xb_plus = x + db

        if not (lo < xa_minus < x < xa_plus < hi):
            raise RuntimeError(
                f"P3B1_R7D7B_LEVEL_A_OUTSIDE_BRACKET uid={uid}"
            )
        if not (lo < xb_minus < x < xb_plus < hi):
            raise RuntimeError(
                f"P3B1_R7D7B_LEVEL_B_OUTSIDE_BRACKET uid={uid}"
            )

        # Unique synthetic numeric ids; D7 helper requires int-convertible ids.
        base_id = uid * 1000 + 700000

        center_a = enrich_center_row(center, "A")
        minus_a = synth_row(
            center, xa_minus, base_id + 11,
            "minus_A", "A", -da
        )
        plus_a = synth_row(
            center, xa_plus, base_id + 12,
            "plus_A", "A", da
        )

        level_a_rows.extend([center_a, minus_a, plus_a])

        center_b = enrich_center_row(center, "B")
        minus_a_b = synth_row(
            center, xa_minus, base_id + 21,
            "minus_A", "B", -da
        )
        plus_a_b = synth_row(
            center, xa_plus, base_id + 22,
            "plus_A", "B", da
        )
        minus_b = synth_row(
            center, xb_minus, base_id + 23,
            "minus_B", "B", -db
        )
        plus_b = synth_row(
            center, xb_plus, base_id + 24,
            "plus_B", "B", db
        )

        level_b_rows.extend(
            [center_b, minus_a_b, plus_a_b, minus_b, plus_b]
        )

        level_a_values.update([x, xa_minus, xa_plus])
        level_b_values.update(
            [x, xa_minus, xa_plus, xb_minus, xb_plus]
        )

        geometry_rows.append(
            {
                "unique_witness_id": uid,
                "base_bar_a_lo": lo,
                "base_bar_a_hi": hi,
                "base_cell_width": width,
                "center_bar_a": x,
                "level_a_delta_bar_a": da,
                "level_a_minus_bar_a": xa_minus,
                "level_a_plus_bar_a": xa_plus,
                "level_b_delta_bar_a": db,
                "level_b_minus_bar_a": xb_minus,
                "level_b_plus_bar_a": xb_plus,
            }
        )

    print("=== P3-B1-R7-D7B SECOND-LEVEL REFINEMENT STABILITY ===")
    print(f"D7_CENTER_WITNESSES={len(centers)}")
    print(f"UNIQUE_CENTER_BAR_A_VALUES={len(set(float(r['bar_a']) for r in centers))}")
    print(f"LEVEL_A_INSERTED_BAR_A_VALUES={len(level_a_values)}")
    print(f"LEVEL_B_INSERTED_BAR_A_VALUES={len(level_b_values)}")
    print(f"LEVEL_A_TEST_ROWS={len(level_a_rows)}")
    print(f"LEVEL_B_TEST_ROWS={len(level_b_rows)}")

    scenario_a = d7.run_axis_scenario(
        "bar_a",
        sorted(level_a_values),
        level_a_rows,
        data,
        p1_cfg,
        p2b_cfg,
        p2c_cfg,
        p3a_cfg,
        protocol,
    )

    scenario_b = d7.run_axis_scenario(
        "bar_a",
        sorted(level_b_values),
        level_b_rows,
        data,
        p1_cfg,
        p2b_cfg,
        p2c_cfg,
        p3a_cfg,
        protocol,
    )

    # R1 fix: restore D7B-only synthetic metadata stripped by the shared
    # D7 scenario helper before role/source mapping.
    attach_input_metadata(scenario_a, level_a_rows)
    attach_input_metadata(scenario_b, level_b_rows)

    structural_a = structural_checks_only(scenario_a)
    structural_b = structural_checks_only(scenario_b)

    structural_pass_a = all(structural_a.values())
    structural_pass_b = all(structural_b.values())

    map_a = map_candidate_rows(scenario_a)
    map_b = map_candidate_rows(scenario_b)

    witness_rows = []
    center_all_levels_strong = 0
    two_sided_a_strong = 0
    two_sided_b_strong = 0
    all_nested_strong = 0

    min_precert_lower_bound = math.inf
    min_precert_multiple = math.inf
    max_center_relative_drift_a_to_b = 0.0
    max_center_relative_drift_d7_to_b = 0.0

    for center in centers:
        uid = int(center["unique_witness_id"])

        # One row per role is expected.
        ca = map_a[(uid, "center")][0]
        ma = map_a[(uid, "minus_A")][0]
        pa = map_a[(uid, "plus_A")][0]

        cb = map_b[(uid, "center")][0]
        ma_b = map_b[(uid, "minus_A")][0]
        pa_b = map_b[(uid, "plus_A")][0]
        mb = map_b[(uid, "minus_B")][0]
        pb = map_b[(uid, "plus_B")][0]

        d7c = d7_center_by_uid[uid]

        center_a_strong = strong_all_profiles(ca)
        center_b_strong = strong_all_profiles(cb)
        center_d7_strong = all(
            str(d7c[f"{name}_strong_q_dependent"]).lower()
            in ("true", "1")
            for name in DIAGNOSTIC_NAMES
        )

        level_a_two_sided = (
            strong_all_profiles(ma)
            and strong_all_profiles(pa)
        )
        level_b_two_sided = (
            strong_all_profiles(mb)
            and strong_all_profiles(pb)
        )
        level_a_replayed_in_b = (
            strong_all_profiles(ma_b)
            and strong_all_profiles(pa_b)
        )

        center_all_levels = (
            center_d7_strong
            and center_a_strong
            and center_b_strong
        )

        nested_strong = (
            center_all_levels
            and level_a_two_sided
            and level_a_replayed_in_b
            and level_b_two_sided
        )

        center_all_levels_strong += int(center_all_levels)
        two_sided_a_strong += int(level_a_two_sided)
        two_sided_b_strong += int(level_b_two_sided)
        all_nested_strong += int(nested_strong)

        row = {
            "unique_witness_id": uid,
            "center_all_levels_strong": bool(center_all_levels),
            "level_a_two_sided_strong": bool(level_a_two_sided),
            "level_a_flanks_replayed_strong_in_level_b":
                bool(level_a_replayed_in_b),
            "level_b_two_sided_strong": bool(level_b_two_sided),
            "all_nested_refinement_points_strong":
                bool(nested_strong),
        }

        spans_for_certificate = []

        for name in DIAGNOSTIC_NAMES:
            d7_span = float(d7c[f"{name}_q_span_m"])
            a_center = float(ca[f"{name}_q_span_m"])
            b_center = float(cb[f"{name}_q_span_m"])

            a_minus = float(ma[f"{name}_q_span_m"])
            a_plus = float(pa[f"{name}_q_span_m"])
            b_minus_a = float(ma_b[f"{name}_q_span_m"])
            b_plus_a = float(pa_b[f"{name}_q_span_m"])
            b_minus = float(mb[f"{name}_q_span_m"])
            b_plus = float(pb[f"{name}_q_span_m"])

            denom_ab = max(abs(a_center), abs(b_center), tol)
            drift_ab = abs(b_center - a_center) / denom_ab

            denom_d7b = max(abs(d7_span), abs(b_center), tol)
            drift_d7b = abs(b_center - d7_span) / denom_d7b

            max_center_relative_drift_a_to_b = max(
                max_center_relative_drift_a_to_b,
                drift_ab,
            )
            max_center_relative_drift_d7_to_b = max(
                max_center_relative_drift_d7_to_b,
                drift_d7b,
            )

            local_min_span = min(
                d7_span,
                a_center,
                b_center,
                a_minus,
                a_plus,
                b_minus_a,
                b_plus_a,
                b_minus,
                b_plus,
            )

            # Outward numerical pre-certificate: subtract a deliberately
            # conservative 2-sided guard from the smallest observed nested
            # refined q-span. This is not a rational/Farkas proof artifact.
            precert_lower = max(
                0.0,
                local_min_span - 2.0 * guard,
            )
            precert_multiple = precert_lower / tol

            spans_for_certificate.append(precert_lower)
            min_precert_lower_bound = min(
                min_precert_lower_bound,
                precert_lower,
            )
            min_precert_multiple = min(
                min_precert_multiple,
                precert_multiple,
            )

            row[f"{name}_d7_center_span_m"] = d7_span
            row[f"{name}_level_a_center_span_m"] = a_center
            row[f"{name}_level_b_center_span_m"] = b_center
            row[f"{name}_level_a_minus_span_m"] = a_minus
            row[f"{name}_level_a_plus_span_m"] = a_plus
            row[f"{name}_level_b_minus_A_span_m"] = b_minus_a
            row[f"{name}_level_b_plus_A_span_m"] = b_plus_a
            row[f"{name}_level_b_minus_B_span_m"] = b_minus
            row[f"{name}_level_b_plus_B_span_m"] = b_plus
            row[f"{name}_center_relative_drift_A_to_B"] = drift_ab
            row[f"{name}_center_relative_drift_D7_to_B"] = drift_d7b
            row[f"{name}_precertificate_lower_bound_m"] = precert_lower
            row[f"{name}_precertificate_lower_bound_multiple_of_tol"] = (
                precert_multiple
            )

        row["minimum_profile_precertificate_lower_bound_m"] = min(
            spans_for_certificate
        )
        row[
            "minimum_profile_precertificate_lower_bound_multiple_of_tol"
        ] = min(spans_for_certificate) / tol

        witness_rows.append(row)

    if min_precert_lower_bound is math.inf:
        min_precert_lower_bound = 0.0
    if min_precert_multiple is math.inf:
        min_precert_multiple = 0.0

    # Original-grid node q-dependence after each nested refinement is
    # scientifically informative but not forced to remain zero.
    def base_qdep(scenario):
        rows = {
            row["profile"]: row
            for row in scenario["profile_rows"]
        }
        return {
            name: int(
                rows[name][
                    "original_base_nodes_q_dependent_after_refinement"
                ]
            )
            for name in DIAGNOSTIC_NAMES
        }

    base_a = base_qdep(scenario_a)
    base_b = base_qdep(scenario_b)

    fixed_point_stability_candidate = bool(
        structural_pass_a
        and structural_pass_b
        and center_all_levels_strong == len(centers)
        and two_sided_a_strong == len(centers)
        and two_sided_b_strong == len(centers)
        and all_nested_strong == len(centers)
        and min_precert_multiple >= min_multiple
    )

    integrity_checks = {
        "D6C_UPSTREAM_PASS":
            d6c_latest.get("status") == "PASS",
        "D7_UPSTREAM_PASS":
            d7_latest.get("status") == "PASS",
        "D7_POSITIVE_REFINED_FIXED_POINT_INTERPRETATION":
            d7_latest.get("interpretation") == expected_d7,
        "D7_CENTER_COUNT_REPRODUCED":
            len(centers) == expected_centers,
        "ALL_WITNESSES_BAR_A_AXIS":
            all(row["axis_name"] == "bar_a" for row in centers),
        "LEVEL_A_STRUCTURAL_GATES":
            bool(structural_pass_a),
        "LEVEL_B_STRUCTURAL_GATES":
            bool(structural_pass_b),
        "LEVEL_B_STRICTLY_REFINES_LEVEL_A":
            set(level_a_values).issubset(level_b_values)
            and len(level_b_values) > len(level_a_values),
    }

    status = "PASS" if all(integrity_checks.values()) else "FAIL"

    if not structural_pass_a or not structural_pass_b:
        interpretation = (
            "SECOND_LEVEL_REFINEMENT_STRUCTURAL_GATE_FAILED"
        )
        next_step = (
            "REPAIR_REFINEMENT_STRUCTURE_BEFORE_ANY_KERNEL_PROMOTION"
        )
    elif center_all_levels_strong != len(centers):
        interpretation = (
            "D7_KERNEL_CENTERS_DO_NOT_ALL_PERSIST_ACROSS_NESTED_FIXED_POINT_REFINEMENTS"
        )
        next_step = (
            "LOCALIZE_NONPERSISTENT_CENTERS_AND_DO_NOT_PROMOTE_KERNEL_CLAIM"
        )
    elif (
        two_sided_a_strong != len(centers)
        or two_sided_b_strong != len(centers)
        or all_nested_strong != len(centers)
    ):
        interpretation = (
            "CENTER_Q_DEPENDENCE_PERSISTS_BUT_TWO_SIDED_FIXED_POINT_NEIGHBORHOOD_IS_NOT_UNIFORMLY_STABLE"
        )
        next_step = (
            "BOUND_THE_Q_DEPENDENT_LAYER_WIDTH_BEFORE_CERTIFICATE_PROMOTION"
        )
    elif min_precert_multiple < min_multiple:
        interpretation = (
            "NESTED_REFINEMENT_PERSISTS_BUT_OUTWARD_NUMERICAL_MARGIN_IS_INSUFFICIENT"
        )
        next_step = (
            "RUN_HIGHER_PRECISION_OR_RATIONAL_RECOMPUTATION_BEFORE_CERTIFICATE"
        )
    else:
        interpretation = (
            "Q_DEPENDENT_KERNEL_LAYER_IS_STABLE_UNDER_TWO_NESTED_TRUE_FIXED_POINT_REFINEMENTS_WITH_POSITIVE_OUTWARD_MARGIN"
        )
        next_step = (
            "RUN_P3B1_R7D8_RATIONAL_OUTWARD_CERTIFICATE_AND_INDEPENDENT_CHECKER_AUDIT"
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    witness_csv = (
        RESULTS_DIR
        / f"P3B1_R7D7B_NESTED_WITNESS_STABILITY_{stamp}.csv"
    )
    geometry_csv = (
        RESULTS_DIR
        / f"P3B1_R7D7B_REFINEMENT_GEOMETRY_{stamp}.csv"
    )
    profile_csv = (
        RESULTS_DIR
        / f"P3B1_R7D7B_PROFILE_AUDIT_{stamp}.csv"
    )
    convergence_csv = (
        RESULTS_DIR
        / f"P3B1_R7D7B_CONVERGENCE_{stamp}.csv"
    )

    write_csv(witness_csv, witness_rows)
    write_csv(geometry_csv, geometry_rows)
    write_csv(
        profile_csv,
        [
            {"refinement_level": "A", **row}
            for row in scenario_a["profile_rows"]
        ]
        +
        [
            {"refinement_level": "B", **row}
            for row in scenario_b["profile_rows"]
        ],
    )
    write_csv(
        convergence_csv,
        [
            {"refinement_level": "A", **row}
            for row in scenario_a["convergence_rows"]
        ]
        +
        [
            {"refinement_level": "B", **row}
            for row in scenario_b["convergence_rows"]
        ],
    )

    output = {
        "schema":
            "SCV_P3B1_R7D7B_NESTED_FIXED_POINT_STABILITY_R1",
        "status":
            status,
        "timestamp_utc":
            stamp,
        "classification":
            (
                "two-level nested local bar_a grid refinement with full "
                "frozen-halo fixed-point re-solves and numerical outward "
                "pre-certificate margin; finite-abstraction evidence only"
            ),
        "integrity_checks":
            {k: bool(v) for k, v in integrity_checks.items()},
        "interpretation":
            interpretation,
        "recommended_next_step":
            next_step,
        "protocol":
            protocol,
        "level_a": {
            "inserted_bar_a_values": sorted(level_a_values),
            "eval_nodes": int(scenario_a["eval_nodes"]),
            "lookup_nodes": int(scenario_a["lookup_nodes"]),
            "structural_checks": structural_a,
            "scenario_status_including_candidate_gate":
                scenario_a["status"],
            "minimum_candidate_q_span_multiple_of_tol":
                float(
                    scenario_a[
                        "minimum_candidate_q_span_multiple_of_tol"
                    ]
                ),
            "base_node_q_dependence_after_refinement":
                base_a,
        },
        "level_b": {
            "inserted_bar_a_values": sorted(level_b_values),
            "eval_nodes": int(scenario_b["eval_nodes"]),
            "lookup_nodes": int(scenario_b["lookup_nodes"]),
            "structural_checks": structural_b,
            "scenario_status_including_candidate_gate":
                scenario_b["status"],
            "minimum_candidate_q_span_multiple_of_tol":
                float(
                    scenario_b[
                        "minimum_candidate_q_span_multiple_of_tol"
                    ]
                ),
            "base_node_q_dependence_after_refinement":
                base_b,
        },
        "aggregate": {
            "d7_center_witnesses": int(len(centers)),
            "unique_center_bar_a_values":
                int(len(set(float(r["bar_a"]) for r in centers))),
            "center_strong_all_three_fixed_point_levels":
                int(center_all_levels_strong),
            "level_a_two_sided_strong_witnesses":
                int(two_sided_a_strong),
            "level_b_two_sided_strong_witnesses":
                int(two_sided_b_strong),
            "all_nested_points_strong_witnesses":
                int(all_nested_strong),
            "minimum_numerical_precertificate_lower_bound_m":
                float(min_precert_lower_bound),
            "minimum_numerical_precertificate_lower_bound_multiple_of_tol":
                float(min_precert_multiple),
            "numerical_precertificate_guard_m":
                float(guard),
            "max_center_relative_drift_A_to_B":
                float(max_center_relative_drift_a_to_b),
            "max_center_relative_drift_D7_to_B":
                float(max_center_relative_drift_d7_to_b),
        },
        "claims": {
            "second_level_refined_fixed_point_stability_candidate":
                bool(fixed_point_stability_candidate),
            "numerical_outward_precertificate_candidate":
                bool(
                    fixed_point_stability_candidate
                    and min_precert_multiple >= min_multiple
                ),
            "scientific_kernel_claim_authorized":
                False,
            "maximal_continuous_kernel_claim_authorized":
                False,
            "implementation_refinement_claim_authorized":
                False,
            "rational_farkas_certificate_authorized":
                False,
            "p6_certified":
                False,
            "reason":
                (
                    "R7-D7B can establish nested finite-grid fixed-point "
                    "stability and a conservative numerical margin only. "
                    "Rational/outward data generation plus an independent "
                    "checker remain required before certificate-backed "
                    "manuscript promotion."
                ),
        },
        "artifacts": {
            "witness_csv": str(witness_csv),
            "geometry_csv": str(geometry_csv),
            "profile_csv": str(profile_csv),
            "convergence_csv": str(convergence_csv),
        },
    }

    result_path = (
        RESULTS_DIR
        / f"P3B1_R7D7B_SECOND_LEVEL_REFINEMENT_{stamp}.json"
    )
    latest_path = RESULTS_DIR / "P3B1_R7D7B_LATEST.json"
    manifest_path = (
        RESULTS_DIR
        / f"P3B1_R7D7B_MANIFEST_{stamp}.sha256"
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
        D7B_CFG_PATH,
        D6C_PATH,
        D7_PATH,
        d6c_occurrence_csv,
        d6c_unique_csv,
        d7_candidate_csv,
        Path(d1.__file__),
        Path(d7.__file__),
        Path(__file__),
        result_path,
        witness_csv,
        geometry_csv,
        profile_csv,
        convergence_csv,
    ]

    manifest = "\n".join(
        f"{sha256_file(path)}  {path}"
        for path in manifest_files
    ) + "\n"
    atomic_write(manifest_path, manifest)

    print("=== P3-B1-R7-D7B DECISION ===")
    print(f"CENTER_WITNESSES={len(centers)}")
    print(
        "CENTER_STRONG_ALL_THREE_FIXED_POINT_LEVELS="
        f"{center_all_levels_strong}"
    )
    print(
        "LEVEL_A_TWO_SIDED_STRONG_WITNESSES="
        f"{two_sided_a_strong}"
    )
    print(
        "LEVEL_B_TWO_SIDED_STRONG_WITNESSES="
        f"{two_sided_b_strong}"
    )
    print(
        "ALL_NESTED_POINTS_STRONG_WITNESSES="
        f"{all_nested_strong}"
    )
    print(
        "MIN_PRECERTIFICATE_LOWER_BOUND_M="
        f"{min_precert_lower_bound:.12g}"
    )
    print(
        "MIN_PRECERTIFICATE_LOWER_BOUND_MULTIPLE_OF_TOL="
        f"{min_precert_multiple:.12g}"
    )
    print(
        "MAX_CENTER_RELATIVE_DRIFT_A_TO_B="
        f"{max_center_relative_drift_a_to_b:.12g}"
    )
    print(
        "MAX_CENTER_RELATIVE_DRIFT_D7_TO_B="
        f"{max_center_relative_drift_d7_to_b:.12g}"
    )
    print(f"INTERPRETATION={interpretation}")
    print(f"RECOMMENDED_NEXT_STEP={next_step}")
    print(f"P3B1_R7D7B_SECOND_LEVEL_REFINEMENT_AUDIT={status}")
    print(
        "SECOND_LEVEL_FIXED_POINT_STABILITY_CANDIDATE="
        + ("YES" if fixed_point_stability_candidate else "NO")
    )
    print(
        "NUMERICAL_OUTWARD_PRECERTIFICATE_CANDIDATE="
        + (
            "YES"
            if output["claims"][
                "numerical_outward_precertificate_candidate"
            ]
            else "NO"
        )
    )
    print("SCIENTIFIC_KERNEL_CLAIM_AUTHORIZED=NO")
    print("RATIONAL_FARKAS_CERTIFICATE_AUTHORIZED=NO")
    print("IMPLEMENTATION_REFINEMENT_CLAIM=NO")
    print(f"RESULT_JSON={result_path}")
    print(f"MANIFEST={manifest_path}")

    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
