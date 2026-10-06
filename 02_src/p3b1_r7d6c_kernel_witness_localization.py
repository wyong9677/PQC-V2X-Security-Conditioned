from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

import p3b1_r7d1_node_level_audit as d1
import p3b1_r7d3_fallback_clipping_audit as d3
import p3b1_r7d6b_adaptive_boundary_bisection as d6b

P1_CFG_PATH = ROOT / "01_config" / "p1_validation_v2.json"
P2B_CFG_PATH = ROOT / "01_config" / "p2b_hybrid_fallback_v1.json"
P2C_CFG_PATH = ROOT / "01_config" / "p2c_switching_guard_v1.json"
P3A_CFG_PATH = ROOT / "01_config" / "p3a_information_contract_v1.json"
D6C_CFG_PATH = (
    ROOT / "01_config" / "p3b1_r7d6c_kernel_witness_protocol_v1.json"
)

D4_PATH = ROOT / "04_results" / "P3B1_R7D4_LATEST.json"
D5_PATH = ROOT / "04_results" / "P3B1_R7D5_LATEST.json"
D6_PATH = ROOT / "04_results" / "P3B1_R7D6_LATEST.json"
D6B_PATH = ROOT / "04_results" / "P3B1_R7D6B_LATEST.json"

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


def quantized_key(x: np.ndarray, atol: float):
    return tuple(int(np.rint(float(v) / atol)) for v in x)


def profile_kernel_stats(
    raw: np.ndarray,
    clipped: np.ndarray,
    tol: float,
):
    raw_span = np.max(raw, axis=1) - np.min(raw, axis=1)
    kernel_component = np.max(clipped, axis=1) - np.min(clipped, axis=1)
    switch_width = np.maximum(raw_span - kernel_component, 0.0)

    q_min = np.argmin(clipped, axis=1)
    q_max = np.argmax(clipped, axis=1)

    return {
        "raw_span": raw_span,
        "kernel_component": kernel_component,
        "switch_width": switch_width,
        "kernel_mask": kernel_component > tol,
        "q_min": q_min,
        "q_max": q_max,
    }


def build_transition_for_points(
    points: np.ndarray,
    data,
    p1_cfg,
    p2b_cfg,
    p2c_cfg,
    p3a_cfg,
):
    flat = [
        np.asarray(points[:, k], dtype=float)
        for k in range(points.shape[1])
    ]

    transition_data = d3.r7.build_transition_data_to_lookup(
        data["eval_cfg"],
        p1_cfg,
        p2b_cfg,
        p2c_cfg,
        p3a_cfg,
        flat,
        data["lookup_axes"],
    )

    completion_invalid = sum(
        int(np.count_nonzero(~tr["completion_valid"]))
        for tr in transition_data["transitions"]
    )
    defer_invalid = sum(
        int(np.count_nonzero(~tr["defer_valid"]))
        for tr in transition_data["transitions"]
    )

    fallback = np.asarray(
        transition_data["fallback_required"],
        dtype=float,
    )

    return flat, transition_data, fallback, completion_invalid, defer_invalid


def evaluate_points(
    points: np.ndarray,
    data,
    p1_cfg,
    p2b_cfg,
    p2c_cfg,
    p3a_cfg,
    diagnostic_names,
    tol,
):
    (
        flat,
        transition_data,
        fallback,
        completion_invalid,
        defer_invalid,
    ) = build_transition_for_points(
        points,
        data,
        p1_cfg,
        p2b_cfg,
        p2c_cfg,
        p3a_cfg,
    )

    profiles = {}

    for name in diagnostic_names:
        raw, clipped, branch, action = d3.raw_operator_slices(
            data["results"][name],
            data["lookup_shape"],
            transition_data,
            fallback,
        )

        stats = profile_kernel_stats(raw, clipped, tol)

        selected, completion_idx, defer_idx = (
            d6b.selected_lookup_signature(
                action,
                transition_data,
            )
        )

        profiles[name] = {
            "raw": raw,
            "clipped": clipped,
            "branch": branch,
            "action": action,
            "stats": stats,
            "selected_action_q0": selected,
            "completion_index_q0": completion_idx,
            "defer_index_q0": defer_idx,
            "action_q_change": d6b.q_label_change(action),
            "branch_q_change": d6b.q_label_change(branch),
        }

    return {
        "flat": flat,
        "transition_data": transition_data,
        "fallback": fallback,
        "completion_invalid": completion_invalid,
        "defer_invalid": defer_invalid,
        "profiles": profiles,
    }


def local_classification(
    center_positive: bool,
    dyadic_pairs: dict,
    ulp_pair: tuple[bool, bool],
):
    if not center_positive:
        return "CENTER_NOT_REPRODUCED"

    if not all(ulp_pair):
        return "MACHINE_ULP_SENSITIVE"

    smallest = min(dyadic_pairs.keys())
    left, right = dyadic_pairs[smallest]

    if left and right:
        return "OPEN_NEIGHBORHOOD_KERNEL_WITNESS_CANDIDATE"
    if left or right:
        return "ONE_SIDED_BOUNDARY_KERNEL_WITNESS_CANDIDATE"
    return "ISOLATED_AT_TESTED_DYADIC_SCALES"


def main() -> int:
    cfg = load_json(D6C_CFG_PATH)
    tol = float(cfg["numeric_tolerance_m"])
    coord_tol = float(cfg["coordinate_dedup_tolerance"])
    subdivisions = int(cfg["stage1_subdivisions"])
    offsets = [float(x) for x in cfg["local_dyadic_offsets_t"]]
    min_center_multiple = float(
        cfg["minimum_center_kernel_multiple_of_tolerance"]
    )

    d4_latest = load_json(D4_PATH)
    d5_latest = load_json(D5_PATH)
    d6_latest = load_json(D6_PATH)
    d6b_latest = load_json(D6B_PATH)

    for label, obj in (
        ("D4", d4_latest),
        ("D5", d5_latest),
        ("D6", d6_latest),
        ("D6B", d6b_latest),
    ):
        if obj.get("status") != "PASS":
            raise RuntimeError(f"P3B1_R7D6C_{label}_UPSTREAM_FAIL")

    expected = "ADAPTIVE_BOUNDARY_REPLAY_REVEALS_Q_DEPENDENT_KERNEL_COMPONENT"
    if d6b_latest.get("interpretation") != expected:
        raise RuntimeError(
            "P3B1_R7D6C_UNEXPECTED_D6B_INTERPRETATION="
            + str(d6b_latest.get("interpretation"))
        )

    expected_occurrences = int(
        d6b_latest["aggregate"]["all_kernel_component_nodes"]
    )

    data = d1.reconstruct()

    p1_cfg = load_json(P1_CFG_PATH)
    p2b_cfg = load_json(P2B_CFG_PATH)
    p2c_cfg = load_json(P2C_CFG_PATH)
    p3a_cfg = load_json(P3A_CFG_PATH)

    diagnostic_names = (
        "diagnostic_fast",
        "diagnostic_nominal",
        "diagnostic_stressed",
    )

    state = np.column_stack(
        [np.asarray(x, dtype=float) for x in data["eval_flat"]]
    )

    base_transition = d3.build_transition_data(data)
    eval_fallback = np.asarray(data["eval_fallback"], dtype=float)

    raw, clipped, _, _ = d3.raw_operator_slices(
        data["results"]["diagnostic_fast"],
        data["lookup_shape"],
        base_transition,
        eval_fallback,
    )

    base_geom = d6b.switch_geometry(raw, clipped, tol)
    base_switch = base_geom["switch_mask"]

    edges = d6b.enumerate_boundary_edges(
        base_switch,
        tuple(int(x) for x in data["eval_shape"]),
    )

    print("=== P3-B1-R7-D6C KERNEL WITNESS LOCALIZATION ===")
    print(f"BOUNDARY_EDGES={len(edges)}")
    print(f"STAGE1_SUBDIVISIONS={subdivisions}")
    print(f"EXPECTED_KERNEL_POSITIVE_OCCURRENCES={expected_occurrences}")

    occurrence_rows = []
    coordinate_to_occurrences = defaultdict(list)

    completion_invalid = 0
    defer_invalid = 0

    batch_size = 64

    for start in range(0, len(edges), batch_size):
        batch = edges[start:start + batch_size]

        result = d6b.evaluate_batch(
            batch,
            state,
            subdivisions,
            data,
            p1_cfg,
            p2b_cfg,
            p2c_cfg,
            p3a_cfg,
            diagnostic_names,
            tol,
        )

        completion_invalid += result["completion_invalid"]
        defer_invalid += result["defer_invalid"]

        npt = subdivisions + 1

        union_kernel = np.zeros(
            len(result["matrix"]),
            dtype=bool,
        )
        profile_stats = {}

        for name in diagnostic_names:
            g = result["geoms"][name]
            # d6b geometry already stores kernel_component.
            kmask = g["kernel_component"] > tol
            union_kernel |= kmask
            profile_stats[name] = {
                "kernel_component": g["kernel_component"],
                "raw_span": g["raw_span"],
                "switch_width": g["switch_width"],
                "kernel_mask": kmask,
            }

        for p in np.flatnonzero(union_kernel):
            local = int(p // npt)
            k = int(p % npt)
            edge = batch[local]
            x = np.asarray(result["matrix"][p], dtype=float)

            key = quantized_key(x, coord_tol)
            occurrence_id = len(occurrence_rows)

            row = {
                "occurrence_id": occurrence_id,
                "edge_id": int(edge["edge_id"]),
                "axis": int(edge["axis"]),
                "axis_name": AXIS_NAMES[int(edge["axis"])],
                "witness_node": int(edge["witness_node"]),
                "nonwitness_node": int(edge["nonwitness_node"]),
                "k": k,
                "t": float(k / subdivisions),
                "v_f": float(x[0]),
                "v_p": float(x[1]),
                "a_f": float(x[2]),
                "bar_a": float(x[3]),
                "bar_u": float(x[4]),
                "age": float(x[5]),
                "fallback_required_m": float(result["fallback"][p]),
            }

            positive_profiles = []

            for name in diagnostic_names:
                ps = profile_stats[name]
                positive = bool(ps["kernel_mask"][p])
                if positive:
                    positive_profiles.append(name)

                row[f"{name}_kernel_positive"] = positive
                row[f"{name}_kernel_component_m"] = float(
                    ps["kernel_component"][p]
                )
                row[f"{name}_raw_span_m"] = float(
                    ps["raw_span"][p]
                )
                row[f"{name}_switch_width_m"] = float(
                    ps["switch_width"][p]
                )

            row["kernel_positive_profiles"] = ";".join(
                positive_profiles
            )
            row["all_diagnostic_profiles_kernel_positive"] = (
                len(positive_profiles) == len(diagnostic_names)
            )

            occurrence_rows.append(row)
            coordinate_to_occurrences[key].append(occurrence_id)

    localized_occurrences = len(occurrence_rows)

    # Aggregate physically duplicate sample coordinates.
    unique_rows = []
    occurrence_to_unique = {}

    for unique_id, (key, ids) in enumerate(
        sorted(
            coordinate_to_occurrences.items(),
            key=lambda kv: min(kv[1]),
        )
    ):
        first = occurrence_rows[ids[0]]
        for occ_id in ids:
            occurrence_to_unique[occ_id] = unique_id

        unique_rows.append(
            {
                "unique_witness_id": unique_id,
                "duplicate_occurrence_count": int(len(ids)),
                "occurrence_ids": ";".join(str(x) for x in ids),
                "edge_ids": ";".join(
                    str(occurrence_rows[x]["edge_id"])
                    for x in ids
                ),
                "axis_names": ";".join(
                    occurrence_rows[x]["axis_name"]
                    for x in ids
                ),
                "v_f": first["v_f"],
                "v_p": first["v_p"],
                "a_f": first["a_f"],
                "bar_a": first["bar_a"],
                "bar_u": first["bar_u"],
                "age": first["age"],
                "all_occurrences_all_profiles_kernel_positive":
                    all(
                        bool(
                            occurrence_rows[x][
                                "all_diagnostic_profiles_kernel_positive"
                            ]
                        )
                        for x in ids
                    ),
            }
        )

    for row in occurrence_rows:
        row["unique_witness_id"] = occurrence_to_unique[
            row["occurrence_id"]
        ]

    unique_count = len(unique_rows)

    print(f"LOCALIZED_KERNEL_POSITIVE_OCCURRENCES={localized_occurrences}")
    print(f"UNIQUE_PHYSICAL_KERNEL_SAMPLE_COORDINATES={unique_count}")
    print(
        "DUPLICATE_COLLAPSE_COUNT="
        f"{localized_occurrences - unique_count}"
    )

    # Local dyadic and ULP stress is performed occurrence-wise because
    # different originating edge directions carry different geometric meaning.
    stress_meta = []
    stress_points = []

    for occ in occurrence_rows:
        edge = edges[int(occ["edge_id"])]
        w = state[int(edge["witness_node"])]
        n = state[int(edge["nonwitness_node"])]
        t0 = float(occ["t"])

        center = (1.0 - t0) * w + t0 * n

        def add(kind, delta_t, x):
            stress_meta.append(
                {
                    "occurrence_id": int(occ["occurrence_id"]),
                    "kind": kind,
                    "delta_t": float(delta_t),
                }
            )
            stress_points.append(np.asarray(x, dtype=float))

        add("center", 0.0, center)

        for delta in offsets:
            tl = max(0.0, t0 - delta)
            tr = min(1.0, t0 + delta)

            xl = (1.0 - tl) * w + tl * n
            xr = (1.0 - tr) * w + tr * n

            add(f"dyadic_minus_{delta:.17g}", -delta, xl)
            add(f"dyadic_plus_{delta:.17g}", delta, xr)

        add(
            "ulp_toward_witness",
            float("nan"),
            np.nextafter(center, w),
        )
        add(
            "ulp_toward_nonwitness",
            float("nan"),
            np.nextafter(center, n),
        )

    stress_points = np.asarray(stress_points, dtype=float)

    stress_eval = evaluate_points(
        stress_points,
        data,
        p1_cfg,
        p2b_cfg,
        p2c_cfg,
        p3a_cfg,
        diagnostic_names,
        tol,
    )

    completion_invalid += stress_eval["completion_invalid"]
    defer_invalid += stress_eval["defer_invalid"]

    by_occurrence = defaultdict(list)
    for i, meta in enumerate(stress_meta):
        by_occurrence[int(meta["occurrence_id"])].append((i, meta))

    stress_rows = []
    classification_counts = Counter()

    open_neighborhood_occurrences = 0
    one_sided_occurrences = 0
    ulp_sensitive_occurrences = 0
    isolated_occurrences = 0
    center_not_reproduced = 0
    all_profiles_center_positive_count = 0
    minimum_positive_center_multiple = math.inf

    smallest_delta = min(offsets)

    for occ in occurrence_rows:
        occ_id = int(occ["occurrence_id"])
        items = by_occurrence[occ_id]
        index_by_kind = {meta["kind"]: i for i, meta in items}

        center_i = index_by_kind["center"]

        center_profile_positive = {}
        center_profile_component = {}

        for name in diagnostic_names:
            stats = stress_eval["profiles"][name]["stats"]
            center_profile_positive[name] = bool(
                stats["kernel_mask"][center_i]
            )
            center_profile_component[name] = float(
                stats["kernel_component"][center_i]
            )

        all_center_positive = all(center_profile_positive.values())
        all_profiles_center_positive_count += int(all_center_positive)

        positive_center_components = [
            value
            for name, value in center_profile_component.items()
            if center_profile_positive[name]
        ]
        if positive_center_components:
            local_multiple = (
                min(positive_center_components) / tol
            )
            minimum_positive_center_multiple = min(
                minimum_positive_center_multiple,
                local_multiple,
            )
        else:
            local_multiple = 0.0

        dyadic_pairs_by_profile = {
            name: {} for name in diagnostic_names
        }

        for delta in offsets:
            minus_kind = f"dyadic_minus_{delta:.17g}"
            plus_kind = f"dyadic_plus_{delta:.17g}"
            mi = index_by_kind[minus_kind]
            pi = index_by_kind[plus_kind]

            for name in diagnostic_names:
                stats = stress_eval["profiles"][name]["stats"]
                dyadic_pairs_by_profile[name][delta] = (
                    bool(stats["kernel_mask"][mi]),
                    bool(stats["kernel_mask"][pi]),
                )

        ulp_w_i = index_by_kind["ulp_toward_witness"]
        ulp_n_i = index_by_kind["ulp_toward_nonwitness"]

        profile_classifications = {}
        for name in diagnostic_names:
            stats = stress_eval["profiles"][name]["stats"]
            cls = local_classification(
                center_profile_positive[name],
                dyadic_pairs_by_profile[name],
                (
                    bool(stats["kernel_mask"][ulp_w_i]),
                    bool(stats["kernel_mask"][ulp_n_i]),
                ),
            )
            profile_classifications[name] = cls

        # Conservative occurrence classification: require all diagnostics
        # to agree on the strongest classification.
        classes = set(profile_classifications.values())
        if len(classes) == 1:
            occurrence_classification = next(iter(classes))
        else:
            occurrence_classification = (
                "PROFILE_SPECIFIC_LOCAL_CLASSIFICATION"
            )

        classification_counts[occurrence_classification] += 1

        open_neighborhood_occurrences += int(
            occurrence_classification
            == "OPEN_NEIGHBORHOOD_KERNEL_WITNESS_CANDIDATE"
        )
        one_sided_occurrences += int(
            occurrence_classification
            == "ONE_SIDED_BOUNDARY_KERNEL_WITNESS_CANDIDATE"
        )
        ulp_sensitive_occurrences += int(
            occurrence_classification
            == "MACHINE_ULP_SENSITIVE"
        )
        isolated_occurrences += int(
            occurrence_classification
            == "ISOLATED_AT_TESTED_DYADIC_SCALES"
        )
        center_not_reproduced += int(
            occurrence_classification
            == "CENTER_NOT_REPRODUCED"
        )

        fast = stress_eval["profiles"]["diagnostic_fast"]
        sig = (
            fast["selected_action_q0"],
            fast["completion_index_q0"],
            fast["defer_index_q0"],
        )

        def signature_at(i):
            return (
                int(sig[0][i]),
                int(sig[1][i]),
                int(sig[2][i]),
            )

        smallest_minus_i = index_by_kind[
            f"dyadic_minus_{smallest_delta:.17g}"
        ]
        smallest_plus_i = index_by_kind[
            f"dyadic_plus_{smallest_delta:.17g}"
        ]

        row = {
            "occurrence_id": occ_id,
            "unique_witness_id": int(occ["unique_witness_id"]),
            "edge_id": int(occ["edge_id"]),
            "axis_name": occ["axis_name"],
            "t": float(occ["t"]),
            "classification": occurrence_classification,
            "all_profiles_center_kernel_positive":
                bool(all_center_positive),
            "minimum_positive_center_kernel_multiple_of_tol":
                float(local_multiple),
            "fast_center_kernel_component_m":
                center_profile_component["diagnostic_fast"],
            "nominal_center_kernel_component_m":
                center_profile_component["diagnostic_nominal"],
            "stressed_center_kernel_component_m":
                center_profile_component["diagnostic_stressed"],
            "fast_classification":
                profile_classifications["diagnostic_fast"],
            "nominal_classification":
                profile_classifications["diagnostic_nominal"],
            "stressed_classification":
                profile_classifications["diagnostic_stressed"],
            "fast_center_signature":
                str(signature_at(center_i)),
            "fast_smallest_minus_signature":
                str(signature_at(smallest_minus_i)),
            "fast_smallest_plus_signature":
                str(signature_at(smallest_plus_i)),
            "fast_ulp_witness_signature":
                str(signature_at(ulp_w_i)),
            "fast_ulp_nonwitness_signature":
                str(signature_at(ulp_n_i)),
            "smallest_tested_delta_t":
                float(smallest_delta),
        }

        for delta in offsets:
            for name in diagnostic_names:
                left, right = dyadic_pairs_by_profile[name][delta]
                tag = (
                    f"{name}_delta_"
                    + f"{delta:.17g}".replace(".", "p")
                )
                row[f"{tag}_minus_kernel_positive"] = bool(left)
                row[f"{tag}_plus_kernel_positive"] = bool(right)

        for name in diagnostic_names:
            stats = stress_eval["profiles"][name]["stats"]
            row[f"{name}_ulp_witness_kernel_positive"] = bool(
                stats["kernel_mask"][ulp_w_i]
            )
            row[f"{name}_ulp_nonwitness_kernel_positive"] = bool(
                stats["kernel_mask"][ulp_n_i]
            )

        stress_rows.append(row)

    if minimum_positive_center_multiple is math.inf:
        minimum_positive_center_multiple = 0.0

    # Unique-witness level promotion requires at least one originating edge
    # occurrence to have an open-neighborhood candidate.
    unique_classification = {}
    for unique in unique_rows:
        uid = int(unique["unique_witness_id"])
        classes = [
            row["classification"]
            for row in stress_rows
            if int(row["unique_witness_id"]) == uid
        ]

        if "OPEN_NEIGHBORHOOD_KERNEL_WITNESS_CANDIDATE" in classes:
            cls = "OPEN_NEIGHBORHOOD_KERNEL_WITNESS_CANDIDATE"
        elif "ONE_SIDED_BOUNDARY_KERNEL_WITNESS_CANDIDATE" in classes:
            cls = "ONE_SIDED_BOUNDARY_KERNEL_WITNESS_CANDIDATE"
        elif "MACHINE_ULP_SENSITIVE" in classes:
            cls = "MACHINE_ULP_SENSITIVE"
        elif "ISOLATED_AT_TESTED_DYADIC_SCALES" in classes:
            cls = "ISOLATED_AT_TESTED_DYADIC_SCALES"
        elif "CENTER_NOT_REPRODUCED" in classes:
            cls = "CENTER_NOT_REPRODUCED"
        else:
            cls = "PROFILE_SPECIFIC_LOCAL_CLASSIFICATION"

        unique["local_classification"] = cls
        unique_classification[uid] = cls

    unique_open_count = sum(
        int(
            row["local_classification"]
            == "OPEN_NEIGHBORHOOD_KERNEL_WITNESS_CANDIDATE"
        )
        for row in unique_rows
    )
    unique_one_sided_count = sum(
        int(
            row["local_classification"]
            == "ONE_SIDED_BOUNDARY_KERNEL_WITNESS_CANDIDATE"
        )
        for row in unique_rows
    )
    unique_ulp_sensitive_count = sum(
        int(
            row["local_classification"]
            == "MACHINE_ULP_SENSITIVE"
        )
        for row in unique_rows
    )

    center_margin_gate = (
        minimum_positive_center_multiple >= min_center_multiple
    )

    if localized_occurrences != expected_occurrences:
        interpretation = (
            "D6B_KERNEL_OCCURRENCE_COUNT_NOT_REPRODUCED"
        )
        next_step = (
            "STOP_AND_RECONCILE_D6B_D6C_REPLAY"
        )
    elif unique_open_count > 0 and center_margin_gate:
        interpretation = (
            "Q_DEPENDENT_KERNEL_COMPONENT_HAS_ULP_STABLE_OPEN_NEIGHBORHOOD_OPERATOR_WITNESSES"
        )
        next_step = (
            "RUN_P3B1_R7D7_REFINED_FIXED_POINT_KERNEL_RECOMPUTATION"
        )
    elif unique_one_sided_count > 0 and unique_ulp_sensitive_count == 0:
        interpretation = (
            "Q_DEPENDENT_KERNEL_COMPONENT_IS_LOCALLY_BOUNDARY_CONFINED_BUT_ULP_STABLE"
        )
        next_step = (
            "RUN_P3B1_R7D7_LOCAL_LOOKUP_GRID_REFINEMENT_BEFORE_FIXED_POINT_PROMOTION"
        )
    elif unique_ulp_sensitive_count > 0:
        interpretation = (
            "Q_DEPENDENT_KERNEL_COMPONENT_IS_MACHINE_ULP_SENSITIVE"
        )
        next_step = (
            "RUN_HIGHER_PRECISION_INDEX_AND_BOUNDARY_ARITHMETIC_BEFORE_KERNEL_RECOMPUTATION"
        )
    else:
        interpretation = (
            "Q_DEPENDENT_KERNEL_COMPONENT_DOES_NOT_PERSIST_AS_AN_OPEN_LOCAL_OPERATOR_WITNESS"
        )
        next_step = (
            "RETAIN_D6B_AS_BOUNDARY_DIAGNOSTIC_AND_DO_NOT_PROMOTE_KERNEL_CLAIM"
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
        "D6B_KERNEL_INTERPRETATION_MATCH":
            d6b_latest.get("interpretation") == expected,
        "STAGE1_KERNEL_OCCURRENCE_COUNT_REPRODUCED":
            localized_occurrences == expected_occurrences,
        "TRANSITION_COVERAGE_COMPLETE":
            completion_invalid == 0
            and defer_invalid == 0,
        "UNIQUE_COORDINATE_SET_NONEMPTY":
            unique_count > 0,
    }

    status = "PASS" if all(integrity_checks.values()) else "FAIL"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    occurrence_csv = (
        RESULTS_DIR
        / f"P3B1_R7D6C_KERNEL_OCCURRENCES_{stamp}.csv"
    )
    unique_csv = (
        RESULTS_DIR
        / f"P3B1_R7D6C_UNIQUE_KERNEL_WITNESSES_{stamp}.csv"
    )
    stress_csv = (
        RESULTS_DIR
        / f"P3B1_R7D6C_LOCAL_PERSISTENCE_ULP_{stamp}.csv"
    )

    write_csv(occurrence_csv, occurrence_rows)
    write_csv(unique_csv, unique_rows)
    write_csv(stress_csv, stress_rows)

    output = {
        "schema":
            "SCV_P3B1_R7D6C_KERNEL_WITNESS_LOCALIZATION_V1",
        "status":
            status,
        "timestamp_utc":
            stamp,
        "classification":
            (
                "operator-level localization and local persistence audit of "
                "R7-D6B q-dependent kernel-component sample occurrences; "
                "not a refined fixed point, maximal kernel, continuous-domain "
                "proof, or implementation-refinement result"
            ),
        "integrity_checks":
            {k: bool(v) for k, v in integrity_checks.items()},
        "interpretation":
            interpretation,
        "recommended_next_step":
            next_step,
        "protocol":
            cfg,
        "aggregate": {
            "expected_stage1_kernel_positive_occurrences":
                int(expected_occurrences),
            "localized_stage1_kernel_positive_occurrences":
                int(localized_occurrences),
            "unique_physical_kernel_sample_coordinates":
                int(unique_count),
            "duplicate_occurrence_collapse_count":
                int(localized_occurrences - unique_count),
            "all_profiles_center_kernel_positive_occurrences":
                int(all_profiles_center_positive_count),
            "open_neighborhood_occurrences":
                int(open_neighborhood_occurrences),
            "one_sided_boundary_occurrences":
                int(one_sided_occurrences),
            "machine_ulp_sensitive_occurrences":
                int(ulp_sensitive_occurrences),
            "isolated_occurrences":
                int(isolated_occurrences),
            "center_not_reproduced_occurrences":
                int(center_not_reproduced),
            "unique_open_neighborhood_witnesses":
                int(unique_open_count),
            "unique_one_sided_boundary_witnesses":
                int(unique_one_sided_count),
            "unique_machine_ulp_sensitive_witnesses":
                int(unique_ulp_sensitive_count),
            "minimum_positive_center_kernel_multiple_of_tol":
                float(minimum_positive_center_multiple),
            "completion_invalid":
                int(completion_invalid),
            "defer_invalid":
                int(defer_invalid),
        },
        "classification_counts":
            dict(classification_counts),
        "claims": {
            "refined_fixed_point_kernel_claim_authorized":
                False,
            "scientific_kernel_claim_authorized":
                False,
            "maximal_kernel_claim_authorized":
                False,
            "continuous_domain_global_claim_authorized":
                False,
            "implementation_refinement_claim_authorized":
                False,
            "operator_level_refined_fixed_point_recompute_candidate":
                bool(
                    status == "PASS"
                    and unique_open_count > 0
                    and center_margin_gate
                    and unique_ulp_sensitive_count == 0
                ),
            "reason":
                (
                    "D6C can justify a refined fixed-point recomputation only. "
                    "A q-dependent kernel claim requires re-solving the frozen-"
                    "halo fixed point on a refined lookup/evaluation grid and "
                    "re-establishing fallback inclusion, service dominance, "
                    "replay, and refinement gates."
                ),
        },
        "artifacts": {
            "occurrence_csv": str(occurrence_csv),
            "unique_witness_csv": str(unique_csv),
            "local_stress_csv": str(stress_csv),
        },
    }

    result_path = (
        RESULTS_DIR
        / f"P3B1_R7D6C_KERNEL_WITNESS_AUDIT_{stamp}.json"
    )
    latest_path = RESULTS_DIR / "P3B1_R7D6C_LATEST.json"
    manifest_path = (
        RESULTS_DIR
        / f"P3B1_R7D6C_MANIFEST_{stamp}.sha256"
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
        D6C_CFG_PATH,
        D4_PATH,
        D5_PATH,
        D6_PATH,
        D6B_PATH,
        Path(d1.__file__),
        Path(d3.__file__),
        Path(d6b.__file__),
        Path(d3.r7.__file__),
        Path(d3.r3.__file__),
        Path(__file__),
        result_path,
        occurrence_csv,
        unique_csv,
        stress_csv,
    ]

    manifest = "\n".join(
        f"{sha256_file(path)}  {path}"
        for path in manifest_files
    ) + "\n"
    atomic_write(manifest_path, manifest)

    print("=== P3-B1-R7-D6C DECISION ===")
    print(
        "LOCALIZED_KERNEL_POSITIVE_OCCURRENCES="
        f"{localized_occurrences}"
    )
    print(
        "UNIQUE_PHYSICAL_KERNEL_SAMPLE_COORDINATES="
        f"{unique_count}"
    )
    print(
        "DUPLICATE_OCCURRENCE_COLLAPSE_COUNT="
        f"{localized_occurrences - unique_count}"
    )
    print(
        "OPEN_NEIGHBORHOOD_OCCURRENCES="
        f"{open_neighborhood_occurrences}"
    )
    print(
        "ONE_SIDED_BOUNDARY_OCCURRENCES="
        f"{one_sided_occurrences}"
    )
    print(
        "MACHINE_ULP_SENSITIVE_OCCURRENCES="
        f"{ulp_sensitive_occurrences}"
    )
    print(
        "UNIQUE_OPEN_NEIGHBORHOOD_WITNESSES="
        f"{unique_open_count}"
    )
    print(
        "MIN_CENTER_KERNEL_MULTIPLE_OF_TOL="
        f"{minimum_positive_center_multiple:.12g}"
    )
    print(f"INTERPRETATION={interpretation}")
    print(f"RECOMMENDED_NEXT_STEP={next_step}")
    print(f"P3B1_R7D6C_KERNEL_WITNESS_AUDIT={status}")
    print(
        "REFINED_FIXED_POINT_RECOMPUTE_CANDIDATE="
        + (
            "YES"
            if output["claims"][
                "operator_level_refined_fixed_point_recompute_candidate"
            ]
            else "NO"
        )
    )
    print("SCIENTIFIC_KERNEL_CLAIM=NO")
    print("MAXIMAL_KERNEL_CLAIM=NO")
    print("CONTINUOUS_DOMAIN_GLOBAL_CLAIM=NO")
    print("IMPLEMENTATION_REFINEMENT_CLAIM=NO")
    print(f"RESULT_JSON={result_path}")
    print(f"MANIFEST={manifest_path}")

    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
