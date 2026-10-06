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

P1_CFG_PATH = ROOT / "01_config" / "p1_validation_v2.json"
P2B_CFG_PATH = ROOT / "01_config" / "p2b_hybrid_fallback_v1.json"
P2C_CFG_PATH = ROOT / "01_config" / "p2c_switching_guard_v1.json"
P3A_CFG_PATH = ROOT / "01_config" / "p3a_information_contract_v1.json"
D6_CFG_PATH = ROOT / "01_config" / "p3b1_r7d6_local_holdout_protocol_v1.json"

D3_PATH = ROOT / "04_results" / "P3B1_R7D3_LATEST.json"
D4_PATH = ROOT / "04_results" / "P3B1_R7D4_LATEST.json"
D5_PATH = ROOT / "04_results" / "P3B1_R7D5_LATEST.json"

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


def quantile_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {
            "min": 0.0,
            "p05": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "max": 0.0,
            "mean": 0.0,
        }
    return {
        "min": float(np.min(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def q_label_change(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[1] <= 1:
        return np.zeros(matrix.shape[0], dtype=bool)
    return np.any(matrix != matrix[:, [0]], axis=1)


def switch_geometry(
    raw: np.ndarray,
    clipped: np.ndarray,
    tol: float,
):
    raw_span = np.max(raw, axis=1) - np.min(raw, axis=1)
    kernel_component = (
        np.max(clipped, axis=1) - np.min(clipped, axis=1)
    )
    switch_width = np.maximum(
        raw_span - kernel_component,
        0.0,
    )
    switch_mask = switch_width > (2.0 * tol)
    robust_radius = 0.5 * switch_width

    return {
        "raw_span": raw_span,
        "kernel_component": kernel_component,
        "switch_width": switch_width,
        "switch_mask": switch_mask,
        "robust_radius": robust_radius,
    }


def enumerate_local_edges(
    switch_mask: np.ndarray,
    eval_shape: tuple[int, ...],
):
    """
    Enumerate each +axis grid edge exactly once.

    Returns
    -------
    interior_edges:
        both endpoints are R7-D5 switch witnesses.
    boundary_edges:
        exactly one endpoint is a switch witness, oriented witness -> nonwitness.
    """
    node_grid = np.arange(
        int(np.prod(eval_shape)),
        dtype=int,
    ).reshape(eval_shape)

    interior_edges = []
    boundary_edges = []

    for axis, size in enumerate(eval_shape):
        if size <= 1:
            continue

        left = [slice(None)] * len(eval_shape)
        right = [slice(None)] * len(eval_shape)
        left[axis] = slice(0, size - 1)
        right[axis] = slice(1, size)

        a = node_grid[tuple(left)].reshape(-1)
        b = node_grid[tuple(right)].reshape(-1)

        ma = switch_mask[a]
        mb = switch_mask[b]

        ww = ma & mb
        for i, j in zip(a[ww], b[ww]):
            interior_edges.append(
                {
                    "axis": axis,
                    "node_a": int(i),
                    "node_b": int(j),
                }
            )

        xor = ma ^ mb
        for i, j, ia in zip(a[xor], b[xor], ma[xor]):
            if bool(ia):
                w, n = int(i), int(j)
            else:
                w, n = int(j), int(i)

            boundary_edges.append(
                {
                    "axis": axis,
                    "witness_node": w,
                    "nonwitness_node": n,
                }
            )

    return interior_edges, boundary_edges


def make_holdouts(
    eval_flat,
    interior_edges,
    boundary_edges,
):
    state = np.column_stack(
        [np.asarray(x, dtype=float) for x in eval_flat]
    )

    records = []

    for edge_id, edge in enumerate(interior_edges):
        i = edge["node_a"]
        j = edge["node_b"]
        x = 0.5 * (state[i] + state[j])

        records.append(
            {
                "kind": "interior_midpoint",
                "edge_id": edge_id,
                "axis": int(edge["axis"]),
                "node_a": i,
                "node_b": j,
                "t_from_witness": "",
                "state": x,
            }
        )

    for edge_id, edge in enumerate(boundary_edges):
        w = edge["witness_node"]
        n = edge["nonwitness_node"]

        for t in (0.25, 0.50, 0.75):
            x = (1.0 - t) * state[w] + t * state[n]
            records.append(
                {
                    "kind": "boundary_fraction",
                    "edge_id": edge_id,
                    "axis": int(edge["axis"]),
                    "witness_node": w,
                    "nonwitness_node": n,
                    "t_from_witness": float(t),
                    "state": x,
                }
            )

    holdout_flat = [
        np.asarray(
            [row["state"][k] for row in records],
            dtype=float,
        )
        for k in range(6)
    ]

    return records, holdout_flat


def base_grid_membership_count(
    holdout_flat,
    eval_axes,
):
    n = len(holdout_flat[0])
    exact = np.ones(n, dtype=bool)

    for values, axis in zip(holdout_flat, eval_axes):
        axis = np.asarray(axis, dtype=float)
        matches = np.zeros(n, dtype=bool)
        for v in axis:
            matches |= np.isclose(
                values,
                v,
                rtol=0.0,
                atol=1.0e-12,
            )
        exact &= matches

    return int(np.count_nonzero(exact))


def boundary_pattern_rows(
    boundary_edges,
    records,
    diagnostic_names,
    masks_by_profile,
):
    # record positions for boundary holdouts, keyed by edge_id and t
    positions = {}
    for pos, row in enumerate(records):
        if row["kind"] != "boundary_fraction":
            continue
        positions[
            (
                int(row["edge_id"]),
                float(row["t_from_witness"]),
            )
        ] = pos

    out = []

    for edge_id, edge in enumerate(boundary_edges):
        row = {
            "edge_id": edge_id,
            "axis": int(edge["axis"]),
            "axis_name": AXIS_NAMES[int(edge["axis"])],
            "witness_node": int(edge["witness_node"]),
            "nonwitness_node": int(edge["nonwitness_node"]),
        }

        for name in diagnostic_names:
            bits = [1]
            for t in (0.25, 0.50, 0.75):
                p = positions[(edge_id, t)]
                bits.append(
                    int(bool(masks_by_profile[name][p]))
                )
            bits.append(0)

            transitions = sum(
                int(bits[k] != bits[k + 1])
                for k in range(len(bits) - 1)
            )
            pattern = "".join(str(x) for x in bits)

            row[f"{name}_pattern"] = pattern
            row[f"{name}_transition_count"] = int(transitions)
            row[f"{name}_reentry"] = bool(transitions > 1)

        out.append(row)

    return out


def main() -> int:
    protocol = load_json(D6_CFG_PATH)
    tol = float(protocol["numeric_tolerance_m"])
    replay_tol = float(protocol["replay_tolerance_m"])
    min_interior_survival = float(
        protocol["interior_midpoint_survival_fraction_min"]
    )
    min_profile_agreement = float(
        protocol[
            "diagnostic_profile_holdout_agreement_fraction_min"
        ]
    )
    max_reentry_fraction = float(
        protocol["boundary_reentry_fraction_max"]
    )
    min_radius_multiple = float(
        protocol[
            "interior_p05_robust_radius_multiple_of_tolerance_min"
        ]
    )

    d3_latest = load_json(D3_PATH)
    d4_latest = load_json(D4_PATH)
    d5_latest = load_json(D5_PATH)

    if d3_latest.get("status") != "PASS":
        raise RuntimeError("P3B1_R7D6_D3_UPSTREAM_FAIL")
    if d4_latest.get("status") != "PASS":
        raise RuntimeError("P3B1_R7D6_D4_UPSTREAM_FAIL")
    if d5_latest.get("status") != "PASS":
        raise RuntimeError("P3B1_R7D6_D5_UPSTREAM_FAIL")
    if not d5_latest["claims"].get(
        "model_level_supervisor_robustness_candidate",
        False,
    ):
        raise RuntimeError("P3B1_R7D6_D5_ROBUSTNESS_CANDIDATE_MISSING")

    data = d1.reconstruct()

    eval_cfg = data["eval_cfg"]
    eval_axes = data["eval_axes"]
    eval_flat = data["eval_flat"]
    eval_shape = tuple(int(x) for x in data["eval_shape"])
    eval_fallback = np.asarray(data["eval_fallback"], dtype=float)
    lookup_axes = data["lookup_axes"]
    lookup_shape = data["lookup_shape"]
    results = data["results"]

    p1_cfg = load_json(P1_CFG_PATH)
    p2b_cfg = load_json(P2B_CFG_PATH)
    p2c_cfg = load_json(P2C_CFG_PATH)
    p3a_cfg = load_json(P3A_CFG_PATH)

    base_transition = d3.build_transition_data(data)

    diagnostic_names = (
        "diagnostic_fast",
        "diagnostic_nominal",
        "diagnostic_stressed",
    )

    # Reconstruct the D5 base switch region from diagnostic_fast.
    fast_raw, fast_clipped, _, _ = d3.raw_operator_slices(
        results["diagnostic_fast"],
        lookup_shape,
        base_transition,
        eval_fallback,
    )
    base_geom = switch_geometry(
        fast_raw,
        fast_clipped,
        tol,
    )
    base_switch = base_geom["switch_mask"]

    base_switch_count = int(np.count_nonzero(base_switch))

    interior_edges, boundary_edges = enumerate_local_edges(
        base_switch,
        eval_shape,
    )

    records, holdout_flat = make_holdouts(
        eval_flat,
        interior_edges,
        boundary_edges,
    )

    if not records:
        raise RuntimeError("P3B1_R7D6_NO_HOLDOUT_RECORDS")

    holdout_transition = d3.r7.build_transition_data_to_lookup(
        eval_cfg,
        p1_cfg,
        p2b_cfg,
        p2c_cfg,
        p3a_cfg,
        holdout_flat,
        lookup_axes,
    )

    completion_invalid = sum(
        int(np.count_nonzero(~t["completion_valid"]))
        for t in holdout_transition["transitions"]
    )
    defer_invalid = sum(
        int(np.count_nonzero(~t["defer_valid"]))
        for t in holdout_transition["transitions"]
    )

    holdout_fallback = np.asarray(
        holdout_transition["fallback_required"],
        dtype=float,
    )

    exact_base_grid_holdouts = base_grid_membership_count(
        holdout_flat,
        eval_axes,
    )

    print("=== P3-B1-R7-D6 LOCAL BOUNDARY HOLDOUT REPLAY ===")
    print(f"BASE_EVAL_GRID_NODES={int(np.prod(eval_shape))}")
    print(f"BASE_SWITCH_NODES={base_switch_count}")
    print(f"INTERIOR_WITNESS_EDGES={len(interior_edges)}")
    print(f"BOUNDARY_WITNESS_EDGES={len(boundary_edges)}")
    print(f"OFFGRID_HOLDOUT_POINTS={len(records)}")
    print(f"EXACT_BASE_GRID_HOLDOUTS={exact_base_grid_holdouts}")
    print(f"COMPLETION_INVALID={completion_invalid}")
    print(f"DEFER_INVALID={defer_invalid}")

    masks_by_profile = {}
    geom_by_profile = {}
    action_change_by_profile = {}
    branch_change_by_profile = {}
    profile_rows = []

    interior_positions = np.asarray(
        [
            i for i, row in enumerate(records)
            if row["kind"] == "interior_midpoint"
        ],
        dtype=int,
    )
    boundary_positions = np.asarray(
        [
            i for i, row in enumerate(records)
            if row["kind"] == "boundary_fraction"
        ],
        dtype=int,
    )

    for name in diagnostic_names:
        raw, clipped, branch, action = d3.raw_operator_slices(
            results[name],
            lookup_shape,
            holdout_transition,
            holdout_fallback,
        )

        geom = switch_geometry(raw, clipped, tol)
        masks_by_profile[name] = geom["switch_mask"]
        geom_by_profile[name] = geom

        action_change = q_label_change(action)
        branch_change = q_label_change(branch)
        action_change_by_profile[name] = action_change
        branch_change_by_profile[name] = branch_change

        interior_mask = geom["switch_mask"][interior_positions]
        interior_survival = (
            float(np.mean(interior_mask))
            if len(interior_mask) else 0.0
        )

        interior_radius = geom["robust_radius"][
            interior_positions
        ][interior_mask]
        radius_stats = quantile_summary(interior_radius)

        kernel_nodes = int(
            np.count_nonzero(
                geom["kernel_component"] > tol
            )
        )

        action_confounded = int(
            np.count_nonzero(
                action_change & geom["switch_mask"]
            )
        )
        branch_on_switch = int(
            np.count_nonzero(
                branch_change & geom["switch_mask"]
            )
        )

        profile_rows.append(
            {
                "profile": name,
                "R": int(results[name]["R"]),
                "holdout_points": int(len(records)),
                "holdout_switch_nodes":
                    int(np.count_nonzero(geom["switch_mask"])),
                "holdout_kernel_component_nodes":
                    kernel_nodes,
                "interior_midpoints":
                    int(len(interior_positions)),
                "interior_midpoint_switch_nodes":
                    int(np.count_nonzero(interior_mask)),
                "interior_midpoint_survival_fraction":
                    interior_survival,
                "interior_robust_radius_min_m":
                    radius_stats["min"],
                "interior_robust_radius_p05_m":
                    radius_stats["p05"],
                "interior_robust_radius_p50_m":
                    radius_stats["p50"],
                "interior_robust_radius_p95_m":
                    radius_stats["p95"],
                "interior_robust_radius_max_m":
                    radius_stats["max"],
                "action_q_change_on_holdout_switch_nodes":
                    action_confounded,
                "branch_q_change_on_holdout_switch_nodes":
                    branch_on_switch,
            }
        )

        print(
            f"{name.upper()} "
            f"R={int(results[name]['R'])} "
            f"HOLDOUT_SWITCH={int(np.count_nonzero(geom['switch_mask']))} "
            f"KERNEL_COMP={kernel_nodes} "
            f"INTERIOR_SURVIVAL={interior_survival:.12g} "
            f"INTERIOR_RHO_P05_M={radius_stats['p05']:.12g} "
            f"ACTION_Q_ON_SWITCH={action_confounded} "
            f"BRANCH_Q_ON_SWITCH={branch_on_switch}"
        )

    # Pointwise profile agreement on all off-grid holdouts.
    fast = masks_by_profile["diagnostic_fast"]
    nominal = masks_by_profile["diagnostic_nominal"]
    stressed = masks_by_profile["diagnostic_stressed"]

    all_agree = (fast == nominal) & (nominal == stressed)
    all_agreement_fraction = float(np.mean(all_agree))

    interior_agreement_fraction = (
        float(np.mean(all_agree[interior_positions]))
        if len(interior_positions) else 0.0
    )
    boundary_agreement_fraction = (
        float(np.mean(all_agree[boundary_positions]))
        if len(boundary_positions) else 0.0
    )

    consensus_switch = fast & nominal & stressed
    consensus_interior = consensus_switch[interior_positions]
    consensus_interior_survival = (
        float(np.mean(consensus_interior))
        if len(consensus_interior) else 0.0
    )

    # Detect any off-grid q-dependence of the viability envelope.
    holdout_kernel_nodes_any_profile = int(
        np.count_nonzero(
            (
                geom_by_profile["diagnostic_fast"][
                    "kernel_component"
                ] > tol
            )
            |
            (
                geom_by_profile["diagnostic_nominal"][
                    "kernel_component"
                ] > tol
            )
            |
            (
                geom_by_profile["diagnostic_stressed"][
                    "kernel_component"
                ] > tol
            )
        )
    )

    # Deconfounding: q-dependent cooperative action must not be needed to
    # create the consensus supervisor switch effect.
    action_confounded_consensus = int(
        np.count_nonzero(
            consensus_switch
            &
            (
                action_change_by_profile["diagnostic_fast"]
                |
                action_change_by_profile["diagnostic_nominal"]
                |
                action_change_by_profile["diagnostic_stressed"]
            )
        )
    )

    boundary_rows = boundary_pattern_rows(
        boundary_edges,
        records,
        diagnostic_names,
        masks_by_profile,
    )

    for row in boundary_rows:
        row["all_profiles_same_pattern"] = (
            row["diagnostic_fast_pattern"]
            ==
            row["diagnostic_nominal_pattern"]
            ==
            row["diagnostic_stressed_pattern"]
        )

    reentry_fraction_by_profile = {}
    for name in diagnostic_names:
        reentry_count = sum(
            int(bool(row[f"{name}_reentry"]))
            for row in boundary_rows
        )
        reentry_fraction_by_profile[name] = (
            float(reentry_count / len(boundary_rows))
            if boundary_rows else 0.0
        )

    max_reentry_observed = max(
        reentry_fraction_by_profile.values()
    ) if reentry_fraction_by_profile else 0.0

    boundary_pattern_agreement = (
        float(
            np.mean(
                [
                    bool(row["all_profiles_same_pattern"])
                    for row in boundary_rows
                ]
            )
        )
        if boundary_rows else 0.0
    )

    diagnostic_rows = {
        row["profile"]: row for row in profile_rows
    }

    min_profile_interior_survival = min(
        diagnostic_rows[name][
            "interior_midpoint_survival_fraction"
        ]
        for name in diagnostic_names
    )

    min_interior_p05_radius = min(
        diagnostic_rows[name]["interior_robust_radius_p05_m"]
        for name in diagnostic_names
    )

    interior_p05_radius_multiple = (
        min_interior_p05_radius / tol
        if tol > 0.0 else float("inf")
    )

    all_holdouts_offgrid_gate = (
        exact_base_grid_holdouts == 0
    )
    coverage_gate = (
        completion_invalid == 0
        and defer_invalid == 0
    )
    kernel_absence_gate = (
        holdout_kernel_nodes_any_profile == 0
    )
    interior_survival_gate = (
        consensus_interior_survival
        >= min_interior_survival
        and
        min_profile_interior_survival
        >= min_interior_survival
    )
    profile_agreement_gate = (
        all_agreement_fraction
        >= min_profile_agreement
    )
    boundary_regular_gate = (
        max_reentry_observed
        <= max_reentry_fraction
    )
    radius_gate = (
        interior_p05_radius_multiple
        >= min_radius_multiple
    )
    action_deconfounding_gate = (
        action_confounded_consensus == 0
    )

    model_level_claim_authorized = bool(
        coverage_gate
        and all_holdouts_offgrid_gate
        and kernel_absence_gate
        and interior_survival_gate
        and profile_agreement_gate
        and boundary_regular_gate
        and radius_gate
        and action_deconfounding_gate
    )

    integrity_checks = {
        "D3_UPSTREAM_PASS":
            d3_latest.get("status") == "PASS",
        "D4_UPSTREAM_PASS":
            d4_latest.get("status") == "PASS",
        "D5_UPSTREAM_PASS":
            d5_latest.get("status") == "PASS",
        "D5_MODEL_LEVEL_ROBUSTNESS_CANDIDATE":
            bool(
                d5_latest["claims"].get(
                    "model_level_supervisor_robustness_candidate",
                    False,
                )
            ),
        "BASE_SWITCH_COUNT_REPRODUCED":
            base_switch_count
            ==
            int(
                d5_latest["aggregate"][
                    "diagnostic_switch_witness_nodes_sum"
                ]
                // 3
            ),
        "LOCAL_EDGE_SET_NONEMPTY":
            len(interior_edges) > 0
            and len(boundary_edges) > 0,
        "HOLDOUT_TRANSITION_COVERAGE":
            coverage_gate,
        "HOLDOUTS_STRICTLY_OFF_BASE_GRID":
            all_holdouts_offgrid_gate,
    }

    status = "PASS" if all(integrity_checks.values()) else "FAIL"

    if not coverage_gate:
        interpretation = (
            "LOCAL_HOLDOUT_REPLAY_BLOCKED_BY_HALO_COVERAGE"
        )
        next_step = (
            "EXPAND_FROZEN_HALO_BEFORE_INTERPRETING_OFFGRID_BOUNDARY"
        )
    elif holdout_kernel_nodes_any_profile > 0:
        interpretation = (
            "OFFGRID_HOLDOUTS_REVEAL_Q_DEPENDENT_VIABILITY_ENVELOPE_MISSED_BY_BASE_GRID"
        )
        next_step = (
            "STOP_SUPERVISOR_PROMOTION_AND_RUN_REFINED_KERNEL_RECOMPUTATION"
        )
    elif not interior_survival_gate:
        interpretation = (
            "BASE_GRID_SWITCH_REGION_FAILS_INTERIOR_OFFGRID_SURVIVAL"
        )
        next_step = (
            "RUN_ADAPTIVE_LOCAL_REFINEMENT_BEFORE_ANY_SUPERVISOR_CLAIM"
        )
    elif not profile_agreement_gate:
        interpretation = (
            "OFFGRID_SWITCH_GEOMETRY_BECOMES_PROFILE_SPECIFIC"
        )
        next_step = (
            "RUN_PROFILE_SPECIFIC_LOCAL_REFINEMENT"
        )
    elif not boundary_regular_gate:
        interpretation = (
            "OFFGRID_BOUNDARY_SHOWS_SUBCELL_REENTRY_OR_OSCILLATION"
        )
        next_step = (
            "RUN_ADAPTIVE_BOUNDARY_BISECTION_AND_HIGH_PRECISION_REPLAY"
        )
    elif not radius_gate:
        interpretation = (
            "OFFGRID_INTERIOR_SWITCH_SURVIVES_BUT_WITH_WEAK_NUMERICAL_MARGIN"
        )
        next_step = (
            "RUN_HIGH_PRECISION_LOCAL_REPLAY_BEFORE_CLAIM"
        )
    elif not action_deconfounding_gate:
        interpretation = (
            "OFFGRID_SUPERVISOR_EFFECT_IS_CONFOUNDED_WITH_ACTION_SELECTION"
        )
        next_step = (
            "DECOMPOSE_ACTION_AND_SUPERVISOR_EFFECTS_BEFORE_CLAIM"
        )
    else:
        interpretation = (
            "Q_DEPENDENT_SUPERVISOR_SWITCH_SURFACE_SURVIVES_LOCAL_OFFGRID_HOLDOUT_REPLAY_WITHOUT_KERNEL_OR_ACTION_CONFOUNDING"
        )
        next_step = (
            "FREEZE_MODEL_LEVEL_SUPERVISORY_RESULT_AND_RUN_P3B1_R7D7_KERNEL_Q_SUPPRESSION_STRUCTURAL_AUDIT"
        )

    # Per-holdout CSV.
    holdout_rows = []
    vf, vp, af, bar_a, bar_u, age = holdout_flat

    for i, rec in enumerate(records):
        row = {
            "holdout_index": i,
            "kind": rec["kind"],
            "edge_id": int(rec["edge_id"]),
            "axis": int(rec["axis"]),
            "axis_name": AXIS_NAMES[int(rec["axis"])],
            "t_from_witness": rec["t_from_witness"],
            "v_f": float(vf[i]),
            "v_p": float(vp[i]),
            "a_f": float(af[i]),
            "bar_a": float(bar_a[i]),
            "bar_u": float(bar_u[i]),
            "age": float(age[i]),
            "fallback_required_m": float(holdout_fallback[i]),
            "diagnostic_profiles_agree":
                bool(all_agree[i]),
            "consensus_switch_witness":
                bool(consensus_switch[i]),
        }

        if rec["kind"] == "interior_midpoint":
            row["node_a"] = int(rec["node_a"])
            row["node_b"] = int(rec["node_b"])
        else:
            row["witness_node"] = int(rec["witness_node"])
            row["nonwitness_node"] = int(rec["nonwitness_node"])

        for name in diagnostic_names:
            g = geom_by_profile[name]
            row[f"{name}_switch"] = bool(g["switch_mask"][i])
            row[f"{name}_switch_width_m"] = float(
                g["switch_width"][i]
            )
            row[f"{name}_robust_radius_m"] = float(
                g["robust_radius"][i]
            )
            row[f"{name}_kernel_component_m"] = float(
                g["kernel_component"][i]
            )
            row[f"{name}_action_q_change"] = bool(
                action_change_by_profile[name][i]
            )
            row[f"{name}_branch_q_change"] = bool(
                branch_change_by_profile[name][i]
            )

        holdout_rows.append(row)

    stamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    profile_csv = (
        RESULTS_DIR
        / f"P3B1_R7D6_PROFILE_HOLDOUT_AUDIT_{stamp}.csv"
    )
    holdout_csv = (
        RESULTS_DIR
        / f"P3B1_R7D6_OFFGRID_HOLDOUT_POINTS_{stamp}.csv"
    )
    boundary_csv = (
        RESULTS_DIR
        / f"P3B1_R7D6_BOUNDARY_EDGE_PATTERNS_{stamp}.csv"
    )

    write_csv(profile_csv, profile_rows)
    write_csv(holdout_csv, holdout_rows)
    write_csv(boundary_csv, boundary_rows)

    output = {
        "schema":
            "SCV_P3B1_R7D6_LOCAL_OFFGRID_HOLDOUT_REPLAY_V1",
        "status":
            status,
        "timestamp_utc":
            stamp,
        "classification":
            (
                "local off-grid replay of the q-dependent certified C/F "
                "supervisor switch region on the frozen-halo fixed point; "
                "not a global continuous-domain kernel proof and not "
                "implementation refinement evidence"
            ),
        "integrity_checks":
            {k: bool(v) for k, v in integrity_checks.items()},
        "scientific_gates": {
            "offgrid_kernel_absence_gate":
                bool(kernel_absence_gate),
            "interior_midpoint_survival_gate":
                bool(interior_survival_gate),
            "diagnostic_profile_holdout_agreement_gate":
                bool(profile_agreement_gate),
            "boundary_subcell_regularity_gate":
                bool(boundary_regular_gate),
            "interior_numerical_radius_gate":
                bool(radius_gate),
            "action_deconfounding_gate":
                bool(action_deconfounding_gate),
        },
        "interpretation":
            interpretation,
        "recommended_next_step":
            next_step,
        "profile_stats":
            profile_rows,
        "boundary_reentry_fraction_by_profile":
            reentry_fraction_by_profile,
        "aggregate": {
            "base_switch_nodes":
                int(base_switch_count),
            "interior_witness_edges":
                int(len(interior_edges)),
            "boundary_witness_edges":
                int(len(boundary_edges)),
            "offgrid_holdout_points":
                int(len(records)),
            "exact_base_grid_holdouts":
                int(exact_base_grid_holdouts),
            "completion_invalid":
                int(completion_invalid),
            "defer_invalid":
                int(defer_invalid),
            "holdout_kernel_component_nodes_any_profile":
                int(holdout_kernel_nodes_any_profile),
            "all_holdout_profile_agreement_fraction":
                float(all_agreement_fraction),
            "interior_profile_agreement_fraction":
                float(interior_agreement_fraction),
            "boundary_profile_agreement_fraction":
                float(boundary_agreement_fraction),
            "boundary_pattern_profile_agreement_fraction":
                float(boundary_pattern_agreement),
            "consensus_interior_midpoint_survival_fraction":
                float(consensus_interior_survival),
            "min_profile_interior_midpoint_survival_fraction":
                float(min_profile_interior_survival),
            "max_boundary_reentry_fraction":
                float(max_reentry_observed),
            "min_interior_p05_robust_radius_m":
                float(min_interior_p05_radius),
            "min_interior_p05_robust_radius_multiple_of_tol":
                float(interior_p05_radius_multiple),
            "action_confounded_consensus_switch_holdouts":
                int(action_confounded_consensus),
        },
        "protocol":
            protocol,
        "claims": {
            "model_level_supervisory_q_sensitivity_claim_authorized":
                bool(model_level_claim_authorized),
            "scientific_kernel_claim_authorized":
                False,
            "continuous_domain_global_claim_authorized":
                False,
            "implementation_refinement_claim_authorized":
                False,
            "p6_certified":
                False,
            "claim_scope":
                (
                    "If authorized, the claim is restricted to the current "
                    "frozen-halo finite abstraction plus the pre-registered "
                    "local off-grid holdout set. It does not establish a "
                    "global continuous-domain theorem or I_c subset A_c."
                ),
        },
        "artifacts": {
            "profile_csv": str(profile_csv),
            "holdout_csv": str(holdout_csv),
            "boundary_pattern_csv": str(boundary_csv),
        },
    }

    result_path = (
        RESULTS_DIR
        / f"P3B1_R7D6_LOCAL_OFFGRID_HOLDOUT_{stamp}.json"
    )
    latest_path = RESULTS_DIR / "P3B1_R7D6_LATEST.json"
    manifest_path = (
        RESULTS_DIR
        / f"P3B1_R7D6_MANIFEST_{stamp}.sha256"
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
        D6_CFG_PATH,
        D3_PATH,
        D4_PATH,
        D5_PATH,
        Path(d1.__file__),
        Path(d3.__file__),
        Path(d3.r7.__file__),
        Path(d3.r3.__file__),
        Path(__file__),
        result_path,
        profile_csv,
        holdout_csv,
        boundary_csv,
    ]

    manifest = "\n".join(
        f"{sha256_file(path)}  {path}"
        for path in manifest_files
    ) + "\n"
    atomic_write(manifest_path, manifest)

    print("=== P3-B1-R7-D6 DECISION ===")
    print(f"OFFGRID_HOLDOUT_POINTS={len(records)}")
    print(
        "HOLDOUT_KERNEL_COMPONENT_NODES_ANY_PROFILE="
        f"{holdout_kernel_nodes_any_profile}"
    )
    print(
        "CONSENSUS_INTERIOR_MIDPOINT_SURVIVAL_FRACTION="
        f"{consensus_interior_survival:.12g}"
    )
    print(
        "ALL_HOLDOUT_PROFILE_AGREEMENT_FRACTION="
        f"{all_agreement_fraction:.12g}"
    )
    print(
        "MAX_BOUNDARY_REENTRY_FRACTION="
        f"{max_reentry_observed:.12g}"
    )
    print(
        "MIN_INTERIOR_P05_ROBUST_RADIUS_M="
        f"{min_interior_p05_radius:.12g}"
    )
    print(
        "MIN_INTERIOR_P05_ROBUST_RADIUS_MULTIPLE_OF_TOL="
        f"{interior_p05_radius_multiple:.12g}"
    )
    print(
        "ACTION_CONFOUNDED_CONSENSUS_SWITCH_HOLDOUTS="
        f"{action_confounded_consensus}"
    )
    print(f"INTERPRETATION={interpretation}")
    print(f"RECOMMENDED_NEXT_STEP={next_step}")
    print(f"P3B1_R7D6_LOCAL_OFFGRID_HOLDOUT={status}")
    print(
        "MODEL_LEVEL_SUPERVISORY_Q_SENSITIVITY_CLAIM_AUTHORIZED="
        + ("YES" if model_level_claim_authorized else "NO")
    )
    print("SCIENTIFIC_KERNEL_CLAIM=NO")
    print("CONTINUOUS_DOMAIN_GLOBAL_CLAIM=NO")
    print("IMPLEMENTATION_REFINEMENT_CLAIM=NO")
    print(f"RESULT_JSON={result_path}")
    print(f"MANIFEST={manifest_path}")

    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
