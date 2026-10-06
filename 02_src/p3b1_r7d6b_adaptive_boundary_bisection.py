from __future__ import annotations

import csv
import hashlib
import json
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

P1_CFG_PATH = ROOT / "01_config" / "p1_validation_v2.json"
P2B_CFG_PATH = ROOT / "01_config" / "p2b_hybrid_fallback_v1.json"
P2C_CFG_PATH = ROOT / "01_config" / "p2c_switching_guard_v1.json"
P3A_CFG_PATH = ROOT / "01_config" / "p3a_information_contract_v1.json"
D6B_CFG_PATH = (
    ROOT / "01_config" / "p3b1_r7d6b_adaptive_boundary_protocol_v1.json"
)

D4_PATH = ROOT / "04_results" / "P3B1_R7D4_LATEST.json"
D5_PATH = ROOT / "04_results" / "P3B1_R7D5_LATEST.json"
D6_PATH = ROOT / "04_results" / "P3B1_R7D6_LATEST.json"

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


def q_label_change(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[1] <= 1:
        return np.zeros(matrix.shape[0], dtype=bool)
    return np.any(matrix != matrix[:, [0]], axis=1)


def switch_geometry(raw: np.ndarray, clipped: np.ndarray, tol: float) -> dict:
    raw_span = np.max(raw, axis=1) - np.min(raw, axis=1)
    kernel_component = np.max(clipped, axis=1) - np.min(clipped, axis=1)
    switch_width = np.maximum(raw_span - kernel_component, 0.0)
    switch_mask = switch_width > (2.0 * tol)
    return {
        "raw_span": raw_span,
        "kernel_component": kernel_component,
        "switch_width": switch_width,
        "switch_mask": switch_mask,
        "robust_radius": 0.5 * switch_width,
    }


def enumerate_boundary_edges(
    switch_mask: np.ndarray,
    eval_shape: tuple[int, ...],
):
    node_grid = np.arange(
        int(np.prod(eval_shape)),
        dtype=int,
    ).reshape(eval_shape)

    edges = []

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

        xor = ma ^ mb
        for i, j, ia in zip(a[xor], b[xor], ma[xor]):
            if bool(ia):
                w, n = int(i), int(j)
            else:
                w, n = int(j), int(i)

            edges.append(
                {
                    "edge_id": len(edges),
                    "axis": int(axis),
                    "witness_node": w,
                    "nonwitness_node": n,
                }
            )

    return edges


def make_edge_points(
    edges,
    state: np.ndarray,
    subdivisions: int,
):
    t = np.linspace(
        0.0,
        1.0,
        subdivisions + 1,
        dtype=float,
    )

    points = []
    edge_ids = []
    local_indices = []

    for local, edge in enumerate(edges):
        w = state[edge["witness_node"]]
        n = state[edge["nonwitness_node"]]
        x = (
            (1.0 - t)[:, None] * w[None, :]
            +
            t[:, None] * n[None, :]
        )
        points.append(x)
        edge_ids.extend([edge["edge_id"]] * len(t))
        local_indices.extend([local] * len(t))

    matrix = np.vstack(points)

    flat = [
        np.asarray(matrix[:, k], dtype=float)
        for k in range(matrix.shape[1])
    ]

    return t, matrix, flat, np.asarray(edge_ids), np.asarray(local_indices)


def selected_lookup_signature(
    action: np.ndarray,
    transition_data: dict,
):
    """
    Use q=0 action selection for a spatial lookup signature.

    D4-D6 found no q-dependent best-action change on the switch region.
    We still return a q-action-change mask separately so any violation is
    explicitly visible.
    """
    action = np.asarray(action)
    selected = action[:, 0].astype(int)
    n = len(selected)

    completion = np.full(n, -1, dtype=np.int64)
    defer = np.full(n, -1, dtype=np.int64)

    for a in np.unique(selected):
        mask = selected == a
        trans = transition_data["transitions"][int(a)]
        completion[mask] = np.asarray(
            trans["completion_index"],
            dtype=np.int64,
        )[mask]
        defer[mask] = np.asarray(
            trans["defer_index"],
            dtype=np.int64,
        )[mask]

    return selected, completion, defer


def signature_changed(
    a0, c0, d0,
    a1, c1, d1,
):
    return bool(
        (a0 != a1)
        or (c0 != c1)
        or (d0 != d1)
    )


def evaluate_batch(
    edges,
    state,
    subdivisions,
    data,
    p1_cfg,
    p2b_cfg,
    p2c_cfg,
    p3a_cfg,
    diagnostic_names,
    tol,
):
    t, matrix, flat, edge_ids, local_indices = make_edge_points(
        edges,
        state,
        subdivisions,
    )

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

    masks = {}
    geoms = {}
    actions = {}
    branches = {}
    signatures = {}
    q_action_change = {}

    for name in diagnostic_names:
        raw, clipped, branch, action = d3.raw_operator_slices(
            data["results"][name],
            data["lookup_shape"],
            transition_data,
            fallback,
        )

        geom = switch_geometry(raw, clipped, tol)

        masks[name] = geom["switch_mask"]
        geoms[name] = geom
        actions[name] = action
        branches[name] = branch
        q_action_change[name] = q_label_change(action)
        signatures[name] = selected_lookup_signature(
            action,
            transition_data,
        )

    return {
        "t": t,
        "matrix": matrix,
        "flat": flat,
        "edge_ids": edge_ids,
        "local_indices": local_indices,
        "transition_data": transition_data,
        "fallback": fallback,
        "masks": masks,
        "geoms": geoms,
        "actions": actions,
        "branches": branches,
        "q_action_change": q_action_change,
        "signatures": signatures,
        "completion_invalid": completion_invalid,
        "defer_invalid": defer_invalid,
    }


def stage_scan(
    edges,
    state,
    subdivisions,
    batch_size,
    data,
    p1_cfg,
    p2b_cfg,
    p2c_cfg,
    p3a_cfg,
    diagnostic_names,
    tol,
    collect_transition_rows=False,
):
    stage_rows = []
    transition_rows = []
    kernel_nodes_any_profile = 0
    profile_mismatch_points = 0
    q_action_change_on_switch = 0
    completion_invalid = 0
    defer_invalid = 0
    endpoint_replay_errors = 0

    axis_reentry_counts = Counter()
    transition_signature_same = 0
    transition_signature_changed = 0

    for start in range(0, len(edges), batch_size):
        batch = edges[start:start + batch_size]

        result = evaluate_batch(
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

        profile_masks = [
            result["masks"][name]
            for name in diagnostic_names
        ]
        agree = (
            (profile_masks[0] == profile_masks[1])
            &
            (profile_masks[1] == profile_masks[2])
        )
        profile_mismatch_points += int(np.count_nonzero(~agree))

        kernel_union = np.zeros(len(agree), dtype=bool)
        for name in diagnostic_names:
            kernel_union |= (
                result["geoms"][name]["kernel_component"] > tol
            )
            q_action_change_on_switch += int(
                np.count_nonzero(
                    result["q_action_change"][name]
                    &
                    result["masks"][name]
                )
            )

        kernel_nodes_any_profile += int(
            np.count_nonzero(kernel_union)
        )

        for local, edge in enumerate(batch):
            lo = local * npt
            hi = lo + npt

            row = {
                "edge_id": int(edge["edge_id"]),
                "axis": int(edge["axis"]),
                "axis_name": AXIS_NAMES[int(edge["axis"])],
                "witness_node": int(edge["witness_node"]),
                "nonwitness_node": int(edge["nonwitness_node"]),
                "subdivisions": int(subdivisions),
            }

            transition_union = set()

            for name in diagnostic_names:
                bits = result["masks"][name][lo:hi]
                trans = np.flatnonzero(bits[:-1] != bits[1:])

                if not bool(bits[0]) or bool(bits[-1]):
                    endpoint_replay_errors += 1

                row[f"{name}_transition_count"] = int(len(trans))
                row[f"{name}_reentry"] = bool(len(trans) > 1)
                row[f"{name}_transition_t"] = ";".join(
                    f"{(int(k) + 0.5) / subdivisions:.12g}"
                    for k in trans
                )

                transition_union.update(int(k) for k in trans)

            row["profiles_transition_counts_equal"] = (
                row["diagnostic_fast_transition_count"]
                ==
                row["diagnostic_nominal_transition_count"]
                ==
                row["diagnostic_stressed_transition_count"]
            )

            row["any_profile_reentry"] = any(
                bool(row[f"{name}_reentry"])
                for name in diagnostic_names
            )

            if row["any_profile_reentry"]:
                axis_reentry_counts[AXIS_NAMES[int(edge["axis"])]] += 1

            # Mechanism attribution uses diagnostic_fast. D6 showed exact
            # profile agreement; any disagreement is reported separately.
            fast_bits = result["masks"]["diagnostic_fast"][lo:hi]
            fast_trans = np.flatnonzero(
                fast_bits[:-1] != fast_bits[1:]
            )

            a, c, d = result["signatures"]["diagnostic_fast"]
            a = a[lo:hi]
            c = c[lo:hi]
            d = d[lo:hi]

            same_count = 0
            changed_count = 0

            for k in fast_trans:
                changed = signature_changed(
                    a[k], c[k], d[k],
                    a[k + 1], c[k + 1], d[k + 1],
                )
                if changed:
                    changed_count += 1
                    transition_signature_changed += 1
                else:
                    same_count += 1
                    transition_signature_same += 1

                if collect_transition_rows:
                    transition_rows.append(
                        {
                            "edge_id": int(edge["edge_id"]),
                            "axis": int(edge["axis"]),
                            "axis_name":
                                AXIS_NAMES[int(edge["axis"])],
                            "witness_node":
                                int(edge["witness_node"]),
                            "nonwitness_node":
                                int(edge["nonwitness_node"]),
                            "subdivisions":
                                int(subdivisions),
                            "interval_index":
                                int(k),
                            "t_left":
                                float(k / subdivisions),
                            "t_right":
                                float((k + 1) / subdivisions),
                            "t_mid":
                                float((k + 0.5) / subdivisions),
                            "left_switch":
                                bool(fast_bits[k]),
                            "right_switch":
                                bool(fast_bits[k + 1]),
                            "lookup_signature_changed":
                                bool(changed),
                            "left_action":
                                int(a[k]),
                            "right_action":
                                int(a[k + 1]),
                            "left_completion_index":
                                int(c[k]),
                            "right_completion_index":
                                int(c[k + 1]),
                            "left_defer_index":
                                int(d[k]),
                            "right_defer_index":
                                int(d[k + 1]),
                        }
                    )

            row["fast_transition_lookup_signature_changed"] = changed_count
            row["fast_transition_lookup_signature_same"] = same_count

            stage_rows.append(row)

    summary = {
        "edge_count": int(len(edges)),
        "subdivisions": int(subdivisions),
        "reentry_edges": int(
            sum(bool(row["any_profile_reentry"]) for row in stage_rows)
        ),
        "single_transition_edges": int(
            sum(
                int(
                    not row["any_profile_reentry"]
                    and
                    row["diagnostic_fast_transition_count"] == 1
                )
                for row in stage_rows
            )
        ),
        "zero_transition_edges": int(
            sum(
                int(row["diagnostic_fast_transition_count"] == 0)
                for row in stage_rows
            )
        ),
        "max_transition_count": int(
            max(
                row["diagnostic_fast_transition_count"]
                for row in stage_rows
            ) if stage_rows else 0
        ),
        "kernel_nodes_any_profile":
            int(kernel_nodes_any_profile),
        "profile_mismatch_points":
            int(profile_mismatch_points),
        "q_action_change_on_switch_observations":
            int(q_action_change_on_switch),
        "completion_invalid":
            int(completion_invalid),
        "defer_invalid":
            int(defer_invalid),
        "endpoint_replay_errors":
            int(endpoint_replay_errors),
        "transition_lookup_signature_changed":
            int(transition_signature_changed),
        "transition_lookup_signature_same":
            int(transition_signature_same),
        "axis_reentry_counts":
            dict(axis_reentry_counts),
    }

    return summary, stage_rows, transition_rows


def make_ulp_points(
    transition_rows,
    state,
):
    rows = []
    points = []

    for transition_id, tr in enumerate(transition_rows):
        w = state[int(tr["witness_node"])]
        n = state[int(tr["nonwitness_node"])]
        t = float(tr["t_mid"])
        mid = (1.0 - t) * w + t * n

        toward_w = np.nextafter(mid, w)
        toward_n = np.nextafter(mid, n)

        for variant, x in (
            ("toward_witness", toward_w),
            ("mid", mid),
            ("toward_nonwitness", toward_n),
        ):
            rows.append(
                {
                    "transition_id": transition_id,
                    "edge_id": int(tr["edge_id"]),
                    "axis": int(tr["axis"]),
                    "variant": variant,
                }
            )
            points.append(x)

    matrix = np.asarray(points, dtype=float)
    flat = [
        np.asarray(matrix[:, k], dtype=float)
        for k in range(matrix.shape[1])
    ]
    return rows, matrix, flat


def ulp_audit(
    transition_rows,
    state,
    batch_size,
    data,
    p1_cfg,
    p2b_cfg,
    p2c_cfg,
    p3a_cfg,
    diagnostic_names,
    tol,
):
    if not transition_rows:
        return {
            "ulp_transition_count": 0,
            "ulp_switch_sensitive_transitions": 0,
            "ulp_lookup_signature_sensitive_transitions": 0,
            "ulp_profile_disagreement_points": 0,
            "kernel_nodes_any_profile": 0,
            "completion_invalid": 0,
            "defer_invalid": 0,
        }, []

    out_rows = []
    switch_sensitive = 0
    signature_sensitive = 0
    profile_disagreement_points = 0
    kernel_nodes_any_profile = 0
    completion_invalid = 0
    defer_invalid = 0

    for start in range(0, len(transition_rows), batch_size):
        trs = transition_rows[start:start + batch_size]
        meta, matrix, flat = make_ulp_points(trs, state)

        transition_data = d3.r7.build_transition_data_to_lookup(
            data["eval_cfg"],
            p1_cfg,
            p2b_cfg,
            p2c_cfg,
            p3a_cfg,
            flat,
            data["lookup_axes"],
        )

        completion_invalid += sum(
            int(np.count_nonzero(~tr["completion_valid"]))
            for tr in transition_data["transitions"]
        )
        defer_invalid += sum(
            int(np.count_nonzero(~tr["defer_valid"]))
            for tr in transition_data["transitions"]
        )

        fallback = np.asarray(
            transition_data["fallback_required"],
            dtype=float,
        )

        masks = {}
        signatures = {}
        kernel_union = np.zeros(len(meta), dtype=bool)

        for name in diagnostic_names:
            raw, clipped, branch, action = d3.raw_operator_slices(
                data["results"][name],
                data["lookup_shape"],
                transition_data,
                fallback,
            )

            geom = switch_geometry(raw, clipped, tol)
            masks[name] = geom["switch_mask"]
            kernel_union |= geom["kernel_component"] > tol
            signatures[name] = selected_lookup_signature(
                action,
                transition_data,
            )

        kernel_nodes_any_profile += int(
            np.count_nonzero(kernel_union)
        )

        agree = (
            (masks[diagnostic_names[0]]
             == masks[diagnostic_names[1]])
            &
            (masks[diagnostic_names[1]]
             == masks[diagnostic_names[2]])
        )
        profile_disagreement_points += int(
            np.count_nonzero(~agree)
        )

        for local, tr in enumerate(trs):
            lo = local * 3
            hi = lo + 3

            fast_bits = masks["diagnostic_fast"][lo:hi]
            switch_is_sensitive = not (
                bool(fast_bits[0])
                ==
                bool(fast_bits[1])
                ==
                bool(fast_bits[2])
            )

            a, c, d = signatures["diagnostic_fast"]
            sig = [
                (int(a[k]), int(c[k]), int(d[k]))
                for k in range(lo, hi)
            ]
            sig_is_sensitive = not (
                sig[0] == sig[1] == sig[2]
            )

            switch_sensitive += int(switch_is_sensitive)
            signature_sensitive += int(sig_is_sensitive)

            out_rows.append(
                {
                    "transition_id": int(start + local),
                    "edge_id": int(tr["edge_id"]),
                    "axis": int(tr["axis"]),
                    "axis_name": AXIS_NAMES[int(tr["axis"])],
                    "t_mid": float(tr["t_mid"]),
                    "toward_witness_switch": bool(fast_bits[0]),
                    "mid_switch": bool(fast_bits[1]),
                    "toward_nonwitness_switch": bool(fast_bits[2]),
                    "ulp_switch_sensitive": bool(switch_is_sensitive),
                    "toward_witness_signature": str(sig[0]),
                    "mid_signature": str(sig[1]),
                    "toward_nonwitness_signature": str(sig[2]),
                    "ulp_lookup_signature_sensitive":
                        bool(sig_is_sensitive),
                    "profiles_agree_all_three_variants":
                        bool(np.all(agree[lo:hi])),
                }
            )

    summary = {
        "ulp_transition_count":
            int(len(transition_rows)),
        "ulp_switch_sensitive_transitions":
            int(switch_sensitive),
        "ulp_lookup_signature_sensitive_transitions":
            int(signature_sensitive),
        "ulp_profile_disagreement_points":
            int(profile_disagreement_points),
        "kernel_nodes_any_profile":
            int(kernel_nodes_any_profile),
        "completion_invalid":
            int(completion_invalid),
        "defer_invalid":
            int(defer_invalid),
    }

    return summary, out_rows


def main() -> int:
    cfg = load_json(D6B_CFG_PATH)
    tol = float(cfg["numeric_tolerance_m"])
    stage1_subdiv = int(cfg["stage1_subdivisions"])
    stage2_subdiv = int(cfg["stage2_subdivisions"])
    batch1 = int(cfg["edge_batch_size_stage1"])
    batch2 = int(cfg["edge_batch_size_stage2"])
    ulp_batch = int(cfg["ulp_transition_batch_size"])

    d4_latest = load_json(D4_PATH)
    d5_latest = load_json(D5_PATH)
    d6_latest = load_json(D6_PATH)

    if d4_latest.get("status") != "PASS":
        raise RuntimeError("P3B1_R7D6B_D4_UPSTREAM_FAIL")
    if d5_latest.get("status") != "PASS":
        raise RuntimeError("P3B1_R7D6B_D5_UPSTREAM_FAIL")
    if d6_latest.get("status") != "PASS":
        raise RuntimeError("P3B1_R7D6B_D6_UPSTREAM_FAIL")

    expected_d6 = "OFFGRID_BOUNDARY_SHOWS_SUBCELL_REENTRY_OR_OSCILLATION"
    if d6_latest.get("interpretation") != expected_d6:
        raise RuntimeError(
            "P3B1_R7D6B_UNEXPECTED_D6_INTERPRETATION="
            + str(d6_latest.get("interpretation"))
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
    base_geom = switch_geometry(raw, clipped, tol)
    base_switch = base_geom["switch_mask"]

    edges = enumerate_boundary_edges(
        base_switch,
        tuple(int(x) for x in data["eval_shape"]),
    )

    d6_boundary_edges = int(
        d6_latest["aggregate"]["boundary_witness_edges"]
    )
    d6_reentry_fraction = float(
        d6_latest["aggregate"]["max_boundary_reentry_fraction"]
    )
    d6_reentry_lower_bound = int(
        round(d6_boundary_edges * d6_reentry_fraction)
    )

    print("=== P3-B1-R7-D6B ADAPTIVE BOUNDARY BISECTION ===")
    print(f"BOUNDARY_EDGES={len(edges)}")
    print(f"D6_REENTRY_EDGES={d6_reentry_lower_bound}")
    print(f"STAGE1_SUBDIVISIONS={stage1_subdiv}")
    print(f"STAGE2_SUBDIVISIONS={stage2_subdiv}")

    stage1_summary, stage1_rows, _ = stage_scan(
        edges,
        state,
        stage1_subdiv,
        batch1,
        data,
        p1_cfg,
        p2b_cfg,
        p2c_cfg,
        p3a_cfg,
        diagnostic_names,
        tol,
        collect_transition_rows=False,
    )

    candidate_ids = {
        int(row["edge_id"])
        for row in stage1_rows
        if bool(row["any_profile_reentry"])
    }
    edge_by_id = {
        int(edge["edge_id"]): edge
        for edge in edges
    }
    candidates = [
        edge_by_id[i]
        for i in sorted(candidate_ids)
    ]

    print("=== STAGE 1: ALL EDGE DYADIC SCAN ===")
    print(
        f"STAGE1_REENTRY_EDGES={stage1_summary['reentry_edges']}"
    )
    print(
        f"STAGE1_MAX_TRANSITIONS={stage1_summary['max_transition_count']}"
    )
    print(
        "STAGE1_PROFILE_MISMATCH_POINTS="
        f"{stage1_summary['profile_mismatch_points']}"
    )
    print(
        "STAGE1_KERNEL_NODES_ANY_PROFILE="
        f"{stage1_summary['kernel_nodes_any_profile']}"
    )

    stage2_summary, stage2_rows, transition_rows = stage_scan(
        candidates,
        state,
        stage2_subdiv,
        batch2,
        data,
        p1_cfg,
        p2b_cfg,
        p2c_cfg,
        p3a_cfg,
        diagnostic_names,
        tol,
        collect_transition_rows=True,
    )

    print("=== STAGE 2: ADAPTIVE REENTRY REFINEMENT ===")
    print(
        f"STAGE2_REFINED_EDGES={stage2_summary['edge_count']}"
    )
    print(
        f"STAGE2_REENTRY_EDGES={stage2_summary['reentry_edges']}"
    )
    print(
        f"STAGE2_MAX_TRANSITIONS={stage2_summary['max_transition_count']}"
    )
    print(
        "STAGE2_LOOKUP_SIGNATURE_CHANGED_TRANSITIONS="
        f"{stage2_summary['transition_lookup_signature_changed']}"
    )
    print(
        "STAGE2_LOOKUP_SIGNATURE_SAME_TRANSITIONS="
        f"{stage2_summary['transition_lookup_signature_same']}"
    )
    print(
        "STAGE2_PROFILE_MISMATCH_POINTS="
        f"{stage2_summary['profile_mismatch_points']}"
    )
    print(
        "STAGE2_KERNEL_NODES_ANY_PROFILE="
        f"{stage2_summary['kernel_nodes_any_profile']}"
    )

    ulp_summary, ulp_rows = ulp_audit(
        transition_rows,
        state,
        ulp_batch,
        data,
        p1_cfg,
        p2b_cfg,
        p2c_cfg,
        p3a_cfg,
        diagnostic_names,
        tol,
    )

    print("=== TRANSITION ULP STRESS ===")
    print(
        f"ULP_TRANSITIONS={ulp_summary['ulp_transition_count']}"
    )
    print(
        "ULP_SWITCH_SENSITIVE_TRANSITIONS="
        f"{ulp_summary['ulp_switch_sensitive_transitions']}"
    )
    print(
        "ULP_LOOKUP_SIGNATURE_SENSITIVE_TRANSITIONS="
        f"{ulp_summary['ulp_lookup_signature_sensitive_transitions']}"
    )
    print(
        "ULP_PROFILE_DISAGREEMENT_POINTS="
        f"{ulp_summary['ulp_profile_disagreement_points']}"
    )

    total_stage2_transitions = (
        stage2_summary["transition_lookup_signature_changed"]
        +
        stage2_summary["transition_lookup_signature_same"]
    )

    if total_stage2_transitions > 0:
        signature_change_fraction = (
            stage2_summary["transition_lookup_signature_changed"]
            /
            total_stage2_transitions
        )
    else:
        signature_change_fraction = 0.0

    all_kernel_nodes = (
        stage1_summary["kernel_nodes_any_profile"]
        +
        stage2_summary["kernel_nodes_any_profile"]
        +
        ulp_summary["kernel_nodes_any_profile"]
    )

    all_profile_mismatches = (
        stage1_summary["profile_mismatch_points"]
        +
        stage2_summary["profile_mismatch_points"]
        +
        ulp_summary["ulp_profile_disagreement_points"]
    )

    all_coverage_invalid = (
        stage1_summary["completion_invalid"]
        +
        stage1_summary["defer_invalid"]
        +
        stage2_summary["completion_invalid"]
        +
        stage2_summary["defer_invalid"]
        +
        ulp_summary["completion_invalid"]
        +
        ulp_summary["defer_invalid"]
    )

    endpoint_errors = (
        stage1_summary["endpoint_replay_errors"]
        +
        stage2_summary["endpoint_replay_errors"]
    )

    d6_reentry_reproduced = (
        stage1_summary["reentry_edges"]
        >= d6_reentry_lower_bound
    )

    ulp_stable = (
        ulp_summary["ulp_switch_sensitive_transitions"] == 0
        and
        ulp_summary["ulp_lookup_signature_sensitive_transitions"] == 0
    )

    if all_kernel_nodes > 0:
        interpretation = (
            "ADAPTIVE_BOUNDARY_REPLAY_REVEALS_Q_DEPENDENT_KERNEL_COMPONENT"
        )
        next_step = (
            "STOP_SUPERVISOR_PROMOTION_AND_RECOMPUTE_REFINED_Q_DEPENDENT_KERNEL"
        )
    elif all_profile_mismatches > 0:
        interpretation = (
            "ADAPTIVE_BOUNDARY_GEOMETRY_IS_SERVICE_PROFILE_SPECIFIC"
        )
        next_step = (
            "RUN_PROFILE_SPECIFIC_REFINED_BOUNDARY_AND_KERNEL_AUDIT"
        )
    elif not ulp_stable:
        interpretation = (
            "BOUNDARY_REENTRY_IS_MACHINE_ULP_SENSITIVE"
        )
        next_step = (
            "RUN_HIGHER_PRECISION_TRANSITION_AND_INDEX_ARITHMETIC_BEFORE_GEOMETRIC_INTERPRETATION"
        )
    elif (
        stage2_summary["transition_lookup_signature_same"] == 0
        and total_stage2_transitions > 0
    ):
        interpretation = (
            "ALL_REFINED_REENTRY_TRANSITIONS_COINCIDE_WITH_LOOKUP_CELL_SIGNATURE_CHANGES"
        )
        next_step = (
            "RUN_P3B1_R7D6C_LOCAL_HALO_REFINEMENT_AND_CONTINUOUS_LOOKUP_ATTRIBUTION"
        )
    elif (
        stage2_summary["transition_lookup_signature_changed"] == 0
        and total_stage2_transitions > 0
    ):
        interpretation = (
            "REFINED_REENTRY_PERSISTS_WITHIN_FIXED_LOOKUP_SIGNATURES"
        )
        next_step = (
            "RUN_P3B1_R7D6C_WITHIN_CELL_NONMONOTONE_SWITCH_GEOMETRY_AUDIT"
        )
    else:
        interpretation = (
            "REFINED_REENTRY_HAS_MIXED_LOOKUP_CELL_AND_WITHIN_CELL_MECHANISMS"
        )
        next_step = (
            "RUN_P3B1_R7D6C_LOCAL_HALO_REFINEMENT_WITH_MECHANISM_SEPARATION"
        )

    integrity_checks = {
        "D4_UPSTREAM_PASS":
            d4_latest.get("status") == "PASS",
        "D5_UPSTREAM_PASS":
            d5_latest.get("status") == "PASS",
        "D6_UPSTREAM_PASS":
            d6_latest.get("status") == "PASS",
        "D6_REENTRY_INTERPRETATION_MATCH":
            d6_latest.get("interpretation") == expected_d6,
        "BOUNDARY_EDGE_COUNT_REPRODUCED":
            len(edges) == d6_boundary_edges,
        "D6_REENTRY_LOWER_BOUND_REPRODUCED":
            bool(d6_reentry_reproduced),
        "TRANSITION_COVERAGE_COMPLETE":
            all_coverage_invalid == 0,
        "EDGE_ENDPOINT_REPLAY":
            endpoint_errors == 0,
    }

    status = "PASS" if all(integrity_checks.values()) else "FAIL"

    scientific_diagnostics = {
        "q_dependent_kernel_component_detected":
            bool(all_kernel_nodes > 0),
        "service_profile_disagreement_detected":
            bool(all_profile_mismatches > 0),
        "machine_ulp_switch_sensitivity_detected":
            bool(
                ulp_summary["ulp_switch_sensitive_transitions"] > 0
            ),
        "machine_ulp_lookup_signature_sensitivity_detected":
            bool(
                ulp_summary[
                    "ulp_lookup_signature_sensitive_transitions"
                ] > 0
            ),
        "all_refined_transitions_lookup_signature_aligned":
            bool(
                total_stage2_transitions > 0
                and
                stage2_summary[
                    "transition_lookup_signature_same"
                ] == 0
            ),
        "within_signature_transition_detected":
            bool(
                stage2_summary[
                    "transition_lookup_signature_same"
                ] > 0
            ),
    }

    stamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    stage1_csv = (
        RESULTS_DIR
        / f"P3B1_R7D6B_STAGE1_EDGE_SCAN_{stamp}.csv"
    )
    stage2_csv = (
        RESULTS_DIR
        / f"P3B1_R7D6B_STAGE2_REFINED_EDGES_{stamp}.csv"
    )
    transition_csv = (
        RESULTS_DIR
        / f"P3B1_R7D6B_REFINED_TRANSITIONS_{stamp}.csv"
    )
    ulp_csv = (
        RESULTS_DIR
        / f"P3B1_R7D6B_TRANSITION_ULP_AUDIT_{stamp}.csv"
    )

    write_csv(stage1_csv, stage1_rows)
    write_csv(stage2_csv, stage2_rows)
    write_csv(transition_csv, transition_rows)
    write_csv(ulp_csv, ulp_rows)

    output = {
        "schema":
            "SCV_P3B1_R7D6B_ADAPTIVE_BOUNDARY_BISECTION_V1",
        "status":
            status,
        "timestamp_utc":
            stamp,
        "classification":
            (
                "adaptive local boundary-topology and lookup-cell attribution "
                "audit; diagnostic only, with no kernel, global continuous-"
                "domain, or implementation-refinement claim authorization"
            ),
        "integrity_checks":
            {k: bool(v) for k, v in integrity_checks.items()},
        "scientific_diagnostics":
            scientific_diagnostics,
        "interpretation":
            interpretation,
        "recommended_next_step":
            next_step,
        "protocol":
            cfg,
        "stage1":
            stage1_summary,
        "stage2":
            stage2_summary,
        "ulp_audit":
            ulp_summary,
        "aggregate": {
            "d6_boundary_edges":
                int(d6_boundary_edges),
            "d6_reentry_edges":
                int(d6_reentry_lower_bound),
            "stage1_reentry_edges":
                int(stage1_summary["reentry_edges"]),
            "stage2_refined_edges":
                int(stage2_summary["edge_count"]),
            "stage2_reentry_edges":
                int(stage2_summary["reentry_edges"]),
            "stage2_total_switch_transitions":
                int(total_stage2_transitions),
            "stage2_lookup_signature_changed_transitions":
                int(
                    stage2_summary[
                        "transition_lookup_signature_changed"
                    ]
                ),
            "stage2_lookup_signature_same_transitions":
                int(
                    stage2_summary[
                        "transition_lookup_signature_same"
                    ]
                ),
            "stage2_lookup_signature_change_fraction":
                float(signature_change_fraction),
            "all_kernel_component_nodes":
                int(all_kernel_nodes),
            "all_profile_mismatch_points":
                int(all_profile_mismatches),
            "all_coverage_invalid":
                int(all_coverage_invalid),
            "endpoint_replay_errors":
                int(endpoint_errors),
        },
        "claims": {
            "model_level_supervisory_claim_authorized":
                False,
            "scientific_kernel_claim_authorized":
                False,
            "continuous_domain_global_claim_authorized":
                False,
            "implementation_refinement_claim_authorized":
                False,
            "p6_certified":
                False,
            "reason":
                (
                    "R7-D6B attributes the D6 reentry mechanism but does not "
                    "remove it. Claim promotion remains blocked until the "
                    "identified lookup-cell or within-cell mechanism is "
                    "resolved by the next targeted refinement audit."
                ),
        },
        "artifacts": {
            "stage1_edge_csv": str(stage1_csv),
            "stage2_edge_csv": str(stage2_csv),
            "transition_csv": str(transition_csv),
            "ulp_csv": str(ulp_csv),
        },
    }

    result_path = (
        RESULTS_DIR
        / f"P3B1_R7D6B_ADAPTIVE_BOUNDARY_AUDIT_{stamp}.json"
    )
    latest_path = RESULTS_DIR / "P3B1_R7D6B_LATEST.json"
    manifest_path = (
        RESULTS_DIR
        / f"P3B1_R7D6B_MANIFEST_{stamp}.sha256"
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
        D6B_CFG_PATH,
        D4_PATH,
        D5_PATH,
        D6_PATH,
        Path(d1.__file__),
        Path(d3.__file__),
        Path(d3.r7.__file__),
        Path(d3.r3.__file__),
        Path(__file__),
        result_path,
        stage1_csv,
        stage2_csv,
        transition_csv,
        ulp_csv,
    ]

    manifest = "\n".join(
        f"{sha256_file(path)}  {path}"
        for path in manifest_files
    ) + "\n"
    atomic_write(manifest_path, manifest)

    print("=== P3-B1-R7-D6B DECISION ===")
    print(f"D6_REENTRY_EDGES={d6_reentry_lower_bound}")
    print(f"STAGE1_REENTRY_EDGES={stage1_summary['reentry_edges']}")
    print(f"STAGE2_REFINED_EDGES={stage2_summary['edge_count']}")
    print(f"STAGE2_REENTRY_EDGES={stage2_summary['reentry_edges']}")
    print(
        "STAGE2_TOTAL_SWITCH_TRANSITIONS="
        f"{total_stage2_transitions}"
    )
    print(
        "LOOKUP_SIGNATURE_CHANGED_TRANSITIONS="
        f"{stage2_summary['transition_lookup_signature_changed']}"
    )
    print(
        "LOOKUP_SIGNATURE_SAME_TRANSITIONS="
        f"{stage2_summary['transition_lookup_signature_same']}"
    )
    print(
        "LOOKUP_SIGNATURE_CHANGE_FRACTION="
        f"{signature_change_fraction:.12g}"
    )
    print(
        "ULP_SWITCH_SENSITIVE_TRANSITIONS="
        f"{ulp_summary['ulp_switch_sensitive_transitions']}"
    )
    print(
        "ULP_LOOKUP_SIGNATURE_SENSITIVE_TRANSITIONS="
        f"{ulp_summary['ulp_lookup_signature_sensitive_transitions']}"
    )
    print(f"ALL_KERNEL_COMPONENT_NODES={all_kernel_nodes}")
    print(f"ALL_PROFILE_MISMATCH_POINTS={all_profile_mismatches}")
    print(f"INTERPRETATION={interpretation}")
    print(f"RECOMMENDED_NEXT_STEP={next_step}")
    print(f"P3B1_R7D6B_ADAPTIVE_BOUNDARY_AUDIT={status}")
    print("MODEL_LEVEL_SUPERVISORY_CLAIM_AUTHORIZED=NO")
    print("SCIENTIFIC_KERNEL_CLAIM=NO")
    print("CONTINUOUS_DOMAIN_GLOBAL_CLAIM=NO")
    print("IMPLEMENTATION_REFINEMENT_CLAIM=NO")
    print(f"RESULT_JSON={result_path}")
    print(f"MANIFEST={manifest_path}")

    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
