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

D2_PATH = ROOT / "04_results" / "P3B1_R7D2_LATEST.json"
D3_PATH = ROOT / "04_results" / "P3B1_R7D3_LATEST.json"

RESULTS_DIR = ROOT / "04_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

AXIS_NAMES = ("v_f", "v_p", "a_f", "bar_a", "bar_u", "age")
TOL = 1.0e-10
REPLAY_TOL = 1.0e-9
MAX_WITNESSES_PER_PROFILE = 128


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


def positive_stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    mask = x > TOL
    vals = x[mask]
    return {
        "nodes": int(np.count_nonzero(mask)),
        "fraction": float(np.mean(mask)) if x.size else 0.0,
        "mean_positive_m": float(np.mean(vals)) if len(vals) else 0.0,
        "p50_positive_m": float(np.quantile(vals, 0.50)) if len(vals) else 0.0,
        "p95_positive_m": float(np.quantile(vals, 0.95)) if len(vals) else 0.0,
        "max_m": float(np.max(x)) if x.size else 0.0,
    }


def any_discrete_q_change(matrix: np.ndarray) -> np.ndarray:
    """
    True at nodes where an integer-valued q-indexed label changes across q.
    """
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[1] <= 1:
        return np.zeros(matrix.shape[0], dtype=bool)
    return np.any(matrix != matrix[:, [0]], axis=1)


def mode_at_gap(
    d: np.ndarray,
    cooperative_required: np.ndarray,
    fallback_required: np.ndarray,
) -> np.ndarray:
    """
    Certified supervisor semantics:

      C : cooperative mode is certified, d >= h_C
      F : cooperative mode is not certified but fallback is, d >= h_F
      X : neither mode is certified

    Priority is C whenever cooperative service remains viable.
    """
    d = np.asarray(d, dtype=float)
    h_c = np.asarray(cooperative_required, dtype=float)
    h_f = np.asarray(fallback_required, dtype=float)

    out = np.full(np.broadcast(d, h_c, h_f).shape, "X", dtype="<U1")
    c = d + TOL >= h_c
    f = (~c) & (d + TOL >= h_f)
    out[c] = "C"
    out[f] = "F"
    return out


def profile_switch_attribution(
    name: str,
    result: dict,
    raw: np.ndarray,
    clipped: np.ndarray,
    branch: np.ndarray,
    action: np.ndarray,
    eval_fallback: np.ndarray,
    eval_flat,
):
    R = int(result["R"])
    n = len(eval_fallback)

    stored = np.asarray(
        result["h_flat"][d1.reconstruct()["eval_node_indices"], :],
        dtype=float,
    )
    replay_error = float(np.max(np.abs(stored - clipped)))

    raw_min = np.min(raw, axis=1)
    raw_max = np.max(raw, axis=1)
    q_argmin = np.argmin(raw, axis=1)
    q_argmax = np.argmax(raw, axis=1)
    raw_span = raw_max - raw_min

    clipped_min = np.min(clipped, axis=1)
    clipped_max = np.max(clipped, axis=1)
    clipped_span = clipped_max - clipped_min

    # Exact decomposition of latent q sensitivity:
    #
    #   raw q span
    #     = component that changes the viability envelope
    #     + component that only moves the safe C/F supervisor switch surface.
    #
    # Because clipped = min(h_F, h_C), the envelope component is exactly
    # range_q[min(h_F, h_C)].
    kernel_component = clipped_span
    switch_component = np.maximum(raw_span - kernel_component, 0.0)

    decomp_error = float(
        np.max(np.abs(raw_span - kernel_component - switch_component))
    )

    raw_q_mask = raw_span > TOL
    kernel_q_mask = kernel_component > TOL
    switch_q_mask = switch_component > TOL

    # Constructive safe-switch witness interval:
    # choose q_low = argmin h_C, q_high = argmax h_C.
    #
    # For any d in
    #   [max(h_F, h_C(q_low)), h_C(q_high))
    # q_low admits C while q_high admits only F.
    witness_low = np.maximum(eval_fallback, raw_min)
    witness_high = raw_max
    witness_width = np.maximum(witness_high - witness_low, 0.0)
    witness_mask = witness_width > (2.0 * TOL)

    # By construction witness_width equals the switch-only component.
    witness_component_error = float(
        np.max(np.abs(witness_width - switch_component))
    )

    qdep_indices = np.flatnonzero(raw_q_mask)
    node_rows = []

    vf, vp, af, bar_a, bar_u, age = [
        np.asarray(x, dtype=float) for x in eval_flat
    ]

    for i in qdep_indices:
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
                "raw_q_span_m": float(raw_span[i]),
                "kernel_component_m": float(kernel_component[i]),
                "safe_switch_component_m": float(switch_component[i]),
                "q_min_index": int(q_argmin[i]),
                "q_min_steps": int(q_argmin[i] + 1),
                "q_max_index": int(q_argmax[i]),
                "q_max_steps": int(q_argmax[i] + 1),
                "action_at_q_min": int(action[i, q_argmin[i]]),
                "action_at_q_max": int(action[i, q_argmax[i]]),
                "branch_at_q_min": int(branch[i, q_argmin[i]]),
                "branch_at_q_max": int(branch[i, q_argmax[i]]),
                "safe_switch_witness_exists": bool(witness_mask[i]),
                "safe_switch_interval_low_m": float(witness_low[i]),
                "safe_switch_interval_high_m": float(witness_high[i]),
                "safe_switch_interval_width_m": float(witness_width[i]),
            }
        )

    witness_rows = []
    witness_idx = np.flatnonzero(witness_mask)
    if len(witness_idx):
        order = witness_idx[np.argsort(witness_width[witness_idx])[::-1]]
    else:
        order = witness_idx

    for i in order[:MAX_WITNESSES_PER_PROFILE]:
        q_lo = int(q_argmin[i])
        q_hi = int(q_argmax[i])
        d_probe = 0.5 * (witness_low[i] + witness_high[i])

        mode_lo = mode_at_gap(
            np.asarray([d_probe]),
            np.asarray([raw[i, q_lo]]),
            np.asarray([eval_fallback[i]]),
        )[0]
        mode_hi = mode_at_gap(
            np.asarray([d_probe]),
            np.asarray([raw[i, q_hi]]),
            np.asarray([eval_fallback[i]]),
        )[0]

        witness_rows.append(
            {
                "profile": name,
                "node_index": int(i),
                "v_f": float(vf[i]),
                "v_p": float(vp[i]),
                "a_f": float(af[i]),
                "bar_a": float(bar_a[i]),
                "bar_u": float(bar_u[i]),
                "age": float(age[i]),
                "q_cooperative_index": q_lo,
                "q_cooperative_steps": q_lo + 1,
                "q_fallback_index": q_hi,
                "q_fallback_steps": q_hi + 1,
                "fallback_required_m": float(eval_fallback[i]),
                "cooperative_required_low_q_m": float(raw[i, q_lo]),
                "cooperative_required_high_q_m": float(raw[i, q_hi]),
                "probe_gap_m": float(d_probe),
                "mode_at_q_cooperative": str(mode_lo),
                "mode_at_q_fallback": str(mode_hi),
                "switch_interval_width_m": float(witness_width[i]),
                "action_at_q_cooperative": int(action[i, q_lo]),
                "action_at_q_fallback": int(action[i, q_hi]),
                "branch_at_q_cooperative": int(branch[i, q_lo]),
                "branch_at_q_fallback": int(branch[i, q_hi]),
            }
        )

    witness_valid = all(
        row["mode_at_q_cooperative"] == "C"
        and row["mode_at_q_fallback"] == "F"
        and row["probe_gap_m"] + TOL >= row["fallback_required_m"]
        and row["probe_gap_m"] + TOL
            >= row["cooperative_required_low_q_m"]
        and row["probe_gap_m"] + TOL
            < row["cooperative_required_high_q_m"]
        for row in witness_rows
    )

    action_q_change = any_discrete_q_change(action)
    branch_q_change = any_discrete_q_change(branch)

    gain_initial = eval_fallback - clipped[:, -1] > TOL

    summary = {
        "profile": name,
        "R": R,
        "evaluation_nodes": int(n),
        "fixed_point_replay_max_error_m": replay_error,
        "raw_q_dependence_nodes": int(np.count_nonzero(raw_q_mask)),
        "kernel_q_component_nodes": int(np.count_nonzero(kernel_q_mask)),
        "safe_switch_q_component_nodes": int(np.count_nonzero(switch_q_mask)),
        "safe_switch_witness_nodes": int(np.count_nonzero(witness_mask)),
        "raw_q_span_max_m": float(np.max(raw_span)),
        "kernel_q_component_max_m": float(np.max(kernel_component)),
        "safe_switch_component_max_m": float(np.max(switch_component)),
        "safe_switch_component_mean_positive_m":
            positive_stats(switch_component)["mean_positive_m"],
        "safe_switch_component_p95_positive_m":
            positive_stats(switch_component)["p95_positive_m"],
        "q_span_decomposition_max_error_m": decomp_error,
        "witness_component_max_error_m": witness_component_error,
        "best_action_q_change_nodes":
            int(np.count_nonzero(action_q_change)),
        "best_branch_q_change_nodes":
            int(np.count_nonzero(branch_q_change)),
        "gain_nodes_initial_slice":
            int(np.count_nonzero(gain_initial)),
        "gain_nodes_with_safe_switch_q_component":
            int(np.count_nonzero(gain_initial & switch_q_mask)),
        "witness_rows_emitted": int(len(witness_rows)),
        "all_emitted_witnesses_valid": bool(witness_valid),
    }

    return summary, node_rows, witness_rows, {
        "raw": raw,
        "clipped": clipped,
        "switch_component": switch_component,
        "kernel_component": kernel_component,
    }


def main() -> int:
    d2 = load_json(D2_PATH)
    d3_latest = load_json(D3_PATH)

    if d2.get("status") != "PASS":
        raise RuntimeError("P3B1_R7D4_D2_UPSTREAM_FAIL")
    if d3_latest.get("status") != "PASS":
        raise RuntimeError("P3B1_R7D4_D3_UPSTREAM_FAIL")

    expected_d3 = (
        "LATENT_Q_EFFECT_EXISTS_BUT_FALLBACK_CLIPPING_REMOVES_IT_OUTSIDE_GAIN_REGION"
    )

    data = d1.reconstruct()
    transition_data = d3.build_transition_data(data)

    eval_idx = data["eval_node_indices"]
    eval_fallback = np.asarray(data["eval_fallback"], dtype=float)
    eval_flat = data["eval_flat"]
    lookup_shape = data["lookup_shape"]
    results = data["results"]

    print("=== P3-B1-R7-D4 SUPERVISORY SWITCH SEMANTICS AUDIT ===")
    print(f"EVAL_GRID_NODES={len(eval_fallback)}")
    print(
        "SUPERVISOR_SEMANTICS="
        "COOPERATIVE_IF_CERTIFIED_ELSE_FALLBACK_IF_CERTIFIED_ELSE_OUTSIDE"
    )

    profile_rows = []
    node_rows = []
    witness_rows = []
    arrays = {}

    for name, result in results.items():
        raw, clipped, branch, action = d3.raw_operator_slices(
            result,
            lookup_shape,
            transition_data,
            eval_fallback,
        )

        # Recompute stored slice here without calling reconstruct again.
        stored = np.asarray(
            result["h_flat"][eval_idx, :],
            dtype=float,
        )
        replay_error = float(np.max(np.abs(stored - clipped)))

        R = int(result["R"])
        raw_min = np.min(raw, axis=1)
        raw_max = np.max(raw, axis=1)
        q_argmin = np.argmin(raw, axis=1)
        q_argmax = np.argmax(raw, axis=1)
        raw_span = raw_max - raw_min
        clipped_span = np.max(clipped, axis=1) - np.min(clipped, axis=1)

        kernel_component = clipped_span
        switch_component = np.maximum(raw_span - kernel_component, 0.0)
        witness_low = np.maximum(eval_fallback, raw_min)
        witness_high = raw_max
        witness_width = np.maximum(witness_high - witness_low, 0.0)

        decomp_error = float(
            np.max(np.abs(raw_span - kernel_component - switch_component))
        )
        witness_component_error = float(
            np.max(np.abs(witness_width - switch_component))
        )

        raw_q_mask = raw_span > TOL
        kernel_q_mask = kernel_component > TOL
        switch_q_mask = switch_component > TOL
        witness_mask = witness_width > (2.0 * TOL)

        action_q_change = any_discrete_q_change(action)
        branch_q_change = any_discrete_q_change(branch)
        gain_initial = eval_fallback - clipped[:, -1] > TOL

        pstats = positive_stats(switch_component)

        row = {
            "profile": name,
            "R": R,
            "evaluation_nodes": int(len(eval_fallback)),
            "fixed_point_replay_max_error_m": replay_error,
            "raw_q_dependence_nodes": int(np.count_nonzero(raw_q_mask)),
            "kernel_q_component_nodes": int(np.count_nonzero(kernel_q_mask)),
            "safe_switch_q_component_nodes": int(np.count_nonzero(switch_q_mask)),
            "safe_switch_witness_nodes": int(np.count_nonzero(witness_mask)),
            "raw_q_span_max_m": float(np.max(raw_span)),
            "kernel_q_component_max_m": float(np.max(kernel_component)),
            "safe_switch_component_max_m": float(np.max(switch_component)),
            "safe_switch_component_mean_positive_m":
                pstats["mean_positive_m"],
            "safe_switch_component_p95_positive_m":
                pstats["p95_positive_m"],
            "q_span_decomposition_max_error_m": decomp_error,
            "witness_component_max_error_m": witness_component_error,
            "best_action_q_change_nodes":
                int(np.count_nonzero(action_q_change)),
            "best_branch_q_change_nodes":
                int(np.count_nonzero(branch_q_change)),
            "gain_nodes_initial_slice":
                int(np.count_nonzero(gain_initial)),
            "gain_nodes_with_safe_switch_q_component":
                int(np.count_nonzero(gain_initial & switch_q_mask)),
        }
        profile_rows.append(row)

        vf, vp, af, bar_a, bar_u, age = [
            np.asarray(x, dtype=float) for x in eval_flat
        ]

        for i in np.flatnonzero(raw_q_mask):
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
                    "raw_q_span_m": float(raw_span[i]),
                    "kernel_component_m": float(kernel_component[i]),
                    "safe_switch_component_m": float(switch_component[i]),
                    "q_min_index": int(q_argmin[i]),
                    "q_min_steps": int(q_argmin[i] + 1),
                    "q_max_index": int(q_argmax[i]),
                    "q_max_steps": int(q_argmax[i] + 1),
                    "action_at_q_min": int(action[i, q_argmin[i]]),
                    "action_at_q_max": int(action[i, q_argmax[i]]),
                    "branch_at_q_min": int(branch[i, q_argmin[i]]),
                    "branch_at_q_max": int(branch[i, q_argmax[i]]),
                    "safe_switch_witness_exists": bool(witness_mask[i]),
                    "safe_switch_interval_low_m": float(witness_low[i]),
                    "safe_switch_interval_high_m": float(witness_high[i]),
                    "safe_switch_interval_width_m": float(witness_width[i]),
                }
            )

        idx = np.flatnonzero(witness_mask)
        if len(idx):
            idx = idx[np.argsort(witness_width[idx])[::-1]]

        emitted = 0
        for i in idx[:MAX_WITNESSES_PER_PROFILE]:
            q_lo = int(q_argmin[i])
            q_hi = int(q_argmax[i])
            d_probe = 0.5 * (witness_low[i] + witness_high[i])

            mode_lo = mode_at_gap(
                np.asarray([d_probe]),
                np.asarray([raw[i, q_lo]]),
                np.asarray([eval_fallback[i]]),
            )[0]
            mode_hi = mode_at_gap(
                np.asarray([d_probe]),
                np.asarray([raw[i, q_hi]]),
                np.asarray([eval_fallback[i]]),
            )[0]

            witness_rows.append(
                {
                    "profile": name,
                    "node_index": int(i),
                    "v_f": float(vf[i]),
                    "v_p": float(vp[i]),
                    "a_f": float(af[i]),
                    "bar_a": float(bar_a[i]),
                    "bar_u": float(bar_u[i]),
                    "age": float(age[i]),
                    "q_cooperative_index": q_lo,
                    "q_cooperative_steps": q_lo + 1,
                    "q_fallback_index": q_hi,
                    "q_fallback_steps": q_hi + 1,
                    "fallback_required_m": float(eval_fallback[i]),
                    "cooperative_required_low_q_m": float(raw[i, q_lo]),
                    "cooperative_required_high_q_m": float(raw[i, q_hi]),
                    "probe_gap_m": float(d_probe),
                    "mode_at_q_cooperative": str(mode_lo),
                    "mode_at_q_fallback": str(mode_hi),
                    "switch_interval_width_m": float(witness_width[i]),
                    "action_at_q_cooperative": int(action[i, q_lo]),
                    "action_at_q_fallback": int(action[i, q_hi]),
                    "branch_at_q_cooperative": int(branch[i, q_lo]),
                    "branch_at_q_fallback": int(branch[i, q_hi]),
                }
            )
            emitted += 1

        row["witness_rows_emitted"] = emitted

        arrays[name] = {
            "raw": raw,
            "clipped": clipped,
            "kernel_component": kernel_component,
            "switch_component": switch_component,
        }

        print(
            f"{name.upper()} "
            f"R={R} "
            f"RAW_Q_DEP={row['raw_q_dependence_nodes']} "
            f"KERNEL_Q_COMP={row['kernel_q_component_nodes']} "
            f"SAFE_SWITCH_Q_COMP={row['safe_switch_q_component_nodes']} "
            f"SWITCH_WITNESS={row['safe_switch_witness_nodes']} "
            f"GAIN_NODES={row['gain_nodes_initial_slice']} "
            f"GAIN_SWITCH_Q={row['gain_nodes_with_safe_switch_q_component']} "
            f"FP_REPLAY_ERR={replay_error:.12g}"
        )

    # Validate emitted constructive witnesses.
    witness_valid = all(
        row["mode_at_q_cooperative"] == "C"
        and row["mode_at_q_fallback"] == "F"
        and row["probe_gap_m"] + TOL >= row["fallback_required_m"]
        and row["probe_gap_m"] + TOL
            >= row["cooperative_required_low_q_m"]
        and row["probe_gap_m"] + TOL
            < row["cooperative_required_high_q_m"]
        for row in witness_rows
    )

    # D3 established equality of the initial (worst-horizon) raw slice
    # across the three diagnostic profiles. Their full q-array shapes may
    # legitimately differ if the service horizons R differ, so D4 only
    # replays the scientifically established comparison.
    diagnostic_raw_max_diff = max(
        float(
            np.max(
                np.abs(
                    arrays[left]["raw"][:, -1]
                    -
                    arrays[right]["raw"][:, -1]
                )
            )
        )
        for left, right in (
            ("diagnostic_fast", "diagnostic_nominal"),
            ("diagnostic_nominal", "diagnostic_stressed"),
            ("diagnostic_fast", "diagnostic_stressed"),
        )
    )

    diagnostic_geometry_identical = diagnostic_raw_max_diff <= TOL

    max_replay = max(
        row["fixed_point_replay_max_error_m"]
        for row in profile_rows
    )
    max_decomp_error = max(
        row["q_span_decomposition_max_error_m"]
        for row in profile_rows
    )
    max_witness_component_error = max(
        row["witness_component_max_error_m"]
        for row in profile_rows
    )

    raw_q_total = sum(
        row["raw_q_dependence_nodes"]
        for row in profile_rows
        if row["R"] > 1
    )
    kernel_q_total = sum(
        row["kernel_q_component_nodes"]
        for row in profile_rows
        if row["R"] > 1
    )
    switch_q_total = sum(
        row["safe_switch_q_component_nodes"]
        for row in profile_rows
        if row["R"] > 1
    )
    witness_total = sum(
        row["safe_switch_witness_nodes"]
        for row in profile_rows
        if row["R"] > 1
    )
    gain_switch_total = sum(
        row["gain_nodes_with_safe_switch_q_component"]
        for row in profile_rows
        if row["R"] > 1
    )
    action_q_total = sum(
        row["best_action_q_change_nodes"]
        for row in profile_rows
        if row["R"] > 1
    )
    branch_q_total = sum(
        row["best_branch_q_change_nodes"]
        for row in profile_rows
        if row["R"] > 1
    )

    if kernel_q_total > 0:
        interpretation = (
            "Q_CHANGES_VIABILITY_ENVELOPE_AND_REQUIRES_KERNEL_RECOMPUTATION"
        )
        next_step = (
            "RUN_KERNEL_LEVEL_Q_RECOMPUTATION_AND_CERTIFICATE_AUDIT_BEFORE_ANY_CLAIM"
        )
    elif switch_q_total > 0:
        interpretation = (
            "Q_CHANGES_CERTIFIED_SUPERVISOR_MODE_BOUNDARY_WITHOUT_CHANGING_VIABILITY_ENVELOPE"
        )
        next_step = (
            "RUN_P3B1_R7D5_SWITCH_BOUNDARY_ROBUSTNESS_AND_MONITORED_REFINEMENT_AUDIT"
        )
    else:
        interpretation = (
            "NO_Q_DEPENDENT_CERTIFIED_SUPERVISOR_SWITCH_SURFACE_FOUND"
        )
        next_step = (
            "REQUIRE_EXPLICIT_Q_AUTOMATON_REDESIGN_BEFORE_FURTHER_EXPANSION"
        )

    integrity_checks = {
        "D2_UPSTREAM_PASS":
            d2.get("status") == "PASS",
        "D3_UPSTREAM_PASS":
            d3_latest.get("status") == "PASS",
        "D3_EXPECTED_ATTRIBUTION_REPRODUCED":
            d3_latest.get("interpretation") == expected_d3,
        "PROFILE_AUDIT_COMPLETE":
            len(profile_rows) == 4,
        "FIXED_POINT_OPERATOR_REPLAY":
            max_replay <= REPLAY_TOL,
        "Q_SPAN_DECOMPOSITION_EXACT":
            max_decomp_error <= REPLAY_TOL,
        "SAFE_SWITCH_INTERVAL_DECOMPOSITION_EXACT":
            max_witness_component_error <= REPLAY_TOL,
        "EMITTED_SWITCH_WITNESSES_CERTIFIED":
            bool(witness_valid),
        "DIAGNOSTIC_INITIAL_RAW_BOUNDARY_IDENTICAL":
            bool(diagnostic_geometry_identical),
    }

    status = "PASS" if all(integrity_checks.values()) else "FAIL"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    profile_csv = (
        RESULTS_DIR
        / f"P3B1_R7D4_SUPERVISOR_PROFILE_AUDIT_{stamp}.csv"
    )
    node_csv = (
        RESULTS_DIR
        / f"P3B1_R7D4_Q_NODE_ATTRIBUTION_{stamp}.csv"
    )
    witness_csv = (
        RESULTS_DIR
        / f"P3B1_R7D4_SAFE_SWITCH_WITNESSES_{stamp}.csv"
    )

    write_csv(profile_csv, profile_rows)
    write_csv(node_csv, node_rows)
    write_csv(witness_csv, witness_rows)

    result_path = (
        RESULTS_DIR
        / f"P3B1_R7D4_SUPERVISORY_SWITCH_AUDIT_{stamp}.json"
    )
    latest_path = RESULTS_DIR / "P3B1_R7D4_LATEST.json"
    manifest_path = RESULTS_DIR / f"P3B1_R7D4_MANIFEST_{stamp}.sha256"

    output = {
        "schema":
            "SCV_P3B1_R7D4_SUPERVISORY_SWITCH_SEMANTICS_V1",
        "status":
            status,
        "timestamp_utc":
            stamp,
        "classification":
            (
                "boundary-conditioned certified supervisor-mode attribution "
                "on the frozen-halo fixed point; preserves the D3 viability "
                "envelope and does not authorize a new kernel claim"
            ),
        "supervisor_semantics": {
            "cooperative":
                "select C when d >= h_C(y,q)",
            "fallback":
                "select F when d < h_C(y,q) and d >= h_F(y)",
            "outside":
                "no certified mode when d < min(h_C(y,q), h_F(y))",
            "envelope":
                "h_supervisor(y,q)=min(h_C(y,q),h_F(y))",
        },
        "integrity_checks":
            {k: bool(v) for k, v in integrity_checks.items()},
        "interpretation":
            interpretation,
        "recommended_next_step":
            next_step,
        "profile_stats":
            profile_rows,
        "aggregate": {
            "raw_q_dependence_nodes_sum":
                int(raw_q_total),
            "kernel_q_component_nodes_sum":
                int(kernel_q_total),
            "safe_switch_q_component_nodes_sum":
                int(switch_q_total),
            "safe_switch_witness_nodes_sum":
                int(witness_total),
            "gain_nodes_with_safe_switch_q_component_sum":
                int(gain_switch_total),
            "best_action_q_change_nodes_sum":
                int(action_q_total),
            "best_branch_q_change_nodes_sum":
                int(branch_q_total),
            "emitted_witness_rows":
                int(len(witness_rows)),
            "max_fixed_point_replay_error_m":
                float(max_replay),
            "max_q_span_decomposition_error_m":
                float(max_decomp_error),
            "max_switch_interval_decomposition_error_m":
                float(max_witness_component_error),
            "diagnostic_initial_raw_boundary_max_abs_difference_m":
                float(diagnostic_raw_max_diff),
        },
        "claims": {
            "scientific_kernel_claim_authorized":
                False,
            "maximal_continuous_kernel_claim":
                False,
            "p6_certified":
                False,
            "supervisory_q_sensitivity_candidate":
                bool(kernel_q_total == 0 and switch_q_total > 0),
            "supervisory_q_sensitivity_claim_authorized":
                False,
            "reason_claim_not_yet_authorized":
                (
                    "D4 establishes constructive certified mode-switch "
                    "witnesses only; robustness under tolerance/refinement "
                    "and monitored implementation semantics remains for R7D5"
                ),
        },
        "artifacts": {
            "profile_csv": str(profile_csv),
            "q_node_attribution_csv": str(node_csv),
            "safe_switch_witness_csv": str(witness_csv),
            "result_json": str(result_path),
            "latest_json": str(latest_path),
            "manifest": str(manifest_path),
        },
    }

    text = json.dumps(output, indent=2, sort_keys=True)
    atomic_write(result_path, text)
    atomic_write(latest_path, text)

    manifest_files = [
        P1_CFG_PATH,
        P2B_CFG_PATH,
        P2C_CFG_PATH,
        P3A_CFG_PATH,
        D2_PATH,
        D3_PATH,
        Path(d1.__file__),
        Path(d3.__file__),
        Path(d3.r7.__file__),
        Path(d3.r3.__file__),
        Path(__file__),
        result_path,
        profile_csv,
        node_csv,
        witness_csv,
    ]

    manifest = "\n".join(
        f"{sha256_file(path)}  {path}"
        for path in manifest_files
    ) + "\n"
    atomic_write(manifest_path, manifest)

    print("=== P3-B1-R7-D4 DECISION ===")
    print(f"RAW_Q_DEPENDENCE_NODES_SUM={raw_q_total}")
    print(f"KERNEL_Q_COMPONENT_NODES_SUM={kernel_q_total}")
    print(f"SAFE_SWITCH_Q_COMPONENT_NODES_SUM={switch_q_total}")
    print(f"SAFE_SWITCH_WITNESS_NODES_SUM={witness_total}")
    print(f"GAIN_SWITCH_Q_COMPONENT_SUM={gain_switch_total}")
    print(f"BEST_ACTION_Q_CHANGE_NODES_SUM={action_q_total}")
    print(f"BEST_BRANCH_Q_CHANGE_NODES_SUM={branch_q_total}")
    print(f"INTERPRETATION={interpretation}")
    print(f"RECOMMENDED_NEXT_STEP={next_step}")
    print(f"P3B1_R7D4_SUPERVISORY_SWITCH_AUDIT={status}")
    print("SCIENTIFIC_KERNEL_CLAIM=NO")
    print(
        "SUPERVISORY_Q_SENSITIVITY_CANDIDATE="
        + (
            "YES"
            if kernel_q_total == 0 and switch_q_total > 0
            else "NO"
        )
    )
    print("SUPERVISORY_Q_SENSITIVITY_CLAIM_AUTHORIZED=NO")
    print(f"RESULT_JSON={result_path}")
    print(f"MANIFEST={manifest_path}")

    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
