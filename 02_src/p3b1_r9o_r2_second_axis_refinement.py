from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
RESULTS = ROOT / "04_results"
RESULTS.mkdir(parents=True, exist_ok=True)
TOL = 1e-9
AXIS_NAMES = ("v_f", "v_p", "a_f", "bar_a", "bar_u", "age")
EXPECTED_R9O = "ONE_AXIS_ADAPTIVE_REFINEMENT_INSUFFICIENT_ENCLOSURE_ERASURE_PERSISTS"


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def choose_second_axis_and_intervals(
    *, r9o, votes: np.ndarray, first_axis: int, cell_hit_counts: np.ndarray,
    cell_shape: tuple[int, ...], eval_axes: list[np.ndarray], lookup_axes: list[np.ndarray],
    max_eval_nodes: int, max_lookup_nodes: int, forced_axis: str,
):
    vv = np.asarray(votes, float).copy()
    vv[first_axis] = -np.inf
    if forced_axis != "auto":
        if forced_axis not in AXIS_NAMES:
            raise ValueError(f"R9OR2_BAD_SECOND_AXIS={forced_axis}")
        idx = AXIS_NAMES.index(forced_axis)
        if idx == first_axis:
            raise ValueError("R9OR2_SECOND_AXIS_MUST_DIFFER_FROM_FIRST_AXIS")
        forced = forced_axis
    else:
        finite = np.isfinite(vv)
        if not np.any(finite):
            raise RuntimeError("R9OR2_NO_SECOND_AXIS_CANDIDATE")
        idx = int(np.argmax(vv))
        forced = AXIS_NAMES[idx]
    return r9o.choose_axis_and_intervals(
        votes=vv,
        cell_hit_counts=cell_hit_counts,
        cell_shape=cell_shape,
        eval_axes=eval_axes,
        lookup_axes=lookup_axes,
        max_eval_nodes=max_eval_nodes,
        max_lookup_nodes=max_lookup_nodes,
        forced_axis=forced,
    )


def self_test() -> None:
    class Dummy:
        @staticmethod
        def choose_axis_and_intervals(**kw):
            axis = AXIS_NAMES.index(kw["forced_axis"])
            return axis, [0], {"axis": kw["forced_axis"]}
    votes = np.array([1.0, 2.0, 3.0, 4.0, 10.0, 9.0])
    axis, ids, meta = choose_second_axis_and_intervals(
        r9o=Dummy,
        votes=votes,
        first_axis=4,
        cell_hit_counts=np.array([1]),
        cell_shape=(1,),
        eval_axes=[np.array([0.0, 1.0])] * 6,
        lookup_axes=[np.array([0.0, 1.0])] * 6,
        max_eval_nodes=100,
        max_lookup_nodes=100,
        forced_axis="auto",
    )
    assert axis == 5 and ids == [0] and meta["axis"] == "age"
    try:
        choose_second_axis_and_intervals(
            r9o=Dummy, votes=votes, first_axis=4, cell_hit_counts=np.array([1]),
            cell_shape=(1,), eval_axes=[np.array([0.0, 1.0])] * 6,
            lookup_axes=[np.array([0.0, 1.0])] * 6,
            max_eval_nodes=100, max_lookup_nodes=100, forced_axis="bar_u",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("same-axis second refinement was not rejected")
    print("R9OR2_INTERNAL_SELF_TEST=PASS", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--action-width", type=float, default=0.5)
    ap.add_argument("--p2c-lower-resolution", type=int, default=257)
    ap.add_argument("--max-eval-nodes", type=int, default=1800000)
    ap.add_argument("--max-lookup-nodes", type=int, default=3300000)
    ap.add_argument("--second-axis", type=str, default="auto")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not (0.0 < args.action_width <= 1.0):
        raise ValueError("R9OR2_BAD_ACTION_WIDTH")

    sys.path.insert(0, str(SRC))
    import p3b1_augmented_fixed_point_v1_r3 as r3
    import p3b0_freshness_service_audit_v1 as b0
    import p2c_switching_guard_v1 as sw
    import p3b1_r5_refinement_attribution as r5
    import p3b1_r6_continuation_halo as r6
    import p3b1_r7_frozen_halo_fixed_point as r7
    import p3b1_r9c_targeted_history_gate as r9c
    import p3b1_r9d_timestamped_service as r9d
    import p3b1_r9e_service_contract_general_action as r9e
    import p3b1_r9h_b_protocolized_paired_bounds as r9hb
    import p3b1_r9h_e_converged_lower_gfp as r9he
    import p3b1_r9h_g_full_augmented_service_gfp as r9hg
    import p3b1_r9k_full_augmented_declared_pair_gfp as r9k
    import p3b1_r9o_adaptive_reachable_gap_refinement as r9o

    up = RESULTS / "P3B1_R9O_LATEST.json"
    if not up.exists():
        raise RuntimeError("R9OR2_MISSING_R9O_LATEST")
    prev = load_json(up)
    if prev.get("status") != "PASS" or prev.get("classification") != EXPECTED_R9O:
        raise RuntimeError(f"R9OR2_R9O_GATE_FAIL={prev.get('classification')}")

    first_axis_name = str(prev.get("refinement_plan", {}).get("axis", ""))
    if first_axis_name not in AXIS_NAMES:
        raise RuntimeError(f"R9OR2_BAD_R9O_FIRST_AXIS={first_axis_name}")
    first_axis = AXIS_NAMES.index(first_axis_name)
    first_intervals = [int(x) for x in prev.get("selected_interval_ids", [])]
    if not first_intervals:
        raise RuntimeError("R9OR2_R9O_FIRST_INTERVALS_MISSING")

    data0 = r9d.reconstruct_timestamped_geometry(r3, r5, r6, r7, r9c)
    Ts = float(data0["Ts"])
    actions = r9hb.refinement_actions(float(args.action_width))

    print("=== P3-B1-R9-O-R2 SECOND-AXIS ADAPTIVE REFINEMENT ===", flush=True)
    print("UPSTREAM_R9O_ONE_AXIS_INSUFFICIENT=PASS", flush=True)
    print("UPSTREAM_R9NR1_REACHABLE_ENCLOSURE_ERASURE=PASS", flush=True)
    print("QUESTION=DOES_ONE_FINAL_ORTHOGONAL_AXIS_REFINEMENT_BREAK_STAGEWISE_CELL_MAX_ERASURE", flush=True)
    print("FINAL_PREDECLARED_REFINEMENT_ROUND=YES", flush=True)
    print("FURTHER_AUTOMATIC_AXIS_REFINEMENT_AUTHORIZED=NO", flush=True)
    print("DEPLOYMENT_PROTOCOL_CERTIFIED=NO", flush=True)
    print("CONTINUOUS_STATE_SEPARATION_CERTIFIED=NO", flush=True)
    print(f"POINT_ACTION_WIDTH={float(args.action_width):.12g}", flush=True)
    print(f"POINT_ACTION_COUNT={len(actions)}", flush=True)
    print(f"R9OR2_FIRST_AXIS={first_axis_name}", flush=True)
    print(f"R9OR2_FIRST_INTERVAL_COUNT={len(first_intervals)}", flush=True)

    # Reconstruct the exact R9-O first-refined geometry from the saved plan.
    data1 = r9o.build_refined_data(
        data=data0, axis=first_axis, interval_ids=first_intervals, r6=r6, r7=r7
    )
    eval1 = int(np.prod(data1["eval_shape"]))
    lookup1 = int(np.prod(data1["lookup_shape"]))
    expected_eval1 = int(prev.get("refined_geometry", {}).get("eval_nodes", -1))
    expected_lookup1 = int(prev.get("refined_geometry", {}).get("lookup_nodes", -1))
    if expected_eval1 > 0 and eval1 != expected_eval1:
        raise RuntimeError(f"R9OR2_FIRST_GEOMETRY_EVAL_MISMATCH={eval1}!={expected_eval1}")
    if expected_lookup1 > 0 and lookup1 != expected_lookup1:
        raise RuntimeError(f"R9OR2_FIRST_GEOMETRY_LOOKUP_MISMATCH={lookup1}!={expected_lookup1}")
    print(f"R9OR2_FIRST_GEOMETRY_REPLAY=PASS eval_nodes={eval1} lookup_nodes={lookup1}", flush=True)

    cfg1, tr1 = r9hg.build_physical_transitions(
        data1, actions, r3, b0, r9e, chunk_size=int(args.chunk_size)
    )
    old_groups1 = r9hg.build_descriptor_range_groups(data1, 2.0 * Ts, b0)
    repl_groups1 = r9hg.build_descriptor_range_groups(data1, 0.0, b0)
    old_age1, ok1 = r9hg.scalar_cell_index(np.asarray(data1["lookup_axes"][5], float), 3.0 * Ts)
    repl_age1, ok2 = r9hg.scalar_cell_index(np.asarray(data1["lookup_axes"][5], float), 1.0 * Ts)
    if not ok1 or not ok2:
        raise RuntimeError("R9OR2_FIRST_ADOPTION_AGE_LOOKUP_FAIL")

    print("R9OR2_STAGE_START=recompute_first_refined_upper_for_second_axis_attribution", flush=True)
    upper1 = r9k.solve_full_service_fixed_point(
        data=data1, cfg=cfg1, transitions=tr1,
        fallback_eval=np.asarray(data1["eval_fallback"], float),
        old_groups=old_groups1, repl_groups=repl_groups1,
        old_adopt_age_cell=old_age1, repl_adopt_age_cell=repl_age1,
        r3=r3, r9hg=r9hg, mode="upper", label="r9or2_first_refined_upper",
    )
    if not upper1.converged:
        raise RuntimeError("R9OR2_FIRST_REFINED_UPPER_NOT_CONVERGED")

    h1 = upper1.h_flat.reshape(tuple(data1["lookup_shape"]) + (r9k.N_STAGES,))
    raw_gap1 = np.abs(h1[..., r9k.OLD_LAST] - h1[..., r9k.REPL_LAST])
    raw_support1 = np.isfinite(raw_gap1) & (raw_gap1 > TOL)
    raw_cells1 = r9o.cell_corner_any(raw_support1, ndims=6)
    if not np.any(raw_cells1):
        raise RuntimeError("R9OR2_FIRST_REFINED_RAW_GAP_VANISHED")

    amp = np.where(np.isfinite(raw_gap1), raw_gap1, 0.0)
    for axis in range(6):
        left = [slice(None)] * amp.ndim
        right = [slice(None)] * amp.ndim
        left[axis] = slice(0, -1)
        right[axis] = slice(1, None)
        amp = np.maximum(amp[tuple(left)], amp[tuple(right)])
    raw_amp1 = amp

    pair_phase = np.asarray(data1["eval_flat"][5], float) >= 3.0 * Ts - 1e-10
    cell_shape1 = tuple(len(a) - 1 for a in data1["lookup_axes"])
    cell_count1 = int(np.prod(cell_shape1))
    hit_counts1 = np.zeros(cell_count1, dtype=np.int64)
    raw_flat1 = raw_cells1.reshape(-1)
    for tr in tr1:
        hold = np.asarray(tr["hold_index"], dtype=np.intp)
        m = pair_phase & raw_flat1[hold]
        if np.any(m):
            hit_counts1 += np.bincount(hold[m], minlength=cell_count1)
    hit_cells1 = np.flatnonzero(hit_counts1 > 0)
    if hit_cells1.size == 0:
        raise RuntimeError("R9OR2_NO_PAIR_PHASE_RAW_GAP_HIT_CELLS_AFTER_FIRST_REFINEMENT")

    attrib_rows, votes = r9o.axis_vote_rows(
        h=h1, hit_cells=hit_cells1, hit_counts=hit_counts1,
        raw_gap_cells=raw_cells1, raw_gap_amplitude=raw_amp1, r9k=r9k,
    )
    second_axis, second_intervals, plan2 = choose_second_axis_and_intervals(
        r9o=r9o, votes=votes, first_axis=first_axis,
        cell_hit_counts=hit_counts1, cell_shape=cell_shape1,
        eval_axes=[np.asarray(a, float) for a in data1["eval_axes"]],
        lookup_axes=[np.asarray(a, float) for a in data1["lookup_axes"]],
        max_eval_nodes=int(args.max_eval_nodes), max_lookup_nodes=int(args.max_lookup_nodes),
        forced_axis=str(args.second_axis),
    )
    second_axis_name = AXIS_NAMES[second_axis]

    print(
        "R9OR2_SECOND_AXIS_ATTRIBUTION "
        + " ".join(f"{AXIS_NAMES[i]}_vote={votes[i]:.12g}" for i in range(6)),
        flush=True,
    )
    print(
        "R9OR2_SECOND_REFINEMENT_PLAN "
        f"first_axis={first_axis_name} second_axis={second_axis_name} "
        f"available_intervals={plan2['available_intervals']} selected_intervals={plan2['selected_intervals']} "
        f"base_eval_nodes={plan2['base_eval_nodes']} projected_eval_nodes={plan2['projected_eval_nodes']} "
        f"base_lookup_nodes={plan2['base_lookup_nodes']} projected_lookup_nodes={plan2['projected_lookup_nodes']} "
        f"budget_truncated={str(plan2['node_budget_truncated']).upper()}",
        flush=True,
    )

    data2 = r9o.build_refined_data(
        data=data1, axis=second_axis, interval_ids=second_intervals, r6=r6, r7=r7
    )
    eval2 = int(np.prod(data2["eval_shape"]))
    lookup2 = int(np.prod(data2["lookup_shape"]))
    if eval2 > args.max_eval_nodes or lookup2 > args.max_lookup_nodes:
        raise RuntimeError(f"R9OR2_NODE_BUDGET_FAIL eval={eval2} lookup={lookup2}")
    print(
        f"R9OR2_REFINED_GEOMETRY first_axis={first_axis_name} second_axis={second_axis_name} "
        f"second_intervals={len(second_intervals)} eval_nodes={eval2} lookup_nodes={lookup2}",
        flush=True,
    )

    print("R9OR2_STAGE_START=build_two_axis_refined_transitions", flush=True)
    cfg2, tr2 = r9hg.build_physical_transitions(
        data2, actions, r3, b0, r9e, chunk_size=int(args.chunk_size)
    )
    old_groups2 = r9hg.build_descriptor_range_groups(data2, 2.0 * Ts, b0)
    repl_groups2 = r9hg.build_descriptor_range_groups(data2, 0.0, b0)
    old_age2, oka = r9hg.scalar_cell_index(np.asarray(data2["lookup_axes"][5], float), 3.0 * Ts)
    repl_age2, okb = r9hg.scalar_cell_index(np.asarray(data2["lookup_axes"][5], float), 1.0 * Ts)
    if not oka or not okb:
        raise RuntimeError("R9OR2_TWO_AXIS_ADOPTION_AGE_LOOKUP_FAIL")

    print("R9OR2_STAGE_START=two_axis_refined_upper_full_gfp", flush=True)
    upper2 = r9k.solve_full_service_fixed_point(
        data=data2, cfg=cfg2, transitions=tr2,
        fallback_eval=np.asarray(data2["eval_fallback"], float),
        old_groups=old_groups2, repl_groups=repl_groups2,
        old_adopt_age_cell=old_age2, repl_adopt_age_cell=repl_age2,
        r3=r3, r9hg=r9hg, mode="upper", label="r9or2_two_axis_upper",
    )
    if not upper2.converged:
        raise RuntimeError("R9OR2_TWO_AXIS_UPPER_NOT_CONVERGED")

    prelim = r9o.evaluate_solution(data=data2, upper=upper2, lower=None, r3=r3, r9k=r9k)
    lower2 = None
    p2c_metrics = None
    if prelim["upper_sensitive_nodes"] > 0:
        print("R9OR2_STAGE_START=two_axis_p2c_lower_and_lower_full_gfp", flush=True)
        fallback_lower, p2c_metrics = r9he.compute_eval_fallback_lower(
            data2, b0, sw, resolution=int(args.p2c_lower_resolution),
            chunk_size=int(args.chunk_size),
        )
        lower2 = r9k.solve_full_service_fixed_point(
            data=data2, cfg=cfg2, transitions=tr2,
            fallback_eval=np.asarray(fallback_lower, float),
            old_groups=old_groups2, repl_groups=repl_groups2,
            old_adopt_age_cell=old_age2, repl_adopt_age_cell=repl_age2,
            r3=r3, r9hg=r9hg, mode="lower", label="r9or2_two_axis_lower",
        )
        if not lower2.converged:
            raise RuntimeError("R9OR2_TWO_AXIS_LOWER_NOT_CONVERGED")

    metrics = r9o.evaluate_solution(data=data2, upper=upper2, lower=lower2, r3=r3, r9k=r9k)
    first_raw_nodes = int(np.count_nonzero(raw_support1))
    first_raw_gap = r9o.finite_max(raw_gap1[raw_support1]) if first_raw_nodes else 0.0

    print(
        "R9OR2_TWO_AXIS_LAST_STAGE "
        f"first_refined_raw_nodes={first_raw_nodes} first_refined_raw_gap_m={first_raw_gap:.12g} "
        f"two_axis_raw_nodes={metrics['raw_last_stage_gap_nodes']} "
        f"two_axis_raw_gap_m={metrics['max_raw_last_stage_gap_m']:.12g} "
        f"two_axis_raw_gap_cells={metrics['raw_gap_containing_cells']} "
        f"two_axis_stagewise_cellmax_gap_cells={metrics['stagewise_cellmax_gap_cells']} "
        f"two_axis_stagewise_cellmax_gap_m={metrics['max_stagewise_cellmax_gap_m']:.12g}",
        flush=True,
    )
    print(
        "R9OR2_TWO_AXIS_PAIR_GFP "
        f"common_nodes={metrics['common_nodes']} upper_sensitive={metrics['upper_sensitive_nodes']} "
        f"max_upper_gap_m={metrics['max_upper_pair_gap_m']:.12g} "
        f"paired_positive={metrics['paired_positive_nodes']} "
        f"max_credential_margin_m={metrics['max_credential_worse_margin_m']:.12g} "
        f"max_fragment_margin_m={metrics['max_fragment_worse_margin_m']:.12g}",
        flush=True,
    )

    if metrics["paired_positive_nodes"] > 0:
        classification = "TWO_AXIS_REFINEMENT_RECOVERS_FULL_GFP_SEPARATION_AND_PAIRED_POINT_MARGIN"
        next_action = "R9P_INTERVALIZE_TWO_AXIS_REFINED_FULL_GRAPH_AND_BUILD_INDEPENDENT_P2C_LOWER_CHECKER"
    elif metrics["upper_sensitive_nodes"] > 0:
        classification = "TWO_AXIS_REFINEMENT_RECOVERS_FULL_GFP_SEPARATION_PAIRED_MARGIN_OPEN"
        next_action = "R9P_STRENGTHEN_TWO_AXIS_LOWER_ENCLOSURE_THEN_INTERVALIZE"
    elif metrics["stagewise_cellmax_gap_cells"] > 0:
        classification = "TWO_AXIS_REFINEMENT_PRESERVES_STAGEWISE_CELLMAX_GAP_BUT_TARGET_PAIR_GFP_STILL_COLLAPSES"
        next_action = "R9P_STOP_POSITIVE_PAIR_SEARCH_AND_REPORT_REFINED_VALUE_TRANSFER_LIMITATION"
    elif metrics["raw_last_stage_gap_nodes"] > 0:
        classification = "TWO_AXIS_PREDECLARED_REFINEMENT_INSUFFICIENT_ENCLOSURE_ERASURE_PERSISTS"
        next_action = "R9P_STOP_POSITIVE_CONTINUOUS_PROMOTION_AND_REPORT_ENCLOSURE_LIMITATION"
    else:
        classification = "RAW_STAGE_GAP_FAILS_TWO_AXIS_REFINEMENT_STABILITY"
        next_action = "R9P_STOP_POSITIVE_CONTINUOUS_PROMOTION_AND_REPORT_REFINEMENT_FALSIFICATION"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result = RESULTS / f"P3B1_R9OR2_RESULT_{stamp}.json"
    latest = RESULTS / "P3B1_R9OR2_LATEST.json"
    attrib_csv = RESULTS / f"P3B1_R9OR2_SECOND_AXIS_ATTRIBUTION_{stamp}.csv"
    intervals_csv = RESULTS / f"P3B1_R9OR2_SECOND_AXIS_INTERVALS_{stamp}.csv"
    manifest = RESULTS / f"P3B1_R9OR2_MANIFEST_{stamp}.sha256"

    for row in attrib_rows:
        row["first_axis"] = first_axis_name
        row["selected_second_axis"] = second_axis_name
    write_csv(attrib_csv, attrib_rows[:4096])
    axis2 = np.asarray(data1["lookup_axes"][second_axis], float)
    interval_rows = []
    for i in second_intervals:
        interval_rows.append({
            "first_axis": first_axis_name,
            "second_axis": second_axis_name,
            "interval_index": int(i),
            "left": float(axis2[i]),
            "right": float(axis2[i + 1]),
            "midpoint": 0.5 * (float(axis2[i]) + float(axis2[i + 1])),
        })
    write_csv(intervals_csv, interval_rows)

    payload = {
        "schema": "P3B1_R9O_R2_SECOND_AXIS_ADAPTIVE_REFINEMENT_V1",
        "status": "PASS",
        "classification": classification,
        "next_action": next_action,
        "final_predeclared_refinement_round": True,
        "further_automatic_axis_refinement_authorized": False,
        "hard_flags": {
            "deployment_protocol_certified": False,
            "continuous_action_interval_gfp_solved": False,
            "p2c_independent_lower_checker": False,
            "continuous_state_separation_certified": False,
        },
        "first_refinement": {
            "axis": first_axis_name,
            "selected_interval_ids": first_intervals,
            "eval_nodes": eval1,
            "lookup_nodes": lookup1,
            "raw_last_stage_gap_nodes": first_raw_nodes,
            "max_raw_last_stage_gap_m": first_raw_gap,
        },
        "second_refinement": {
            "axis": second_axis_name,
            "selected_interval_ids": [int(i) for i in second_intervals],
            "plan": plan2,
            "axis_votes_after_first_refinement": {AXIS_NAMES[i]: float(votes[i]) for i in range(6)},
            "eval_nodes": eval2,
            "lookup_nodes": lookup2,
        },
        "metrics": metrics,
        "p2c_lower_metrics": p2c_metrics,
        "artifacts": {
            "second_axis_attribution_csv": str(attrib_csv),
            "second_axis_intervals_csv": str(intervals_csv),
        },
    }
    text = json.dumps(payload, indent=2, sort_keys=True, default=lambda x: x.item() if isinstance(x, np.generic) else x) + "\n"
    atomic_write(result, text)
    atomic_write(latest, text)
    files = [result, latest, attrib_csv, intervals_csv]
    atomic_write(manifest, "\n".join(f"{sha256_file(p)}  {p.name}" for p in files) + "\n")

    print("=== R9-O-R2 DECISION ===", flush=True)
    print("R9OR2_SECOND_AXIS_REFINEMENT_GATE=PASS", flush=True)
    print(f"R9OR2_CLASSIFICATION={classification}", flush=True)
    print("FINAL_PREDECLARED_REFINEMENT_ROUND=YES", flush=True)
    print("FURTHER_AUTOMATIC_AXIS_REFINEMENT_AUTHORIZED=NO", flush=True)
    print("DEPLOYMENT_PROTOCOL_CERTIFIED=NO", flush=True)
    print("CONTINUOUS_ACTION_INTERVAL_GFP_SOLVED=NO", flush=True)
    print("P2C_INDEPENDENT_LOWER_CHECKER=NO", flush=True)
    print("CONTINUOUS_STATE_SEPARATION_CERTIFIED=NO", flush=True)
    print("R9OR2_EXECUTION=PASS", flush=True)
    print(f"R9OR2_NEXT_ACTION={next_action}", flush=True)
    print(f"RESULT_JSON={result}", flush=True)
    print(f"SECOND_AXIS_ATTRIBUTION_CSV={attrib_csv}", flush=True)
    print(f"SECOND_AXIS_INTERVALS_CSV={intervals_csv}", flush=True)
    print(f"MANIFEST={manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
