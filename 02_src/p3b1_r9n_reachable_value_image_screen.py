from __future__ import annotations

import argparse
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
CONFIG = ROOT / "01_config"
RESULTS = ROOT / "04_results"
RESULTS.mkdir(parents=True, exist_ok=True)
TOL = 1e-9
EXPECTED_R9M = "PAIR_SUCCESSOR_IMAGE_AVOIDS_ALL_GLOBAL_LAST_STAGE_VALUE_DIFFERENCE"
EXPECTED_R9K = "FULL_AUGMENTED_R9J_DIAGNOSTIC_GFP_COLLAPSES_NEW_PAIR"


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
    seen = set()
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


def finite_sup_abs(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    aa = np.asarray(a, float)
    bb = np.asarray(b, float)
    if mask is not None:
        aa = aa[mask]
        bb = bb[mask]
    good = np.isfinite(aa) & np.isfinite(bb)
    if not np.any(good):
        return 0.0
    return float(np.max(np.abs(aa[good] - bb[good])))


def common_predecessors(contract: dict) -> dict[tuple[str, str], list[str]]:
    succ: dict[str, set[str]] = defaultdict(set)
    for e in contract.get("edges", []):
        succ[str(e["src"])].add(str(e["dst"]))
    out: dict[tuple[str, str], list[str]] = defaultdict(list)
    for pred, ss in succ.items():
        names = sorted(ss)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                out[(names[i], names[j])].append(pred)
    return out


def self_test() -> None:
    # Global value gap can exist while a source image misses it completely.
    gap_support = np.asarray([False, True, False, True])
    hold = np.asarray([0, 2, 0], dtype=np.intp)
    assert int(np.count_nonzero(gap_support[hold])) == 0
    hold2 = np.asarray([1, 2, 3], dtype=np.intp)
    assert int(np.count_nonzero(gap_support[hold2])) == 2

    # Value difference can survive future -> continuation -> requirement -> min.
    step = np.asarray([0.0, 0.0])
    close = np.asarray([0.0, 0.0])
    f1 = np.asarray([1.0, 4.0])
    f2 = np.asarray([2.0, 1.0])
    c1 = np.maximum(0.0, close + f1)
    c2 = np.maximum(0.0, close + f2)
    r1 = np.maximum(step, c1)
    r2 = np.maximum(step, c2)
    assert np.any(np.abs(r1 - r2) > TOL)
    assert abs(float(np.min(r1)) - float(np.min(r2))) <= TOL  # controller may erase it
    print("R9N_INTERNAL_SELF_TEST=PASS", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--action-width", type=float, default=0.5)
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.action_width <= 0.0 or args.action_width > 1.0:
        raise ValueError("R9N_BAD_ACTION_WIDTH")

    sys.path.insert(0, str(SRC))
    import p3b1_augmented_fixed_point_v1_r3 as r3
    import p3b0_freshness_service_audit_v1 as b0
    import p3b1_r5_refinement_attribution as r5
    import p3b1_r6_continuation_halo as r6
    import p3b1_r7_frozen_halo_fixed_point as r7
    import p3b1_r9c_targeted_history_gate as r9c
    import p3b1_r9d_timestamped_service as r9d
    import p3b1_r9e_service_contract_general_action as r9e
    import p3b1_r9h_b_protocolized_paired_bounds as r9hb
    import p3b1_r9h_g_full_augmented_service_gfp as r9hg
    import p3b1_r9k_full_augmented_declared_pair_gfp as r9k

    r9m_path = RESULTS / "P3B1_R9M_LATEST.json"
    r9k_path = RESULTS / "P3B1_R9K_LATEST.json"
    contract_path = CONFIG / "p3b1_r9j_declared_diagnostic_service_contract_v2.json"
    for p in (r9m_path, r9k_path, contract_path):
        if not p.exists():
            raise RuntimeError(f"R9N_MISSING_UPSTREAM={p}")
    r9m = load_json(r9m_path)
    r9ku = load_json(r9k_path)
    contract = load_json(contract_path)
    if r9m.get("status") != "PASS" or r9m.get("classification") != EXPECTED_R9M:
        raise RuntimeError(f"R9N_R9M_GATE_FAIL={r9m.get('classification')}")
    if r9ku.get("status") != "PASS" or r9ku.get("classification") != EXPECTED_R9K:
        raise RuntimeError(f"R9N_R9K_GATE_FAIL={r9ku.get('classification')}")
    mm = r9m.get("metrics", {})
    if int(mm.get("reachable_future_value_difference_tests", -1)) != 0:
        raise RuntimeError("R9N_EXPECTED_ZERO_R9M_REACHABLE_FUTURE_DIFF")
    if float(mm.get("global_last_stage_gap_m", 0.0)) <= TOL:
        raise RuntimeError("R9N_EXPECTED_GLOBAL_LAST_STAGE_GAP")

    data = r9d.reconstruct_timestamped_geometry(r3, r5, r6, r7, r9c)
    Ts = float(data["Ts"])
    actions = r9hb.refinement_actions(float(args.action_width))

    print("=== P3-B1-R9-N REACHABLE SUCCESSOR VALUE-IMAGE SCREEN ===", flush=True)
    print("UPSTREAM_R9M_SUCCESSOR_IMAGE_MISS=PASS", flush=True)
    print("UPSTREAM_R9K_FULL_GFP_COLLAPSE=PASS", flush=True)
    print("QUESTION=WHICH_SERVICE_STATE_PAIRS_HAVE_BELLMAN_ACTIVE_VALUE_DIFFERENCE_ON_THEIR_ACTUAL_REACHABLE_SUCCESSOR_IMAGE", flush=True)
    print("TRACE_LABEL_DIFFERENCE_IS_SCREENING_CRITERION=NO", flush=True)
    print("DEPLOYMENT_PROTOCOL_CERTIFIED=NO", flush=True)
    print("CONTINUOUS_STATE_SEPARATION_CERTIFIED=NO", flush=True)
    print(f"POINT_ACTION_WIDTH={float(args.action_width):.12g}", flush=True)
    print(f"POINT_ACTION_COUNT={len(actions)}", flush=True)

    cfg, transitions = r9hg.build_physical_transitions(
        data, actions, r3, b0, r9e, chunk_size=int(args.chunk_size)
    )
    age = np.asarray(data["eval_flat"][5], float)
    masks = r9k.stage_masks(age, Ts)

    old_groups = r9hg.build_descriptor_range_groups(data, 2.0 * Ts, b0)
    repl_groups = r9hg.build_descriptor_range_groups(data, 0.0, b0)
    old_age_cell, old_ok = r9hg.scalar_cell_index(np.asarray(data["lookup_axes"][5], float), 3.0 * Ts)
    repl_age_cell, repl_ok = r9hg.scalar_cell_index(np.asarray(data["lookup_axes"][5], float), 1.0 * Ts)
    if not old_ok or not repl_ok:
        raise RuntimeError("R9N_ADOPTION_AGE_LOOKUP_FAIL")

    print("R9N_STAGE_START=recompute_actual_upper_full_gfp", flush=True)
    upper = r9k.solve_full_service_fixed_point(
        data=data, cfg=cfg, transitions=transitions,
        fallback_eval=np.asarray(data["eval_fallback"], float),
        old_groups=old_groups, repl_groups=repl_groups,
        old_adopt_age_cell=old_age_cell, repl_adopt_age_cell=repl_age_cell,
        r3=r3, r9hg=r9hg, mode="upper", label="r9n_actual_upper",
    )
    if not upper.converged:
        raise RuntimeError("R9N_UPPER_GFP_NOT_CONVERGED")

    lookup_shape = tuple(data["lookup_shape"])
    eval_idx = np.asarray(data["eval_idx"], dtype=np.int64)
    h = upper.h_flat.reshape(lookup_shape + (r9k.N_STAGES,))
    cells = r3.cell_corner_max(h)
    cflat = [cells[..., s].reshape(-1) for s in range(r9k.N_STAGES)]
    ue = np.asarray(upper.h_flat[eval_idx, :], float)
    n = len(eval_idx)

    # R9-M's specific gap support: OLD_LAST vs REPL_LAST cell values.
    old_repl_cell_gap = np.abs(cflat[r9k.OLD_LAST] - cflat[r9k.REPL_LAST])
    old_repl_support = np.isfinite(old_repl_cell_gap) & (old_repl_cell_gap > TOL)
    old_repl_support_cells = int(np.count_nonzero(old_repl_support))
    max_cell_gap = float(np.max(old_repl_cell_gap[old_repl_support])) if old_repl_support_cells else 0.0

    preimage_all_tests = 0
    preimage_pair_phase_tests = 0
    preimage_before_pair_tests = 0
    preimage_nodes_all = np.zeros(n, bool)
    preimage_nodes_pair = np.zeros(n, bool)
    earliest_source_age = math.inf
    preimage_rows: list[dict] = []

    # Pairwise action-level screen accumulators.
    pairs = [(a, b) for a in range(r9k.N_STAGES) for b in range(a + 1, r9k.N_STAGES)]
    accum: dict[tuple[int, int], dict] = {}
    for a, b in pairs:
        common = masks[a] & masks[b]
        accum[(a, b)] = {
            "common": common,
            "future_nodes": np.zeros(n, bool),
            "cont_nodes": np.zeros(n, bool),
            "req_nodes": np.zeros(n, bool),
            "future_tests": 0,
            "cont_tests": 0,
            "req_tests": 0,
            "max_future": 0.0,
            "max_cont": 0.0,
            "max_req": 0.0,
        }

    best_cont = np.full((n, r9k.N_STAGES), np.inf, dtype=float)
    best_req = np.full((n, r9k.N_STAGES), np.inf, dtype=float)

    vf, vp, af, ba, bu, ag = [np.asarray(x, float) for x in data["eval_flat"]]

    for tr in transitions:
        hold = np.asarray(tr["hold_index"], dtype=np.intp)
        hit = old_repl_support[hold]
        preimage_all_tests += int(np.count_nonzero(hit))
        preimage_nodes_all |= hit
        pair_phase = age >= 3.0 * Ts - 1e-10
        hp = hit & pair_phase
        hb = hit & ~pair_phase
        preimage_pair_phase_tests += int(np.count_nonzero(hp))
        preimage_before_pair_tests += int(np.count_nonzero(hb))
        preimage_nodes_pair |= hp
        if np.any(hit):
            earliest_source_age = min(earliest_source_age, float(np.min(age[hit])))
            ids = np.flatnonzero(hit)
            # Keep only a few deterministic witnesses per action.
            for i in ids[:3]:
                preimage_rows.append({
                    "kind": "OLD_REPL_VALUE_GAP_PREIMAGE",
                    "action": float(tr.get("action", math.nan)),
                    "node_index": int(i),
                    "v_f": float(vf[i]), "v_p": float(vp[i]), "a_f": float(af[i]),
                    "bar_a": float(ba[i]), "bar_u": float(bu[i]), "age_s": float(ag[i]),
                    "pair_phase_age_ge_3Ts": bool(pair_phase[i]),
                    "successor_cell_gap_m": float(old_repl_cell_gap[hold[i]]),
                })

        hold_stage = [cflat[s][hold] for s in range(r9k.N_STAGES)]
        adopt_old = r9hg.robust_adopt_future(cells[..., r9k.POST], tr, old_groups, old_age_cell)
        adopt_repl = r9hg.robust_adopt_future(cells[..., r9k.POST], tr, repl_groups, repl_age_cell)
        futures = r9k.service_future_algebra(
            post_hold=hold_stage[r9k.POST], entry_hold=hold_stage[r9k.ENTRY],
            frag_hold=hold_stage[r9k.FRAG], cred_hold=hold_stage[r9k.CRED],
            old_last_hold=hold_stage[r9k.OLD_LAST], repl_last_hold=hold_stage[r9k.REPL_LAST],
            adopt_old=adopt_old, adopt_repl=adopt_repl,
        )
        step = np.asarray(tr["step_loss_upper"], float)
        close = np.asarray(tr["closing_end"], float)
        conts = [np.maximum(close + np.asarray(f, float), 0.0) for f in futures]
        reqs = [np.maximum(step, c) for c in conts]
        for s in range(r9k.N_STAGES):
            best_cont[:, s] = np.minimum(best_cont[:, s], conts[s])
            best_req[:, s] = np.minimum(best_req[:, s], reqs[s])

        for a, b in pairs:
            ac = accum[(a, b)]
            common = ac["common"]
            fd = common & np.isfinite(futures[a]) & np.isfinite(futures[b]) & (np.abs(futures[a] - futures[b]) > TOL)
            cd = common & np.isfinite(conts[a]) & np.isfinite(conts[b]) & (np.abs(conts[a] - conts[b]) > TOL)
            rd = common & np.isfinite(reqs[a]) & np.isfinite(reqs[b]) & (np.abs(reqs[a] - reqs[b]) > TOL)
            ac["future_tests"] += int(np.count_nonzero(fd))
            ac["cont_tests"] += int(np.count_nonzero(cd))
            ac["req_tests"] += int(np.count_nonzero(rd))
            ac["future_nodes"] |= fd
            ac["cont_nodes"] |= cd
            ac["req_nodes"] |= rd
            ac["max_future"] = max(ac["max_future"], finite_sup_abs(futures[a], futures[b], common))
            ac["max_cont"] = max(ac["max_cont"], finite_sup_abs(conts[a], conts[b], common))
            ac["max_req"] = max(ac["max_req"], finite_sup_abs(reqs[a], reqs[b], common))

    if not math.isfinite(earliest_source_age):
        earliest_source_age = -1.0

    cp = common_predecessors(contract)
    states = contract.get("states", {})
    declared_pair = frozenset(contract.get("comparison_pair", []))
    rows: list[dict] = []
    target_rows: list[dict] = []
    diagnostic_active_rows: list[dict] = []

    for a, b in pairs:
        name_a, name_b = r9k.STAGE_NAMES[a], r9k.STAGE_NAMES[b]
        ac = accum[(a, b)]
        common = ac["common"]
        common_n = int(np.count_nonzero(common))
        value_gap = finite_sup_abs(ue[:, a], ue[:, b], common)
        current_value_diff_nodes = int(np.count_nonzero(common & (np.abs(ue[:, a] - ue[:, b]) > TOL)))
        cont_min_gap = finite_sup_abs(best_cont[:, a], best_cont[:, b], common)
        req_min_gap = finite_sup_abs(best_req[:, a], best_req[:, b], common)
        cont_min_nodes = int(np.count_nonzero(common & (np.abs(best_cont[:, a] - best_cont[:, b]) > TOL)))
        req_min_nodes = int(np.count_nonzero(common & (np.abs(best_req[:, a] - best_req[:, b]) > TOL)))

        sa, sb = states.get(name_a, {}), states.get(name_b, {})
        same_liveness = bool(sa and sb and sa.get("liveness_bound") == sb.get("liveness_bound"))
        preds = cp.get(tuple(sorted((name_a, name_b))), [])
        # Pairwise matched-current-chi evidence is only explicitly declared for
        # the R9-J comparison pair.  Do not infer pairwise equality merely from
        # per-state booleans.
        matched_chi_declared = frozenset((name_a, name_b)) == declared_pair
        target_eligible = bool(same_liveness and matched_chi_declared and preds)
        value_active = bool(req_min_nodes > 0 and req_min_gap > TOL)
        row = {
            "state_a": name_a, "state_b": name_b,
            "common_nodes": common_n,
            "same_liveness": same_liveness,
            "common_predecessors": ";".join(preds),
            "matched_current_chi_pair_declared": matched_chi_declared,
            "target_eligible": target_eligible,
            "future_diff_tests": ac["future_tests"],
            "future_diff_nodes": int(np.count_nonzero(common & ac["future_nodes"])),
            "max_future_gap_m": ac["max_future"],
            "continuation_diff_tests": ac["cont_tests"],
            "continuation_diff_nodes": int(np.count_nonzero(common & ac["cont_nodes"])),
            "max_continuation_gap_m": ac["max_cont"],
            "requirement_diff_tests": ac["req_tests"],
            "requirement_diff_nodes": int(np.count_nonzero(common & ac["req_nodes"])),
            "max_requirement_gap_m": ac["max_req"],
            "continuation_control_min_diff_nodes": cont_min_nodes,
            "max_continuation_control_min_gap_m": cont_min_gap,
            "requirement_control_min_diff_nodes": req_min_nodes,
            "max_requirement_control_min_gap_m": req_min_gap,
            "current_full_gfp_value_diff_nodes": current_value_diff_nodes,
            "max_current_full_gfp_value_gap_m": value_gap,
            "value_active_before_fallback": value_active,
        }
        rows.append(row)
        if target_eligible:
            target_rows.append(row)
        if value_active or value_gap > TOL:
            diagnostic_active_rows.append(row)

    rows.sort(key=lambda r: (
        int(r["target_eligible"]),
        r["max_requirement_control_min_gap_m"],
        r["max_current_full_gfp_value_gap_m"],
        r["max_future_gap_m"],
    ), reverse=True)
    diagnostic_active_rows.sort(key=lambda r: (
        r["max_requirement_control_min_gap_m"],
        r["max_current_full_gfp_value_gap_m"],
    ), reverse=True)

    target_value_active = sum(1 for r in target_rows if r["value_active_before_fallback"])
    target_current_gap = sum(1 for r in target_rows if r["max_current_full_gfp_value_gap_m"] > TOL)
    value_active_pairs = sum(1 for r in rows if r["value_active_before_fallback"])
    current_gap_pairs = sum(1 for r in rows if r["max_current_full_gfp_value_gap_m"] > TOL)

    print(
        "R9N_OLD_REPL_VALUE_GAP_SUPPORT "
        f"cells={old_repl_support_cells} max_cell_gap_m={max_cell_gap:.12g} "
        f"preimage_all_tests={preimage_all_tests} preimage_all_nodes={int(np.count_nonzero(preimage_nodes_all))} "
        f"preimage_before_pair_phase_tests={preimage_before_pair_tests} "
        f"preimage_pair_phase_tests={preimage_pair_phase_tests} "
        f"preimage_pair_phase_nodes={int(np.count_nonzero(preimage_nodes_pair))} "
        f"earliest_source_age_s={earliest_source_age:.12g}",
        flush=True,
    )
    print(
        "R9N_PAIR_VALUE_SCREEN "
        f"pairs={len(rows)} target_eligible_pairs={len(target_rows)} "
        f"value_active_pairs={value_active_pairs} current_gfp_gap_pairs={current_gap_pairs} "
        f"target_value_active_pairs={target_value_active} target_current_gfp_gap_pairs={target_current_gap}",
        flush=True,
    )
    for r in rows[:5]:
        print(
            "R9N_PAIR "
            f"pair={r['state_a']}|{r['state_b']} target={int(r['target_eligible'])} "
            f"future_nodes={r['future_diff_nodes']} req_min_nodes={r['requirement_control_min_diff_nodes']} "
            f"req_min_gap_m={r['max_requirement_control_min_gap_m']:.12g} "
            f"full_gfp_gap_nodes={r['current_full_gfp_value_diff_nodes']} "
            f"full_gfp_gap_m={r['max_current_full_gfp_value_gap_m']:.12g}",
            flush=True,
        )

    # Decision logic.  No new expensive GFP should be launched unless a pair is
    # value-active on the reachable image and also satisfies the target semantic
    # constraints.  Diagnostic value gaps without pairwise matched-chi evidence
    # are reported but not promoted.
    if target_value_active > 0 or target_current_gap > 0:
        classification = "TARGET_ELIGIBLE_MATCHED_CHI_PAIR_HAS_REACHABLE_VALUE_NONDominance"
        next_action = "R9O_VERIFY_TARGET_PAIR_SEMANTICS_AND_PROMOTE_ONLY_THEN_TO_INTERVAL_CERTIFICATION"
    elif preimage_all_tests > 0 and preimage_pair_phase_tests == 0:
        classification = "GLOBAL_LAST_STAGE_VALUE_GAP_HAS_PREIMAGE_BUT_IS_PHASE_UNREACHABLE_FROM_MATCHED_CHI_PAIR"
        next_action = "R9O_TEST_WHETHER_VALUE_GAP_PHASE_CAN_BE_REALIZED_BY_A_DECLARED_SAME_CHI_COMMON_HISTORY_STATE_WITHOUT_CHANGING_PHYSICS"
    elif preimage_all_tests == 0:
        classification = "GLOBAL_LAST_STAGE_VALUE_GAP_HAS_NO_ONE_STEP_PHYSICAL_PREIMAGE_IN_CURRENT_GRID"
        next_action = "R9O_REASSESS_PHYSICAL_SUCCESSOR_GEOMETRY_OR_STOP_MATCHED_CHI_POSITIVE_PROMOTION_FOR_CURRENT_MODEL"
    elif value_active_pairs > 0 or current_gap_pairs > 0:
        classification = "VALUE_ACTIVE_PAIRS_EXIST_BUT_NONE_SATISFY_DECLARED_MATCHED_CHI_TARGET_CONSTRAINTS"
        next_action = "R9O_AUDIT_WHETHER_ANY_VALUE_ACTIVE_PAIR_CAN_BE_REALIZED_WITH_MATCHED_CURRENT_CHI_FROM_PROTOCOL_EVIDENCE"
    else:
        classification = "NO_BELLMAN_ACTIVE_SERVICE_PAIR_FOUND_ON_CURRENT_REACHABLE_VALUE_IMAGE"
        next_action = "R9O_STOP_NEW_PAIR_GFP_AND_REASSESS_MODEL_COUPLING"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result = RESULTS / f"P3B1_R9N_RESULT_{stamp}.json"
    latest = RESULTS / "P3B1_R9N_LATEST.json"
    csvp = RESULTS / f"P3B1_R9N_PAIR_VALUE_SCREEN_{stamp}.csv"
    pre_csv = RESULTS / f"P3B1_R9N_VALUE_GAP_PREIMAGE_{stamp}.csv"
    lemma = RESULTS / f"P3B1_R9N_REACHABLE_IMAGE_NECESSITY_LEMMA_{stamp}.tex"
    manifest = RESULTS / f"P3B1_R9N_MANIFEST_{stamp}.sha256"

    write_csv(csvp, rows)
    write_csv(pre_csv, preimage_rows[:256])
    tex = r"""% R9-N reachable-successor value-image necessity lemma
\begin{lemma}[Reachable value-image necessity]
For two service states to induce different values under a Bellman operator with
common physical transition, it is not sufficient that their downstream value
functions differ somewhere on the augmented grid.  A necessary condition at a
source state is that at least one admissible controlled successor lies in a
region where the relevant downstream continuation values differ, and that this
difference survives the subsequent Bellman max/min operations.  Therefore a
global downstream value gap whose support has empty intersection with the
reachable successor image of the compared source states cannot produce a
source-state viability separation.
\end{lemma}
% This is an algebraic necessity statement for the instantiated point-action
% Bellman screen; continuous-action/state promotion remains separate.
"""
    atomic_write(lemma, tex)

    out = {
        "schema": "SCV_P3B1_R9N_REACHABLE_VALUE_IMAGE_SCREEN_V1",
        "status": "PASS",
        "timestamp_utc": stamp,
        "classification": classification,
        "next_action": next_action,
        "continuous_state_separation_certified": False,
        "deployment_protocol_certified": False,
        "metrics": {
            "point_action_width": float(args.action_width),
            "point_action_count": len(actions),
            "old_replacement_gap_support_cells": old_repl_support_cells,
            "max_old_replacement_cell_gap_m": max_cell_gap,
            "old_replacement_gap_preimage_all_tests": preimage_all_tests,
            "old_replacement_gap_preimage_all_nodes": int(np.count_nonzero(preimage_nodes_all)),
            "old_replacement_gap_preimage_before_pair_phase_tests": preimage_before_pair_tests,
            "old_replacement_gap_preimage_pair_phase_tests": preimage_pair_phase_tests,
            "old_replacement_gap_preimage_pair_phase_nodes": int(np.count_nonzero(preimage_nodes_pair)),
            "earliest_gap_preimage_source_age_s": earliest_source_age,
            "screened_stage_pairs": len(rows),
            "target_eligible_pairs": len(target_rows),
            "value_active_pairs_before_fallback": value_active_pairs,
            "current_full_gfp_gap_pairs": current_gap_pairs,
            "target_value_active_pairs": target_value_active,
            "target_current_full_gfp_gap_pairs": target_current_gap,
        },
        "top_pairs": rows[:5],
        "gates": {
            "upstream_r9m_successor_image_miss": True,
            "upstream_r9k_full_gfp_collapsed": True,
            "trace_only_pair_search_stopped": True,
            "reachable_successor_value_image_screen_completed": True,
            "deployment_protocol_certified": False,
            "continuous_state_separation_certified": False,
        },
        "artifacts": {
            "pair_value_screen_csv": str(csvp),
            "value_gap_preimage_csv": str(pre_csv),
            "conditional_lemma_tex": str(lemma),
        },
    }
    text = json.dumps(out, indent=2, sort_keys=True)
    atomic_write(result, text)
    atomic_write(latest, text)
    mfiles = [Path(__file__), result, csvp, pre_csv, lemma, r9m_path, r9k_path, contract_path]
    atomic_write(manifest, "".join(f"{sha256_file(p)}  {p}\n" for p in mfiles if p.exists()))

    print("=== R9-N DECISION ===", flush=True)
    print("R9N_REACHABLE_VALUE_IMAGE_SCREEN_GATE=PASS", flush=True)
    print(f"R9N_CLASSIFICATION={classification}", flush=True)
    print("TRACE_NONDOMINANCE_SUFFICIENT_FOR_GFP_SEPARATION=NO", flush=True)
    print("CONTINUOUS_STATE_SEPARATION_CERTIFIED=NO", flush=True)
    print("R9N_EXECUTION=PASS", flush=True)
    print(f"R9N_NEXT_ACTION={next_action}", flush=True)
    print(f"RESULT_JSON={result}", flush=True)
    print(f"PAIR_VALUE_SCREEN_CSV={csvp}", flush=True)
    print(f"VALUE_GAP_PREIMAGE_CSV={pre_csv}", flush=True)
    print(f"CONDITIONAL_LEMMA_TEX={lemma}", flush=True)
    print(f"MANIFEST={manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
