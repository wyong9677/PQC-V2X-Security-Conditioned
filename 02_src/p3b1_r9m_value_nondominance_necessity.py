from __future__ import annotations

import argparse
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
RESULTS = ROOT / "04_results"
RESULTS.mkdir(parents=True, exist_ok=True)
TOL = 1e-9
EXPECTED_R9L = "CANDIDATE_FATE_DIFFERENCE_REMAINS_NONBINDING_EVEN_UNDER_FORCED_ADOPTION_COUNTERFACTUAL"
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


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def top_local_indices(score: np.ndarray, mask: np.ndarray, k: int = 8, largest: bool = True) -> np.ndarray:
    ids = np.flatnonzero(mask & np.isfinite(score))
    if ids.size == 0:
        return np.asarray([], dtype=np.int64)
    vals = score[ids]
    if ids.size <= k:
        order = np.argsort(vals)
        if largest:
            order = order[::-1]
        return ids[order]
    if largest:
        part = np.argpartition(vals, -k)[-k:]
        part = part[np.argsort(vals[part])[::-1]]
    else:
        part = np.argpartition(vals, k - 1)[:k]
        part = part[np.argsort(vals[part])]
    return ids[part]


def self_test() -> None:
    # Three exact algebra cases for B_i=max(step,C_i).
    step = np.asarray([5.0, 2.0, 1.0])
    c1 = np.asarray([3.0, 3.0, 4.0])
    c2 = np.asarray([4.0, 4.0, 2.0])
    b1 = np.maximum(step, c1)
    b2 = np.maximum(step, c2)
    # case 0: continuation differs but common step erases it
    assert b1[0] == b2[0] == 5.0
    # case 1: both continuations active, difference survives
    assert b1[1] == 3.0 and b2[1] == 4.0
    # case 2: one continuation active, difference survives
    assert b1[2] == 4.0 and b2[2] == 2.0
    diff = np.abs(c1 - c2) > TOL
    erased = diff & (np.abs(b1 - b2) <= TOL)
    assert erased.tolist() == [True, False, False]
    deficit = np.maximum(step - np.maximum(c1, c2), 0.0)
    assert np.allclose(deficit, [1.0, 0.0, 0.0])

    # min_u can erase actionwise continuation differences.
    # q1=[1,4], q2=[2,1] => each action differs, but both minima are 1.
    q1 = np.asarray([1.0, 4.0])
    q2 = np.asarray([2.0, 1.0])
    assert np.any(np.abs(q1 - q2) > TOL)
    assert abs(float(np.min(q1)) - float(np.min(q2))) <= TOL
    print("R9M_INTERNAL_SELF_TEST=PASS", flush=True)


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
        raise ValueError("R9M_BAD_ACTION_WIDTH")

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

    r9l_path = RESULTS / "P3B1_R9L_LATEST.json"
    r9k_path = RESULTS / "P3B1_R9K_LATEST.json"
    if not r9l_path.exists() or not r9k_path.exists():
        raise RuntimeError("R9M_MISSING_UPSTREAM_LATEST")
    r9l = load_json(r9l_path)
    r9ku = load_json(r9k_path)
    if r9l.get("status") != "PASS" or r9l.get("classification") != EXPECTED_R9L:
        raise RuntimeError(f"R9M_R9L_GATE_FAIL classification={r9l.get('classification')}")
    if r9ku.get("status") != "PASS" or r9ku.get("classification") != EXPECTED_R9K:
        raise RuntimeError(f"R9M_R9K_GATE_FAIL classification={r9ku.get('classification')}")
    lm = r9l.get("metrics", {})
    if int(lm.get("actionwise_pair_difference_nodes", -1)) != 0:
        raise RuntimeError("R9M_EXPECTED_ZERO_ACTIONWISE_PAIR_DIFFERENCE")
    if float(lm.get("max_old_vs_replacement_last_gap_m", 0.0)) <= TOL:
        raise RuntimeError("R9M_EXPECTED_MATERIAL_LAST_STAGE_VALUE_GAP")

    data = r9d.reconstruct_timestamped_geometry(r3, r5, r6, r7, r9c)
    Ts = float(data["Ts"])
    actions = r9hb.refinement_actions(float(args.action_width))

    print("=== P3-B1-R9-M VALUE-NONDOMINANCE NECESSITY AUDIT ===", flush=True)
    print("UPSTREAM_R9L_FORCED_ADOPT_STILL_COLLAPSED=PASS", flush=True)
    print("UPSTREAM_R9K_FULL_GFP_COLLAPSE=PASS", flush=True)
    print("QUESTION=WHERE_DOES_LAST_STAGE_VALUE_DIFFERENCE_FAIL_TO_BECOME_BELLMAN_ACTIVE", flush=True)
    print("TRACE_NONDOMINANCE_IS_SUFFICIENT_FOR_GFP_SEPARATION=NO", flush=True)
    print("CONTINUOUS_VALUE_NONDOMINANCE_THEOREM_CERTIFIED=NO", flush=True)
    print("CONTINUOUS_STATE_SEPARATION_CERTIFIED=NO", flush=True)
    print(f"POINT_ACTION_WIDTH={float(args.action_width):.12g}", flush=True)
    print(f"POINT_ACTION_COUNT={len(actions)}", flush=True)

    cfg, transitions = r9hg.build_physical_transitions(
        data, actions, r3, b0, r9e, chunk_size=int(args.chunk_size)
    )
    age = np.asarray(data["eval_flat"][5], float)
    common = age >= 3.0 * Ts - 1e-10
    last_mask = age >= 4.0 * Ts - 1e-10
    common_n = int(np.count_nonzero(common))

    old_groups = r9hg.build_descriptor_range_groups(data, 2.0 * Ts, b0)
    repl_groups = r9hg.build_descriptor_range_groups(data, 0.0, b0)
    old_age_cell, old_ok = r9hg.scalar_cell_index(np.asarray(data["lookup_axes"][5], float), 3.0 * Ts)
    repl_age_cell, repl_ok = r9hg.scalar_cell_index(np.asarray(data["lookup_axes"][5], float), 1.0 * Ts)
    if not old_ok or not repl_ok:
        raise RuntimeError("R9M_ADOPTION_AGE_LOOKUP_FAIL")

    print("R9M_STAGE_START=recompute_actual_upper_full_gfp", flush=True)
    upper = r9k.solve_full_service_fixed_point(
        data=data, cfg=cfg, transitions=transitions,
        fallback_eval=np.asarray(data["eval_fallback"], float),
        old_groups=old_groups, repl_groups=repl_groups,
        old_adopt_age_cell=old_age_cell, repl_adopt_age_cell=repl_age_cell,
        r3=r3, r9hg=r9hg, mode="upper", label="r9m_actual_upper",
    )
    if not upper.converged:
        raise RuntimeError("R9M_UPPER_GFP_NOT_CONVERGED")

    lookup_shape = tuple(data["lookup_shape"])
    eval_idx = np.asarray(data["eval_idx"], dtype=np.int64)
    h = upper.h_flat.reshape(lookup_shape + (r9k.N_STAGES,))
    cells = r3.cell_corner_max(h)
    cflat = [cells[..., s].reshape(-1) for s in range(r9k.N_STAGES)]
    ue = np.asarray(upper.h_flat[eval_idx, :], float)
    actual_pair_gap = finite_sup_abs(ue[:, r9k.FRAG], ue[:, r9k.CRED], common)
    global_last_gap = finite_sup_abs(ue[:, r9k.OLD_LAST], ue[:, r9k.REPL_LAST], last_mask)
    if actual_pair_gap > 1e-8:
        raise RuntimeError(f"R9M_UPSTREAM_PAIR_COLLAPSE_REPRO_FAIL gap={actual_pair_gap}")

    n = len(eval_idx)
    best_cont_old = np.full(n, np.inf)
    best_cont_repl = np.full(n, np.inf)
    best_req_old = np.full(n, np.inf)
    best_req_repl = np.full(n, np.inf)

    any_future_diff = np.zeros(n, bool)
    any_cont_diff = np.zeros(n, bool)
    any_req_diff = np.zeros(n, bool)
    any_zero_clip_erasure = np.zeros(n, bool)
    any_step_erasure = np.zeros(n, bool)

    total_tests = 0
    future_diff_tests = 0
    continuation_diff_tests = 0
    requirement_diff_tests = 0
    zero_clip_erasure_tests = 0
    step_loss_erasure_tests = 0
    step_dominates_both_tests = 0
    continuation_active_tests = 0
    max_future_gap = 0.0
    max_cont_gap = 0.0
    max_req_gap = 0.0
    min_activation_deficit = math.inf
    max_activation_deficit = 0.0
    sum_activation_deficit = 0.0
    n_activation_deficit = 0
    witness_rows: list[dict] = []

    vf, vp, af, ba, bu, ag = [np.asarray(x, float) for x in data["eval_flat"]]

    for tr in transitions:
        action = float(tr.get("action", math.nan))
        hold = np.asarray(tr["hold_index"], dtype=np.intp)
        f_old = np.asarray(cflat[r9k.OLD_LAST][hold], float)
        f_repl = np.asarray(cflat[r9k.REPL_LAST][hold], float)
        step = np.asarray(tr["step_loss_upper"], float)
        close = np.asarray(tr["closing_end"], float)
        raw_old = close + f_old
        raw_repl = close + f_repl
        cont_old = np.maximum(raw_old, 0.0)
        cont_repl = np.maximum(raw_repl, 0.0)
        req_old = np.maximum(step, cont_old)
        req_repl = np.maximum(step, cont_repl)

        finite = common & np.isfinite(f_old) & np.isfinite(f_repl) & np.isfinite(step) & np.isfinite(close)
        fd = finite & (np.abs(f_old - f_repl) > TOL)
        cd = finite & (np.abs(cont_old - cont_repl) > TOL)
        rd = finite & (np.abs(req_old - req_repl) > TOL)
        zero_erase = fd & ~cd
        step_erase = cd & ~rd
        step_both = finite & (step >= cont_old - TOL) & (step >= cont_repl - TOL)
        active = finite & ((cont_old > step + TOL) | (cont_repl > step + TOL))

        total_tests += int(np.count_nonzero(finite))
        future_diff_tests += int(np.count_nonzero(fd))
        continuation_diff_tests += int(np.count_nonzero(cd))
        requirement_diff_tests += int(np.count_nonzero(rd))
        zero_clip_erasure_tests += int(np.count_nonzero(zero_erase))
        step_loss_erasure_tests += int(np.count_nonzero(step_erase))
        step_dominates_both_tests += int(np.count_nonzero(step_both))
        continuation_active_tests += int(np.count_nonzero(active))
        any_future_diff |= fd
        any_cont_diff |= cd
        any_req_diff |= rd
        any_zero_clip_erasure |= zero_erase
        any_step_erasure |= step_erase

        max_future_gap = max(max_future_gap, finite_sup_abs(f_old, f_repl, finite))
        max_cont_gap = max(max_cont_gap, finite_sup_abs(cont_old, cont_repl, finite))
        max_req_gap = max(max_req_gap, finite_sup_abs(req_old, req_repl, finite))

        if np.any(step_erase):
            deficit = np.maximum(step - np.maximum(cont_old, cont_repl), 0.0)
            vals = deficit[step_erase]
            if vals.size:
                min_activation_deficit = min(min_activation_deficit, float(np.min(vals)))
                max_activation_deficit = max(max_activation_deficit, float(np.max(vals)))
                sum_activation_deficit += float(np.sum(vals))
                n_activation_deficit += int(vals.size)

        best_cont_old = np.minimum(best_cont_old, cont_old)
        best_cont_repl = np.minimum(best_cont_repl, cont_repl)
        best_req_old = np.minimum(best_req_old, req_old)
        best_req_repl = np.minimum(best_req_repl, req_repl)

        # Compact deterministic witness sampling: largest reachable future gaps
        # and smallest positive activation deficits for this action.
        future_gap = np.abs(f_old - f_repl)
        for i in top_local_indices(future_gap, fd, k=4, largest=True):
            witness_rows.append({
                "kind": "MAX_REACHABLE_FUTURE_GAP", "action": action, "node_index": int(i),
                "v_f": float(vf[i]), "v_p": float(vp[i]), "a_f": float(af[i]),
                "bar_a": float(ba[i]), "bar_u": float(bu[i]), "age_s": float(ag[i]),
                "old_last_successor_value_m": float(f_old[i]),
                "replacement_last_successor_value_m": float(f_repl[i]),
                "future_gap_m": float(future_gap[i]), "closing_end_m": float(close[i]),
                "continuation_old_m": float(cont_old[i]), "continuation_replacement_m": float(cont_repl[i]),
                "step_loss_m": float(step[i]), "requirement_old_m": float(req_old[i]),
                "requirement_replacement_m": float(req_repl[i]),
                "activation_deficit_m": float(max(step[i] - max(cont_old[i], cont_repl[i]), 0.0)),
            })
        deficit = np.maximum(step - np.maximum(cont_old, cont_repl), 0.0)
        for i in top_local_indices(deficit, step_erase & (deficit > TOL), k=4, largest=False):
            witness_rows.append({
                "kind": "MIN_STEP_ERASURE_DEFICIT", "action": action, "node_index": int(i),
                "v_f": float(vf[i]), "v_p": float(vp[i]), "a_f": float(af[i]),
                "bar_a": float(ba[i]), "bar_u": float(bu[i]), "age_s": float(ag[i]),
                "old_last_successor_value_m": float(f_old[i]),
                "replacement_last_successor_value_m": float(f_repl[i]),
                "future_gap_m": float(future_gap[i]), "closing_end_m": float(close[i]),
                "continuation_old_m": float(cont_old[i]), "continuation_replacement_m": float(cont_repl[i]),
                "step_loss_m": float(step[i]), "requirement_old_m": float(req_old[i]),
                "requirement_replacement_m": float(req_repl[i]),
                "activation_deficit_m": float(deficit[i]),
            })

    future_diff_nodes = int(np.count_nonzero(common & any_future_diff))
    cont_diff_nodes = int(np.count_nonzero(common & any_cont_diff))
    req_diff_nodes = int(np.count_nonzero(common & any_req_diff))
    zero_erase_nodes = int(np.count_nonzero(common & any_zero_clip_erasure))
    step_erase_nodes = int(np.count_nonzero(common & any_step_erasure))

    cont_min_diff = common & (np.abs(best_cont_old - best_cont_repl) > TOL)
    req_min_diff = common & (np.abs(best_req_old - best_req_repl) > TOL)
    cont_min_diff_nodes = int(np.count_nonzero(cont_min_diff))
    req_min_diff_nodes = int(np.count_nonzero(req_min_diff))
    max_cont_min_gap = finite_sup_abs(best_cont_old, best_cont_repl, common)
    max_req_min_gap = finite_sup_abs(best_req_old, best_req_repl, common)

    if requirement_diff_tests != 0 or req_diff_nodes != 0 or req_min_diff_nodes != 0 or max_req_gap > 1e-8:
        raise RuntimeError(
            "R9M_R9L_ACTIONWISE_ZERO_REPRO_FAIL "
            f"tests={requirement_diff_tests} nodes={req_diff_nodes} min_nodes={req_min_diff_nodes} max={max_req_gap}"
        )

    if not math.isfinite(min_activation_deficit):
        min_activation_deficit = 0.0
    mean_activation_deficit = (
        sum_activation_deficit / n_activation_deficit if n_activation_deficit else 0.0
    )

    print(
        "R9M_REACHABLE_SUCCESSOR_VALUE "
        f"tests={total_tests} future_diff_tests={future_diff_tests} future_diff_nodes={future_diff_nodes} "
        f"max_future_gap_m={max_future_gap:.12g} global_last_stage_gap_m={global_last_gap:.12g}",
        flush=True,
    )
    print(
        "R9M_CONTINUATION_TRANSFER "
        f"continuation_diff_tests={continuation_diff_tests} continuation_diff_nodes={cont_diff_nodes} "
        f"zero_clip_erasure_tests={zero_clip_erasure_tests} zero_clip_erasure_nodes={zero_erase_nodes} "
        f"max_continuation_gap_m={max_cont_gap:.12g}",
        flush=True,
    )
    print(
        "R9M_PHYSICAL_STEP_DOMINANCE "
        f"requirement_diff_tests={requirement_diff_tests} requirement_diff_nodes={req_diff_nodes} "
        f"step_loss_erasure_tests={step_loss_erasure_tests} step_loss_erasure_nodes={step_erase_nodes} "
        f"step_dominates_both_tests={step_dominates_both_tests} continuation_active_tests={continuation_active_tests} "
        f"min_activation_deficit_m={min_activation_deficit:.12g} "
        f"mean_activation_deficit_m={mean_activation_deficit:.12g} max_activation_deficit_m={max_activation_deficit:.12g}",
        flush=True,
    )
    print(
        "R9M_CONTROL_MIN_COUNTERFACTUAL "
        f"continuation_only_min_diff_nodes={cont_min_diff_nodes}/{common_n} "
        f"max_continuation_only_min_gap_m={max_cont_min_gap:.12g} "
        f"actual_requirement_min_diff_nodes={req_min_diff_nodes}/{common_n} "
        f"max_actual_requirement_min_gap_m={max_req_min_gap:.12g}",
        flush=True,
    )

    # Point-action necessity audit: a trace/value distinction can affect this
    # pair only if it survives the successor image, zero clipping, the common
    # physical step-loss max, and control minimization.  This is an algebraic
    # statement for the instantiated point-action Bellman operator; it is NOT a
    # continuous-domain theorem.
    necessity_gate = (
        actual_pair_gap <= 1e-8
        and requirement_diff_tests == 0
        and req_min_diff_nodes == 0
        and float(lm.get("max_old_vs_replacement_last_gap_m", 0.0)) > TOL
    )

    if future_diff_tests == 0:
        classification = "PAIR_SUCCESSOR_IMAGE_AVOIDS_ALL_GLOBAL_LAST_STAGE_VALUE_DIFFERENCE"
        next_action = "R9N_SCREEN_SERVICE_PAIRS_ON_REACHABLE_SUCCESSOR_VALUE_IMAGE_BEFORE_ANY_NEW_GFP"
    elif continuation_diff_tests == 0:
        classification = "ZERO_CLIPPING_ERASES_REACHABLE_LAST_STAGE_VALUE_DIFFERENCE"
        next_action = "R9N_SCREEN_FOR_NONZERO_CONTROL_CONTINUATION_VALUE_DIFFERENCE"
    elif cont_min_diff_nodes == 0:
        classification = "CONTROL_MINIMIZATION_ERASES_REACHABLE_CONTINUATION_VALUE_NONDOMINANCE"
        next_action = "R9N_SCREEN_PAIRS_BY_CONTROL_OPTIMIZED_CONTINUATION_VALUE_NONDOMINANCE"
    elif requirement_diff_tests == 0 and step_loss_erasure_tests > 0:
        classification = "PHYSICAL_ONE_STEP_LOSS_DOMINATES_ALL_BELLMAN_ACTIVE_SERVICE_VALUE_DIFFERENCE"
        next_action = "R9N_SCREEN_PAIRS_FOR_BELLMAN_ACTIVE_CONTINUATION_DIFFERENCE_ABOVE_PHYSICAL_STEP_LOSS"
    else:
        classification = "MIXED_VALUE_DOMINANCE_REQUIRES_TARGETED_BELLMAN_SCREEN"
        next_action = "R9N_BUILD_VALUE_LEVEL_BELLMAN_NONDominance_SCREEN"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result = RESULTS / f"P3B1_R9M_RESULT_{stamp}.json"
    latest = RESULTS / "P3B1_R9M_LATEST.json"
    csvp = RESULTS / f"P3B1_R9M_VALUE_WITNESSES_{stamp}.csv"
    theorem = RESULTS / f"P3B1_R9M_VALUE_NONDominance_LEMMA_{stamp}.tex"
    manifest = RESULTS / f"P3B1_R9M_MANIFEST_{stamp}.sha256"

    # Dedupe compact witness rows.
    uniq = {}
    for row in witness_rows:
        uniq[(row["kind"], row["action"], row["node_index"])] = row
    witness_rows = list(uniq.values())
    witness_rows.sort(key=lambda r: (r["kind"], r["activation_deficit_m"], -r["future_gap_m"]))
    write_csv(csvp, witness_rows[:160])

    tex = r"""% R9-M point-action value-nondominance necessity lemma
\begin{lemma}[Bellman-active value distinction is necessary for stage separation]
Consider two service states $q_1,q_2$ that share the same physical one-step
loss $L(x,u)$, closing term $g(x,u)$, control set, and fallback cap.  Let
\[
 C_i(x,u)=\max\{0,\,g(x,u)+V_i(x^+)\},\qquad
 B_i(x,u)=\max\{L(x,u),\,C_i(x,u)\}.
\]
For the instantiated point-action Bellman operator, if
$B_1(x,u)=B_2(x,u)$ for every admissible action at a state $x$, then the
cooperative minima are equal at $x$ and a common fallback cap cannot create a
service-state difference there.  Consequently, trace-set non-dominance or a
successor-value difference is not sufficient for viability separation: the
value distinction must survive successor reachability, clipping, the
one-step physical maximum, and control minimization.
\end{lemma}
% This lemma is algebraic for the instantiated point-action operator.  It is
% not a continuous-action or continuous-state certificate.
"""
    atomic_write(theorem, tex)

    out = {
        "schema": "SCV_P3B1_R9M_VALUE_NONDOMINANCE_NECESSITY_V1",
        "status": "PASS",
        "timestamp_utc": stamp,
        "classification": classification,
        "next_action": next_action,
        "continuous_state_separation_certified": False,
        "continuous_value_nondominance_theorem_certified": False,
        "trace_nondominance_sufficient_for_gfp_separation": False,
        "metrics": {
            "point_action_width": float(args.action_width),
            "point_action_count": len(actions),
            "common_nodes": common_n,
            "total_state_action_tests": total_tests,
            "global_last_stage_gap_m": global_last_gap,
            "reachable_future_value_difference_tests": future_diff_tests,
            "reachable_future_value_difference_nodes": future_diff_nodes,
            "max_reachable_future_value_gap_m": max_future_gap,
            "continuation_difference_tests": continuation_diff_tests,
            "continuation_difference_nodes": cont_diff_nodes,
            "zero_clip_erasure_tests": zero_clip_erasure_tests,
            "zero_clip_erasure_nodes": zero_erase_nodes,
            "max_continuation_gap_m": max_cont_gap,
            "bellman_requirement_difference_tests": requirement_diff_tests,
            "bellman_requirement_difference_nodes": req_diff_nodes,
            "step_loss_erasure_tests": step_loss_erasure_tests,
            "step_loss_erasure_nodes": step_erase_nodes,
            "step_dominates_both_tests": step_dominates_both_tests,
            "continuation_active_tests": continuation_active_tests,
            "min_activation_deficit_m": min_activation_deficit,
            "mean_activation_deficit_m": mean_activation_deficit,
            "max_activation_deficit_m": max_activation_deficit,
            "continuation_only_control_min_difference_nodes": cont_min_diff_nodes,
            "max_continuation_only_control_min_gap_m": max_cont_min_gap,
            "actual_requirement_control_min_difference_nodes": req_min_diff_nodes,
            "max_actual_requirement_control_min_gap_m": max_req_min_gap,
            "actual_full_gfp_pair_gap_m": actual_pair_gap,
        },
        "gates": {
            "upstream_r9l_forced_adopt_still_collapsed": True,
            "upstream_r9k_full_gfp_collapsed": True,
            "material_global_last_stage_value_difference": global_last_gap > TOL,
            "point_action_value_nondominance_necessity_audit": bool(necessity_gate),
            "continuous_action_value_nondominance_theorem_certified": False,
            "deployment_protocol_certified": False,
            "continuous_state_separation_certified": False,
        },
        "artifacts": {
            "value_witness_csv": str(csvp),
            "conditional_lemma_tex": str(theorem),
        },
    }
    text = json.dumps(out, indent=2, sort_keys=True)
    atomic_write(result, text)
    atomic_write(latest, text)
    mfiles = [Path(__file__), result, csvp, theorem, r9l_path, r9k_path]
    atomic_write(manifest, "".join(f"{sha256_file(p)}  {p}\n" for p in mfiles if p.exists()))

    print("=== R9-M DECISION ===", flush=True)
    print(f"R9M_POINT_ACTION_VALUE_NONDOMINANCE_NECESSITY_GATE={'PASS' if necessity_gate else 'FAIL'}", flush=True)
    print(f"R9M_CLASSIFICATION={classification}", flush=True)
    print("TRACE_NONDOMINANCE_SUFFICIENT_FOR_GFP_SEPARATION=NO", flush=True)
    print("CONTINUOUS_VALUE_NONDOMINANCE_THEOREM_CERTIFIED=NO", flush=True)
    print("CONTINUOUS_STATE_SEPARATION_CERTIFIED=NO", flush=True)
    print("R9M_EXECUTION=PASS", flush=True)
    print(f"R9M_NEXT_ACTION={next_action}", flush=True)
    print(f"RESULT_JSON={result}", flush=True)
    print(f"VALUE_WITNESS_CSV={csvp}", flush=True)
    print(f"CONDITIONAL_LEMMA_TEX={theorem}", flush=True)
    print(f"MANIFEST={manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
