from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
RESULTS = ROOT / "04_results"
RESULTS.mkdir(parents=True, exist_ok=True)
TOL = 1e-9
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


def finite_max(x: np.ndarray, default: float = 0.0) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.max(x)) if x.size else float(default)


def finite_sup_abs(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if mask is not None:
        a = a[mask]
        b = b[mask]
    good = np.isfinite(a) & np.isfinite(b)
    if not np.any(good):
        return 0.0
    return float(np.max(np.abs(a[good] - b[good])))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


@dataclass
class UpperSolve:
    h_flat: np.ndarray
    converged: bool
    iterations: int
    final_delta: float


def solve_forced_adopt_upper(
    *, data: dict, cfg: dict, transitions: list[dict], old_groups, repl_groups,
    old_age_cell: int, repl_age_cell: int, r3, r9k, r9hg,
) -> UpperSolve:
    """Diagnostic counterfactual only: eligible output must be adopted.

    Fallback remains available.  This is intentionally NOT the manuscript
    protocol and is never promoted to a certificate.  Its only purpose is to
    test whether optional hold erases an otherwise material old-vs-replacement
    candidate distinction.
    """
    lookup_shape = tuple(data["lookup_shape"])
    lookup_nodes = int(np.prod(lookup_shape))
    eval_idx = np.asarray(data["eval_idx"], dtype=np.int64)
    n = len(eval_idx)
    age = np.asarray(data["eval_flat"][5], float)
    masks = r9k.stage_masks(age, float(data["Ts"]))
    base = np.asarray(data["lookup_fallback"], float)
    h = np.repeat(base[:, None], r9k.N_STAGES, axis=1)
    fallback_eval = np.asarray(data["eval_fallback"], float)
    tol = float(cfg["fixed_point"]["tolerance_m"])
    max_iter = int(cfg["fixed_point"]["max_iterations"])

    for it in range(1, max_iter + 1):
        shaped = h.reshape(lookup_shape + (r9k.N_STAGES,))
        cells = r3.cell_corner_max(shaped)
        cflat = [cells[..., s].reshape(-1) for s in range(r9k.N_STAGES)]
        old = h[eval_idx, :].copy()
        best = np.full((n, r9k.N_STAGES), np.inf, dtype=float)

        for tr in transitions:
            hold = np.asarray(tr["hold_index"], dtype=np.intp)
            hs = [cflat[s][hold] for s in range(r9k.N_STAGES)]
            adopt_old = r9hg.robust_adopt_future(cells[..., r9k.POST], tr, old_groups, old_age_cell)
            adopt_repl = r9hg.robust_adopt_future(cells[..., r9k.POST], tr, repl_groups, repl_age_cell)
            futures = (
                hs[r9k.ENTRY],
                np.maximum(hs[r9k.FRAG], hs[r9k.CRED]),
                hs[r9k.OLD_LAST],
                hs[r9k.REPL_LAST],
                adopt_old,
                adopt_repl,
            )
            step = np.asarray(tr["step_loss_upper"], float)
            close = np.asarray(tr["closing_end"], float)
            for s, fut in enumerate(futures):
                req = np.maximum(step, close + np.asarray(fut, float))
                req = np.maximum(req, 0.0)
                best[:, s] = np.minimum(best[:, s], req)

        cand = np.minimum(fallback_eval[:, None], best)
        new = old.copy()
        for s, m in enumerate(masks):
            new[m, s] = np.minimum(old[m, s], cand[m, s])
        delta = finite_sup_abs(new, old)
        h[eval_idx, :] = new
        if it == 1 or it % 5 == 0 or delta <= tol:
            print(
                f"R9L_COUNTERFACTUAL_GFP_ITER mode=forced_adopt iteration={it} delta_m={delta:.12g}",
                flush=True,
            )
        if delta <= tol:
            return UpperSolve(h, True, it, delta)
    return UpperSolve(h, False, max_iter, delta)


def self_test() -> None:
    post = np.array([4.0, 2.0, 3.0])
    old = np.array([2.0, 3.0, 1.0])
    repl = np.array([1.0, 4.0, 3.0])
    old_real = np.minimum(old, post)
    repl_real = np.minimum(repl, post)
    direct = np.abs(old - repl) > 1e-12
    erased = direct & (np.abs(old_real - repl_real) <= 1e-12)
    assert direct.tolist() == [True, True, True]
    assert erased.tolist() == [False, True, False]
    fb = np.array([1.0, 3.0])
    a = np.array([2.0, 2.0])
    b = np.array([4.0, 2.5])
    assert np.allclose(np.minimum(fb, a), [1.0, 2.0])
    assert np.allclose(np.minimum(fb, b), [1.0, 2.5])
    print("R9L_INTERNAL_SELF_TEST=PASS", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--action-width", type=float, default=0.5)
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0

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

    r9k_latest = RESULTS / "P3B1_R9K_LATEST.json"
    if not r9k_latest.exists():
        raise RuntimeError("R9L_MISSING_R9K_LATEST")
    upstream = load_json(r9k_latest)
    if upstream.get("status") != "PASS" or upstream.get("classification") != EXPECTED_R9K:
        raise RuntimeError(f"R9L_R9K_GATE_FAIL classification={upstream.get('classification')}")
    if int(upstream.get("metrics", {}).get("upper_stage_sensitive_nodes", -1)) != 0:
        raise RuntimeError("R9L_EXPECTED_R9K_ZERO_UPPER_SENSITIVITY")

    data = r9d.reconstruct_timestamped_geometry(r3, r5, r6, r7, r9c)
    Ts = float(data["Ts"])
    actions = r9hb.refinement_actions(float(args.action_width))

    print("=== P3-B1-R9-L FULL-GFP COLLAPSE MECHANISM AUDIT ===", flush=True)
    print("UPSTREAM_R9K_FULL_GFP_COLLAPSE=PASS", flush=True)
    print("QUESTION=WHAT_ERASES_THE_R9J_TRACE_NONDominance_IN_ROBUST_MAXIMAL_VIABILITY", flush=True)
    print("COUNTERFACTUAL_FORCED_ADOPT_IS_CERTIFICATE=NO", flush=True)
    print("DEPLOYMENT_PROTOCOL_CERTIFIED=NO", flush=True)
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
    last_n = int(np.count_nonzero(last_mask))

    old_groups = r9hg.build_descriptor_range_groups(data, 2.0 * Ts, b0)
    repl_groups = r9hg.build_descriptor_range_groups(data, 0.0, b0)
    old_age_cell, old_ok = r9hg.scalar_cell_index(np.asarray(data["lookup_axes"][5], float), 3.0 * Ts)
    repl_age_cell, repl_ok = r9hg.scalar_cell_index(np.asarray(data["lookup_axes"][5], float), 1.0 * Ts)
    if not old_ok or not repl_ok:
        raise RuntimeError("R9L_ADOPTION_AGE_LOOKUP_FAIL")

    print("R9L_STAGE_START=recompute_actual_upper_full_gfp", flush=True)
    actual = r9k.solve_full_service_fixed_point(
        data=data, cfg=cfg, transitions=transitions,
        fallback_eval=np.asarray(data["eval_fallback"], float),
        old_groups=old_groups, repl_groups=repl_groups,
        old_adopt_age_cell=old_age_cell, repl_adopt_age_cell=repl_age_cell,
        r3=r3, r9hg=r9hg, mode="upper", label="r9l_actual_upper",
    )
    if not actual.converged:
        raise RuntimeError("R9L_ACTUAL_UPPER_GFP_NOT_CONVERGED")

    lookup_shape = tuple(data["lookup_shape"])
    eval_idx = np.asarray(data["eval_idx"], dtype=np.int64)
    h = actual.h_flat.reshape(lookup_shape + (r9k.N_STAGES,))
    cells = r3.cell_corner_max(h)
    cflat = [cells[..., s].reshape(-1) for s in range(r9k.N_STAGES)]
    n = len(eval_idx)
    fallback = np.asarray(data["eval_fallback"], float)

    best_frag = np.full(n, np.inf)
    best_cred = np.full(n, np.inf)
    best_old_last = np.full(n, np.inf)
    best_repl_last = np.full(n, np.inf)
    any_actionwise_pair_diff = np.zeros(n, dtype=bool)
    any_direct_adopt_diff = np.zeros(n, dtype=bool)
    any_after_hold_diff = np.zeros(n, dtype=bool)
    any_hold_erasure = np.zeros(n, dtype=bool)
    any_old_adopt_improves = np.zeros(n, dtype=bool)
    any_repl_adopt_improves = np.zeros(n, dtype=bool)
    all_hold_dominates_both = np.ones(n, dtype=bool)
    max_direct_adopt_diff = 0.0
    max_after_hold_diff = 0.0
    max_actionwise_pair_gap = 0.0

    total_last_action_tests = 0
    direct_adopt_diff_tests = 0
    after_hold_diff_tests = 0
    hold_erasure_tests = 0
    hold_dominates_both_tests = 0
    old_adopt_improve_tests = 0
    repl_adopt_improve_tests = 0
    actionwise_pair_diff_tests = 0

    for tr in transitions:
        hold = np.asarray(tr["hold_index"], dtype=np.intp)
        hs = [cflat[s][hold] for s in range(r9k.N_STAGES)]
        post_hold = hs[r9k.POST]
        adopt_old = r9hg.robust_adopt_future(cells[..., r9k.POST], tr, old_groups, old_age_cell)
        adopt_repl = r9hg.robust_adopt_future(cells[..., r9k.POST], tr, repl_groups, repl_age_cell)
        old_real = np.minimum(adopt_old, post_hold)
        repl_real = np.minimum(adopt_repl, post_hold)

        finite_last = last_mask & np.isfinite(adopt_old) & np.isfinite(adopt_repl) & np.isfinite(post_hold)
        direct = finite_last & (np.abs(adopt_old - adopt_repl) > TOL)
        after = finite_last & (np.abs(old_real - repl_real) > TOL)
        hold_erase = direct & ~after
        old_improve = finite_last & (adopt_old < post_hold - TOL)
        repl_improve = finite_last & (adopt_repl < post_hold - TOL)
        hold_both = finite_last & (post_hold <= adopt_old + TOL) & (post_hold <= adopt_repl + TOL)

        total_last_action_tests += int(np.count_nonzero(finite_last))
        direct_adopt_diff_tests += int(np.count_nonzero(direct))
        after_hold_diff_tests += int(np.count_nonzero(after))
        hold_erasure_tests += int(np.count_nonzero(hold_erase))
        old_adopt_improve_tests += int(np.count_nonzero(old_improve))
        repl_adopt_improve_tests += int(np.count_nonzero(repl_improve))
        hold_dominates_both_tests += int(np.count_nonzero(hold_both))
        any_direct_adopt_diff |= direct
        any_after_hold_diff |= after
        any_hold_erasure |= hold_erase
        any_old_adopt_improves |= old_improve
        any_repl_adopt_improves |= repl_improve
        all_hold_dominates_both &= (~last_mask) | hold_both
        max_direct_adopt_diff = max(max_direct_adopt_diff, finite_sup_abs(adopt_old, adopt_repl, finite_last))
        max_after_hold_diff = max(max_after_hold_diff, finite_sup_abs(old_real, repl_real, finite_last))

        step = np.asarray(tr["step_loss_upper"], float)
        close = np.asarray(tr["closing_end"], float)
        req_frag = np.maximum(step, np.maximum(close + hs[r9k.OLD_LAST], 0.0))
        req_cred = np.maximum(step, np.maximum(close + hs[r9k.REPL_LAST], 0.0))
        req_old_last = np.maximum(step, np.maximum(close + old_real, 0.0))
        req_repl_last = np.maximum(step, np.maximum(close + repl_real, 0.0))
        adiff = common & (np.abs(req_frag - req_cred) > TOL)
        any_actionwise_pair_diff |= adiff
        actionwise_pair_diff_tests += int(np.count_nonzero(adiff))
        max_actionwise_pair_gap = max(max_actionwise_pair_gap, finite_sup_abs(req_frag, req_cred, common))
        best_frag = np.minimum(best_frag, req_frag)
        best_cred = np.minimum(best_cred, req_cred)
        best_old_last = np.minimum(best_old_last, req_old_last)
        best_repl_last = np.minimum(best_repl_last, req_repl_last)

    coop_diff = common & (np.abs(best_frag - best_cred) > TOL)
    capped_frag = np.minimum(fallback, best_frag)
    capped_cred = np.minimum(fallback, best_cred)
    capped_diff = common & (np.abs(capped_frag - capped_cred) > TOL)
    action_erase = common & any_actionwise_pair_diff & ~coop_diff
    fallback_erase = common & coop_diff & ~capped_diff
    fallback_binds_both = common & (fallback <= best_frag + TOL) & (fallback <= best_cred + TOL)

    ue = np.asarray(actual.h_flat[eval_idx, :], float)
    actual_pair_gap = finite_sup_abs(ue[:, r9k.FRAG], ue[:, r9k.CRED], common)
    last_stage_gap = finite_sup_abs(ue[:, r9k.OLD_LAST], ue[:, r9k.REPL_LAST], last_mask)
    one_step_capped_gap = finite_sup_abs(capped_frag, capped_cred, common)
    if actual_pair_gap > 1e-8 or one_step_capped_gap > 1e-8:
        raise RuntimeError(
            f"R9L_UPSTREAM_COLLAPSE_REPRO_FAIL actual_gap={actual_pair_gap} one_step_gap={one_step_capped_gap}"
        )

    print(
        "R9L_ELIGIBLE_ADOPTION_ATTRIBUTION "
        f"last_nodes={last_n} action_tests={total_last_action_tests} "
        f"direct_adopt_diff_tests={direct_adopt_diff_tests} "
        f"after_optional_hold_diff_tests={after_hold_diff_tests} "
        f"hold_erasure_tests={hold_erasure_tests} "
        f"hold_dominates_both_tests={hold_dominates_both_tests} "
        f"max_direct_adopt_diff_m={max_direct_adopt_diff:.12g} "
        f"max_after_hold_diff_m={max_after_hold_diff:.12g}",
        flush=True,
    )
    print(
        "R9L_PAIR_OPERATOR_ATTRIBUTION "
        f"common_nodes={common_n} actionwise_diff_tests={actionwise_pair_diff_tests} "
        f"actionwise_diff_nodes={int(np.count_nonzero(common & any_actionwise_pair_diff))} "
        f"cooperative_min_diff_nodes={int(np.count_nonzero(coop_diff))} "
        f"action_min_erasure_nodes={int(np.count_nonzero(action_erase))} "
        f"fallback_cap_erasure_nodes={int(np.count_nonzero(fallback_erase))} "
        f"fallback_binds_both_nodes={int(np.count_nonzero(fallback_binds_both))} "
        f"max_actionwise_gap_m={max_actionwise_pair_gap:.12g} "
        f"max_cooperative_min_gap_m={finite_sup_abs(best_frag,best_cred,common):.12g}",
        flush=True,
    )
    print(
        "R9L_STAGE_IDENTITY "
        f"max_old_vs_replacement_last_gap_m={last_stage_gap:.12g} "
        f"max_fragment_vs_credential_gap_m={actual_pair_gap:.12g}",
        flush=True,
    )

    print("R9L_STAGE_START=forced_adopt_counterfactual_full_gfp", flush=True)
    forced = solve_forced_adopt_upper(
        data=data, cfg=cfg, transitions=transitions,
        old_groups=old_groups, repl_groups=repl_groups,
        old_age_cell=old_age_cell, repl_age_cell=repl_age_cell,
        r3=r3, r9k=r9k, r9hg=r9hg,
    )
    if not forced.converged:
        raise RuntimeError("R9L_FORCED_ADOPT_COUNTERFACTUAL_NOT_CONVERGED")
    fe = np.asarray(forced.h_flat[eval_idx, :], float)
    forced_signed = fe[:, r9k.CRED] - fe[:, r9k.FRAG]
    forced_sensitive = common & (np.abs(forced_signed) > TOL)
    forced_sensitive_n = int(np.count_nonzero(forced_sensitive))
    forced_max_gap = finite_max(np.abs(forced_signed[common])) if common_n else 0.0
    forced_cred_more = int(np.count_nonzero(common & (forced_signed > TOL)))
    forced_frag_more = int(np.count_nonzero(common & (forced_signed < -TOL)))
    print(
        "R9L_FORCED_ADOPT_COUNTERFACTUAL "
        f"converged=YES iterations={forced.iterations} "
        f"sensitive_nodes={forced_sensitive_n}/{common_n} "
        f"credential_more={forced_cred_more} fragment_more={forced_frag_more} "
        f"max_gap_m={forced_max_gap:.12g}",
        flush=True,
    )

    direct_nodes = int(np.count_nonzero(last_mask & any_direct_adopt_diff))
    after_nodes = int(np.count_nonzero(last_mask & any_after_hold_diff))
    hold_erasure_nodes = int(np.count_nonzero(last_mask & any_hold_erasure))
    hold_all_nodes = int(np.count_nonzero(last_mask & all_hold_dominates_both))
    action_diff_nodes = int(np.count_nonzero(common & any_actionwise_pair_diff))
    coop_diff_nodes = int(np.count_nonzero(coop_diff))
    fallback_erase_nodes = int(np.count_nonzero(fallback_erase))
    action_erase_nodes = int(np.count_nonzero(action_erase))

    if direct_nodes > 0 and after_nodes == 0 and forced_sensitive_n > 0:
        classification = "OPTIONAL_HOLD_DOMINANCE_ERASES_MATERIAL_CANDIDATE_FATE_DIFFERENCE"
        next_action = "R9M_SCREEN_FOR_SERVICE_STATES_WITH_DIFFERENT_WORST_CASE_NONADOPTION_CONTINUATIONS"
    elif direct_nodes == 0:
        classification = "ROBUST_CHI_ENVELOPE_ERASES_OLD_VS_REPLACEMENT_CANDIDATE_DIFFERENCE_BEFORE_ADOPTION"
        next_action = "R9M_BUILD_CORRELATION_PRESERVING_CHI_ENCLOSURE_BEFORE_ANY_NEW_PAIR_SEARCH"
    elif after_nodes > 0 and action_erase_nodes > 0 and coop_diff_nodes == 0:
        classification = "CONTROL_ACTION_MINIMIZATION_ERASES_SERVICE_CONTINUATION_DIFFERENCE"
        next_action = "R9M_SCREEN_WORST_CASE_CONTROL_VALUE_NONDOMINATED_PAIRS_BEFORE_GFP"
    elif fallback_erase_nodes > 0:
        classification = "FALLBACK_CAP_ERASES_SERVICE_CONTINUATION_DIFFERENCE"
        next_action = "R9M_PROVE_FALLBACK_DOMINANCE_OR_IDENTIFY_STATES_WHERE_COOPERATIVE_SERVICE_IS_NECESSARY"
    elif forced_sensitive_n == 0:
        classification = "CANDIDATE_FATE_DIFFERENCE_REMAINS_NONBINDING_EVEN_UNDER_FORCED_ADOPTION_COUNTERFACTUAL"
        next_action = "R9M_STOP_TRACE_ONLY_PAIR_SEARCH_AND_TEST_VALUE_NONDOMINANCE_NECESSITY"
    else:
        classification = "MULTIPLE_COLLAPSE_MECHANISMS_REQUIRE_VALUE_LEVEL_SCREENING"
        next_action = "R9M_BUILD_WORST_CASE_CONTROL_VALUE_NONDOMINANCE_SCREEN"

    rows = []
    ids = np.flatnonzero(common)
    if len(ids):
        score = np.maximum.reduce([
            np.abs(best_frag - best_cred),
            np.abs(forced_signed),
            np.abs(ue[:, r9k.OLD_LAST] - ue[:, r9k.REPL_LAST]),
        ])
        order = ids[np.argsort(score[ids])[::-1][:64]]
        vf, vp, af, ba, bu, ag = [np.asarray(x, float) for x in data["eval_flat"]]
        for i in order:
            rows.append({
                "node_index": int(i), "v_f": float(vf[i]), "v_p": float(vp[i]),
                "a_f": float(af[i]), "bar_a": float(ba[i]), "bar_u": float(bu[i]),
                "age_s": float(ag[i]), "actual_fragment_m": float(ue[i, r9k.FRAG]),
                "actual_credential_m": float(ue[i, r9k.CRED]),
                "old_last_m": float(ue[i, r9k.OLD_LAST]),
                "replacement_last_m": float(ue[i, r9k.REPL_LAST]),
                "coop_fragment_m": float(best_frag[i]), "coop_credential_m": float(best_cred[i]),
                "fallback_m": float(fallback[i]),
                "forced_fragment_m": float(fe[i, r9k.FRAG]),
                "forced_credential_m": float(fe[i, r9k.CRED]),
                "forced_signed_credential_minus_fragment_m": float(forced_signed[i]),
            })

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result = RESULTS / f"P3B1_R9L_RESULT_{stamp}.json"
    latest = RESULTS / "P3B1_R9L_LATEST.json"
    csvp = RESULTS / f"P3B1_R9L_ATTRIBUTION_WITNESSES_{stamp}.csv"
    manifest = RESULTS / f"P3B1_R9L_MANIFEST_{stamp}.sha256"
    write_csv(csvp, rows)
    out = {
        "schema": "SCV_P3B1_R9L_COLLAPSE_MECHANISM_AUDIT_V1",
        "status": "PASS",
        "timestamp_utc": stamp,
        "classification": classification,
        "next_action": next_action,
        "continuous_state_separation_certified": False,
        "deployment_protocol_certified": False,
        "counterfactual_forced_adopt_is_certificate": False,
        "metrics": {
            "common_nodes": common_n,
            "eligible_last_nodes": last_n,
            "direct_adopt_difference_nodes": direct_nodes,
            "after_optional_hold_difference_nodes": after_nodes,
            "hold_erasure_nodes": hold_erasure_nodes,
            "hold_dominates_both_for_all_actions_nodes": hold_all_nodes,
            "max_direct_adopt_difference_m": max_direct_adopt_diff,
            "max_after_optional_hold_difference_m": max_after_hold_diff,
            "actionwise_pair_difference_nodes": action_diff_nodes,
            "action_min_erasure_nodes": action_erase_nodes,
            "cooperative_min_difference_nodes": coop_diff_nodes,
            "fallback_cap_erasure_nodes": fallback_erase_nodes,
            "fallback_binds_both_nodes": int(np.count_nonzero(fallback_binds_both)),
            "max_actionwise_pair_gap_m": max_actionwise_pair_gap,
            "max_cooperative_min_gap_m": finite_sup_abs(best_frag, best_cred, common),
            "max_old_vs_replacement_last_gap_m": last_stage_gap,
            "max_actual_fragment_vs_credential_gap_m": actual_pair_gap,
            "forced_adopt_sensitive_nodes": forced_sensitive_n,
            "forced_adopt_max_gap_m": forced_max_gap,
            "forced_adopt_credential_more_nodes": forced_cred_more,
            "forced_adopt_fragment_more_nodes": forced_frag_more,
            "forced_adopt_iterations": forced.iterations,
        },
        "gates": {
            "upstream_r9k_full_gfp_collapse": True,
            "actual_upper_gfp_reproduced": True,
            "actual_pair_gap_zero": actual_pair_gap <= 1e-8,
            "forced_adopt_counterfactual_converged": forced.converged,
            "forced_adopt_is_protocol_semantics": False,
            "deployment_protocol_certified": False,
            "continuous_action_interval_gfp_solved": False,
            "continuous_state_separation_certified": False,
        },
        "artifacts": {"attribution_witness_csv": str(csvp)},
    }
    text = json.dumps(out, indent=2, sort_keys=True)
    atomic_write(result, text)
    atomic_write(latest, text)
    mfiles = [Path(__file__), result, csvp, r9k_latest]
    atomic_write(manifest, "".join(f"{sha256_file(p)}  {p}\n" for p in mfiles if p.exists()))

    print("=== R9-L DECISION ===", flush=True)
    print(f"R9L_CLASSIFICATION={classification}", flush=True)
    print("R9L_COUNTERFACTUAL_FORCED_ADOPT_IS_CERTIFICATE=NO", flush=True)
    print("DEPLOYMENT_PROTOCOL_CERTIFIED=NO", flush=True)
    print("CONTINUOUS_STATE_SEPARATION_CERTIFIED=NO", flush=True)
    print("R9L_EXECUTION=PASS", flush=True)
    print(f"R9L_NEXT_ACTION={next_action}", flush=True)
    print(f"RESULT_JSON={result}", flush=True)
    print(f"ATTRIBUTION_WITNESS_CSV={csvp}", flush=True)
    print(f"MANIFEST={manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
