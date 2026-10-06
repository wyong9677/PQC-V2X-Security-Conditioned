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
CONFIG = ROOT / "01_config"
RESULTS = ROOT / "04_results"
RESULTS.mkdir(parents=True, exist_ok=True)

# R9-J declared diagnostic-v2 states.
POST = 0
ENTRY = 1
FRAG = 2
CRED = 3
OLD_LAST = 4
REPL_LAST = 5
STAGE_NAMES = (
    "POST_TERMINATION_ENTRY",
    "SAME_CANDIDATE_ENTRY",
    "FRAGMENT_CARRY",
    "CREDENTIAL_DECISION",
    "VERIFY_OLD_LAST",
    "VERIFY_REPLACEMENT_LAST",
)
N_STAGES = len(STAGE_NAMES)
TOL = 1e-9
EXPECTED_R9J = (
    "DECLARED_DIAGNOSTIC_NONDominated_MATCHED_CHI_PAIR_FOUND_"
    "READY_FOR_FULL_AUGMENTED_GFP_NOT_DEPLOYMENT_CERTIFIED"
)


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def finite_sup_abs(a: np.ndarray, b: np.ndarray) -> float:
    d = np.abs(np.asarray(a, float) - np.asarray(b, float))
    d = d[np.isfinite(d)]
    return float(np.max(d)) if d.size else 0.0


def service_future_algebra(
    *, post_hold: np.ndarray, entry_hold: np.ndarray,
    frag_hold: np.ndarray, cred_hold: np.ndarray,
    old_last_hold: np.ndarray, repl_last_hold: np.ndarray,
    adopt_old: np.ndarray, adopt_repl: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Closed R9-J service graph future terms, before physical closing loss.

    Service uncertainty is adversarial only where the declared contract has
    multiple outgoing branches (SAME_CANDIDATE_ENTRY). Eligible outputs use the
    causal adoption rule: controller may adopt or hold after observing output.
    Rejected output at CREDENTIAL_DECISION is not adoptable and moves to the
    replacement-candidate last-verification state.
    """
    f_post = entry_hold
    f_entry = np.maximum(frag_hold, cred_hold)
    f_frag = old_last_hold
    f_cred = repl_last_hold
    f_old_last = np.minimum(adopt_old, post_hold)
    f_repl_last = np.minimum(adopt_repl, post_hold)
    return f_post, f_entry, f_frag, f_cred, f_old_last, f_repl_last


def stage_masks(age: np.ndarray, Ts: float) -> list[np.ndarray]:
    """Reachability lower-age masks for the declared diagnostic cycle.

    POST may be entered after adopting the fresh replacement output (age Ts).
    POST->ENTRY consumes one no-output interval; ENTRY->pair consumes another;
    pair->last consumes one more.  These are semantic reachability guards only,
    not a continuous-domain certificate.
    """
    a = np.asarray(age, float)
    eps = 1e-10
    return [
        a >= 1.0 * Ts - eps,  # POST_TERMINATION_ENTRY
        a >= 2.0 * Ts - eps,  # SAME_CANDIDATE_ENTRY
        a >= 3.0 * Ts - eps,  # FRAGMENT_CARRY
        a >= 3.0 * Ts - eps,  # CREDENTIAL_DECISION
        a >= 4.0 * Ts - eps,  # VERIFY_OLD_LAST
        a >= 4.0 * Ts - eps,  # VERIFY_REPLACEMENT_LAST
    ]


@dataclass
class SolveResult:
    h_flat: np.ndarray
    converged: bool
    iterations: int
    final_delta: float
    monotonicity_violations: int


def solve_full_service_fixed_point(
    *, data: dict, cfg: dict, transitions: list[dict], fallback_eval: np.ndarray,
    old_groups, repl_groups, old_adopt_age_cell: int, repl_adopt_age_cell: int,
    r3, r9hg, mode: str, label: str,
) -> SolveResult:
    lookup_shape = tuple(data["lookup_shape"])
    lookup_nodes = int(np.prod(lookup_shape))
    eval_idx = np.asarray(data["eval_idx"], dtype=np.int64)
    n = len(eval_idx)
    age = np.asarray(data["eval_flat"][5], float)
    Ts = float(data["Ts"])
    masks = stage_masks(age, Ts)
    tol = float(cfg["fixed_point"]["tolerance_m"])
    max_iter = int(cfg["fixed_point"]["max_iterations"])
    lips = 0.5 * (
        float(data["p2b"]["state_domain"]["v_f_max"])
        + float(data["p2b"]["state_domain"]["v_p_max"])
    ) * (Ts / (int(cfg["one_step"]["trajectory_points"]) - 1))

    if mode == "upper":
        base = np.asarray(data["lookup_fallback"], float)
        h = np.repeat(base[:, None], N_STAGES, axis=1)
    elif mode == "lower":
        h = np.zeros((lookup_nodes, N_STAGES), dtype=float)
    else:
        raise ValueError(mode)

    violations = 0
    delta = math.inf
    for it in range(1, max_iter + 1):
        shaped = h.reshape(lookup_shape + (N_STAGES,))
        cells = r3.cell_corner_max(shaped) if mode == "upper" else r9hg.cell_corner_min(shaped)
        cflat = [cells[..., s].reshape(-1) for s in range(N_STAGES)]
        old = h[eval_idx, :].copy()
        best = np.full((n, N_STAGES), np.inf, dtype=float)

        for tr in transitions:
            hold = np.asarray(tr["hold_index"], dtype=np.intp)
            hold_stage = [cflat[s][hold] for s in range(N_STAGES)]

            # Eligible OLD_LAST output carries the original candidate generated
            # three intervals earlier at completion.  Replacement output is one
            # interval old at completion.  The continuation state is POST.
            adopt_old = r9hg.robust_adopt_future(
                cells[..., POST], tr, old_groups, old_adopt_age_cell
            )
            adopt_repl = r9hg.robust_adopt_future(
                cells[..., POST], tr, repl_groups, repl_adopt_age_cell
            )

            futures = service_future_algebra(
                post_hold=hold_stage[POST],
                entry_hold=hold_stage[ENTRY],
                frag_hold=hold_stage[FRAG],
                cred_hold=hold_stage[CRED],
                old_last_hold=hold_stage[OLD_LAST],
                repl_last_hold=hold_stage[REPL_LAST],
                adopt_old=adopt_old,
                adopt_repl=adopt_repl,
            )

            step = np.asarray(tr["step_loss_upper"], float)
            if mode == "lower":
                # build_physical_transitions stores sampled max + Lipschitz
                # correction; removing the correction leaves a sound lower
                # element for the continuous in-step maximum.
                step = np.maximum(step - lips, 0.0)
            close = np.asarray(tr["closing_end"], float)
            for s, fut in enumerate(futures):
                req = np.maximum(step, close + np.asarray(fut, float))
                req = np.maximum(req, 0.0)
                best[:, s] = np.minimum(best[:, s], req)

        cand = np.minimum(np.asarray(fallback_eval, float)[:, None], best)
        new = old.copy()
        for s, m in enumerate(masks):
            if mode == "upper":
                new[m, s] = np.minimum(old[m, s], cand[m, s])
            else:
                bad = cand[m, s] < old[m, s] - max(1e-9, 10.0 * tol)
                violations += int(np.count_nonzero(bad))
                if np.any(bad):
                    worst = float(np.min(cand[m, s][bad] - old[m, s][bad]))
                    raise RuntimeError(
                        f"R9K_LOWER_MONOTONICITY_FAIL stage={STAGE_NAMES[s]} worst={worst}"
                    )
                new[m, s] = np.maximum(old[m, s], cand[m, s])

        delta = finite_sup_abs(new, old)
        h[eval_idx, :] = new
        if it == 1 or it % 5 == 0 or delta <= tol:
            print(
                f"R9K_GFP_ITER label={label} mode={mode} iteration={it} "
                f"delta_m={delta:.12g} monotonicity_violations={violations}",
                flush=True,
            )
        if delta <= tol:
            return SolveResult(h, True, it, delta, violations)
    return SolveResult(h, False, max_iter, delta, violations)


def top_witness_rows(
    *, data: dict, ue: np.ndarray, le: np.ndarray, common_mask: np.ndarray,
    max_rows: int = 64,
) -> list[dict]:
    signed = ue[:, CRED] - ue[:, FRAG]
    margin_cred_worse = le[:, CRED] - ue[:, FRAG]
    margin_frag_worse = le[:, FRAG] - ue[:, CRED]
    score = np.maximum.reduce([
        np.abs(signed),
        np.maximum(margin_cred_worse, 0.0),
        np.maximum(margin_frag_worse, 0.0),
    ])
    ids = np.flatnonzero(common_mask)
    if len(ids) == 0:
        return []
    order = ids[np.argsort(score[ids])[::-1][:max_rows]]
    vf, vp, af, ba, bu, age = [np.asarray(x, float) for x in data["eval_flat"]]
    rows: list[dict] = []
    for i in order:
        rows.append({
            "node_index": int(i),
            "v_f": float(vf[i]),
            "v_p": float(vp[i]),
            "a_f": float(af[i]),
            "bar_a": float(ba[i]),
            "bar_u": float(bu[i]),
            "adopted_age_s": float(age[i]),
            "upper_fragment_carry_m": float(ue[i, FRAG]),
            "upper_credential_decision_m": float(ue[i, CRED]),
            "upper_signed_credential_minus_fragment_m": float(signed[i]),
            "lower_fragment_carry_m": float(le[i, FRAG]),
            "lower_credential_decision_m": float(le[i, CRED]),
            "paired_credential_worse_margin_m": float(margin_cred_worse[i]),
            "paired_fragment_worse_margin_m": float(margin_frag_worse[i]),
        })
    return rows


def self_test() -> None:
    x = np.asarray([1.0, 2.0])
    vals = service_future_algebra(
        post_hold=np.asarray([10.0, 10.0]),
        entry_hold=np.asarray([3.0, 4.0]),
        frag_hold=np.asarray([5.0, 2.0]),
        cred_hold=np.asarray([1.0, 6.0]),
        old_last_hold=np.asarray([7.0, 7.0]),
        repl_last_hold=np.asarray([8.0, 8.0]),
        adopt_old=np.asarray([5.0, 5.0]),
        adopt_repl=np.asarray([1.0, 1.0]),
    )
    assert np.allclose(vals[POST], [3.0, 4.0])
    assert np.allclose(vals[ENTRY], [5.0, 6.0])
    assert np.allclose(vals[FRAG], [7.0, 7.0])
    assert np.allclose(vals[CRED], [8.0, 8.0])
    assert np.allclose(vals[OLD_LAST], [5.0, 5.0])
    assert np.allclose(vals[REPL_LAST], [1.0, 1.0])
    m = stage_masks(np.asarray([0.1, 0.2, 0.3, 0.4]), 0.1)
    assert bool(m[POST][0]) and not bool(m[ENTRY][0])
    assert bool(m[FRAG][2]) and bool(m[CRED][2])
    assert bool(m[OLD_LAST][3]) and bool(m[REPL_LAST][3])
    del x
    print("R9K_INTERNAL_SELF_TEST=PASS", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--action-width", type=float, default=0.5)
    ap.add_argument("--p2c-lower-resolution", type=int, default=257)
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.action_width <= 0.0 or args.action_width > 1.0:
        raise ValueError("R9K_BAD_ACTION_WIDTH")

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

    r9j_latest = RESULTS / "P3B1_R9J_LATEST.json"
    if not r9j_latest.exists():
        raise RuntimeError("R9K_MISSING_R9J_LATEST")
    r9j = load_json(r9j_latest)
    if r9j.get("status") != "PASS" or r9j.get("classification") != EXPECTED_R9J:
        raise RuntimeError(
            f"R9K_R9J_GATE_FAIL classification={r9j.get('classification')}"
        )
    pair_j = r9j.get("pair_screen", {})
    gates_j = r9j.get("gates", {})
    if not bool(pair_j.get("target_pair_pass", False)):
        raise RuntimeError("R9K_R9J_TARGET_PAIR_NOT_PASS")
    if bool(gates_j.get("deployment_transition_relation_certified", False)):
        raise RuntimeError("R9K_UNEXPECTED_DEPLOYMENT_CERTIFICATION")

    contract_path = CONFIG / "p3b1_r9j_declared_diagnostic_service_contract_v2.json"
    if not contract_path.exists():
        raise RuntimeError("R9K_MISSING_R9J_CONTRACT_CONFIG")
    contract = load_json(contract_path)
    if contract.get("comparison_pair") != ["FRAGMENT_CARRY", "CREDENTIAL_DECISION"]:
        raise RuntimeError("R9K_COMPARISON_PAIR_CHANGED")
    if contract.get("deployment_certified") is not False:
        raise RuntimeError("R9K_CONTRACT_DEPLOYMENT_FLAG_BAD")

    data = r9d.reconstruct_timestamped_geometry(r3, r5, r6, r7, r9c)
    Ts = float(data["Ts"])
    profile = "diagnostic_fast"
    R = r7.service_horizon(profile, data["p1"]["diagnostic_service_profiles"][profile])
    if R != 2:
        raise RuntimeError("R9K_EXPECTED_DIAGNOSTIC_FAST_R2")

    actions = r9hb.refinement_actions(float(args.action_width))
    print("=== P3-B1-R9-K FULL AUGMENTED DECLARED-PAIR GFP ===", flush=True)
    print("UPSTREAM_R9J_NONDominated_PAIR=PASS", flush=True)
    print("DECLARED_DIAGNOSTIC_CONTRACT_V2=PASS", flush=True)
    print("DEPLOYMENT_PROTOCOL_CERTIFIED=NO", flush=True)
    print("COMPARISON_PAIR=FRAGMENT_CARRY|CREDENTIAL_DECISION", flush=True)
    print("MATCHED_CURRENT_CHI=YES", flush=True)
    print("SAME_LIVENESS_BOUND_STEPS=2", flush=True)
    print("COMMON_PREDECESSOR=SAME_CANDIDATE_ENTRY", flush=True)
    print(
        "SERVICE_GRAPH=POST_TERMINATION_ENTRY,SAME_CANDIDATE_ENTRY,FRAGMENT_CARRY,"
        "CREDENTIAL_DECISION,VERIFY_OLD_LAST,VERIFY_REPLACEMENT_LAST",
        flush=True,
    )
    print("RESTART_IS_INTERNAL_TO_FIXED_POINT=YES", flush=True)
    print("OLD_CANDIDATE_GENERATION_TIME_PRESERVED=YES", flush=True)
    print("REPLACEMENT_GENERATION_TIME_RESET_ON_REJECTION=YES", flush=True)
    print("OLD_ELIGIBLE_AGE_AT_COMPLETION=3Ts", flush=True)
    print("REPLACEMENT_ELIGIBLE_AGE_AT_COMPLETION=Ts", flush=True)
    print("EXPLICIT_CHI_CORRELATIONS_CERTIFIED=NO", flush=True)
    print(f"POINT_ACTION_WIDTH={float(args.action_width):.12g}", flush=True)
    print(f"POINT_ACTION_COUNT={len(actions)}", flush=True)
    print("CONTINUOUS_STATE_SEPARATION_CERTIFIED=NO", flush=True)

    cfg, transitions = r9hg.build_physical_transitions(
        data, actions, r3, b0, r9e, chunk_size=int(args.chunk_size)
    )

    print("R9K_STAGE_START=build_old_and_replacement_chi_rectangles", flush=True)
    old_groups = r9hg.build_descriptor_range_groups(data, 2.0 * Ts, b0)
    repl_groups = r9hg.build_descriptor_range_groups(data, 0.0, b0)
    old_age_cell, old_ok = r9hg.scalar_cell_index(
        np.asarray(data["lookup_axes"][5], float), 3.0 * Ts
    )
    repl_age_cell, repl_ok = r9hg.scalar_cell_index(
        np.asarray(data["lookup_axes"][5], float), 1.0 * Ts
    )
    if not old_ok or not repl_ok:
        raise RuntimeError("R9K_ADOPTION_AGE_LOOKUP_FAIL")
    age = np.asarray(data["eval_flat"][5], float)
    if np.any((age >= 4.0 * Ts - 1e-10) & ~old_groups.valid_mask):
        raise RuntimeError("R9K_OLD_CHI_ENVELOPE_COVERAGE_FAIL")
    if np.any((age >= 4.0 * Ts - 1e-10) & ~repl_groups.valid_mask):
        raise RuntimeError("R9K_REPLACEMENT_CHI_ENVELOPE_COVERAGE_FAIL")
    print(
        f"R9K_CHI_RECTANGLES old_unique={old_groups.unique_rectangles} "
        f"old_max_cells={old_groups.max_rectangle_cells} "
        f"replacement_unique={repl_groups.unique_rectangles} "
        f"replacement_max_cells={repl_groups.max_rectangle_cells}",
        flush=True,
    )

    print("R9K_STAGE_START=descending_upper_full_declared_service_gfp", flush=True)
    upper = solve_full_service_fixed_point(
        data=data, cfg=cfg, transitions=transitions,
        fallback_eval=np.asarray(data["eval_fallback"], float),
        old_groups=old_groups, repl_groups=repl_groups,
        old_adopt_age_cell=old_age_cell, repl_adopt_age_cell=repl_age_cell,
        r3=r3, r9hg=r9hg, mode="upper", label="r9k_upper_full_declared",
    )
    if not upper.converged:
        raise RuntimeError("R9K_UPPER_GFP_NOT_CONVERGED")

    print("R9K_STAGE_START=global_p2c_lower_bracket", flush=True)
    fallback_lower, p2c_metrics = r9he.compute_eval_fallback_lower(
        data, b0, sw,
        resolution=int(args.p2c_lower_resolution),
        chunk_size=int(args.chunk_size),
    )
    print("R9K_STAGE_START=ascending_lower_full_declared_service_gfp", flush=True)
    lower = solve_full_service_fixed_point(
        data=data, cfg=cfg, transitions=transitions,
        fallback_eval=np.asarray(fallback_lower, float),
        old_groups=old_groups, repl_groups=repl_groups,
        old_adopt_age_cell=old_age_cell, repl_adopt_age_cell=repl_age_cell,
        r3=r3, r9hg=r9hg, mode="lower", label="r9k_lower_full_declared",
    )
    if not lower.converged:
        raise RuntimeError("R9K_LOWER_GFP_NOT_CONVERGED")

    eval_idx = np.asarray(data["eval_idx"], dtype=np.int64)
    ue = np.asarray(upper.h_flat[eval_idx, :], float)
    le = np.asarray(lower.h_flat[eval_idx, :], float)
    violation = le > ue + 1e-8
    if np.any(violation):
        raise RuntimeError(
            f"R9K_FULL_GRAPH_SANDWICH_FAIL count={int(np.count_nonzero(violation))}"
        )
    sandwich_slack = float(np.max(le - ue))

    # Both comparison states are reached two no-output intervals after POST and
    # one interval after SAME_CANDIDATE_ENTRY; common-history phase requires the
    # adopted information to be at least 3Ts old.
    common = age >= 3.0 * Ts - 1e-10
    signed = ue[:, CRED] - ue[:, FRAG]
    sensitive = common & (np.abs(signed) > TOL)
    cred_more = common & (signed > TOL)
    frag_more = common & (signed < -TOL)

    margin_cred_worse = le[:, CRED] - ue[:, FRAG]
    margin_frag_worse = le[:, FRAG] - ue[:, CRED]
    paired_cred = common & (margin_cred_worse > TOL)
    paired_frag = common & (margin_frag_worse > TOL)
    paired_any = paired_cred | paired_frag

    common_n = int(np.count_nonzero(common))
    sensitive_n = int(np.count_nonzero(sensitive))
    cred_more_n = int(np.count_nonzero(cred_more))
    frag_more_n = int(np.count_nonzero(frag_more))
    paired_cred_n = int(np.count_nonzero(paired_cred))
    paired_frag_n = int(np.count_nonzero(paired_frag))
    paired_any_n = int(np.count_nonzero(paired_any))
    max_upper_gap = float(np.max(np.abs(signed[common]))) if common_n else 0.0
    max_cred_margin = float(np.max(margin_cred_worse[common])) if common_n else 0.0
    max_frag_margin = float(np.max(margin_frag_worse[common])) if common_n else 0.0
    positive_margins = np.concatenate([
        margin_cred_worse[paired_cred], margin_frag_worse[paired_frag]
    ])
    min_positive_margin = float(np.min(positive_margins)) if positive_margins.size else 0.0

    if paired_any_n > 0:
        classification = (
            "FULL_AUGMENTED_R9J_DIAGNOSTIC_GFP_PRESERVES_NONDominated_PAIR_"
            "AND_PAIRED_POINT_MARGIN_FOUND"
        )
        next_action = (
            "R9L_INTERVALIZE_R9K_FULL_GRAPH_AND_BUILD_INDEPENDENT_P2C_LOWER_CHECKER"
        )
    elif sensitive_n > 0:
        classification = (
            "FULL_AUGMENTED_R9J_DIAGNOSTIC_GFP_PRESERVES_STAGE_SEPARATION_"
            "BUT_PAIRED_POINT_MARGIN_OPEN"
        )
        next_action = (
            "R9L_STRENGTHEN_R9K_FULL_GRAPH_LOWER_ENCLOSURE_BEFORE_CONTINUOUS_PROMOTION"
        )
    else:
        classification = "FULL_AUGMENTED_R9J_DIAGNOSTIC_GFP_COLLAPSES_NEW_PAIR"
        next_action = (
            "R9L_REASSESS_DECLARED_DIAGNOSTIC_CONTRACT_OR_STOP_POSITIVE_CONTINUOUS_PROMOTION"
        )

    rows = top_witness_rows(data=data, ue=ue, le=le, common_mask=common, max_rows=64)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result = RESULTS / f"P3B1_R9K_RESULT_{stamp}.json"
    latest = RESULTS / "P3B1_R9K_LATEST.json"
    csvp = RESULTS / f"P3B1_R9K_TOP_WITNESSES_{stamp}.csv"
    manifest = RESULTS / f"P3B1_R9K_MANIFEST_{stamp}.sha256"
    write_csv(csvp, rows)

    out = {
        "schema": "SCV_P3B1_R9K_FULL_AUGMENTED_DECLARED_PAIR_GFP_V1",
        "status": "PASS",
        "timestamp_utc": stamp,
        "classification": classification,
        "next_action": next_action,
        "deployment_protocol_certified": False,
        "continuous_state_separation_certified": False,
        "full_augmented_declared_diagnostic_point_gfp_solved": True,
        "comparison_pair": ["FRAGMENT_CARRY", "CREDENTIAL_DECISION"],
        "metrics": {
            "point_action_width": float(args.action_width),
            "point_action_count": len(actions),
            "upper_iterations": int(upper.iterations),
            "upper_final_delta_m": float(upper.final_delta),
            "lower_iterations": int(lower.iterations),
            "lower_final_delta_m": float(lower.final_delta),
            "full_graph_sandwich_max_lower_minus_upper_m": sandwich_slack,
            "common_history_semantic_nodes": common_n,
            "upper_stage_sensitive_nodes": sensitive_n,
            "credential_more_demanding_upper_nodes": cred_more_n,
            "fragment_more_demanding_upper_nodes": frag_more_n,
            "max_upper_pair_gap_m": max_upper_gap,
            "paired_credential_worse_positive_nodes": paired_cred_n,
            "paired_fragment_worse_positive_nodes": paired_frag_n,
            "paired_positive_nodes": paired_any_n,
            "max_paired_credential_worse_margin_m": max_cred_margin,
            "max_paired_fragment_worse_margin_m": max_frag_margin,
            "min_positive_paired_margin_m": min_positive_margin,
            "old_candidate_pending_age_at_last_s": 2.0 * Ts,
            "old_candidate_eligible_age_s": 3.0 * Ts,
            "replacement_pending_age_at_last_s": 0.0,
            "replacement_eligible_age_s": 1.0 * Ts,
            "old_chi_unique_rectangles": old_groups.unique_rectangles,
            "old_chi_max_cells": old_groups.max_rectangle_cells,
            "replacement_chi_unique_rectangles": repl_groups.unique_rectangles,
            "replacement_chi_max_cells": repl_groups.max_rectangle_cells,
            "p2c_lower_resolution": int(args.p2c_lower_resolution),
            **{f"p2c_{k}": v for k, v in p2c_metrics.items()},
        },
        "gates": {
            "upstream_r9j_nondominated_pair": True,
            "declared_diagnostic_contract_v2": True,
            "deployment_protocol_certified": False,
            "matched_current_chi": True,
            "same_liveness_two_steps": True,
            "common_predecessor": True,
            "restart_internal_to_fixed_point": True,
            "old_generation_time_preserved": True,
            "replacement_generation_time_reset_on_rejection": True,
            "upper_full_graph_gfp_converged": bool(upper.converged),
            "lower_full_graph_gfp_converged": bool(lower.converged),
            "point_model_sandwich_pass": True,
            "explicit_chi_correlations_certified": False,
            "continuous_action_interval_gfp_solved": False,
            "independent_p2c_lower_checker_pass": False,
            "continuous_state_cell_interval_gfp": False,
        },
        "artifacts": {"top_witness_csv": str(csvp)},
    }
    text = json.dumps(out, indent=2, sort_keys=True)
    atomic_write(result, text)
    atomic_write(latest, text)
    mfiles = [
        Path(__file__), result, csvp, contract_path,
        RESULTS / "P3B1_R9J_LATEST.json",
        RESULTS / "P3B1_R9I_LATEST.json",
    ]
    atomic_write(
        manifest,
        "".join(f"{sha256_file(p)}  {p}\n" for p in mfiles if p.exists()),
    )

    print("=== R9-K DECISION ===", flush=True)
    print(
        f"R9K_FULL_GFP upper_converged=YES upper_iterations={upper.iterations} "
        f"lower_converged=YES lower_iterations={lower.iterations}",
        flush=True,
    )
    print(
        f"R9K_PAIR_DOMAIN common_nodes={common_n} upper_sensitive={sensitive_n} "
        f"credential_more={cred_more_n} fragment_more={frag_more_n} "
        f"max_upper_gap_m={max_upper_gap:.12g}",
        flush=True,
    )
    print(
        f"R9K_PAIRED_POINT_MARGIN positive={paired_any_n}/{common_n} "
        f"credential_worse={paired_cred_n} fragment_worse={paired_frag_n} "
        f"max_credential_margin_m={max_cred_margin:.12g} "
        f"max_fragment_margin_m={max_frag_margin:.12g} "
        f"min_positive_margin_m={min_positive_margin:.12g}",
        flush=True,
    )
    print("FULL_AUGMENTED_DECLARED_DIAGNOSTIC_POINT_GFP_SOLVED=YES", flush=True)
    print("DEPLOYMENT_PROTOCOL_CERTIFIED=NO", flush=True)
    print("CONTINUOUS_ACTION_INTERVAL_GFP_SOLVED=NO", flush=True)
    print("P2C_INDEPENDENT_LOWER_CHECKER=NO", flush=True)
    print("R9K_EXECUTION=PASS", flush=True)
    print(f"R9K_CLASSIFICATION={classification}", flush=True)
    print("CONTINUOUS_STATE_SEPARATION_CERTIFIED=NO", flush=True)
    print(f"R9K_NEXT_ACTION={next_action}", flush=True)
    print(f"RESULT_JSON={result}", flush=True)
    print(f"TOP_WITNESS_CSV={csvp}", flush=True)
    print(f"MANIFEST={manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
