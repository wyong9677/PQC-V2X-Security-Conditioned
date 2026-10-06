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

ROOT = Path.home() / "Desktop/paper set/PQC_V2X_Security_Conditioned/numerical_experiments"
SRC = ROOT / "02_src"
RESULTS = ROOT / "04_results"
TOL = 1.0e-10


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
    fields: list[str] = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                fields.append(k); seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def resolve_artifact(recorded: str | Path) -> Path:
    p = Path(recorded)
    if p.is_file():
        return p
    q = RESULTS / p.name
    if q.is_file():
        return q
    raise FileNotFoundError(f"R9HB_ARTIFACT_NOT_FOUND recorded={p} recovered={q}")


def as_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"1", "true", "yes", "y", "pass"}


def refinement_actions(width: float, *, u_min=-6.0, u_max=2.5) -> list[float]:
    # -6 is the physical fallback command and therefore is not used as a
    # cooperative feasible sample.  It remains the closure endpoint of the
    # interval cover.  nextafter gives a deterministic interior representative.
    first = float(np.nextafter(float(u_min), math.inf))
    xs = [first]
    x = u_min + width
    while x < u_max - 1e-12:
        xs.append(float(x)); x += width
    xs.append(float(u_max))
    return sorted(set(xs))


def load_stage_tests(r9g: dict) -> list[dict]:
    p = resolve_artifact(r9g["artifacts"]["matched_chi_tests_csv"])
    with p.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = [r for r in rows if float(r.get("full_abs_progress_gap_m") or 0.0) > TOL]
    if not out:
        raise RuntimeError("R9HB_NO_UPSTREAM_MATCHED_CHI_SENSITIVE_TESTS")
    return out


def full_action_restart_transitions(data, actions, R, r3, b0, r9e, *, chunk_size=4096):
    """Timestamp-correct restart transitions using the generalized propagator.

    This removes R9-D's negative-equilibrium restriction.  It is still a point-
    action continuation, not an interval-action theorem certificate; that fact is
    explicitly kept as a promotion gate in the result.
    """
    import copy
    cfg = copy.deepcopy(data["eval_cfg"])
    cfg["cooperative_actions"] = [float(a) for a in actions]
    vf, vp, af, bar_a, bar_u, age = [np.asarray(x, float) for x in data["eval_flat"]]
    n = len(vf)
    p1, p2b, p3a = data["p1"], data["p2b"], data["p3a"]
    Ts = float(data["Ts"])
    J = float(cfg["information_contract"]["slew_rate"])
    nt = int(cfg["one_step"]["trajectory_points"])
    times = np.linspace(0.0, Ts, nt)
    sd = p2b["state_domain"]
    speed_bound = float(sd["v_f_max"]) + float(sd["v_p_max"])
    lips = 0.5 * speed_bound * (Ts / (nt - 1))
    axes = data["lookup_axes"]
    tau = float(p2b["follower"]["tau"])
    w = float(p1["uncertainty"]["follower_actuation_abs"])

    outs = []
    for a in actions:
        outs.append({
            "action": float(a),
            "step_loss_upper": np.empty(n),
            "closing_end": np.empty(n),
            "defer_index": np.empty(n, dtype=np.int64),
            "defer_valid": np.empty(n, dtype=bool),
            "adopt_index_by_q": np.empty((R, n), dtype=np.int64),
            "adopt_valid_by_q": np.empty((R, n), dtype=bool),
        })

    for start in range(0, n, int(chunk_size)):
        stop = min(n, start + int(chunk_size)); sl = slice(start, stop)
        vfc, vpc, afc = vf[sl], vp[sl], af[sl]
        bac, buc, agc = bar_a[sl], bar_u[sl], age[sl]
        ap_lower = b0.predecessor_acceleration_lower(agc, bac, buc, J, p3a, p2b)
        Pp, Vp, Ap, Up, _ = r3.predecessor_motion_with_stop(vpc, ap_lower, buc, agc, times, J, p3a)
        vp_end, ap_end, up_end = Vp[:, -1], Ap[:, -1], Up[:, -1]
        age_hold = agc + Ts
        for out in outs:
            a = float(out["action"])
            Pf, Vf, Af, _ = r9e.generalized_follower_motion(vfc, afc, a, times, tau, w)
            closing = Pf - Pp
            out["step_loss_upper"][sl] = np.maximum(np.max(closing, axis=1) + lips, 0.0)
            out["closing_end"][sl] = Pf[:, -1] - Pp[:, -1]
            didx, dvalid = r3.locate_cells(axes, [Vf[:, -1], vp_end, Af[:, -1], bac, buc, age_hold])
            out["defer_index"][sl] = didx; out["defer_valid"][sl] = dvalid
            m = stop - start
            for q in range(1, R + 1):
                c_age = np.full(m, (R - q + 1) * Ts)
                cidx, cvalid = r3.locate_cells(axes, [Vf[:, -1], vp_end, Af[:, -1], ap_end, up_end, c_age])
                out["adopt_index_by_q"][q-1, sl] = cidx
                out["adopt_valid_by_q"][q-1, sl] = cvalid
        if start == 0 or stop == n or (stop // int(chunk_size)) % 16 == 0:
            print(f"R9HB_RESTART_TRANSITION_PROGRESS states={stop}/{n} actions={len(actions)}", flush=True)

    invalid = sum(int(np.count_nonzero(~o["defer_valid"])) + int(np.count_nonzero(~o["adopt_valid_by_q"])) for o in outs)
    if invalid:
        raise RuntimeError(f"R9HB_RESTART_LOOKUP_HALO_FAIL invalid={invalid}")
    return cfg, {"fallback_required": np.asarray(data["eval_fallback"], float), "transitions": outs, "lipschitz_correction_m": lips}


def self_test() -> None:
    for w, expected_min in [(0.25, 35), (0.125, 69), (0.5, 18)]:
        a = refinement_actions(w)
        assert len(a) >= expected_min - 1
        assert a[0] > -6.0 and a[-1] == 2.5
    print("R9HB_INTERNAL_SELF_TEST=PASS")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--restart-action-width", type=float, default=0.5)
    ap.add_argument("--refinement-widths", default="0.25,0.125,0.0625,0.03125")
    args = ap.parse_args()
    if args.self_test:
        self_test(); return 0

    widths = [float(x) for x in args.refinement_widths.split(",") if x.strip()]
    if not widths or any(w <= 0 or w > 0.5 for w in widths):
        raise ValueError("R9HB_BAD_REFINEMENT_WIDTHS")
    if sorted(widths, reverse=True) != widths:
        raise ValueError("R9HB_REFINEMENT_WIDTHS_MUST_DESCEND")

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
    import p3b1_r9f_sparse_chi_envelope_graph as r9f
    import p3b1_r9g_matched_chi_branching_service as r9g
    import p3b1_r9h_a_r1_reachable_protocol as r9har1

    ha = json.loads((RESULTS / "P3B1_R9HA_R1_LATEST.json").read_text(encoding="utf-8"))
    exp = "READY_FOR_R9H_B_REACHABLE_PROTOCOLIZED_SPARSE_GFP_AND_INTERVAL_INNER_OUTER"
    if ha.get("status") != "PASS" or ha.get("classification") != exp:
        raise RuntimeError(f"R9HB_R9HAR1_GATE_FAIL classification={ha.get('classification')}")
    if not ha.get("protocol_graph", {}).get("pass", False):
        raise RuntimeError("R9HB_PROTOCOL_REACHABILITY_NOT_PASS")
    if int(ha.get("metrics", {}).get("interval_monotonicity_failures", 1)) != 0 or int(ha.get("metrics", {}).get("interval_lookup_failures", 1)) != 0:
        raise RuntimeError("R9HB_INTERVAL_PREFLIGHT_NOT_PASS")

    rg = json.loads((RESULTS / "P3B1_R9G_LATEST.json").read_text(encoding="utf-8"))
    if rg.get("classification") != "MATCHED_CHI_SERVICE_PROGRESS_SENSITIVITY_FOUND_DIAGNOSTIC_BRANCHING_AUDIT":
        raise RuntimeError("R9HB_R9G_GATE_FAIL")
    tests = load_stage_tests(rg)

    data = r9d.reconstruct_timestamped_geometry(r3, r5, r6, r7, r9c)
    profile = "diagnostic_fast"
    R = r7.service_horizon(profile, data["p1"]["diagnostic_service_profiles"][profile])
    if R != 2:
        raise RuntimeError("R9HB_EXPECTED_R2")
    Ts = float(data["Ts"])

    print("=== P3-B1-R9-H-B PROTOCOLIZED PAIRED-BOUND AUDIT ===", flush=True)
    print("UPSTREAM_R9HAR1_REACHABILITY=PASS", flush=True)
    print("UPSTREAM_R9HA_INTERVAL_PREFLIGHT=PASS", flush=True)
    print("MATCHED_CHI_PROGRESS_UPSTREAM=PASS", flush=True)
    print("PROTOCOL_GRAPH=PENDING_ENTRY,FRAGMENTED,VERIFYING,VERIFYING_LAST", flush=True)
    print("CO_REACHABLE_MATCHED_PAIR=YES", flush=True)
    print("DEPLOYMENT_PROTOCOL_CERTIFIED=NO", flush=True)
    print("P2C_LOWER_BOUND_SEMANTICS_CERTIFIED=NO", flush=True)
    print("CONTINUOUS_STATE_CELL_ENCLOSURE_CERTIFIED=NO", flush=True)
    print("CONTINUOUS_STATE_SEPARATION_CERTIFIED=NO", flush=True)

    # Restart continuation: substantially denser than R9-G, and now capable of
    # positive actions.  Still point-action based, therefore explicitly not used
    # as a theorem-grade continuous-action promotion gate.
    restart_actions = refinement_actions(float(args.restart_action_width))
    print(f"R9HB_STAGE_START=generalized_restart_continuation actions={len(restart_actions)}", flush=True)
    cfg_restart, td_restart = full_action_restart_transitions(
        data, restart_actions, R, r3, b0, r9e, chunk_size=args.chunk_size
    )
    sol = r9d.solve_timestamped_causal(
        R, cfg_restart, data["eval_idx"], data["lookup_shape"],
        data["lookup_fallback"], data["eval_fallback"], td_restart, r3,
        label="r9hb_generalized_restart_continuation",
    )
    if not sol.converged:
        raise RuntimeError("R9HB_RESTART_CONTINUATION_NOT_CONVERGED")
    restart_cells = r9g.build_restart_cell_values(sol.h_flat, data, R, r3)

    vf, vp, af, ba, bu, age = [np.asarray(x, float) for x in data["eval_flat"]]
    J = float(data["eval_cfg"]["information_contract"]["slew_rate"])
    fallback_cache: dict = {}
    rows: list[dict] = []
    summary: list[dict] = []
    stable_sensitive = 0
    max_finest_gap = 0.0
    max_refinement_drift = 0.0
    sign_flip_count = 0

    # Group by physical node + pending age; these are the 16 matched-chi tests.
    for k, row in enumerate(tests, start=1):
        i = int(row["node_index"]); b = float(row["pending_age_s"])
        st = {"v_f": float(vf[i]), "v_p": float(vp[i]), "a_f": float(af[i]),
              "bar_a": float(ba[i]), "bar_u": float(bu[i]), "age": float(age[i])}
        env = r9g.matched_pending_envelope(
            adopted_age=st["age"], pending_age=b, bar_a=st["bar_a"], bar_u=st["bar_u"],
            J=J, b0=b0, p3a=data["p3a"], p2b=data["p2b"],
        )
        if not env.get("valid", False):
            raise RuntimeError(f"R9HB_MATCHED_CHI_ENV_INVALID node={i} pending_age={b}")

        vals = []
        for w in widths:
            actions = refinement_actions(w)
            z = r9g.matched_stage_pair(
                st=st, env=env, pending_age=b, actions=actions,
                restart_cells=restart_cells, data=data, r3=r3, b0=b0, sw=sw,
                r9e=r9e, r9f=r9f, fallback_cache=fallback_cache,
            )
            if not z.get("valid", False):
                raise RuntimeError(f"R9HB_STAGE_PAIR_INVALID node={i} width={w}")
            signed = float(z["signed_verify_minus_fragment_m"])
            gap = float(z["abs_progress_gap_m"])
            vals.append((w, signed, gap, z))
            rows.append({
                "node_index": i, "pending_age_s": b, "action_width": w,
                "action_count": len(actions),
                "fragmented_required_gap_m": z["fragmented_required_gap_m"],
                "verifying_required_gap_m": z["verifying_required_gap_m"],
                "signed_verify_minus_fragment_m": signed,
                "abs_progress_gap_m": gap,
                "fragmented_best_action": z["fragmented_best_action"],
                "verifying_best_action": z["verifying_best_action"],
                "matched_chi": True,
            })
        signed_seq = [v[1] for v in vals]
        gap_seq = [v[2] for v in vals]
        drift = max(abs(gap_seq[j] - gap_seq[j-1]) for j in range(1, len(gap_seq))) if len(gap_seq) > 1 else 0.0
        same_sign = all(x > TOL for x in signed_seq) or all(x < -TOL for x in signed_seq)
        if not same_sign:
            sign_flip_count += 1
        finest = gap_seq[-1]
        # Numerical stability gate only; this is deliberately NOT labeled a
        # theorem-grade lower bound.
        stable = bool(same_sign and finest > 10.0 * max(drift, TOL))
        if stable:
            stable_sensitive += 1
        max_finest_gap = max(max_finest_gap, finest)
        max_refinement_drift = max(max_refinement_drift, drift)
        summary.append({
            "node_index": i, "pending_age_s": b,
            "coarsest_gap_m": gap_seq[0], "finest_gap_m": finest,
            "max_refinement_drift_m": drift,
            "sign_stable": same_sign,
            "numerically_stable_progress_gap": stable,
        })
        print(f"R9HB_PROGRESS tests={k}/{len(tests)} stable={stable_sensitive} finest_gap_m={finest:.12g}", flush=True)

    # These two unresolved theorem gates are inherited from the continuous
    # closure audit and intentionally prevent a false YES.
    p2c_lower_certified = False
    state_cell_enclosure_certified = False
    restart_interval_certified = False
    theorem_promotion = bool(
        stable_sensitive > 0 and p2c_lower_certified and
        state_cell_enclosure_certified and restart_interval_certified
    )

    if theorem_promotion:
        classification = "CONTINUOUS_MATCHED_CHI_SERVICE_PROGRESS_SEPARATION_CERTIFIED"
        next_action = "R9H_C_PACKAGE_THEOREM_CERTIFICATES_AND_MANUSCRIPT_UPDATE"
    elif stable_sensitive > 0:
        classification = "MATCHED_CHI_GAP_PERSISTS_UNDER_ACTION_REFINEMENT_CONTINUOUS_PROMOTION_BLOCKED_BY_REMAINING_THEOREM_GATES"
        next_action = "R9H_C_CERTIFY_STATE_CELL_ENCLOSURE_RESTART_INTERVAL_AND_P2C_LOWER_SEMANTICS"
    else:
        classification = "MATCHED_CHI_GAP_NOT_STABLE_UNDER_ACTION_REFINEMENT"
        next_action = "R9H_C_REFINE_ACTION_INTERVALS_OR_REASSESS_SERVICE_PROGRESS_CLAIM"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_json = RESULTS / f"P3B1_R9HB_RESULT_{stamp}.json"
    latest_json = RESULTS / "P3B1_R9HB_LATEST.json"
    rows_csv = RESULTS / f"P3B1_R9HB_ACTION_REFINEMENT_{stamp}.csv"
    summary_csv = RESULTS / f"P3B1_R9HB_PAIRED_SUMMARY_{stamp}.csv"
    manifest = RESULTS / f"P3B1_R9HB_MANIFEST_{stamp}.sha256"
    write_csv(rows_csv, rows); write_csv(summary_csv, summary)

    out = {
        "schema": "SCV_P3B1_R9HB_PROTOCOLIZED_PAIRED_BOUNDS_V1",
        "status": "PASS",
        "timestamp_utc": stamp,
        "classification": classification,
        "next_action": next_action,
        "continuous_state_separation_certified": theorem_promotion,
        "matched_chi_service_progress_numerically_stable": stable_sensitive > 0,
        "full_augmented_gfp_solved": False,
        "metrics": {
            "matched_chi_tests": len(tests),
            "stable_sensitive_tests": stable_sensitive,
            "sign_flip_tests": sign_flip_count,
            "max_finest_gap_m": max_finest_gap,
            "max_action_refinement_drift_m": max_refinement_drift,
            "restart_action_width": float(args.restart_action_width),
            "restart_action_count": len(restart_actions),
            "action_refinement_widths": widths,
        },
        "gates": {
            "r9har1_reachable_protocol": True,
            "r9ha_interval_preflight": True,
            "matched_chi": True,
            "same_pending_generation_time": True,
            "same_two_step_liveness_bound": True,
            "generalized_restart_continuation_converged": True,
            "action_refinement_sign_stable_some_tests": stable_sensitive > 0,
            "restart_continuous_action_interval_certified": restart_interval_certified,
            "continuous_state_cell_enclosure_certified": state_cell_enclosure_certified,
            "p2c_lower_bound_semantics_certified": p2c_lower_certified,
            "deployment_protocol_certified": False,
        },
        "artifacts": {"action_refinement_csv": str(rows_csv), "paired_summary_csv": str(summary_csv)},
    }
    text = json.dumps(out, indent=2, sort_keys=True)
    atomic_write(result_json, text); atomic_write(latest_json, text)
    mfiles = [Path(__file__), result_json, rows_csv, summary_csv, RESULTS / "P3B1_R9HA_R1_LATEST.json", RESULTS / "P3B1_R9G_LATEST.json"]
    atomic_write(manifest, "".join(f"{sha256_file(p)}  {p}\n" for p in mfiles if p.exists()))

    print("=== R9-H-B DECISION ===", flush=True)
    print(f"R9HB_ACTION_REFINEMENT tests={len(tests)} stable_sensitive={stable_sensitive} sign_flip={sign_flip_count} max_finest_gap_m={max_finest_gap:.12g} max_drift_m={max_refinement_drift:.12g}", flush=True)
    print(f"R9HB_RESTART_CONTINUATION converged=YES point_action_width={args.restart_action_width:.12g} interval_certified=NO", flush=True)
    print("FULL_AUGMENTED_GFP_SOLVED=NO", flush=True)
    print("P2C_LOWER_BOUND_SEMANTICS_CERTIFIED=NO", flush=True)
    print("CONTINUOUS_STATE_CELL_ENCLOSURE_CERTIFIED=NO", flush=True)
    print("CONTINUOUS_ACTION_RESTART_INTERVAL_CERTIFIED=NO", flush=True)
    print("R9HB_EXECUTION=PASS", flush=True)
    print(f"R9HB_CLASSIFICATION={classification}", flush=True)
    print(f"CONTINUOUS_STATE_SEPARATION_CERTIFIED={'YES' if theorem_promotion else 'NO'}", flush=True)
    print(f"R9HB_NEXT_ACTION={next_action}", flush=True)
    print(f"RESULT_JSON={result_json}", flush=True)
    print(f"MANIFEST={manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
