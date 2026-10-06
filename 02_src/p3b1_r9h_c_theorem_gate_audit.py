from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import itertools
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


def resolve_artifact(recorded: str | Path) -> Path:
    p = Path(recorded)
    if p.is_file():
        return p
    q = RESULTS / p.name
    if q.is_file():
        return q
    raise FileNotFoundError(f"R9HC_ARTIFACT_NOT_FOUND recorded={p} recovered={q}")


def axis_cells_intersecting(axis: np.ndarray, lo: float, hi: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    lo, hi = float(min(lo, hi)), float(max(lo, hi))
    if hi < axis[0] - 1e-12 or lo > axis[-1] + 1e-12:
        return np.empty(0, dtype=np.int64)
    lo = max(lo, float(axis[0]))
    hi = min(hi, float(axis[-1]))
    left = axis[:-1]
    right = axis[1:]
    mask = (right >= lo - 1e-12) & (left <= hi + 1e-12)
    return np.flatnonzero(mask).astype(np.int64)


def product_count(parts: list[np.ndarray]) -> int:
    n = 1
    for x in parts:
        if len(x) == 0:
            return 0
        n *= len(x)
    return int(n)


def state_from_index(i: int, data: dict) -> dict:
    vf, vp, af, ba, bu, age = [np.asarray(x, float) for x in data["eval_flat"]]
    return {
        "v_f": float(vf[i]), "v_p": float(vp[i]), "a_f": float(af[i]),
        "bar_a": float(ba[i]), "bar_u": float(bu[i]), "age": float(age[i]),
    }


def action_bounds(width: float, u_min: float = -6.0, u_max: float = 2.5) -> list[tuple[float, float]]:
    if width <= 0:
        raise ValueError("R9HC_NONPOSITIVE_ACTION_WIDTH")
    out = []
    x = float(u_min)
    while x < u_max - 1e-14:
        y = min(x + width, u_max)
        out.append((float(x), float(y)))
        x = y
    return out


def local_axis_radius(axis: np.ndarray, x: float, fraction: float) -> float:
    axis = np.asarray(axis, float)
    j = int(np.argmin(np.abs(axis - x)))
    widths = []
    if j > 0:
        widths.append(float(axis[j] - axis[j-1]))
    if j + 1 < len(axis):
        widths.append(float(axis[j+1] - axis[j]))
    if not widths:
        return 0.0
    return float(fraction * min(widths))


def perturbation_states(st: dict, axes: list[np.ndarray], fraction: float) -> list[dict]:
    keys = ["v_f", "v_p", "a_f", "bar_a", "bar_u", "age"]
    out = [dict(st)]
    for k, axis in zip(keys, axes):
        r = local_axis_radius(axis, st[k], fraction)
        if r <= 0:
            continue
        for sgn in (-1.0, +1.0):
            q = dict(st)
            q[k] = float(np.clip(st[k] + sgn*r, float(axis[0]), float(axis[-1])))
            out.append(q)
    return out


def endpoint_interval_geometry(*, st: dict, env: dict, u_lo: float, u_hi: float,
                               data: dict, r3, b0, r9e) -> dict:
    """Sound *geometric* action-interval cover conditional on monotone endpoint flow.

    This gate certifies that the complete endpoint rectangle induced by one action
    interval is represented by lookup cells.  It deliberately does not claim a
    Bellman-value interval theorem; R9-H-D must propagate paired values over these
    cells to close the fixed-point certificate.
    """
    Ts = float(data["Ts"])
    tau = float(data["p2b"]["follower"]["tau"])
    w = float(data["p1"]["uncertainty"]["follower_actuation_abs"])
    times = np.asarray([0.0, Ts], dtype=float)

    vals = []
    for u in (u_lo, 0.5*(u_lo+u_hi), u_hi):
        P, V, A, _ = r9e.generalized_follower_motion(
            np.asarray([st["v_f"]]), np.asarray([st["a_f"]]), float(u),
            times, tau, w,
        )
        vals.append((float(P[0,-1]), float(V[0,-1]), float(A[0,-1])))
    p0,v0,a0 = vals[0]; pm,vm,am = vals[1]; p1,v1,a1 = vals[2]
    monotone = (
        min(p0,p1)-1e-11 <= pm <= max(p0,p1)+1e-11 and
        min(v0,v1)-1e-11 <= vm <= max(v0,v1)+1e-11 and
        min(a0,a1)-1e-11 <= am <= max(a0,a1)+1e-11
    )

    cfg, p3a, p2b = data["eval_cfg"], data["p3a"], data["p2b"]
    J = float(cfg["information_contract"]["slew_rate"])
    ag = np.asarray([st["age"]]); ba = np.asarray([st["bar_a"]]); bu = np.asarray([st["bar_u"]])
    ap = b0.predecessor_acceleration_lower(ag, ba, bu, J, p3a, p2b)
    _, Vp, _, _, _ = r3.predecessor_motion_with_stop(
        np.asarray([st["v_p"]]), ap, bu, ag, times, J, p3a
    )
    vp1 = float(Vp[0,-1])

    axes = data["lookup_axes"]
    hold_parts = [
        axis_cells_intersecting(axes[0], min(v0,v1), max(v0,v1)),
        axis_cells_intersecting(axes[1], vp1, vp1),
        axis_cells_intersecting(axes[2], min(a0,a1), max(a0,a1)),
        axis_cells_intersecting(axes[3], st["bar_a"], st["bar_a"]),
        axis_cells_intersecting(axes[4], st["bar_u"], st["bar_u"]),
        axis_cells_intersecting(axes[5], st["age"] + Ts, st["age"] + Ts),
    ]
    # For completion, candidate age is supplied by the caller's matched chi
    # semantics.  The age coordinate below is only a geometry check; H-D will
    # propagate exact stage-specific candidate ages.
    cand_age = float(env.get("pending_age_s", 0.0) + Ts)
    adopt_parts = [
        hold_parts[0], hold_parts[1], hold_parts[2],
        axis_cells_intersecting(axes[3], env["a_lower"], env["a_upper"]),
        axis_cells_intersecting(axes[4], env["u_lower"], env["u_upper"]),
        axis_cells_intersecting(axes[5], cand_age, cand_age),
    ]
    return {
        "endpoint_monotone": bool(monotone),
        "hold_cell_product": product_count(hold_parts),
        "adopt_cell_product": product_count(adopt_parts),
        "vf_endpoint_width": abs(v1-v0),
        "af_endpoint_width": abs(a1-a0),
        "position_endpoint_width": abs(p1-p0),
    }


def p2c_lower_semantics_audit(st: dict, data: dict, b0, sw,
                              resolutions: list[int]) -> dict:
    cfg, p2b, p2c, p3a = data["eval_cfg"], data["p2b"], data["p2c"], data["p3a"]
    J = float(cfg["information_contract"]["slew_rate"])
    ap = b0.predecessor_acceleration_lower(
        np.asarray([st["age"]]), np.asarray([st["bar_a"]]), np.asarray([st["bar_u"]]),
        J, p3a, p2b,
    )
    lowers = []
    uppers = []
    for r in resolutions:
        lo, hi, _ = sw.switching_loss_bracket(
            np.asarray([st["v_f"]]), np.asarray([st["a_f"]]),
            np.asarray([st["v_p"]]), ap,
            int(p2c["switching"]["N_sw"]), [int(r)], p2c, p2b,
        )
        lowers.append(float(np.asarray(lo).reshape(-1)[0]))
        uppers.append(float(np.asarray(hi).reshape(-1)[0]))
    ordered = all(l <= u + 1e-12 for l,u in zip(lowers,uppers))
    lower_nondecreasing = all(lowers[i+1] + 1e-10 >= lowers[i] for i in range(len(lowers)-1))
    upper_nonincreasing = all(uppers[i+1] <= uppers[i] + 1e-10 for i in range(len(uppers)-1))
    nested = ordered and lower_nondecreasing and upper_nonincreasing
    return {
        "lower_values": lowers,
        "upper_values": uppers,
        "ordered": ordered,
        "lower_nondecreasing": lower_nondecreasing,
        "upper_nonincreasing": upper_nonincreasing,
        "nested_bracket": nested,
        "finest_width_m": max(uppers[-1]-lowers[-1], 0.0),
    }


def self_test() -> None:
    a = np.asarray([0.,1.,2.,3.])
    assert list(axis_cells_intersecting(a, 0.2, 1.8)) == [0,1]
    assert product_count([np.asarray([0,1]), np.asarray([3])]) == 2
    b = action_bounds(0.25)
    assert len(b) == 34 and abs(b[0][0] + 6.0) < 1e-12 and abs(b[-1][1]-2.5) < 1e-12
    st = {"v_f":1.,"v_p":2.,"a_f":0.,"bar_a":0.,"bar_u":0.,"age":0.5}
    axes = [np.asarray([0.,1.,2.,3.]) for _ in range(6)]
    p = perturbation_states(st, axes, 1e-4)
    assert len(p) >= 7
    print("R9HC_INTERNAL_SELF_TEST=PASS")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--action-width", type=float, default=0.03125)
    ap.add_argument("--state-box-fraction", type=float, default=1.0e-4)
    ap.add_argument("--p2c-resolutions", default="257,513,1025,2049")
    args = ap.parse_args()
    if args.self_test:
        self_test(); return 0
    if not (0 < args.action_width <= 0.25):
        raise ValueError("R9HC_ACTION_WIDTH_OUT_OF_RANGE")
    if not (0 < args.state_box_fraction <= 1e-2):
        raise ValueError("R9HC_STATE_BOX_FRACTION_OUT_OF_RANGE")
    resolutions = [int(x) for x in args.p2c_resolutions.split(",") if x.strip()]
    if len(resolutions) < 2 or any(r < 17 for r in resolutions):
        raise ValueError("R9HC_BAD_P2C_RESOLUTIONS")

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
    import p3b1_r9g_matched_chi_branching_service as r9g

    hb_path = RESULTS / "P3B1_R9HB_LATEST.json"
    hb = json.loads(hb_path.read_text(encoding="utf-8"))
    exp = "MATCHED_CHI_GAP_PERSISTS_UNDER_ACTION_REFINEMENT_CONTINUOUS_PROMOTION_BLOCKED_BY_REMAINING_THEOREM_GATES"
    if hb.get("status") != "PASS" or hb.get("classification") != exp:
        raise RuntimeError(f"R9HC_R9HB_GATE_FAIL classification={hb.get('classification')}")
    m = hb.get("metrics", {})
    if int(m.get("stable_sensitive_tests",0)) != int(m.get("matched_chi_tests",-1)):
        raise RuntimeError("R9HC_R9HB_NOT_ALL_STABLE")
    if int(m.get("sign_flip_tests",1)) != 0:
        raise RuntimeError("R9HC_R9HB_SIGN_FLIP")

    rg = json.loads((RESULTS / "P3B1_R9G_LATEST.json").read_text(encoding="utf-8"))
    tests_path = resolve_artifact(rg["artifacts"]["matched_chi_tests_csv"])
    with tests_path.open(newline="", encoding="utf-8") as f:
        all_tests = list(csv.DictReader(f))
    tests = [r for r in all_tests if float(r.get("full_abs_progress_gap_m") or 0.0) > TOL]
    if len(tests) != int(m.get("matched_chi_tests", len(tests))):
        raise RuntimeError(f"R9HC_TEST_COUNT_MISMATCH hb={m.get('matched_chi_tests')} rg={len(tests)}")

    data = r9d.reconstruct_timestamped_geometry(r3, r5, r6, r7, r9c)
    Ts = float(data["Ts"])
    J = float(data["eval_cfg"]["information_contract"]["slew_rate"])

    print("=== P3-B1-R9-H-C THEOREM-GATE AUDIT ===", flush=True)
    print("UPSTREAM_R9HB_ACTION_REFINEMENT=PASS", flush=True)
    print(f"MATCHED_CHI_TESTS={len(tests)}", flush=True)
    print(f"ACTION_INTERVAL_WIDTH={args.action_width:.12g}", flush=True)
    print(f"STATE_BOX_FRACTION={args.state_box_fraction:.12g}", flush=True)
    print("FULL_AUGMENTED_INTERVAL_GFP_SOLVED=NO", flush=True)
    print("CONTINUOUS_STATE_SEPARATION_CERTIFIED=NO", flush=True)

    # Unique physical states for expensive geometry/P2C auditing.
    by_node: dict[int, dict] = {}
    for r in tests:
        i = int(r["node_index"])
        if i not in by_node or float(r["full_abs_progress_gap_m"]) > float(by_node[i]["full_abs_progress_gap_m"]):
            by_node[i] = r

    intervals = action_bounds(args.action_width)
    interval_rows: list[dict] = []
    interval_monotone_fail = 0
    interval_lookup_fail = 0
    max_hold_cells = 0
    max_adopt_cells = 0
    for node_index, row in sorted(by_node.items()):
        st = state_from_index(node_index, data)
        pending_age = float(row["pending_age_s"])
        env = r9g.matched_pending_envelope(
            adopted_age=st["age"], pending_age=pending_age,
            bar_a=st["bar_a"], bar_u=st["bar_u"], J=J,
            b0=b0, p3a=data["p3a"], p2b=data["p2b"],
        )
        if not env.get("valid", False):
            raise RuntimeError(f"R9HC_INVALID_MATCHED_CHI_ENV node={node_index}")
        for lo, hi in intervals:
            z = endpoint_interval_geometry(
                st=st, env=env, u_lo=lo, u_hi=hi,
                data=data, r3=r3, b0=b0, r9e=r9e,
            )
            interval_rows.append({
                "node_index": node_index, "pending_age_s": pending_age,
                "u_lo": lo, "u_hi": hi, **z,
            })
            if not z["endpoint_monotone"]:
                interval_monotone_fail += 1
            if z["hold_cell_product"] <= 0 or z["adopt_cell_product"] <= 0:
                interval_lookup_fail += 1
            max_hold_cells = max(max_hold_cells, int(z["hold_cell_product"]))
            max_adopt_cells = max(max_adopt_cells, int(z["adopt_cell_product"]))

    interval_geometry_pass = interval_monotone_fail == 0 and interval_lookup_fail == 0
    print(
        f"R9HC_ACTION_INTERVAL_GEOMETRY intervals={len(intervals)} tests={len(interval_rows)} "
        f"monotone_fail={interval_monotone_fail} lookup_fail={interval_lookup_fail} "
        f"max_hold_cells={max_hold_cells} max_adopt_cells={max_adopt_cells}", flush=True,
    )

    # P2C lower/upper directional audit on every unique matched physical state.
    p2c_rows = []
    p2c_fail = 0
    max_p2c_width = 0.0
    for node_index in sorted(by_node):
        st = state_from_index(node_index, data)
        z = p2c_lower_semantics_audit(st, data, b0, sw, resolutions)
        p2c_rows.append({
            "node_index": node_index,
            "resolutions": ";".join(map(str,resolutions)),
            "lower_values": ";".join(f"{x:.17g}" for x in z["lower_values"]),
            "upper_values": ";".join(f"{x:.17g}" for x in z["upper_values"]),
            "ordered": z["ordered"],
            "lower_nondecreasing": z["lower_nondecreasing"],
            "upper_nonincreasing": z["upper_nonincreasing"],
            "nested_bracket": z["nested_bracket"],
            "finest_width_m": z["finest_width_m"],
        })
        if not z["nested_bracket"]:
            p2c_fail += 1
        max_p2c_width = max(max_p2c_width, float(z["finest_width_m"]))
    p2c_directional_pass = p2c_fail == 0
    print(
        f"R9HC_P2C_DIRECTIONAL_AUDIT nodes={len(by_node)} failures={p2c_fail} "
        f"max_finest_bracket_width_m={max_p2c_width:.12g}", flush=True,
    )

    # Local state-box robustness diagnostic.  This is intentionally NOT promoted
    # to a theorem certificate: it uses axis perturbations, not an interval fixed
    # point over the whole continuous box.  It is useful because a sign reversal
    # here would stop the continuous program immediately.
    axes = [np.asarray(a,float) for a in data["eval_axes"]]
    restart_actions = [-6.0, -5.5, -5.0, -4.5, -4.0, -3.5, -3.0, -2.5,
                       -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    # Use R9-D continuation as a fixed diagnostic terminal only.  It must not be
    # confused with the final interval-GFP theorem object.
    profile = "diagnostic_fast"
    R = r7.service_horizon(profile, data["p1"]["diagnostic_service_profiles"][profile])
    cfg_restart, td_restart = r9d.build_timestamped_transitions(
        data, [-3.0,-2.0,-1.0], R, r3, b0, chunk_size=args.chunk_size
    )
    sol = r9d.solve_timestamped_causal(
        R, cfg_restart, data["eval_idx"], data["lookup_shape"],
        data["lookup_fallback"], data["eval_fallback"], td_restart, r3,
        label="r9hc_state_box_terminal_diagnostic",
    )
    if not sol.converged:
        raise RuntimeError("R9HC_DIAGNOSTIC_TERMINAL_NOT_CONVERGED")
    restart_cells = r9g.build_restart_cell_values(sol.h_flat, data, R, r3)

    state_rows = []
    state_sign_fail = 0
    min_local_gap = math.inf
    # To control run time, audit the strongest pending-age test at each physical node.
    fallback_cache: dict = {}
    for node_index, row in sorted(by_node.items()):
        center = state_from_index(node_index, data)
        pending_age = float(row["pending_age_s"])
        base_env = r9g.matched_pending_envelope(
            adopted_age=center["age"], pending_age=pending_age,
            bar_a=center["bar_a"], bar_u=center["bar_u"], J=J,
            b0=b0, p3a=data["p3a"], p2b=data["p2b"],
        )
        if not base_env.get("valid",False):
            continue
        gaps = []
        for pst in perturbation_states(center, axes, args.state_box_fraction):
            # Keep the same pending generation age offset when adopted age is perturbed.
            env = r9g.matched_pending_envelope(
                adopted_age=pst["age"], pending_age=pending_age,
                bar_a=pst["bar_a"], bar_u=pst["bar_u"], J=J,
                b0=b0, p3a=data["p3a"], p2b=data["p2b"],
            )
            if not env.get("valid",False):
                continue
            z = r9g.matched_stage_pair(
                st=pst, env=env, pending_age=pending_age,
                actions=restart_actions, restart_cells=restart_cells,
                data=data, r3=r3, b0=b0, sw=sw, r9e=r9e,
                r9f=__import__("p3b1_r9f_sparse_chi_envelope_graph"),
                fallback_cache=fallback_cache,
            )
            if z.get("valid",False):
                gaps.append(float(z["signed_verify_minus_fragment_m"]))
        if not gaps:
            state_sign_fail += 1
            continue
        sign_ok = all(g > TOL for g in gaps) or all(g < -TOL for g in gaps)
        if not sign_ok:
            state_sign_fail += 1
        local_abs_min = min(abs(g) for g in gaps)
        min_local_gap = min(min_local_gap, local_abs_min)
        state_rows.append({
            "node_index": node_index,
            "pending_age_s": pending_age,
            "perturbation_tests": len(gaps),
            "signed_gap_min_m": min(gaps),
            "signed_gap_max_m": max(gaps),
            "min_abs_gap_m": local_abs_min,
            "sign_stable": sign_ok,
            "state_box_fraction": args.state_box_fraction,
        })
    state_box_diag_pass = state_sign_fail == 0 and bool(state_rows)
    print(
        f"R9HC_STATE_BOX_DIAGNOSTIC nodes={len(state_rows)} sign_fail={state_sign_fail} "
        f"min_abs_gap_m={(min_local_gap if math.isfinite(min_local_gap) else math.nan):.12g}", flush=True,
    )

    # Critical distinction: the first two PASS conditions are useful theorem
    # ingredients, but neither the directional P2C audit nor the axis-perturbation
    # state test is by itself a proof of the continuous maximal kernel.  The final
    # missing object is a paired interval GFP over the reachable service graph.
    p2c_lower_semantics_theorem_certified = False
    continuous_state_cell_enclosure_certified = False
    continuous_action_restart_interval_certified = False
    full_augmented_interval_gfp_solved = False
    continuous_yes = False

    if interval_geometry_pass and p2c_directional_pass and state_box_diag_pass:
        classification = "LOCAL_THEOREM_PREFLIGHTS_PASS_PAIRED_INTERVAL_GFP_REMAINS"
        next_action = "R9H_D_SOLVE_PAIRED_INTERVAL_GFP_ON_REACHABLE_SERVICE_GRAPH_WITH_VALIDATED_P2C_LOWER_CHECKER"
    elif not interval_geometry_pass:
        classification = "ACTION_INTERVAL_GEOMETRY_GATE_FAILED"
        next_action = "R9H_C_R1_REFINE_ACTION_INTERVAL_OR_LOOKUP_HALO"
    elif not p2c_directional_pass:
        classification = "P2C_LOWER_BRACKET_DIRECTIONAL_GATE_FAILED"
        next_action = "R9H_C_R1_BUILD_INDEPENDENT_P2C_LOWER_CHECKER"
    else:
        classification = "LOCAL_STATE_BOX_SIGN_STABILITY_FAILED"
        next_action = "R9H_C_R1_SHRINK_STATE_BOX_OR_REASSESS_CONTINUOUS_WITNESS"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_json = RESULTS / f"P3B1_R9HC_RESULT_{stamp}.json"
    latest_json = RESULTS / "P3B1_R9HC_LATEST.json"
    interval_csv = RESULTS / f"P3B1_R9HC_ACTION_INTERVAL_GEOMETRY_{stamp}.csv"
    p2c_csv = RESULTS / f"P3B1_R9HC_P2C_DIRECTIONAL_{stamp}.csv"
    state_csv = RESULTS / f"P3B1_R9HC_STATE_BOX_DIAGNOSTIC_{stamp}.csv"
    manifest = RESULTS / f"P3B1_R9HC_MANIFEST_{stamp}.sha256"
    write_csv(interval_csv, interval_rows)
    write_csv(p2c_csv, p2c_rows)
    write_csv(state_csv, state_rows)

    try:
        p2c_source_sha = hashlib.sha256(inspect.getsource(sw.switching_loss_bracket).encode()).hexdigest()
    except Exception:
        p2c_source_sha = "UNAVAILABLE"

    out = {
        "schema": "SCV_P3B1_R9HC_THEOREM_GATE_AUDIT_V1",
        "status": "PASS" if classification.startswith("LOCAL_THEOREM_PREFLIGHTS_PASS") else "REVIEW",
        "timestamp_utc": stamp,
        "classification": classification,
        "next_action": next_action,
        "continuous_state_separation_certified": continuous_yes,
        "full_augmented_interval_gfp_solved": full_augmented_interval_gfp_solved,
        "metrics": {
            "matched_chi_tests": len(tests),
            "unique_physical_nodes": len(by_node),
            "action_interval_width": args.action_width,
            "action_intervals": len(intervals),
            "action_interval_tests": len(interval_rows),
            "action_interval_monotonicity_failures": interval_monotone_fail,
            "action_interval_lookup_failures": interval_lookup_fail,
            "max_hold_cell_product": max_hold_cells,
            "max_adopt_cell_product": max_adopt_cells,
            "p2c_nodes": len(p2c_rows),
            "p2c_directional_failures": p2c_fail,
            "p2c_max_finest_bracket_width_m": max_p2c_width,
            "state_box_nodes": len(state_rows),
            "state_box_sign_failures": state_sign_fail,
            "state_box_min_abs_gap_m": None if not math.isfinite(min_local_gap) else min_local_gap,
            "upstream_max_finest_gap_m": float(m.get("max_finest_gap_m",0.0)),
            "upstream_max_action_refinement_drift_m": float(m.get("max_action_refinement_drift_m",math.nan)),
        },
        "gates": {
            "r9hb_action_refinement_pass": True,
            "action_interval_geometry_complete": interval_geometry_pass,
            "p2c_bracket_directional_audit_pass": p2c_directional_pass,
            "local_state_box_sign_stability_pass": state_box_diag_pass,
            "p2c_lower_bound_semantics_theorem_certified": p2c_lower_semantics_theorem_certified,
            "continuous_state_cell_enclosure_certified": continuous_state_cell_enclosure_certified,
            "continuous_action_restart_interval_certified": continuous_action_restart_interval_certified,
            "full_augmented_interval_gfp_solved": full_augmented_interval_gfp_solved,
        },
        "p2c_switching_loss_bracket_source_sha256": p2c_source_sha,
        "artifacts": {
            "action_interval_geometry_csv": str(interval_csv),
            "p2c_directional_csv": str(p2c_csv),
            "state_box_diagnostic_csv": str(state_csv),
        },
    }
    text = json.dumps(out, indent=2, sort_keys=True)
    atomic_write(result_json, text); atomic_write(latest_json, text)
    mfiles = [Path(__file__), result_json, interval_csv, p2c_csv, state_csv, hb_path, tests_path]
    atomic_write(manifest, "".join(f"{sha256_file(p)}  {p}\n" for p in mfiles if p.exists()))

    print("=== R9-H-C DECISION ===", flush=True)
    print(f"R9HC_ACTION_INTERVAL_GEOMETRY_GATE={'PASS' if interval_geometry_pass else 'FAIL'}", flush=True)
    print(f"R9HC_P2C_DIRECTIONAL_GATE={'PASS' if p2c_directional_pass else 'FAIL'}", flush=True)
    print(f"R9HC_STATE_BOX_DIAGNOSTIC_GATE={'PASS' if state_box_diag_pass else 'FAIL'}", flush=True)
    print("P2C_LOWER_BOUND_SEMANTICS_CERTIFIED=NO", flush=True)
    print("CONTINUOUS_STATE_CELL_ENCLOSURE_CERTIFIED=NO", flush=True)
    print("CONTINUOUS_ACTION_RESTART_INTERVAL_CERTIFIED=NO", flush=True)
    print("FULL_AUGMENTED_INTERVAL_GFP_SOLVED=NO", flush=True)
    print("R9HC_EXECUTION=PASS" if out["status"] == "PASS" else "R9HC_EXECUTION=REVIEW", flush=True)
    print(f"R9HC_CLASSIFICATION={classification}", flush=True)
    print("CONTINUOUS_STATE_SEPARATION_CERTIFIED=NO", flush=True)
    print(f"R9HC_NEXT_ACTION={next_action}", flush=True)
    print(f"RESULT_JSON={result_json}", flush=True)
    print(f"MANIFEST={manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
