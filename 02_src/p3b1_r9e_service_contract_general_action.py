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
CFGDIR = ROOT / "01_config"
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
    fields = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def raw_acceleration(a0, u, tau, w, t):
    ainf = u + tau * w
    return ainf + (a0 - ainf) * np.exp(-np.asarray(t, dtype=float) / tau)


def raw_velocity(v0, a0, u, tau, w, t):
    t = np.asarray(t, dtype=float)
    ainf = u + tau * w
    return v0 + ainf * t + tau * (a0 - ainf) * (1.0 - np.exp(-t / tau))


def raw_position(v0, a0, u, tau, w, t):
    t = np.asarray(t, dtype=float)
    ainf = u + tau * w
    return (
        v0 * t
        + 0.5 * ainf * t * t
        + tau * (a0 - ainf) * (t - tau * (1.0 - np.exp(-t / tau)))
    )


def accel_zero_time(a0: float, ainf: float, tau: float):
    den = a0 - ainf
    if abs(den) < 1e-15:
        return None
    ratio = -ainf / den
    if not (0.0 < ratio < 1.0):
        return None
    t = -tau * math.log(ratio)
    return t if math.isfinite(t) and t > 0.0 else None


def first_stop_time(v0: float, a0: float, u: float, tau: float, w: float, horizon: float) -> float:
    """First zero of raw velocity on [0,horizon], or +inf if none.

    a(t) is affine in exp(-t/tau), so it crosses zero at most once. Therefore
    velocity is monotone on at most two subintervals. We bracket the first
    root on those monotone pieces and bisect it. This supports braking,
    coasting, and positive-equilibrium cooperative commands without assuming
    negative equilibrium acceleration.
    """
    if v0 <= 0.0:
        return 0.0
    ainf = float(u + tau * w)
    breaks = [0.0]
    tz = accel_zero_time(float(a0), ainf, float(tau))
    if tz is not None and tz < horizon:
        breaks.append(float(tz))
    breaks.append(float(horizon))
    breaks = sorted(set(breaks))

    def vel(t: float) -> float:
        return float(raw_velocity(v0, a0, u, tau, w, t))

    for lo, hi in zip(breaks[:-1], breaks[1:]):
        vlo, vhi = vel(lo), vel(hi)
        if vlo <= 0.0:
            return lo
        if vhi <= 0.0:
            # On a monotone segment with positive left endpoint and nonpositive
            # right endpoint, the first root is unique.
            a, b = lo, hi
            for _ in range(80):
                m = 0.5 * (a + b)
                if vel(m) > 0.0:
                    a = m
                else:
                    b = m
            return 0.5 * (a + b)
    return math.inf


def generalized_follower_motion(vf, af, action: float, times, tau: float, w: float):
    vf = np.asarray(vf, dtype=float)
    af = np.asarray(af, dtype=float)
    times = np.asarray(times, dtype=float)
    H = float(np.max(times))
    n = len(vf)
    P = np.empty((n, len(times)), dtype=float)
    V = np.empty_like(P)
    A = np.empty_like(P)
    stops = np.empty(n, dtype=float)
    for i in range(n):
        ts = first_stop_time(float(vf[i]), float(af[i]), float(action), tau, w, H)
        stops[i] = ts
        T = np.minimum(times, ts) if math.isfinite(ts) else times
        P[i, :] = raw_position(vf[i], af[i], action, tau, w, T)
        V[i, :] = raw_velocity(vf[i], af[i], action, tau, w, T)
        A[i, :] = raw_acceleration(af[i], action, tau, w, T)
        if math.isfinite(ts):
            stopped = times >= ts
            V[i, stopped] = 0.0
            A[i, stopped] = 0.0
    return P, V, A, stops


def canonical_pending_descriptor(age, bar_a, bar_u, q: int, R: int, Ts: float, J: float, b0, p3a, p2b):
    """Canonical explicit pending descriptor for the diagnostic countdown.

    The pending message is generated (R-q)*Ts before the current sample.
    Its acceleration content is the existing certified lower predictor at that
    generation instant, reconstructed from the adopted tuple.  The held-command
    component is kept at bar_u.  This makes chi explicit enough to test whether
    the legacy current-endpoint completion proxy changes the Bellman branch.

    This is a diagnostic canonical descriptor, not a complete deployment
    protocol model and not a continuous-domain certificate.
    """
    p_age = (R - q) * Ts
    elapsed = np.asarray(age, dtype=float) - p_age
    valid = elapsed >= -1e-12
    elapsed_clipped = np.maximum(elapsed, 0.0)
    a_pend = b0.predecessor_acceleration_lower(
        elapsed_clipped, np.asarray(bar_a, float), np.asarray(bar_u, float),
        J, p3a, p2b,
    )
    u_pend = np.asarray(bar_u, dtype=float).copy()
    return a_pend, u_pend, np.full_like(a_pend, p_age), valid


def lookup_future(cell_flat, q_index, index, valid):
    out = np.full(len(index), np.inf, dtype=float)
    m = np.asarray(valid, dtype=bool)
    out[m] = cell_flat[q_index][np.asarray(index, dtype=np.int64)[m]]
    return out


def one_step_requirement_for_action(
    *, i: int, q: int, action: float, data, h_flat, R: int, r3, b0,
    use_general_motion: bool, canonical_descriptor: bool,
):
    cfg, p1, p2b, p3a = data["eval_cfg"], data["p1"], data["p2b"], data["p3a"]
    vf, vp, af, bar_a, bar_u, age = [np.asarray(x, float) for x in data["eval_flat"]]
    Ts = float(data["Ts"])
    J = float(cfg["information_contract"]["slew_rate"])
    nt = int(cfg["one_step"]["trajectory_points"])
    times = np.linspace(0.0, Ts, nt)
    tau = float(p2b["follower"]["tau"])
    w = float(p1["uncertainty"]["follower_actuation_abs"])
    sd = p2b["state_domain"]
    speed_bound = float(sd["v_f_max"]) + float(sd["v_p_max"])
    lips = 0.5 * speed_bound * (Ts / (nt - 1))

    sl = slice(i, i + 1)
    ap_lower = b0.predecessor_acceleration_lower(
        age[sl], bar_a[sl], bar_u[sl], J, p3a, p2b
    )
    Pp, Vp, Ap, Up, _ = r3.predecessor_motion_with_stop(
        vp[sl], ap_lower, bar_u[sl], age[sl], times, J, p3a
    )
    if use_general_motion:
        Pf, Vf, Af, _ = generalized_follower_motion(
            vf[sl], af[sl], action, times, tau, w
        )
    else:
        Pf, Vf, Af = r3.follower_motion(
            vf[sl], af[sl], action, times, p1, p2b
        )
    closing = Pf - Pp
    step_loss = float(max(np.max(closing, axis=1)[0] + lips, 0.0))
    closing_end = float(Pf[0, -1] - Pp[0, -1])
    axes = data["lookup_axes"]
    defer_vals = [
        Vf[:, -1], Vp[:, -1], Af[:, -1], bar_a[sl], bar_u[sl], age[sl] + Ts
    ]
    didx, dvalid = r3.locate_cells(axes, defer_vals)

    if canonical_descriptor:
        pa, pu, pending_age, pvalid = canonical_pending_descriptor(
            age[sl], bar_a[sl], bar_u[sl], q, R, Ts, J, b0, p3a, p2b
        )
        completion_age = pending_age + Ts
        adopt_vals = [
            Vf[:, -1], Vp[:, -1], Af[:, -1], pa, pu, completion_age
        ]
        cidx, cvalid = r3.locate_cells(axes, adopt_vals)
        cvalid = cvalid & pvalid
        descriptor_bar_a = float(pa[0])
        descriptor_bar_u = float(pu[0])
        descriptor_age = float(completion_age[0])
    else:
        adopt_vals = [
            Vf[:, -1], Vp[:, -1], Af[:, -1], Ap[:, -1], Up[:, -1],
            np.full(1, (R - q + 1) * Ts, dtype=float),
        ]
        cidx, cvalid = r3.locate_cells(axes, adopt_vals)
        descriptor_bar_a = float(Ap[0, -1])
        descriptor_bar_u = float(Up[0, -1])
        descriptor_age = float((R - q + 1) * Ts)

    h = h_flat.reshape(data["lookup_shape"] + (R,))
    cell = r3.cell_corner_max(h)
    cell_flat = [cell[..., qq].reshape(-1) for qq in range(R)]
    hold = lookup_future(cell_flat, R - 1, didx, dvalid)
    adopt = lookup_future(cell_flat, R - 1, cidx, cvalid)
    completion = np.minimum(adopt, hold)
    future = completion
    if q > 1:
        defer = lookup_future(cell_flat, q - 2, didx, dvalid)
        future = np.maximum(completion, defer)
    req = max(step_loss, closing_end + float(future[0]))
    req = max(req, 0.0)
    projected = min(float(data["eval_fallback"][i]), req)
    return {
        "valid": bool(dvalid[0] and cvalid[0] and math.isfinite(projected)),
        "projected_required_gap_m": float(projected),
        "raw_required_gap_m": float(req),
        "step_loss_upper_m": step_loss,
        "closing_end_m": closing_end,
        "adopt_index": int(cidx[0]),
        "defer_index": int(didx[0]),
        "descriptor_bar_a": descriptor_bar_a,
        "descriptor_bar_u": descriptor_bar_u,
        "descriptor_age": descriptor_age,
    }


def self_test() -> None:
    # Positive-equilibrium case: acceleration starts negative and recovers;
    # generalized logic must not throw and must preserve nonnegative speed after stop.
    times = np.linspace(0.0, 0.1, 9)
    P, V, A, stops = generalized_follower_motion(
        np.asarray([5.0, 0.01]), np.asarray([-1.0, -3.0]), 2.5, times, 0.4, 0.2
    )
    assert P.shape == V.shape == A.shape == (2, 9)
    assert np.all(V >= -1e-12)
    assert len(stops) == 2
    # Pure braking case must produce a finite stop if horizon is long enough.
    t = first_stop_time(1.0, -1.0, -6.0, 0.4, 0.2, 2.0)
    assert math.isfinite(t) and 0.0 < t < 2.0
    print("R9E_INTERNAL_SELF_TEST=PASS", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--chunk-size", type=int, default=4096)
    args = ap.parse_args()
    self_test()
    if args.self_test:
        return 0

    sys.path.insert(0, str(SRC))
    import p3b1_augmented_fixed_point_v1_r3 as r3
    import p3b0_freshness_service_audit_v1 as b0
    import p3b1_r9d_timestamped_service as r9d

    latest = RESULTS / "P3B1_R9D_LATEST.json"
    d = json.loads(latest.read_text(encoding="utf-8"))
    if d.get("status") != "PASS":
        raise RuntimeError("R9E_UPSTREAM_R9D_NOT_PASS")
    if d.get("classification") != "Q_LAYER_PERSISTS_ON_TIMESTAMPED_HISTORY_ALIGNED_GRID_PENDING_CONTENT_UNRESOLVED":
        raise RuntimeError("R9E_UNEXPECTED_R9D_CLASSIFICATION=" + str(d.get("classification")))

    data = r9d.reconstruct_timestamped_geometry(r3, __import__("p3b1_r5_refinement_attribution"), __import__("p3b1_r6_continuation_halo"), __import__("p3b1_r7_frozen_halo_fixed_point"), __import__("p3b1_r9c_targeted_history_gate"))
    profile = d.get("profile", "diagnostic_fast")
    profiles = data["p1"]["diagnostic_service_profiles"]
    R = max(1, int(profiles[profile]["eligible_bound_steps"]))
    Ts = float(data["Ts"])

    print("=== P3-B1-R9-E SERVICE-CONTRACT + GENERAL-ACTION AUDIT ===", flush=True)
    print(f"PROFILE={profile} R={R} Ts={Ts:.12g}", flush=True)
    print("THEORY_Q_STATE_REQUIRED=(stage,timing,pending_descriptor)", flush=True)
    print("R9D_PENDING_CONTENT_OMITTED=YES", flush=True)
    print("R9E_PENDING_DESCRIPTOR_MODEL=CANONICAL_HELD_PREDICTOR_LOWER_BOUND", flush=True)
    print("R9E_PENDING_DESCRIPTOR_FULL_PROTOCOL_CERTIFIED=NO", flush=True)
    print("R9E_GENERAL_ACTION_PROPAGATOR=STOP_EVENT_IF_AND_ONLY_IF_ZERO_SPEED_REACHED", flush=True)
    print("CONTINUOUS_STATE_SEPARATION_CERTIFIED=NO", flush=True)

    frozen_actions = [-3.0, -2.0, -1.0]
    print("R9E_STAGE_START=recompute_r9d_timestamped_continuation", flush=True)
    cfg_frozen, td = r9d.build_timestamped_transitions(
        data, frozen_actions, R, r3, b0, chunk_size=args.chunk_size
    )
    sol = r9d.solve_timestamped_causal(
        R, cfg_frozen, data["eval_idx"], data["lookup_shape"],
        data["lookup_fallback"], data["eval_fallback"], td, r3,
        label="r9e_timestamped_continuation",
    )
    if not sol.converged:
        raise RuntimeError("R9E_BASELINE_CONTINUATION_NOT_CONVERGED")
    semantic_mask, _ = r9d.timestamped_semantic_mask(
        np.asarray(data["eval_flat"][-1], float), R, Ts
    )
    widx, spans = r9d.qdep_indices(sol.h_flat, data["eval_idx"], semantic_mask)
    if len(widx) != int(d["metrics"]["frozen_semantic"]["qdep_nodes"]):
        raise RuntimeError(
            f"R9E_R9D_WITNESS_REPRO_FAIL expected={d['metrics']['frozen_semantic']['qdep_nodes']} actual={len(widx)}"
        )
    print(f"R9E_R9D_WITNESS_REPRO=PASS count={len(widx)}", flush=True)

    # Regression: new general propagator must exactly/closely reproduce legacy
    # braking actions on the witness states before it is used outside that set.
    vf, vp, af, ba, bu, age = [np.asarray(x, float) for x in data["eval_flat"]]
    tau = float(data["p2b"]["follower"]["tau"])
    w = float(data["p1"]["uncertainty"]["follower_actuation_abs"])
    times = np.linspace(0.0, Ts, int(data["eval_cfg"]["one_step"]["trajectory_points"]))
    sample = widx[: min(44, len(widx))]
    max_reg = 0.0
    for action in frozen_actions:
        P0, V0, A0 = r3.follower_motion(vf[sample], af[sample], action, times, data["p1"], data["p2b"])
        P1, V1, A1, _ = generalized_follower_motion(vf[sample], af[sample], action, times, tau, w)
        max_reg = max(max_reg, float(np.max(np.abs(P0-P1))), float(np.max(np.abs(V0-V1))), float(np.max(np.abs(A0-A1))))
    regression_pass = max_reg <= 5e-10
    print(f"R9E_GENERAL_PROPAGATOR_REGRESSION pass={regression_pass} max_error={max_reg:.3e}", flush=True)
    if not regression_pass:
        raise RuntimeError("R9E_GENERAL_PROPAGATOR_REGRESSION_FAIL")

    descriptor_rows = []
    action_rows = []
    descriptor_material = 0
    descriptor_cell_change = 0
    max_descriptor_req_delta = 0.0
    max_descriptor_a_shift = 0.0
    action_material = 0
    max_action_improvement = 0.0
    invalid_general_actions = 0
    action_grid = sorted(set([
        -6.0, -5.0, -4.0, -3.0, -2.5, -2.0, -1.5, -1.0,
        -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5,
    ]))

    print("R9E_STAGE_START=pending_descriptor_and_action_witness_audit", flush=True)
    for c, i in enumerate(widx, start=1):
        for q in range(1, R + 1):
            if not semantic_mask[i, q-1]:
                continue
            legacy_best = math.inf
            canonical_best = math.inf
            baseline_canonical_best = math.inf
            best_action = None
            for action in action_grid:
                try:
                    can = one_step_requirement_for_action(
                        i=int(i), q=q, action=float(action), data=data,
                        h_flat=sol.h_flat, R=R, r3=r3, b0=b0,
                        use_general_motion=True, canonical_descriptor=True,
                    )
                except Exception:
                    can = {"valid": False}
                if not can.get("valid", False):
                    invalid_general_actions += 1
                    action_rows.append({
                        "node_index": int(i), "q": q, "action": action,
                        "valid": False,
                    })
                    continue
                canonical_best = min(canonical_best, can["projected_required_gap_m"])
                if action in frozen_actions:
                    baseline_canonical_best = min(baseline_canonical_best, can["projected_required_gap_m"])
                if best_action is None or can["projected_required_gap_m"] < best_action[1]:
                    best_action = (action, can["projected_required_gap_m"])
                action_rows.append({
                    "node_index": int(i), "q": q, "action": action,
                    "valid": True,
                    "canonical_required_gap_m": can["projected_required_gap_m"],
                })

            # Compare old current-endpoint proxy and explicit canonical pending
            # descriptor on the frozen three-action policy class only.
            old_candidates, can_candidates = [], []
            for action in frozen_actions:
                old = one_step_requirement_for_action(
                    i=int(i), q=q, action=action, data=data,
                    h_flat=sol.h_flat, R=R, r3=r3, b0=b0,
                    use_general_motion=False, canonical_descriptor=False,
                )
                can = one_step_requirement_for_action(
                    i=int(i), q=q, action=action, data=data,
                    h_flat=sol.h_flat, R=R, r3=r3, b0=b0,
                    use_general_motion=True, canonical_descriptor=True,
                )
                if old["valid"]:
                    old_candidates.append(old)
                if can["valid"]:
                    can_candidates.append(can)
            old_best = min((x["projected_required_gap_m"] for x in old_candidates), default=math.inf)
            can_best = min((x["projected_required_gap_m"] for x in can_candidates), default=math.inf)
            req_delta = can_best - old_best if math.isfinite(old_best) and math.isfinite(can_best) else math.nan
            if math.isfinite(req_delta):
                max_descriptor_req_delta = max(max_descriptor_req_delta, abs(req_delta))
                if abs(req_delta) > TOL:
                    descriptor_material += 1
            if old_candidates and can_candidates:
                # compare the best representatives' adoption cell and message a
                o = min(old_candidates, key=lambda x: x["projected_required_gap_m"])
                ca = min(can_candidates, key=lambda x: x["projected_required_gap_m"])
                if o["adopt_index"] != ca["adopt_index"]:
                    descriptor_cell_change += 1
                max_descriptor_a_shift = max(
                    max_descriptor_a_shift,
                    abs(o["descriptor_bar_a"] - ca["descriptor_bar_a"]),
                )
                descriptor_rows.append({
                    "node_index": int(i), "q": q,
                    "v_f": float(vf[i]), "v_p": float(vp[i]), "a_f": float(af[i]),
                    "bar_a": float(ba[i]), "bar_u": float(bu[i]), "age": float(age[i]),
                    "q_span_r9d_m": float(spans[i]),
                    "legacy_best_gap_m": old_best,
                    "canonical_best_gap_m": can_best,
                    "canonical_minus_legacy_m": req_delta,
                    "legacy_adopt_index": int(o["adopt_index"]),
                    "canonical_adopt_index": int(ca["adopt_index"]),
                    "legacy_descriptor_bar_a": o["descriptor_bar_a"],
                    "canonical_descriptor_bar_a": ca["descriptor_bar_a"],
                    "canonical_descriptor_age": ca["descriptor_age"],
                    "coarse_full_action_best_gap_m": canonical_best,
                    "frozen_action_best_gap_m": baseline_canonical_best,
                    "coarse_best_action": best_action[0] if best_action else "",
                })
            if math.isfinite(baseline_canonical_best) and math.isfinite(canonical_best):
                improve = baseline_canonical_best - canonical_best
                max_action_improvement = max(max_action_improvement, improve)
                if improve > TOL:
                    action_material += 1
        if c == 1 or c % 8 == 0 or c == len(widx):
            print(f"R9E_PROGRESS witnesses={c}/{len(widx)}", flush=True)

    if descriptor_material > 0 or descriptor_cell_change > 0:
        classification = "PENDING_DESCRIPTOR_MATERIAL_ON_TIMESTAMPED_Q_WITNESSES"
        next_action = "R9F_BUILD_SPARSE_AUGMENTED_SERVICE_GRAPH_WITH_EXPLICIT_CHI_AND_PSI"
    elif action_material > 0:
        classification = "COARSE_GENERAL_ACTION_CLASS_MATERIAL_ON_Q_WITNESSES"
        next_action = "R9F_INTERVAL_ACTION_COVER_PLUS_EXPLICIT_SERVICE_GRAPH"
    else:
        classification = "PENDING_DESCRIPTOR_CANONICAL_AUDIT_AND_COARSE_ACTION_DISCOVERY_NONMATERIAL"
        next_action = "R9F_FULL_EXPLICIT_SERVICE_GRAPH_AND_INTERVAL_ACTION_CERTIFICATE"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_json = RESULTS / f"P3B1_R9E_RESULT_{stamp}.json"
    latest_json = RESULTS / "P3B1_R9E_LATEST.json"
    descriptor_csv = RESULTS / f"P3B1_R9E_PENDING_DESCRIPTOR_{stamp}.csv"
    action_csv = RESULTS / f"P3B1_R9E_GENERAL_ACTION_DISCOVERY_{stamp}.csv"
    manifest = RESULTS / f"P3B1_R9E_MANIFEST_{stamp}.sha256"
    write_csv(descriptor_csv, descriptor_rows)
    write_csv(action_csv, action_rows)

    out = {
        "schema": "SCV_P3B1_R9E_SERVICE_CONTRACT_GENERAL_ACTION_V1",
        "status": "PASS",
        "timestamp_utc": stamp,
        "classification": classification,
        "next_action": next_action,
        "continuous_state_separation_certified": False,
        "theory_status": {
            "viability_fixed_point_definition_retained": True,
            "numerical_markov_sufficiency_closed": False,
            "pending_descriptor_model": "canonical held-predictor lower-bound diagnostic only",
            "post_output_bookkeeping_Psi_explicit": False,
        },
        "metrics": {
            "r9d_witness_count": int(len(widx)),
            "descriptor_material_state_q_pairs": int(descriptor_material),
            "descriptor_adoption_cell_changes": int(descriptor_cell_change),
            "max_descriptor_required_gap_change_m": float(max_descriptor_req_delta),
            "max_descriptor_bar_a_shift": float(max_descriptor_a_shift),
            "coarse_general_action_material_state_q_pairs": int(action_material),
            "max_coarse_action_improvement_m": float(max_action_improvement),
            "invalid_general_action_trials": int(invalid_general_actions),
            "general_propagator_regression_max_error": float(max_reg),
        },
        "gates": {
            "upstream_r9d_pass": True,
            "r9d_witness_reproduction": int(len(widx)) == int(d["metrics"]["frozen_semantic"]["qdep_nodes"]),
            "general_propagator_regression": bool(regression_pass),
            "pending_content_full_protocol_certificate": False,
            "post_output_service_bookkeeping_explicit": False,
        },
        "artifacts": {
            "pending_descriptor_csv": str(descriptor_csv),
            "general_action_csv": str(action_csv),
        },
    }
    text = json.dumps(out, indent=2, sort_keys=True)
    atomic_write(result_json, text)
    atomic_write(latest_json, text)
    files = [Path(__file__), result_json, descriptor_csv, action_csv, latest]
    atomic_write(manifest, "".join(f"{sha256_file(p)}  {p}\n" for p in files if p.exists()))

    print("=== R9-E DECISION ===", flush=True)
    print(f"R9E_PENDING_DESCRIPTOR material_pairs={descriptor_material} cell_changes={descriptor_cell_change} max_gap_change_m={max_descriptor_req_delta:.12g} max_bar_a_shift={max_descriptor_a_shift:.12g}", flush=True)
    print(f"R9E_GENERAL_ACTION material_pairs={action_material} max_improvement_m={max_action_improvement:.12g} invalid_trials={invalid_general_actions}", flush=True)
    print("R9E_EXECUTION=PASS", flush=True)
    print(f"R9E_CLASSIFICATION={classification}", flush=True)
    print("CONTINUOUS_STATE_SEPARATION_CERTIFIED=NO", flush=True)
    print("MARKOV_SUFFICIENCY_CERTIFIED=NO", flush=True)
    print(f"R9E_NEXT_ACTION={next_action}", flush=True)
    print(f"RESULT_JSON={result_json}", flush=True)
    print(f"MANIFEST={manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
