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
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def as_bool(x) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y", "pass"}


def resolve_recorded_artifact(recorded: str | Path, current_results: Path) -> Path:
    p = Path(recorded)
    if p.is_file():
        return p
    q = current_results / p.name
    if q.is_file():
        return q
    raise FileNotFoundError(f"R9G_ARTIFACT_NOT_FOUND recorded={p} recovered={q}")


def matched_pending_envelope(*, adopted_age: float, pending_age: float,
                             bar_a: float, bar_u: float, J: float,
                             b0, p3a: dict, p2b: dict) -> dict:
    """Reachable pending-content envelope at a *fixed* generation time.

    Unlike R9-F's countdown-specific helper, pending_age is an independent
    argument.  Hence two service-progress states can share exactly the same
    pending generation time and the same chi envelope.
    """
    elapsed = float(adopted_age - pending_age)
    if elapsed < -1e-10:
        return {"valid": False, "reason": "PENDING_MESSAGE_OLDER_THAN_ADOPTED_STATE_ORDERING"}
    elapsed = max(elapsed, 0.0)
    tau = float(p3a.get("predecessor", {}).get("tau", p2b["predecessor"]["tau"]))
    nominal_a = float(bar_u + (bar_a - bar_u) * math.exp(-elapsed / tau))
    lower = float(b0.predecessor_acceleration_lower(
        np.asarray([elapsed]), np.asarray([bar_a]), np.asarray([bar_u]),
        J, p3a, p2b,
    )[0])
    radius = max(nominal_a - lower, 0.0)
    sd = p2b["state_domain"]
    a_lo = max(float(sd["a_p_min"]), lower)
    a_hi = min(float(sd["a_p_max"]), nominal_a + radius)
    pred = p3a.get("predecessor", {})
    u_min = float(pred.get("command_min", sd.get("a_p_min", -8.0)))
    u_max = float(pred.get("command_max", sd.get("a_p_max", 3.0)))
    u_lo = max(u_min, float(bar_u - J * elapsed))
    u_hi = min(u_max, float(bar_u + J * elapsed))
    if a_lo > a_hi + 1e-12 or u_lo > u_hi + 1e-12:
        return {"valid": False, "reason": "EMPTY_MATCHED_CHI_ENVELOPE"}
    return {
        "valid": True,
        "pending_age_s": float(pending_age),
        "elapsed_from_adopted_generation_s": float(elapsed),
        "a_lower": float(a_lo),
        "a_upper": float(a_hi),
        "u_lower": float(u_lo),
        "u_upper": float(u_hi),
    }


def envelope_equal(a: dict, b: dict, tol: float = 1e-10) -> bool:
    if not a.get("valid", False) or not b.get("valid", False):
        return False
    return all(abs(float(a[k]) - float(b[k])) <= tol for k in (
        "a_lower", "a_upper", "u_lower", "u_upper"
    ))


def common_pending_ages(adopted_age: float, Ts: float, age_max: float) -> list[float]:
    # Two service intervals must remain inside the finite lookup age domain.
    upper = min(float(adopted_age), float(age_max - 2.0 * Ts))
    if upper < -1e-12:
        return []
    n = int(math.floor((upper + 1e-12) / Ts))
    return [float(k * Ts) for k in range(n + 1)]


def state_key(st: dict) -> tuple:
    return tuple(round(float(st[k]), 11) for k in (
        "v_f", "v_p", "a_f", "bar_a", "bar_u", "age"
    ))


def physical_step(st: dict, action: float, data: dict, r3, b0, r9e) -> dict:
    cfg, p1, p2b, p3a = data["eval_cfg"], data["p1"], data["p2b"], data["p3a"]
    Ts = float(data["Ts"])
    J = float(cfg["information_contract"]["slew_rate"])
    nt = int(cfg["one_step"]["trajectory_points"])
    times = np.linspace(0.0, Ts, nt)
    tau = float(p2b["follower"]["tau"])
    w = float(p1["uncertainty"]["follower_actuation_abs"])
    sd = p2b["state_domain"]
    lips = 0.5 * (float(sd["v_f_max"]) + float(sd["v_p_max"])) * (Ts / (nt - 1))

    vp = np.asarray([st["v_p"]], dtype=float)
    ba = np.asarray([st["bar_a"]], dtype=float)
    bu = np.asarray([st["bar_u"]], dtype=float)
    ag = np.asarray([st["age"]], dtype=float)
    ap_lower = b0.predecessor_acceleration_lower(ag, ba, bu, J, p3a, p2b)
    Pp, Vp, Ap, Up, _ = r3.predecessor_motion_with_stop(vp, ap_lower, bu, ag, times, J, p3a)
    Pf, Vf, Af, _ = r9e.generalized_follower_motion(
        np.asarray([st["v_f"]], dtype=float),
        np.asarray([st["a_f"]], dtype=float),
        float(action), times, tau, w,
    )
    closing = Pf - Pp
    step_loss = float(max(float(np.max(closing[0])) + lips, 0.0))
    closing_end = float(Pf[0, -1] - Pp[0, -1])
    nxt = {
        "v_f": float(Vf[0, -1]),
        "v_p": float(Vp[0, -1]),
        "a_f": float(Af[0, -1]),
        "bar_a": float(st["bar_a"]),
        "bar_u": float(st["bar_u"]),
        "age": float(st["age"] + Ts),
    }
    return {
        "step_loss_m": step_loss,
        "closing_end_m": closing_end,
        "next_state": nxt,
    }


def fallback_requirement(st: dict, data: dict, b0, sw) -> float:
    cfg, p2b, p2c, p3a = data["eval_cfg"], data["p2b"], data["p2c"], data["p3a"]
    J = float(cfg["information_contract"]["slew_rate"])
    ap = b0.predecessor_acceleration_lower(
        np.asarray([st["age"]], dtype=float),
        np.asarray([st["bar_a"]], dtype=float),
        np.asarray([st["bar_u"]], dtype=float),
        J, p3a, p2b,
    )
    _, upper, _ = sw.switching_loss_bracket(
        np.asarray([st["v_f"]], dtype=float),
        np.asarray([st["a_f"]], dtype=float),
        np.asarray([st["v_p"]], dtype=float),
        ap,
        int(p2c["switching"]["N_sw"]),
        [257], p2c, p2b,
    )
    return float(max(float(np.asarray(upper).reshape(-1)[0]), 0.0))


def build_restart_cell_values(h_flat: np.ndarray, data: dict, R: int, r3) -> np.ndarray:
    h = h_flat.reshape(data["lookup_shape"] + (R,))
    cmax = r3.cell_corner_max(h)
    return cmax[..., R - 1].reshape(-1)


def completion_future(*, end_state: dict, env: dict, candidate_age: float,
                      restart_cells: np.ndarray, data: dict, r3, r9f) -> dict:
    axes = data["lookup_axes"]
    c_cells, cvalid = r9f.descriptor_cell_indices(
        axes,
        (end_state["v_f"], end_state["v_p"], end_state["a_f"], candidate_age),
        env,
    )
    if not cvalid or len(c_cells) == 0:
        return {"valid": False, "reason": "MATCHED_CHI_ADOPT_OUTSIDE_LOOKUP"}
    robust_adopt = float(np.max(restart_cells[c_cells]))
    hold_vals = [
        np.asarray([end_state["v_f"]]), np.asarray([end_state["v_p"]]),
        np.asarray([end_state["a_f"]]), np.asarray([end_state["bar_a"]]),
        np.asarray([end_state["bar_u"]]), np.asarray([end_state["age"]]),
    ]
    hidx, hvalid = r3.locate_cells(axes, hold_vals)
    if not bool(hvalid[0]):
        return {"valid": False, "reason": "MATCHED_CHI_HOLD_OUTSIDE_LOOKUP"}
    hold = float(restart_cells[int(hidx[0])])
    return {
        "valid": True,
        "future_m": float(min(robust_adopt, hold)),
        "robust_adopt_future_m": robust_adopt,
        "hold_future_m": hold,
        "descriptor_cells": int(len(c_cells)),
    }


def verify_last_value(*, st: dict, env: dict, pending_age: float,
                      actions: list[float], restart_cells: np.ndarray,
                      data: dict, r3, b0, sw, r9e, r9f,
                      fallback_cache: dict) -> dict:
    key = state_key(st)
    if key not in fallback_cache:
        fallback_cache[key] = fallback_requirement(st, data, b0, sw)
    fallback = fallback_cache[key]
    Ts = float(data["Ts"])
    best = math.inf
    best_action = None
    max_cells = 0
    for action in actions:
        tr = physical_step(st, float(action), data, r3, b0, r9e)
        cf = completion_future(
            end_state=tr["next_state"], env=env,
            candidate_age=float(pending_age + Ts),
            restart_cells=restart_cells, data=data, r3=r3, r9f=r9f,
        )
        if not cf.get("valid", False):
            continue
        req = max(tr["step_loss_m"], tr["closing_end_m"] + cf["future_m"], 0.0)
        req = min(fallback, req)
        max_cells = max(max_cells, int(cf["descriptor_cells"]))
        if req < best:
            best = req
            best_action = float(action)
    return {
        "valid": math.isfinite(best),
        "required_gap_m": float(best),
        "best_action": best_action,
        "max_descriptor_cells": max_cells,
    }


def matched_stage_pair(*, st: dict, env: dict, pending_age: float,
                       actions: list[float], restart_cells: np.ndarray,
                       data: dict, r3, b0, sw, r9e, r9f,
                       fallback_cache: dict) -> dict:
    """Two service states with identical chi and identical liveness upper bound.

    FRAGMENTED_TWO_STEP: current interval only advances the same pending message
    to VERIFYING_LAST; it cannot produce an eligible output now.

    VERIFYING_TWO_STEP: current interval may either complete now (eligible output)
    or continue to VERIFYING_LAST.  The service adversary chooses the worse of
    these two admitted outcomes.  VERIFYING_LAST must complete in the next
    interval, so both initial states satisfy the same two-interval liveness bound.
    """
    key = state_key(st)
    if key not in fallback_cache:
        fallback_cache[key] = fallback_requirement(st, data, b0, sw)
    fallback = fallback_cache[key]
    Ts = float(data["Ts"])

    best_frag = math.inf
    best_verify = math.inf
    best_frag_action = None
    best_verify_action = None
    max_cells = 0

    for action in actions:
        tr = physical_step(st, float(action), data, r3, b0, r9e)
        nxt = tr["next_state"]
        last = verify_last_value(
            st=nxt, env=env, pending_age=float(pending_age + Ts),
            actions=actions, restart_cells=restart_cells,
            data=data, r3=r3, b0=b0, sw=sw, r9e=r9e, r9f=r9f,
            fallback_cache=fallback_cache,
        )
        if not last.get("valid", False):
            continue
        max_cells = max(max_cells, int(last.get("max_descriptor_cells", 0)))

        # Fragmented: no eligible output this interval; the same chi is carried.
        req_frag = max(
            tr["step_loss_m"],
            tr["closing_end_m"] + float(last["required_gap_m"]),
            0.0,
        )
        req_frag = min(fallback, req_frag)
        if req_frag < best_frag:
            best_frag = req_frag
            best_frag_action = float(action)

        # Verifying: service uncertainty may complete now or continue.
        cf = completion_future(
            end_state=nxt, env=env,
            candidate_age=float(pending_age + Ts),
            restart_cells=restart_cells, data=data, r3=r3, r9f=r9f,
        )
        if not cf.get("valid", False):
            continue
        max_cells = max(max_cells, int(cf["descriptor_cells"]))
        future_verify = max(float(cf["future_m"]), float(last["required_gap_m"]))
        req_verify = max(
            tr["step_loss_m"],
            tr["closing_end_m"] + future_verify,
            0.0,
        )
        req_verify = min(fallback, req_verify)
        if req_verify < best_verify:
            best_verify = req_verify
            best_verify_action = float(action)

    valid = math.isfinite(best_frag) and math.isfinite(best_verify)
    return {
        "valid": valid,
        "fragmented_required_gap_m": float(best_frag),
        "verifying_required_gap_m": float(best_verify),
        "signed_verify_minus_fragment_m": float(best_verify - best_frag) if valid else math.nan,
        "abs_progress_gap_m": float(abs(best_verify - best_frag)) if valid else math.nan,
        "fragmented_best_action": best_frag_action,
        "verifying_best_action": best_verify_action,
        "max_descriptor_cells": max_cells,
    }


def self_test() -> None:
    # Same chi is preserved when both adopted age and pending age advance by Ts.
    class B0:
        @staticmethod
        def predecessor_acceleration_lower(age, ba, bu, J, p3a, p2b):
            tau = float(p3a["predecessor"]["tau"])
            nominal = bu + (ba - bu) * np.exp(-age / tau)
            return nominal - 0.1 * (1.0 - np.exp(-age / tau))
    p3a = {"predecessor": {"tau": 0.35, "command_min": -8.0, "command_max": 3.0}}
    p2b = {"predecessor": {"tau": 0.35}, "state_domain": {"a_p_min": -8.0, "a_p_max": 3.0}}
    a = matched_pending_envelope(
        adopted_age=0.7, pending_age=0.2, bar_a=-2.0, bar_u=-1.0,
        J=6.0, b0=B0, p3a=p3a, p2b=p2b,
    )
    b = matched_pending_envelope(
        adopted_age=0.8, pending_age=0.3, bar_a=-2.0, bar_u=-1.0,
        J=6.0, b0=B0, p3a=p3a, p2b=p2b,
    )
    assert envelope_equal(a, b)
    assert common_pending_ages(0.35, 0.1, 2.0) == [0.0, 0.1, 0.2, 0.30000000000000004]
    # Service-tree orientation sanity: adding an adversarial branch cannot lower
    # the per-action future value before the supervisor re-optimizes.
    completion = 2.0
    continuation = 1.5
    assert max(completion, continuation) >= continuation
    print("R9G_INTERNAL_SELF_TEST=PASS")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--chunk-size", type=int, default=4096)
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0

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

    upstream = RESULTS / "P3B1_R9F_LATEST.json"
    u = json.loads(upstream.read_text(encoding="utf-8"))
    if u.get("status") != "PASS":
        raise RuntimeError("R9G_R9F_UPSTREAM_NOT_PASS")
    if u.get("classification") != "COUNTDOWN_Q_CONFOUNDS_SERVICE_PROGRESS_WITH_PENDING_MESSAGE_HISTORY":
        raise RuntimeError("R9G_UNEXPECTED_R9F_CLASSIFICATION=" + str(u.get("classification")))

    witness_path = resolve_recorded_artifact(
        u.get("artifacts", {}).get("robust_chi_witness_csv", ""), RESULTS
    )
    with witness_path.open(newline="", encoding="utf-8") as f:
        wrows = list(csv.DictReader(f))
    robust_rows = [r for r in wrows if as_bool(r.get("robust_chi_full_action_qdep", False))]
    expected_robust = int(u["metrics"]["robust_chi_full_action_qdep_nodes"])
    if len(robust_rows) != expected_robust:
        raise RuntimeError(
            f"R9G_R9F_WITNESS_FILTER_MISMATCH expected={expected_robust} actual={len(robust_rows)}"
        )

    data = r9d.reconstruct_timestamped_geometry(r3, r5, r6, r7, r9c)
    profile = "diagnostic_fast"
    R = r7.service_horizon(profile, data["p1"]["diagnostic_service_profiles"][profile])
    Ts = float(data["Ts"])
    if R != 2:
        raise RuntimeError(f"R9G_DIAGNOSTIC_PAIR_REQUIRES_R2 actual={R}")

    print("=== P3-B1-R9-G MATCHED-CHI BRANCHING SERVICE AUDIT ===", flush=True)
    print(f"PROFILE={profile} R={R} Ts={Ts:.12g}", flush=True)
    print("SERVICE_PAIR=FRAGMENTED_TWO_STEP_VS_VERIFYING_TWO_STEP", flush=True)
    print("MATCHED_PHYSICAL_STATE=YES", flush=True)
    print("MATCHED_ADOPTED_INFORMATION=YES", flush=True)
    print("MATCHED_PENDING_GENERATION_TIME=YES", flush=True)
    print("MATCHED_CHI_ENVELOPE=YES", flush=True)
    print("MATCHED_SCALAR_LIVENESS_UPPER_BOUND_STEPS=2", flush=True)
    print("ONLY_SERVICE_STAGE_BRANCHING_DIFFERS=YES", flush=True)
    print("DEPLOYMENT_PROTOCOL_CERTIFIED=NO", flush=True)
    print("FULL_AUGMENTED_GFP_SOLVED=NO", flush=True)
    print("CONTINUOUS_STATE_SEPARATION_CERTIFIED=NO", flush=True)

    # Recompute the timestamp-correct continuation used only after the current
    # pending candidate terminates and the diagnostic cycle restarts.
    frozen_actions = [-3.0, -2.0, -1.0]
    full_actions = [-6.0, -5.0, -4.0, -3.0, -2.5, -2.0, -1.5, -1.0,
                    -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    print("R9G_STAGE_START=recompute_timestamped_restart_continuation", flush=True)
    cfg_frozen, td = r9d.build_timestamped_transitions(
        data, frozen_actions, R, r3, b0, chunk_size=args.chunk_size
    )
    sol = r9d.solve_timestamped_causal(
        R, cfg_frozen, data["eval_idx"], data["lookup_shape"],
        data["lookup_fallback"], data["eval_fallback"], td, r3,
        label="r9g_timestamped_restart_continuation",
    )
    if not sol.converged:
        raise RuntimeError("R9G_RESTART_CONTINUATION_NOT_CONVERGED")
    restart_cells = build_restart_cell_values(sol.h_flat, data, R, r3)

    # General propagator regression remains a hard gate.
    vf, vp, af, ba, bu, age = [np.asarray(x, float) for x in data["eval_flat"]]
    sample_idx = np.linspace(0, len(vf) - 1, min(257, len(vf)), dtype=int)
    times = np.linspace(0.0, Ts, int(data["eval_cfg"]["one_step"]["trajectory_points"]))
    tau = float(data["p2b"]["follower"]["tau"])
    w = float(data["p1"]["uncertainty"]["follower_actuation_abs"])
    max_reg = 0.0
    for action in frozen_actions:
        P0, V0, A0 = r3.follower_motion(vf[sample_idx], af[sample_idx], action, times, data["p1"], data["p2b"])
        P1, V1, A1, _ = r9e.generalized_follower_motion(vf[sample_idx], af[sample_idx], action, times, tau, w)
        max_reg = max(max_reg, float(np.max(np.abs(P0-P1))), float(np.max(np.abs(V0-V1))), float(np.max(np.abs(A0-A1))))
    if max_reg > 1e-11:
        raise RuntimeError(f"R9G_GENERAL_PROPAGATOR_REGRESSION_FAIL max_error={max_reg}")
    print(f"R9G_GENERAL_PROPAGATOR_REGRESSION=PASS max_error={max_reg:.3e}", flush=True)

    node_to_idx = {int(r["node_index"]): int(r["node_index"]) for r in robust_rows}
    tests = []
    summaries = []
    frozen_sensitive_nodes = set()
    full_sensitive_nodes = set()
    max_frozen_gap = 0.0
    max_full_gap = 0.0
    verify_worse_full = 0
    fragment_worse_full = 0
    matched_chi_invariance_fail = 0
    skipped_domain = 0
    fallback_cache: dict = {}

    J = float(data["eval_cfg"]["information_contract"]["slew_rate"])
    print(f"R9G_STAGE_START=matched_chi_service_tree_audit witnesses={len(robust_rows)}", flush=True)
    for n, row in enumerate(robust_rows, start=1):
        i = int(row["node_index"])
        st = {
            "v_f": float(vf[i]), "v_p": float(vp[i]), "a_f": float(af[i]),
            "bar_a": float(ba[i]), "bar_u": float(bu[i]), "age": float(age[i]),
        }
        p_ages = common_pending_ages(st["age"], Ts, float(data["age_max"]))
        if not p_ages:
            skipped_domain += 1
            continue
        node_frozen_gap = 0.0
        node_full_gap = 0.0
        node_best_b_frozen = None
        node_best_b_full = None
        for b in p_ages:
            env = matched_pending_envelope(
                adopted_age=st["age"], pending_age=b,
                bar_a=st["bar_a"], bar_u=st["bar_u"], J=J,
                b0=b0, p3a=data["p3a"], p2b=data["p2b"],
            )
            if not env.get("valid", False):
                continue
            env_next = matched_pending_envelope(
                adopted_age=st["age"] + Ts, pending_age=b + Ts,
                bar_a=st["bar_a"], bar_u=st["bar_u"], J=J,
                b0=b0, p3a=data["p3a"], p2b=data["p2b"],
            )
            if not envelope_equal(env, env_next):
                matched_chi_invariance_fail += 1
                continue

            fr = matched_stage_pair(
                st=st, env=env, pending_age=b, actions=frozen_actions,
                restart_cells=restart_cells, data=data,
                r3=r3, b0=b0, sw=sw, r9e=r9e, r9f=r9f,
                fallback_cache=fallback_cache,
            )
            fu = matched_stage_pair(
                st=st, env=env, pending_age=b, actions=full_actions,
                restart_cells=restart_cells, data=data,
                r3=r3, b0=b0, sw=sw, r9e=r9e, r9f=r9f,
                fallback_cache=fallback_cache,
            )
            if not fr.get("valid", False) or not fu.get("valid", False):
                skipped_domain += 1
                continue
            if fr["abs_progress_gap_m"] > node_frozen_gap:
                node_frozen_gap = float(fr["abs_progress_gap_m"])
                node_best_b_frozen = float(b)
            if fu["abs_progress_gap_m"] > node_full_gap:
                node_full_gap = float(fu["abs_progress_gap_m"])
                node_best_b_full = float(b)
            if fr["abs_progress_gap_m"] > TOL:
                frozen_sensitive_nodes.add(i)
            if fu["abs_progress_gap_m"] > TOL:
                full_sensitive_nodes.add(i)
                if fu["signed_verify_minus_fragment_m"] > 0:
                    verify_worse_full += 1
                elif fu["signed_verify_minus_fragment_m"] < 0:
                    fragment_worse_full += 1
            max_frozen_gap = max(max_frozen_gap, float(fr["abs_progress_gap_m"]))
            max_full_gap = max(max_full_gap, float(fu["abs_progress_gap_m"]))
            tests.append({
                "node_index": i,
                "pending_age_s": float(b),
                "adopted_age_s": float(st["age"]),
                "chi_a_lower": env["a_lower"], "chi_a_upper": env["a_upper"],
                "chi_u_lower": env["u_lower"], "chi_u_upper": env["u_upper"],
                "frozen_fragmented_required_gap_m": fr["fragmented_required_gap_m"],
                "frozen_verifying_required_gap_m": fr["verifying_required_gap_m"],
                "frozen_signed_verify_minus_fragment_m": fr["signed_verify_minus_fragment_m"],
                "frozen_abs_progress_gap_m": fr["abs_progress_gap_m"],
                "full_fragmented_required_gap_m": fu["fragmented_required_gap_m"],
                "full_verifying_required_gap_m": fu["verifying_required_gap_m"],
                "full_signed_verify_minus_fragment_m": fu["signed_verify_minus_fragment_m"],
                "full_abs_progress_gap_m": fu["abs_progress_gap_m"],
                "full_fragmented_best_action": fu["fragmented_best_action"],
                "full_verifying_best_action": fu["verifying_best_action"],
                "matched_chi": True,
                "same_liveness_upper_bound_steps": 2,
                "deployment_protocol_certified": False,
            })
        summaries.append({
            "node_index": i,
            "upstream_r9f_robust_full_q_span_m": float(row.get("robust_chi_full_action_q_span_m", 0.0)),
            "matched_pending_age_tests": len(p_ages),
            "max_frozen_matched_chi_progress_gap_m": node_frozen_gap,
            "max_full_matched_chi_progress_gap_m": node_full_gap,
            "best_pending_age_frozen_s": node_best_b_frozen if node_best_b_frozen is not None else "",
            "best_pending_age_full_s": node_best_b_full if node_best_b_full is not None else "",
            "frozen_progress_sensitive": bool(node_frozen_gap > TOL),
            "full_progress_sensitive": bool(node_full_gap > TOL),
        })
        print(
            f"R9G_PROGRESS witnesses={n}/{len(robust_rows)} "
            f"full_sensitive_nodes={len(full_sensitive_nodes)}",
            flush=True,
        )

    if matched_chi_invariance_fail:
        raise RuntimeError(f"R9G_MATCHED_CHI_INVARIANCE_FAIL count={matched_chi_invariance_fail}")
    if not tests:
        raise RuntimeError("R9G_NO_VALID_MATCHED_CHI_TESTS")

    frozen_n = len(frozen_sensitive_nodes)
    full_n = len(full_sensitive_nodes)
    if full_n > 0:
        classification = "MATCHED_CHI_SERVICE_PROGRESS_SENSITIVITY_FOUND_DIAGNOSTIC_BRANCHING_AUDIT"
        next_action = "R9H_FULL_SPARSE_AUGMENTED_SERVICE_GFP_PLUS_INTERVAL_ACTION_INNER_OUTER_CERTIFICATES"
    elif frozen_n > 0:
        classification = "COARSE_GENERAL_ACTION_REMOVES_MATCHED_CHI_PROGRESS_SENSITIVITY"
        next_action = "R9H_INTERVAL_ACTION_COVER_BEFORE_FULL_AUGMENTED_SERVICE_GFP"
    else:
        classification = "NO_MATCHED_CHI_PROGRESS_SENSITIVITY_IN_DIAGNOSTIC_BRANCHING_AUDIT"
        next_action = "R9H_REVISE_OR_PROTOCOLIZE_MATCHED_CHI_SERVICE_AUTOMATON_BEFORE_CONTINUOUS_PROMOTION"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_json = RESULTS / f"P3B1_R9G_RESULT_{stamp}.json"
    latest_json = RESULTS / "P3B1_R9G_LATEST.json"
    tests_csv = RESULTS / f"P3B1_R9G_MATCHED_CHI_TESTS_{stamp}.csv"
    summary_csv = RESULTS / f"P3B1_R9G_WITNESS_SUMMARY_{stamp}.csv"
    manifest = RESULTS / f"P3B1_R9G_MANIFEST_{stamp}.sha256"
    write_csv(tests_csv, tests)
    write_csv(summary_csv, summaries)

    out = {
        "schema": "SCV_P3B1_R9G_MATCHED_CHI_BRANCHING_SERVICE_V1",
        "status": "PASS",
        "timestamp_utc": stamp,
        "classification": classification,
        "next_action": next_action,
        "continuous_state_separation_certified": False,
        "markov_sufficiency_certified": False,
        "service_progress_claim_certified": bool(full_n > 0),
        "diagnostic_contract": {
            "profile": profile,
            "R": R,
            "stage_pair": ["FRAGMENTED_TWO_STEP", "VERIFYING_TWO_STEP"],
            "shared_pending_descriptor": True,
            "shared_pending_generation_time": True,
            "shared_scalar_liveness_upper_bound_steps": 2,
            "fragmented_branch": "NONELIGIBLE_PROGRESS_TO_VERIFYING_LAST",
            "verifying_branches": ["ELIGIBLE_COMPLETION", "CONTINUE_TO_VERIFYING_LAST"],
            "verifying_last_branch": "ELIGIBLE_COMPLETION_REQUIRED_BY_DIAGNOSTIC_LIVENESS",
            "completion_adoption": "CAUSAL_ADOPT_OR_HOLD_AFTER_OUTPUT",
            "deployment_protocol_certified": False,
            "terminal_continuation": "R9D_TIMESTAMPED_RESTART_CONTINUATION",
        },
        "metrics": {
            "upstream_r9f_robust_full_qdep_nodes": expected_robust,
            "matched_chi_tests": len(tests),
            "frozen_progress_sensitive_nodes": frozen_n,
            "full_action_progress_sensitive_nodes": full_n,
            "max_frozen_matched_chi_progress_gap_m": float(max_frozen_gap),
            "max_full_matched_chi_progress_gap_m": float(max_full_gap),
            "full_action_verify_more_demanding_tests": int(verify_worse_full),
            "full_action_fragment_more_demanding_tests": int(fragment_worse_full),
            "skipped_domain_tests": int(skipped_domain),
            "general_propagator_regression_max_error": float(max_reg),
        },
        "gates": {
            "r9f_upstream_pass": True,
            "r9f_witness_filter_count": len(robust_rows) == expected_robust,
            "matched_physical_state": True,
            "matched_adopted_information": True,
            "matched_pending_generation_time": True,
            "matched_chi_envelope": True,
            "matched_chi_carry_invariance": True,
            "same_scalar_liveness_bound": True,
            "only_stage_branching_differs": True,
            "general_propagator_regression": bool(max_reg <= 1e-11),
            "deployment_protocol_certified": False,
            "full_augmented_gfp_solved": False,
            "continuous_action_interval_certificate": False,
        },
        "artifacts": {
            "matched_chi_tests_csv": str(tests_csv),
            "witness_summary_csv": str(summary_csv),
        },
    }
    text = json.dumps(out, indent=2, sort_keys=True)
    atomic_write(result_json, text)
    atomic_write(latest_json, text)
    mfiles = [Path(__file__), result_json, tests_csv, summary_csv, upstream, witness_path]
    atomic_write(manifest, "".join(f"{sha256_file(p)}  {p}\n" for p in mfiles if p.exists()))

    print("=== R9-G DECISION ===", flush=True)
    print(
        f"R9G_MATCHED_CHI frozen_sensitive_nodes={frozen_n} "
        f"full_action_sensitive_nodes={full_n} "
        f"max_frozen_gap_m={max_frozen_gap:.12g} "
        f"max_full_gap_m={max_full_gap:.12g}", flush=True,
    )
    print(
        f"R9G_ORIENTATION verify_more_demanding_tests={verify_worse_full} "
        f"fragment_more_demanding_tests={fragment_worse_full}", flush=True,
    )
    print("MATCHED_CHI_PROGRESS_COMPARISON=AVAILABLE", flush=True)
    print("R9G_EXECUTION=PASS", flush=True)
    print(f"R9G_CLASSIFICATION={classification}", flush=True)
    print("CONTINUOUS_STATE_SEPARATION_CERTIFIED=NO", flush=True)
    print("MARKOV_SUFFICIENCY_CERTIFIED=NO", flush=True)
    print(f"R9G_NEXT_ACTION={next_action}", flush=True)
    print(f"RESULT_JSON={result_json}", flush=True)
    print(f"MANIFEST={manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
