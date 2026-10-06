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
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k); fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def spatial_cell_corner_max(nodal: np.ndarray) -> np.ndarray:
    """Maximum over the 2^6 physical/information corners; keep q discrete.

    IMPORTANT: this is NOT a sound lower bound.  R9-H-F uses it only as an
    attribution ceiling: if even this optimistic replacement cannot restore a
    paired margin, then ordinary tightening of the existing cell-min lower
    continuation cannot be the sole explanation of the failure.
    """
    out = np.asarray(nodal, dtype=float)
    for axis in range(6):
        left = [slice(None)] * out.ndim
        right = [slice(None)] * out.ndim
        left[axis] = slice(0, -1)
        right[axis] = slice(1, None)
        out = np.maximum(out[tuple(left)], out[tuple(right)])
    return out


def build_restart_cells(h_flat: np.ndarray, data: dict, R: int, *, mode: str) -> np.ndarray:
    h = np.asarray(h_flat, float).reshape(tuple(data["lookup_shape"]) + (R,))
    if mode == "min":
        c = h
        for axis in range(6):
            l = [slice(None)] * c.ndim; r = [slice(None)] * c.ndim
            l[axis] = slice(0, -1); r[axis] = slice(1, None)
            c = np.minimum(c[tuple(l)], c[tuple(r)])
    elif mode == "max":
        c = spatial_cell_corner_max(h)
    else:
        raise ValueError(mode)
    return c[..., R - 1].reshape(-1)


def point_stage_lower(*, st: dict, env: dict, pending_age: float,
                      actions: list[float], cells: np.ndarray,
                      data: dict, r3, b0, sw, r9e, r9f, r9hd,
                      p2c_resolution: int) -> dict:
    """Point-action service-tree lower evaluator.

    With ``cells`` equal to corner-min restart cells this is a conditional lower
    calculation for the sampled action model.  With corner-max cells it is only
    an optimistic attribution diagnostic and is never promoted as a certificate.
    """
    Ts = float(data["Ts"])
    fb_lo, fb_hi = r9hd.p2c_lower_candidate(st, data, b0, sw, int(p2c_resolution))

    def verify_last(s1: dict, p_age: float) -> float:
        fb1, _ = r9hd.p2c_lower_candidate(s1, data, b0, sw, int(p2c_resolution))
        best = math.inf
        for u in actions:
            tr = __import__("p3b1_r9g_matched_chi_branching_service").physical_step(
                s1, float(u), data, r3, b0, r9e
            )
            cf = r9hd.lower_completion_future(
                end_state=tr["next_state"], env=env,
                candidate_age=float(p_age + Ts), lower_cells=cells,
                data=data, r3=r3, r9f=r9f,
            )
            if not cf.get("valid", False):
                continue
            req = max(0.0, float(tr["step_loss_m"]),
                      float(tr["closing_end_m"]) + float(cf["future_m"]))
            best = min(best, req)
        return min(float(fb1), best) if math.isfinite(best) else math.inf

    best_f = math.inf
    best_v = math.inf
    best_fu = None
    best_vu = None
    for u in actions:
        tr = __import__("p3b1_r9g_matched_chi_branching_service").physical_step(
            st, float(u), data, r3, b0, r9e
        )
        nxt = tr["next_state"]
        last = verify_last(nxt, float(pending_age + Ts))
        if not math.isfinite(last):
            continue
        reqf = min(float(fb_lo), max(0.0, float(tr["step_loss_m"]),
                                      float(tr["closing_end_m"]) + last))
        if reqf < best_f:
            best_f = reqf; best_fu = float(u)

        cf = r9hd.lower_completion_future(
            end_state=nxt, env=env,
            candidate_age=float(pending_age + Ts), lower_cells=cells,
            data=data, r3=r3, r9f=r9f,
        )
        if not cf.get("valid", False):
            continue
        fut = max(float(cf["future_m"]), last)
        reqv = min(float(fb_lo), max(0.0, float(tr["step_loss_m"]),
                                      float(tr["closing_end_m"]) + fut))
        if reqv < best_v:
            best_v = reqv; best_vu = float(u)

    return {
        "valid": math.isfinite(best_f) and math.isfinite(best_v),
        "fragmented_lower_m": float(best_f),
        "verifying_lower_m": float(best_v),
        "fragmented_best_action": best_fu,
        "verifying_best_action": best_vu,
        "p2c_lower_candidate_m": float(fb_lo),
        "p2c_upper_candidate_m": float(fb_hi),
    }


def self_test() -> None:
    z = np.zeros((2,2,2,2,2,2,2), dtype=float)
    z[1,1,1,1,1,1,:] = [1.0, 2.0]
    c = spatial_cell_corner_max(z)
    assert c.shape == (1,1,1,1,1,1,2)
    assert np.allclose(c.reshape(-1), [1.0, 2.0])
    print("R9HF_INTERNAL_SELF_TEST=PASS")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--restart-action-width", type=float, default=0.25)
    ap.add_argument("--stage-action-width", type=float, default=0.03125)
    ap.add_argument("--p2c-resolution", type=int, default=257)
    args = ap.parse_args()
    if args.self_test:
        self_test(); return 0
    if args.restart_action_width <= 0 or args.stage_action_width <= 0:
        raise ValueError("R9HF_BAD_ACTION_WIDTH")

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
    import p3b1_r9h_b_protocolized_paired_bounds as r9hb
    import p3b1_r9h_c_theorem_gate_audit as r9hc
    import p3b1_r9h_d_paired_interval_gfp as r9hd
    import p3b1_r9h_e_converged_lower_gfp as r9he

    he_path = RESULTS / "P3B1_R9HE_LATEST.json"
    he = json.loads(he_path.read_text(encoding="utf-8"))
    expected = "CONVERGED_POINT_ACTION_LOWER_FIXED_POINT_STILL_NO_PAIRED_MARGIN"
    if he.get("status") != "PASS" or he.get("classification") != expected:
        raise RuntimeError(f"R9HF_R9HE_GATE_FAIL classification={he.get('classification')}")
    hm = he.get("metrics", {})
    if int(hm.get("positive_paired_margin_tests", -1)) != 0:
        raise RuntimeError("R9HF_EXPECTED_ZERO_UPSTREAM_MARGIN")
    if int(hm.get("restart_binding_tests", 0)) <= 0:
        raise RuntimeError("R9HF_EXPECTED_RESTART_BINDING")

    hc = json.loads((RESULTS / "P3B1_R9HC_LATEST.json").read_text(encoding="utf-8"))
    if hc.get("status") != "PASS":
        raise RuntimeError("R9HF_R9HC_NOT_PASS")
    rg = json.loads((RESULTS / "P3B1_R9G_LATEST.json").read_text(encoding="utf-8"))
    tests = r9hb.load_stage_tests(rg)

    data = r9d.reconstruct_timestamped_geometry(r3, r5, r6, r7, r9c)
    profile = "diagnostic_fast"
    R = r7.service_horizon(profile, data["p1"]["diagnostic_service_profiles"][profile])
    if R != 2:
        raise RuntimeError("R9HF_EXPECTED_R2")
    J = float(data["eval_cfg"]["information_contract"]["slew_rate"])

    print("=== P3-B1-R9-H-F RESTART-LOWER ATTRIBUTION AUDIT ===", flush=True)
    print("UPSTREAM_R9HE_CONVERGED_LOWER_GFP=PASS", flush=True)
    print(f"MATCHED_CHI_TESTS={len(tests)}", flush=True)
    print("QUESTION=IS_ZERO_PAIRED_MARGIN_CAUSED_BY_ACTION_RELAXATION_CELL_MIN_OR_RESTART_SERVICE_SEMANTICS", flush=True)
    print("CELL_MAX_OBJECT=CLEARLY_DIAGNOSTIC_NOT_A_CERTIFICATE", flush=True)
    print("CONTINUOUS_STATE_SEPARATION_CERTIFIED=NO", flush=True)

    restart_actions = r9hb.refinement_actions(float(args.restart_action_width))
    print(f"R9HF_STAGE_START=shared_restart_transition_build actions={len(restart_actions)}", flush=True)
    cfg_r, td = r9hb.full_action_restart_transitions(
        data, restart_actions, R, r3, b0, r9e, chunk_size=int(args.chunk_size)
    )

    upper = r9d.solve_timestamped_causal(
        R, cfg_r, data["eval_idx"], data["lookup_shape"],
        data["lookup_fallback"], data["eval_fallback"], td, r3,
        label="r9hf_upper_same_action_grid",
    )
    if not upper.converged:
        raise RuntimeError("R9HF_UPPER_NOT_CONVERGED")
    upper_cells = r9g.build_restart_cell_values(upper.h_flat, data, R, r3)

    print("R9HF_STAGE_START=global_p2c_lower_bracket", flush=True)
    fallback_lower, fb_metrics = r9he.compute_eval_fallback_lower(
        data, b0, sw, resolution=int(args.p2c_resolution), chunk_size=int(args.chunk_size)
    )

    print("R9HF_STAGE_START=rebuild_converged_point_action_lower_fixed_point", flush=True)
    lower = r9he.solve_point_action_lower_fixed_point(
        R, cfg_r, np.asarray(data["eval_idx"], dtype=np.int64), tuple(data["lookup_shape"]),
        fallback_lower, td, label="r9hf_point_action_lower",
    )
    if not lower.converged:
        raise RuntimeError("R9HF_LOWER_NOT_CONVERGED")

    lower_min_cells = build_restart_cells(lower.h_flat, data, R, mode="min")
    lower_max_cells = build_restart_cells(lower.h_flat, data, R, mode="max")

    # Sanity: the diagnostic cell-max object must dominate the sound cell-min object.
    if np.any(lower_max_cells + 1e-12 < lower_min_cells):
        raise RuntimeError("R9HF_CELL_MAX_BELOW_CELL_MIN")

    stage_actions = r9hb.refinement_actions(float(args.stage_action_width))
    intervals = r9hc.action_bounds(float(args.stage_action_width))
    vf, vp, af, ba, bu, age = [np.asarray(x, float) for x in data["eval_flat"]]
    fallback_cache: dict = {}

    rows: list[dict] = []
    diagnostic_positive = 0
    sampled_positive = 0
    interval_positive = 0
    action_relaxation_dominance_fail = 0
    max_spatial_headroom = 0.0
    max_action_relaxation_loss = 0.0

    for k, row in enumerate(tests, start=1):
        i = int(row["node_index"]); pending_age = float(row["pending_age_s"])
        st = {
            "v_f": float(vf[i]), "v_p": float(vp[i]), "a_f": float(af[i]),
            "bar_a": float(ba[i]), "bar_u": float(bu[i]), "age": float(age[i]),
        }
        env = r9g.matched_pending_envelope(
            adopted_age=st["age"], pending_age=pending_age,
            bar_a=st["bar_a"], bar_u=st["bar_u"], J=J,
            b0=b0, p3a=data["p3a"], p2b=data["p2b"],
        )
        if not env.get("valid", False):
            raise RuntimeError(f"R9HF_MATCHED_ENV_INVALID node={i}")

        up = r9g.matched_stage_pair(
            st=st, env=env, pending_age=pending_age,
            actions=stage_actions, restart_cells=upper_cells,
            data=data, r3=r3, b0=b0, sw=sw, r9e=r9e, r9f=r9f,
            fallback_cache=fallback_cache,
        )
        lo_point = point_stage_lower(
            st=st, env=env, pending_age=pending_age,
            actions=stage_actions, cells=lower_min_cells,
            data=data, r3=r3, b0=b0, sw=sw, r9e=r9e, r9f=r9f, r9hd=r9hd,
            p2c_resolution=int(args.p2c_resolution),
        )
        lo_interval = r9hd.interval_relaxed_stage_lower(
            st=st, env=env, pending_age=pending_age,
            intervals=intervals, lower_cells=lower_min_cells,
            data=data, r3=r3, b0=b0, sw=sw, r9e=r9e, r9f=r9f,
            p2c_resolution=int(args.p2c_resolution),
        )
        diag_max = point_stage_lower(
            st=st, env=env, pending_age=pending_age,
            actions=stage_actions, cells=lower_max_cells,
            data=data, r3=r3, b0=b0, sw=sw, r9e=r9e, r9f=r9f, r9hd=r9hd,
            p2c_resolution=int(args.p2c_resolution),
        )
        if not all(z.get("valid", False) for z in (up, lo_point, lo_interval, diag_max)):
            raise RuntimeError(f"R9HF_PAIR_INVALID node={i} pending_age={pending_age}")

        upper_frag = float(up["fragmented_required_gap_m"])
        m_point = float(lo_point["verifying_lower_m"] - upper_frag)
        m_interval = float(lo_interval["verifying_lower_m"] - upper_frag)
        m_diag = float(diag_max["verifying_lower_m"] - upper_frag)

        sampled_positive += int(m_point > TOL)
        interval_positive += int(m_interval > TOL)
        diagnostic_positive += int(m_diag > TOL)
        if float(lo_interval["verifying_lower_m"]) > float(lo_point["verifying_lower_m"]) + 1e-8:
            action_relaxation_dominance_fail += 1

        spatial_headroom = float(diag_max["verifying_lower_m"] - lo_point["verifying_lower_m"])
        action_loss = float(lo_point["verifying_lower_m"] - lo_interval["verifying_lower_m"])
        max_spatial_headroom = max(max_spatial_headroom, spatial_headroom)
        max_action_relaxation_loss = max(max_action_relaxation_loss, action_loss)

        rows.append({
            "node_index": i,
            "pending_age_s": pending_age,
            "fragmented_upper_m": upper_frag,
            "verifying_point_cellmin_lower_m": lo_point["verifying_lower_m"],
            "verifying_interval_cellmin_lower_m": lo_interval["verifying_lower_m"],
            "verifying_cellmax_diagnostic_m": diag_max["verifying_lower_m"],
            "sampled_point_margin_m": m_point,
            "interval_relaxed_margin_m": m_interval,
            "cellmax_diagnostic_margin_m": m_diag,
            "spatial_cell_headroom_m": spatial_headroom,
            "action_relaxation_loss_m": action_loss,
            "point_best_action": lo_point["verifying_best_action"],
            "diagnostic_best_action": diag_max["verifying_best_action"],
            "p2c_lower_candidate_m": lo_point["p2c_lower_candidate_m"],
            "cellmax_is_certificate": False,
        })
        print(
            f"R9HF_PROGRESS tests={k}/{len(tests)} diag_positive={diagnostic_positive} "
            f"point_margin_m={m_point:.12g} interval_margin_m={m_interval:.12g} "
            f"cellmax_diag_margin_m={m_diag:.12g}",
            flush=True,
        )

    if action_relaxation_dominance_fail:
        raise RuntimeError(
            f"R9HF_INTERVAL_RELAXATION_DIRECTION_FAIL count={action_relaxation_dominance_fail}"
        )

    # Classification is deliberately diagnostic.  A positive cell-max margin says
    # only that spatial lower-enclosure tightness could plausibly recover a paired
    # certificate.  It is NOT itself a proof.
    if diagnostic_positive > 0:
        classification = "STATE_CELL_LOWER_ENCLOSURE_IS_PRIMARY_REMAINING_BLOCKER_DIAGNOSTIC_HEADROOM_EXISTS"
        next_action = "R9H_G_BUILD_LOCAL_ADAPTIVE_RESTART_CELL_REFINEMENT_WITH_INTERVAL_SOUND_LOWER_ENCLOSURE"
    else:
        classification = "RESTART_SERVICE_SEMANTICS_BIND_BEYOND_EXISTING_CELL_MIN_CONSERVATISM"
        next_action = "R9H_G_BUILD_FULL_SPARSE_AUGMENTED_SERVICE_RESTART_GFP_BEFORE_FURTHER_CONTINUOUS_PROMOTION"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result = RESULTS / f"P3B1_R9HF_RESULT_{stamp}.json"
    latest = RESULTS / "P3B1_R9HF_LATEST.json"
    csvp = RESULTS / f"P3B1_R9HF_RESTART_LOWER_ATTRIBUTION_{stamp}.csv"
    manifest = RESULTS / f"P3B1_R9HF_MANIFEST_{stamp}.sha256"
    write_csv(csvp, rows)

    out = {
        "schema": "SCV_P3B1_R9HF_RESTART_LOWER_ATTRIBUTION_V1",
        "status": "PASS",
        "timestamp_utc": stamp,
        "classification": classification,
        "next_action": next_action,
        "continuous_state_separation_certified": False,
        "metrics": {
            "matched_chi_tests": len(tests),
            "sampled_point_positive_margin_tests": sampled_positive,
            "interval_relaxed_positive_margin_tests": interval_positive,
            "cellmax_diagnostic_positive_margin_tests": diagnostic_positive,
            "max_spatial_cell_headroom_m": max_spatial_headroom,
            "max_action_relaxation_loss_m": max_action_relaxation_loss,
            "action_relaxation_direction_failures": action_relaxation_dominance_fail,
            "lower_iterations": lower.iterations,
            "lower_final_delta_m": lower.final_delta,
            "restart_action_width": float(args.restart_action_width),
            "stage_action_width": float(args.stage_action_width),
            "stage_action_count": len(stage_actions),
            "stage_interval_count": len(intervals),
            "p2c_resolution": int(args.p2c_resolution),
            **{f"p2c_{k}": v for k, v in fb_metrics.items()},
        },
        "gates": {
            "upstream_r9he_pass": True,
            "recomputed_lower_fixed_point_converged": bool(lower.converged),
            "interval_relaxation_not_stronger_than_point_lower": action_relaxation_dominance_fail == 0,
            "cellmax_object_is_diagnostic_only": True,
            "continuous_action_lower_gfp_certified": False,
            "independent_p2c_lower_checker_pass": False,
            "full_augmented_interval_gfp_solved": False,
        },
        "artifacts": {"restart_lower_attribution_csv": str(csvp)},
    }
    text = json.dumps(out, indent=2, sort_keys=True)
    atomic_write(result, text); atomic_write(latest, text)
    mfiles = [
        Path(__file__), result, csvp,
        RESULTS / "P3B1_R9HE_LATEST.json",
        RESULTS / "P3B1_R9HD_LATEST.json",
        RESULTS / "P3B1_R9HC_LATEST.json",
        RESULTS / "P3B1_R9G_LATEST.json",
    ]
    atomic_write(manifest, "".join(f"{sha256_file(p)}  {p}\n" for p in mfiles if p.exists()))

    print("=== R9-H-F DECISION ===", flush=True)
    print(
        f"R9HF_ATTRIBUTION sampled_point_positive={sampled_positive}/{len(tests)} "
        f"interval_relaxed_positive={interval_positive}/{len(tests)} "
        f"cellmax_diagnostic_positive={diagnostic_positive}/{len(tests)}",
        flush=True,
    )
    print(
        f"R9HF_HEADROOM max_spatial_cell_headroom_m={max_spatial_headroom:.12g} "
        f"max_action_relaxation_loss_m={max_action_relaxation_loss:.12g}",
        flush=True,
    )
    print("R9HF_CELL_MAX_DIAGNOSTIC_IS_CERTIFICATE=NO", flush=True)
    print("CONTINUOUS_ACTION_LOWER_GFP_CERTIFIED=NO", flush=True)
    print("P2C_INDEPENDENT_LOWER_CHECKER=NO", flush=True)
    print("FULL_AUGMENTED_INTERVAL_GFP_SOLVED=NO", flush=True)
    print("R9HF_EXECUTION=PASS", flush=True)
    print(f"R9HF_CLASSIFICATION={classification}", flush=True)
    print("CONTINUOUS_STATE_SEPARATION_CERTIFIED=NO", flush=True)
    print(f"R9HF_NEXT_ACTION={next_action}", flush=True)
    print(f"RESULT_JSON={result}", flush=True)
    print(f"MANIFEST={manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
