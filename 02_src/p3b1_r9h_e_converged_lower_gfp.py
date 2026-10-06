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


def spatial_cell_corner_min(nodal: np.ndarray) -> np.ndarray:
    """Minimum over the 2^6 physical/information corners; keep q discrete."""
    out = np.asarray(nodal, dtype=float)
    for axis in range(6):
        left = [slice(None)] * out.ndim
        right = [slice(None)] * out.ndim
        left[axis] = slice(0, -1)
        right[axis] = slice(1, None)
        out = np.minimum(out[tuple(left)], out[tuple(right)])
    return out


def finite_sup_abs(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if not np.any(m):
        return 0.0
    return float(np.max(np.abs(a[m] - b[m])))


def lookup_lower(cell_flat_q: np.ndarray, index: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.zeros(len(index), dtype=float)
    m = np.asarray(valid, dtype=bool)
    out[m] = cell_flat_q[np.asarray(index[m], dtype=np.int64)]
    # Invalid lookup is assigned zero, which is fail-safe for a lower bound.
    return out


def compute_eval_fallback_lower(data: dict, b0, sw, *, resolution: int, chunk_size: int) -> tuple[np.ndarray, dict]:
    """Conditional P2-C lower bracket on all evaluation nodes.

    This uses the existing switching_loss_bracket implementation and therefore is
    *not* the independent P2-C checker required for continuous promotion.  It is
    useful for closing the lower fixed point of the current numerical model.
    """
    cfg, p2b, p2c, p3a = data["eval_cfg"], data["p2b"], data["p2c"], data["p3a"]
    vf, vp, af, ba, bu, age = [np.asarray(x, float) for x in data["eval_flat"]]
    J = float(cfg["information_contract"]["slew_rate"])
    n = len(vf)
    lo_all = np.empty(n, dtype=float)
    hi_all = np.empty(n, dtype=float)
    for start in range(0, n, int(chunk_size)):
        stop = min(n, start + int(chunk_size)); sl = slice(start, stop)
        ap = b0.predecessor_acceleration_lower(age[sl], ba[sl], bu[sl], J, p3a, p2b)
        lo, hi, _ = sw.switching_loss_bracket(
            vf[sl], af[sl], vp[sl], ap,
            int(p2c["switching"]["N_sw"]), [int(resolution)], p2c, p2b,
        )
        lo_all[sl] = np.maximum(np.asarray(lo, float).reshape(-1), 0.0)
        hi_all[sl] = np.maximum(np.asarray(hi, float).reshape(-1), 0.0)
        if start == 0 or stop == n or (stop // int(chunk_size)) % 32 == 0:
            print(f"R9HE_P2C_LOWER_PROGRESS states={stop}/{n}", flush=True)
    ordered = lo_all <= hi_all + 1e-10
    legacy_upper = np.asarray(data["eval_fallback"], float)
    legacy_order = lo_all <= legacy_upper + 1e-8
    metrics = {
        "ordered_failures_same_resolution": int(np.count_nonzero(~ordered)),
        "lower_above_legacy_upper_failures": int(np.count_nonzero(~legacy_order)),
        "max_same_resolution_width_m": float(np.max(np.maximum(hi_all - lo_all, 0.0))),
        "p95_same_resolution_width_m": float(np.quantile(np.maximum(hi_all - lo_all, 0.0), 0.95)),
        "max_lower_m": float(np.max(lo_all)),
    }
    if metrics["ordered_failures_same_resolution"]:
        raise RuntimeError("R9HE_P2C_BRACKET_ORDER_FAIL")
    if metrics["lower_above_legacy_upper_failures"]:
        raise RuntimeError("R9HE_P2C_LOWER_EXCEEDS_LEGACY_UPPER")
    return lo_all, metrics


@dataclass
class LowerSolve:
    h_flat: np.ndarray
    converged: bool
    iterations: int
    final_delta: float
    monotonicity_violations: int


def solve_point_action_lower_fixed_point(
    R: int,
    cfg: dict,
    eval_idx: np.ndarray,
    lookup_shape: tuple,
    eval_fallback_lower: np.ndarray,
    transition_data: dict,
    *,
    max_iterations: int | None = None,
    label: str = "r9he_lower",
) -> LowerSolve:
    """Ascending lower fixed point for the *point-action* restart model.

    - cell continuation uses corner minima;
    - within-step loss removes the positive temporal upper correction;
    - P2-C uses its reported lower bracket;
    - the action set is still sampled, so this is NOT a theorem-grade lower
      bound for the continuous action interval.  That gate remains explicitly NO.
    """
    lookup_nodes = int(np.prod(lookup_shape))
    h_flat = np.zeros((lookup_nodes, R), dtype=float)
    tol = float(cfg["fixed_point"]["tolerance_m"])
    max_iter = int(max_iterations or cfg["fixed_point"]["max_iterations"])
    lips = float(transition_data.get("lipschitz_correction_m", 0.0))
    violations = 0
    delta = math.inf

    for it in range(1, max_iter + 1):
        h = h_flat.reshape(tuple(lookup_shape) + (R,))
        cmin = spatial_cell_corner_min(h)
        cflat = [cmin[..., qi].reshape(-1) for qi in range(R)]
        old = h_flat[np.asarray(eval_idx, dtype=np.int64), :].copy()
        new = old.copy()

        for q in range(1, R + 1):
            qi = q - 1
            best = np.full(len(eval_idx), np.inf, dtype=float)
            for tr in transition_data["transitions"]:
                adopt = lookup_lower(
                    cflat[R - 1], tr["adopt_index_by_q"][qi], tr["adopt_valid_by_q"][qi]
                )
                hold = lookup_lower(cflat[R - 1], tr["defer_index"], tr["defer_valid"])
                completion = np.minimum(adopt, hold)
                future = completion
                if q > 1:
                    defer = lookup_lower(cflat[q - 2], tr["defer_index"], tr["defer_valid"])
                    future = np.maximum(completion, defer)

                # step_loss_upper = sampled maximum + positive temporal cover.
                # Removing that cover gives a lower element for the sampled
                # within-step loss of this point action.
                step_lower = np.maximum(np.asarray(tr["step_loss_upper"], float) - lips, 0.0)
                required = np.maximum(step_lower, np.asarray(tr["closing_end"], float) + future)
                required = np.maximum(required, 0.0)
                best = np.minimum(best, required)

            cand = np.minimum(np.asarray(eval_fallback_lower, float), best)
            bad = cand < old[:, qi] - max(1e-9, 10.0 * tol)
            violations += int(np.count_nonzero(bad))
            if np.any(bad):
                worst = float(np.min(cand[bad] - old[bad, qi]))
                raise RuntimeError(f"R9HE_LOWER_MONOTONICITY_FAIL q={q} worst_delta={worst}")
            # Suppress floating roundoff only; the mathematical sequence should ascend.
            new[:, qi] = np.maximum(old[:, qi], cand)

        delta = finite_sup_abs(new, old)
        h_flat[np.asarray(eval_idx, dtype=np.int64), :] = new
        if it == 1 or it % 5 == 0 or delta <= tol:
            print(
                f"R9HE_LOWER_ITER label={label} iteration={it} delta_m={delta:.12g} "
                f"monotonicity_violations={violations}",
                flush=True,
            )
        if delta <= tol:
            return LowerSolve(h_flat, True, it, delta, violations)
    return LowerSolve(h_flat, False, max_iter, delta, violations)


def build_lower_restart_cells(h_flat: np.ndarray, data: dict, R: int) -> np.ndarray:
    h = np.asarray(h_flat, float).reshape(tuple(data["lookup_shape"]) + (R,))
    cmin = spatial_cell_corner_min(h)
    return cmin[..., R - 1].reshape(-1)


def self_test() -> None:
    x = np.arange(2 * 3 * 4 * 2, dtype=float).reshape(2, 3, 4, 2)
    # Synthetic 3-spatial-axis analogue of corner min sanity.
    y = x.copy()
    for axis in range(3):
        l = [slice(None)] * y.ndim; r = [slice(None)] * y.ndim
        l[axis] = slice(0, -1); r[axis] = slice(1, None)
        y = np.minimum(y[tuple(l)], y[tuple(r)])
    assert y.shape == (1, 2, 3, 2)
    # Real helper expects six spatial axes.
    z = np.zeros((2, 2, 2, 2, 2, 2, 2), dtype=float)
    z[1,1,1,1,1,1,:] = 1.0
    c = spatial_cell_corner_min(z)
    assert c.shape == (1,1,1,1,1,1,2)
    assert np.all(c == 0.0)
    print("R9HE_INTERNAL_SELF_TEST=PASS")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--restart-action-width", type=float, default=0.25)
    ap.add_argument("--stage-interval-width", type=float, default=0.03125)
    ap.add_argument("--fallback-lower-resolution", type=int, default=257)
    ap.add_argument("--max-lower-iterations", type=int, default=0)
    args = ap.parse_args()
    if args.self_test:
        self_test(); return 0
    if args.restart_action_width <= 0 or args.stage_interval_width <= 0:
        raise ValueError("R9HE_BAD_ACTION_WIDTH")
    if args.fallback_lower_resolution < 17:
        raise ValueError("R9HE_BAD_P2C_RESOLUTION")

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

    hd = json.loads((RESULTS / "P3B1_R9HD_LATEST.json").read_text(encoding="utf-8"))
    expected = "ONE_STEP_LOWER_BOUND_TOO_WEAK_FOR_PAIRED_INTERVAL_SEPARATION"
    if hd.get("status") != "PASS" or hd.get("classification") != expected:
        raise RuntimeError(f"R9HE_R9HD_GATE_FAIL classification={hd.get('classification')}")
    if int(hd.get("metrics", {}).get("candidate_positive_tests", -1)) != 0:
        raise RuntimeError("R9HE_EXPECTED_ZERO_H_D_CANDIDATES")

    hc = json.loads((RESULTS / "P3B1_R9HC_LATEST.json").read_text(encoding="utf-8"))
    rg = json.loads((RESULTS / "P3B1_R9G_LATEST.json").read_text(encoding="utf-8"))
    tests = r9hb.load_stage_tests(rg)
    data = r9d.reconstruct_timestamped_geometry(r3, r5, r6, r7, r9c)
    profile = "diagnostic_fast"
    R = r7.service_horizon(profile, data["p1"]["diagnostic_service_profiles"][profile])
    if R != 2:
        raise RuntimeError("R9HE_EXPECTED_R2")
    J = float(data["eval_cfg"]["information_contract"]["slew_rate"])

    print("=== P3-B1-R9-H-E CONVERGED LOWER FIXED-POINT AUDIT ===", flush=True)
    print("UPSTREAM_R9HD_ONE_STEP_LOWER_WEAK=PASS", flush=True)
    print(f"MATCHED_CHI_TESTS={len(tests)}", flush=True)
    print(f"RESTART_POINT_ACTION_WIDTH={args.restart_action_width:.12g}", flush=True)
    print(f"STAGE_INTERVAL_WIDTH={args.stage_interval_width:.12g}", flush=True)
    print(f"P2C_LOWER_RESOLUTION={args.fallback_lower_resolution}", flush=True)
    print("LOWER_FIXED_POINT_MODEL=POINT_ACTION_RESTART_WITH_CELL_CORNER_MINIMA", flush=True)
    print("CONTINUOUS_ACTION_LOWER_GFP_CERTIFIED=NO", flush=True)
    print("P2C_INDEPENDENT_LOWER_CHECKER=NO", flush=True)
    print("CONTINUOUS_STATE_SEPARATION_CERTIFIED=NO", flush=True)

    actions = r9hb.refinement_actions(float(args.restart_action_width))
    print(f"R9HE_STAGE_START=shared_restart_transition_build actions={len(actions)}", flush=True)
    cfg_restart, td = r9hb.full_action_restart_transitions(
        data, actions, R, r3, b0, r9e, chunk_size=args.chunk_size
    )

    # Upper fixed point on the same action grid, used solely for sandwich diagnostics.
    upper = r9d.solve_timestamped_causal(
        R, cfg_restart, data["eval_idx"], data["lookup_shape"],
        data["lookup_fallback"], data["eval_fallback"], td, r3,
        label="r9he_upper_same_action_grid",
    )
    if not upper.converged:
        raise RuntimeError("R9HE_UPPER_NOT_CONVERGED")
    upper_cells = r9g.build_restart_cell_values(upper.h_flat, data, R, r3)

    print("R9HE_STAGE_START=global_p2c_lower_bracket", flush=True)
    fallback_lower, fb_metrics = compute_eval_fallback_lower(
        data, b0, sw, resolution=int(args.fallback_lower_resolution), chunk_size=int(args.chunk_size)
    )
    print(
        "R9HE_P2C_LOWER "
        f"ordered_failures={fb_metrics['ordered_failures_same_resolution']} "
        f"legacy_upper_failures={fb_metrics['lower_above_legacy_upper_failures']} "
        f"max_width_m={fb_metrics['max_same_resolution_width_m']:.12g} "
        f"p95_width_m={fb_metrics['p95_same_resolution_width_m']:.12g}",
        flush=True,
    )

    print("R9HE_STAGE_START=ascending_point_action_lower_fixed_point", flush=True)
    lower = solve_point_action_lower_fixed_point(
        R, cfg_restart, np.asarray(data["eval_idx"], dtype=np.int64), tuple(data["lookup_shape"]),
        fallback_lower, td,
        max_iterations=(None if args.max_lower_iterations <= 0 else int(args.max_lower_iterations)),
        label="r9he_point_action_lower",
    )
    if not lower.converged:
        raise RuntimeError("R9HE_LOWER_FIXED_POINT_NOT_CONVERGED")
    lower_cells = build_lower_restart_cells(lower.h_flat, data, R)

    # The lower point-action object should never exceed the upper object on the
    # same sampled action model if the directional constructions are coherent.
    eidx = np.asarray(data["eval_idx"], dtype=np.int64)
    diff = lower.h_flat[eidx, :] - upper.h_flat[eidx, :]
    order_viol = int(np.count_nonzero(diff > 1e-8))
    max_lower_minus_upper = float(np.max(diff))
    if order_viol:
        raise RuntimeError(
            f"R9HE_POINT_MODEL_SANDWICH_ORDER_FAIL violations={order_viol} max={max_lower_minus_upper}"
        )
    print(
        f"R9HE_POINT_MODEL_SANDWICH lower_le_upper=PASS violations={order_viol} "
        f"max_lower_minus_upper_m={max_lower_minus_upper:.12g}",
        flush=True,
    )

    intervals = r9hc.action_bounds(float(args.stage_interval_width))
    vf, vp, af, ba, bu, age = [np.asarray(x, float) for x in data["eval_flat"]]
    fallback_cache: dict = {}
    rows: list[dict] = []
    positive = 0
    p2c_binding = 0
    restart_binding = 0
    min_margin = math.inf
    max_margin = -math.inf

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
            raise RuntimeError(f"R9HE_MATCHED_ENV_INVALID node={i}")

        upper_pair = r9g.matched_stage_pair(
            st=st, env=env, pending_age=pending_age,
            actions=r9hb.refinement_actions(float(args.stage_interval_width)),
            restart_cells=upper_cells, data=data,
            r3=r3, b0=b0, sw=sw, r9e=r9e, r9f=r9f,
            fallback_cache=fallback_cache,
        )
        lower_pair = r9hd.interval_relaxed_stage_lower(
            st=st, env=env, pending_age=pending_age,
            intervals=intervals, lower_cells=lower_cells, data=data,
            r3=r3, b0=b0, sw=sw, r9e=r9e, r9f=r9f,
            p2c_resolution=int(args.fallback_lower_resolution),
        )
        if not upper_pair.get("valid", False) or not lower_pair.get("valid", False):
            raise RuntimeError(f"R9HE_PAIR_INVALID node={i} pending_age={pending_age}")

        margin = float(lower_pair["verifying_lower_m"] - upper_pair["fragmented_required_gap_m"])
        is_pos = margin > TOL
        positive += int(is_pos)
        min_margin = min(min_margin, margin); max_margin = max(max_margin, margin)

        # Binding diagnosis.  If verifying lower equals the P2-C lower bracket,
        # fallback clipping is the active lower-side bottleneck.  Otherwise the
        # restart/service continuation remains the active lower bottleneck.
        p2c_bind = abs(
            float(lower_pair["verifying_lower_m"]) - float(lower_pair["p2c_lower_candidate_m"])
        ) <= 1e-8
        p2c_binding += int(p2c_bind)
        restart_binding += int(not p2c_bind)

        rows.append({
            "node_index": i,
            "pending_age_s": pending_age,
            "fragmented_upper_m": upper_pair["fragmented_required_gap_m"],
            "verifying_upper_m": upper_pair["verifying_required_gap_m"],
            "fragmented_lower_m": lower_pair["fragmented_lower_m"],
            "verifying_lower_m": lower_pair["verifying_lower_m"],
            "paired_margin_m": margin,
            "paired_margin_positive": is_pos,
            "p2c_lower_candidate_m": lower_pair["p2c_lower_candidate_m"],
            "p2c_upper_candidate_m": lower_pair["p2c_upper_candidate_m"],
            "lower_binding_reason": "P2C_LOWER_CLIPPING" if p2c_bind else "RESTART_OR_SERVICE_CONTINUATION",
        })
        print(
            f"R9HE_PROGRESS tests={k}/{len(tests)} positive={positive} "
            f"margin_m={margin:.12g} binding={'P2C' if p2c_bind else 'RESTART'}",
            flush=True,
        )

    # This round deliberately cannot emit a continuous theorem: the converged
    # lower restart object is for a dense *point-action* model and the P2-C lower
    # bracket is not independently rederived.
    continuous_yes = False
    continuous_action_lower_certified = False
    independent_p2c = False
    full_interval_gfp = False

    if positive > 0:
        classification = "CONVERGED_POINT_ACTION_LOWER_FIXED_POINT_RESTORES_PAIRED_MARGIN_CONTINUOUS_INTERVAL_AND_INDEPENDENT_P2C_GATES_REMAIN"
        next_action = "R9H_F_BUILD_INTERVAL_SOUND_LOWER_RESTART_AND_INDEPENDENT_P2C_LOWER_CHECKER"
    elif p2c_binding == len(tests):
        classification = "CONVERGED_LOWER_FIXED_POINT_REMAINS_P2C_LOWER_CLIPPED"
        next_action = "R9H_F_BUILD_INDEPENDENT_ANALYTIC_P2C_LOWER_CHECKER_BEFORE_FURTHER_GFP_REFINEMENT"
    else:
        classification = "CONVERGED_POINT_ACTION_LOWER_FIXED_POINT_STILL_NO_PAIRED_MARGIN"
        next_action = "R9H_F_STRENGTHEN_INTERVAL_SOUND_LOWER_CONTINUATION_AND_REASSESS_BINDING_STRUCTURE"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result = RESULTS / f"P3B1_R9HE_RESULT_{stamp}.json"
    latest = RESULTS / "P3B1_R9HE_LATEST.json"
    csvp = RESULTS / f"P3B1_R9HE_CONVERGED_LOWER_BOUNDS_{stamp}.csv"
    manifest = RESULTS / f"P3B1_R9HE_MANIFEST_{stamp}.sha256"
    write_csv(csvp, rows)

    out = {
        "schema": "SCV_P3B1_R9HE_CONVERGED_POINT_ACTION_LOWER_FIXED_POINT_V1",
        "status": "PASS",
        "timestamp_utc": stamp,
        "classification": classification,
        "next_action": next_action,
        "continuous_state_separation_certified": continuous_yes,
        "metrics": {
            "matched_chi_tests": len(tests),
            "positive_paired_margin_tests": positive,
            "min_paired_margin_m": None if not math.isfinite(min_margin) else min_margin,
            "max_paired_margin_m": None if not math.isfinite(max_margin) else max_margin,
            "p2c_binding_tests": p2c_binding,
            "restart_binding_tests": restart_binding,
            "lower_iterations": lower.iterations,
            "lower_final_delta_m": lower.final_delta,
            "lower_monotonicity_violations": lower.monotonicity_violations,
            "upper_iterations": upper.iterations,
            "restart_action_width": args.restart_action_width,
            "restart_point_actions": len(actions),
            "stage_interval_width": args.stage_interval_width,
            "stage_intervals": len(intervals),
            "fallback_lower_resolution": args.fallback_lower_resolution,
            "point_model_max_lower_minus_upper_m": max_lower_minus_upper,
            **{f"p2c_{k}": v for k, v in fb_metrics.items()},
            "upstream_state_box_min_abs_gap_m": hc.get("metrics", {}).get("state_box_min_abs_gap_m"),
        },
        "gates": {
            "upstream_r9hd_pass": True,
            "upper_same_action_fixed_point_converged": bool(upper.converged),
            "point_action_lower_fixed_point_converged": bool(lower.converged),
            "point_model_lower_le_upper": order_viol == 0,
            "p2c_existing_lower_bracket_ordered": fb_metrics["ordered_failures_same_resolution"] == 0,
            "continuous_action_lower_gfp_certified": continuous_action_lower_certified,
            "independent_p2c_lower_checker_pass": independent_p2c,
            "full_augmented_interval_gfp_solved": full_interval_gfp,
        },
        "artifacts": {"converged_lower_bounds_csv": str(csvp)},
    }
    text = json.dumps(out, indent=2, sort_keys=True)
    atomic_write(result, text); atomic_write(latest, text)
    mfiles = [
        Path(__file__), result, csvp,
        RESULTS / "P3B1_R9HD_LATEST.json",
        RESULTS / "P3B1_R9HC_LATEST.json",
        RESULTS / "P3B1_R9HB_LATEST.json",
        RESULTS / "P3B1_R9G_LATEST.json",
    ]
    atomic_write(manifest, "".join(f"{sha256_file(p)}  {p}\n" for p in mfiles if p.exists()))

    print("=== R9-H-E DECISION ===", flush=True)
    print(
        f"R9HE_LOWER_FIXED_POINT converged={'YES' if lower.converged else 'NO'} "
        f"iterations={lower.iterations} final_delta_m={lower.final_delta:.12g}",
        flush=True,
    )
    print(
        f"R9HE_PAIRED_REASSESS positive_tests={positive}/{len(tests)} "
        f"min_margin_m={min_margin:.12g} max_margin_m={max_margin:.12g}",
        flush=True,
    )
    print(
        f"R9HE_BINDING p2c_lower_clipping_tests={p2c_binding} "
        f"restart_or_service_tests={restart_binding}",
        flush=True,
    )
    print("CONTINUOUS_ACTION_LOWER_GFP_CERTIFIED=NO", flush=True)
    print("P2C_INDEPENDENT_LOWER_CHECKER=NO", flush=True)
    print("FULL_AUGMENTED_INTERVAL_GFP_SOLVED=NO", flush=True)
    print("R9HE_EXECUTION=PASS", flush=True)
    print(f"R9HE_CLASSIFICATION={classification}", flush=True)
    print("CONTINUOUS_STATE_SEPARATION_CERTIFIED=NO", flush=True)
    print(f"R9HE_NEXT_ACTION={next_action}", flush=True)
    print(f"RESULT_JSON={result}", flush=True)
    print(f"MANIFEST={manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
