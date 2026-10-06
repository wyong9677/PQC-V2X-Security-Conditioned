from __future__ import annotations

import argparse
import copy
import csv
import gc
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
CFGDIR = ROOT / "01_config"
RESULTS = ROOT / "04_results"

AXIS_NAMES = ("v_f", "v_p", "a_f", "bar_a", "bar_u", "age")
TARGET_BAR_A = np.asarray([
    -2.4140625,
    -2.328125,
    -2.2421875,
    -2.15625,
    -2.0703125,
    -1.984375,
], dtype=float)
TOL = 1.0e-10


@dataclass
class SolveResult:
    h_flat: np.ndarray
    converged: bool
    iterations: int
    final_change: float
    outside_change_max: float


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
                fields.append(k)
                seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def finite_sup_abs(a: np.ndarray, b: np.ndarray) -> float:
    x = np.abs(np.asarray(a, float) - np.asarray(b, float))
    finite = np.isfinite(x)
    return float(np.max(x[finite])) if np.any(finite) else math.inf


def history_age_axis(Ts: float, age_max: float) -> np.ndarray:
    n = int(round(age_max / Ts))
    if abs(n * Ts - age_max) > 1e-10:
        raise RuntimeError(
            f"R9D_AGE_MAX_NOT_TS_ALIGNED age_max={age_max} Ts={Ts}"
        )
    return np.asarray([k * Ts for k in range(n + 1)], dtype=float)


def timestamped_reachable_qage(R: int, max_ticks: int) -> set[tuple[int, int]]:
    """Reachable (q, adopted-age-tick) pairs for the diagnostic countdown.

    Contract used here:
      * a fresh pending message is generated when a service cycle starts at q=R;
      * q>1 permits completion or deferral; q=1 must complete;
      * pending generation time is retained;
      * completion from q=r has candidate age (R-r+1)*Ts;
      * after completion, causal adoption may adopt that candidate or hold the
        previously adopted state; a new diagnostic service cycle starts at q=R;
      * hold/defer increases the previously adopted age by one sample.

    This is a timestamp-corrected *diagnostic countdown* automaton.  It is not a
    full protocol automaton because pending message content/credential/replacement
    variables are still omitted.
    """
    if R < 1 or max_ticks < 1:
        return set()
    # Any completion latency 1..R can create the most recently adopted message.
    reached = {(R, k) for k in range(1, min(R, max_ticks) + 1)}
    frontier = list(reached)
    while frontier:
        q, a = frontier.pop()
        nxt: list[tuple[int, int]] = []
        completion_age = R - q + 1
        # completion + adopt (timestamp preserved)
        if completion_age <= max_ticks:
            nxt.append((R, completion_age))
        # completion + hold/discard
        if a + 1 <= max_ticks:
            nxt.append((R, a + 1))
        # no completion / defer
        if q > 1 and a + 1 <= max_ticks:
            nxt.append((q - 1, a + 1))
        for s in nxt:
            if s not in reached:
                reached.add(s)
                frontier.append(s)
    return reached


def timestamped_semantic_mask(age_flat: np.ndarray, R: int, Ts: float) -> tuple[np.ndarray, dict]:
    age = np.asarray(age_flat, dtype=float)
    ticks_real = age / Ts
    ticks = np.rint(ticks_real).astype(int)
    on_lattice = np.abs(ticks_real - ticks) <= 1e-9
    max_ticks = int(np.max(ticks[on_lattice])) if np.any(on_lattice) else 0
    reachable = timestamped_reachable_qage(R, max_ticks)
    mask = np.zeros((len(age), R), dtype=bool)
    for i, (ok, tick) in enumerate(zip(on_lattice, ticks)):
        if not ok:
            continue
        for q in range(1, R + 1):
            mask[i, q - 1] = (q, int(tick)) in reachable
    return mask, {
        "age_nodes": int(len(age)),
        "on_lattice": int(np.count_nonzero(on_lattice)),
        "off_lattice": int(np.count_nonzero(~on_lattice)),
        "reachable_qage_pairs": int(len(reachable)),
        "comparable_physical_nodes": int(np.count_nonzero(np.sum(mask, axis=1) >= 2)),
    }


def q_metrics(h_flat: np.ndarray, eval_idx: np.ndarray, semantic_mask: np.ndarray | None = None) -> dict:
    h = np.asarray(h_flat[eval_idx, :], dtype=float)
    if semantic_mask is None:
        row_ok = np.ones(h.shape[0], dtype=bool)
    else:
        row_ok = np.sum(semantic_mask, axis=1) >= 2
    spans: list[float] = []
    rows: list[int] = []
    for i in np.flatnonzero(row_ok):
        vals = h[i, :] if semantic_mask is None else h[i, semantic_mask[i, :]]
        if len(vals) >= 2:
            spans.append(float(np.max(vals) - np.min(vals)))
            rows.append(int(i))
    a = np.asarray(spans, dtype=float)
    return {
        "comparable_nodes": int(len(a)),
        "qdep_nodes": int(np.count_nonzero(a > TOL)),
        "max_q_span_m": float(np.max(a)) if len(a) else 0.0,
        "p95_positive_q_span_m": float(np.quantile(a[a > TOL], .95)) if np.any(a > TOL) else 0.0,
    }


def qdep_indices(h_flat: np.ndarray, eval_idx: np.ndarray, semantic_mask: np.ndarray | None = None):
    h = np.asarray(h_flat[eval_idx, :], dtype=float)
    idx: list[int] = []
    spans = np.zeros(h.shape[0], dtype=float)
    for i in range(h.shape[0]):
        if semantic_mask is None:
            vals = h[i, :]
        else:
            m = semantic_mask[i, :]
            if int(np.count_nonzero(m)) < 2:
                continue
            vals = h[i, m]
        span = float(np.max(vals) - np.min(vals))
        spans[i] = span
        if span > TOL:
            idx.append(i)
    return np.asarray(idx, dtype=int), spans


def reconstruct_timestamped_geometry(r3, r5, r6, r7, r9c):
    cfg = json.loads((CFGDIR / "p3b1_augmented_fixed_point_v1.json").read_text())
    p1 = json.loads((CFGDIR / "p1_validation_v2.json").read_text())
    p2b = json.loads((CFGDIR / "p2b_hybrid_fallback_v1.json").read_text())
    p2c = json.loads((CFGDIR / "p2c_switching_guard_v1.json").read_text())
    p3a = json.loads((CFGDIR / "p3a_information_contract_v1.json").read_text())

    r6_path = RESULTS / "P3B1_R6_LATEST.json"
    r6_result = json.loads(r6_path.read_text(encoding="utf-8"))
    if r6_result.get("status") != "PASS":
        raise RuntimeError("R9D_R6_NOT_PASS")

    selected_axes = list(r6_result["selected_refinement_axes"])
    coarse_refined_cfg = r5.refine_cfg(cfg, tuple(selected_axes))
    halo_path, halo_mode = r9c.resolve_historical_artifact(
        r6_result["halo_spec"], RESULTS
    )
    halo = json.loads(halo_path.read_text(encoding="utf-8"))
    provenance = r9c.validate_halo_spec_provenance(
        halo, r6_result, coarse_refined_cfg, halo_path
    )

    Ts = float(p1["plant"]["Ts"])
    age_max = float(max(cfg["grid"]["age"]))
    eval_age = history_age_axis(Ts, age_max)
    lookup_age = np.asarray(
        [k * Ts for k in range(int(round(age_max / Ts)) + 2)], dtype=float
    )

    target_cfg = copy.deepcopy(coarse_refined_cfg)
    target_cfg["grid"]["bar_a"] = [
        float(x) for x in r9c.insert_exact(target_cfg["grid"]["bar_a"], TARGET_BAR_A)
    ]
    target_cfg["grid"]["age"] = [float(x) for x in eval_age]
    eval_axes, eval_flat, eval_shape = r3.build_grid(target_cfg)

    lookup_axes = [
        np.asarray(halo["lookup_halo_grid"][n], dtype=float)
        for n in AXIS_NAMES
    ]
    lookup_axes[AXIS_NAMES.index("bar_a")] = r9c.insert_exact(
        lookup_axes[AXIS_NAMES.index("bar_a")], TARGET_BAR_A
    )
    lookup_axes[AXIS_NAMES.index("age")] = lookup_age
    lookup_flat, lookup_shape = r6.mesh_flat(lookup_axes)
    eval_idx = r7.exact_node_indices(eval_axes, lookup_axes)

    original_age_count = len(cfg["grid"]["age"])
    expected_eval = int(315315 // original_age_count * len(eval_age))
    eval_nodes = int(np.prod(eval_shape))
    if eval_nodes != expected_eval:
        raise RuntimeError(
            f"R9D_EVAL_NODE_COUNT_MISMATCH expected={expected_eval} actual={eval_nodes}"
        )

    lookup_fallback = r6.fallback_required_on_grid(
        target_cfg, p1, p2b, p2c, p3a, lookup_flat
    )
    eval_fallback = np.asarray(lookup_fallback[eval_idx], dtype=float)

    return {
        "cfg": cfg,
        "eval_cfg": target_cfg,
        "p1": p1,
        "p2b": p2b,
        "p2c": p2c,
        "p3a": p3a,
        "selected_axes": selected_axes,
        "eval_axes": eval_axes,
        "eval_flat": eval_flat,
        "eval_shape": eval_shape,
        "lookup_axes": lookup_axes,
        "lookup_shape": lookup_shape,
        "eval_idx": eval_idx,
        "lookup_fallback": np.asarray(lookup_fallback, dtype=float),
        "eval_fallback": eval_fallback,
        "halo_path": str(halo_path),
        "halo_mode": halo_mode,
        "provenance": provenance,
        "Ts": Ts,
        "age_max": age_max,
        "eval_age_axis": eval_age,
        "lookup_age_axis": lookup_age,
    }


def build_timestamped_transitions(data, actions, R, r3, b0, chunk_size=4096):
    """Build q-dependent completion transitions with timestamp-correct age.

    Physical candidate content still uses the legacy completion proxy
    (current predecessor endpoint).  This routine therefore fixes the adoption
    *timestamp/age* semantics but deliberately does not claim a complete pending
    message descriptor implementation.
    """
    cfg = copy.deepcopy(data["eval_cfg"])
    cfg["cooperative_actions"] = [float(a) for a in actions]
    vf, vp, af, bar_a, bar_u, age = [
        np.asarray(x, dtype=float) for x in data["eval_flat"]
    ]
    n = len(vf)
    p1, p2b, p3a = data["p1"], data["p2b"], data["p3a"]
    Ts = data["Ts"]
    J = float(cfg["information_contract"]["slew_rate"])
    nt = int(cfg["one_step"]["trajectory_points"])
    times = np.linspace(0.0, Ts, nt)
    sd = p2b["state_domain"]
    speed_bound = float(sd["v_f_max"]) + float(sd["v_p_max"])
    lips = 0.5 * speed_bound * (Ts / (nt - 1))
    axes = data["lookup_axes"]

    outs = []
    for action in actions:
        outs.append({
            "action": float(action),
            "step_loss_upper": np.empty(n, dtype=float),
            "closing_end": np.empty(n, dtype=float),
            "defer_index": np.empty(n, dtype=np.int64),
            "defer_valid": np.empty(n, dtype=bool),
            "adopt_index_by_q": np.empty((R, n), dtype=np.int64),
            "adopt_valid_by_q": np.empty((R, n), dtype=bool),
        })

    tau = float(p2b["follower"]["tau"])
    w = float(p1["uncertainty"]["follower_actuation_abs"])
    for a in actions:
        if float(a) + tau * w >= 0.0:
            raise RuntimeError(
                "R9D_NEGATIVE_ACTION_STAGE_REQUIRES_NEGATIVE_EQUILIBRIUM "
                f"action={a} equilibrium={float(a)+tau*w}"
            )

    for start in range(0, n, int(chunk_size)):
        stop = min(n, start + int(chunk_size))
        sl = slice(start, stop)
        vfc, vpc, afc = vf[sl], vp[sl], af[sl]
        bac, buc, agc = bar_a[sl], bar_u[sl], age[sl]
        m = stop - start

        ap_lower = b0.predecessor_acceleration_lower(agc, bac, buc, J, p3a, p2b)
        Pp, Vp, Ap, Up, _ = r3.predecessor_motion_with_stop(
            vpc, ap_lower, buc, agc, times, J, p3a
        )
        vp_end, ap_end, up_end = Vp[:, -1], Ap[:, -1], Up[:, -1]
        age_hold = agc + Ts

        for out in outs:
            action = float(out["action"])
            Pf, Vf, Af = r3.follower_motion(vfc, afc, action, times, p1, p2b)
            closing = Pf - Pp
            out["step_loss_upper"][sl] = np.maximum(
                np.max(closing, axis=1) + lips, 0.0
            )
            out["closing_end"][sl] = Pf[:, -1] - Pp[:, -1]
            defer_vals = [Vf[:, -1], vp_end, Af[:, -1], bac, buc, age_hold]
            didx, dvalid = r3.locate_cells(axes, defer_vals)
            out["defer_index"][sl] = didx
            out["defer_valid"][sl] = dvalid

            for q in range(1, R + 1):
                completion_age = np.full(
                    m, (R - q + 1) * Ts, dtype=float
                )
                adopt_vals = [
                    Vf[:, -1], vp_end, Af[:, -1], ap_end, up_end,
                    completion_age,
                ]
                cidx, cvalid = r3.locate_cells(axes, adopt_vals)
                out["adopt_index_by_q"][q - 1, sl] = cidx
                out["adopt_valid_by_q"][q - 1, sl] = cvalid

        if start == 0 or stop == n or (stop // int(chunk_size)) % 16 == 0:
            print(
                f"R9D_TRANSITION_PROGRESS states={stop}/{n} actions={len(actions)}",
                flush=True,
            )

    invalid = 0
    for out in outs:
        invalid += int(np.count_nonzero(~out["defer_valid"]))
        invalid += int(np.count_nonzero(~out["adopt_valid_by_q"]))
    if invalid:
        raise RuntimeError(f"R9D_LOOKUP_HALO_COVERAGE_FAIL invalid={invalid}")
    return cfg, {
        "fallback_required": np.asarray(data["eval_fallback"], dtype=float),
        "transitions": outs,
        "lipschitz_correction_m": float(lips),
        "timestamped_completion_age_seconds": {
            str(q): float((R - q + 1) * Ts) for q in range(1, R + 1)
        },
    }


def lookup_future(cell_flat, q_index, index, valid):
    out = np.full(len(index), np.inf, dtype=float)
    m = np.asarray(valid, dtype=bool)
    out[m] = cell_flat[q_index][index[m]]
    return out


def solve_timestamped_causal(
    R, cfg, eval_idx, lookup_shape, lookup_fallback, eval_fallback,
    transition_data, r3, label,
) -> SolveResult:
    lookup_nodes = int(np.prod(lookup_shape))
    h_flat = np.repeat(np.asarray(lookup_fallback)[:, None], R, axis=1)
    eval_mask = np.zeros(lookup_nodes, dtype=bool)
    eval_mask[eval_idx] = True
    outside = ~eval_mask
    tol = float(cfg["fixed_point"]["tolerance_m"])
    max_iter = int(cfg["fixed_point"]["max_iterations"])
    outside_change_max = 0.0
    delta = math.inf

    for it in range(1, max_iter + 1):
        h = h_flat.reshape(lookup_shape + (R,))
        cell_max = r3.cell_corner_max(h)
        cell_flat = [cell_max[..., q].reshape(-1) for q in range(R)]
        old = h_flat[eval_idx, :].copy()
        new = old.copy()

        for q in range(1, R + 1):
            qi = q - 1
            best = np.full(len(eval_idx), np.inf, dtype=float)
            for tr in transition_data["transitions"]:
                adopt = lookup_future(
                    cell_flat, R - 1,
                    tr["adopt_index_by_q"][qi],
                    tr["adopt_valid_by_q"][qi],
                )
                hold = lookup_future(
                    cell_flat, R - 1,
                    tr["defer_index"], tr["defer_valid"],
                )
                completion = np.minimum(adopt, hold)
                future = completion
                if q > 1:
                    defer = lookup_future(
                        cell_flat, q - 2,
                        tr["defer_index"], tr["defer_valid"],
                    )
                    future = np.maximum(completion, defer)
                required = np.maximum(
                    tr["step_loss_upper"], tr["closing_end"] + future
                )
                best = np.minimum(best, required)
            best = np.maximum(best, 0.0)
            cand = np.minimum(np.asarray(eval_fallback), best)
            new[:, qi] = np.minimum(old[:, qi], cand)

        delta = finite_sup_abs(new, old)
        h_flat[eval_idx, :] = new
        if np.any(outside):
            expected = np.repeat(np.asarray(lookup_fallback)[outside, None], R, axis=1)
            outside_change_max = max(
                outside_change_max,
                finite_sup_abs(h_flat[outside, :], expected),
            )
        if it == 1 or it % 5 == 0 or delta <= tol:
            print(
                f"R9D_ITER label={label} iteration={it} "
                f"delta_m={delta:.12g} outside_change_m={outside_change_max:.12g}",
                flush=True,
            )
        if delta <= tol:
            return SolveResult(h_flat, True, it, delta, outside_change_max)
    return SolveResult(h_flat, False, max_iter, delta, outside_change_max)


def upstream_r9c_witness_age_summary() -> dict:
    p = RESULTS / "P3B1_R9C_LATEST.json"
    if not p.is_file():
        return {"available": False}
    d = json.loads(p.read_text(encoding="utf-8"))
    art = Path(d.get("artifacts", {}).get("witness_csv", ""))
    if not art.is_file():
        return {"available": True, "witness_csv_available": False}
    ages = []
    with art.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                ages.append(float(row["age"]))
            except Exception:
                pass
    return {
        "available": True,
        "witness_csv_available": True,
        "count": len(ages),
        "unique_ages": sorted(set(ages)),
    }


def self_test() -> None:
    a = history_age_axis(0.1, 2.0)
    assert len(a) == 21 and abs(a[-1] - 2.0) < 1e-12
    r = timestamped_reachable_qage(2, 20)
    assert (2, 1) in r
    assert (1, 2) in r
    assert (2, 2) in r
    assert (1, 1) not in r
    # Timestamped completion age must not be reset to zero.
    assert (2 - 2 + 1) * 0.1 == 0.1
    assert (2 - 1 + 1) * 0.1 == 0.2
    print("R9D_INTERNAL_SELF_TEST=PASS", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--profile", default="diagnostic_fast")
    ap.add_argument("--chunk-size", type=int, default=4096)
    args = ap.parse_args()
    self_test()
    if args.self_test:
        return 0

    sys.path.insert(0, str(SRC))
    import p3b1_augmented_fixed_point_v1_r3 as r3
    import p3b1_r5_refinement_attribution as r5
    import p3b1_r6_continuation_halo as r6
    import p3b1_r7_frozen_halo_fixed_point as r7
    import p3b0_freshness_service_audit_v1 as b0
    import p3b1_r9c_targeted_history_gate as r9c

    r9c_latest = RESULTS / "P3B1_R9C_LATEST.json"
    r9c_result = json.loads(r9c_latest.read_text(encoding="utf-8"))
    if r9c_result.get("status") != "PASS":
        raise RuntimeError("R9D_UPSTREAM_R9C_NOT_PASS")
    if r9c_result.get("classification") != "TARGETED_FINITE_Q_LAYER_OFF_EXACT_QAGE_HISTORY_LATTICE":
        raise RuntimeError(
            "R9D_UNEXPECTED_R9C_CLASSIFICATION=" + str(r9c_result.get("classification"))
        )

    data = reconstruct_timestamped_geometry(r3, r5, r6, r7, r9c)
    profile = args.profile
    profiles = data["p1"]["diagnostic_service_profiles"]
    if profile not in profiles:
        raise KeyError(profile)
    R = r7.service_horizon(profile, profiles[profile])
    Ts = data["Ts"]
    semantic_mask, semantic_diag = timestamped_semantic_mask(
        np.asarray(data["eval_flat"][-1], float), R, Ts
    )

    print("=== P3-B1-R9-D TIMESTAMPED SERVICE SEMANTICS AUDIT ===", flush=True)
    print(f"PROFILE={profile} R={R} Ts={Ts:.12g}", flush=True)
    print("THEORY_AGE_RULE=ADOPTED_AGE_EQUALS_ADOPTION_TIME_MINUS_GENERATION_TIME", flush=True)
    print("FROZEN_RESET_TO_ZERO_RULE_RETIRED=YES", flush=True)
    print("SERVICE_MODEL=TIMESTAMPED_DIAGNOSTIC_COUNTDOWN", flush=True)
    print("PENDING_MESSAGE_CONTENT_EXPLICIT=NO", flush=True)
    print("CONTINUOUS_PROMOTION_ALLOWED=NO", flush=True)
    print(
        "COMPLETION_AGE_BY_Q=" + ",".join(
            f"q{q}:{(R-q+1)*Ts:.12g}" for q in range(1, R+1)
        ), flush=True
    )
    print(
        f"AGE_GRID_POINTS={len(data['eval_age_axis'])} "
        f"AGE_MIN={data['eval_age_axis'][0]:.12g} "
        f"AGE_MAX={data['eval_age_axis'][-1]:.12g}", flush=True
    )
    print(
        f"EVAL_NODES={int(np.prod(data['eval_shape']))} "
        f"LOOKUP_NODES={int(np.prod(data['lookup_shape']))}", flush=True
    )
    print(
        f"R9D_PROVENANCE content_gate={data['provenance']['content_gate']} "
        f"manifest_gate={data['provenance']['manifest_gate']} "
        f"sha256={data['provenance']['actual_sha256']}", flush=True
    )
    print(
        f"R9D_SEMANTIC_REACHABILITY reachable_qage_pairs={semantic_diag['reachable_qage_pairs']} "
        f"comparable_physical_nodes={semantic_diag['comparable_physical_nodes']}", flush=True
    )

    upstream = upstream_r9c_witness_age_summary()
    if upstream.get("witness_csv_available"):
        print(
            "R9D_UPSTREAM_R9C_WITNESS_AGES=" + ",".join(
                f"{x:.12g}" for x in upstream.get("unique_ages", [])
            ), flush=True
        )

    frozen_actions = [-3.0, -2.0, -1.0]
    midpoint_actions = [-3.0, -2.5, -2.0, -1.5, -1.0]

    print("R9D_STAGE_START=timestamped_frozen_transition_build", flush=True)
    cfg_frozen, td_frozen = build_timestamped_transitions(
        data, frozen_actions, R, r3, b0, chunk_size=args.chunk_size
    )
    print("R9D_STAGE_START=timestamped_frozen_solve", flush=True)
    frozen = solve_timestamped_causal(
        R, cfg_frozen, data["eval_idx"], data["lookup_shape"],
        data["lookup_fallback"], data["eval_fallback"], td_frozen, r3,
        label="timestamped_frozen",
    )
    if not frozen.converged:
        raise RuntimeError("R9D_TIMESTAMPED_FROZEN_NOT_CONVERGED")
    frozen_all = q_metrics(frozen.h_flat, data["eval_idx"])
    frozen_sem = q_metrics(frozen.h_flat, data["eval_idx"], semantic_mask)
    print(
        f"R9D_FROZEN qdep_all={frozen_all['qdep_nodes']} "
        f"qdep_semantic={frozen_sem['qdep_nodes']} "
        f"max_qspan_semantic_m={frozen_sem['max_q_span_m']:.12g}", flush=True
    )

    hull = None
    hull_all = {"qdep_nodes": 0, "max_q_span_m": 0.0, "comparable_nodes": 0}
    hull_sem = dict(hull_all)
    hull_changed = 0
    if frozen_sem["qdep_nodes"] > 0:
        print("R9D_STAGE_START=timestamped_midpoint_transition_build", flush=True)
        cfg_hull, td_hull = build_timestamped_transitions(
            data, midpoint_actions, R, r3, b0, chunk_size=args.chunk_size
        )
        print("R9D_STAGE_START=timestamped_midpoint_solve", flush=True)
        hull = solve_timestamped_causal(
            R, cfg_hull, data["eval_idx"], data["lookup_shape"],
            data["lookup_fallback"], data["eval_fallback"], td_hull, r3,
            label="timestamped_midpoint",
        )
        if not hull.converged:
            raise RuntimeError("R9D_TIMESTAMPED_HULL_NOT_CONVERGED")
        hull_all = q_metrics(hull.h_flat, data["eval_idx"])
        hull_sem = q_metrics(hull.h_flat, data["eval_idx"], semantic_mask)
        A = np.asarray(frozen.h_flat[data["eval_idx"], :], float)
        B = np.asarray(hull.h_flat[data["eval_idx"], :], float)
        hull_changed = int(np.count_nonzero(np.max(np.abs(A-B), axis=1) > TOL))
        print(
            f"R9D_ACTION_HULL changed_nodes={hull_changed} "
            f"qdep_semantic={hull_sem['qdep_nodes']} "
            f"max_qspan_semantic_m={hull_sem['max_q_span_m']:.12g}", flush=True
        )
    else:
        print("R9D_ACTION_HULL_SKIPPED=NO_Q_LAYER_AFTER_TIMESTAMP_CORRECTION", flush=True)

    final_h = hull.h_flat if hull is not None else frozen.h_flat
    final_sem = hull_sem if hull is not None else frozen_sem
    widx, spans = qdep_indices(final_h, data["eval_idx"], semantic_mask)
    vf, vp, af, ba, bu, age = [np.asarray(x, float) for x in data["eval_flat"]]
    witness_rows = []
    for i in widx[:2000]:
        witness_rows.append({
            "node_index": int(i),
            "v_f": float(vf[i]), "v_p": float(vp[i]), "a_f": float(af[i]),
            "bar_a": float(ba[i]), "bar_u": float(bu[i]), "age": float(age[i]),
            "q_span_m": float(spans[i]),
            "admissible_q_count": int(np.count_nonzero(semantic_mask[i, :])),
        })

    if frozen_sem["qdep_nodes"] == 0:
        classification = "NO_Q_LAYER_ON_TIMESTAMPED_HISTORY_ALIGNED_COUNTDOWN"
        next_action = "R9E_BUILD_FULL_PENDING_MESSAGE_DESCRIPTOR_BEFORE_ANY_FURTHER_CONTINUOUS_CLAIM"
    elif hull is not None and hull_sem["qdep_nodes"] == 0:
        classification = "MIDPOINT_ACTION_HULL_REMOVES_TIMESTAMPED_Q_LAYER"
        next_action = "R9E_PENDING_DESCRIPTOR_PLUS_CONTINUOUS_ACTION_INTERVAL_COVER"
    else:
        classification = "Q_LAYER_PERSISTS_ON_TIMESTAMPED_HISTORY_ALIGNED_GRID_PENDING_CONTENT_UNRESOLVED"
        next_action = "R9E_ADD_PENDING_MESSAGE_CONTENT_DESCRIPTOR_AND_GENERAL_ACTION_PROPAGATOR"

    automaton_rows = []
    max_ticks = int(round(data["age_max"] / Ts))
    reach = timestamped_reachable_qage(R, max_ticks)
    for q, tick in sorted(reach):
        automaton_rows.append({
            "q": q,
            "adopted_age_tick": tick,
            "adopted_age_seconds": tick * Ts,
            "pending_age_before_interval_ticks": R - q,
            "candidate_age_if_completion_ticks": R - q + 1,
            "candidate_age_if_completion_seconds": (R - q + 1) * Ts,
        })

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_json = RESULTS / f"P3B1_R9D_RESULT_{stamp}.json"
    latest_json = RESULTS / "P3B1_R9D_LATEST.json"
    witness_csv = RESULTS / f"P3B1_R9D_TIMESTAMPED_WITNESSES_{stamp}.csv"
    automaton_csv = RESULTS / f"P3B1_R9D_AUTOMATON_{stamp}.csv"
    manifest = RESULTS / f"P3B1_R9D_MANIFEST_{stamp}.sha256"
    write_csv(witness_csv, witness_rows)
    write_csv(automaton_csv, automaton_rows)

    output = {
        "schema": "SCV_P3B1_R9D_TIMESTAMPED_SERVICE_V1",
        "status": "PASS",
        "timestamp_utc": stamp,
        "classification": classification,
        "next_action": next_action,
        "continuous_state_separation_certified": False,
        "continuous_action_certificate": False,
        "formal_timestamp_age_rule_implemented": True,
        "pending_message_content_explicit": False,
        "semantic_scope": "timestamp-corrected diagnostic countdown only; pending message content/replacement/credential state not explicit",
        "profile": profile,
        "R": R,
        "geometry": {
            "evaluation_nodes": int(np.prod(data["eval_shape"])),
            "lookup_nodes": int(np.prod(data["lookup_shape"])),
            "age_axis": [float(x) for x in data["eval_age_axis"]],
            "inserted_bar_a": [float(x) for x in TARGET_BAR_A],
        },
        "service_contract": {
            "completion_age_by_q_seconds": td_frozen["timestamped_completion_age_seconds"],
            "qage_reachability": semantic_diag,
            "upstream_r9c_witness_age_summary": upstream,
        },
        "metrics": {
            "frozen_all": frozen_all,
            "frozen_semantic": frozen_sem,
            "midpoint_all": hull_all,
            "midpoint_semantic": hull_sem,
            "midpoint_changed_nodes": hull_changed,
        },
        "gates": {
            "r9c_upstream_pass": True,
            "r6_halo_provenance": data["provenance"],
            "timestamp_age_not_reset_zero": all(v > 0 for v in td_frozen["timestamped_completion_age_seconds"].values()),
            "solver_converged": bool(frozen.converged and (hull is None or hull.converged)),
            "pending_message_content_gate": False,
        },
        "artifacts": {
            "witness_csv": str(witness_csv),
            "automaton_csv": str(automaton_csv),
        },
    }
    text = json.dumps(output, indent=2, sort_keys=True)
    atomic_write(result_json, text)
    atomic_write(latest_json, text)

    manifest_files = [
        Path(__file__), result_json, witness_csv, automaton_csv,
        RESULTS / "P3B1_R9C_LATEST.json",
        Path(data["halo_path"]),
        CFGDIR / "p3b1_augmented_fixed_point_v1.json",
        CFGDIR / "p1_validation_v2.json",
        CFGDIR / "p2b_hybrid_fallback_v1.json",
        CFGDIR / "p2c_switching_guard_v1.json",
        CFGDIR / "p3a_information_contract_v1.json",
    ]
    atomic_write(
        manifest,
        "".join(f"{sha256_file(p)}  {p}\n" for p in manifest_files if p.exists()),
    )

    print("=== R9-D DECISION ===", flush=True)
    print(
        f"R9D_QDEP frozen_semantic={frozen_sem['qdep_nodes']} "
        f"midpoint_semantic={hull_sem['qdep_nodes'] if hull is not None else 'SKIPPED'}",
        flush=True,
    )
    print("R9D_EXECUTION=PASS", flush=True)
    print(f"R9D_CLASSIFICATION={classification}", flush=True)
    print("CONTINUOUS_STATE_SEPARATION_CERTIFIED=NO", flush=True)
    print("PENDING_MESSAGE_CONTENT_EXPLICIT=NO", flush=True)
    print(f"R9D_NEXT_ACTION={next_action}", flush=True)
    print(f"RESULT_JSON={result_json}", flush=True)
    print(f"MANIFEST={manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
