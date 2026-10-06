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

ENTRY = 0
FRAGMENTED = 1
VERIFYING = 2
VERIFYING_LAST = 3
STAGE_NAMES = ("PENDING_ENTRY", "FRAGMENTED", "VERIFYING", "VERIFYING_LAST")


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


def finite_sup_abs(a: np.ndarray, b: np.ndarray) -> float:
    d = np.abs(np.asarray(a, float) - np.asarray(b, float))
    d = d[np.isfinite(d)]
    return float(np.max(d)) if d.size else 0.0


def cell_corner_min(nodal: np.ndarray) -> np.ndarray:
    out = np.asarray(nodal, dtype=float)
    for axis in range(6):
        l = [slice(None)] * out.ndim
        r = [slice(None)] * out.ndim
        l[axis] = slice(0, -1)
        r[axis] = slice(1, None)
        out = np.minimum(out[tuple(l)], out[tuple(r)])
    return out


def vector_interval_cells(axis: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized equivalent of R9-F cells_intersecting_interval.

    Returns inclusive first/last cell indices and a validity mask.
    Exact node hits deliberately include both adjacent cells when possible,
    matching the conservative interval-intersection semantics of R9-F.
    """
    axis = np.asarray(axis, float)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    a = np.minimum(lo, hi)
    b = np.maximum(lo, hi)
    valid = (b >= axis[0] - 1e-12) & (a <= axis[-1] + 1e-12)
    aa = np.maximum(a, axis[0])
    bb = np.minimum(b, axis[-1])
    # first cell with right endpoint >= lo
    first = np.searchsorted(axis[1:], aa - 1e-12, side="left")
    # last cell with left endpoint <= hi
    last = np.searchsorted(axis[:-1], bb + 1e-12, side="right") - 1
    first = np.clip(first, 0, len(axis) - 2).astype(np.int16)
    last = np.clip(last, 0, len(axis) - 2).astype(np.int16)
    valid &= first <= last
    return first, last, valid


@dataclass
class RangeGroups:
    valid_mask: np.ndarray
    groups: list[tuple[np.ndarray, int, int, int, int]]
    max_rectangle_cells: int
    unique_rectangles: int


def build_descriptor_range_groups(data: dict, pending_age: float, b0) -> RangeGroups:
    """Build robust chi rectangles for every evaluation state.

    The service stage determines pending_age.  The rectangle itself is the same
    stale-state/slew over-approximation used in R9-F/R9-G, but now it is part of
    a closed service/restart fixed-point operator rather than an external
    terminal continuation.
    """
    vf, vp, af, ba, bu, age = [np.asarray(x, float) for x in data["eval_flat"]]
    del vf, vp, af
    J = float(data["eval_cfg"]["information_contract"]["slew_rate"])
    p3a, p2b = data["p3a"], data["p2b"]
    elapsed = age - float(pending_age)
    valid = elapsed >= -1e-10
    e = np.maximum(elapsed, 0.0)
    tau = float(p3a.get("predecessor", {}).get("tau", p2b["predecessor"]["tau"]))
    nominal = bu + (ba - bu) * np.exp(-e / tau)
    lower = np.asarray(
        b0.predecessor_acceleration_lower(e, ba, bu, J, p3a, p2b), float
    )
    radius = np.maximum(nominal - lower, 0.0)
    sd = p2b["state_domain"]
    a_lo = np.maximum(float(sd["a_p_min"]), lower)
    a_hi = np.minimum(float(sd["a_p_max"]), nominal + radius)
    pred = p3a.get("predecessor", {})
    u_min = float(pred.get("command_min", sd.get("a_p_min", -8.0)))
    u_max = float(pred.get("command_max", sd.get("a_p_max", 3.0)))
    u_lo = np.maximum(u_min, bu - J * e)
    u_hi = np.minimum(u_max, bu + J * e)
    valid &= a_lo <= a_hi + 1e-12
    valid &= u_lo <= u_hi + 1e-12

    ba_axis = np.asarray(data["lookup_axes"][3], float)
    bu_axis = np.asarray(data["lookup_axes"][4], float)
    ba0, ba1, ok_a = vector_interval_cells(ba_axis, a_lo, a_hi)
    bu0, bu1, ok_u = vector_interval_cells(bu_axis, u_lo, u_hi)
    valid &= ok_a & ok_u

    # Compress millions of source states into a small number of distinct
    # rectangle classes.  Physical coordinates are handled action-by-action.
    n_ba = len(ba_axis) - 1
    n_bu = len(bu_axis) - 1
    key = (((ba0.astype(np.int64) * n_ba + ba1.astype(np.int64)) * n_bu + bu0.astype(np.int64)) * n_bu + bu1.astype(np.int64))
    groups: list[tuple[np.ndarray, int, int, int, int]] = []
    max_cells = 0
    if np.any(valid):
        for k in np.unique(key[valid]):
            pos = np.flatnonzero(valid & (key == k)).astype(np.int32)
            p = int(pos[0])
            x0, x1, y0, y1 = int(ba0[p]), int(ba1[p]), int(bu0[p]), int(bu1[p])
            max_cells = max(max_cells, (x1 - x0 + 1) * (y1 - y0 + 1))
            groups.append((pos, x0, x1, y0, y1))
    return RangeGroups(valid.astype(bool), groups, int(max_cells), len(groups))


def scalar_cell_index(axis: np.ndarray, x: float) -> tuple[int, bool]:
    axis = np.asarray(axis, float)
    ok = bool(x >= axis[0] - 1e-12 and x <= axis[-1] + 1e-12)
    i = int(np.searchsorted(axis, x, side="right") - 1)
    i = int(np.clip(i, 0, len(axis) - 2))
    return i, ok


def build_physical_transitions(data: dict, actions: list[float], r3, b0, r9e, *, chunk_size: int) -> tuple[dict, list[dict]]:
    """Generalized physical transitions without legacy adopt proxies.

    Only the common physical step and the hold/carry cell are stored.  Adoption
    successors are generated from the stage-specific chi rectangle during the
    fixed-point solve, which avoids importing the old current-endpoint adoption
    approximation into the closed service graph.
    """
    import copy

    cfg = copy.deepcopy(data["eval_cfg"])
    cfg["cooperative_actions"] = [float(a) for a in actions]
    vf, vp, af, ba, bu, age = [np.asarray(x, float) for x in data["eval_flat"]]
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
    cell_shape = tuple(len(a) - 1 for a in axes)
    tau = float(p2b["follower"]["tau"])
    w = float(p1["uncertainty"]["follower_actuation_abs"])

    outs: list[dict] = []
    for a in actions:
        outs.append({
            "action": float(a),
            "step_loss_upper": np.empty(n, dtype=float),
            "closing_end": np.empty(n, dtype=float),
            "hold_index": np.empty(n, dtype=np.int32),
            "vf_cell": np.empty(n, dtype=np.int16),
            "vp_cell": np.empty(n, dtype=np.int16),
            "af_cell": np.empty(n, dtype=np.int16),
        })

    for start in range(0, n, int(chunk_size)):
        stop = min(n, start + int(chunk_size))
        sl = slice(start, stop)
        vfc, vpc, afc = vf[sl], vp[sl], af[sl]
        bac, buc, agc = ba[sl], bu[sl], age[sl]
        ap = b0.predecessor_acceleration_lower(agc, bac, buc, J, p3a, p2b)
        Pp, Vp, Ap, Up, _ = r3.predecessor_motion_with_stop(vpc, ap, buc, agc, times, J, p3a)
        del Ap, Up
        vp_end = Vp[:, -1]
        for out in outs:
            Pf, Vf, Af, _ = r9e.generalized_follower_motion(
                vfc, afc, float(out["action"]), times, tau, w
            )
            closing = Pf - Pp
            out["step_loss_upper"][sl] = np.maximum(np.max(closing, axis=1) + lips, 0.0)
            out["closing_end"][sl] = Pf[:, -1] - Pp[:, -1]
            idx, ok = r3.locate_cells(
                axes,
                [Vf[:, -1], vp_end, Af[:, -1], bac, buc, agc + Ts],
            )
            if not bool(np.all(ok)):
                bad = int(np.count_nonzero(~ok))
                raise RuntimeError(f"R9HG_HOLD_LOOKUP_HALO_FAIL action={out['action']} invalid={bad}")
            idx32 = np.asarray(idx, dtype=np.int32)
            out["hold_index"][sl] = idx32
            coords = np.unravel_index(np.asarray(idx, dtype=np.int64), cell_shape)
            out["vf_cell"][sl] = np.asarray(coords[0], dtype=np.int16)
            out["vp_cell"][sl] = np.asarray(coords[1], dtype=np.int16)
            out["af_cell"][sl] = np.asarray(coords[2], dtype=np.int16)
        if start == 0 or stop == n or (stop // int(chunk_size)) % 16 == 0:
            print(f"R9HG_TRANSITION_PROGRESS states={stop}/{n} actions={len(actions)}", flush=True)
    return cfg, outs


def robust_adopt_future(entry_cells: np.ndarray, tr: dict, groups: RangeGroups, age_cell: int) -> np.ndarray:
    """Maximize PENDING_ENTRY continuation over every chi-intersecting cell."""
    n = len(tr["hold_index"])
    out = np.full(n, np.inf, dtype=float)
    c0 = np.asarray(tr["vf_cell"], dtype=np.intp)
    c1 = np.asarray(tr["vp_cell"], dtype=np.intp)
    c2 = np.asarray(tr["af_cell"], dtype=np.intp)
    for pos32, ba0, ba1, bu0, bu1 in groups.groups:
        pos = np.asarray(pos32, dtype=np.intp)
        vals = np.full(len(pos), -np.inf, dtype=float)
        for iba in range(ba0, ba1 + 1):
            for ibu in range(bu0, bu1 + 1):
                vals = np.maximum(
                    vals,
                    entry_cells[c0[pos], c1[pos], c2[pos], iba, ibu, age_cell],
                )
        out[pos] = vals
    if np.any(groups.valid_mask & ~np.isfinite(out)):
        raise RuntimeError("R9HG_DESCRIPTOR_RECTANGLE_LOOKUP_FAIL")
    return out


@dataclass
class StageSolve:
    h_flat: np.ndarray
    converged: bool
    iterations: int
    final_delta: float
    monotonicity_violations: int


def stage_masks(data: dict) -> list[np.ndarray]:
    age = np.asarray(data["eval_flat"][5], float)
    Ts = float(data["Ts"])
    return [
        np.ones(len(age), dtype=bool),
        age >= Ts - 1e-10,
        age >= Ts - 1e-10,
        age >= 2.0 * Ts - 1e-10,
    ]


def solve_stage_fixed_point(
    *, data: dict, cfg: dict, transitions: list[dict], fallback_eval: np.ndarray,
    groups_v: RangeGroups, groups_last: RangeGroups, candidate_age_v_cell: int,
    candidate_age_last_cell: int, r3, mode: str, label: str,
) -> StageSolve:
    lookup_shape = tuple(data["lookup_shape"])
    lookup_nodes = int(np.prod(lookup_shape))
    eval_idx = np.asarray(data["eval_idx"], dtype=np.int64)
    n = len(eval_idx)
    masks = stage_masks(data)
    tol = float(cfg["fixed_point"]["tolerance_m"])
    max_iter = int(cfg["fixed_point"]["max_iterations"])
    lips = 0.5 * (
        float(data["p2b"]["state_domain"]["v_f_max"])
        + float(data["p2b"]["state_domain"]["v_p_max"])
    ) * (float(data["Ts"]) / (int(cfg["one_step"]["trajectory_points"]) - 1))

    if mode == "upper":
        base = np.asarray(data["lookup_fallback"], float)
        h = np.repeat(base[:, None], 4, axis=1)
    elif mode == "lower":
        h = np.zeros((lookup_nodes, 4), dtype=float)
    else:
        raise ValueError(mode)

    violations = 0
    delta = math.inf
    cell_shape = tuple(x - 1 for x in lookup_shape)
    for it in range(1, max_iter + 1):
        shaped = h.reshape(lookup_shape + (4,))
        cells = r3.cell_corner_max(shaped) if mode == "upper" else cell_corner_min(shaped)
        cflat = [cells[..., s].reshape(-1) for s in range(4)]
        old = h[eval_idx, :].copy()
        best = np.full((n, 4), np.inf, dtype=float)

        for tr in transitions:
            hold = np.asarray(tr["hold_index"], dtype=np.intp)
            entry_hold = cflat[ENTRY][hold]
            f_hold = cflat[FRAGMENTED][hold]
            v_hold = cflat[VERIFYING][hold]
            last_hold = cflat[VERIFYING_LAST][hold]

            adopt_v = robust_adopt_future(cells[..., ENTRY], tr, groups_v, candidate_age_v_cell)
            adopt_last = robust_adopt_future(cells[..., ENTRY], tr, groups_last, candidate_age_last_cell)
            comp_v = np.minimum(adopt_v, entry_hold)
            comp_last = np.minimum(adopt_last, entry_hold)

            future_entry = np.maximum(f_hold, v_hold)
            future_frag = last_hold
            future_verify = np.maximum(comp_v, last_hold)
            future_last = comp_last

            step = np.asarray(tr["step_loss_upper"], float)
            if mode == "lower":
                step = np.maximum(step - lips, 0.0)
            close = np.asarray(tr["closing_end"], float)
            futures = (future_entry, future_frag, future_verify, future_last)
            for s, fut in enumerate(futures):
                req = np.maximum(step, close + fut)
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
                        f"R9HG_LOWER_MONOTONICITY_FAIL stage={STAGE_NAMES[s]} worst={worst}"
                    )
                new[m, s] = np.maximum(old[m, s], cand[m, s])
        delta = finite_sup_abs(new, old)
        h[eval_idx, :] = new
        if it == 1 or it % 5 == 0 or delta <= tol:
            print(
                f"R9HG_GFP_ITER label={label} mode={mode} iteration={it} "
                f"delta_m={delta:.12g} monotonicity_violations={violations}",
                flush=True,
            )
        if delta <= tol:
            return StageSolve(h, True, it, delta, violations)
    return StageSolve(h, False, max_iter, delta, violations)


def self_test() -> None:
    axis = np.asarray([0.0, 1.0, 2.0])
    f, l, ok = vector_interval_cells(axis, np.asarray([1.0, 0.2]), np.asarray([1.0, 1.8]))
    # exact x=1 intersects both [0,1] and [1,2]
    assert bool(ok[0]) and int(f[0]) == 0 and int(l[0]) == 1
    assert bool(ok[1]) and int(f[1]) == 0 and int(l[1]) == 1

    cell = np.zeros((1, 1, 1, 2, 2, 1), dtype=float)
    cell[0, 0, 0, 0, 0, 0] = 1.0
    cell[0, 0, 0, 1, 1, 0] = 4.0
    rg = RangeGroups(
        valid_mask=np.asarray([True]),
        groups=[(np.asarray([0], dtype=np.int32), 0, 1, 0, 1)],
        max_rectangle_cells=4,
        unique_rectangles=1,
    )
    tr = {
        "hold_index": np.asarray([0], dtype=np.int32),
        "vf_cell": np.asarray([0], dtype=np.int16),
        "vp_cell": np.asarray([0], dtype=np.int16),
        "af_cell": np.asarray([0], dtype=np.int16),
    }
    z = robust_adopt_future(cell, tr, rg, 0)
    assert abs(float(z[0]) - 4.0) < 1e-12
    print("R9HG_INTERNAL_SELF_TEST=PASS")


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
        raise ValueError("R9HG_BAD_ACTION_WIDTH")

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
    import p3b1_r9h_b_protocolized_paired_bounds as r9hb
    import p3b1_r9h_e_converged_lower_gfp as r9he

    hf = json.loads((RESULTS / "P3B1_R9HF_LATEST.json").read_text(encoding="utf-8"))
    expected = "RESTART_SERVICE_SEMANTICS_BIND_BEYOND_EXISTING_CELL_MIN_CONSERVATISM"
    if hf.get("status") != "PASS" or hf.get("classification") != expected:
        raise RuntimeError(f"R9HG_R9HF_GATE_FAIL classification={hf.get('classification')}")
    if int(hf.get("metrics", {}).get("cellmax_diagnostic_positive_margin_tests", -1)) != 0:
        raise RuntimeError("R9HG_EXPECTED_ZERO_CELLMAX_DIAGNOSTIC")

    ha = json.loads((RESULTS / "P3B1_R9HA_R1_LATEST.json").read_text(encoding="utf-8"))
    if not ha.get("protocol_graph", {}).get("pass", False):
        raise RuntimeError("R9HG_PROTOCOL_GRAPH_NOT_CLOSED")
    rg = json.loads((RESULTS / "P3B1_R9G_LATEST.json").read_text(encoding="utf-8"))
    tests = r9hb.load_stage_tests(rg)

    data = r9d.reconstruct_timestamped_geometry(r3, r5, r6, r7, r9c)
    Ts = float(data["Ts"])
    profile = "diagnostic_fast"
    R = r7.service_horizon(profile, data["p1"]["diagnostic_service_profiles"][profile])
    if R != 2:
        raise RuntimeError("R9HG_EXPECTED_R2")

    actions = r9hb.refinement_actions(float(args.action_width))
    print("=== P3-B1-R9-H-G FULL AUGMENTED SERVICE/RESTART GFP ===", flush=True)
    print("UPSTREAM_R9HF_RESTART_SEMANTICS_BINDING=PASS", flush=True)
    print("PROTOCOL_GRAPH=PENDING_ENTRY,FRAGMENTED,VERIFYING,VERIFYING_LAST", flush=True)
    print("RESTART_IS_INTERNAL_TO_FIXED_POINT=YES", flush=True)
    print("SERVICE_STAGE_TIMING=PENDING_ENTRY:0,FRAGMENTED:Ts,VERIFYING:Ts,VERIFYING_LAST:2Ts", flush=True)
    print("COMPLETION_RESTART_TARGET=PENDING_ENTRY", flush=True)
    print("CHI_MODEL=ROBUST_STAGE_CONDITIONED_ENVELOPE", flush=True)
    print("EXPLICIT_CHI_CORRELATIONS_CERTIFIED=NO", flush=True)
    print(f"POINT_ACTION_WIDTH={float(args.action_width):.12g}", flush=True)
    print(f"POINT_ACTION_COUNT={len(actions)}", flush=True)
    print("CONTINUOUS_STATE_SEPARATION_CERTIFIED=NO", flush=True)

    cfg, transitions = build_physical_transitions(
        data, actions, r3, b0, r9e, chunk_size=int(args.chunk_size)
    )

    print("R9HG_STAGE_START=build_stage_conditioned_chi_rectangles", flush=True)
    groups_v = build_descriptor_range_groups(data, Ts, b0)
    groups_last = build_descriptor_range_groups(data, 2.0 * Ts, b0)
    age_v, ok_v = scalar_cell_index(np.asarray(data["lookup_axes"][5], float), 2.0 * Ts)
    age_l, ok_l = scalar_cell_index(np.asarray(data["lookup_axes"][5], float), 3.0 * Ts)
    if not ok_v or not ok_l:
        raise RuntimeError("R9HG_COMPLETION_AGE_LOOKUP_FAIL")
    age = np.asarray(data["eval_flat"][5], float)
    valid_v_expected = age >= Ts - 1e-10
    valid_l_expected = age >= 2.0 * Ts - 1e-10
    if np.any(valid_v_expected & ~groups_v.valid_mask):
        raise RuntimeError("R9HG_VERIFYING_CHI_ENVELOPE_COVERAGE_FAIL")
    if np.any(valid_l_expected & ~groups_last.valid_mask):
        raise RuntimeError("R9HG_LAST_CHI_ENVELOPE_COVERAGE_FAIL")
    print(
        f"R9HG_CHI_RECTANGLES verifying_unique={groups_v.unique_rectangles} "
        f"verifying_max_cells={groups_v.max_rectangle_cells} "
        f"last_unique={groups_last.unique_rectangles} "
        f"last_max_cells={groups_last.max_rectangle_cells}",
        flush=True,
    )

    print("R9HG_STAGE_START=descending_upper_full_service_gfp", flush=True)
    upper = solve_stage_fixed_point(
        data=data, cfg=cfg, transitions=transitions,
        fallback_eval=np.asarray(data["eval_fallback"], float),
        groups_v=groups_v, groups_last=groups_last,
        candidate_age_v_cell=age_v, candidate_age_last_cell=age_l,
        r3=r3, mode="upper", label="r9hg_upper_full_service",
    )
    if not upper.converged:
        raise RuntimeError("R9HG_UPPER_FULL_SERVICE_GFP_NOT_CONVERGED")

    print("R9HG_STAGE_START=global_p2c_lower_bracket", flush=True)
    fallback_lower, p2c_metrics = r9he.compute_eval_fallback_lower(
        data, b0, sw, resolution=int(args.p2c_lower_resolution), chunk_size=int(args.chunk_size)
    )
    print("R9HG_STAGE_START=ascending_lower_full_service_gfp", flush=True)
    lower = solve_stage_fixed_point(
        data=data, cfg=cfg, transitions=transitions,
        fallback_eval=np.asarray(fallback_lower, float),
        groups_v=groups_v, groups_last=groups_last,
        candidate_age_v_cell=age_v, candidate_age_last_cell=age_l,
        r3=r3, mode="lower", label="r9hg_lower_full_service",
    )
    if not lower.converged:
        raise RuntimeError("R9HG_LOWER_FULL_SERVICE_GFP_NOT_CONVERGED")

    eval_idx = np.asarray(data["eval_idx"], dtype=np.int64)
    ue = np.asarray(upper.h_flat[eval_idx, :], float)
    le = np.asarray(lower.h_flat[eval_idx, :], float)
    if np.any(le > ue + 1e-8):
        bad = int(np.count_nonzero(le > ue + 1e-8))
        raise RuntimeError(f"R9HG_FULL_GRAPH_SANDWICH_FAIL count={bad}")
    sandwich_slack = float(np.max(le - ue))

    # The complete service contract fixes the phase of pending-message age.
    # R9-G intentionally scanned several matched pending ages; only b=Ts is the
    # FRAGMENTED/VERIFYING phase reached one interval after PENDING_ENTRY.
    aligned_rows = [r for r in tests if abs(float(r["pending_age_s"]) - Ts) <= 1e-9]
    unique_nodes = sorted({int(r["node_index"]) for r in tests})
    aligned_nodes = sorted({int(r["node_index"]) for r in aligned_rows})
    if not aligned_nodes:
        # Keep the audit executable even if the old diagnostic CSV did not happen
        # to include b=Ts; compare the unique R9-G witness physical nodes at the
        # protocol-correct F/V stage phase.
        aligned_nodes = [i for i in unique_nodes if age[i] >= Ts - 1e-10]

    valid_compare = age >= Ts - 1e-10
    upper_signed_all = ue[:, VERIFYING] - ue[:, FRAGMENTED]
    all_verify_more = int(np.count_nonzero(valid_compare & (upper_signed_all > TOL)))
    all_fragment_more = int(np.count_nonzero(valid_compare & (upper_signed_all < -TOL)))

    rows: list[dict] = []
    witness_upper_sensitive = 0
    witness_paired_positive = 0
    max_upper_gap = 0.0
    max_paired_margin = 0.0
    min_positive_paired = math.inf
    for i in aligned_nodes:
        upper_f = float(ue[i, FRAGMENTED])
        upper_v = float(ue[i, VERIFYING])
        lower_f = float(le[i, FRAGMENTED])
        lower_v = float(le[i, VERIFYING])
        signed = upper_v - upper_f
        paired = lower_v - upper_f
        sens = abs(signed) > TOL
        pos = paired > TOL
        witness_upper_sensitive += int(sens)
        witness_paired_positive += int(pos)
        max_upper_gap = max(max_upper_gap, abs(signed))
        max_paired_margin = max(max_paired_margin, paired)
        if pos:
            min_positive_paired = min(min_positive_paired, paired)
        rows.append({
            "node_index": int(i),
            "age_s": float(age[i]),
            "protocol_pending_age_s": Ts,
            "upper_fragmented_m": upper_f,
            "upper_verifying_m": upper_v,
            "upper_signed_verify_minus_fragment_m": signed,
            "lower_fragmented_m": lower_f,
            "lower_verifying_m": lower_v,
            "paired_lower_verify_minus_upper_fragment_m": paired,
            "upper_stage_sensitive": bool(sens),
            "paired_point_action_margin_positive": bool(pos),
        })

    if witness_upper_sensitive > 0 and witness_paired_positive > 0:
        classification = "FULL_AUGMENTED_SERVICE_RESTART_GFP_PRESERVES_STAGE_SEPARATION_AND_PAIRED_POINT_MARGIN_FOUND"
        next_action = "R9H_H_INTERVALIZE_FULL_AUGMENTED_GFP_AND_BUILD_INDEPENDENT_P2C_LOWER_CHECKER"
    elif witness_upper_sensitive > 0:
        classification = "FULL_AUGMENTED_SERVICE_RESTART_GFP_PRESERVES_STAGE_SEPARATION_BUT_PAIRED_LOWER_MARGIN_REMAINS_OPEN"
        next_action = "R9H_H_BUILD_INTERVAL_SOUND_FULL_GRAPH_LOWER_OPERATOR_AND_REASSESS_PAIRED_MARGIN"
    else:
        classification = "FULL_AUGMENTED_SERVICE_RESTART_GFP_COLLAPSES_PRIOR_LOCAL_STAGE_GAP"
        next_action = "R9H_H_REASSESS_CONTINUOUS_PROMOTION_SCOPE_FOR_CURRENT_DIAGNOSTIC_SERVICE_CONTRACT"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result = RESULTS / f"P3B1_R9HG_RESULT_{stamp}.json"
    latest = RESULTS / "P3B1_R9HG_LATEST.json"
    csvp = RESULTS / f"P3B1_R9HG_FULL_SERVICE_WITNESSES_{stamp}.csv"
    manifest = RESULTS / f"P3B1_R9HG_MANIFEST_{stamp}.sha256"
    write_csv(csvp, rows)

    out = {
        "schema": "SCV_P3B1_R9HG_FULL_AUGMENTED_SERVICE_RESTART_GFP_V1",
        "status": "PASS",
        "timestamp_utc": stamp,
        "classification": classification,
        "next_action": next_action,
        "continuous_state_separation_certified": False,
        "full_augmented_service_restart_point_gfp_solved": True,
        "metrics": {
            "point_action_width": float(args.action_width),
            "point_action_count": len(actions),
            "upper_iterations": int(upper.iterations),
            "upper_final_delta_m": float(upper.final_delta),
            "lower_iterations": int(lower.iterations),
            "lower_final_delta_m": float(lower.final_delta),
            "full_graph_sandwich_max_lower_minus_upper_m": sandwich_slack,
            "r9g_sensitive_test_rows": len(tests),
            "r9g_phase_aligned_test_rows": len(aligned_rows),
            "unique_witness_nodes": len(unique_nodes),
            "phase_aligned_witness_nodes": len(aligned_nodes),
            "witness_upper_sensitive_nodes": int(witness_upper_sensitive),
            "witness_paired_point_positive_nodes": int(witness_paired_positive),
            "max_witness_upper_stage_gap_m": float(max_upper_gap),
            "max_witness_paired_point_margin_m": float(max_paired_margin),
            "min_positive_witness_paired_point_margin_m": 0.0 if not math.isfinite(min_positive_paired) else float(min_positive_paired),
            "all_semantic_verify_more_demanding_nodes": all_verify_more,
            "all_semantic_fragment_more_demanding_nodes": all_fragment_more,
            "verifying_chi_unique_rectangles": groups_v.unique_rectangles,
            "verifying_chi_max_cells": groups_v.max_rectangle_cells,
            "last_chi_unique_rectangles": groups_last.unique_rectangles,
            "last_chi_max_cells": groups_last.max_rectangle_cells,
            "p2c_lower_resolution": int(args.p2c_lower_resolution),
            **{f"p2c_{k}": v for k, v in p2c_metrics.items()},
        },
        "gates": {
            "upstream_r9hf_restart_semantics_binding": True,
            "reachable_protocol_graph_closed": True,
            "restart_internal_to_fixed_point": True,
            "stage_timing_explicit": True,
            "robust_stage_conditioned_chi_envelope": True,
            "explicit_chi_correlations_certified": False,
            "upper_full_service_gfp_converged": bool(upper.converged),
            "lower_full_service_gfp_converged": bool(lower.converged),
            "point_model_sandwich_pass": True,
            "continuous_action_interval_gfp_solved": False,
            "independent_p2c_lower_checker_pass": False,
            "continuous_state_cell_interval_gfp": False,
        },
        "artifacts": {"full_service_witness_csv": str(csvp)},
    }
    text = json.dumps(out, indent=2, sort_keys=True)
    atomic_write(result, text)
    atomic_write(latest, text)
    mfiles = [
        Path(__file__), result, csvp,
        RESULTS / "P3B1_R9HF_LATEST.json",
        RESULTS / "P3B1_R9HE_LATEST.json",
        RESULTS / "P3B1_R9HA_R1_LATEST.json",
        RESULTS / "P3B1_R9G_LATEST.json",
    ]
    atomic_write(manifest, "".join(f"{sha256_file(p)}  {p}\n" for p in mfiles if p.exists()))

    print("=== R9-H-G DECISION ===", flush=True)
    print(
        f"R9HG_FULL_SERVICE_GFP upper_converged=YES upper_iterations={upper.iterations} "
        f"lower_converged=YES lower_iterations={lower.iterations}",
        flush=True,
    )
    print(
        f"R9HG_PHASE_ALIGNMENT r9g_tests={len(tests)} aligned_tests={len(aligned_rows)} "
        f"unique_nodes={len(unique_nodes)} compared_nodes={len(aligned_nodes)}",
        flush=True,
    )
    print(
        f"R9HG_STAGE_SEPARATION witness_upper_sensitive={witness_upper_sensitive}/{len(aligned_nodes)} "
        f"max_upper_gap_m={max_upper_gap:.12g} verify_more_all={all_verify_more} "
        f"fragment_more_all={all_fragment_more}",
        flush=True,
    )
    print(
        f"R9HG_PAIRED_POINT_MARGIN positive={witness_paired_positive}/{len(aligned_nodes)} "
        f"max_margin_m={max_paired_margin:.12g}",
        flush=True,
    )
    print("FULL_AUGMENTED_SERVICE_RESTART_POINT_GFP_SOLVED=YES", flush=True)
    print("CONTINUOUS_ACTION_INTERVAL_GFP_SOLVED=NO", flush=True)
    print("P2C_INDEPENDENT_LOWER_CHECKER=NO", flush=True)
    print("R9HG_EXECUTION=PASS", flush=True)
    print(f"R9HG_CLASSIFICATION={classification}", flush=True)
    print("CONTINUOUS_STATE_SEPARATION_CERTIFIED=NO", flush=True)
    print(f"R9HG_NEXT_ACTION={next_action}", flush=True)
    print(f"RESULT_JSON={result}", flush=True)
    print(f"MANIFEST={manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
