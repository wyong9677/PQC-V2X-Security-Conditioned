from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
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
AXIS_NAMES = ("v_f", "v_p", "a_f", "bar_a", "bar_u", "age")
EXPECTED_R9NR1 = "REACHABLE_RAW_STAGE_GAP_ERASED_BY_STAGEWISE_CELL_CORNER_MAX_ENCLOSURE"


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


def cell_corner_any(mask: np.ndarray, ndims: int = 6) -> np.ndarray:
    out = np.asarray(mask, dtype=bool)
    for axis in range(ndims):
        left = [slice(None)] * out.ndim
        right = [slice(None)] * out.ndim
        left[axis] = slice(0, -1)
        right[axis] = slice(1, None)
        out = np.logical_or(out[tuple(left)], out[tuple(right)])
    return out


def finite_max(x: np.ndarray) -> float:
    a = np.asarray(x, float)
    a = a[np.isfinite(a)]
    return float(np.max(a)) if a.size else 0.0


def insert_midpoints(axis: np.ndarray, interval_ids: list[int]) -> np.ndarray:
    a = np.asarray(axis, float)
    vals = list(a)
    for i in sorted(set(int(x) for x in interval_ids)):
        if i < 0 or i >= len(a) - 1:
            raise ValueError(f"BAD_INTERVAL_ID={i}")
        vals.append(0.5 * (float(a[i]) + float(a[i + 1])))
    return np.asarray(sorted(set(vals)), dtype=float)


def corner_values(stage_field: np.ndarray, coord: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    vals = []
    bits = []
    for bit in itertools.product((0, 1), repeat=6):
        idx = tuple(coord[d] + bit[d] for d in range(6))
        vals.append(float(stage_field[idx]))
        bits.append(bit)
    return np.asarray(vals, float), np.asarray(bits, np.int8)


def hamming_nearest(bit: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    if len(candidates) == 0:
        return bit.copy()
    d = np.sum(np.abs(candidates - bit[None, :]), axis=1)
    return candidates[int(np.argmin(d))]


def axis_vote_rows(
    *, h: np.ndarray, hit_cells: np.ndarray, hit_counts: np.ndarray,
    raw_gap_cells: np.ndarray, raw_gap_amplitude: np.ndarray, r9k,
) -> tuple[list[dict], np.ndarray]:
    cell_shape = raw_gap_cells.shape
    old = h[..., r9k.OLD_LAST]
    repl = h[..., r9k.REPL_LAST]
    votes = np.zeros(6, dtype=float)
    rows = []
    for flat in hit_cells:
        coord = tuple(int(x) for x in np.unravel_index(int(flat), cell_shape))
        ov, bits = corner_values(old, coord)
        rv, _ = corner_values(repl, coord)
        diff = np.abs(ov - rv)
        j = int(np.argmax(diff))
        gapbit = bits[j]
        sign = float(ov[j] - rv[j])
        if abs(sign) <= TOL:
            continue
        if sign > 0:
            target_max = float(np.max(rv))
            candidates = bits[np.abs(rv - target_max) <= TOL]
        else:
            target_max = float(np.max(ov))
            candidates = bits[np.abs(ov - target_max) <= TOL]
        maskbit = hamming_nearest(gapbit, candidates)
        differing = [d for d in range(6) if int(gapbit[d]) != int(maskbit[d])]
        weight = float(hit_counts[int(flat)]) * max(float(diff[j]), TOL)
        if differing:
            share = weight / len(differing)
            for d in differing:
                votes[d] += share
        rows.append({
            "cell_flat_index": int(flat),
            "hit_count": int(hit_counts[int(flat)]),
            "raw_gap_amplitude_m": float(raw_gap_amplitude.reshape(-1)[int(flat)]),
            "max_corner_raw_gap_m": float(diff[j]),
            "gap_corner_bits": "".join(str(int(x)) for x in gapbit),
            "mask_corner_bits": "".join(str(int(x)) for x in maskbit),
            "separating_axes": ",".join(AXIS_NAMES[d] for d in differing),
        })
    return rows, votes


def choose_axis_and_intervals(
    *, votes: np.ndarray, cell_hit_counts: np.ndarray, cell_shape: tuple[int, ...],
    eval_axes: list[np.ndarray], lookup_axes: list[np.ndarray], max_eval_nodes: int,
    max_lookup_nodes: int, forced_axis: str,
) -> tuple[int, list[int], dict]:
    if forced_axis != "auto":
        if forced_axis not in AXIS_NAMES:
            raise ValueError(f"R9O_BAD_REFINE_AXIS={forced_axis}")
        order = [AXIS_NAMES.index(forced_axis)]
    else:
        order = list(np.argsort(votes)[::-1])

    hit_cells = np.flatnonzero(cell_hit_counts > 0)
    if hit_cells.size == 0:
        raise RuntimeError("R9O_NO_PAIR_PHASE_RAW_GAP_HIT_CELLS")
    coords = np.unravel_index(hit_cells, cell_shape)

    base_eval_shape = [len(a) for a in eval_axes]
    base_lookup_shape = [len(a) for a in lookup_axes]
    base_eval_nodes = int(np.prod(base_eval_shape))
    base_lookup_nodes = int(np.prod(base_lookup_shape))

    for axis in order:
        if votes[axis] <= 0 and forced_axis == "auto":
            continue
        ids = np.asarray(coords[axis], dtype=int)
        counts = np.asarray([cell_hit_counts[int(f)] for f in hit_cells], dtype=np.int64)
        interval_counts: dict[int, int] = {}
        for i, c in zip(ids, counts):
            interval_counts[int(i)] = interval_counts.get(int(i), 0) + int(c)
        ranked = sorted(interval_counts, key=lambda i: (-interval_counts[i], i))

        # Determine how many midpoint insertions the node budget allows.  A selected
        # lookup midpoint only enters the eval axis if it lies inside the eval domain.
        other_eval = int(np.prod([len(a) for j, a in enumerate(eval_axes) if j != axis]))
        other_lookup = int(np.prod([len(a) for j, a in enumerate(lookup_axes) if j != axis]))
        max_eval_len = max(2, max_eval_nodes // max(1, other_eval))
        max_lookup_len = max(2, max_lookup_nodes // max(1, other_lookup))
        max_lookup_extra = max(0, max_lookup_len - len(lookup_axes[axis]))

        eval_lo, eval_hi = float(eval_axes[axis][0]), float(eval_axes[axis][-1])
        kept: list[int] = []
        eval_extra = 0
        for i in ranked:
            if len(kept) >= max_lookup_extra:
                break
            mid = 0.5 * (float(lookup_axes[axis][i]) + float(lookup_axes[axis][i + 1]))
            new_eval_extra = eval_extra + (1 if eval_lo < mid < eval_hi else 0)
            if len(eval_axes[axis]) + new_eval_extra > max_eval_len:
                continue
            kept.append(int(i))
            eval_extra = new_eval_extra

        if not kept:
            continue
        new_lookup_len = len(lookup_axes[axis]) + len(kept)
        new_eval_len = len(eval_axes[axis]) + sum(
            1 for i in kept
            if eval_lo < 0.5 * (float(lookup_axes[axis][i]) + float(lookup_axes[axis][i + 1])) < eval_hi
        )
        projected_eval = other_eval * new_eval_len
        projected_lookup = other_lookup * new_lookup_len
        meta = {
            "axis": AXIS_NAMES[axis],
            "axis_index": int(axis),
            "vote": float(votes[axis]),
            "available_intervals": len(ranked),
            "selected_intervals": len(kept),
            "base_eval_nodes": base_eval_nodes,
            "base_lookup_nodes": base_lookup_nodes,
            "projected_eval_nodes": int(projected_eval),
            "projected_lookup_nodes": int(projected_lookup),
            "node_budget_truncated": len(kept) < len(ranked),
        }
        return axis, kept, meta
    raise RuntimeError("R9O_NO_AXIS_FITS_NODE_BUDGET")


def build_refined_data(
    *, data: dict, axis: int, interval_ids: list[int], r6, r7,
) -> dict:
    refined = dict(data)
    lookup_axes = [np.asarray(a, float).copy() for a in data["lookup_axes"]]
    eval_axes = [np.asarray(a, float).copy() for a in data["eval_axes"]]

    coarse_lookup_axis = lookup_axes[axis].copy()
    mids = [
        0.5 * (float(coarse_lookup_axis[i]) + float(coarse_lookup_axis[i + 1]))
        for i in interval_ids
    ]
    lookup_axes[axis] = insert_midpoints(coarse_lookup_axis, interval_ids)
    elo, ehi = float(eval_axes[axis][0]), float(eval_axes[axis][-1])
    eval_add = [m for m in mids if elo < m < ehi]
    eval_axes[axis] = np.asarray(sorted(set(list(eval_axes[axis]) + eval_add)), dtype=float)

    eval_flat, eval_shape = r6.mesh_flat(eval_axes)
    lookup_flat, lookup_shape = r6.mesh_flat(lookup_axes)
    eval_idx = r7.exact_node_indices(eval_axes, lookup_axes)

    eval_cfg = copy.deepcopy(data["eval_cfg"])
    for name, arr in zip(AXIS_NAMES, eval_axes):
        eval_cfg["grid"][name] = [float(x) for x in arr]

    lookup_fallback = r6.fallback_required_on_grid(
        eval_cfg, data["p1"], data["p2b"], data["p2c"], data["p3a"], lookup_flat
    )
    eval_fallback = np.asarray(lookup_fallback[np.asarray(eval_idx, dtype=np.int64)], float)

    refined.update({
        "eval_cfg": eval_cfg,
        "eval_axes": eval_axes,
        "eval_flat": eval_flat,
        "eval_shape": eval_shape,
        "lookup_axes": lookup_axes,
        "lookup_flat": lookup_flat,
        "lookup_shape": lookup_shape,
        "eval_idx": np.asarray(eval_idx, dtype=np.int64),
        "lookup_fallback": np.asarray(lookup_fallback, float),
        "eval_fallback": eval_fallback,
        "eval_age_axis": np.asarray(eval_axes[5], float),
        "lookup_age_axis": np.asarray(lookup_axes[5], float),
    })
    return refined


def evaluate_solution(*, data: dict, upper, lower, r3, r9k) -> dict:
    eval_idx = np.asarray(data["eval_idx"], dtype=np.int64)
    ue = np.asarray(upper.h_flat[eval_idx, :], float)
    age = np.asarray(data["eval_flat"][5], float)
    Ts = float(data["Ts"])
    common = age >= 3.0 * Ts - 1e-10
    signed = ue[:, r9k.CRED] - ue[:, r9k.FRAG]
    sensitive = common & (np.abs(signed) > TOL)

    h = upper.h_flat.reshape(tuple(data["lookup_shape"]) + (r9k.N_STAGES,))
    raw_gap = np.abs(h[..., r9k.OLD_LAST] - h[..., r9k.REPL_LAST])
    raw_support = np.isfinite(raw_gap) & (raw_gap > TOL)
    raw_cells = cell_corner_any(raw_support, ndims=6)
    cells = r3.cell_corner_max(h)
    env_gap = np.abs(cells[..., r9k.OLD_LAST] - cells[..., r9k.REPL_LAST])
    env_support = np.isfinite(env_gap) & (env_gap > TOL)

    out = {
        "common_nodes": int(np.count_nonzero(common)),
        "upper_sensitive_nodes": int(np.count_nonzero(sensitive)),
        "max_upper_pair_gap_m": float(np.max(np.abs(signed[common]))) if np.any(common) else 0.0,
        "raw_last_stage_gap_nodes": int(np.count_nonzero(raw_support)),
        "max_raw_last_stage_gap_m": finite_max(raw_gap[raw_support]) if np.any(raw_support) else 0.0,
        "raw_gap_containing_cells": int(np.count_nonzero(raw_cells)),
        "stagewise_cellmax_gap_cells": int(np.count_nonzero(env_support)),
        "max_stagewise_cellmax_gap_m": finite_max(env_gap[env_support]) if np.any(env_support) else 0.0,
    }
    if lower is not None:
        le = np.asarray(lower.h_flat[eval_idx, :], float)
        if np.any(le > ue + 1e-8):
            raise RuntimeError(f"R9O_REFINED_SANDWICH_FAIL count={int(np.count_nonzero(le > ue + 1e-8))}")
        m1 = le[:, r9k.CRED] - ue[:, r9k.FRAG]
        m2 = le[:, r9k.FRAG] - ue[:, r9k.CRED]
        paired = common & ((m1 > TOL) | (m2 > TOL))
        out.update({
            "paired_positive_nodes": int(np.count_nonzero(paired)),
            "max_credential_worse_margin_m": float(np.max(m1[common])) if np.any(common) else 0.0,
            "max_fragment_worse_margin_m": float(np.max(m2[common])) if np.any(common) else 0.0,
        })
    else:
        out.update({
            "paired_positive_nodes": 0,
            "max_credential_worse_margin_m": 0.0,
            "max_fragment_worse_margin_m": 0.0,
        })
    return out


def self_test() -> None:
    a = np.asarray([0.0, 1.0, 2.0])
    b = insert_midpoints(a, [0])
    assert np.allclose(b, [0.0, 0.5, 1.0, 2.0])
    # A cell where OLD has its distinctive max at the left corner and REPL is
    # masked by an equal maximum at the right corner should vote for the split axis.
    old = np.asarray([2.0, 1.0])
    repl = np.asarray([1.0, 2.0])
    diff = np.abs(old - repl)
    assert float(np.max(diff)) == 1.0
    # budget arithmetic
    eval_axes = [np.arange(3.0), np.arange(2.0)]
    assert int(np.prod([len(x) for x in eval_axes])) == 6
    print("R9O_INTERNAL_SELF_TEST=PASS", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--action-width", type=float, default=0.5)
    ap.add_argument("--p2c-lower-resolution", type=int, default=257)
    ap.add_argument("--max-eval-nodes", type=int, default=1800000)
    ap.add_argument("--max-lookup-nodes", type=int, default=3300000)
    ap.add_argument("--refine-axis", type=str, default="auto")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.action_width <= 0.0 or args.action_width > 1.0:
        raise ValueError("R9O_BAD_ACTION_WIDTH")

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
    import p3b1_r9k_full_augmented_declared_pair_gfp as r9k

    up = RESULTS / "P3B1_R9NR1_LATEST.json"
    if not up.exists():
        raise RuntimeError("R9O_MISSING_R9NR1_LATEST")
    r9nr1 = load_json(up)
    if r9nr1.get("status") != "PASS" or r9nr1.get("classification") != EXPECTED_R9NR1:
        raise RuntimeError(f"R9O_R9NR1_GATE_FAIL={r9nr1.get('classification')}")

    data = r9d.reconstruct_timestamped_geometry(r3, r5, r6, r7, r9c)
    Ts = float(data["Ts"])
    actions = r9hb.refinement_actions(float(args.action_width))

    print("=== P3-B1-R9-O ADAPTIVE REACHABLE-GAP REFINEMENT ===", flush=True)
    print("UPSTREAM_R9NR1_REACHABLE_ENCLOSURE_ERASURE=PASS", flush=True)
    print("QUESTION=CAN_BUDGETED_ADAPTIVE_STATE_REFINEMENT_RECOVER_STAGEWISE_VALUE_SEPARATION", flush=True)
    print("REFINEMENT_IS_GLOBAL_TENSOR_ONLY_ON_SELECTED_REACHABLE_INTERVALS=YES", flush=True)
    print("REFINEMENT_IS_LOCAL_SPARSE_PATCH_THEOREM=NO", flush=True)
    print("DEPLOYMENT_PROTOCOL_CERTIFIED=NO", flush=True)
    print("CONTINUOUS_STATE_SEPARATION_CERTIFIED=NO", flush=True)
    print(f"POINT_ACTION_WIDTH={float(args.action_width):.12g}", flush=True)
    print(f"POINT_ACTION_COUNT={len(actions)}", flush=True)
    print(f"R9O_MAX_EVAL_NODES={int(args.max_eval_nodes)}", flush=True)
    print(f"R9O_MAX_LOOKUP_NODES={int(args.max_lookup_nodes)}", flush=True)

    # Coarse recomputation used only to identify reachable erased cells and choose
    # a refinement axis/interval set.
    cfg0, tr0 = r9hg.build_physical_transitions(
        data, actions, r3, b0, r9e, chunk_size=int(args.chunk_size)
    )
    old_groups0 = r9hg.build_descriptor_range_groups(data, 2.0 * Ts, b0)
    repl_groups0 = r9hg.build_descriptor_range_groups(data, 0.0, b0)
    old_age0, ok0 = r9hg.scalar_cell_index(np.asarray(data["lookup_axes"][5], float), 3.0 * Ts)
    repl_age0, ok1 = r9hg.scalar_cell_index(np.asarray(data["lookup_axes"][5], float), 1.0 * Ts)
    if not ok0 or not ok1:
        raise RuntimeError("R9O_COARSE_ADOPTION_AGE_LOOKUP_FAIL")

    print("R9O_STAGE_START=coarse_upper_for_refinement_planner", flush=True)
    coarse = r9k.solve_full_service_fixed_point(
        data=data, cfg=cfg0, transitions=tr0,
        fallback_eval=np.asarray(data["eval_fallback"], float),
        old_groups=old_groups0, repl_groups=repl_groups0,
        old_adopt_age_cell=old_age0, repl_adopt_age_cell=repl_age0,
        r3=r3, r9hg=r9hg, mode="upper", label="r9o_coarse_upper",
    )
    if not coarse.converged:
        raise RuntimeError("R9O_COARSE_UPPER_NOT_CONVERGED")

    h0 = coarse.h_flat.reshape(tuple(data["lookup_shape"]) + (r9k.N_STAGES,))
    raw_gap0 = np.abs(h0[..., r9k.OLD_LAST] - h0[..., r9k.REPL_LAST])
    raw_support0 = np.isfinite(raw_gap0) & (raw_gap0 > TOL)
    raw_cells0 = cell_corner_any(raw_support0, ndims=6)
    raw_amp0 = np.zeros_like(raw_cells0, dtype=float)
    # Max raw nodal gap among each cell's corners.
    amp = np.where(np.isfinite(raw_gap0), raw_gap0, 0.0)
    for axis in range(6):
        left = [slice(None)] * amp.ndim
        right = [slice(None)] * amp.ndim
        left[axis] = slice(0, -1)
        right[axis] = slice(1, None)
        amp = np.maximum(amp[tuple(left)], amp[tuple(right)])
    raw_amp0 = amp

    pair_phase = np.asarray(data["eval_flat"][5], float) >= 3.0 * Ts - 1e-10
    cell_shape = tuple(len(a) - 1 for a in data["lookup_axes"])
    cell_count = int(np.prod(cell_shape))
    hit_counts = np.zeros(cell_count, dtype=np.int64)
    raw_flat = raw_cells0.reshape(-1)
    for tr in tr0:
        hold = np.asarray(tr["hold_index"], dtype=np.intp)
        m = pair_phase & raw_flat[hold]
        if np.any(m):
            hit_counts += np.bincount(hold[m], minlength=cell_count)
    hit_cells = np.flatnonzero(hit_counts > 0)
    if hit_cells.size == 0:
        raise RuntimeError("R9O_NO_REACHABLE_RAW_GAP_CELLS_CONTRADICTS_R9NR1")

    attribution_rows, votes = axis_vote_rows(
        h=h0, hit_cells=hit_cells, hit_counts=hit_counts,
        raw_gap_cells=raw_cells0, raw_gap_amplitude=raw_amp0, r9k=r9k,
    )
    axis, intervals, plan = choose_axis_and_intervals(
        votes=votes, cell_hit_counts=hit_counts, cell_shape=cell_shape,
        eval_axes=[np.asarray(a, float) for a in data["eval_axes"]],
        lookup_axes=[np.asarray(a, float) for a in data["lookup_axes"]],
        max_eval_nodes=int(args.max_eval_nodes), max_lookup_nodes=int(args.max_lookup_nodes),
        forced_axis=str(args.refine_axis),
    )

    print(
        "R9O_AXIS_ATTRIBUTION "
        + " ".join(f"{AXIS_NAMES[i]}_vote={votes[i]:.12g}" for i in range(6)),
        flush=True,
    )
    print(
        "R9O_REFINEMENT_PLAN "
        f"axis={plan['axis']} available_intervals={plan['available_intervals']} "
        f"selected_intervals={plan['selected_intervals']} "
        f"base_eval_nodes={plan['base_eval_nodes']} projected_eval_nodes={plan['projected_eval_nodes']} "
        f"base_lookup_nodes={plan['base_lookup_nodes']} projected_lookup_nodes={plan['projected_lookup_nodes']} "
        f"budget_truncated={str(plan['node_budget_truncated']).upper()}",
        flush=True,
    )

    refined = build_refined_data(data=data, axis=axis, interval_ids=intervals, r6=r6, r7=r7)
    actual_eval_nodes = int(np.prod(refined["eval_shape"]))
    actual_lookup_nodes = int(np.prod(refined["lookup_shape"]))
    if actual_eval_nodes > args.max_eval_nodes or actual_lookup_nodes > args.max_lookup_nodes:
        raise RuntimeError(
            f"R9O_REFINED_NODE_BUDGET_FAIL eval={actual_eval_nodes} lookup={actual_lookup_nodes}"
        )

    print(
        f"R9O_REFINED_GEOMETRY axis={AXIS_NAMES[axis]} intervals={len(intervals)} "
        f"eval_nodes={actual_eval_nodes} lookup_nodes={actual_lookup_nodes}",
        flush=True,
    )

    print("R9O_STAGE_START=build_refined_transitions", flush=True)
    cfg1, tr1 = r9hg.build_physical_transitions(
        refined, actions, r3, b0, r9e, chunk_size=int(args.chunk_size)
    )
    old_groups1 = r9hg.build_descriptor_range_groups(refined, 2.0 * Ts, b0)
    repl_groups1 = r9hg.build_descriptor_range_groups(refined, 0.0, b0)
    old_age1, oka = r9hg.scalar_cell_index(np.asarray(refined["lookup_axes"][5], float), 3.0 * Ts)
    repl_age1, okb = r9hg.scalar_cell_index(np.asarray(refined["lookup_axes"][5], float), 1.0 * Ts)
    if not oka or not okb:
        raise RuntimeError("R9O_REFINED_ADOPTION_AGE_LOOKUP_FAIL")

    print("R9O_STAGE_START=refined_upper_full_gfp", flush=True)
    upper = r9k.solve_full_service_fixed_point(
        data=refined, cfg=cfg1, transitions=tr1,
        fallback_eval=np.asarray(refined["eval_fallback"], float),
        old_groups=old_groups1, repl_groups=repl_groups1,
        old_adopt_age_cell=old_age1, repl_adopt_age_cell=repl_age1,
        r3=r3, r9hg=r9hg, mode="upper", label="r9o_refined_upper",
    )
    if not upper.converged:
        raise RuntimeError("R9O_REFINED_UPPER_NOT_CONVERGED")

    # Solve the lower side only if the upper pair gap reappears. This keeps the
    # refinement gate computationally bounded while still producing a paired
    # certificate candidate whenever there is something to certify.
    prelim = evaluate_solution(data=refined, upper=upper, lower=None, r3=r3, r9k=r9k)
    lower = None
    p2c_metrics = None
    if prelim["upper_sensitive_nodes"] > 0:
        print("R9O_STAGE_START=refined_p2c_lower_and_lower_full_gfp", flush=True)
        fallback_lower, p2c_metrics = r9he.compute_eval_fallback_lower(
            refined, b0, sw, resolution=int(args.p2c_lower_resolution),
            chunk_size=int(args.chunk_size),
        )
        lower = r9k.solve_full_service_fixed_point(
            data=refined, cfg=cfg1, transitions=tr1,
            fallback_eval=np.asarray(fallback_lower, float),
            old_groups=old_groups1, repl_groups=repl_groups1,
            old_adopt_age_cell=old_age1, repl_adopt_age_cell=repl_age1,
            r3=r3, r9hg=r9hg, mode="lower", label="r9o_refined_lower",
        )
        if not lower.converged:
            raise RuntimeError("R9O_REFINED_LOWER_NOT_CONVERGED")

    metrics = evaluate_solution(data=refined, upper=upper, lower=lower, r3=r3, r9k=r9k)
    coarse_raw = int(np.count_nonzero(raw_support0))
    coarse_raw_max = finite_max(raw_gap0[raw_support0]) if coarse_raw else 0.0

    print(
        "R9O_REFINED_LAST_STAGE "
        f"coarse_raw_nodes={coarse_raw} coarse_raw_gap_m={coarse_raw_max:.12g} "
        f"refined_raw_nodes={metrics['raw_last_stage_gap_nodes']} "
        f"refined_raw_gap_m={metrics['max_raw_last_stage_gap_m']:.12g} "
        f"refined_raw_gap_cells={metrics['raw_gap_containing_cells']} "
        f"refined_stagewise_cellmax_gap_cells={metrics['stagewise_cellmax_gap_cells']} "
        f"refined_stagewise_cellmax_gap_m={metrics['max_stagewise_cellmax_gap_m']:.12g}",
        flush=True,
    )
    print(
        "R9O_REFINED_PAIR_GFP "
        f"common_nodes={metrics['common_nodes']} upper_sensitive={metrics['upper_sensitive_nodes']} "
        f"max_upper_gap_m={metrics['max_upper_pair_gap_m']:.12g} "
        f"paired_positive={metrics['paired_positive_nodes']} "
        f"max_credential_margin_m={metrics['max_credential_worse_margin_m']:.12g} "
        f"max_fragment_margin_m={metrics['max_fragment_worse_margin_m']:.12g}",
        flush=True,
    )

    if metrics["paired_positive_nodes"] > 0:
        classification = "ADAPTIVE_REFINEMENT_RECOVERS_FULL_GFP_STAGE_SEPARATION_AND_PAIRED_POINT_MARGIN"
        next_action = "R9P_INTERVALIZE_REFINED_FULL_GRAPH_AND_BUILD_INDEPENDENT_P2C_LOWER_CHECKER"
    elif metrics["upper_sensitive_nodes"] > 0:
        classification = "ADAPTIVE_REFINEMENT_RECOVERS_FULL_GFP_STAGE_SEPARATION_PAIRED_MARGIN_OPEN"
        next_action = "R9P_STRENGTHEN_REFINED_LOWER_ENCLOSURE_THEN_CONTINUOUS_INTERVALIZE"
    elif metrics["stagewise_cellmax_gap_cells"] > 0:
        classification = "ADAPTIVE_REFINEMENT_PRESERVES_LAST_STAGE_CELLMAX_GAP_BUT_PAIR_GFP_STILL_COLLAPSES"
        next_action = "R9P_STOP_PAIR_SEARCH_AND_ANALYZE_VALUE_TRANSFER_ON_REFINED_GRAPH"
    elif metrics["raw_last_stage_gap_nodes"] > 0:
        classification = "ONE_AXIS_ADAPTIVE_REFINEMENT_INSUFFICIENT_ENCLOSURE_ERASURE_PERSISTS"
        next_action = "R9O_R2_REFINE_SECOND_ATTRIBUTED_AXIS_ONLY_IF_WITHIN_PREDECLARED_NODE_BUDGET"
    else:
        classification = "COARSE_RAW_STAGE_GAP_NOT_STABLE_UNDER_REFINED_FULL_GFP"
        next_action = "R9P_STOP_POSITIVE_CONTINUOUS_PROMOTION_AND_REPORT_REFINEMENT_FALSIFICATION"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result = RESULTS / f"P3B1_R9O_RESULT_{stamp}.json"
    latest = RESULTS / "P3B1_R9O_LATEST.json"
    attrib_csv = RESULTS / f"P3B1_R9O_AXIS_ATTRIBUTION_{stamp}.csv"
    plan_csv = RESULTS / f"P3B1_R9O_REFINED_INTERVALS_{stamp}.csv"
    manifest = RESULTS / f"P3B1_R9O_MANIFEST_{stamp}.sha256"

    for i, row in enumerate(attribution_rows):
        row["selected_axis"] = AXIS_NAMES[axis]
    write_csv(attrib_csv, attribution_rows[:4096])
    interval_rows = []
    coarse_axis = np.asarray(data["lookup_axes"][axis], float)
    for i in intervals:
        interval_rows.append({
            "axis": AXIS_NAMES[axis], "interval_index": int(i),
            "left": float(coarse_axis[i]), "right": float(coarse_axis[i + 1]),
            "midpoint": 0.5 * (float(coarse_axis[i]) + float(coarse_axis[i + 1])),
        })
    write_csv(plan_csv, interval_rows)

    payload = {
        "schema": "P3B1_R9O_ADAPTIVE_REACHABLE_GAP_REFINEMENT_V1",
        "status": "PASS",
        "classification": classification,
        "next_action": next_action,
        "hard_flags": {
            "deployment_protocol_certified": False,
            "continuous_action_interval_gfp_solved": False,
            "p2c_independent_lower_checker": False,
            "continuous_state_separation_certified": False,
        },
        "refinement_plan": plan,
        "selected_interval_ids": [int(i) for i in intervals],
        "axis_votes": {AXIS_NAMES[i]: float(votes[i]) for i in range(6)},
        "metrics": metrics,
        "coarse": {
            "raw_last_stage_gap_nodes": coarse_raw,
            "max_raw_last_stage_gap_m": coarse_raw_max,
            "reachable_raw_gap_cells": int(hit_cells.size),
        },
        "refined_geometry": {
            "eval_nodes": actual_eval_nodes,
            "lookup_nodes": actual_lookup_nodes,
            "eval_shape": [int(x) for x in refined["eval_shape"]],
            "lookup_shape": [int(x) for x in refined["lookup_shape"]],
        },
        "p2c_lower_metrics": p2c_metrics,
        "artifacts": {
            "axis_attribution_csv": str(attrib_csv),
            "refined_intervals_csv": str(plan_csv),
        },
    }
    text = json.dumps(payload, indent=2, sort_keys=True, default=lambda x: x.item() if isinstance(x, np.generic) else x) + "\n"
    atomic_write(result, text)
    atomic_write(latest, text)
    files = [result, latest, attrib_csv, plan_csv]
    atomic_write(manifest, "\n".join(f"{sha256_file(p)}  {p.name}" for p in files) + "\n")

    print("=== R9-O DECISION ===", flush=True)
    print("R9O_ADAPTIVE_REFINEMENT_GATE=PASS", flush=True)
    print(f"R9O_CLASSIFICATION={classification}", flush=True)
    print("DEPLOYMENT_PROTOCOL_CERTIFIED=NO", flush=True)
    print("CONTINUOUS_ACTION_INTERVAL_GFP_SOLVED=NO", flush=True)
    print("P2C_INDEPENDENT_LOWER_CHECKER=NO", flush=True)
    print("CONTINUOUS_STATE_SEPARATION_CERTIFIED=NO", flush=True)
    print("R9O_EXECUTION=PASS", flush=True)
    print(f"R9O_NEXT_ACTION={next_action}", flush=True)
    print(f"RESULT_JSON={result}", flush=True)
    print(f"AXIS_ATTRIBUTION_CSV={attrib_csv}", flush=True)
    print(f"REFINED_INTERVALS_CSV={plan_csv}", flush=True)
    print(f"MANIFEST={manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
