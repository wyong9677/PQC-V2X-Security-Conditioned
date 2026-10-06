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

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
RESULTS = ROOT / "04_results"
RESULTS.mkdir(parents=True, exist_ok=True)
TOL = 1e-9
EXPECTED_R9N = "GLOBAL_LAST_STAGE_VALUE_GAP_HAS_NO_ONE_STEP_PHYSICAL_PREIMAGE_IN_CURRENT_GRID"
EXPECTED_R9M = "PAIR_SUCCESSOR_IMAGE_AVOIDS_ALL_GLOBAL_LAST_STAGE_VALUE_DIFFERENCE"


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
    seen = set()
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


def cell_corner_max_scalar(values: np.ndarray, ndims: int = 6) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    for axis in range(ndims):
        left = [slice(None)] * out.ndim
        right = [slice(None)] * out.ndim
        left[axis] = slice(0, -1)
        right[axis] = slice(1, None)
        out = np.maximum(out[tuple(left)], out[tuple(right)])
    return out


def finite_max(x: np.ndarray) -> float:
    a = np.asarray(x, float)
    a = a[np.isfinite(a)]
    return float(np.max(a)) if a.size else 0.0


def self_test() -> None:
    # Two stage nodal fields can differ at a corner while their stage-wise
    # cell maxima remain identical. This is the precise R9-N ambiguity.
    a = np.asarray([0.0, 2.0, 0.0])
    b = np.asarray([1.0, 2.0, 0.0])
    raw = np.abs(a - b) > TOL
    raw_cells = cell_corner_any(raw, ndims=1)
    ca = cell_corner_max_scalar(a, ndims=1)
    cb = cell_corner_max_scalar(b, ndims=1)
    env = np.abs(ca - cb) > TOL
    assert raw_cells.tolist() == [True, False]
    assert env.tolist() == [False, False]
    # A successor may hit a cell containing a raw gap even when the enclosed
    # stage maxima are equal on that cell.
    hold = np.asarray([0, 1], dtype=np.intp)
    assert int(np.count_nonzero(raw_cells[hold])) == 1
    assert int(np.count_nonzero(env[hold])) == 0
    print("R9NR1_INTERNAL_SELF_TEST=PASS", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--action-width", type=float, default=0.5)
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.action_width <= 0.0 or args.action_width > 1.0:
        raise ValueError("R9NR1_BAD_ACTION_WIDTH")

    sys.path.insert(0, str(SRC))
    import p3b1_augmented_fixed_point_v1_r3 as r3
    import p3b0_freshness_service_audit_v1 as b0
    import p3b1_r5_refinement_attribution as r5
    import p3b1_r6_continuation_halo as r6
    import p3b1_r7_frozen_halo_fixed_point as r7
    import p3b1_r9c_targeted_history_gate as r9c
    import p3b1_r9d_timestamped_service as r9d
    import p3b1_r9e_service_contract_general_action as r9e
    import p3b1_r9h_b_protocolized_paired_bounds as r9hb
    import p3b1_r9h_g_full_augmented_service_gfp as r9hg
    import p3b1_r9k_full_augmented_declared_pair_gfp as r9k

    r9n_path = RESULTS / "P3B1_R9N_LATEST.json"
    r9m_path = RESULTS / "P3B1_R9M_LATEST.json"
    for p in (r9n_path, r9m_path):
        if not p.exists():
            raise RuntimeError(f"R9NR1_MISSING_UPSTREAM={p}")
    r9n = load_json(r9n_path)
    r9m = load_json(r9m_path)
    if r9n.get("status") != "PASS" or r9n.get("classification") != EXPECTED_R9N:
        raise RuntimeError(f"R9NR1_R9N_GATE_FAIL={r9n.get('classification')}")
    if r9m.get("status") != "PASS" or r9m.get("classification") != EXPECTED_R9M:
        raise RuntimeError(f"R9NR1_R9M_GATE_FAIL={r9m.get('classification')}")
    global_gap = float(r9m.get("metrics", {}).get("global_last_stage_gap_m", 0.0))
    if global_gap <= TOL:
        raise RuntimeError("R9NR1_EXPECTED_POSITIVE_RAW_GLOBAL_GAP")

    data = r9d.reconstruct_timestamped_geometry(r3, r5, r6, r7, r9c)
    Ts = float(data["Ts"])
    actions = r9hb.refinement_actions(float(args.action_width))

    print("=== P3-B1-R9-N-R1 ENCLOSURE-ERASURE CORRECTION AUDIT ===", flush=True)
    print("UPSTREAM_R9N_NO_PREIMAGE_CLASSIFICATION=PASS", flush=True)
    print("UPSTREAM_R9M_RAW_LAST_STAGE_GAP_POSITIVE=PASS", flush=True)
    print("QUESTION=DID_R9N_CONFUSE_RAW_NODAL_GAP_SUPPORT_WITH_STAGEWISE_CELL_MAX_GAP_SUPPORT", flush=True)
    print("R9N_ORIGINAL_CLASSIFICATION_ACCEPTED_AS_FINAL=NO", flush=True)
    print("CONTINUOUS_STATE_SEPARATION_CERTIFIED=NO", flush=True)
    print(f"POINT_ACTION_WIDTH={float(args.action_width):.12g}", flush=True)
    print(f"POINT_ACTION_COUNT={len(actions)}", flush=True)

    cfg, transitions = r9hg.build_physical_transitions(
        data, actions, r3, b0, r9e, chunk_size=int(args.chunk_size)
    )
    age = np.asarray(data["eval_flat"][5], float)
    masks = r9k.stage_masks(age, Ts)

    old_groups = r9hg.build_descriptor_range_groups(data, 2.0 * Ts, b0)
    repl_groups = r9hg.build_descriptor_range_groups(data, 0.0, b0)
    old_age_cell, old_ok = r9hg.scalar_cell_index(np.asarray(data["lookup_axes"][5], float), 3.0 * Ts)
    repl_age_cell, repl_ok = r9hg.scalar_cell_index(np.asarray(data["lookup_axes"][5], float), 1.0 * Ts)
    if not old_ok or not repl_ok:
        raise RuntimeError("R9NR1_ADOPTION_AGE_LOOKUP_FAIL")

    print("R9NR1_STAGE_START=recompute_actual_upper_full_gfp", flush=True)
    upper = r9k.solve_full_service_fixed_point(
        data=data, cfg=cfg, transitions=transitions,
        fallback_eval=np.asarray(data["eval_fallback"], float),
        old_groups=old_groups, repl_groups=repl_groups,
        old_adopt_age_cell=old_age_cell, repl_adopt_age_cell=repl_age_cell,
        r3=r3, r9hg=r9hg, mode="upper", label="r9nr1_actual_upper",
    )
    if not upper.converged:
        raise RuntimeError("R9NR1_UPPER_GFP_NOT_CONVERGED")

    lookup_shape = tuple(data["lookup_shape"])
    eval_idx = np.asarray(data["eval_idx"], dtype=np.int64)
    h = upper.h_flat.reshape(lookup_shape + (r9k.N_STAGES,))
    ue = np.asarray(upper.h_flat[eval_idx, :], float)

    # RAW nodal support: this is what R9-L's 0.2067608 m stage-identity
    # diagnostic measures on the eval nodes.
    raw_lookup_gap = np.abs(h[..., r9k.OLD_LAST] - h[..., r9k.REPL_LAST])
    raw_lookup_support = np.isfinite(raw_lookup_gap) & (raw_lookup_gap > TOL)
    raw_lookup_nodes = int(np.count_nonzero(raw_lookup_support))
    raw_lookup_max = finite_max(raw_lookup_gap[raw_lookup_support]) if raw_lookup_nodes else 0.0

    last_common = masks[r9k.OLD_LAST] & masks[r9k.REPL_LAST]
    raw_eval_gap = np.abs(ue[:, r9k.OLD_LAST] - ue[:, r9k.REPL_LAST])
    raw_eval_support = last_common & np.isfinite(raw_eval_gap) & (raw_eval_gap > TOL)
    raw_eval_nodes = int(np.count_nonzero(raw_eval_support))
    raw_eval_max = finite_max(raw_eval_gap[raw_eval_support]) if raw_eval_nodes else 0.0
    if raw_eval_nodes == 0:
        raise RuntimeError("R9NR1_RAW_EVAL_GAP_DISAPPEARED_CONTRADICTS_UPSTREAM")

    # A cell is raw-gap-containing if ANY of its 2^6 nodal corners has a
    # nonzero OLD_LAST/REPL_LAST difference.
    raw_gap_cells = cell_corner_any(raw_lookup_support, ndims=6)
    raw_gap_cell_count = int(np.count_nonzero(raw_gap_cells))
    raw_gap_amplitude_cells = cell_corner_max_scalar(
        np.where(np.isfinite(raw_lookup_gap), raw_lookup_gap, 0.0), ndims=6
    )
    raw_gap_cell_max = finite_max(raw_gap_amplitude_cells[raw_gap_cells]) if raw_gap_cell_count else 0.0

    # R9-N's original object: compare the two STAGE-WISE cell maxima.
    stage_cells = r3.cell_corner_max(h)
    enclosed_gap = np.abs(
        stage_cells[..., r9k.OLD_LAST] - stage_cells[..., r9k.REPL_LAST]
    )
    enclosed_support = np.isfinite(enclosed_gap) & (enclosed_gap > TOL)
    enclosed_gap_cells = int(np.count_nonzero(enclosed_support))
    enclosed_gap_max = finite_max(enclosed_gap[enclosed_support]) if enclosed_gap_cells else 0.0

    erased_cells = raw_gap_cells & ~enclosed_support
    erased_cell_count = int(np.count_nonzero(erased_cells))
    preserved_cells = raw_gap_cells & enclosed_support
    preserved_cell_count = int(np.count_nonzero(preserved_cells))

    raw_flat = raw_gap_cells.reshape(-1)
    enclosed_flat = enclosed_support.reshape(-1)
    erased_flat = erased_cells.reshape(-1)
    amplitude_flat = raw_gap_amplitude_cells.reshape(-1)

    n = len(eval_idx)
    pair_phase = age >= 3.0 * Ts - 1e-10
    raw_preimage_tests = 0
    raw_preimage_nodes = np.zeros(n, bool)
    raw_pair_tests = 0
    raw_pair_nodes = np.zeros(n, bool)
    enclosed_preimage_tests = 0
    enclosed_pair_tests = 0
    erased_preimage_tests = 0
    erased_pair_tests = 0
    earliest_raw_age = math.inf
    witnesses: list[dict] = []
    vf, vp, af, ba, bu, ag = [np.asarray(x, float) for x in data["eval_flat"]]

    for tr in transitions:
        hold = np.asarray(tr["hold_index"], dtype=np.intp)
        raw_hit = raw_flat[hold]
        env_hit = enclosed_flat[hold]
        era_hit = erased_flat[hold]
        raw_preimage_tests += int(np.count_nonzero(raw_hit))
        enclosed_preimage_tests += int(np.count_nonzero(env_hit))
        erased_preimage_tests += int(np.count_nonzero(era_hit))
        raw_preimage_nodes |= raw_hit
        rp = raw_hit & pair_phase
        ep = env_hit & pair_phase
        xp = era_hit & pair_phase
        raw_pair_tests += int(np.count_nonzero(rp))
        enclosed_pair_tests += int(np.count_nonzero(ep))
        erased_pair_tests += int(np.count_nonzero(xp))
        raw_pair_nodes |= rp
        if np.any(raw_hit):
            earliest_raw_age = min(earliest_raw_age, float(np.min(age[raw_hit])))
            for i in np.flatnonzero(raw_hit)[:3]:
                witnesses.append({
                    "action": float(tr.get("action", math.nan)),
                    "node_index": int(i),
                    "v_f": float(vf[i]), "v_p": float(vp[i]), "a_f": float(af[i]),
                    "bar_a": float(ba[i]), "bar_u": float(bu[i]), "age_s": float(ag[i]),
                    "pair_phase_age_ge_3Ts": bool(pair_phase[i]),
                    "raw_gap_cell": True,
                    "stagewise_cellmax_gap": bool(env_hit[i]),
                    "enclosure_erased": bool(era_hit[i]),
                    "raw_cell_max_corner_gap_m": float(amplitude_flat[hold[i]]),
                })
    if not math.isfinite(earliest_raw_age):
        earliest_raw_age = -1.0

    print(
        "R9NR1_RAW_NODAL_GAP "
        f"lookup_nodes={raw_lookup_nodes} eval_semantic_nodes={raw_eval_nodes} "
        f"max_lookup_gap_m={raw_lookup_max:.12g} max_eval_gap_m={raw_eval_max:.12g} "
        f"upstream_global_gap_m={global_gap:.12g}",
        flush=True,
    )
    print(
        "R9NR1_CELL_ENCLOSURE "
        f"raw_gap_containing_cells={raw_gap_cell_count} "
        f"raw_cell_max_corner_gap_m={raw_gap_cell_max:.12g} "
        f"stagewise_cellmax_gap_cells={enclosed_gap_cells} "
        f"stagewise_cellmax_max_gap_m={enclosed_gap_max:.12g} "
        f"enclosure_erased_cells={erased_cell_count} preserved_cells={preserved_cell_count}",
        flush=True,
    )
    print(
        "R9NR1_PREIMAGE_COMPARISON "
        f"raw_gap_cell_preimage_tests={raw_preimage_tests} "
        f"raw_gap_cell_preimage_nodes={int(np.count_nonzero(raw_preimage_nodes))} "
        f"raw_gap_pair_phase_tests={raw_pair_tests} "
        f"raw_gap_pair_phase_nodes={int(np.count_nonzero(raw_pair_nodes))} "
        f"stagewise_cellmax_gap_preimage_tests={enclosed_preimage_tests} "
        f"stagewise_cellmax_gap_pair_phase_tests={enclosed_pair_tests} "
        f"enclosure_erased_preimage_tests={erased_preimage_tests} "
        f"enclosure_erased_pair_phase_tests={erased_pair_tests} "
        f"earliest_raw_preimage_age_s={earliest_raw_age:.12g}",
        flush=True,
    )

    # Decision: do not preserve R9-N's no-preimage classification unless the
    # RAW-gap-containing cells themselves have no preimage.
    if raw_gap_cell_count > 0 and raw_pair_tests > 0 and enclosed_pair_tests == 0:
        classification = "REACHABLE_RAW_STAGE_GAP_ERASED_BY_STAGEWISE_CELL_CORNER_MAX_ENCLOSURE"
        next_action = "R9O_ADAPTIVELY_REFINE_ONLY_REACHABLE_RAW_GAP_CELLS_AND_RECOMPUTE_FULL_GFP"
    elif raw_gap_cell_count > 0 and raw_preimage_tests > 0 and raw_pair_tests == 0:
        classification = "RAW_STAGE_GAP_HAS_PREIMAGE_BUT_NOT_AT_MATCHED_PAIR_PHASE"
        next_action = "R9O_PHASE_ALIGN_SERVICE_PAIR_WITH_RAW_VALUE_GAP_PREIMAGE_BEFORE_NEW_GFP"
    elif raw_gap_cell_count > 0 and raw_preimage_tests == 0:
        classification = "RAW_STAGE_GAP_CELLS_EXIST_BUT_ONE_STEP_SUCCESSOR_IMAGE_MISSES_THEM"
        next_action = "R9O_REASSESS_PHYSICAL_SUCCESSOR_GEOMETRY_WITH_RAW_NOT_CELLMAX_SUPPORT"
    elif enclosed_pair_tests > 0:
        classification = "STAGEWISE_CELLMAX_GAP_IS_REACHABLE_BUT_PRIOR_VALUE_TRANSFER_AUDIT_INCONSISTENT"
        next_action = "R9O_AUDIT_VALUE_TRANSFER_INDEXING_BEFORE_ANY_PROMOTION"
    else:
        classification = "ENCLOSURE_CORRECTION_INCONCLUSIVE_FAIL_CLOSED"
        next_action = "R9O_MANUAL_GEOMETRY_AUDIT_REQUIRED"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result = RESULTS / f"P3B1_R9NR1_RESULT_{stamp}.json"
    latest = RESULTS / "P3B1_R9NR1_LATEST.json"
    csvp = RESULTS / f"P3B1_R9NR1_WITNESSES_{stamp}.csv"
    manifest = RESULTS / f"P3B1_R9NR1_MANIFEST_{stamp}.sha256"
    write_csv(csvp, witnesses[:256])

    payload = {
        "schema": "P3B1_R9NR1_ENCLOSURE_ERASURE_CORRECTION_V1",
        "status": "PASS",
        "classification": classification,
        "next_action": next_action,
        "hard_flags": {
            "r9n_original_no_preimage_classification_accepted_as_final": False,
            "continuous_state_separation_certified": False,
        },
        "metrics": {
            "raw_lookup_gap_nodes": raw_lookup_nodes,
            "raw_eval_semantic_gap_nodes": raw_eval_nodes,
            "max_raw_lookup_gap_m": raw_lookup_max,
            "max_raw_eval_gap_m": raw_eval_max,
            "upstream_global_last_stage_gap_m": global_gap,
            "raw_gap_containing_cells": raw_gap_cell_count,
            "raw_cell_max_corner_gap_m": raw_gap_cell_max,
            "stagewise_cellmax_gap_cells": enclosed_gap_cells,
            "stagewise_cellmax_max_gap_m": enclosed_gap_max,
            "enclosure_erased_cells": erased_cell_count,
            "preserved_gap_cells": preserved_cell_count,
            "raw_gap_cell_preimage_tests": raw_preimage_tests,
            "raw_gap_cell_preimage_nodes": int(np.count_nonzero(raw_preimage_nodes)),
            "raw_gap_pair_phase_tests": raw_pair_tests,
            "raw_gap_pair_phase_nodes": int(np.count_nonzero(raw_pair_nodes)),
            "stagewise_cellmax_gap_preimage_tests": enclosed_preimage_tests,
            "stagewise_cellmax_gap_pair_phase_tests": enclosed_pair_tests,
            "enclosure_erased_preimage_tests": erased_preimage_tests,
            "enclosure_erased_pair_phase_tests": erased_pair_tests,
            "earliest_raw_preimage_age_s": earliest_raw_age,
        },
        "artifacts": {"witness_csv": str(csvp)},
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    atomic_write(result, text)
    atomic_write(latest, text)

    files = [result, latest, csvp]
    lines = [f"{sha256_file(p)}  {p.name}" for p in files]
    atomic_write(manifest, "\n".join(lines) + "\n")

    print("=== R9-N-R1 DECISION ===", flush=True)
    print("R9NR1_ENCLOSURE_CORRECTION_GATE=PASS", flush=True)
    print(f"R9NR1_CLASSIFICATION={classification}", flush=True)
    print("R9N_ORIGINAL_CLASSIFICATION_ACCEPTED_AS_FINAL=NO", flush=True)
    print("CONTINUOUS_STATE_SEPARATION_CERTIFIED=NO", flush=True)
    print("R9NR1_EXECUTION=PASS", flush=True)
    print(f"R9NR1_NEXT_ACTION={next_action}", flush=True)
    print(f"RESULT_JSON={result}", flush=True)
    print(f"WITNESS_CSV={csvp}", flush=True)
    print(f"MANIFEST={manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
