from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path.home() / "Desktop/paper set/PQC_V2X_Security_Conditioned/numerical_experiments"
SRC = ROOT / "02_src"
CFGDIR = ROOT / "01_config"
RESULTS = ROOT / "04_results"

TOL = 1.0e-10
AXIS_NAMES = ("v_f", "v_p", "a_f", "bar_a", "bar_u", "age")


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
                seen.add(k)
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def resolve_historical_artifact(recorded_path: str | Path, current_dir: Path) -> tuple[Path, str]:
    """Resolve a historical absolute artifact path after workspace migration.

    Resolution is deliberately conservative:
      1. use the recorded path if it still exists;
      2. otherwise recover only the same basename from the current authoritative
         directory;
      3. otherwise fail.  No fuzzy filename substitution is allowed.
    """
    recorded = Path(recorded_path).expanduser()
    if recorded.is_file():
        return recorded.resolve(), "RECORDED_PATH"
    recovered = current_dir / recorded.name
    if recovered.is_file():
        return recovered.resolve(), "CURRENT_RESULTS_BASENAME_RECOVERY"
    raise FileNotFoundError(
        "R9B_HISTORICAL_ARTIFACT_NOT_FOUND "
        f"recorded={recorded} recovered_candidate={recovered}"
    )


def manifest_expected_hash_for_basename(results_dir: Path, basename: str) -> tuple[str | None, Path | None]:
    """Read R6 manifests as text and recover the archived hash for basename."""
    manifests = sorted(
        results_dir.glob("P3B1_R6_MANIFEST_*.sha256"),
        key=lambda q: q.stat().st_mtime,
        reverse=True,
    )
    for manifest in manifests:
        for raw in manifest.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or "  " not in line:
                continue
            digest, recorded = line.split("  ", 1)
            if Path(recorded.strip()).name == basename and len(digest) == 64:
                return digest.lower(), manifest
    return None, None


def validate_halo_spec_provenance(
    halo_spec: dict,
    r6_result: dict,
    eval_cfg: dict,
    resolved_path: Path,
) -> dict:
    selected_r6 = list(r6_result.get("selected_refinement_axes", []))
    selected_halo = list(halo_spec.get("selected_refinement_axes", []))
    if selected_r6 != selected_halo:
        raise RuntimeError(
            "R9B_HALO_SPEC_SELECTED_AXES_MISMATCH "
            f"r6={selected_r6} halo={selected_halo}"
        )

    evaluation_grid = halo_spec.get("evaluation_grid")
    lookup_grid = halo_spec.get("lookup_halo_grid")
    if not isinstance(evaluation_grid, dict) or not isinstance(lookup_grid, dict):
        raise RuntimeError("R9B_HALO_SPEC_GRID_KEYS_MISSING")

    for name in AXIS_NAMES:
        if name not in evaluation_grid or name not in lookup_grid:
            raise RuntimeError(f"R9B_HALO_SPEC_AXIS_MISSING={name}")
        expected = np.asarray(eval_cfg["grid"][name], dtype=float)
        recorded = np.asarray(evaluation_grid[name], dtype=float)
        lookup = np.asarray(lookup_grid[name], dtype=float)
        if expected.shape != recorded.shape or not np.allclose(
            expected, recorded, rtol=0.0, atol=1e-12
        ):
            raise RuntimeError(f"R9B_HALO_EVALUATION_GRID_MISMATCH={name}")
        if lookup.ndim != 1 or len(lookup) < 2 or not np.all(np.diff(lookup) > 0.0):
            raise RuntimeError(f"R9B_HALO_LOOKUP_AXIS_INVALID={name}")
        # Every evaluation node must be represented exactly in the halo.
        for x in expected:
            if not np.any(np.isclose(lookup, x, rtol=0.0, atol=1e-12)):
                raise RuntimeError(f"R9B_HALO_MISSING_EVAL_NODE={name}:{x}")

    product = int(np.prod([len(lookup_grid[name]) for name in AXIS_NAMES]))
    declared = int(r6_result.get("lookup_grid_nodes", product))
    if product != declared:
        raise RuntimeError(
            f"R9B_HALO_NODE_COUNT_MISMATCH product={product} declared={declared}"
        )

    actual_sha = sha256_file(resolved_path)
    expected_sha, manifest = manifest_expected_hash_for_basename(
        RESULTS, resolved_path.name
    )
    manifest_gate = "NOT_AVAILABLE"
    if expected_sha is not None:
        if actual_sha.lower() != expected_sha.lower():
            raise RuntimeError(
                "R9B_HALO_SPEC_MANIFEST_HASH_MISMATCH "
                f"expected={expected_sha} actual={actual_sha} manifest={manifest}"
            )
        manifest_gate = "PASS"

    return {
        "content_gate": "PASS",
        "actual_sha256": actual_sha,
        "manifest_expected_sha256": expected_sha,
        "manifest_path": str(manifest) if manifest else None,
        "manifest_gate": manifest_gate,
        "lookup_grid_nodes": product,
    }


def reachable_q_age_ticks(R: int, max_ticks: int) -> set[tuple[int, int]]:
    """Exact reachability for the frozen diagnostic countdown automaton.

    Start immediately after adoption at (q=R, age=0). For q>1 the service may
    defer or complete. For q=1 completion is mandatory. Upon completion the
    causal adoption map may adopt (age reset) or hold/discard (old age + 1), and
    a fresh service cycle starts at q=R.
    """
    start = (R, 0)
    reached = {start}
    frontier = [start]
    while frontier:
        q, a = frontier.pop()
        if a >= max_ticks:
            continue
        nxt: list[tuple[int, int]] = []
        # completion + adopt / completion + hold
        nxt.append((R, 0))
        nxt.append((R, a + 1))
        if q > 1:
            nxt.append((q - 1, a + 1))
        for s in nxt:
            if s[1] <= max_ticks and s not in reached:
                reached.add(s)
                frontier.append(s)
    return reached


def qage_gate_rows(R_values=(1, 2, 5, 10), max_ticks=20) -> list[dict]:
    rows = []
    for R in R_values:
        reachable = reachable_q_age_ticks(R, max_ticks)
        canonical = {
            (q, a)
            for q in range(1, R + 1)
            for a in range(max_ticks + 1)
            if a >= (R - q)
        }
        rows.append({
            "R": R,
            "max_ticks": max_ticks,
            "reachable_pairs": len(reachable),
            "canonical_pairs": len(canonical),
            "missing_from_reachable": len(canonical - reachable),
            "extra_vs_canonical": len(reachable - canonical),
            "canonical_qage_gate_exact": reachable == canonical,
        })
    return rows


def self_test() -> None:
    rows = qage_gate_rows()
    assert all(r["canonical_qage_gate_exact"] for r in rows), rows
    # Basic causality ordering: completion then adopt/hold, not forced adoption.
    r = reachable_q_age_ticks(2, 3)
    assert (2, 1) in r  # completion+hold
    assert (1, 1) in r  # defer
    # Historical-path migration recovery must preserve exact basename.
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "04_results"
        d.mkdir()
        f = d / "artifact.json"
        f.write_text("{}\n", encoding="utf-8")
        resolved, mode = resolve_historical_artifact(
            "/obsolete/project/04_results/artifact.json", d
        )
        assert resolved == f.resolve()
        assert mode == "CURRENT_RESULTS_BASENAME_RECOVERY"
        try:
            resolve_historical_artifact(
                "/obsolete/project/04_results/not-the-same-name.json", d
            )
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("resolver accepted a fuzzy/nonexistent artifact")
    print("R9B_R1_INTERNAL_SELF_TEST=PASS", flush=True)


def finite_sup_abs(a: np.ndarray, b: np.ndarray) -> float:
    x = np.abs(np.asarray(a, float) - np.asarray(b, float))
    finite = np.isfinite(x)
    return float(np.max(x[finite])) if np.any(finite) else math.inf


def lookup_future(cell_flat, q_index, index, valid):
    out = np.full(len(index), np.inf, dtype=float)
    mask = np.asarray(valid, dtype=bool)
    out[mask] = cell_flat[q_index][index[mask]]
    return out


def solve_causal_frozen_halo(
    R: int,
    cfg: dict,
    eval_node_indices: np.ndarray,
    lookup_shape: tuple[int, ...],
    lookup_fallback: np.ndarray,
    eval_fallback: np.ndarray,
    transition_data: dict,
    r3,
    label: str = "causal",
) -> SolveResult:
    lookup_nodes = int(np.prod(lookup_shape))
    h_flat = np.repeat(np.asarray(lookup_fallback)[:, None], R, axis=1)
    eval_mask = np.zeros(lookup_nodes, dtype=bool)
    eval_mask[eval_node_indices] = True
    outside = ~eval_mask
    tol = float(cfg["fixed_point"]["tolerance_m"])
    max_iter = int(cfg["fixed_point"]["max_iterations"])
    outside_change_max = 0.0

    for it in range(1, max_iter + 1):
        h = h_flat.reshape(lookup_shape + (R,))
        cell_max = r3.cell_corner_max(h)
        cell_flat = [cell_max[..., q].reshape(-1) for q in range(R)]
        old_eval = h_flat[eval_node_indices, :].copy()
        new_eval = old_eval.copy()

        for r in range(1, R + 1):
            q_index = r - 1
            best = np.full(len(eval_node_indices), np.inf, dtype=float)
            for trans in transition_data["transitions"]:
                adopt_future = lookup_future(
                    cell_flat, R - 1,
                    trans["completion_index"], trans["completion_valid"]
                )
                hold_future = lookup_future(
                    cell_flat, R - 1,
                    trans["defer_index"], trans["defer_valid"]
                )
                completion_future = np.minimum(adopt_future, hold_future)
                future = completion_future
                if r > 1:
                    defer_future = lookup_future(
                        cell_flat, r - 2,
                        trans["defer_index"], trans["defer_valid"]
                    )
                    future = np.maximum(completion_future, defer_future)
                required = np.maximum(
                    trans["step_loss_upper"],
                    trans["closing_end"] + future,
                )
                best = np.minimum(best, required)

            best = np.maximum(best, 0.0)
            candidate = np.minimum(np.asarray(eval_fallback), best)
            new_eval[:, q_index] = np.minimum(old_eval[:, q_index], candidate)

        delta = finite_sup_abs(new_eval, old_eval)
        h_flat[eval_node_indices, :] = new_eval
        if np.any(outside):
            expected = np.repeat(np.asarray(lookup_fallback)[outside, None], R, axis=1)
            outside_change_max = max(
                outside_change_max,
                finite_sup_abs(h_flat[outside, :], expected),
            )
        if it == 1 or it % 5 == 0 or delta <= tol:
            print(
                f"R9B_ITER label={label} iteration={it} "
                f"delta_m={delta:.12g} outside_change_m={outside_change_max:.12g}",
                flush=True,
            )
        if delta <= tol:
            return SolveResult(h_flat, True, it, delta, outside_change_max)

    return SolveResult(h_flat, False, max_iter, delta, outside_change_max)


def q_metrics(h_flat: np.ndarray, eval_idx: np.ndarray, semantic_mask: np.ndarray | None = None) -> dict:
    h = np.asarray(h_flat[eval_idx, :], dtype=float)
    if semantic_mask is None:
        row_ok = np.ones(h.shape[0], dtype=bool)
    else:
        # A physical node is comparable only if >=2 q values are semantically admissible.
        row_ok = np.sum(semantic_mask, axis=1) >= 2
    if h.shape[1] <= 1 or not np.any(row_ok):
        return {"comparable_nodes": int(np.count_nonzero(row_ok)), "qdep_nodes": 0, "max_q_span_m": 0.0}
    spans = []
    for i in np.flatnonzero(row_ok):
        vals = h[i, :] if semantic_mask is None else h[i, semantic_mask[i, :]]
        if len(vals) >= 2:
            spans.append(float(np.max(vals) - np.min(vals)))
    a = np.asarray(spans, dtype=float)
    return {
        "comparable_nodes": int(len(a)),
        "qdep_nodes": int(np.count_nonzero(a > TOL)),
        "max_q_span_m": float(np.max(a)) if len(a) else 0.0,
        "p95_positive_q_span_m": float(np.quantile(a[a > TOL], .95)) if np.any(a > TOL) else 0.0,
    }


def exact_qage_mask(age_flat: np.ndarray, R: int, Ts: float, tol: float = 1e-9) -> tuple[np.ndarray, dict]:
    """Exact grid-node reachability under the frozen age-reset-to-zero model.

    Because the frozen completion endpoint sets age=0 and subsequent defer/hold
    increments age by exactly Ts, only ages on the Ts lattice are exact sampled
    states. Non-lattice grid nodes remain useful interpolation support but are not
    promoted as semantically reachable sampled states.
    """
    age = np.asarray(age_flat, dtype=float)
    ticks_real = age / Ts
    ticks = np.rint(ticks_real).astype(int)
    on_lattice = np.abs(ticks_real - ticks) <= tol
    max_ticks = int(np.max(ticks[on_lattice])) if np.any(on_lattice) else 0
    reach = reachable_q_age_ticks(R, max_ticks)
    mask = np.zeros((len(age), R), dtype=bool)
    for i in range(len(age)):
        if not on_lattice[i]:
            continue
        for q in range(1, R + 1):
            mask[i, q - 1] = (q, int(ticks[i])) in reach
    diag = {
        "age_nodes": int(len(age)),
        "age_nodes_on_Ts_lattice": int(np.count_nonzero(on_lattice)),
        "age_nodes_off_Ts_lattice": int(np.count_nonzero(~on_lattice)),
        "fraction_on_Ts_lattice": float(np.mean(on_lattice)),
    }
    return mask, diag



TARGET_BAR_A = np.asarray([
    -2.4140625,
    -2.328125,
    -2.2421875,
    -2.15625,
    -2.0703125,
    -1.984375,
], dtype=float)
EXPECTED_TARGET_EVAL_NODES = 315315
EXPECTED_TARGET_LOOKUP_NODES = 710775


def insert_exact(axis, values):
    return np.asarray(sorted(set(float(x) for x in np.concatenate([
        np.asarray(axis, dtype=float), np.asarray(values, dtype=float)
    ]))), dtype=float)


def reconstruct_targeted_geometry(r3, r5, r6, r7):
    """Reconstruct the actual 315315-node targeted witness abstraction.

    R6 is the 143325-node v_p/bar_u refinement plus a frozen halo.  The
    manuscript-facing targeted solve then inserts the six discovery-stage bar_a
    coordinates into both evaluation and lookup grids.  R9-C reconstructs that
    exact geometry instead of auditing the earlier R6 grid.
    """
    cfg = json.loads((CFGDIR / "p3b1_augmented_fixed_point_v1.json").read_text())
    p1 = json.loads((CFGDIR / "p1_validation_v2.json").read_text())
    p2b = json.loads((CFGDIR / "p2b_hybrid_fallback_v1.json").read_text())
    p2c = json.loads((CFGDIR / "p2c_switching_guard_v1.json").read_text())
    p3a = json.loads((CFGDIR / "p3a_information_contract_v1.json").read_text())

    r6_path = RESULTS / "P3B1_R6_LATEST.json"
    r6_result = json.loads(r6_path.read_text(encoding="utf-8"))
    if r6_result.get("status") != "PASS":
        raise RuntimeError("R6_NOT_PASS")

    selected_axes = list(r6_result["selected_refinement_axes"])
    coarse_refined_cfg = r5.refine_cfg(cfg, tuple(selected_axes))

    recorded_halo_spec = str(r6_result["halo_spec"])
    halo_spec_path, halo_resolution_mode = resolve_historical_artifact(
        recorded_halo_spec, RESULTS
    )
    halo_spec = json.loads(halo_spec_path.read_text(encoding="utf-8"))
    provenance = validate_halo_spec_provenance(
        halo_spec, r6_result, coarse_refined_cfg, halo_spec_path
    )

    target_cfg = copy.deepcopy(coarse_refined_cfg)
    target_cfg["grid"]["bar_a"] = [
        float(x) for x in insert_exact(
            target_cfg["grid"]["bar_a"], TARGET_BAR_A
        )
    ]
    eval_axes, eval_flat, eval_shape = r3.build_grid(target_cfg)

    lookup_axes = [
        np.asarray(halo_spec["lookup_halo_grid"][n], dtype=float)
        for n in AXIS_NAMES
    ]
    j_bara = AXIS_NAMES.index("bar_a")
    lookup_axes[j_bara] = insert_exact(lookup_axes[j_bara], TARGET_BAR_A)

    lookup_flat, lookup_shape = r6.mesh_flat(lookup_axes)
    eval_idx = r7.exact_node_indices(eval_axes, lookup_axes)

    eval_nodes = int(np.prod(eval_shape))
    lookup_nodes = int(np.prod(lookup_shape))
    if eval_nodes != EXPECTED_TARGET_EVAL_NODES:
        raise RuntimeError(
            f"R9C_TARGET_EVAL_NODE_COUNT_MISMATCH={eval_nodes}"
        )
    if lookup_nodes != EXPECTED_TARGET_LOOKUP_NODES:
        raise RuntimeError(
            f"R9C_TARGET_LOOKUP_NODE_COUNT_MISMATCH={lookup_nodes}"
        )

    # Recompute the fallback requirement on the exact targeted halo.  This is
    # the authoritative safe way to move from the R6 halo to the 6-point bar_a
    # insertion; no interpolation of h_F is used.
    lookup_fallback = r6.fallback_required_on_grid(
        target_cfg, p1, p2b, p2c, p3a, lookup_flat
    )
    eval_fallback = lookup_fallback[eval_idx]

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
        "lookup_fallback": lookup_fallback,
        "eval_fallback": eval_fallback,
        "r6_latest_path": str(r6_path),
        "r6_latest_sha256": sha256_file(r6_path),
        "halo_spec_recorded_path": recorded_halo_spec,
        "halo_spec_resolved_path": str(halo_spec_path),
        "halo_spec_resolution_mode": halo_resolution_mode,
        "halo_spec_provenance": provenance,
    }


def build_transition_data_chunked(data, actions, r3, b0, chunk_size=8192):
    """Memory-bounded transition construction for the targeted grid.

    This reproduces the frozen R3 one-step semantics for negative cooperative
    actions, but processes states in chunks so the 315315 x 65 trajectory arrays
    are never materialized for the entire grid at once.
    """
    cfg = copy.deepcopy(data["eval_cfg"])
    cfg["cooperative_actions"] = [float(a) for a in actions]
    vf, vp, af, bar_a, bar_u, age = [
        np.asarray(x, dtype=float) for x in data["eval_flat"]
    ]
    n = len(vf)
    p1 = data["p1"]
    p2b = data["p2b"]
    p3a = data["p3a"]
    axes = data["lookup_axes"]
    Ts = float(p1["plant"]["Ts"])
    J = float(cfg["information_contract"]["slew_rate"])
    nt = int(cfg["one_step"]["trajectory_points"])
    times = np.linspace(0.0, Ts, nt)
    sd = p2b["state_domain"]
    speed_bound = float(sd["v_f_max"]) + float(sd["v_p_max"])
    dt_sample = Ts / (nt - 1)
    lipschitz_correction = 0.5 * speed_bound * dt_sample

    outs = []
    for action in actions:
        outs.append({
            "action": float(action),
            "step_loss_upper": np.empty(n, dtype=float),
            "closing_end": np.empty(n, dtype=float),
            "completion_index": np.empty(n, dtype=np.int64),
            "completion_valid": np.empty(n, dtype=bool),
            "defer_index": np.empty(n, dtype=np.int64),
            "defer_valid": np.empty(n, dtype=bool),
        })

    for start in range(0, n, int(chunk_size)):
        stop = min(n, start + int(chunk_size))
        sl = slice(start, stop)
        vfc, vpc, afc = vf[sl], vp[sl], af[sl]
        bac, buc, agc = bar_a[sl], bar_u[sl], age[sl]
        m = stop - start

        ap_lower = b0.predecessor_acceleration_lower(
            agc, bac, buc, J, p3a, p2b
        )
        Pp, Vp, Ap, Up, _ = r3.predecessor_motion_with_stop(
            vpc, ap_lower, buc, agc, times, J, p3a
        )
        pp_end = Pp[:, -1]
        vp_end = Vp[:, -1]
        ap_end = Ap[:, -1]
        up_end = Up[:, -1]
        age_defer = agc + Ts

        for out in outs:
            action = float(out["action"])
            # All stages before generalized-action discovery use braking actions
            # for which the legacy stop-aware propagator is defined.
            tau = float(p2b["follower"]["tau"])
            w = float(p1["uncertainty"]["follower_actuation_abs"])
            if action + tau * w >= 0.0:
                raise RuntimeError(
                    "R9C_CHUNKED_LEGACY_PROPAGATOR_REQUIRES_NEGATIVE_EQUILIBRIUM "
                    f"action={action} equilibrium={action + tau*w}"
                )
            Pf, Vf, Af = r3.follower_motion(
                vfc, afc, action, times, p1, p2b
            )
            closing = Pf - Pp
            out["step_loss_upper"][sl] = np.maximum(
                np.max(closing, axis=1) + lipschitz_correction, 0.0
            )
            out["closing_end"][sl] = Pf[:, -1] - pp_end

            defer_values = [
                Vf[:, -1], vp_end, Af[:, -1], bac, buc, age_defer
            ]
            completion_values = [
                Vf[:, -1], vp_end, Af[:, -1], ap_end, up_end,
                np.zeros(m, dtype=float),
            ]
            didx, dvalid = r3.locate_cells(axes, defer_values)
            cidx, cvalid = r3.locate_cells(axes, completion_values)
            out["defer_index"][sl] = didx
            out["defer_valid"][sl] = dvalid
            out["completion_index"][sl] = cidx
            out["completion_valid"][sl] = cvalid

        if start == 0 or stop == n or (stop // int(chunk_size)) % 8 == 0:
            print(
                f"R9C_TRANSITION_PROGRESS states={stop}/{n} actions={len(actions)}",
                flush=True,
            )

    invalid = sum(
        int(np.count_nonzero(~o["completion_valid"]))
        + int(np.count_nonzero(~o["defer_valid"]))
        for o in outs
    )
    if invalid:
        raise RuntimeError(f"R9C_TARGET_HALO_COVERAGE_FAIL invalid={invalid}")
    return cfg, {
        "fallback_required": np.asarray(data["eval_fallback"], dtype=float),
        "transitions": outs,
        "lipschitz_correction_m": float(lipschitz_correction),
    }


def frozen_transition_equivalence_gate(data, td_chunk, actions, r7):
    """Compare chunked transitions with the original builder on a sparse sample."""
    sample = np.linspace(0, len(data["eval_flat"][0]) - 1, 257, dtype=int)
    flat = [np.asarray(x)[sample] for x in data["eval_flat"]]
    cfg = copy.deepcopy(data["eval_cfg"])
    cfg["cooperative_actions"] = [float(a) for a in actions]
    original = r7.build_transition_data_to_lookup(
        cfg, data["p1"], data["p2b"], data["p2c"], data["p3a"],
        flat, data["lookup_axes"]
    )
    max_float = 0.0
    mismatch = 0
    for a, b in zip(td_chunk["transitions"], original["transitions"]):
        max_float = max(
            max_float,
            float(np.max(np.abs(a["step_loss_upper"][sample] - b["step_loss_upper"]))),
            float(np.max(np.abs(a["closing_end"][sample] - b["closing_end"]))),
        )
        mismatch += int(np.count_nonzero(a["completion_index"][sample] != b["completion_index"]))
        mismatch += int(np.count_nonzero(a["defer_index"][sample] != b["defer_index"]))
        mismatch += int(np.count_nonzero(a["completion_valid"][sample] != b["completion_valid"]))
        mismatch += int(np.count_nonzero(a["defer_valid"][sample] != b["defer_valid"]))
    passed = max_float <= 1e-12 and mismatch == 0
    return {
        "pass": bool(passed),
        "max_float_error": float(max_float),
        "index_or_valid_mismatch": int(mismatch),
        "sample_count": int(len(sample)),
    }


def qdep_indices(h_flat, eval_idx, semantic_mask=None):
    h = np.asarray(h_flat[eval_idx, :], dtype=float)
    if semantic_mask is None:
        spans = np.max(h, axis=1) - np.min(h, axis=1)
        return np.flatnonzero(spans > TOL), spans
    spans = np.zeros(h.shape[0], dtype=float)
    comparable = np.sum(semantic_mask, axis=1) >= 2
    for i in np.flatnonzero(comparable):
        v = h[i, semantic_mask[i]]
        spans[i] = float(np.max(v) - np.min(v))
    return np.flatnonzero(spans > TOL), spans


def generalized_follower_single(v0, a0, action, times, p1, p2b, r3):
    """Discovery-only general cooperative propagator.

    For negative equilibrium acceleration we preserve the frozen stop-aware
    dynamics exactly.  For nonnegative equilibrium we use the exact unconstrained
    lag trajectory only when velocity stays nonnegative throughout the sampled
    interval.  Cases requiring unilateral stop/restart mechanics are explicitly
    returned as AMBIGUOUS and are never used to claim a safe action.
    """
    tau = float(p2b["follower"]["tau"])
    w = float(p1["uncertainty"]["follower_actuation_abs"])
    eq = float(action) + tau * w
    vf = np.asarray([float(v0)], dtype=float)
    af = np.asarray([float(a0)], dtype=float)
    if eq < -1e-12:
        P, V, A = r3.follower_motion(vf, af, float(action), times, p1, p2b)
        return P[0], V[0], A[0], "LEGACY_STOP_AWARE"

    t = np.asarray(times, dtype=float)
    P, V, A = r3.lag_linear_motion(
        vf[:, None], af[:, None], float(action), 0.0,
        t[None, :], tau, w
    )
    P, V, A = P[0], V[0], A[0]
    if float(np.min(V)) < -1e-10:
        return P, V, A, "AMBIGUOUS_UNILATERAL_CONTACT"
    return P, V, A, "RAW_NONNEGATIVE_SPEED"


def witness_action_discovery(data, causal_h_flat, R, witness_idx, r3, b0):
    """Coarse discovery over the wider action interval at exact witness nodes.

    This is not a continuous-action certificate.  It only decides whether a
    broader action class can obviously beat the midpoint hull at the surviving
    semantically admissible q-dependent nodes.
    """
    if len(witness_idx) == 0:
        return [], {"scanned_witnesses": 0, "material_witnesses": 0}

    p1, p2b, p3a = data["p1"], data["p2b"], data["p3a"]
    cfg = data["eval_cfg"]
    Ts = float(p1["plant"]["Ts"])
    J = float(cfg["information_contract"]["slew_rate"])
    nt = int(cfg["one_step"]["trajectory_points"])
    times = np.linspace(0.0, Ts, nt)
    sd = p2b["state_domain"]
    lips = 0.5 * (float(sd["v_f_max"]) + float(sd["v_p_max"])) * Ts / (nt - 1)
    h = np.asarray(causal_h_flat, dtype=float).reshape(data["lookup_shape"] + (R,))
    cell = r3.cell_corner_max(h)
    cell_flat = [cell[..., q].reshape(-1) for q in range(R)]

    # u=-6 is the instantiated fallback command and has a mode-semantics
    # ambiguity if reused as a cooperative command.  Discovery scans strictly
    # above it and reports that endpoint separately.
    actions = np.unique(np.concatenate([
        np.linspace(-5.75, 2.5, 34),
        np.asarray([-3.0, -2.5, -2.0, -1.5, -1.0]),
    ]))

    vf, vp, af, bara, baru, age = [np.asarray(x, float) for x in data["eval_flat"]]
    rows = []
    material = 0
    for i in witness_idx:
        ap0 = b0.predecessor_acceleration_lower(
            np.asarray([age[i]]), np.asarray([bara[i]]), np.asarray([baru[i]]),
            J, p3a, p2b
        )
        Pp, Vp, Ap, Up, _ = r3.predecessor_motion_with_stop(
            np.asarray([vp[i]]), ap0, np.asarray([baru[i]]),
            np.asarray([age[i]]), times, J, p3a
        )
        base_threshold = np.asarray(causal_h_flat[data["eval_idx"][i], :], float)
        best_scan = base_threshold.copy()
        best_action = [None] * R
        valid_actions = 0
        ambiguous_actions = 0
        outside_actions = 0
        for action in actions:
            Pf, Vf, Af, status = generalized_follower_single(
                vf[i], af[i], float(action), times, p1, p2b, r3
            )
            if status == "AMBIGUOUS_UNILATERAL_CONTACT":
                ambiguous_actions += 1
                continue
            closing = Pf - Pp[0]
            step_loss = max(float(np.max(closing)) + lips, 0.0)
            closing_end = float(Pf[-1] - Pp[0, -1])
            defer_values = [
                np.asarray([Vf[-1]]), np.asarray([Vp[0, -1]]), np.asarray([Af[-1]]),
                np.asarray([bara[i]]), np.asarray([baru[i]]), np.asarray([age[i] + Ts]),
            ]
            completion_values = [
                np.asarray([Vf[-1]]), np.asarray([Vp[0, -1]]), np.asarray([Af[-1]]),
                np.asarray([Ap[0, -1]]), np.asarray([Up[0, -1]]), np.asarray([0.0]),
            ]
            didx, dvalid = r3.locate_cells(data["lookup_axes"], defer_values)
            cidx, cvalid = r3.locate_cells(data["lookup_axes"], completion_values)
            if not (bool(dvalid[0]) and bool(cvalid[0])):
                outside_actions += 1
                continue
            valid_actions += 1
            adopt = float(cell_flat[R - 1][cidx[0]])
            hold = float(cell_flat[R - 1][didx[0]])
            completion = min(adopt, hold)
            for q in range(1, R + 1):
                future = completion
                if q > 1:
                    future = max(future, float(cell_flat[q - 2][didx[0]]))
                req = max(step_loss, closing_end + future, 0.0)
                req = min(float(data["eval_fallback"][i]), req)
                if req < best_scan[q - 1]:
                    best_scan[q - 1] = req
                    best_action[q - 1] = float(action)
        reduction = base_threshold - best_scan
        is_material = float(np.max(reduction)) > TOL
        material += int(is_material)
        rows.append({
            "node_index": int(i),
            "v_f": float(vf[i]), "v_p": float(vp[i]), "a_f": float(af[i]),
            "bar_a": float(bara[i]), "bar_u": float(baru[i]), "age": float(age[i]),
            "base_q_span_m": float(np.max(base_threshold) - np.min(base_threshold)),
            "scan_q_span_m": float(np.max(best_scan) - np.min(best_scan)),
            "max_threshold_reduction_m": float(np.max(reduction)),
            "material": bool(is_material),
            "valid_actions": int(valid_actions),
            "ambiguous_unilateral_actions": int(ambiguous_actions),
            "outside_halo_actions": int(outside_actions),
            "best_action_q1": best_action[0],
            "best_action_qR": best_action[-1],
        })
    return rows, {
        "scanned_witnesses": int(len(witness_idx)),
        "material_witnesses": int(material),
        "action_grid_count": int(len(actions)),
        "fallback_endpoint_u_min_mode_ambiguous": True,
        "scan_is_discovery_only": True,
    }


def self_test_r9c():
    a = insert_exact(np.asarray([-2.5, 0.25]), [-2.4140625])
    assert np.all(np.diff(a) > 0)
    assert np.any(np.isclose(a, -2.4140625, atol=0, rtol=0))
    assert EXPECTED_TARGET_EVAL_NODES == 315315
    assert EXPECTED_TARGET_LOOKUP_NODES == 710775
    print("R9C_INTERNAL_SELF_TEST=PASS", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--profile", default="diagnostic_fast")
    ap.add_argument("--chunk-size", type=int, default=8192)
    args = ap.parse_args()
    self_test()
    self_test_r9c()
    if args.self_test:
        return 0

    sys.path.insert(0, str(SRC))
    import p3b1_augmented_fixed_point_v1_r3 as r3
    import p3b1_r5_refinement_attribution as r5
    import p3b1_r6_continuation_halo as r6
    import p3b1_r7_frozen_halo_fixed_point as r7
    import p3b0_freshness_service_audit_v1 as b0

    data = reconstruct_targeted_geometry(r3, r5, r6, r7)
    profile = args.profile
    profiles = data["p1"]["diagnostic_service_profiles"]
    if profile not in profiles:
        raise KeyError(profile)
    R = r7.service_horizon(profile, profiles[profile])
    Ts = float(data["p1"]["plant"]["Ts"])

    print("=== P3-B1-R9-C TARGETED HISTORY-ALIGNED CLOSURE GATE ===", flush=True)
    print(f"PROFILE={profile} R={R}", flush=True)
    print("GEOMETRY=TARGETED_315315_WITH_SIX_BAR_A_INSERTIONS", flush=True)
    print(f"BASE_REFINEMENT_AXES={','.join(data['selected_axes'])}", flush=True)
    print("TARGETED_INSERT_AXIS=bar_a", flush=True)
    print(
        "TARGETED_BAR_A=" + ",".join(f"{x:.10g}" for x in TARGET_BAR_A),
        flush=True,
    )
    print(
        f"R9C_INPUT_RECOVERY mode={data['halo_spec_resolution_mode']} "
        f"resolved={data['halo_spec_resolved_path']}", flush=True
    )
    print(
        f"R9C_PROVENANCE content_gate={data['halo_spec_provenance']['content_gate']} "
        f"manifest_gate={data['halo_spec_provenance']['manifest_gate']} "
        f"sha256={data['halo_spec_provenance']['actual_sha256']}", flush=True
    )
    print(
        f"EVAL_NODES={int(np.prod(data['eval_shape']))} "
        f"LOOKUP_NODES={int(np.prod(data['lookup_shape']))}", flush=True
    )

    semantic_mask, age_diag = exact_qage_mask(
        np.asarray(data["eval_flat"][-1], float), R, Ts
    )
    print(
        f"QAGE_TS_LATTICE on={age_diag['age_nodes_on_Ts_lattice']} "
        f"off={age_diag['age_nodes_off_Ts_lattice']} "
        f"fraction={age_diag['fraction_on_Ts_lattice']:.12g}", flush=True
    )

    frozen_actions = [-3.0, -2.0, -1.0]
    hull_actions = [-3.0, -2.5, -2.0, -1.5, -1.0]

    print("R9C_STAGE_START=build_frozen_transitions_chunked", flush=True)
    cfg_frozen, td_frozen = build_transition_data_chunked(
        data, frozen_actions, r3, b0, chunk_size=args.chunk_size
    )
    eq_gate = frozen_transition_equivalence_gate(
        data, td_frozen, frozen_actions, r7
    )
    print(
        f"R9C_FROZEN_TRANSITION_EQUIVALENCE pass={eq_gate['pass']} "
        f"max_float_error={eq_gate['max_float_error']:.3e} "
        f"mismatch={eq_gate['index_or_valid_mismatch']}", flush=True
    )
    if not eq_gate["pass"]:
        raise RuntimeError("R9C_FROZEN_TRANSITION_EQUIVALENCE_FAIL")

    print("R9C_STAGE_START=legacy_targeted", flush=True)
    legacy = r7.solve_frozen_halo_profile(
        profile, R, cfg_frozen, data["eval_idx"], len(data["eval_idx"]),
        data["lookup_shape"], data["lookup_fallback"], data["eval_fallback"],
        td_frozen,
    )
    legacy_metrics = q_metrics(legacy["h_flat"], data["eval_idx"])
    legacy_exact = q_metrics(
        legacy["h_flat"], data["eval_idx"], semantic_mask
    )
    print(
        f"R9C_LEGACY_TARGETED converged={legacy['converged']} "
        f"iterations={legacy['iterations']} qdep_all={legacy_metrics['qdep_nodes']} "
        f"qdep_exact_qage={legacy_exact['qdep_nodes']} "
        f"max_qspan_m={legacy_metrics['max_q_span_m']:.12g}", flush=True
    )
    if not legacy["converged"]:
        raise RuntimeError("R9C_LEGACY_TARGETED_NOT_CONVERGED")
    if legacy_metrics["qdep_nodes"] != 16:
        raise RuntimeError(
            "R9C_TARGETED_BASELINE_REPRODUCTION_FAIL "
            f"expected_qdep=16 actual={legacy_metrics['qdep_nodes']}"
        )

    print("R9C_STAGE_START=causal_targeted", flush=True)
    causal = solve_causal_frozen_halo(
        R, cfg_frozen, data["eval_idx"], data["lookup_shape"],
        data["lookup_fallback"], data["eval_fallback"], td_frozen, r3,
        label="r9c_causal_targeted",
    )
    if not causal.converged:
        raise RuntimeError("R9C_CAUSAL_TARGETED_NOT_CONVERGED")
    causal_metrics = q_metrics(causal.h_flat, data["eval_idx"])
    causal_exact = q_metrics(causal.h_flat, data["eval_idx"], semantic_mask)

    A = np.asarray(legacy["h_flat"][data["eval_idx"], :], float)
    B = np.asarray(causal.h_flat[data["eval_idx"], :], float)
    causal_changed = int(
        np.count_nonzero(np.max(np.abs(A - B), axis=1) > TOL)
    )
    print(
        f"R9C_CAUSAL changed_nodes={causal_changed} "
        f"qdep_all={causal_metrics['qdep_nodes']} "
        f"qdep_exact_qage={causal_exact['qdep_nodes']} "
        f"max_reduction_m={float(np.max(A-B)):.12g}", flush=True
    )

    print("R9C_STAGE_START=build_midpoint_transitions_chunked", flush=True)
    cfg_hull, td_hull = build_transition_data_chunked(
        data, hull_actions, r3, b0, chunk_size=args.chunk_size
    )
    print("R9C_STAGE_START=causal_midpoint_targeted", flush=True)
    hull = solve_causal_frozen_halo(
        R, cfg_hull, data["eval_idx"], data["lookup_shape"],
        data["lookup_fallback"], data["eval_fallback"], td_hull, r3,
        label="r9c_causal_hull_targeted",
    )
    if not hull.converged:
        raise RuntimeError("R9C_HULL_TARGETED_NOT_CONVERGED")
    hull_metrics = q_metrics(hull.h_flat, data["eval_idx"])
    hull_exact = q_metrics(hull.h_flat, data["eval_idx"], semantic_mask)
    C = np.asarray(hull.h_flat[data["eval_idx"], :], float)
    hull_changed = int(
        np.count_nonzero(np.max(np.abs(B - C), axis=1) > TOL)
    )
    print(
        f"R9C_ACTION_HULL changed_nodes={hull_changed} "
        f"qdep_all={hull_metrics['qdep_nodes']} "
        f"qdep_exact_qage={hull_exact['qdep_nodes']} "
        f"max_reduction_m={float(np.max(B-C)):.12g}", flush=True
    )

    legacy_widx, legacy_spans = qdep_indices(
        legacy["h_flat"], data["eval_idx"]
    )
    exact_widx, exact_spans = qdep_indices(
        hull.h_flat, data["eval_idx"], semantic_mask
    )

    vf, vp, af, bara, baru, age = [
        np.asarray(x, float) for x in data["eval_flat"]
    ]
    witness_rows = []
    for i in legacy_widx:
        witness_rows.append({
            "node_index": int(i),
            "v_f": float(vf[i]), "v_p": float(vp[i]), "a_f": float(af[i]),
            "bar_a": float(bara[i]), "bar_u": float(baru[i]), "age": float(age[i]),
            "legacy_qspan_m": float(legacy_spans[i]),
            "causal_qspan_m": float(np.max(B[i]) - np.min(B[i])),
            "hull_qspan_m": float(np.max(C[i]) - np.min(C[i])),
            "qage_admissible_count": int(np.sum(semantic_mask[i, :])),
            "semantically_comparable": bool(np.sum(semantic_mask[i, :]) >= 2),
            "survives_exact_qage": bool(i in set(int(x) for x in exact_widx)),
        })

    action_rows = []
    action_summary = {
        "scanned_witnesses": 0,
        "material_witnesses": 0,
        "scan_is_discovery_only": True,
    }
    # Efficiency/correctness gate: wider action discovery has theorem value only
    # after a q-dependent layer survives corrected causal semantics on exact
    # history-admissible q-age states.
    if hull_exact["qdep_nodes"] > 0:
        print("R9C_STAGE_START=general_action_witness_discovery", flush=True)
        action_rows, action_summary = witness_action_discovery(
            data, hull.h_flat, R, exact_widx, r3, b0
        )
        print(
            f"R9C_GENERAL_ACTION_DISCOVERY witnesses={action_summary['scanned_witnesses']} "
            f"material={action_summary['material_witnesses']} "
            f"action_grid={action_summary.get('action_grid_count',0)}", flush=True
        )
    else:
        print(
            "R9C_GENERAL_ACTION_DISCOVERY_SKIPPED="
            "NO_Q_LAYER_ON_EXACT_HISTORY_ADMISSIBLE_STATES",
            flush=True,
        )

    if legacy_exact["qdep_nodes"] == 0:
        classification = "TARGETED_FINITE_Q_LAYER_OFF_EXACT_QAGE_HISTORY_LATTICE"
        next_action = "R9D_REBUILD_HISTORY_ALIGNED_AGE_GRID_AND_SERVICE_AUTOMATON"
    elif causal_exact["qdep_nodes"] == 0:
        classification = "CAUSAL_ADOPTION_REMOVES_EXACT_HISTORY_Q_LAYER"
        next_action = "R9D_RECOMPUTE_FINITE_CERTIFICATES_WITH_CAUSAL_ADOPTION"
    elif hull_exact["qdep_nodes"] == 0:
        classification = "MIDPOINT_ACTION_HULL_REMOVES_EXACT_HISTORY_Q_LAYER"
        next_action = "R9D_CONTINUOUS_ACTION_MODEL_AND_INTERVAL_COVER"
    elif action_summary.get("material_witnesses", 0) > 0:
        classification = "WIDER_ACTION_CLASS_MATERIAL_AT_EXACT_HISTORY_Q_WITNESSES"
        next_action = "R9D_GENERAL_HYBRID_PROPAGATOR_AND_ACTION_INTERVAL_BRANCH_BOUND"
    else:
        classification = "Q_LAYER_PERSISTS_AFTER_CAUSAL_HISTORY_AND_MIDPOINT_ACTION_GATES"
        next_action = "R9D_CONTINUOUS_INNER_OUTER_KERNEL_ON_CORRECTED_MODEL"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    qage_csv = RESULTS / f"P3B1_R9C_QAGE_{stamp}.csv"
    witness_csv = RESULTS / f"P3B1_R9C_TARGETED_WITNESSES_{stamp}.csv"
    action_csv = RESULTS / f"P3B1_R9C_ACTION_DISCOVERY_{stamp}.csv"
    result_json = RESULTS / f"P3B1_R9C_RESULT_{stamp}.json"
    latest_json = RESULTS / "P3B1_R9C_LATEST.json"
    manifest = RESULTS / f"P3B1_R9C_MANIFEST_{stamp}.sha256"

    write_csv(qage_csv, qage_gate_rows((1,2,5,10), max_ticks=int(round(2.0/Ts))))
    write_csv(witness_csv, witness_rows)
    write_csv(action_csv, action_rows)

    output = {
        "schema": "SCV_P3B1_R9C_TARGETED_HISTORY_GATE_V1",
        "status": "PASS",
        "timestamp_utc": stamp,
        "classification": classification,
        "next_action": next_action,
        "continuous_state_separation_certified": False,
        "continuous_action_certificate": False,
        "semantic_admissibility_scope": "exact q-age lattice under frozen countdown/reset model only",
        "geometry": {
            "evaluation_nodes": int(np.prod(data["eval_shape"])),
            "lookup_nodes": int(np.prod(data["lookup_shape"])),
            "base_refinement_axes": data["selected_axes"],
            "inserted_bar_a": [float(x) for x in TARGET_BAR_A],
        },
        "gates": {
            "r6_halo_provenance": data["halo_spec_provenance"],
            "frozen_transition_equivalence": eq_gate,
            "targeted_baseline_qdep_exactly_16": legacy_metrics["qdep_nodes"] == 16,
            "all_solvers_converged": bool(legacy["converged"] and causal.converged and hull.converged),
        },
        "metrics": {
            "legacy_all": legacy_metrics,
            "legacy_exact_qage": legacy_exact,
            "causal_all": causal_metrics,
            "causal_exact_qage": causal_exact,
            "hull_all": hull_metrics,
            "hull_exact_qage": hull_exact,
            "causal_changed_nodes": causal_changed,
            "hull_changed_nodes": hull_changed,
            "action_discovery": action_summary,
            "qage_lattice": age_diag,
        },
        "artifacts": {
            "qage_csv": str(qage_csv),
            "witness_csv": str(witness_csv),
            "action_discovery_csv": str(action_csv),
        },
    }
    text = json.dumps(output, indent=2, sort_keys=True)
    atomic_write(result_json, text)
    atomic_write(latest_json, text)

    manifest_files = [
        Path(__file__),
        result_json,
        qage_csv,
        witness_csv,
        action_csv,
        Path(data["halo_spec_resolved_path"]),
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

    print("=== R9-C DECISION ===", flush=True)
    print(
        f"R9C_QDEP legacy_all={legacy_metrics['qdep_nodes']} "
        f"legacy_exact_qage={legacy_exact['qdep_nodes']} "
        f"causal_all={causal_metrics['qdep_nodes']} "
        f"causal_exact_qage={causal_exact['qdep_nodes']} "
        f"hull_all={hull_metrics['qdep_nodes']} "
        f"hull_exact_qage={hull_exact['qdep_nodes']}", flush=True
    )
    print(f"R9C_EXECUTION=PASS", flush=True)
    print(f"R9C_CLASSIFICATION={classification}", flush=True)
    print("CONTINUOUS_STATE_SEPARATION_CERTIFIED=NO", flush=True)
    print(f"R9C_NEXT_ACTION={next_action}", flush=True)
    print(f"RESULT_JSON={result_json}", flush=True)
    print(f"MANIFEST={manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
