from __future__ import annotations

import copy
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
sys.path.insert(0, str(SRC))

import p2c_switching_guard_v1 as sw
import p3b0_freshness_service_audit_v1 as b0
import p3b1_augmented_fixed_point_v1_r3 as r3
import p3b1_r4_bellman_decomposition as r4
import p3b1_r5_refinement_attribution as r5

CFG_PATH = ROOT / "01_config" / "p3b1_augmented_fixed_point_v1.json"
P1_CFG_PATH = ROOT / "01_config" / "p1_validation_v2.json"
P2B_CFG_PATH = ROOT / "01_config" / "p2b_hybrid_fallback_v1.json"
P2C_CFG_PATH = ROOT / "01_config" / "p2c_switching_guard_v1.json"
P3A_CFG_PATH = ROOT / "01_config" / "p3a_information_contract_v1.json"
R5B_PATH = ROOT / "04_results" / "P3B1_R5B_LATEST.json"

RESULTS_DIR = ROOT / "04_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

AXIS_NAMES = ("v_f", "v_p", "a_f", "bar_a", "bar_u", "age")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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

    fieldnames = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def make_eval_cfg(cfg: dict, selected_axes: list[str]) -> dict:
    return r5.refine_cfg(cfg, tuple(selected_axes))


def successor_values_for_action(
    cfg: dict,
    p1_cfg: dict,
    p2b_cfg: dict,
    p3a_cfg: dict,
    flat,
    action: float,
) -> dict:
    endpoint = r4.endpoint_values_for_action(
        cfg,
        p1_cfg,
        p2b_cfg,
        p3a_cfg,
        flat,
        float(action),
    )
    return {
        "completion": [
            np.asarray(x, dtype=float)
            for x in endpoint["completion"]
        ],
        "defer": [
            np.asarray(x, dtype=float)
            for x in endpoint["defer"]
        ],
        "closing_end":
            np.asarray(endpoint["closing_end"], dtype=float),
    }


def nominal_spacing(axis: np.ndarray) -> float:
    d = np.diff(axis)
    return float(np.median(d))


def extend_axis_to_cover(
    axis: np.ndarray,
    observed_min: float,
    observed_max: float,
) -> np.ndarray:
    """
    Extend only the lookup axis.  Evaluation-domain nodes remain untouched.

    One nominal-cell outward padding is added beyond the observed successor
    extrema, so exact-boundary successor values do not sit on the artificial
    outer edge of the lookup mesh.
    """
    axis = np.asarray(axis, dtype=float)
    step = nominal_spacing(axis)

    lo = float(axis[0])
    hi = float(axis[-1])

    extra_lo = []
    extra_hi = []

    if observed_min < lo - 1e-12:
        target = observed_min - step
        x = lo - step
        while x > target:
            extra_lo.append(x)
            x -= step
        extra_lo.append(x)

    if observed_max > hi + 1e-12:
        target = observed_max + step
        x = hi + step
        while x < target:
            extra_hi.append(x)
            x += step
        extra_hi.append(x)

    merged = np.concatenate(
        [
            np.asarray(extra_lo[::-1], dtype=float),
            axis,
            np.asarray(extra_hi, dtype=float),
        ]
    )

    return np.unique(merged)


def build_halo_axes(eval_axes, successor_sets: list[list[np.ndarray]]):
    mins = [
        min(float(np.min(values[j])) for values in successor_sets)
        for j in range(6)
    ]
    maxs = [
        max(float(np.max(values[j])) for values in successor_sets)
        for j in range(6)
    ]

    halo_axes = [
        extend_axis_to_cover(
            axis,
            mins[j],
            maxs[j],
        )
        for j, axis in enumerate(eval_axes)
    ]

    return halo_axes, mins, maxs


def mesh_flat(axes):
    mesh = np.meshgrid(*axes, indexing="ij")
    flat = [x.reshape(-1) for x in mesh]
    shape = tuple(len(axis) for axis in axes)
    return flat, shape


def fallback_required_on_grid(
    cfg: dict,
    p1_cfg: dict,
    p2b_cfg: dict,
    p2c_cfg: dict,
    p3a_cfg: dict,
    flat,
) -> np.ndarray:
    vf, vp, af, bar_a, bar_u, age = flat

    J = float(cfg["information_contract"]["slew_rate"])

    ap_lower = b0.predecessor_acceleration_lower(
        age,
        bar_a,
        bar_u,
        J,
        p3a_cfg,
        p2b_cfg,
    )

    _, fallback_upper, _ = sw.switching_loss_bracket(
        vf,
        af,
        vp,
        ap_lower,
        int(p2c_cfg["switching"]["N_sw"]),
        [257],
        p2c_cfg,
        p2b_cfg,
    )

    return np.maximum(
        np.asarray(fallback_upper, dtype=float),
        0.0,
    )


def cell_corner_min(nodal: np.ndarray) -> np.ndarray:
    out = nodal
    for axis in range(6):
        left = [slice(None)] * out.ndim
        right = [slice(None)] * out.ndim
        left[axis] = slice(0, -1)
        right[axis] = slice(1, None)
        out = np.minimum(out[tuple(left)], out[tuple(right)])
    return out


def axis_extension_summary(eval_axes, halo_axes):
    rows = []
    for name, e, h in zip(AXIS_NAMES, eval_axes, halo_axes):
        rows.append(
            {
                "axis": name,
                "eval_min": float(e[0]),
                "eval_max": float(e[-1]),
                "eval_nodes": int(len(e)),
                "lookup_min": float(h[0]),
                "lookup_max": float(h[-1]),
                "lookup_nodes": int(len(h)),
                "lower_extension": float(e[0] - h[0]),
                "upper_extension": float(h[-1] - e[-1]),
            }
        )
    return rows


def halo_decomposition_for_action(
    action: float,
    eval_cfg: dict,
    p1_cfg: dict,
    p2b_cfg: dict,
    p2c_cfg: dict,
    p3a_cfg: dict,
    eval_axes,
    eval_flat,
    eval_shape,
    halo_axes,
    halo_fallback,
    halo_shape,
):
    # Step loss / closing end on the evaluation nodes.
    local_cfg = copy.deepcopy(eval_cfg)
    local_cfg["cooperative_actions"] = [float(action)]

    td_eval = r3.build_transition_data(
        local_cfg,
        p1_cfg,
        p2b_cfg,
        p2c_cfg,
        p3a_cfg,
        eval_axes,
        eval_flat,
    )

    trans = td_eval["transitions"][0]
    eval_fallback = np.asarray(
        td_eval["fallback_required"],
        dtype=float,
    )

    succ = successor_values_for_action(
        eval_cfg,
        p1_cfg,
        p2b_cfg,
        p3a_cfg,
        eval_flat,
        float(action),
    )

    completion_idx, completion_valid = r3.locate_cells(
        halo_axes,
        succ["completion"],
    )

    defer_idx, defer_valid = r3.locate_cells(
        halo_axes,
        succ["defer"],
    )

    # R=1 ideal completion is sufficient to test whether the halo removes
    # artificial boundary truncation from the first Bellman improvement.
    h_lookup = halo_fallback.reshape(halo_shape)[..., None]

    cmax = r3.cell_corner_max(h_lookup)[..., 0].reshape(-1)
    cmin = cell_corner_min(h_lookup)[..., 0].reshape(-1)

    future_max = np.full(eval_fallback.size, np.inf, dtype=float)
    future_min = np.full(eval_fallback.size, np.inf, dtype=float)

    future_max[completion_valid] = cmax[
        completion_idx[completion_valid]
    ]
    future_min[completion_valid] = cmin[
        completion_idx[completion_valid]
    ]

    cont_max = trans["closing_end"] + future_max
    cont_min = trans["closing_end"] + future_min

    required_max = np.maximum(
        trans["step_loss_upper"],
        cont_max,
    )

    required_min = np.maximum(
        trans["step_loss_upper"],
        cont_min,
    )

    tol = 1e-10

    strict_max = required_max < eval_fallback - tol
    strict_min = required_min < eval_fallback - tol

    finite = completion_valid & np.isfinite(future_max)
    penalty = future_max[finite] - future_min[finite]

    return {
        "action": float(action),
        "eval_nodes": int(eval_fallback.size),
        "completion_invalid_count":
            int(np.count_nonzero(~completion_valid)),
        "defer_invalid_count":
            int(np.count_nonzero(~defer_valid)),
        "completion_invalid_fraction":
            float(np.mean(~completion_valid)),
        "defer_invalid_fraction":
            float(np.mean(~defer_valid)),
        "strict_gain_count":
            int(np.count_nonzero(strict_max)),
        "strict_gain_fraction":
            float(np.mean(strict_max)),
        "optimistic_gain_count":
            int(np.count_nonzero(strict_min)),
        "optimistic_gain_fraction":
            float(np.mean(strict_min)),
        "corner_penalty_mean_m":
            float(np.mean(penalty)),
        "corner_penalty_p95_m":
            float(np.quantile(penalty, 0.95)),
        "corner_penalty_max_m":
            float(np.max(penalty)),
        "fallback_required_mean_m":
            float(np.mean(eval_fallback)),
        "required_conservative_mean_m":
            float(
                np.mean(required_max[np.isfinite(required_max)])
            ),
    }


def main():
    cfg = load_json(CFG_PATH)
    p1_cfg = load_json(P1_CFG_PATH)
    p2b_cfg = load_json(P2B_CFG_PATH)
    p2c_cfg = load_json(P2C_CFG_PATH)
    p3a_cfg = load_json(P3A_CFG_PATH)
    r5b = load_json(R5B_PATH)

    selected_axes = list(r5b["selected_axes"])
    eval_cfg = make_eval_cfg(cfg, selected_axes)

    eval_axes, eval_flat, eval_shape = r3.build_grid(eval_cfg)

    stop = r3.endpoint_semantics_audit(
        eval_cfg,
        p1_cfg,
        p2b_cfg,
        p3a_cfg,
        eval_flat,
    )

    if not stop["pass"]:
        raise RuntimeError("P3B1_R6_STOP_SEMANTICS_GATE=FAIL")

    actions = [
        float(x)
        for x in cfg["cooperative_actions"]
    ]

    successor_sets = []

    for action in actions:
        succ = successor_values_for_action(
            eval_cfg,
            p1_cfg,
            p2b_cfg,
            p3a_cfg,
            eval_flat,
            action,
        )
        successor_sets.append(succ["completion"])
        successor_sets.append(succ["defer"])

    halo_axes, succ_min, succ_max = build_halo_axes(
        eval_axes,
        successor_sets,
    )

    halo_flat, halo_shape = mesh_flat(halo_axes)

    halo_fallback = fallback_required_on_grid(
        eval_cfg,
        p1_cfg,
        p2b_cfg,
        p2c_cfg,
        p3a_cfg,
        halo_flat,
    )

    print("=== P3-B1-R6 CONTINUATION HALO AUDIT ===")
    print("STOP_SEMANTICS_GATE=PASS")
    print(
        "SELECTED_REFINEMENT_AXES="
        + ",".join(selected_axes)
    )
    print(
        "EVAL_GRID_NODES="
        f"{int(np.prod(eval_shape))}"
    )
    print(
        "LOOKUP_GRID_NODES="
        f"{int(np.prod(halo_shape))}"
    )

    extension_rows = axis_extension_summary(
        eval_axes,
        halo_axes,
    )

    print("=== HALO AXES ===")
    for row in extension_rows:
        print(
            "AXIS="
            f"{row['axis']} "
            "EVAL=["
            f"{row['eval_min']:.9g},"
            f"{row['eval_max']:.9g}] "
            "LOOKUP=["
            f"{row['lookup_min']:.9g},"
            f"{row['lookup_max']:.9g}] "
            "N="
            f"{row['eval_nodes']}->"
            f"{row['lookup_nodes']}"
        )

    action_rows = []

    for action in actions:
        row = halo_decomposition_for_action(
            action,
            eval_cfg,
            p1_cfg,
            p2b_cfg,
            p2c_cfg,
            p3a_cfg,
            eval_axes,
            eval_flat,
            eval_shape,
            halo_axes,
            halo_fallback,
            halo_shape,
        )

        action_rows.append(row)

        print(
            "ACTION="
            f"{action:.1f} "
            "COMPLETION_INVALID="
            f"{row['completion_invalid_count']} "
            "DEFER_INVALID="
            f"{row['defer_invalid_count']} "
            "STRICT_GAIN="
            f"{row['strict_gain_count']} "
            "STRICT_RATE="
            f"{row['strict_gain_fraction']:.12g} "
            "P95_PENALTY_M="
            f"{row['corner_penalty_p95_m']:.9g}"
        )

    total_invalid = sum(
        row["completion_invalid_count"]
        +
        row["defer_invalid_count"]
        for row in action_rows
    )

    current_action = min(
        actions,
        key=lambda x: abs(x - (-3.0)),
    )

    current_row = next(
        row
        for row in action_rows
        if row["action"] == current_action
    )

    checks = {
        "STOP_SEMANTICS_GATE":
            bool(stop["pass"]),
        "EVALUATION_DOMAIN_UNCHANGED":
            all(
                np.array_equal(
                    np.asarray(eval_cfg["grid"][name], dtype=float),
                    np.asarray(eval_axes[j], dtype=float),
                )
                for j, name in enumerate(AXIS_NAMES)
            ),
        "HALO_COVERS_ALL_SUCCESSORS":
            total_invalid == 0,
        "CURRENT_ACTION_STRICT_GAIN_SURVIVES_HALO":
            current_row["strict_gain_count"] > 0,
        "LOOKUP_GRID_FINITE":
            bool(
                np.isfinite(
                    halo_fallback
                ).all()
            ),
    }

    status = "PASS" if all(checks.values()) else "FAIL"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    axis_csv = RESULTS_DIR / f"P3B1_R6_HALO_AXES_{stamp}.csv"
    action_csv = RESULTS_DIR / f"P3B1_R6_HALO_ACTIONS_{stamp}.csv"

    write_csv(axis_csv, extension_rows)
    write_csv(action_csv, action_rows)

    halo_spec = {
        name: [float(x) for x in axis]
        for name, axis in zip(AXIS_NAMES, halo_axes)
    }

    halo_spec_path = RESULTS_DIR / f"P3B1_R6_HALO_SPEC_{stamp}.json"
    atomic_write(
        halo_spec_path,
        json.dumps(
            {
                "selected_refinement_axes": selected_axes,
                "evaluation_grid": eval_cfg["grid"],
                "lookup_halo_grid": halo_spec,
                "successor_min": {
                    name: float(succ_min[j])
                    for j, name in enumerate(AXIS_NAMES)
                },
                "successor_max": {
                    name: float(succ_max[j])
                    for j, name in enumerate(AXIS_NAMES)
                },
            },
            indent=2,
            sort_keys=True,
        ),
    )

    output = {
        "schema":
            "SCV_P3B1_R6_CONTINUATION_HALO_AUDIT_V1",
        "status":
            status,
        "timestamp_utc":
            stamp,
        "classification":
            (
                "continuation lookup-halo diagnostic; "
                "evaluation domain unchanged; not a kernel result"
            ),
        "checks":
            checks,
        "selected_refinement_axes":
            selected_axes,
        "evaluation_grid_nodes":
            int(np.prod(eval_shape)),
        "lookup_grid_nodes":
            int(np.prod(halo_shape)),
        "axis_extensions":
            extension_rows,
        "action_results":
            action_rows,
        "current_action":
            current_action,
        "current_action_result":
            current_row,
        "total_successor_invalid_count":
            int(total_invalid),
        "halo_spec":
            str(halo_spec_path),
        "scientific_kernel_claim_authorized":
            False,
        "next_action":
            (
                "RUN_R7_REFINED_FIXED_POINT_WITH_FROZEN_HALO"
                if status == "PASS"
                else
                "REPAIR_HALO_COVERAGE_BEFORE_FIXED_POINT"
            ),
    }

    result_path = RESULTS_DIR / f"P3B1_R6_HALO_AUDIT_{stamp}.json"
    latest_path = RESULTS_DIR / "P3B1_R6_LATEST.json"

    text = json.dumps(output, indent=2, sort_keys=True)
    atomic_write(result_path, text)
    atomic_write(latest_path, text)

    manifest_path = RESULTS_DIR / f"P3B1_R6_MANIFEST_{stamp}.sha256"
    manifest_files = [
        CFG_PATH,
        P1_CFG_PATH,
        P2B_CFG_PATH,
        P2C_CFG_PATH,
        P3A_CFG_PATH,
        R5B_PATH,
        Path(__file__),
        result_path,
        halo_spec_path,
        axis_csv,
        action_csv,
    ]

    manifest = "\n".join(
        f"{sha256_file(path)}  {path}"
        for path in manifest_files
    ) + "\n"

    atomic_write(manifest_path, manifest)

    print("=== P3-B1-R6 DECISION ===")
    print(f"HALO_TOTAL_INVALID={total_invalid}")
    print(
        "CURRENT_ACTION_STRICT_GAIN="
        f"{current_row['strict_gain_count']}"
    )
    print(
        "CURRENT_ACTION_P95_PENALTY_M="
        f"{current_row['corner_penalty_p95_m']:.12g}"
    )
    print(f"P3B1_R6_HALO_AUDIT={status}")
    print("SCIENTIFIC_KERNEL_CLAIM=NO")
    print(f"NEXT_ACTION={output['next_action']}")
    print(f"RESULT_JSON={result_path}")
    print(f"HALO_SPEC={halo_spec_path}")
    print(f"MANIFEST={manifest_path}")

    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
