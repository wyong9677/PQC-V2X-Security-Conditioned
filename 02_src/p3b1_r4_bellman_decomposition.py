from __future__ import annotations

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

import p3b1_augmented_fixed_point_v1_r3 as r3
import p3b0_freshness_service_audit_v1 as b0

CFG_PATH = ROOT / "01_config" / "p3b1_augmented_fixed_point_v1.json"
P1_CFG_PATH = ROOT / "01_config" / "p1_validation_v2.json"
P2B_CFG_PATH = ROOT / "01_config" / "p2b_hybrid_fallback_v1.json"
P2C_CFG_PATH = ROOT / "01_config" / "p2c_switching_guard_v1.json"
P3A_CFG_PATH = ROOT / "01_config" / "p3a_information_contract_v1.json"

RESULTS_DIR = ROOT / "04_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def cell_corner_min(nodal: np.ndarray) -> np.ndarray:
    out = nodal
    for axis in range(6):
        left = [slice(None)] * out.ndim
        right = [slice(None)] * out.ndim
        left[axis] = slice(0, -1)
        right[axis] = slice(1, None)
        out = np.minimum(out[tuple(left)], out[tuple(right)])
    return out


def endpoint_values_for_action(
    cfg: dict,
    p1_cfg: dict,
    p2b_cfg: dict,
    p3a_cfg: dict,
    flat: list[np.ndarray],
    action: float,
):
    vf, vp, af, bar_a, bar_u, age = flat

    Ts = float(p1_cfg["plant"]["Ts"])
    J = float(cfg["information_contract"]["slew_rate"])
    nt = int(cfg["one_step"]["trajectory_points"])
    times = np.linspace(0.0, Ts, nt)

    ap_lower = b0.predecessor_acceleration_lower(
        age, bar_a, bar_u, J, p3a_cfg, p2b_cfg
    )

    Pp, Vp, Ap, Up, _ = r3.predecessor_motion_with_stop(
        vp, ap_lower, bar_u, age, times, J, p3a_cfg
    )

    Pf, Vf, Af = r3.follower_motion(
        vf, af, float(action), times, p1_cfg, p2b_cfg
    )

    completion = [
        Vf[:, -1],
        Vp[:, -1],
        Af[:, -1],
        Ap[:, -1],
        Up[:, -1],
        np.zeros(len(vf), dtype=float),
    ]

    defer = [
        Vf[:, -1],
        Vp[:, -1],
        Af[:, -1],
        bar_a,
        bar_u,
        age + Ts,
    ]

    return {
        "completion": completion,
        "defer": defer,
        "closing_end": Pf[:, -1] - Pp[:, -1],
    }


def axis_domain_violations(axes, values) -> dict:
    names = ("v_f", "v_p", "a_f", "bar_a", "bar_u", "age")
    result = {}
    for name, axis, x in zip(names, axes, values):
        x = np.asarray(x, dtype=float)
        result[f"{name}_below"] = int(np.count_nonzero(x < axis[0] - 1e-12))
        result[f"{name}_above"] = int(np.count_nonzero(x > axis[-1] + 1e-12))
    return result


def action_decomposition(
    action: float,
    cfg: dict,
    p1_cfg: dict,
    p2b_cfg: dict,
    p2c_cfg: dict,
    p3a_cfg: dict,
    axes,
    flat,
    shape,
):
    local_cfg = json.loads(json.dumps(cfg))
    local_cfg["cooperative_actions"] = [float(action)]

    td = r3.build_transition_data(
        local_cfg,
        p1_cfg,
        p2b_cfg,
        p2c_cfg,
        p3a_cfg,
        axes,
        flat,
    )

    trans = td["transitions"][0]
    fallback = td["fallback_required"].reshape(shape)

    h = fallback[..., None]
    cmax = r3.cell_corner_max(h)[..., 0].reshape(-1)
    cmin = cell_corner_min(h)[..., 0].reshape(-1)
    fallback_flat = fallback.reshape(-1)

    idx = trans["completion_index"]
    valid = trans["completion_valid"]

    future_max = np.full(fallback_flat.size, np.inf, dtype=float)
    future_min = np.full(fallback_flat.size, np.inf, dtype=float)
    future_max[valid] = cmax[idx[valid]]
    future_min[valid] = cmin[idx[valid]]

    cont_max = trans["closing_end"] + future_max
    cont_min = trans["closing_end"] + future_min

    required_max = np.maximum(trans["step_loss_upper"], cont_max)
    required_min = np.maximum(trans["step_loss_upper"], cont_min)

    tol = 1e-10

    gain_max = required_max < fallback_flat - tol
    gain_min = required_min < fallback_flat - tol
    step_only_gain = trans["step_loss_upper"] < fallback_flat - tol

    valid_penalty = future_max[valid] - future_min[valid]

    endpoint = endpoint_values_for_action(
        local_cfg,
        p1_cfg,
        p2b_cfg,
        p3a_cfg,
        flat,
        action,
    )

    completion_violations = axis_domain_violations(
        axes, endpoint["completion"]
    )
    defer_violations = axis_domain_violations(
        axes, endpoint["defer"]
    )

    return {
        "action": float(action),
        "node_count": int(fallback_flat.size),
        "completion_invalid_count": int(np.count_nonzero(~valid)),
        "completion_invalid_fraction": float(np.mean(~valid)),
        "step_only_gain_count": int(np.count_nonzero(step_only_gain)),
        "conservative_corner_gain_count": int(np.count_nonzero(gain_max)),
        "optimistic_corner_gain_count": int(np.count_nonzero(gain_min)),
        "continuation_dominant_count": int(
            np.count_nonzero(cont_max >= trans["step_loss_upper"])
        ),
        "future_corner_penalty_mean_m": (
            float(np.mean(valid_penalty)) if valid_penalty.size else math.inf
        ),
        "future_corner_penalty_p95_m": (
            float(np.quantile(valid_penalty, 0.95))
            if valid_penalty.size else math.inf
        ),
        "future_corner_penalty_max_m": (
            float(np.max(valid_penalty)) if valid_penalty.size else math.inf
        ),
        "fallback_required_mean_m": float(np.mean(fallback_flat)),
        "step_loss_upper_mean_m": float(np.mean(trans["step_loss_upper"])),
        "required_conservative_mean_m": float(
            np.mean(required_max[np.isfinite(required_max)])
        ) if np.any(np.isfinite(required_max)) else math.inf,
        "required_optimistic_mean_m": float(
            np.mean(required_min[np.isfinite(required_min)])
        ) if np.any(np.isfinite(required_min)) else math.inf,
        **{
            f"completion_{k}": v
            for k, v in completion_violations.items()
        },
        **{
            f"defer_{k}": v
            for k, v in defer_violations.items()
        },
    }


def classify_root_cause(rows: list[dict]) -> str:
    best_conservative = max(
        row["conservative_corner_gain_count"] for row in rows
    )
    best_optimistic = max(
        row["optimistic_corner_gain_count"] for row in rows
    )
    best_step = max(
        row["step_only_gain_count"] for row in rows
    )
    invalid = max(
        row["completion_invalid_fraction"] for row in rows
    )

    if best_conservative > 0:
        return "STRICT_GAIN_EXISTS_BEFORE_FIXED_POINT"
    if best_optimistic > 0:
        return "CELL_CORNER_CONSERVATISM_SUPPRESSES_GAIN"
    if best_step > 0 and invalid > 0.01:
        return "CONTINUATION_OR_DOMAIN_TRUNCATION_SUPPRESSES_GAIN"
    if best_step > 0:
        return "CONTINUATION_REQUIREMENT_SUPPRESSES_GAIN"
    return "ONE_STEP_ACTION_SET_DOES_NOT_BEAT_FALLBACK_REQUIREMENT"


def main():
    cfg = load_json(CFG_PATH)
    p1_cfg = load_json(P1_CFG_PATH)
    p2b_cfg = load_json(P2B_CFG_PATH)
    p2c_cfg = load_json(P2C_CFG_PATH)
    p3a_cfg = load_json(P3A_CFG_PATH)

    axes, flat, shape = r3.build_grid(cfg)

    stop = r3.endpoint_semantics_audit(
        cfg,
        p1_cfg,
        p2b_cfg,
        p3a_cfg,
        flat,
    )

    if not stop["pass"]:
        raise RuntimeError("P3B1_R4_STOP_SEMANTICS_GATE=FAIL")

    print("=== P3-B1-R4 BELLMAN DECOMPOSITION ===")
    print("STOP_SEMANTICS_GATE=PASS")

    actions = [-6.0, -5.0, -4.0, -3.0, -2.0, -1.0]

    rows = []
    for action in actions:
        row = action_decomposition(
            action,
            cfg,
            p1_cfg,
            p2b_cfg,
            p2c_cfg,
            p3a_cfg,
            axes,
            flat,
            shape,
        )
        rows.append(row)

        print(
            "ACTION="
            f"{action:.1f} "
            "INVALID="
            f"{row['completion_invalid_count']} "
            "STEP_GAIN="
            f"{row['step_only_gain_count']} "
            "CORNER_MAX_GAIN="
            f"{row['conservative_corner_gain_count']} "
            "CORNER_MIN_GAIN="
            f"{row['optimistic_corner_gain_count']} "
            "P95_CORNER_PENALTY_M="
            f"{row['future_corner_penalty_p95_m']:.9g}"
        )

    cause = classify_root_cause(rows)

    checks = {
        "STOP_SEMANTICS_GATE": bool(stop["pass"]),
        "AUTHORITY_DECOMPOSITION_COMPLETE": len(rows) == 6,
        "FINITE_VALID_DIAGNOSTICS": all(
            math.isfinite(row["fallback_required_mean_m"])
            and math.isfinite(row["step_loss_upper_mean_m"])
            and math.isfinite(row["future_corner_penalty_p95_m"])
            for row in rows
        ),
    }

    status = "PASS" if all(checks.values()) else "FAIL"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    csv_path = RESULTS_DIR / f"P3B1_R4_BELLMAN_DECOMPOSITION_{stamp}.csv"
    write_csv(csv_path, rows)

    output = {
        "schema": "SCV_P3B1_R4_BELLMAN_DECOMPOSITION_V1",
        "status": status,
        "timestamp_utc": stamp,
        "classification": (
            "Bellman root-cause diagnostic only; not a viability-kernel result"
        ),
        "checks": checks,
        "stop_semantics": stop,
        "root_cause_classification": cause,
        "actions": rows,
        "scientific_claim_authorized": False,
    }

    result_path = RESULTS_DIR / f"P3B1_R4_BELLMAN_DIAGNOSTIC_{stamp}.json"
    latest_path = RESULTS_DIR / "P3B1_R4_LATEST.json"

    text = json.dumps(output, indent=2, sort_keys=True)
    atomic_write(result_path, text)
    atomic_write(latest_path, text)

    manifest_path = RESULTS_DIR / f"P3B1_R4_MANIFEST_{stamp}.sha256"
    manifest_files = [
        CFG_PATH,
        P1_CFG_PATH,
        P2B_CFG_PATH,
        P2C_CFG_PATH,
        P3A_CFG_PATH,
        Path(__file__),
        result_path,
        csv_path,
    ]
    manifest = "\n".join(
        f"{sha256_file(path)}  {path}"
        for path in manifest_files
    ) + "\n"
    atomic_write(manifest_path, manifest)

    print("=== ROOT CAUSE ===")
    print(f"ROOT_CAUSE_CLASSIFICATION={cause}")
    print(f"P3B1_R4_BELLMAN_DIAGNOSTIC={status}")
    print("SCIENTIFIC_KERNEL_CLAIM=NO")
    print(f"RESULT_JSON={result_path}")
    print(f"CSV={csv_path}")
    print(f"MANIFEST={manifest_path}")

    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
