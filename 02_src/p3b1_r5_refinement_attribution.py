from __future__ import annotations

import copy
import csv
import gc
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
import p3b1_r4_bellman_decomposition as r4

CFG_PATH = ROOT / "01_config" / "p3b1_augmented_fixed_point_v1.json"
P1_CFG_PATH = ROOT / "01_config" / "p1_validation_v2.json"
P2B_CFG_PATH = ROOT / "01_config" / "p2b_hybrid_fallback_v1.json"
P2C_CFG_PATH = ROOT / "01_config" / "p2c_switching_guard_v1.json"
P3A_CFG_PATH = ROOT / "01_config" / "p3a_information_contract_v1.json"

RESULTS_DIR = ROOT / "04_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

AXES = ("v_f", "v_p", "a_f", "bar_a", "bar_u", "age")


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


def write_csv(
    path: Path,
    rows: list[dict],
):
    if not rows:

        path.write_text(
            "",
            encoding="utf-8",
        )

        return

    fieldnames = []
    seen = set()

    for row in rows:

        for key in row.keys():

            if key not in seen:

                seen.add(key)
                fieldnames.append(key)

    with path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in rows:

            writer.writerow(
                {
                    key:
                        row.get(
                            key,
                            "",
                        )
                    for key
                    in fieldnames
                }
            )



def refined_values(values) -> list[float]:
    x = np.asarray(values, dtype=float)
    mids = 0.5 * (x[:-1] + x[1:])
    y = np.sort(np.concatenate([x, mids]))
    return [float(v) for v in y]


def refine_cfg(cfg: dict, axis_names: tuple[str, ...]) -> dict:
    local = copy.deepcopy(cfg)
    for axis in axis_names:
        local["grid"][axis] = refined_values(local["grid"][axis])
    return local


def top_boundary_violations(row: dict, limit: int = 12) -> list[dict]:
    candidates = []
    for key, value in row.items():
        if not (
            key.startswith("completion_")
            or key.startswith("defer_")
        ):
            continue
        if not (
            key.endswith("_below")
            or key.endswith("_above")
        ):
            continue
        count = int(value)
        if count > 0:
            candidates.append(
                {
                    "boundary": key,
                    "count": count,
                }
            )
    candidates.sort(
        key=lambda item: item["count"],
        reverse=True,
    )
    return candidates[:limit]


def run_scenario(
    label: str,
    cfg: dict,
    action: float,
    p1_cfg,
    p2b_cfg,
    p2c_cfg,
    p3a_cfg,
) -> dict:
    axes, flat, shape = r3.build_grid(cfg)

    row = r4.action_decomposition(
        float(action),
        cfg,
        p1_cfg,
        p2b_cfg,
        p2c_cfg,
        p3a_cfg,
        axes,
        flat,
        shape,
    )

    nodes = int(row["node_count"])

    result = {
        "scenario": label,
        "action": float(action),
        "node_count": nodes,
        "completion_invalid_count":
            int(row["completion_invalid_count"]),
        "completion_invalid_fraction":
            float(row["completion_invalid_fraction"]),
        "step_gain_count":
            int(row["step_only_gain_count"]),
        "step_gain_fraction":
            float(row["step_only_gain_count"] / nodes),
        "conservative_gain_count":
            int(row["conservative_corner_gain_count"]),
        "conservative_gain_fraction":
            float(row["conservative_corner_gain_count"] / nodes),
        "optimistic_gain_count":
            int(row["optimistic_corner_gain_count"]),
        "optimistic_gain_fraction":
            float(row["optimistic_corner_gain_count"] / nodes),
        "continuation_dominant_fraction":
            float(row["continuation_dominant_count"] / nodes),
        "corner_penalty_mean_m":
            float(row["future_corner_penalty_mean_m"]),
        "corner_penalty_p95_m":
            float(row["future_corner_penalty_p95_m"]),
        "corner_penalty_max_m":
            float(row["future_corner_penalty_max_m"]),
        "fallback_required_mean_m":
            float(row["fallback_required_mean_m"]),
        "step_loss_upper_mean_m":
            float(row["step_loss_upper_mean_m"]),
        "required_conservative_mean_m":
            float(row["required_conservative_mean_m"]),
        "required_optimistic_mean_m":
            float(row["required_optimistic_mean_m"]),
    }

    for key, value in row.items():
        if key.startswith("completion_") or key.startswith("defer_"):
            if key not in result:
                result[key] = value

    del row, axes, flat
    gc.collect()
    return result


def choose_top_axes(single_rows: list[dict], baseline_p95: float) -> list[str]:
    scored = []
    for row in single_rows:
        axis = row["scenario"].replace("REFINE_", "")
        reduction = baseline_p95 - row["corner_penalty_p95_m"]
        scored.append((reduction, axis))
    scored.sort(reverse=True)
    return [axis for _, axis in scored[:2]]


def main():
    cfg = load_json(CFG_PATH)
    p1_cfg = load_json(P1_CFG_PATH)
    p2b_cfg = load_json(P2B_CFG_PATH)
    p2c_cfg = load_json(P2C_CFG_PATH)
    p3a_cfg = load_json(P3A_CFG_PATH)

    axes0, flat0, _shape0 = r3.build_grid(cfg)
    stop = r3.endpoint_semantics_audit(
        cfg, p1_cfg, p2b_cfg, p3a_cfg, flat0
    )
    if not stop["pass"]:
        raise RuntimeError("P3B1_R5_STOP_SEMANTICS_GATE=FAIL")
    del axes0, flat0
    gc.collect()

    print("=== P3-B1-R5 ANISOTROPIC REFINEMENT ATTRIBUTION ===")
    print("STOP_SEMANTICS_GATE=PASS")

    # Use u=-6 only for attribution: R4 showed that even the strongest
    # diagnostic authority was suppressed by the same corner-max mechanism.
    baseline = run_scenario(
        "BASELINE",
        cfg,
        -6.0,
        p1_cfg,
        p2b_cfg,
        p2c_cfg,
        p3a_cfg,
    )

    print(
        "BASELINE "
        f"N={baseline['node_count']} "
        f"INVALID={baseline['completion_invalid_count']} "
        f"STEP_GAIN={baseline['step_gain_count']} "
        f"CORNER_MAX_GAIN={baseline['conservative_gain_count']} "
        f"CORNER_MIN_GAIN={baseline['optimistic_gain_count']} "
        f"P95_PENALTY_M={baseline['corner_penalty_p95_m']:.9g}"
    )

    boundary = top_boundary_violations(baseline)
    print("=== BASELINE DOMAIN-BOUNDARY ATTRIBUTION ===")
    if boundary:
        for item in boundary:
            print(
                f"{item['boundary'].upper()}="
                f"{item['count']}"
            )
    else:
        print("NO_BOUNDARY_VIOLATIONS=YES")

    single_rows = []

    for axis in AXES:
        local = refine_cfg(cfg, (axis,))
        row = run_scenario(
            f"REFINE_{axis}",
            local,
            -6.0,
            p1_cfg,
            p2b_cfg,
            p2c_cfg,
            p3a_cfg,
        )
        row["p95_penalty_reduction_m"] = (
            baseline["corner_penalty_p95_m"]
            -
            row["corner_penalty_p95_m"]
        )
        row["p95_penalty_reduction_fraction"] = (
            row["p95_penalty_reduction_m"]
            /
            baseline["corner_penalty_p95_m"]
        )
        single_rows.append(row)

        print(
            f"AXIS={axis} "
            f"N={row['node_count']} "
            f"INVALID_FRAC={row['completion_invalid_fraction']:.6f} "
            f"CORNER_MAX_GAIN={row['conservative_gain_count']} "
            f"CORNER_MIN_GAIN={row['optimistic_gain_count']} "
            f"P95_PENALTY_M={row['corner_penalty_p95_m']:.9g} "
            f"REDUCTION_FRAC={row['p95_penalty_reduction_fraction']:.6f}"
        )

    top_axes = choose_top_axes(
        single_rows,
        baseline["corner_penalty_p95_m"],
    )

    print(
        "TOP_REFINEMENT_AXES="
        + ",".join(top_axes)
    )

    pair_cfg = refine_cfg(
        cfg,
        tuple(top_axes),
    )

    pair_m6 = run_scenario(
        "PAIR_" + "_".join(top_axes) + "_M6",
        pair_cfg,
        -6.0,
        p1_cfg,
        p2b_cfg,
        p2c_cfg,
        p3a_cfg,
    )

    # Also test the paper-facing candidate action u=-3 on the selected pair.
    pair_m3 = run_scenario(
        "PAIR_" + "_".join(top_axes) + "_M3",
        pair_cfg,
        -3.0,
        p1_cfg,
        p2b_cfg,
        p2c_cfg,
        p3a_cfg,
    )

    for row in (pair_m6, pair_m3):
        row["p95_penalty_reduction_m"] = (
            baseline["corner_penalty_p95_m"]
            -
            row["corner_penalty_p95_m"]
        )
        row["p95_penalty_reduction_fraction"] = (
            row["p95_penalty_reduction_m"]
            /
            baseline["corner_penalty_p95_m"]
        )

        print(
            f"PAIR={row['scenario']} "
            f"N={row['node_count']} "
            f"INVALID_FRAC={row['completion_invalid_fraction']:.6f} "
            f"CORNER_MAX_GAIN={row['conservative_gain_count']} "
            f"CORNER_MIN_GAIN={row['optimistic_gain_count']} "
            f"P95_PENALTY_M={row['corner_penalty_p95_m']:.9g} "
            f"REDUCTION_FRAC={row['p95_penalty_reduction_fraction']:.6f}"
        )

    single_best_gain = max(
        row["conservative_gain_count"]
        for row in single_rows
    )

    pair_gain = max(
        pair_m6["conservative_gain_count"],
        pair_m3["conservative_gain_count"],
    )

    pair_reduction = max(
        pair_m6["p95_penalty_reduction_fraction"],
        pair_m3["p95_penalty_reduction_fraction"],
    )

    invalid_material = (
        baseline["completion_invalid_fraction"] >= 0.05
    )

    if pair_gain > 0:
        recommendation = (
            "ANISOTROPIC_PAIR_REFINEMENT_RECOVERS_STRICT_GAIN"
        )
    elif single_best_gain > 0:
        recommendation = (
            "SINGLE_AXIS_REFINEMENT_RECOVERS_STRICT_GAIN"
        )
    elif pair_reduction >= 0.50 and invalid_material:
        recommendation = (
            "REFINEMENT_WORKS_BUT_DOMAIN_HALO_ALSO_REQUIRED"
        )
    elif pair_reduction >= 0.50:
        recommendation = (
            "MULTI_AXIS_ADAPTIVE_REFINEMENT_REQUIRED"
        )
    elif invalid_material:
        recommendation = (
            "DOMAIN_HALO_AND_CELL_ENCLOSURE_REDESIGN_REQUIRED"
        )
    else:
        recommendation = (
            "CELL_ENCLOSURE_REDESIGN_REQUIRED"
        )

    checks = {
        "STOP_SEMANTICS_GATE":
            bool(stop["pass"]),
        "SIX_SINGLE_AXIS_SCANS_COMPLETE":
            len(single_rows) == 6,
        "PAIR_SCAN_COMPLETE":
            pair_m6["node_count"] > 0
            and pair_m3["node_count"] > 0,
        "BASELINE_OPTIMISTIC_GAIN_EXISTS":
            baseline["optimistic_gain_count"] > 0,
        "BASELINE_CONSERVATIVE_GAIN_SUPPRESSED":
            baseline["conservative_gain_count"] == 0,
    }

    status = "PASS" if all(checks.values()) else "FAIL"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    rows = [baseline] + single_rows + [pair_m6, pair_m3]
    csv_path = (
        RESULTS_DIR
        /
        f"P3B1_R5_REFINEMENT_ATTRIBUTION_{stamp}.csv"
    )
    write_csv(csv_path, rows)

    output = {
        "schema":
            "SCV_P3B1_R5_ANISOTROPIC_REFINEMENT_ATTRIBUTION_V1",
        "status":
            status,
        "timestamp_utc":
            stamp,
        "classification":
            "diagnostic refinement attribution; not a kernel result",
        "checks":
            checks,
        "stop_semantics":
            stop,
        "baseline":
            baseline,
        "baseline_boundary_violations":
            boundary,
        "single_axis_refinement":
            single_rows,
        "selected_pair_axes":
            top_axes,
        "pair_refinement_m6":
            pair_m6,
        "pair_refinement_m3":
            pair_m3,
        "recommendation":
            recommendation,
        "scientific_kernel_claim_authorized":
            False,
    }

    result_path = (
        RESULTS_DIR
        /
        f"P3B1_R5_REFINEMENT_DIAGNOSTIC_{stamp}.json"
    )
    latest_path = (
        RESULTS_DIR
        /
        "P3B1_R5_LATEST.json"
    )

    text = json.dumps(
        output,
        indent=2,
        sort_keys=True,
    )
    atomic_write(result_path, text)
    atomic_write(latest_path, text)

    manifest_path = (
        RESULTS_DIR
        /
        f"P3B1_R5_MANIFEST_{stamp}.sha256"
    )

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

    print("=== P3-B1-R5 DECISION ===")
    print(f"RECOMMENDATION={recommendation}")
    print(
        "SELECTED_PAIR="
        + ",".join(top_axes)
    )
    print(f"P3B1_R5_REFINEMENT_DIAGNOSTIC={status}")
    print("SCIENTIFIC_KERNEL_CLAIM=NO")
    print(f"RESULT_JSON={result_path}")
    print(f"CSV={csv_path}")
    print(f"MANIFEST={manifest_path}")

    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
