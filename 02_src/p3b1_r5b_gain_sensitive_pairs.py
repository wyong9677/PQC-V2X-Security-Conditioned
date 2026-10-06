from __future__ import annotations

import copy
import csv
import gc
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

import p3b1_r5_refinement_attribution as r5

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


def write_union_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: row.get(key, "") for key in fieldnames}
            )


def refined_cfg(cfg: dict, axes: tuple[str, ...]) -> dict:
    return r5.refine_cfg(cfg, axes)


def main():
    cfg = load_json(CFG_PATH)
    p1_cfg = load_json(P1_CFG_PATH)
    p2b_cfg = load_json(P2B_CFG_PATH)
    p2c_cfg = load_json(P2C_CFG_PATH)
    p3a_cfg = load_json(P3A_CFG_PATH)

    # Multi-objective pairs:
    #   v_f / v_p reduce global cell-corner penalty;
    #   bar_u recovered the largest single-axis strict-gain count in R5;
    #   age is included because it is both a security state and recovered
    #   strict gain on its own.
    pair_axes = [
        ("v_f", "v_p"),
        ("v_f", "bar_u"),
        ("v_p", "bar_u"),
        ("age", "bar_u"),
    ]

    rows = []

    print("=== P3-B1-R5B GAIN-SENSITIVE PAIR SCAN ===")

    for axes in pair_axes:
        local = refined_cfg(cfg, axes)

        for action in (-6.0, -3.0):
            row = r5.run_scenario(
                "PAIR_" + "_".join(axes) + f"_M{abs(int(action))}",
                local,
                action,
                p1_cfg,
                p2b_cfg,
                p2c_cfg,
                p3a_cfg,
            )

            row["refined_axes"] = ",".join(axes)
            row["candidate_action"] = action
            row["strict_gain_rate"] = (
                row["conservative_gain_count"]
                /
                row["node_count"]
            )
            row["optimistic_gain_rate"] = (
                row["optimistic_gain_count"]
                /
                row["node_count"]
            )

            rows.append(row)

            print(
                "PAIR="
                f"{','.join(axes)} "
                "ACTION="
                f"{action:.1f} "
                "N="
                f"{row['node_count']} "
                "INVALID_FRAC="
                f"{row['completion_invalid_fraction']:.6f} "
                "STRICT_GAIN="
                f"{row['conservative_gain_count']} "
                "STRICT_RATE="
                f"{row['strict_gain_rate']:.9g} "
                "OPT_GAIN="
                f"{row['optimistic_gain_count']} "
                "P95_PENALTY_M="
                f"{row['corner_penalty_p95_m']:.9g}"
            )


    # Selection is made for the current paper-facing candidate action -3.
    m3_rows = [
        row for row in rows
        if row["candidate_action"] == -3.0
    ]

    # Primary score: recover robust strict-gain geometry.
    # Secondary score: reduce global corner penalty.
    # Tertiary score: reduce domain invalid fraction.
    selected = max(
        m3_rows,
        key=lambda row: (
            row["strict_gain_rate"],
            -row["corner_penalty_p95_m"],
            -row["completion_invalid_fraction"],
        ),
    )

    print("=== P3-B1-R5B SELECTION ===")
    print(
        "SELECTED_AXES="
        + selected["refined_axes"]
    )
    print(
        "SELECTED_STRICT_GAIN="
        f"{selected['conservative_gain_count']}"
    )
    print(
        "SELECTED_STRICT_GAIN_RATE="
        f"{selected['strict_gain_rate']:.12g}"
    )
    print(
        "SELECTED_P95_PENALTY_M="
        f"{selected['corner_penalty_p95_m']:.12g}"
    )
    print(
        "SELECTED_INVALID_FRACTION="
        f"{selected['completion_invalid_fraction']:.12g}"
    )

    checks = {
        "PAIR_SCAN_COMPLETE":
            len(rows) == 8,
        "CURRENT_ACTION_PAIR_RESULTS_COMPLETE":
            len(m3_rows) == 4,
        "STRICT_GAIN_RECOVERED_FOR_CURRENT_ACTION":
            max(
                row["conservative_gain_count"]
                for row in m3_rows
            ) > 0,
        "SELECTED_PAIR_VALID":
            selected["node_count"] > 0,
    }

    status = "PASS" if all(checks.values()) else "FAIL"

    if selected["completion_invalid_fraction"] >= 0.05:
        next_action = (
            "BUILD_CONTINUATION_HALO_AROUND_SELECTED_REFINEMENT"
        )
    else:
        next_action = (
            "RUN_SELECTED_REFINED_FIXED_POINT"
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    csv_path = RESULTS_DIR / f"P3B1_R5B_PAIR_SCAN_{stamp}.csv"
    write_union_csv(csv_path, rows)

    output = {
        "schema": "SCV_P3B1_R5B_GAIN_SENSITIVE_PAIR_SCAN_V1",
        "status": status,
        "timestamp_utc": stamp,
        "classification": (
            "multi-objective refinement diagnostic; not a kernel result"
        ),
        "checks": checks,
        "pair_results": rows,
        "selected_axes": selected["refined_axes"].split(","),
        "selected_current_action_result": selected,
        "next_action": next_action,
        "scientific_kernel_claim_authorized": False,
    }

    result_path = RESULTS_DIR / f"P3B1_R5B_PAIR_DIAGNOSTIC_{stamp}.json"
    latest_path = RESULTS_DIR / "P3B1_R5B_LATEST.json"

    text = json.dumps(output, indent=2, sort_keys=True)
    atomic_write(result_path, text)
    atomic_write(latest_path, text)

    manifest_path = RESULTS_DIR / f"P3B1_R5B_MANIFEST_{stamp}.sha256"
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

    print(f"NEXT_ACTION={next_action}")
    print(f"P3B1_R5B_PAIR_DIAGNOSTIC={status}")
    print("SCIENTIFIC_KERNEL_CLAIM=NO")
    print(f"RESULT_JSON={result_path}")
    print(f"CSV={csv_path}")
    print(f"MANIFEST={manifest_path}")

    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
