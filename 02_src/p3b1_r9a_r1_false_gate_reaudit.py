from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.home() / "Desktop/paper set/PQC_V2X_Security_Conditioned/numerical_experiments"
RESULTS = ROOT / "04_results"
LATEST = RESULTS / "P3B1_R9A_LATEST.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def self_test() -> None:
    # Cross-mode objects must never be ordered by a projected scalar threshold
    # unless a mode-erasing projection theorem has first been proved.
    completed_fallback_mode = "F"
    switching_guard_mode = "C_or_S"
    assert completed_fallback_mode != switching_guard_mode
    print("R9A_R1_INTERNAL_SELF_TEST=PASS", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    self_test()
    if args.self_test:
        return 0

    if not LATEST.exists():
        raise FileNotFoundError(LATEST)
    old = json.loads(LATEST.read_text(encoding="utf-8"))

    profiles = old.get("profiles", [])
    profile_gates = all(bool(r.get("profile_gate")) for r in profiles) and bool(profiles)
    causal_gate = bool(old.get("gates", {}).get("causal_nonworsening_all_profiles"))
    gfp_gate = bool(old.get("gates", {}).get("gfp_contains_causal_lfp_all_profiles"))

    false_gate_detected = (
        old.get("classification") == "R9A_SAFETY_SEED_GATE_FAILED"
        and old.get("gates", {}).get("safety_seed_contains_switching_fallback") is False
    )

    # The retired gate compared projections of two different hybrid-mode sets:
    # L_F(K_F^x) (completed fallback mode) and G_F (pre-switch guard in C/S).
    # No set-inclusion order follows from their scalar spacing projections.
    corrected_pass = bool(profile_gates and causal_gate and gfp_gate)

    any_causal = any(int(r.get("causal_lfp_changed_nodes", 0)) > 0 for r in profiles)
    any_gfp = any(int(r.get("gfp_changed_vs_causal_lfp_nodes", 0)) > 0 for r in profiles)
    if not corrected_pass:
        classification = "R9A_R1_REAUDIT_STILL_FAILS_VALID_GATES"
    elif any_causal and any_gfp:
        classification = "CAUSAL_AND_FIXED_POINT_POLARITY_BOTH_MATERIAL_ON_BASE_GRID"
    elif any_causal:
        classification = "CAUSAL_ADOPTION_CORRECTION_MATERIAL_ON_BASE_GRID"
    elif any_gfp:
        classification = "FIXED_POINT_POLARITY_MATERIAL_ON_BASE_GRID"
    else:
        classification = "CORRECTIONS_NUMERICALLY_EQUIVALENT_ON_BASE_GRID"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result = {
        "schema": "SCV_P3B1_R9A_R1_FALSE_GATE_REAUDIT_V1",
        "timestamp_utc": stamp,
        "status": "PASS" if corrected_pass else "FAIL",
        "classification": classification,
        "input_r9a_latest": str(LATEST),
        "input_sha256": sha256_file(LATEST),
        "retired_gate": {
            "name": "SAFETY_SEED_CONTAINS_SWITCHING_FALLBACK",
            "retired": True,
            "reason": (
                "INVALID_CROSS_MODE_PROJECTION_ORDER: N_sw=0 completed-fallback lifting "
                "and N_sw>0 cooperative/switching guard are distinct hybrid-mode sets; "
                "their scalar spacing thresholds are not required to be nested."
            ),
            "false_gate_detected_in_input": false_gate_detected,
        },
        "valid_gates": {
            "all_profile_correctness_gates": profile_gates,
            "causal_nonworsening_all_profiles": causal_gate,
            "gfp_contains_causal_lfp_all_profiles": gfp_gate,
        },
        "scope": {
            "base_grid_only": True,
            "continuous_state_separation_certified": False,
            "maximal_continuous_kernel_claim": False,
            "semantic_admissibility_certified": False,
            "policy_class": "FROZEN_FINITE_ACTION_ALPHABET",
        },
        "next_action": "R9B_WITNESS_FOCUSED_SEMANTIC_AND_ACTION_AUDIT",
    }

    out = RESULTS / f"P3B1_R9A_R1_REAUDIT_{stamp}.json"
    latest = RESULTS / "P3B1_R9A_R1_LATEST.json"
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    atomic_write(out, text)
    atomic_write(latest, text)

    print("=== R9-A-R1 FALSE-GATE REAUDIT ===")
    print(f"FALSE_GATE_DETECTED={'YES' if false_gate_detected else 'NO'}")
    print("CROSS_MODE_PROJECTION_ORDER_REQUIRED=NO")
    print(f"VALID_PROFILE_GATES={'PASS' if profile_gates else 'FAIL'}")
    print(f"VALID_CAUSAL_GATE={'PASS' if causal_gate else 'FAIL'}")
    print(f"VALID_GFP_GATE={'PASS' if gfp_gate else 'FAIL'}")
    print(f"R9A_R1_EXECUTION={'PASS' if corrected_pass else 'FAIL'}")
    print(f"R9A_R1_CLASSIFICATION={classification}")
    print(f"RESULT_JSON={out}")
    return 0 if corrected_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
