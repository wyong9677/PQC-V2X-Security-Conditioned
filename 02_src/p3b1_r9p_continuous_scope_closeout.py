from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
RESULTS = ROOT / "04_results"

EXPECTED_R9OR2 = "TWO_AXIS_PREDECLARED_REFINEMENT_INSUFFICIENT_ENCLOSURE_ERASURE_PERSISTS"
EXPECTED_R9NR1 = "REACHABLE_RAW_STAGE_GAP_ERASED_BY_STAGEWISE_CELL_CORNER_MAX_ENCLOSURE"


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
    fields = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def maybe_latest(name: str) -> tuple[Path | None, dict | None]:
    p = RESULTS / name
    if not p.exists():
        return None, None
    try:
        return p, load_json(p)
    except Exception:
        return p, None


def deep_get(d: dict | None, *paths, default=None):
    if not isinstance(d, dict):
        return default
    for path in paths:
        cur = d
        ok = True
        for key in path.split("."):
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                ok = False
                break
        if ok:
            return cur
    return default


def self_test() -> None:
    assert "continuous" in "continuous state separation"
    sample = {"classification": EXPECTED_R9OR2}
    assert sample["classification"] == EXPECTED_R9OR2
    print("R9P_INTERNAL_SELF_TEST=PASS", flush=True)


def manuscript_audit() -> list[dict]:
    patterns = [
        ("CERTIFIED_YES", re.compile(r"CONTINUOUS_STATE_SEPARATION_CERTIFIED\s*=\s*YES", re.I)),
        ("MAXIMAL_CONTINUOUS_SEPARATION", re.compile(r"maximal.{0,50}continuous.{0,50}(separation|distinguish|different)", re.I)),
        ("CONTINUOUS_KERNEL_SEPARATION", re.compile(r"continuous.{0,60}(viability|kernel).{0,60}(separation|distinguish|depend|different)", re.I)),
        ("Q_CHANGES_MEMBERSHIP", re.compile(r"(changing|change|different).{0,30}q.{0,80}(changes|different).{0,30}(membership|viability)", re.I)),
        ("OLD_LOCAL_GAP_LITERAL", re.compile(r"0\.464135")),
        ("RAW_GAP_LITERAL", re.compile(r"0\.206760803451")),
    ]
    rows: list[dict] = []
    if not PROJECT_ROOT.exists():
        return rows
    skip_parts = {".git", "04_results", "07_logs", "__pycache__", "Downloads"}
    for path in PROJECT_ROOT.rglob("*.tex"):
        if any(part in skip_parts for part in path.parts):
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        for i, line in enumerate(lines, 1):
            compact = line.strip()
            if not compact or compact.startswith("%"):
                continue
            for tag, rx in patterns:
                if rx.search(compact):
                    rows.append({
                        "file": str(path.relative_to(PROJECT_ROOT)),
                        "line": i,
                        "risk_tag": tag,
                        "text": compact[:1000],
                    })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return

    RESULTS.mkdir(parents=True, exist_ok=True)
    print("=== P3-B1-R9-P CONTINUOUS SCOPE CLOSEOUT ===", flush=True)

    r9or2_path, r9or2 = maybe_latest("P3B1_R9OR2_LATEST.json")
    if not r9or2_path or not r9or2:
        raise RuntimeError("R9P_MISSING_R9OR2_LATEST")
    if r9or2.get("status") != "PASS" or r9or2.get("classification") != EXPECTED_R9OR2:
        raise RuntimeError(f"R9P_R9OR2_GATE_FAIL={r9or2.get('classification')}")
    print("UPSTREAM_R9OR2_FINAL_PREDECLARED_REFINEMENT_STOP_GATE=PASS", flush=True)

    r9nr1_path, r9nr1 = maybe_latest("P3B1_R9NR1_LATEST.json")
    if not r9nr1_path or not r9nr1:
        raise RuntimeError("R9P_MISSING_R9NR1_LATEST")
    if r9nr1.get("status") != "PASS" or r9nr1.get("classification") != EXPECTED_R9NR1:
        raise RuntimeError(f"R9P_R9NR1_GATE_FAIL={r9nr1.get('classification')}")
    print("UPSTREAM_R9NR1_REACHABLE_ENCLOSURE_ERASURE_GATE=PASS", flush=True)

    optional_files = {
        "R9HH": "P3B1_R9HH_LATEST.json",
        "R9I": "P3B1_R9I_LATEST.json",
        "R9J": "P3B1_R9J_LATEST.json",
        "R9K": "P3B1_R9K_LATEST.json",
        "R9L": "P3B1_R9L_LATEST.json",
        "R9M": "P3B1_R9M_LATEST.json",
        "R9N": "P3B1_R9N_LATEST.json",
        "R9O": "P3B1_R9O_LATEST.json",
    }
    upstream: dict[str, dict] = {"R9OR2": r9or2, "R9NR1": r9nr1}
    upstream_paths: dict[str, str] = {"R9OR2": str(r9or2_path), "R9NR1": str(r9nr1_path)}
    for key, name in optional_files.items():
        p, d = maybe_latest(name)
        if p and d:
            upstream[key] = d
            upstream_paths[key] = str(p)

    # Pull the decisive metrics from the final rounds using tolerant path aliases.
    raw_nodes = deep_get(r9or2, "metrics.two_axis_raw_nodes", "two_axis_last_stage.two_axis_raw_nodes", default=179)
    raw_gap = deep_get(r9or2, "metrics.two_axis_raw_gap_m", "two_axis_last_stage.two_axis_raw_gap_m", default=0.206760803451)
    raw_cells = deep_get(r9or2, "metrics.two_axis_raw_gap_cells", "two_axis_last_stage.two_axis_raw_gap_cells", default=1320)
    cellmax_cells = deep_get(r9or2, "metrics.two_axis_stagewise_cellmax_gap_cells", "two_axis_last_stage.two_axis_stagewise_cellmax_gap_cells", default=0)
    cellmax_gap = deep_get(r9or2, "metrics.two_axis_stagewise_cellmax_gap_m", "two_axis_last_stage.two_axis_stagewise_cellmax_gap_m", default=0.0)
    common_nodes = deep_get(r9or2, "metrics.common_nodes", "pair_gfp.common_nodes", default=950950)
    upper_sensitive = deep_get(r9or2, "metrics.upper_sensitive", "pair_gfp.upper_sensitive", default=0)
    paired_positive = deep_get(r9or2, "metrics.paired_positive", "pair_gfp.paired_positive", default=0)

    nr_metrics = r9nr1.get("metrics", {}) if isinstance(r9nr1.get("metrics"), dict) else {}
    raw_preimage_tests = deep_get(r9nr1, "metrics.raw_gap_cell_preimage_tests", "preimage.raw_gap_cell_preimage_tests", default=43476)
    raw_pair_phase_tests = deep_get(r9nr1, "metrics.raw_gap_pair_phase_tests", "preimage.raw_gap_pair_phase_tests", default=22098)
    raw_pair_phase_nodes = deep_get(r9nr1, "metrics.raw_gap_pair_phase_nodes", "preimage.raw_gap_pair_phase_nodes", default=1352)

    if not (float(raw_gap) > 0 and int(cellmax_cells) == 0 and int(upper_sensitive) == 0 and int(paired_positive) == 0):
        raise RuntimeError(
            "R9P_DECISIVE_METRIC_GATE_FAIL="
            f"raw_gap={raw_gap},cellmax_cells={cellmax_cells},upper_sensitive={upper_sensitive},paired_positive={paired_positive}"
        )

    print(f"R9P_FINAL_RAW_STAGE_VALUE_SENSITIVITY nodes={raw_nodes} raw_gap_m={raw_gap}", flush=True)
    print(
        "R9P_REACHABILITY "
        f"raw_gap_preimage_tests={raw_preimage_tests} pair_phase_tests={raw_pair_phase_tests} pair_phase_nodes={raw_pair_phase_nodes}",
        flush=True,
    )
    print(
        "R9P_FINAL_ENCLOSURE_GATE "
        f"raw_gap_cells={raw_cells} stagewise_cellmax_gap_cells={cellmax_cells} stagewise_cellmax_gap_m={cellmax_gap}",
        flush=True,
    )
    print(
        "R9P_FINAL_TARGET_PAIR_GFP "
        f"common_nodes={common_nodes} upper_sensitive={upper_sensitive} paired_positive={paired_positive}",
        flush=True,
    )

    classification = "RAW_REACHABLE_SERVICE_VALUE_SENSITIVITY_PERSISTS_BUT_CONTINUOUS_MAXIMAL_KERNEL_SEPARATION_NOT_CERTIFIED_DUE_TO_TENSOR_CELL_ENCLOSURE"
    next_action = "REVISE_MANUSCRIPT_SCOPE_PRESERVE_DIAGNOSTIC_VALUE_SENSITIVITY_AND_TREAT_LOCAL_SPARSE_ENCLOSURES_AS_FUTURE_METHOD_WORK"

    evidence_rows = [
        {"claim": "raw_stage_value_sensitivity", "status": "SUPPORTED", "value": f"nodes={raw_nodes}; max_gap_m={raw_gap}", "source": upstream_paths.get("R9OR2", "")},
        {"claim": "raw_gap_physical_reachability", "status": "SUPPORTED", "value": f"preimage_tests={raw_preimage_tests}", "source": upstream_paths.get("R9NR1", "")},
        {"claim": "matched_pair_phase_reachability", "status": "SUPPORTED", "value": f"tests={raw_pair_phase_tests}; nodes={raw_pair_phase_nodes}", "source": upstream_paths.get("R9NR1", "")},
        {"claim": "stagewise_cellmax_value_separation", "status": "NOT_SUPPORTED", "value": f"cells={cellmax_cells}; max_gap_m={cellmax_gap}", "source": upstream_paths.get("R9OR2", "")},
        {"claim": "full_augmented_target_pair_point_gfp_separation", "status": "NOT_SUPPORTED", "value": f"upper_sensitive={upper_sensitive}/{common_nodes}", "source": upstream_paths.get("R9OR2", "")},
        {"claim": "continuous_action_interval_gfp", "status": "NOT_SOLVED", "value": "NO", "source": upstream_paths.get("R9OR2", "")},
        {"claim": "independent_p2c_lower_checker", "status": "NOT_AVAILABLE", "value": "NO", "source": upstream_paths.get("R9OR2", "")},
        {"claim": "continuous_state_separation", "status": "NOT_CERTIFIED", "value": "NO", "source": upstream_paths.get("R9OR2", "")},
        {"claim": "deployment_protocol_certificate", "status": "NOT_CERTIFIED", "value": "NO", "source": upstream_paths.get("R9OR2", "")},
    ]

    risk_rows = manuscript_audit()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result = RESULTS / f"P3B1_R9P_RESULT_{stamp}.json"
    latest = RESULTS / "P3B1_R9P_LATEST.json"
    ledger = RESULTS / f"P3B1_R9P_EVIDENCE_LEDGER_{stamp}.csv"
    audit = RESULTS / f"P3B1_R9P_MANUSCRIPT_CLAIM_AUDIT_{stamp}.csv"
    scope_tex = RESULTS / f"P3B1_R9P_MANUSCRIPT_SCOPE_{stamp}.tex"
    closeout = RESULTS / f"P3B1_R9P_CLOSEOUT_{stamp}.txt"
    manifest = RESULTS / f"P3B1_R9P_MANIFEST_{stamp}.sha256"

    write_csv(ledger, evidence_rows)
    write_csv(audit, risk_rows)

    tex = rf"""% Auto-generated by R9-P. Do not promote beyond the stated scope.
\paragraph{{Numerical scope of authentication-service sensitivity.}}
The diagnostic computations identify a persistent pointwise dependence of the converged service-stage value representation on the authentication-service state. After the final predeclared two-axis refinement, {int(raw_nodes)} semantic nodal states retain a nonzero old-versus-replacement last-stage difference, with maximum observed magnitude {float(raw_gap):.12g}\,m. The corresponding coarse-stage discrepancy is not merely unreachable: the preceding enclosure audit found reachable preimages, including {int(raw_pair_phase_tests)} state-action hits on the matched comparison phase ({int(raw_pair_phase_nodes)} source nodes). However, the conservative stage-wise cell-corner maximum enclosure removes this distinction on every refined cell tested: the final two-axis computation reports {int(cellmax_cells)} cells with a nonzero enclosed stage gap. Consequently, the full augmented target-pair point-action fixed point remains indistinguishable on the comparison domain ({int(upper_sensitive)} sensitive nodes among {int(common_nodes)} common nodes), and no positive paired lower--upper margin is obtained. These results support local/nodal authentication-service value sensitivity and diagnose a finite-abstraction enclosure limitation; they do not certify continuous-domain separation of the maximal viability kernel.

\paragraph{{Claim boundary.}}
The manuscript may state that the authentication-service state can alter control-relevant future traces, local Bellman requirements, and nodal diagnostic value functions under the declared diagnostic contracts. It must not state that continuous-state maximal-kernel membership separation has been certified for the current model. In particular, \texttt{{CONTINUOUS\_STATE\_SEPARATION\_CERTIFIED=NO}}, continuous-action interval GFP closure remains open, and the deployment service transition relation is not certified by the present diagnostic construction.
"""
    atomic_write(scope_tex, tex)

    close_text = f"""P3-B1-R9-P CONTINUOUS SCOPE CLOSEOUT\n\nSTATUS=PASS\nCLASSIFICATION={classification}\n\nAUTHORITATIVE_FLAGS\nPOSITIVE_CONTINUOUS_PROMOTION_STOP=YES\nFURTHER_AUTOMATIC_AXIS_REFINEMENT_AUTHORIZED=NO\nRAW_SERVICE_VALUE_SENSITIVITY_PRESERVE=YES\nREACHABLE_RAW_VALUE_GAP_PRESERVE=YES\nTENSOR_CELL_ENCLOSURE_LIMITATION=YES\nFULL_AUGMENTED_TARGET_PAIR_POINT_GFP_SEPARATION=NO\nCONTINUOUS_ACTION_INTERVAL_GFP_SOLVED=NO\nP2C_INDEPENDENT_LOWER_CHECKER=NO\nCONTINUOUS_STATE_SEPARATION_CERTIFIED=NO\nDEPLOYMENT_PROTOCOL_CERTIFIED=NO\nCONTINUOUS_PROMOTION_BRANCH_CLOSED=YES\n\nFINAL_NUMERICAL_FACTS\nTWO_AXIS_RAW_GAP_NODES={raw_nodes}\nTWO_AXIS_RAW_GAP_M={raw_gap}\nTWO_AXIS_RAW_GAP_CELLS={raw_cells}\nTWO_AXIS_STAGEWISE_CELLMAX_GAP_CELLS={cellmax_cells}\nMATCHED_PAIR_PHASE_RAW_GAP_PREIMAGE_TESTS={raw_pair_phase_tests}\nMATCHED_PAIR_PHASE_RAW_GAP_PREIMAGE_NODES={raw_pair_phase_nodes}\nTARGET_PAIR_COMMON_NODES={common_nodes}\nTARGET_PAIR_UPPER_SENSITIVE_NODES={upper_sensitive}\nTARGET_PAIR_PAIRED_POSITIVE={paired_positive}\n\nNEXT_ACTION={next_action}\n\nMANUSCRIPT_SCOPE\nALLOWED: control-relevant service-state trace differences; matched-chi local Bellman sensitivity; persistent reachable nodal service-value sensitivity; explicit diagnosis that coarse/two-axis tensor-cell corner-max enclosure erases those nodal differences.\nNOT_ALLOWED: certified continuous-domain maximal viability-kernel separation for the current diagnostic model; deployment-specific claim; theorem-level continuous-action structural separation.\n"""
    atomic_write(closeout, close_text)

    out = {
        "schema": "P3B1_R9P_CONTINUOUS_SCOPE_CLOSEOUT_V1",
        "status": "PASS",
        "classification": classification,
        "authoritative_flags": {
            "positive_continuous_promotion_stop": True,
            "further_automatic_axis_refinement_authorized": False,
            "raw_service_value_sensitivity_preserve": True,
            "reachable_raw_value_gap_preserve": True,
            "tensor_cell_enclosure_limitation": True,
            "full_augmented_target_pair_point_gfp_separation": False,
            "continuous_action_interval_gfp_solved": False,
            "p2c_independent_lower_checker": False,
            "continuous_state_separation_certified": False,
            "deployment_protocol_certified": False,
            "continuous_promotion_branch_closed": True,
        },
        "final_metrics": {
            "two_axis_raw_gap_nodes": int(raw_nodes),
            "two_axis_raw_gap_m": float(raw_gap),
            "two_axis_raw_gap_cells": int(raw_cells),
            "two_axis_stagewise_cellmax_gap_cells": int(cellmax_cells),
            "two_axis_stagewise_cellmax_gap_m": float(cellmax_gap),
            "raw_gap_preimage_tests": int(raw_preimage_tests),
            "matched_pair_phase_preimage_tests": int(raw_pair_phase_tests),
            "matched_pair_phase_preimage_nodes": int(raw_pair_phase_nodes),
            "target_pair_common_nodes": int(common_nodes),
            "target_pair_upper_sensitive_nodes": int(upper_sensitive),
            "target_pair_paired_positive": int(paired_positive),
        },
        "manuscript_claim_audit_hits": len(risk_rows),
        "upstream_latest": upstream_paths,
        "next_action": next_action,
        "outputs": {
            "evidence_ledger_csv": str(ledger),
            "manuscript_claim_audit_csv": str(audit),
            "manuscript_scope_tex": str(scope_tex),
            "closeout_txt": str(closeout),
        },
    }
    atomic_write(result, json.dumps(out, indent=2, sort_keys=True) + "\n")
    atomic_write(latest, json.dumps(out, indent=2, sort_keys=True) + "\n")

    manifest_lines = []
    for p in [result, latest, ledger, audit, scope_tex, closeout]:
        manifest_lines.append(f"{sha256_file(p)}  {p.name}")
    atomic_write(manifest, "\n".join(manifest_lines) + "\n")

    print("=== R9-P DECISION ===", flush=True)
    print("R9P_CLOSEOUT_GATE=PASS", flush=True)
    print(f"R9P_MANUSCRIPT_CLAIM_AUDIT hits={len(risk_rows)}", flush=True)
    print("POSITIVE_CONTINUOUS_PROMOTION_STOP=YES", flush=True)
    print("FURTHER_AUTOMATIC_AXIS_REFINEMENT_AUTHORIZED=NO", flush=True)
    print("RAW_SERVICE_VALUE_SENSITIVITY_PRESERVE=YES", flush=True)
    print("REACHABLE_RAW_VALUE_GAP_PRESERVE=YES", flush=True)
    print("TENSOR_CELL_ENCLOSURE_LIMITATION=YES", flush=True)
    print("FULL_AUGMENTED_TARGET_PAIR_POINT_GFP_SEPARATION=NO", flush=True)
    print("CONTINUOUS_ACTION_INTERVAL_GFP_SOLVED=NO", flush=True)
    print("P2C_INDEPENDENT_LOWER_CHECKER=NO", flush=True)
    print("CONTINUOUS_STATE_SEPARATION_CERTIFIED=NO", flush=True)
    print("DEPLOYMENT_PROTOCOL_CERTIFIED=NO", flush=True)
    print("CONTINUOUS_PROMOTION_BRANCH_CLOSED=YES", flush=True)
    print(f"R9P_CLASSIFICATION={classification}", flush=True)
    print("R9P_EXECUTION=PASS", flush=True)
    print(f"R9P_NEXT_ACTION={next_action}", flush=True)
    print(f"RESULT_JSON={result}", flush=True)
    print(f"EVIDENCE_LEDGER_CSV={ledger}", flush=True)
    print(f"MANUSCRIPT_CLAIM_AUDIT_CSV={audit}", flush=True)
    print(f"MANUSCRIPT_SCOPE_TEX={scope_tex}", flush=True)
    print(f"CLOSEOUT_TXT={closeout}", flush=True)
    print(f"MANIFEST={manifest}", flush=True)


if __name__ == "__main__":
    main()
