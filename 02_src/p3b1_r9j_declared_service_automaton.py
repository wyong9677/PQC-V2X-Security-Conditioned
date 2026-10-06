from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
RESULTS = ROOT / "04_results"
CONFIG = ROOT / "01_config"

EXPECTED_R9I = "CURRENT_PAIR_STRUCTURALLY_COLLAPSED_AND_NO_CERTIFIED_NONDominated_REPLACEMENT_PAIR_YET"


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def locate_manuscript() -> Path | None:
    candidates = [
        ROOT.parent / "Section3_System_and_Authentication_Service_Model_revised.tex",
        ROOT.parent / "PQC_V2X_Security_Conditioned.tex",
        ROOT.parent / "PQC_V2X_Security_Conditioned(5).tex",
    ]
    for p in candidates:
        if p.exists():
            return p
    tex = sorted(ROOT.parent.glob("*.tex"))
    return tex[0] if tex else None


def semantic_evidence(path: Path | None) -> dict:
    if path is None:
        return {"found": False, "all_required_semantics": False}
    t = path.read_text(encoding="utf-8", errors="replace")
    low = t.lower()
    def has(*parts: str) -> bool:
        return all(p.lower() in low for p in parts)
    checks = {
        "found": True,
        "path": str(path),
        "stateful_service": has("stateful service"),
        "stage_fragmented": has("fragmented"),
        "stage_verifying": has("verifying"),
        "outcome_eligible": has("eligible"),
        "outcome_rejected": has("rejected"),
        "outcome_timeout": has("timeout"),
        "replacement_semantics": has("replacement"),
        "retransmission_semantics": has("retransmission"),
        "generation_time_preserved": has("generation time", "preserved"),
        "finite_service_horizon": has("finite service horizon"),
        "same_horizon_trace_difference_example": has("same two-interval certified liveness bound", "fragmented", "verification stage"),
        "pair_explicitly_not_viability_claim": has("conceptual service-state counterexample", "does not assert that this particular pair has different viability membership"),
        "exact_transition_relation_profile_specific": has("parameters required to define", "mathfrak r_c"),
    }
    required = [
        "stateful_service", "stage_fragmented", "stage_verifying",
        "outcome_eligible", "outcome_rejected", "replacement_semantics",
        "generation_time_preserved", "finite_service_horizon",
        "same_horizon_trace_difference_example",
    ]
    checks["all_required_semantics"] = all(checks[k] for k in required)
    return checks


def p1_fast_profile() -> dict:
    p = CONFIG / "p1_validation_v2.json"
    if not p.exists():
        return {"found": False}
    d = load_json(p)
    prof = d.get("diagnostic_service_profiles", {}).get("diagnostic_fast")
    if not isinstance(prof, dict):
        return {"found": False, "path": str(p)}
    return {
        "found": True,
        "path": str(p),
        "status": d.get("status"),
        "eligible_bound_steps": int(prof.get("eligible_bound_steps", -1)),
        "max_loss_burst": int(prof.get("max_loss_burst", -1)),
    }


def validate_contract(spec: dict, sem: dict, fast: dict) -> dict:
    states = spec.get("states", {})
    edges = spec.get("edges", [])
    names = set(states)
    edge_ok = all(str(e.get("src")) in names and str(e.get("dst")) in names for e in edges)
    outcomes = {str(e.get("outcome")) for e in edges}
    supported_outcomes = outcomes <= {"NoOutput", "Eligible", "Rejected", "Timeout"}
    mechanisms = {str(e.get("candidate_update", "preserve")) for e in edges}
    supported_mechanisms = mechanisms <= {"preserve", "terminate", "terminate_and_replace", "new_candidate"}
    liveness_pair = spec.get("comparison_pair", [])
    same_liveness = False
    matched_current = False
    if len(liveness_pair) == 2 and all(x in states for x in liveness_pair):
        a, b = liveness_pair
        same_liveness = states[a].get("liveness_bound") == states[b].get("liveness_bound") == 2
        matched_current = bool(states[a].get("matched_current_chi")) and bool(states[b].get("matched_current_chi"))
    return {
        "schema_ok": spec.get("schema") == "SCV_R9J_DECLARED_DIAGNOSTIC_SERVICE_CONTRACT_V2",
        "contract_type_ok": spec.get("contract_type") == "DECLARED_DIAGNOSTIC_NOT_DEPLOYMENT",
        "deployment_certified_false": spec.get("deployment_certified") is False,
        "edge_endpoints_valid": edge_ok,
        "outcomes_supported": supported_outcomes,
        "candidate_updates_supported": supported_mechanisms,
        "manuscript_semantics_support": bool(sem.get("all_required_semantics")),
        "diagnostic_fast_two_step_support": bool(fast.get("found") and fast.get("eligible_bound_steps") == 2),
        "comparison_same_liveness_two": same_liveness,
        "comparison_matched_current_chi": matched_current,
    }


def enriched_traces(spec: dict, horizon: int) -> dict[str, set[tuple[str, ...]]]:
    states = set(spec.get("states", {}))
    adj: dict[str, list[dict]] = defaultdict(list)
    for e in spec.get("edges", []):
        src, dst = str(e["src"]), str(e["dst"])
        if src not in states or dst not in states:
            raise RuntimeError(f"R9J_BAD_EDGE={src}->{dst}")
        adj[src].append(e)
    out: dict[str, set[tuple[str, ...]]] = {s: set() for s in states}
    for s0 in states:
        q = deque([(s0, tuple(), 0)])
        seen = set()
        while q:
            s, tr, d = q.popleft()
            key = (s, tr, d)
            if key in seen:
                continue
            seen.add(key)
            if d >= horizon or not adj[s]:
                out[s0].add(tr)
                continue
            for e in adj[s]:
                dst = str(e["dst"])
                token = "|".join([
                    str(e.get("outcome", "NoOutput")),
                    str(e.get("candidate_update", "preserve")),
                    str(spec["states"][dst].get("stage", dst)),
                ])
                q.append((dst, tr + (token,), d + 1))
    return out


def common_predecessors(spec: dict) -> dict[tuple[str, str], list[str]]:
    succ: dict[str, set[str]] = defaultdict(set)
    for e in spec.get("edges", []):
        succ[str(e["src"])].add(str(e["dst"]))
    out: dict[tuple[str, str], list[str]] = defaultdict(list)
    for pred, sset in succ.items():
        ss = sorted(sset)
        for i in range(len(ss)):
            for j in range(i + 1, len(ss)):
                out[(ss[i], ss[j])].append(pred)
    return out


def screen_pairs(spec: dict, horizon: int) -> tuple[list[dict], dict[str, set[tuple[str, ...]]]]:
    traces = enriched_traces(spec, horizon)
    cp = common_predecessors(spec)
    states = spec["states"]
    rows: list[dict] = []
    names = sorted(states)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            ta, tb = traces[a], traces[b]
            la, lb = states[a].get("liveness_bound"), states[b].get("liveness_bound")
            same_liveness = la is not None and la == lb
            matched_current_chi = bool(states[a].get("matched_current_chi")) and bool(states[b].get("matched_current_chi"))
            preds = cp.get((a, b), cp.get((b, a), []))
            a_sub_b = ta.issubset(tb)
            b_sub_a = tb.issubset(ta)
            nondominated = (not a_sub_b) and (not b_sub_a)
            symmetric_difference = len(ta.symmetric_difference(tb))
            candidate = bool(same_liveness and matched_current_chi and preds and nondominated)
            rows.append({
                "state_a": a,
                "state_b": b,
                "stage_a": states[a].get("stage"),
                "stage_b": states[b].get("stage"),
                "liveness_a": la,
                "liveness_b": lb,
                "same_liveness": same_liveness,
                "matched_current_chi": matched_current_chi,
                "common_predecessors": ";".join(preds),
                "trace_count_a": len(ta),
                "trace_count_b": len(tb),
                "a_trace_subset_b": a_sub_b,
                "b_trace_subset_a": b_sub_a,
                "trace_nondominated": nondominated,
                "trace_symmetric_difference": symmetric_difference,
                "screen_candidate": candidate,
            })
    return rows, traces


def self_test() -> None:
    spec = declared_contract_template()
    sem = {"all_required_semantics": True}
    fast = {"found": True, "eligible_bound_steps": 2}
    gates = validate_contract(spec, sem, fast)
    assert all(gates.values()), gates
    rows, traces = screen_pairs(spec, 4)
    cand = [r for r in rows if r["screen_candidate"]]
    assert cand, "expected at least one nondominated candidate"
    target = {tuple(sorted(spec["comparison_pair"]))}
    assert any(tuple(sorted((r["state_a"], r["state_b"]))) in target for r in cand)
    a, b = spec["comparison_pair"]
    assert traces[a] != traces[b]
    print("R9J_INTERNAL_SELF_TEST=PASS", flush=True)


def declared_contract_template() -> dict:
    return {
        "schema": "SCV_R9J_DECLARED_DIAGNOSTIC_SERVICE_CONTRACT_V2",
        "status": "DECLARED_DIAGNOSTIC_CONTRACT_FOR_THEOREM_FALSIFICATION_NOT_DEPLOYMENT_DATA",
        "contract_type": "DECLARED_DIAGNOSTIC_NOT_DEPLOYMENT",
        "deployment_certified": False,
        "profile_basis": "diagnostic_fast_two_step_liveness",
        "scientific_boundary": [
            "This contract is an explicit mathematical diagnostic instantiation of service mechanisms already admitted by the manuscript.",
            "It is not asserted to be a measured ML-DSA/SLH-DSA or production V2X deployment.",
            "Rejection/replacement are used because the manuscript explicitly permits them in q and R_c; they are not presented as observed frequencies.",
            "Any manuscript positive continuous result obtained from this contract must be labeled diagnostic-contract conditioned unless later mapped to deployment evidence.",
        ],
        "comparison_pair": ["FRAGMENT_CARRY", "CREDENTIAL_DECISION"],
        "states": {
            "SAME_CANDIDATE_ENTRY": {
                "stage": "Received",
                "liveness_bound": 3,
                "matched_current_chi": True,
                "meaning": "Common-history predecessor holding the same pending message/generation time before the service fork.",
            },
            "FRAGMENT_CARRY": {
                "stage": "Fragmented",
                "liveness_bound": 2,
                "matched_current_chi": True,
                "meaning": "One fragment remains; current candidate and generation time are preserved.",
            },
            "CREDENTIAL_DECISION": {
                "stage": "Verifying",
                "liveness_bound": 2,
                "matched_current_chi": True,
                "meaning": "Same current candidate enters a credential/eligibility decision; rejection may terminate it and invoke replacement.",
            },
            "VERIFY_OLD_LAST": {
                "stage": "Verifying",
                "liveness_bound": 1,
                "matched_current_chi": True,
                "meaning": "Last verification interval for the original candidate.",
            },
            "VERIFY_REPLACEMENT_LAST": {
                "stage": "Verifying",
                "liveness_bound": 1,
                "matched_current_chi": False,
                "meaning": "Last verification interval for a newly generated replacement candidate.",
            },
            "POST_TERMINATION_ENTRY": {
                "stage": "Idle",
                "liveness_bound": 3,
                "matched_current_chi": False,
                "meaning": "Post-output restart state; future candidate cycle restarts.",
            },
        },
        "edges": [
            {
                "src": "SAME_CANDIDATE_ENTRY",
                "dst": "FRAGMENT_CARRY",
                "event": "fragment_path_realized",
                "outcome": "NoOutput",
                "candidate_update": "preserve",
                "evidence_class": "DECLARED_DIAGNOSTIC_EDGE",
            },
            {
                "src": "SAME_CANDIDATE_ENTRY",
                "dst": "CREDENTIAL_DECISION",
                "event": "credential_path_realized",
                "outcome": "NoOutput",
                "candidate_update": "preserve",
                "evidence_class": "DECLARED_DIAGNOSTIC_EDGE",
            },
            {
                "src": "FRAGMENT_CARRY",
                "dst": "VERIFY_OLD_LAST",
                "event": "remaining_fragment_delivered",
                "outcome": "NoOutput",
                "candidate_update": "preserve",
                "evidence_class": "MANUSCRIPT_CONCEPTUAL_SEMANTIC",
            },
            {
                "src": "VERIFY_OLD_LAST",
                "dst": "POST_TERMINATION_ENTRY",
                "event": "verification_completes",
                "outcome": "Eligible",
                "candidate_update": "terminate",
                "evidence_class": "DECLARED_DIAGNOSTIC_EDGE",
            },
            {
                "src": "CREDENTIAL_DECISION",
                "dst": "VERIFY_REPLACEMENT_LAST",
                "event": "credential_or_protocol_rejects_then_replacement_generated",
                "outcome": "Rejected",
                "candidate_update": "terminate_and_replace",
                "evidence_class": "DECLARED_DIAGNOSTIC_EDGE_USING_MANUSCRIPT_PERMITTED_REJECTION_REPLACEMENT",
            },
            {
                "src": "VERIFY_REPLACEMENT_LAST",
                "dst": "POST_TERMINATION_ENTRY",
                "event": "replacement_verification_completes",
                "outcome": "Eligible",
                "candidate_update": "terminate",
                "evidence_class": "DECLARED_DIAGNOSTIC_EDGE",
            },
            {
                "src": "POST_TERMINATION_ENTRY",
                "dst": "SAME_CANDIDATE_ENTRY",
                "event": "new_cycle_candidate_available",
                "outcome": "NoOutput",
                "candidate_update": "new_candidate",
                "evidence_class": "DECLARED_DIAGNOSTIC_RESTART_EDGE",
            },
        ],
        "screening_horizon": 4,
    }


def latex_contract(spec: dict, sem: dict, fast: dict, top: dict | None) -> str:
    pair = spec.get("comparison_pair", ["?", "?"])
    top_text = "none" if top is None else f"{top['state_a']} vs. {top['state_b']}"
    return rf"""% Auto-generated R9-J diagnostic service contract note.
% This is NOT deployment evidence.
\paragraph{{Declared diagnostic service contract v2.}}
For theorem falsification only, we instantiate a two-step diagnostic service
contract using mechanisms already admitted by the model: fragmentation,
verification, protocol rejection, and pending-message replacement.  The
contract is not identified with a measured post-quantum V2X deployment.
The compared states are \texttt{{{pair[0]}}} and
\texttt{{{pair[1]}}}.  They share the same current pending message descriptor
and the same current generation time, have the same certified scalar liveness
bound $R=2$, and are co-reachable from a common predecessor.  Their future
control-observable service traces are intentionally non-dominated: the first
preserves the old candidate through a remaining-fragment path, whereas the
second admits rejection of that same candidate followed by generation and
verification of a replacement candidate.  This contract may be used to test
whether equal scalar liveness but different control-relevant service semantics
can generate a full augmented-kernel separation.  Any positive result remains
conditioned on this declared diagnostic contract unless a deployment-refinement
certificate is supplied.

% Manuscript semantic support gate: {sem.get('all_required_semantics')}
% diagnostic_fast eligible steps: {fast.get('eligible_bound_steps')}
% top screened pair: {top_text}
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--contract", default="")
    ap.add_argument("--horizon", type=int, default=4)
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not (1 <= args.horizon <= 12):
        raise ValueError("R9J_BAD_HORIZON")

    r9i_path = RESULTS / "P3B1_R9I_LATEST.json"
    if not r9i_path.exists():
        raise RuntimeError("R9J_MISSING_R9I_LATEST")
    r9i = load_json(r9i_path)
    if r9i.get("status") != "PASS" or r9i.get("classification") != EXPECTED_R9I:
        raise RuntimeError(f"R9J_R9I_GATE_FAIL={r9i.get('classification')}")

    manuscript = locate_manuscript()
    sem = semantic_evidence(manuscript)
    fast = p1_fast_profile()

    contract_path = Path(args.contract).expanduser().resolve() if args.contract else CONFIG / "p3b1_r9j_declared_diagnostic_service_contract_v2.json"
    if not contract_path.exists():
        atomic_write(contract_path, json.dumps(declared_contract_template(), indent=2, sort_keys=True) + "\n")
    spec = load_json(contract_path)
    gates = validate_contract(spec, sem, fast)
    if not all(gates.values()):
        raise RuntimeError("R9J_CONTRACT_VALIDATION_FAIL=" + json.dumps(gates, sort_keys=True))

    rows, traces = screen_pairs(spec, int(args.horizon))
    candidates = [r for r in rows if r["screen_candidate"]]
    candidates.sort(key=lambda r: (-int(r["trace_symmetric_difference"]), r["state_a"], r["state_b"]))
    top = candidates[0] if candidates else None
    target_pair = tuple(sorted(spec["comparison_pair"]))
    target = next((r for r in candidates if tuple(sorted((r["state_a"], r["state_b"]))) == target_pair), None)

    # Strict evidence boundary: the manuscript specifies the semantic categories
    # and profile ingredients but not a deployment-complete transition table.
    strict_profile_complete = False
    diagnostic_contract_valid = all(gates.values())
    deployment_certified = False

    if target is not None:
        classification = "DECLARED_DIAGNOSTIC_NONDominated_MATCHED_CHI_PAIR_FOUND_READY_FOR_FULL_AUGMENTED_GFP_NOT_DEPLOYMENT_CERTIFIED"
        next_action = "R9K_BUILD_MATCHED_CHI_FULL_AUGMENTED_GFP_FOR_R9J_DECLARED_DIAGNOSTIC_PAIR"
    else:
        classification = "DECLARED_DIAGNOSTIC_CONTRACT_VALID_BUT_NO_NONDominated_MATCHED_CHI_PAIR_FOUND"
        next_action = "R9J_R1_REVISE_DIAGNOSTIC_CONTRACT_WITHOUT_EXCEEDING_MANUSCRIPT_SEMANTICS"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result = RESULTS / f"P3B1_R9J_RESULT_{stamp}.json"
    latest = RESULTS / "P3B1_R9J_LATEST.json"
    pair_csv = RESULTS / f"P3B1_R9J_PAIR_SCREEN_{stamp}.csv"
    trace_json = RESULTS / f"P3B1_R9J_TRACE_SETS_{stamp}.json"
    evidence_csv = RESULTS / f"P3B1_R9J_EVIDENCE_LEDGER_{stamp}.csv"
    contract_copy = RESULTS / f"P3B1_R9J_DECLARED_DIAGNOSTIC_CONTRACT_{stamp}.json"
    note_tex = RESULTS / f"P3B1_R9J_DECLARED_DIAGNOSTIC_CONTRACT_{stamp}.tex"
    manifest = RESULTS / f"P3B1_R9J_MANIFEST_{stamp}.sha256"

    write_csv(pair_csv, rows)
    trace_serial = {k: [list(x) for x in sorted(v)] for k, v in traces.items()}
    atomic_write(trace_json, json.dumps(trace_serial, indent=2, sort_keys=True) + "\n")
    atomic_write(contract_copy, json.dumps(spec, indent=2, sort_keys=True) + "\n")
    atomic_write(note_tex, latex_contract(spec, sem, fast, top))

    evidence_rows = []
    for k, v in sem.items():
        if k not in {"path", "found"}:
            evidence_rows.append({"layer": "manuscript_semantics", "item": k, "supported": bool(v), "source": sem.get("path", "")})
    evidence_rows.extend([
        {"layer": "numerical_profile", "item": "diagnostic_fast_eligible_bound_steps_2", "supported": bool(fast.get("found") and fast.get("eligible_bound_steps") == 2), "source": fast.get("path", "")},
        {"layer": "deployment_transition_relation", "item": "complete_R_c_edge_table", "supported": False, "source": "not supplied by current manuscript/config"},
        {"layer": "diagnostic_contract_v2", "item": "explicit_new_declared_contract", "supported": diagnostic_contract_valid, "source": str(contract_path)},
    ])
    write_csv(evidence_csv, evidence_rows)

    out = {
        "schema": "SCV_P3B1_R9J_DECLARED_SERVICE_AUTOMATON_V2",
        "status": "PASS",
        "timestamp_utc": stamp,
        "classification": classification,
        "next_action": next_action,
        "continuous_state_separation_certified": False,
        "upstream": {
            "r9i_classification": r9i.get("classification"),
            "current_pair_structurally_collapsed": True,
        },
        "evidence_boundary": {
            "manuscript_semantics": sem,
            "diagnostic_fast_profile": fast,
            "strict_manuscript_or_deployment_transition_relation_complete": strict_profile_complete,
            "deployment_certified": deployment_certified,
            "declared_diagnostic_contract_validated": diagnostic_contract_valid,
            "diagnostic_contract_is_not_deployment_evidence": True,
        },
        "contract_validation_gates": gates,
        "pair_screen": {
            "horizon": int(args.horizon),
            "pairs_screened": len(rows),
            "nondominated_matched_chi_candidates": len(candidates),
            "target_pair": spec.get("comparison_pair"),
            "target_pair_pass": target is not None,
            "top_candidate": top,
            "all_candidates": candidates,
        },
        "gates": {
            "upstream_r9i_pass": True,
            "manuscript_semantics_support": bool(sem.get("all_required_semantics")),
            "diagnostic_fast_two_step_support": bool(fast.get("found") and fast.get("eligible_bound_steps") == 2),
            "declared_diagnostic_contract_valid": diagnostic_contract_valid,
            "matched_chi_same_liveness_common_history_nondominated_pair_found": target is not None,
            "deployment_transition_relation_certified": False,
            "continuous_state_separation_certified": False,
        },
        "artifacts": {
            "pair_screen_csv": str(pair_csv),
            "trace_sets_json": str(trace_json),
            "evidence_ledger_csv": str(evidence_csv),
            "declared_contract_json": str(contract_copy),
            "declared_contract_tex": str(note_tex),
        },
    }
    text = json.dumps(out, indent=2, sort_keys=True) + "\n"
    atomic_write(result, text)
    atomic_write(latest, text)

    manifest_files = [Path(__file__), r9i_path, contract_path, result, pair_csv, trace_json, evidence_csv, contract_copy, note_tex]
    if manuscript is not None:
        manifest_files.append(manuscript)
    p1 = CONFIG / "p1_validation_v2.json"
    if p1.exists():
        manifest_files.append(p1)
    atomic_write(manifest, "".join(f"{sha256_file(p)}  {p}\n" for p in manifest_files if p.exists()))

    print("=== P3-B1-R9-J DECLARED SERVICE-AUTOMATON INSTANTIATION ===", flush=True)
    print("UPSTREAM_R9I_NO_CERTIFIED_REPLACEMENT_PAIR=PASS", flush=True)
    print(f"R9J_MANUSCRIPT_SEMANTIC_GATE={'PASS' if sem.get('all_required_semantics') else 'FAIL'}", flush=True)
    print(f"R9J_DIAGNOSTIC_FAST_TWO_STEP_GATE={'PASS' if fast.get('found') and fast.get('eligible_bound_steps') == 2 else 'FAIL'}", flush=True)
    print("R9J_STRICT_DEPLOYMENT_TRANSITION_RELATION_COMPLETE=NO", flush=True)
    print(f"R9J_DECLARED_DIAGNOSTIC_CONTRACT_GATE={'PASS' if diagnostic_contract_valid else 'FAIL'}", flush=True)
    print("R9J_DECLARED_DIAGNOSTIC_CONTRACT_IS_DEPLOYMENT_CERTIFIED=NO", flush=True)
    print(f"R9J_PAIR_SCREEN pairs={len(rows)} candidates={len(candidates)} target_pair_pass={'YES' if target is not None else 'NO'}", flush=True)
    if top is not None:
        print(f"R9J_TOP_PAIR={top['state_a']}|{top['state_b']} same_liveness={top['same_liveness']} matched_chi={top['matched_current_chi']} common_predecessor={top['common_predecessors']} trace_symdiff={top['trace_symmetric_difference']}", flush=True)
    print("CONTINUOUS_STATE_SEPARATION_CERTIFIED=NO", flush=True)
    print("R9J_EXECUTION=PASS", flush=True)
    print(f"R9J_CLASSIFICATION={classification}", flush=True)
    print(f"R9J_NEXT_ACTION={next_action}", flush=True)
    print(f"RESULT_JSON={result}", flush=True)
    print(f"PAIR_SCREEN_CSV={pair_csv}", flush=True)
    print(f"EVIDENCE_LEDGER_CSV={evidence_csv}", flush=True)
    print(f"DECLARED_CONTRACT_JSON={contract_copy}", flush=True)
    print(f"DECLARED_CONTRACT_TEX={note_tex}", flush=True)
    print(f"MANIFEST={manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
