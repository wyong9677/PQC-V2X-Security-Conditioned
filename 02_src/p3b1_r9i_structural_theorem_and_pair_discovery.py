from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
RESULTS = ROOT / "04_results"
CONFIG = ROOT / "01_config"

EXPECTED_R9HH = "FULL_GFP_STAGE_COLLAPSE_EXPLAINED_BY_ACTIONWISE_VERIFY_COMPLETION_BRANCH_DOMINANCE"
TOL = 1.0e-12


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
        w.writerows(rows)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def theorem_identity(completion: float, delayed: float) -> tuple[float, float]:
    """Conditional identity used by the current-pair theorem.

    FRAGMENTED future = delayed.
    VERIFYING future = max(completion, delayed).
    If completion <= delayed then the two futures are identical.
    """
    frag = delayed
    verify = max(completion, delayed)
    return frag, verify


def self_test() -> None:
    for c, d in [(0.0, 1.0), (1.0, 1.0), (-3.0, 2.0)]:
        f, v = theorem_identity(c, d)
        assert abs(f - v) <= TOL
    f, v = theorem_identity(2.0, 1.0)
    assert v > f

    spec = {
        "deployment_certified": True,
        "states": {
            "A": {"liveness_bound": 2},
            "B": {"liveness_bound": 2},
            "C": {"liveness_bound": 1},
        },
        "edges": [
            {"src": "A", "dst": "C", "outcome": "NoOutput"},
            {"src": "C", "dst": "C", "outcome": "Eligible"},
            {"src": "B", "dst": "C", "outcome": "Rejected"},
        ],
    }
    traces = bounded_traces(spec, horizon=2)
    assert traces["A"] != traces["B"]
    assert not traces["A"].issubset(traces["B"])
    assert not traces["B"].issubset(traces["A"])
    print("R9I_INTERNAL_SELF_TEST=PASS", flush=True)


def operator_structure_audit(source_path: Path) -> dict:
    """Audit exact source structure, not numerical output.

    This does NOT certify continuous-action dominance.  It certifies that the
    implemented Bellman branches have the intended symbolic form.
    """
    text = source_path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    assignments: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in {"comp_v", "future_frag", "future_verify", "future_last"}:
                assignments[name] = ast.unparse(node.value)

    checks = {
        "comp_v_is_safe_adopt_min": bool(re.search(r"minimum\s*\(\s*adopt_v\s*,\s*entry_hold\s*\)", assignments.get("comp_v", ""))),
        "fragment_future_is_last": assignments.get("future_frag", "").strip() == "last_hold",
        "verify_future_is_max_completion_last": bool(re.search(r"maximum\s*\(\s*comp_v\s*,\s*last_hold\s*\)", assignments.get("future_verify", ""))),
        "last_future_is_completion": assignments.get("future_last", "").strip() == "comp_last",
    }
    checks["pass"] = all(checks.values())
    checks["assignments"] = assignments
    return checks


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


def manuscript_semantic_audit(path: Path | None) -> dict:
    if path is None:
        return {
            "found": False,
            "declares_stateful_service": False,
            "declares_outcomes": False,
            "declares_non_age_only_features": False,
            "declares_fragmented_vs_verifying_conceptual_pair": False,
            "explicitly_says_pair_not_viability_claim": False,
        }
    t = path.read_text(encoding="utf-8", errors="replace")
    def has(*parts: str) -> bool:
        return all(p.lower() in t.lower() for p in parts)
    return {
        "found": True,
        "path": str(path),
        "declares_stateful_service": has("stateful service", "authentication-service state"),
        "declares_outcomes": has("NoOutput", "Eligible", "Rejected", "Timeout"),
        "declares_non_age_only_features": has("retransmission", "replacement", "credential"),
        "declares_fragmented_vs_verifying_conceptual_pair": has("fragmented pending message", "verification stage"),
        "explicitly_says_pair_not_viability_claim": has("conceptual service-state counterexample", "does not assert that this particular pair has different viability membership"),
    }


def bounded_traces(spec: dict, horizon: int) -> dict[str, set[tuple[str, ...]]]:
    states = set(spec.get("states", {}))
    adj: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for e in spec.get("edges", []):
        src, dst = str(e["src"]), str(e["dst"])
        if src not in states or dst not in states:
            raise RuntimeError(f"R9I_BAD_EDGE={src}->{dst}")
        outcome = str(e.get("outcome", "NoOutput"))
        adj[src].append((dst, outcome))

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
            for dst, outcome in adj[s]:
                q.append((dst, tr + (outcome,), d + 1))
    return out


def common_predecessor_pairs(spec: dict) -> set[tuple[str, str]]:
    succ: dict[str, set[str]] = defaultdict(set)
    for e in spec.get("edges", []):
        succ[str(e["src"])].add(str(e["dst"]))
    pairs: set[tuple[str, str]] = set()
    for _, sset in succ.items():
        ss = sorted(sset)
        for i in range(len(ss)):
            for j in range(i + 1, len(ss)):
                pairs.add((ss[i], ss[j]))
    return pairs


def screen_pairs(spec: dict, horizon: int) -> tuple[list[dict], dict[str, set[tuple[str, ...]]]]:
    traces = bounded_traces(spec, horizon)
    common = common_predecessor_pairs(spec)
    states = spec.get("states", {})
    rows: list[dict] = []
    names = sorted(states)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            la = states[a].get("liveness_bound")
            lb = states[b].get("liveness_bound")
            same_liveness = la is not None and la == lb
            ta, tb = traces[a], traces[b]
            a_sub_b = ta.issubset(tb)
            b_sub_a = tb.issubset(ta)
            nondominated = (not a_sub_b) and (not b_sub_a)
            coreach = (a, b) in common or (b, a) in common
            rows.append({
                "state_a": a,
                "state_b": b,
                "liveness_a": la,
                "liveness_b": lb,
                "same_liveness": same_liveness,
                "common_predecessor_one_step": coreach,
                "trace_count_a": len(ta),
                "trace_count_b": len(tb),
                "a_trace_subset_b": a_sub_b,
                "b_trace_subset_a": b_sub_a,
                "trace_nondominated": nondominated,
                "screen_candidate": bool(same_liveness and coreach and nondominated),
            })
    return rows, traces


def default_template() -> dict:
    return {
        "schema": "SCV_R9I_SERVICE_AUTOMATON_INSTANTIATION_TEMPLATE_V1",
        "deployment_certified": False,
        "status": "TEMPLATE_REQUIRES_PROTOCOL_EVIDENCE",
        "instructions": [
            "Populate states and edges only from the declared implementation/service contract.",
            "Do not add rejection, timeout, retransmission, replacement, or credential branches merely to create separation.",
            "For a positive GFP candidate, preserve same physical/adopted information, same pending generation time/chi where required, same scalar liveness bound, and common-history co-reachability.",
            "Set deployment_certified=true only after every edge/liveness bound is supported by protocol or implementation evidence.",
        ],
        "states": {
            "PENDING_ENTRY": {"liveness_bound": None, "evidence": "TO_FILL"},
            "FRAGMENTED": {"liveness_bound": 2, "evidence": "manuscript conceptual example only"},
            "VERIFYING": {"liveness_bound": 2, "evidence": "manuscript conceptual example only"},
            "RETRANSMIT": {"liveness_bound": None, "evidence": "declared possible behavior; exact bound/edge TO_FILL"},
            "REJECTED": {"liveness_bound": None, "evidence": "declared observable outcome; exact edge TO_FILL"},
            "TIMEOUT": {"liveness_bound": None, "evidence": "declared observable outcome; exact edge TO_FILL"},
            "REPLACED": {"liveness_bound": None, "evidence": "declared possible behavior; exact edge TO_FILL"},
            "CREDENTIAL": {"liveness_bound": None, "evidence": "declared control-relevant dimension; exact edge TO_FILL"},
        },
        "edges": [],
        "screening_horizon": 4,
    }


def write_theorem_note(path: Path, hh: dict, op: dict) -> None:
    m = hh.get("metrics", {})
    text = r"""% Auto-generated R9-I structural theorem note.
\begin{proposition}[Conditional collapse of the fragmented/verifying pair]
Consider the current diagnostic service operator in which the fragmented state
has continuation $H_{\rm last}$ and the verifying state has continuation
$\max\{H_{\rm comp},H_{\rm last}\}$.  If
\[
H_{\rm comp}(z,u)\le H_{\rm last}(z,u)
\]
for every admissible augmented state $z$ and admissible control $u$, then the
two Bellman updates are identical and therefore their greatest fixed-point
thresholds coincide on their common admissible domain.
\end{proposition}

\noindent The proposition is an exact algebraic implication.  The present
R9-H-H computation verifies its premise on the sampled point-action audit but
does not by itself certify the premise over the entire continuous action/state
domain.  Therefore it must not be reported as a continuous-state separation
certificate or as an unconditional continuous-action impossibility theorem.
"""
    text += "\n%% Sampled evidence: branch_tests=%s, violations=%s, max_completion_minus_last_m=%s\n" % (
        m.get("branch_tests"),
        m.get("completion_branch_dominance_violations"),
        m.get("max_completion_minus_last_m"),
    )
    text += "%% Operator-source-structure-pass=%s\n" % op.get("pass")
    atomic_write(path, text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--automaton", type=str, default="")
    ap.add_argument("--horizon", type=int, default=4)
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.horizon < 1 or args.horizon > 12:
        raise ValueError("R9I_BAD_HORIZON")

    hh_path = RESULTS / "P3B1_R9HH_LATEST.json"
    if not hh_path.exists():
        raise RuntimeError("R9I_MISSING_R9HH_LATEST")
    hh = load_json(hh_path)
    if hh.get("status") != "PASS" or hh.get("classification") != EXPECTED_R9HH:
        raise RuntimeError(f"R9I_R9HH_GATE_FAIL={hh.get('classification')}")
    hm = hh.get("metrics", {})
    if int(hm.get("completion_branch_dominance_violations", -1)) != 0:
        raise RuntimeError("R9I_EXPECTED_ZERO_R9HH_BRANCH_VIOLATIONS")
    if int(hm.get("fixed_point_nonzero_stage_nodes", -1)) != 0:
        raise RuntimeError("R9I_EXPECTED_R9HH_FIXED_POINT_EQUALITY")

    hg_src = SRC / "p3b1_r9h_g_full_augmented_service_gfp.py"
    if not hg_src.exists():
        raise RuntimeError("R9I_MISSING_R9HG_SOURCE")
    op = operator_structure_audit(hg_src)
    if not op.get("pass"):
        raise RuntimeError(f"R9I_OPERATOR_STRUCTURE_GATE_FAIL={op}")

    manuscript = locate_manuscript()
    sem = manuscript_semantic_audit(manuscript)

    if args.automaton:
        auto_path = Path(args.automaton).expanduser().resolve()
    else:
        auto_path = CONFIG / "p3b1_r9i_service_automaton_candidates.json"
    if not auto_path.exists():
        atomic_write(auto_path, json.dumps(default_template(), indent=2, sort_keys=True) + "\n")
    spec = load_json(auto_path)
    rows, traces = screen_pairs(spec, int(args.horizon))
    screen_candidates = [r for r in rows if r["screen_candidate"]]
    deployment = bool(spec.get("deployment_certified", False))
    deployment_candidates = screen_candidates if deployment else []

    # Current pair status is deliberately split into what is exact and what is not.
    current_pair_sampled_structural = (
        int(hm.get("completion_branch_dominance_violations", -1)) == 0
        and int(hm.get("fixed_point_nonzero_stage_nodes", -1)) == 0
        and bool(op.get("pass"))
    )
    continuous_action_impossibility = False

    if deployment_candidates:
        classification = "DEPLOYMENT_CERTIFIED_NONDominated_SERVICE_PAIRS_FOUND_CURRENT_PAIR_COLLAPSE_DOES_NOT_CLOSE_GLOBAL_CONTINUOUS_GOAL"
        next_action = "R9J_BUILD_MATCHED_CHI_FULL_AUGMENTED_GFP_FOR_TOP_DEPLOYMENT_CERTIFIED_NONDominated_PAIR"
    elif screen_candidates:
        classification = "SEMANTIC_NONDominated_PAIR_CANDIDATES_EXIST_BUT_DEPLOYMENT_AUTOMATON_NOT_CERTIFIED"
        next_action = "R9J_CERTIFY_SERVICE_AUTOMATON_EDGES_AND_LIVENESS_BEFORE_NEW_GFP"
    else:
        classification = "CURRENT_PAIR_STRUCTURALLY_COLLAPSED_AND_NO_CERTIFIED_NONDominated_REPLACEMENT_PAIR_YET"
        next_action = "R9J_INSTANTIATE_DECLARED_SERVICE_AUTOMATON_FROM_PROTOCOL_EVIDENCE_THEN_SCREEN_NONDominated_PAIRS"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result = RESULTS / f"P3B1_R9I_RESULT_{stamp}.json"
    latest = RESULTS / "P3B1_R9I_LATEST.json"
    pair_csv = RESULTS / f"P3B1_R9I_PAIR_SCREEN_{stamp}.csv"
    trace_json = RESULTS / f"P3B1_R9I_TRACE_SETS_{stamp}.json"
    theorem_tex = RESULTS / f"P3B1_R9I_CURRENT_PAIR_THEOREM_{stamp}.tex"
    manifest = RESULTS / f"P3B1_R9I_MANIFEST_{stamp}.sha256"
    template_out = RESULTS / f"P3B1_R9I_SERVICE_AUTOMATON_TEMPLATE_{stamp}.json"

    write_csv(pair_csv, rows)
    trace_serial = {k: [list(x) for x in sorted(v)] for k, v in traces.items()}
    atomic_write(trace_json, json.dumps(trace_serial, indent=2, sort_keys=True) + "\n")
    atomic_write(template_out, json.dumps(default_template(), indent=2, sort_keys=True) + "\n")
    write_theorem_note(theorem_tex, hh, op)

    out = {
        "schema": "SCV_P3B1_R9I_STRUCTURAL_THEOREM_AND_PAIR_DISCOVERY_V1",
        "status": "PASS",
        "timestamp_utc": stamp,
        "classification": classification,
        "next_action": next_action,
        "continuous_state_separation_certified": False,
        "current_pair": {
            "pair": ["FRAGMENTED", "VERIFYING"],
            "sampled_point_action_structural_equivalence": current_pair_sampled_structural,
            "sampled_branch_tests": int(hm.get("branch_tests", 0)),
            "sampled_branch_violations": int(hm.get("completion_branch_dominance_violations", 0)),
            "fixed_point_nonzero_stage_nodes": int(hm.get("fixed_point_nonzero_stage_nodes", 0)),
            "continuous_action_structural_impossibility_certified": continuous_action_impossibility,
            "theorem_status": "EXACT_CONDITIONAL_THEOREM_WITH_SAMPLED_PREMISE_EVIDENCE_NOT_UNCONDITIONAL_CONTINUOUS_THEOREM",
        },
        "operator_structure": op,
        "manuscript_semantics": sem,
        "pair_discovery": {
            "automaton_path": str(auto_path),
            "deployment_certified": deployment,
            "screening_horizon": int(args.horizon),
            "pairs_screened": len(rows),
            "nondominated_screen_candidates": len(screen_candidates),
            "deployment_certified_nondominated_candidates": len(deployment_candidates),
            "candidate_pairs": screen_candidates,
        },
        "gates": {
            "upstream_r9hh_structural_collapse": True,
            "operator_source_structure": bool(op.get("pass")),
            "current_pair_sampled_structural_equivalence": current_pair_sampled_structural,
            "current_pair_continuous_action_impossibility_certified": False,
            "deployment_service_automaton_certified": deployment,
            "deployment_nondominated_replacement_pair_found": bool(deployment_candidates),
        },
        "artifacts": {
            "pair_screen_csv": str(pair_csv),
            "trace_sets_json": str(trace_json),
            "conditional_theorem_tex": str(theorem_tex),
            "service_automaton_template_json": str(template_out),
        },
    }
    text = json.dumps(out, indent=2, sort_keys=True) + "\n"
    atomic_write(result, text)
    atomic_write(latest, text)

    manifest_files = [Path(__file__), hh_path, hg_src, auto_path, result, pair_csv, trace_json, theorem_tex, template_out]
    atomic_write(manifest, "".join(f"{sha256_file(p)}  {p}\n" for p in manifest_files if p.exists()))

    print("=== P3-B1-R9-I STRUCTURAL THEOREM + SERVICE-PAIR DISCOVERY ===", flush=True)
    print("UPSTREAM_R9HH_STRUCTURAL_COLLAPSE=PASS", flush=True)
    print(f"R9I_OPERATOR_STRUCTURE_GATE={'PASS' if op.get('pass') else 'FAIL'}", flush=True)
    print(f"R9I_CURRENT_PAIR_SAMPLED_STRUCTURAL_EQUIVALENCE={'YES' if current_pair_sampled_structural else 'NO'}", flush=True)
    print("R9I_CURRENT_PAIR_CONTINUOUS_ACTION_IMPOSSIBILITY_CERTIFIED=NO", flush=True)
    print(f"R9I_MANUSCRIPT_SERVICE_SEMANTICS_FOUND={'YES' if sem.get('found') else 'NO'}", flush=True)
    print(f"R9I_AUTOMATON_DEPLOYMENT_CERTIFIED={'YES' if deployment else 'NO'}", flush=True)
    print(f"R9I_PAIR_SCREEN pairs={len(rows)} nondominated_candidates={len(screen_candidates)} deployment_certified_nondominated={len(deployment_candidates)}", flush=True)
    print("CONTINUOUS_STATE_SEPARATION_CERTIFIED=NO", flush=True)
    print("R9I_EXECUTION=PASS", flush=True)
    print(f"R9I_CLASSIFICATION={classification}", flush=True)
    print(f"R9I_NEXT_ACTION={next_action}", flush=True)
    print(f"RESULT_JSON={result}", flush=True)
    print(f"PAIR_SCREEN_CSV={pair_csv}", flush=True)
    print(f"CONDITIONAL_THEOREM_TEX={theorem_tex}", flush=True)
    print(f"SERVICE_AUTOMATON_TEMPLATE={template_out}", flush=True)
    print(f"MANIFEST={manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
