from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.home() / "Desktop/paper set/PQC_V2X_Security_Conditioned/numerical_experiments"
RESULTS = ROOT / "04_results"
LOGS = ROOT / "07_logs"


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
                fields.append(k); seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def resolve_artifact(recorded: str | Path) -> Path:
    p = Path(recorded)
    if p.is_file():
        return p
    q = RESULTS / p.name
    if q.is_file():
        return q
    raise FileNotFoundError(f"R9HAR1_ARTIFACT_NOT_FOUND recorded={p} recovered={q}")


@dataclass(frozen=True)
class Edge:
    source: str
    event: str
    target: str
    observable: bool
    eligible_output: bool
    chi_rule: str
    pending_time_rule: str
    candidate_rule: str


def protocol_edges() -> list[Edge]:
    # A new pending message first enters PENDING_ENTRY. During one service step,
    # the *same candidate* can progress slowly to FRAGMENTED or faster to
    # VERIFYING. Thus the matched-chi pair is not merely algebraically defined;
    # both states are semantically co-reachable from one common history.
    return [
        Edge("PENDING_ENTRY", "service_progress_slow", "FRAGMENTED", True, False,
             "CARRY_CURRENT_CHI", "CARRY_CURRENT_PENDING_GENERATION_TIME", "CARRY_SAME_CANDIDATE"),
        Edge("PENDING_ENTRY", "service_progress_fast", "VERIFYING", True, False,
             "CARRY_CURRENT_CHI", "CARRY_CURRENT_PENDING_GENERATION_TIME", "CARRY_SAME_CANDIDATE"),
        Edge("FRAGMENTED", "fragment_delivered", "VERIFYING_LAST", True, False,
             "CARRY_CURRENT_CHI", "CARRY_CURRENT_PENDING_GENERATION_TIME", "CARRY_SAME_CANDIDATE"),
        Edge("VERIFYING", "verification_completes", "PENDING_ENTRY", True, True,
             "TERMINATE_CURRENT_CHI_AND_GENERATE_NEW_CHI", "RESET_PENDING_GENERATION_TIME_TO_COMPLETION_SAMPLE", "TERMINATE_AND_GENERATE_NEW_CANDIDATE"),
        Edge("VERIFYING", "verification_defers", "VERIFYING_LAST", True, False,
             "CARRY_CURRENT_CHI", "CARRY_CURRENT_PENDING_GENERATION_TIME", "CARRY_SAME_CANDIDATE"),
        Edge("VERIFYING_LAST", "verification_completes", "PENDING_ENTRY", True, True,
             "TERMINATE_CURRENT_CHI_AND_GENERATE_NEW_CHI", "RESET_PENDING_GENERATION_TIME_TO_COMPLETION_SAMPLE", "TERMINATE_AND_GENERATE_NEW_CANDIDATE"),
    ]


def reachable_nodes(edges: list[Edge], start: str) -> set[str]:
    adj: dict[str, list[str]] = {}
    for e in edges:
        adj.setdefault(e.source, []).append(e.target)
    seen = {start}
    q = deque([start])
    while q:
        x = q.popleft()
        for y in adj.get(x, []):
            if y not in seen:
                seen.add(y); q.append(y)
    return seen


def matched_pair_common_predecessor(edges: list[Edge]) -> bool:
    outs = [e for e in edges if e.source == "PENDING_ENTRY" and not e.eligible_output]
    targets = {e.target for e in outs}
    if not {"FRAGMENTED", "VERIFYING"}.issubset(targets):
        return False
    pair = [e for e in outs if e.target in {"FRAGMENTED", "VERIFYING"}]
    return all(
        e.chi_rule == "CARRY_CURRENT_CHI"
        and e.pending_time_rule == "CARRY_CURRENT_PENDING_GENERATION_TIME"
        and e.candidate_rule == "CARRY_SAME_CANDIDATE"
        for e in pair
    )


def worst_steps_to_completion(edges: list[Edge], start: str) -> int:
    # Evaluate the current-candidate graph only; eligible completion terminates
    # the current candidate and therefore ends this liveness calculation.
    out = {n: [] for n in {e.source for e in edges} | {e.target for e in edges}}
    for e in edges:
        out.setdefault(e.source, []).append(e)
    memo: dict[str, int] = {}
    visiting: set[str] = set()

    def rec(node: str) -> int:
        if node in memo:
            return memo[node]
        if node in visiting:
            raise RuntimeError(f"R9HAR1_CURRENT_CANDIDATE_CYCLE node={node}")
        visiting.add(node)
        vals = []
        for e in out.get(node, []):
            if e.eligible_output:
                vals.append(1)
            else:
                vals.append(1 + rec(e.target))
        visiting.remove(node)
        if not vals:
            raise RuntimeError(f"R9HAR1_NO_CURRENT_CANDIDATE_COMPLETION_PATH node={node}")
        memo[node] = max(vals)
        return memo[node]

    return rec(start)


def validate_graph(edges: list[Edge]) -> dict:
    nodes = sorted({e.source for e in edges} | {e.target for e in edges})
    required = {"PENDING_ENTRY", "FRAGMENTED", "VERIFYING", "VERIFYING_LAST"}
    out = {n: [e for e in edges if e.source == n] for n in nodes}
    no_dead_ends = all(out.get(n) for n in required)
    reach = reachable_nodes(edges, "PENDING_ENTRY")
    all_reachable = required.issubset(reach)
    completion = [e for e in edges if e.eligible_output]
    restart_closed = bool(completion) and all(e.target == "PENDING_ENTRY" for e in completion)
    carry_total = all(
        (e.eligible_output and e.candidate_rule == "TERMINATE_AND_GENERATE_NEW_CANDIDATE")
        or ((not e.eligible_output) and e.candidate_rule == "CARRY_SAME_CANDIDATE")
        for e in edges
    )
    chi_total = all(
        (e.eligible_output and e.chi_rule.startswith("TERMINATE_"))
        or ((not e.eligible_output) and e.chi_rule == "CARRY_CURRENT_CHI")
        for e in edges
    )
    time_total = all(
        (e.eligible_output and e.pending_time_rule.startswith("RESET_"))
        or ((not e.eligible_output) and e.pending_time_rule == "CARRY_CURRENT_PENDING_GENERATION_TIME")
        for e in edges
    )
    co_reachable = matched_pair_common_predecessor(edges)
    f_steps = worst_steps_to_completion(edges, "FRAGMENTED")
    v_steps = worst_steps_to_completion(edges, "VERIFYING")
    same_two_step_bound = (f_steps == 2 and v_steps == 2)
    return {
        "nodes": nodes,
        "edge_count": len(edges),
        "no_dead_ends": bool(no_dead_ends),
        "all_nodes_reachable_from_restart": bool(all_reachable),
        "restart_closed": bool(restart_closed),
        "candidate_update_total": bool(carry_total),
        "chi_update_total": bool(chi_total),
        "pending_time_update_total": bool(time_total),
        "matched_pair_common_predecessor": bool(co_reachable),
        "fragmented_worst_completion_steps": int(f_steps),
        "verifying_worst_completion_steps": int(v_steps),
        "same_two_step_liveness_bound": bool(same_two_step_bound),
        "pass": bool(no_dead_ends and all_reachable and restart_closed and carry_total and chi_total and time_total and co_reachable and same_two_step_bound),
    }


def detect_v1_unreachable_protocol(graph_csv: Path) -> dict:
    with graph_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    incoming_verify = [r for r in rows if r.get("target") == "VERIFYING"]
    nodes = sorted({r.get("source") for r in rows} | {r.get("target") for r in rows})
    return {
        "v1_nodes": nodes,
        "v1_incoming_to_verifying": len(incoming_verify),
        "false_reachability_gate_detected": len(incoming_verify) == 0,
    }


def self_test() -> None:
    g = validate_graph(protocol_edges())
    assert g["pass"]
    assert g["all_nodes_reachable_from_restart"]
    assert g["matched_pair_common_predecessor"]
    assert g["fragmented_worst_completion_steps"] == 2
    assert g["verifying_worst_completion_steps"] == 2
    print("R9HA_R1_INTERNAL_SELF_TEST=PASS")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test(); return 0

    upstream = RESULTS / "P3B1_R9HA_LATEST.json"
    d = json.loads(upstream.read_text(encoding="utf-8"))
    expected = "READY_FOR_R9H_B_SPARSE_AUGMENTED_GFP_AND_INTERVAL_INNER_OUTER"
    if d.get("status") != "PASS" or d.get("classification") != expected:
        raise RuntimeError(f"R9HAR1_UPSTREAM_R9HA_GATE_FAIL classification={d.get('classification')}")

    graph_v1 = resolve_artifact(d["artifacts"]["protocol_graph_csv"])
    preflight_v1 = resolve_artifact(d["artifacts"]["interval_preflight_csv"])
    defect = detect_v1_unreachable_protocol(graph_v1)
    if not defect["false_reachability_gate_detected"]:
        raise RuntimeError("R9HAR1_EXPECTED_V1_REACHABILITY_DEFECT_NOT_FOUND")

    with preflight_v1.open(newline="", encoding="utf-8") as f:
        pre_rows = list(csv.DictReader(f))
    if not pre_rows:
        raise RuntimeError("R9HAR1_EMPTY_INTERVAL_PREFLIGHT")
    monotone_fail = sum(str(r.get("endpoint_monotone_midpoint_gate", "")).lower() not in {"true", "1", "yes"} for r in pre_rows)
    lookup_fail = sum(int(float(r.get("hold_intersected_cells_product") or 0)) <= 0 or int(float(r.get("adopt_intersected_cells_product") or 0)) <= 0 for r in pre_rows)
    interval_gate = monotone_fail == 0 and lookup_fail == 0
    if not interval_gate:
        raise RuntimeError(f"R9HAR1_INTERVAL_PREFLIGHT_REGRESSION monotone_fail={monotone_fail} lookup_fail={lookup_fail}")

    edges = protocol_edges()
    gate = validate_graph(edges)
    if not gate["pass"]:
        raise RuntimeError("R9HAR1_REACHABLE_PROTOCOL_GATE_FAIL")

    # Copy the already validated interval preflight rows into a new authoritative
    # R1 artifact so later retirement of R9-H-A v1 does not break provenance.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_json = RESULTS / f"P3B1_R9HA_R1_RESULT_{stamp}.json"
    latest_json = RESULTS / "P3B1_R9HA_R1_LATEST.json"
    graph_csv = RESULTS / f"P3B1_R9HA_R1_REACHABLE_PROTOCOL_{stamp}.csv"
    preflight_csv = RESULTS / f"P3B1_R9HA_R1_INTERVAL_PREFLIGHT_COPY_{stamp}.csv"
    manifest = RESULTS / f"P3B1_R9HA_R1_MANIFEST_{stamp}.sha256"
    write_csv(graph_csv, [e.__dict__ for e in edges])
    write_csv(preflight_csv, pre_rows)

    classification = "READY_FOR_R9H_B_REACHABLE_PROTOCOLIZED_SPARSE_GFP_AND_INTERVAL_INNER_OUTER"
    next_action = "R9H_B_SOLVE_REACHABLE_PROTOCOLIZED_SPARSE_GFP_THEN_INTERVAL_ACTION_PAIRED_BOUNDS"
    out = {
        "schema": "SCV_P3B1_R9HA_R1_REACHABLE_PROTOCOL_PREFLIGHT_V1",
        "status": "PASS",
        "timestamp_utc": stamp,
        "classification": classification,
        "next_action": next_action,
        "continuous_state_separation_certified": False,
        "full_augmented_gfp_solved": False,
        "continuous_action_interval_certificate": False,
        "r9ha_v1_false_protocol_reachability_gate_detected": True,
        "v1_defect": defect,
        "protocol_graph": gate,
        "metrics": {
            "action_interval_width": float(d["metrics"]["action_interval_width"]),
            "action_intervals": int(d["metrics"]["action_intervals"]),
            "interval_tests": len(pre_rows),
            "interval_monotonicity_failures": int(monotone_fail),
            "interval_lookup_failures": int(lookup_fail),
            "upstream_max_matched_chi_gap_m": float(d["metrics"]["upstream_max_matched_chi_gap_m"]),
        },
        "gates": {
            "v1_reachability_defect_detected": True,
            "all_protocol_nodes_semantically_reachable": gate["all_nodes_reachable_from_restart"],
            "matched_pair_common_predecessor_same_candidate": gate["matched_pair_common_predecessor"],
            "same_pending_generation_time_on_common_branch": True,
            "same_chi_on_common_branch": True,
            "same_two_step_liveness_bound": gate["same_two_step_liveness_bound"],
            "restart_rule_total": gate["restart_closed"],
            "candidate_update_total": gate["candidate_update_total"],
            "chi_update_total": gate["chi_update_total"],
            "pending_time_update_total": gate["pending_time_update_total"],
            "interval_endpoint_monotonicity": monotone_fail == 0,
            "interval_lookup_coverage": lookup_fail == 0,
            "deployment_protocol_certified": False,
            "full_augmented_gfp_solved": False,
            "continuous_action_interval_certificate": False,
        },
        "artifacts": {
            "reachable_protocol_graph_csv": str(graph_csv),
            "interval_preflight_csv": str(preflight_csv),
        },
    }
    text = json.dumps(out, indent=2, sort_keys=True)
    atomic_write(result_json, text); atomic_write(latest_json, text)
    files = [Path(__file__), result_json, graph_csv, preflight_csv, upstream, graph_v1, preflight_v1]
    atomic_write(manifest, "".join(f"{sha256_file(p)}  {p}\n" for p in files if p.exists()))

    print("=== P3-B1-R9-H-A-R1 REACHABLE-PROTOCOL REAUDIT ===", flush=True)
    print("FALSE_PROTOCOL_REACHABILITY_GATE_DETECTED=YES", flush=True)
    print(f"R9HA_V1_VERIFYING_INCOMING_EDGES={defect['v1_incoming_to_verifying']}", flush=True)
    print("COMMON_PREDECESSOR=PENDING_ENTRY", flush=True)
    print("MATCHED_PAIR_CO_REACHABLE_FROM_SAME_CANDIDATE=YES", flush=True)
    print("MATCHED_PENDING_GENERATION_TIME=YES", flush=True)
    print("MATCHED_CHI_ENVELOPE=YES", flush=True)
    print(f"FRAGMENTED_WORST_COMPLETION_STEPS={gate['fragmented_worst_completion_steps']}", flush=True)
    print(f"VERIFYING_WORST_COMPLETION_STEPS={gate['verifying_worst_completion_steps']}", flush=True)
    print("SAME_TWO_STEP_LIVENESS_BOUND=YES", flush=True)
    print(f"R9HAR1_PROTOCOL nodes={len(gate['nodes'])} edges={gate['edge_count']} reachability=PASS closure=PASS", flush=True)
    print(f"R9HAR1_INTERVAL_REUSE tests={len(pre_rows)} monotone_fail={monotone_fail} lookup_fail={lookup_fail}", flush=True)
    print("R9HAR1_EXECUTION=PASS", flush=True)
    print(f"R9HAR1_CLASSIFICATION={classification}", flush=True)
    print("FULL_AUGMENTED_GFP_SOLVED=NO", flush=True)
    print("CONTINUOUS_STATE_SEPARATION_CERTIFIED=NO", flush=True)
    print(f"R9HAR1_NEXT_ACTION={next_action}", flush=True)
    print(f"RESULT_JSON={result_json}", flush=True)
    print(f"MANIFEST={manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
