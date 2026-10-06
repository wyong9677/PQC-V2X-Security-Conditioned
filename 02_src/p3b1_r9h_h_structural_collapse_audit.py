from __future__ import annotations

import argparse
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
RESULTS = ROOT / "04_results"

ENTRY, FRAGMENTED, VERIFYING, VERIFYING_LAST = range(4)
TOL = 1.0e-10
EXPECTED_R9HG = "FULL_AUGMENTED_SERVICE_RESTART_GFP_COLLAPSES_PRIOR_LOCAL_STAGE_GAP"


def atomic_write(path: Path, text: str) -> None:
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
    fields=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def self_test() -> None:
    comp=np.asarray([1.0,2.0,3.0]); last=np.asarray([2.0,2.0,4.0])
    future=np.maximum(comp,last)
    assert np.allclose(future,last)
    assert int(np.count_nonzero(comp-last > 1e-12)) == 0
    print("R9HH_INTERNAL_SELF_TEST=PASS", flush=True)


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--chunk-size", type=int, default=4096)
    args=ap.parse_args()
    if args.self_test:
        self_test(); return 0

    sys.path.insert(0, str(SRC))
    import p3b1_augmented_fixed_point_v1_r3 as r3
    import p3b0_freshness_service_audit_v1 as b0
    import p3b1_r5_refinement_attribution as r5
    import p3b1_r6_continuation_halo as r6
    import p3b1_r7_frozen_halo_fixed_point as r7
    import p3b1_r9c_targeted_history_gate as r9c
    import p3b1_r9d_timestamped_service as r9d
    import p3b1_r9e_service_contract_general_action as r9e
    import p3b1_r9h_b_protocolized_paired_bounds as r9hb
    import p3b1_r9h_g_full_augmented_service_gfp as r9hg

    latest=RESULTS/"P3B1_R9HG_LATEST.json"
    if not latest.exists(): raise RuntimeError("R9HH_MISSING_R9HG_LATEST")
    hg=json.loads(latest.read_text(encoding="utf-8"))
    if hg.get("status") != "PASS" or hg.get("classification") != EXPECTED_R9HG:
        raise RuntimeError(f"R9HH_R9HG_GATE_FAIL={hg.get('classification')}")
    met=hg.get("metrics",{})
    if int(met.get("all_semantic_verify_more_demanding_nodes",-1)) != 0 or int(met.get("all_semantic_fragment_more_demanding_nodes",-1)) != 0:
        raise RuntimeError("R9HH_EXPECTED_GLOBAL_STAGE_COLLAPSE_NOT_PRESENT")

    width=float(met.get("point_action_width",0.5))
    print("=== P3-B1-R9-H-H STRUCTURAL COLLAPSE AUDIT ===", flush=True)
    print("UPSTREAM_R9HG_FULL_GFP_COLLAPSE=PASS", flush=True)
    print(f"POINT_ACTION_WIDTH={width:.12g}", flush=True)
    print("QUESTION=IS_FULL_GFP_STAGE_EQUALITY_EXPLAINED_BY_ACTIONWISE_BRANCH_DOMINANCE", flush=True)
    print("CONTINUOUS_STATE_SEPARATION_CERTIFIED=NO", flush=True)

    data=r9d.reconstruct_timestamped_geometry(r3,r5,r6,r7,r9c)
    Ts=float(data["Ts"])
    actions=r9hb.refinement_actions(width)
    cfg, transitions=r9hg.build_physical_transitions(data, actions, r3, b0, r9e, chunk_size=int(args.chunk_size))
    groups_v=r9hg.build_descriptor_range_groups(data, Ts, b0)
    groups_last=r9hg.build_descriptor_range_groups(data, 2.0*Ts, b0)
    age_v, ok_v=r9hg.scalar_cell_index(np.asarray(data["lookup_axes"][5],float),2.0*Ts)
    age_l, ok_l=r9hg.scalar_cell_index(np.asarray(data["lookup_axes"][5],float),3.0*Ts)
    if not ok_v or not ok_l: raise RuntimeError("R9HH_COMPLETION_AGE_LOOKUP_FAIL")

    print("R9HH_STAGE_START=recompute_upper_full_service_gfp", flush=True)
    upper=r9hg.solve_stage_fixed_point(
        data=data,cfg=cfg,transitions=transitions,
        fallback_eval=np.asarray(data["eval_fallback"],float),
        groups_v=groups_v,groups_last=groups_last,
        candidate_age_v_cell=age_v,candidate_age_last_cell=age_l,
        r3=r3,mode="upper",label="r9hh_upper_full_service",
    )
    if not upper.converged: raise RuntimeError("R9HH_UPPER_GFP_NOT_CONVERGED")

    lookup_shape=tuple(data["lookup_shape"])
    eval_idx=np.asarray(data["eval_idx"],dtype=np.int64)
    shaped=np.asarray(upper.h_flat,float).reshape(lookup_shape+(4,))
    cells=r3.cell_corner_max(shaped)
    cflat=[cells[...,s].reshape(-1) for s in range(4)]
    age=np.asarray(data["eval_flat"][5],float)
    valid=age >= Ts - 1e-10
    ue=np.asarray(upper.h_flat[eval_idx,:],float)
    fixed_diff=ue[:,VERIFYING]-ue[:,FRAGMENTED]
    max_fixed_abs=float(np.max(np.abs(fixed_diff[valid]))) if np.any(valid) else 0.0
    fixed_nonzero=int(np.count_nonzero(valid & (np.abs(fixed_diff)>TOL)))

    rows=[]
    total_positive=0
    total_tests=0
    max_comp_minus_last=-math.inf
    max_req_diff=-math.inf
    action_positive=0
    for tr in transitions:
        hold=np.asarray(tr["hold_index"],dtype=np.intp)
        entry_hold=cflat[ENTRY][hold]
        last_hold=cflat[VERIFYING_LAST][hold]
        adopt_v=r9hg.robust_adopt_future(cells[...,ENTRY],tr,groups_v,age_v)
        comp_v=np.minimum(adopt_v,entry_hold)
        delta=comp_v-last_hold
        step=np.asarray(tr["step_loss_upper"],float)
        close=np.asarray(tr["closing_end"],float)
        req_f=np.maximum(np.maximum(step,close+last_hold),0.0)
        future_v=np.maximum(comp_v,last_hold)
        req_v=np.maximum(np.maximum(step,close+future_v),0.0)
        req_diff=req_v-req_f
        dv=delta[valid]; rv=req_diff[valid]
        pos=int(np.count_nonzero(dv>TOL))
        reqpos=int(np.count_nonzero(rv>TOL))
        total_positive += pos
        total_tests += int(np.count_nonzero(valid))
        action_positive += int(pos>0)
        local_max=float(np.max(dv)) if len(dv) else -math.inf
        local_req=float(np.max(rv)) if len(rv) else -math.inf
        max_comp_minus_last=max(max_comp_minus_last,local_max)
        max_req_diff=max(max_req_diff,local_req)
        rows.append({
            "action":float(tr["action"]),
            "valid_nodes":int(np.count_nonzero(valid)),
            "completion_branch_exceeds_last_nodes":pos,
            "verify_requirement_exceeds_fragment_nodes":reqpos,
            "max_completion_minus_last_m":local_max,
            "max_verify_req_minus_fragment_req_m":local_req,
        })

    dominance = total_positive == 0 and max_comp_minus_last <= TOL and max_req_diff <= TOL
    if dominance and fixed_nonzero==0:
        classification="FULL_GFP_STAGE_COLLAPSE_EXPLAINED_BY_ACTIONWISE_VERIFY_COMPLETION_BRANCH_DOMINANCE"
        next_action="R9H_I_FREEZE_CURRENT_DIAGNOSTIC_CONTINUOUS_PROMOTION_AND_REVISE_CLAIM_SCOPE"
    else:
        classification="FULL_GFP_STAGE_COLLAPSE_NOT_EXPLAINED_BY_SIMPLE_BRANCH_DOMINANCE"
        next_action="R9H_I_REFINE_FULL_AUGMENTED_GFP_ACTION_GRID_BEFORE_SCOPE_CLOSURE"

    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result=RESULTS/f"P3B1_R9HH_RESULT_{stamp}.json"
    latest_out=RESULTS/"P3B1_R9HH_LATEST.json"
    csvp=RESULTS/f"P3B1_R9HH_BRANCH_DOMINANCE_{stamp}.csv"
    manifest=RESULTS/f"P3B1_R9HH_MANIFEST_{stamp}.sha256"
    write_csv(csvp,rows)
    out={
      "schema":"SCV_P3B1_R9HH_STRUCTURAL_COLLAPSE_AUDIT_V1",
      "status":"PASS",
      "timestamp_utc":stamp,
      "classification":classification,
      "next_action":next_action,
      "continuous_state_separation_certified":False,
      "metrics":{
        "point_action_width":width,
        "point_action_count":len(actions),
        "valid_semantic_compare_nodes":int(np.count_nonzero(valid)),
        "fixed_point_nonzero_stage_nodes":fixed_nonzero,
        "max_fixed_point_stage_abs_gap_m":max_fixed_abs,
        "branch_tests":total_tests,
        "actions_with_completion_branch_dominance_violation":action_positive,
        "completion_branch_dominance_violations":total_positive,
        "max_completion_minus_last_m":float(max_comp_minus_last),
        "max_verify_requirement_minus_fragment_requirement_m":float(max_req_diff),
      },
      "gates":{
        "upstream_r9hg_full_gfp_collapse":True,
        "upper_full_service_gfp_reconverged":True,
        "fragmented_verifying_fixed_point_equal_on_semantic_domain":fixed_nonzero==0,
        "verify_completion_branch_never_exceeds_last_branch_on_sampled_actions":dominance,
        "continuous_action_structural_dominance_certified":False,
      },
      "artifacts":{"branch_dominance_csv":str(csvp)},
    }
    text=json.dumps(out,indent=2,sort_keys=True)+"\n"
    atomic_write(result,text); atomic_write(latest_out,text)
    files=[Path(__file__),latest,result,csvp]
    atomic_write(manifest,"".join(f"{sha256_file(p)}  {p}\n" for p in files))

    print("=== R9-H-H DECISION ===", flush=True)
    print(f"R9HH_FIXED_POINT_EQUAL semantic_nonzero_nodes={fixed_nonzero} max_abs_gap_m={max_fixed_abs:.12g}", flush=True)
    print(f"R9HH_BRANCH_DOMINANCE tests={total_tests} violations={total_positive} actions_with_violation={action_positive} max_completion_minus_last_m={max_comp_minus_last:.12g}", flush=True)
    print(f"R9HH_REQUIREMENT_IDENTITY max_verify_minus_fragment_m={max_req_diff:.12g}", flush=True)
    print(f"R9HH_STRUCTURAL_DOMINANCE_GATE={'PASS' if dominance else 'FAIL'}", flush=True)
    print("CONTINUOUS_ACTION_STRUCTURAL_DOMINANCE_CERTIFIED=NO", flush=True)
    print("CONTINUOUS_STATE_SEPARATION_CERTIFIED=NO", flush=True)
    print("R9HH_EXECUTION=PASS", flush=True)
    print(f"R9HH_CLASSIFICATION={classification}", flush=True)
    print(f"R9HH_NEXT_ACTION={next_action}", flush=True)
    print(f"RESULT_JSON={result}", flush=True)
    print(f"MANIFEST={manifest}", flush=True)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
