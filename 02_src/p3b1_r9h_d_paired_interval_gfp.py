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

ROOT = Path.home() / "Desktop/paper set/PQC_V2X_Security_Conditioned/numerical_experiments"
SRC = ROOT / "02_src"
RESULTS = ROOT / "04_results"
TOL = 1.0e-10


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
                fields.append(k); seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows: w.writerow({k:r.get(k,"") for k in fields})


def resolve_artifact(recorded: str | Path) -> Path:
    p=Path(recorded)
    if p.is_file(): return p
    q=RESULTS/p.name
    if q.is_file(): return q
    raise FileNotFoundError(f"R9HD_ARTIFACT_NOT_FOUND recorded={p} recovered={q}")


def cell_corner_min(arr: np.ndarray) -> np.ndarray:
    out=np.asarray(arr)
    for axis in range(out.ndim):
        left=[slice(None)]*out.ndim; right=[slice(None)]*out.ndim
        left[axis]=slice(0,-1); right[axis]=slice(1,None)
        out=np.minimum(out[tuple(left)], out[tuple(right)])
    return out


def immediate_restart_lower(data, r3, b0, r9e, *, chunk_size=4096) -> np.ndarray:
    """Sound one-step lower bound for restart continuation on lookup nodes.

    Any infinite-horizon viable spacing must at least cover the unavoidable
    within-sample closing loss.  Since follower position is nondecreasing in the
    constant command, the infimum over the cooperative closure [-6,2.5] occurs at
    u=-6.  This routine deliberately does NOT call the result a full GFP lower
    bound; it is only the first monotone lower element used by the paired audit.
    """
    cfg,p1,p2b,p3a=data["eval_cfg"],data["p1"],data["p2b"],data["p3a"]
    # Update only semantically admitted evaluation nodes. Lookup-halo nodes are
    # kept at zero, which is conservative for a lower bound and avoids silently
    # treating the age=2.1 interpolation halo as an admitted service state.
    vf,vp,af,ba,bu,age=[np.asarray(x,float) for x in data["eval_flat"]]
    n=len(vf); Ts=float(data["Ts"]); J=float(cfg["information_contract"]["slew_rate"])
    nt=int(cfg["one_step"]["trajectory_points"]); times=np.linspace(0.0,Ts,nt)
    tau=float(p2b["follower"]["tau"]); w=float(p1["uncertainty"]["follower_actuation_abs"])
    sd=p2b["state_domain"]
    lips=0.5*(float(sd["v_f_max"])+float(sd["v_p_max"]))*(Ts/(nt-1))
    eval_lower=np.empty(n,dtype=float)
    for start in range(0,n,int(chunk_size)):
        stop=min(n,start+int(chunk_size)); sl=slice(start,stop)
        ap=b0.predecessor_acceleration_lower(age[sl],ba[sl],bu[sl],J,p3a,p2b)
        Pp,_,_,_,_=r3.predecessor_motion_with_stop(vp[sl],ap,bu[sl],age[sl],times,J,p3a)
        Pf,_,_,_=r9e.generalized_follower_motion(vf[sl],af[sl],-6.0,times,tau,w)
        closing=Pf-Pp
        # For a theorem lower bound we must not add the positive temporal-grid
        # Lipschitz correction used by the upper certificate.  Use sampled loss
        # itself minus the worst temporal interpolation amount and clip at zero.
        sampled=np.max(closing,axis=1)
        eval_lower[sl]=np.maximum(sampled-lips,0.0)
        if start==0 or stop==n or (stop//int(chunk_size))%32==0:
            print(f"R9HD_LOWER_RESTART_PROGRESS states={stop}/{n}",flush=True)
    lookup_nodes=int(np.prod(data["lookup_shape"]))
    out=np.zeros(lookup_nodes,dtype=float)
    out[np.asarray(data["eval_idx"],dtype=np.int64)]=eval_lower
    return out


def p2c_lower_candidate(st: dict, data: dict, b0, sw, resolution: int) -> float:
    cfg,p2b,p2c,p3a=data["eval_cfg"],data["p2b"],data["p2c"],data["p3a"]
    J=float(cfg["information_contract"]["slew_rate"])
    ap=b0.predecessor_acceleration_lower(np.asarray([st["age"]]),np.asarray([st["bar_a"]]),np.asarray([st["bar_u"]]),J,p3a,p2b)
    lo,hi,_=sw.switching_loss_bracket(np.asarray([st["v_f"]]),np.asarray([st["a_f"]]),np.asarray([st["v_p"]]),ap,int(p2c["switching"]["N_sw"]),[int(resolution)],p2c,p2b)
    lo=float(np.asarray(lo).reshape(-1)[0]); hi=float(np.asarray(hi).reshape(-1)[0])
    return max(lo,0.0), max(hi,0.0)


def lookup_min_over_descriptor(end_state: dict, env: dict, candidate_age: float,
                               lower_cells: np.ndarray, data: dict, r3, r9f) -> tuple[float,bool,int]:
    cells,valid=r9f.descriptor_cell_indices(
        data["lookup_axes"],
        (end_state["v_f"],end_state["v_p"],end_state["a_f"],candidate_age),
        env,
    )
    if not valid or len(cells)==0: return math.inf,False,0
    return float(np.min(lower_cells[cells])),True,int(len(cells))


def hold_lower(end_state: dict, lower_cells: np.ndarray, data: dict, r3) -> tuple[float,bool]:
    vals=[np.asarray([end_state[k]]) for k in ("v_f","v_p","a_f","bar_a","bar_u","age")]
    idx,valid=r3.locate_cells(data["lookup_axes"],vals)
    if not bool(valid[0]): return math.inf,False
    return float(lower_cells[int(idx[0])]),True


def lower_completion_future(*, end_state: dict, env: dict, candidate_age: float,
                            lower_cells: np.ndarray, data: dict, r3, r9f) -> dict:
    adopt,ok,n=lookup_min_over_descriptor(end_state,env,candidate_age,lower_cells,data,r3,r9f)
    hold,hok=hold_lower(end_state,lower_cells,data,r3)
    if not ok or not hok: return {"valid":False}
    return {"valid":True,"future_m":min(adopt,hold),"descriptor_cells":n}


def interval_relaxed_stage_lower(*, st: dict, env: dict, pending_age: float,
                                 intervals: list[tuple[float,float]], lower_cells: np.ndarray,
                                 data: dict, r3,b0,sw,r9e,r9f,p2c_resolution:int) -> dict:
    """Optimistic lower Bellman bound on the two-step matched-chi service tree.

    The action is relaxed interval-by-interval.  The lower calculation uses the
    safer action endpoint u_lo for immediate motion and the minimum continuation
    over descriptor cells.  This is intentionally optimistic and therefore can
    only lower the true requirement.  P2-C is inserted through its reported lower
    bracket; independent semantic validation remains a separate hard gate.
    """
    Ts=float(data["Ts"])
    fb_lo,fb_hi=p2c_lower_candidate(st,data,b0,sw,p2c_resolution)

    def verifying_last_lower(s1:dict,p_age:float)->float:
        fb1,_=p2c_lower_candidate(s1,data,b0,sw,p2c_resolution)
        best=math.inf
        for lo,hi in intervals:
            tr=__import__("p3b1_r9g_matched_chi_branching_service").physical_step(s1,float(lo),data,r3,b0,r9e)
            cf=lower_completion_future(end_state=tr["next_state"],env=env,candidate_age=float(p_age+Ts),lower_cells=lower_cells,data=data,r3=r3,r9f=r9f)
            if not cf.get("valid",False): continue
            req=max(0.0,tr["step_loss_m"],tr["closing_end_m"]+float(cf["future_m"]))
            best=min(best,req)
        return min(fb1,best) if math.isfinite(best) else math.inf

    best_frag=math.inf; best_verify=math.inf
    for lo,hi in intervals:
        tr=__import__("p3b1_r9g_matched_chi_branching_service").physical_step(st,float(lo),data,r3,b0,r9e)
        nxt=tr["next_state"]
        last=verifying_last_lower(nxt,float(pending_age+Ts))
        if not math.isfinite(last): continue
        reqf=max(0.0,tr["step_loss_m"],tr["closing_end_m"]+last)
        best_frag=min(best_frag,min(fb_lo,reqf))
        cf=lower_completion_future(end_state=nxt,env=env,candidate_age=float(pending_age+Ts),lower_cells=lower_cells,data=data,r3=r3,r9f=r9f)
        if not cf.get("valid",False): continue
        fut=max(float(cf["future_m"]),last)
        reqv=max(0.0,tr["step_loss_m"],tr["closing_end_m"]+fut)
        best_verify=min(best_verify,min(fb_lo,reqv))
    return {
        "valid":math.isfinite(best_frag) and math.isfinite(best_verify),
        "fragmented_lower_m":float(best_frag),
        "verifying_lower_m":float(best_verify),
        "p2c_lower_candidate_m":float(fb_lo),
        "p2c_upper_candidate_m":float(fb_hi),
    }


def self_test() -> None:
    x=np.arange(2*3*4,dtype=float).reshape(2,3,4)
    y=cell_corner_min(x)
    assert y.shape==(1,2,3)
    assert float(y[0,0,0])==0.0
    print("R9HD_INTERNAL_SELF_TEST=PASS")


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--self-test",action="store_true")
    ap.add_argument("--chunk-size",type=int,default=4096)
    ap.add_argument("--restart-action-width",type=float,default=0.25)
    ap.add_argument("--interval-width",type=float,default=0.03125)
    ap.add_argument("--p2c-resolution",type=int,default=4097)
    args=ap.parse_args()
    if args.self_test: self_test(); return 0
    if args.restart_action_width<=0 or args.interval_width<=0: raise ValueError("R9HD_BAD_ACTION_WIDTH")

    sys.path.insert(0,str(SRC))
    import p3b1_augmented_fixed_point_v1_r3 as r3
    import p3b0_freshness_service_audit_v1 as b0
    import p2c_switching_guard_v1 as sw
    import p3b1_r5_refinement_attribution as r5
    import p3b1_r6_continuation_halo as r6
    import p3b1_r7_frozen_halo_fixed_point as r7
    import p3b1_r9c_targeted_history_gate as r9c
    import p3b1_r9d_timestamped_service as r9d
    import p3b1_r9e_service_contract_general_action as r9e
    import p3b1_r9f_sparse_chi_envelope_graph as r9f
    import p3b1_r9g_matched_chi_branching_service as r9g
    import p3b1_r9h_b_protocolized_paired_bounds as r9hb
    import p3b1_r9h_c_theorem_gate_audit as r9hc

    hc=json.loads((RESULTS/"P3B1_R9HC_LATEST.json").read_text(encoding="utf-8"))
    exp="LOCAL_THEOREM_PREFLIGHTS_PASS_PAIRED_INTERVAL_GFP_REMAINS"
    if hc.get("status")!="PASS" or hc.get("classification")!=exp:
        raise RuntimeError(f"R9HD_R9HC_GATE_FAIL classification={hc.get('classification')}")
    gates=hc.get("gates",{})
    for key in ("action_interval_geometry_complete","p2c_bracket_directional_audit_pass","local_state_box_sign_stability_pass"):
        if not gates.get(key,False): raise RuntimeError(f"R9HD_UPSTREAM_GATE_FAIL={key}")

    hb=json.loads((RESULTS/"P3B1_R9HB_LATEST.json").read_text(encoding="utf-8"))
    rg=json.loads((RESULTS/"P3B1_R9G_LATEST.json").read_text(encoding="utf-8"))
    tests=r9hb.load_stage_tests(rg)
    data=r9d.reconstruct_timestamped_geometry(r3,r5,r6,r7,r9c)
    profile="diagnostic_fast"; R=r7.service_horizon(profile,data["p1"]["diagnostic_service_profiles"][profile])
    if R!=2: raise RuntimeError("R9HD_EXPECTED_R2")
    Ts=float(data["Ts"]); J=float(data["eval_cfg"]["information_contract"]["slew_rate"])

    print("=== P3-B1-R9-H-D PAIRED INTERVAL-GFP CERTIFICATE ATTEMPT ===",flush=True)
    print("UPSTREAM_R9HC_LOCAL_THEOREM_PREFLIGHTS=PASS",flush=True)
    print(f"MATCHED_CHI_TESTS={len(tests)}",flush=True)
    print(f"INTERVAL_WIDTH={args.interval_width:.12g}",flush=True)
    print(f"P2C_INDEPENDENT_REPLAY_RESOLUTION={args.p2c_resolution}",flush=True)
    print("P2C_LOWER_BOUND_SEMANTICS_CERTIFIED=NO",flush=True)
    print("FULL_AUGMENTED_INTERVAL_GFP_SOLVED=NO",flush=True)
    print("CONTINUOUS_STATE_SEPARATION_CERTIFIED=NO",flush=True)

    # Pessimistic / safe upper restart continuation: feasible point actions over
    # the whole plant range, cell-corner maxima, causal adopt/hold semantics.
    restart_actions=r9hb.refinement_actions(float(args.restart_action_width))
    print(f"R9HD_STAGE_START=upper_restart_continuation actions={len(restart_actions)}",flush=True)
    cfg_u,td_u=r9hb.full_action_restart_transitions(data,restart_actions,R,r3,b0,r9e,chunk_size=args.chunk_size)
    sol_u=r9d.solve_timestamped_causal(R,cfg_u,data["eval_idx"],data["lookup_shape"],data["lookup_fallback"],data["eval_fallback"],td_u,r3,label="r9hd_upper_restart")
    if not sol_u.converged: raise RuntimeError("R9HD_UPPER_RESTART_NOT_CONVERGED")
    upper_cells=r9g.build_restart_cell_values(sol_u.h_flat,data,R,r3)

    # Optimistic lower element.  It is sound as a one-step lower bound but is not
    # yet a full lower GFP; H-D reports that distinction explicitly.
    print("R9HD_STAGE_START=analytic_one_step_restart_lower",flush=True)
    lower_nodes=immediate_restart_lower(data,r3,b0,r9e,chunk_size=args.chunk_size)
    lower_grid=lower_nodes.reshape(data["lookup_shape"])
    lower_cells=cell_corner_min(lower_grid).reshape(-1)

    intervals=r9hc.action_bounds(float(args.interval_width))
    vf,vp,af,ba,bu,age=[np.asarray(x,float) for x in data["eval_flat"]]
    fallback_cache={}; rows=[]
    candidate_positive=0; min_candidate_margin=math.inf; max_candidate_margin=-math.inf
    for k,row in enumerate(tests,start=1):
        i=int(row["node_index"]); b=float(row["pending_age_s"])
        st={"v_f":float(vf[i]),"v_p":float(vp[i]),"a_f":float(af[i]),"bar_a":float(ba[i]),"bar_u":float(bu[i]),"age":float(age[i])}
        env=r9g.matched_pending_envelope(adopted_age=st["age"],pending_age=b,bar_a=st["bar_a"],bar_u=st["bar_u"],J=J,b0=b0,p3a=data["p3a"],p2b=data["p2b"])
        if not env.get("valid",False): raise RuntimeError(f"R9HD_MATCHED_ENV_INVALID node={i}")
        upper=r9g.matched_stage_pair(st=st,env=env,pending_age=b,actions=r9hb.refinement_actions(args.interval_width),restart_cells=upper_cells,data=data,r3=r3,b0=b0,sw=sw,r9e=r9e,r9f=r9f,fallback_cache=fallback_cache)
        lower=interval_relaxed_stage_lower(st=st,env=env,pending_age=b,intervals=intervals,lower_cells=lower_cells,data=data,r3=r3,b0=b0,sw=sw,r9e=r9e,r9f=r9f,p2c_resolution=args.p2c_resolution)
        if not upper.get("valid",False) or not lower.get("valid",False): raise RuntimeError(f"R9HD_PAIR_INVALID node={i} pending_age={b}")
        # For orientation observed upstream, a certificate would require
        # lower(VERIFYING) > upper(FRAGMENTED).
        margin=float(lower["verifying_lower_m"]-upper["fragmented_required_gap_m"])
        if margin>TOL: candidate_positive+=1
        min_candidate_margin=min(min_candidate_margin,margin); max_candidate_margin=max(max_candidate_margin,margin)
        rows.append({
            "node_index":i,"pending_age_s":b,
            "fragmented_upper_m":upper["fragmented_required_gap_m"],
            "verifying_upper_m":upper["verifying_required_gap_m"],
            "fragmented_lower_m":lower["fragmented_lower_m"],
            "verifying_lower_m":lower["verifying_lower_m"],
            "candidate_interval_separation_margin_m":margin,
            "candidate_positive":margin>TOL,
            "p2c_lower_candidate_m":lower["p2c_lower_candidate_m"],
            "p2c_upper_candidate_m":lower["p2c_upper_candidate_m"],
        })
        print(f"R9HD_PROGRESS tests={k}/{len(tests)} candidate_positive={candidate_positive} margin_m={margin:.12g}",flush=True)

    # Hard truth conditions.  The lower restart object above is only the first
    # monotone lower element, not the converged lower GFP; the P2-C lower result
    # still comes from the same bracket implementation.  Therefore H-D is an
    # honest certificate attempt and cannot emit YES yet.
    lower_restart_gfp_converged=False
    p2c_independent_lower_checker=False
    state_cell_interval_gfp=False
    full_augmented_interval_gfp=False
    continuous_yes=False

    if candidate_positive>0:
        classification="PAIRED_INTERVAL_SEPARATION_CANDIDATE_SURVIVES_ONE_STEP_LOWER_BOUND_FULL_LOWER_GFP_AND_INDEPENDENT_P2C_CHECKER_REMAIN"
        next_action="R9H_E_CONVERGE_OPTIMISTIC_LOWER_GFP_AND_BUILD_INDEPENDENT_ANALYTIC_P2C_LOWER_REPLAY"
    else:
        classification="ONE_STEP_LOWER_BOUND_TOO_WEAK_FOR_PAIRED_INTERVAL_SEPARATION"
        next_action="R9H_E_CONVERGE_OPTIMISTIC_LOWER_GFP_BEFORE_REASSESSING_CONTINUOUS_SEPARATION"

    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result=RESULTS/f"P3B1_R9HD_RESULT_{stamp}.json"; latest=RESULTS/"P3B1_R9HD_LATEST.json"
    csvp=RESULTS/f"P3B1_R9HD_PAIRED_INTERVAL_BOUNDS_{stamp}.csv"; manifest=RESULTS/f"P3B1_R9HD_MANIFEST_{stamp}.sha256"
    write_csv(csvp,rows)
    out={
        "schema":"SCV_P3B1_R9HD_PAIRED_INTERVAL_GFP_ATTEMPT_V1",
        "status":"PASS",
        "timestamp_utc":stamp,
        "classification":classification,
        "next_action":next_action,
        "continuous_state_separation_certified":continuous_yes,
        "full_augmented_interval_gfp_solved":full_augmented_interval_gfp,
        "metrics":{
            "matched_chi_tests":len(tests),"candidate_positive_tests":candidate_positive,
            "min_candidate_margin_m":None if not math.isfinite(min_candidate_margin) else min_candidate_margin,
            "max_candidate_margin_m":None if not math.isfinite(max_candidate_margin) else max_candidate_margin,
            "interval_width":args.interval_width,"action_intervals":len(intervals),
            "restart_upper_action_width":args.restart_action_width,"restart_upper_actions":len(restart_actions),
            "p2c_resolution":args.p2c_resolution,
            "upstream_state_box_min_abs_gap_m":hc.get("metrics",{}).get("state_box_min_abs_gap_m"),
            "upstream_p2c_max_finest_bracket_width_m":hc.get("metrics",{}).get("p2c_max_finest_bracket_width_m"),
        },
        "gates":{
            "upstream_r9hc_pass":True,
            "upper_restart_continuation_converged":bool(sol_u.converged),
            "analytic_one_step_restart_lower_constructed":True,
            "candidate_paired_margin_positive_somewhere":candidate_positive>0,
            "lower_restart_gfp_converged":lower_restart_gfp_converged,
            "independent_p2c_lower_checker_pass":p2c_independent_lower_checker,
            "continuous_state_cell_interval_gfp":state_cell_interval_gfp,
            "full_augmented_interval_gfp_solved":full_augmented_interval_gfp,
        },
        "artifacts":{"paired_interval_bounds_csv":str(csvp)},
    }
    text=json.dumps(out,indent=2,sort_keys=True); atomic_write(result,text); atomic_write(latest,text)
    mf=[Path(__file__),result,csvp,RESULTS/"P3B1_R9HC_LATEST.json",RESULTS/"P3B1_R9HB_LATEST.json",RESULTS/"P3B1_R9G_LATEST.json"]
    atomic_write(manifest,"".join(f"{sha256_file(p)}  {p}\n" for p in mf if p.exists()))

    print("=== R9-H-D DECISION ===",flush=True)
    print(f"R9HD_PAIRED_CANDIDATE positive_tests={candidate_positive}/{len(tests)} min_margin_m={min_candidate_margin:.12g} max_margin_m={max_candidate_margin:.12g}",flush=True)
    print("UPPER_RESTART_CONTINUATION_CONVERGED=YES",flush=True)
    print("OPTIMISTIC_LOWER_RESTART_ONE_STEP=YES",flush=True)
    print("OPTIMISTIC_LOWER_RESTART_GFP_CONVERGED=NO",flush=True)
    print("P2C_INDEPENDENT_LOWER_CHECKER=NO",flush=True)
    print("CONTINUOUS_STATE_CELL_INTERVAL_GFP=NO",flush=True)
    print("FULL_AUGMENTED_INTERVAL_GFP_SOLVED=NO",flush=True)
    print("R9HD_EXECUTION=PASS",flush=True)
    print(f"R9HD_CLASSIFICATION={classification}",flush=True)
    print("CONTINUOUS_STATE_SEPARATION_CERTIFIED=NO",flush=True)
    print(f"R9HD_NEXT_ACTION={next_action}",flush=True)
    print(f"RESULT_JSON={result}",flush=True); print(f"MANIFEST={manifest}",flush=True)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
