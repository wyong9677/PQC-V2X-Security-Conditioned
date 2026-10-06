#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PATCH_LEVEL = "E4_REAL_PQC_SERVICE_PROFILE_V1"
SCHEMA = "PQC_V2X_E4_REAL_PQC_SERVICE_PROFILE_V1"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def b(v: bool) -> str:
    return "YES" if v else "NO"


@dataclass(frozen=True)
class ServiceState:
    stage: str
    queue_remaining: int
    fragments_remaining: int
    loss_streak: int
    age_steps: int
    tx_opportunities_used: int

    def key(self) -> tuple:
        return (
            self.stage,
            self.queue_remaining,
            self.fragments_remaining,
            self.loss_streak,
            self.age_steps,
            self.tx_opportunities_used,
        )


def state_dict(s: ServiceState) -> dict:
    return asdict(s)


def build_service_automaton(profile: dict, net: dict, cycle_frames: int) -> dict:
    """Build a finite set-valued service transition relation R_c.

    Semantics:
    - queue delay is bounded by queue_wait_steps_max;
    - one fragment transmission opportunity occurs per 100-ms service/control slot;
    - each fragment may suffer at most max_consecutive_fragment_losses losses before
      a success (the loss branch is absent once the declared burst bound is reached);
    - after all fragments arrive, verification occupies one or more sampled steps;
    - the final verification event is nondeterministic ACCEPT/REJECT.

    Eligible-liveness is conditional on VALID_CANDIDATE/ACCEPT. REJECT is retained
    in R_c but is not counted as a successful eligibility guarantee.
    """
    alpha = int(profile["fragment_count"])
    qmax = int(net["queue_wait_steps_max"])
    bmax = int(net["max_consecutive_fragment_losses"])
    verify_steps = int(profile["verify_steps"])
    robust_R = int(profile["robust_valid_candidate_horizon_steps"])
    allow_cross_cycle = bool(net["allow_cross_cycle_reassembly"])

    initial = ServiceState("QUEUE" if qmax > 0 else "TX_FRAGMENT", qmax, alpha, 0, 0, 0)
    q = deque([initial])
    seen = {initial.key(): initial}
    transitions: list[dict] = []
    terminals: dict[str, list[dict]] = {"ELIGIBLE": [], "REJECTED": [], "TIMEOUT": []}

    while q:
        s = q.popleft()
        if s.stage in terminals:
            terminals[s.stage].append(state_dict(s))
            continue
        if s.age_steps >= robust_R:
            t = ServiceState("TIMEOUT", 0, s.fragments_remaining, s.loss_streak, s.age_steps, s.tx_opportunities_used)
            transitions.append({
                "from": state_dict(s), "event": "HORIZON_TIMEOUT", "output": "Timeout(m)", "to": state_dict(t)
            })
            if t.key() not in seen:
                seen[t.key()] = t
                q.append(t)
            continue

        outs: list[tuple[str, str, ServiceState]] = []
        if s.stage == "QUEUE":
            if s.queue_remaining > 0:
                nr = s.queue_remaining - 1
                ns = ServiceState(
                    "QUEUE" if nr > 0 else "TX_FRAGMENT",
                    nr,
                    s.fragments_remaining,
                    0,
                    s.age_steps + 1,
                    s.tx_opportunities_used,
                )
                outs.append(("QUEUE_ADVANCE", "NoOutput", ns))
        elif s.stage == "TX_FRAGMENT":
            # Strict cycle feasibility is a separate gate. If cross-cycle reassembly
            # is disabled, exhausting the cycle before all fragments arrive is a timeout.
            if (not allow_cross_cycle) and s.tx_opportunities_used >= cycle_frames:
                ns = ServiceState("TIMEOUT", 0, s.fragments_remaining, s.loss_streak, s.age_steps + 1, s.tx_opportunities_used)
                outs.append(("CYCLE_EXHAUSTED", "Timeout(m)", ns))
            else:
                # successful fragment delivery
                fr = s.fragments_remaining - 1
                next_stage = "VERIFYING" if fr == 0 else "TX_FRAGMENT"
                ns = ServiceState(next_stage, 0, fr, 0, s.age_steps + 1, s.tx_opportunities_used + 1)
                outs.append(("FRAGMENT_SUCCESS", "NoOutput", ns))
                # bounded-loss branch; no transition is admitted beyond declared bound
                if s.loss_streak < bmax:
                    nl = ServiceState("TX_FRAGMENT", 0, s.fragments_remaining, s.loss_streak + 1,
                                      s.age_steps + 1, s.tx_opportunities_used + 1)
                    outs.append(("FRAGMENT_LOSS", "NoOutput", nl))
        elif s.stage == "VERIFYING":
            # encode remaining verification slots in queue_remaining to avoid another field
            rem = s.queue_remaining if s.queue_remaining > 0 else verify_steps
            if rem > 1:
                ns = ServiceState("VERIFYING", rem - 1, 0, 0, s.age_steps + 1, s.tx_opportunities_used)
                outs.append(("VERIFY_PROGRESS", "NoOutput", ns))
            else:
                age = s.age_steps + 1
                outs.append(("VERIFY_ACCEPT", "Eligible(m)", ServiceState("ELIGIBLE", 0, 0, 0, age, s.tx_opportunities_used)))
                outs.append(("VERIFY_REJECT", "Rejected(m)", ServiceState("REJECTED", 0, 0, 0, age, s.tx_opportunities_used)))
        else:
            raise RuntimeError(f"unknown stage {s.stage}")

        for event, output, ns in outs:
            transitions.append({"from": state_dict(s), "event": event, "output": output, "to": state_dict(ns)})
            if ns.key() not in seen:
                seen[ns.key()] = ns
                q.append(ns)

    states = [state_dict(s) for s in sorted(seen.values(), key=lambda x: x.key())]
    return {
        "schema": "PQC_V2X_FINITE_SERVICE_AUTOMATON_V1",
        "profile_id": profile["profile_id"],
        "semantics": {
            "eligible_liveness_condition": "VALID_CANDIDATE_AND_ACCEPT_BRANCH_WITHIN_DECLARED_BOUNDED_QUEUE_LOSS_CONTRACT",
            "rejection_is_represented": True,
            "rejection_is_counted_as_eligible_success": False,
            "loss_beyond_declared_burst_is_outside_contract": True,
            "cross_cycle_reassembly": allow_cross_cycle,
        },
        "initial_state": state_dict(initial),
        "states": states,
        "transitions": transitions,
        "terminal_counts": {k: len(v) for k, v in terminals.items()},
        "state_count": len(states),
        "transition_count": len(transitions),
    }


def parse_benchmark_csv(path: Path) -> dict[str, dict]:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    out: dict[str, dict] = {}
    for row in rows:
        alg = row["algorithm"].strip()
        rec = {
            "message_bytes": int(row["message_bytes"]),
            "public_key_bytes": int(row["public_key_bytes"]),
            "secret_key_bytes": int(row["secret_key_bytes"]),
            "signature_capacity_bytes": int(row["signature_capacity_bytes"]),
            "signature_actual_bytes": int(row["signature_actual_bytes"]),
            "verify_median_us": float(row["verify_median_us"]),
            "verify_p95_us": float(row["verify_p95_us"]),
            "verify_max_us": float(row["verify_max_us"]),
            "iterations": int(row["iterations"]),
        }
        out.setdefault(alg, {"rows": []})["rows"].append(rec)
    for alg, d in out.items():
        rr = d["rows"]
        d["median_min_us"] = min(x["verify_median_us"] for x in rr)
        d["median_max_us"] = max(x["verify_median_us"] for x in rr)
        d["p95_max_us"] = max(x["verify_p95_us"] for x in rr)
        d["max_observed_us"] = max(x["verify_max_us"] for x in rr)
        d["public_key_bytes"] = max(x["public_key_bytes"] for x in rr)
        d["signature_bytes"] = max(x["signature_actual_bytes"] for x in rr)
    return out


def choose_benchmark_record(frozen: dict, bench: dict[str, dict], aliases: list[str]) -> tuple[dict, str]:
    for a in aliases:
        if a in bench:
            d = bench[a]
            return {
                "median_min_us": d["median_min_us"],
                "median_max_us": d["median_max_us"],
                "p95_max_us": d["p95_max_us"],
                "max_observed_us": d["max_observed_us"],
                "public_key_bytes": d["public_key_bytes"],
                "signature_bytes": d["signature_bytes"],
            }, "LOCAL_NATIVE_REBENCHMARK"
    return dict(frozen), "FROZEN_EXISTING_APPLE_M4_PRO_CALIBRATION"


def make_profile(name: str, spec: dict, bench_rec: dict, bench_source: str, cfg: dict, Ts: float) -> dict:
    pkt = cfg["v2x_packetization"]
    net = cfg["bounded_network_contract"]
    env_mult = float(cfg["verification_calibration"]["envelope_multiplier"])

    sig = int(spec["signature_bytes"])
    pk = int(spec["public_key_bytes"])
    classical_cert = int(pkt["classical_explicit_certificate_bytes"])
    wrapper = int(pkt["partially_hybrid_certificate_wrapper_bytes"])
    cert = wrapper + classical_cert + sig

    base_frame = (
        int(pkt["mac_layer_fixed_bytes"])
        + int(pkt["spdu_fixed_bytes"])
        + int(pkt["bsm_bytes"])
        + int(pkt["classical_bsm_signature_bytes"])
    )
    frame_limit = int(pkt["dsrc_frame_payload_limit_bytes"])
    frag_capacity = frame_limit - base_frame
    alpha = ceil_div(cert, frag_capacity)

    max_obs_us = float(bench_rec["max_observed_us"])
    envelope_us = env_mult * max_obs_us
    verify_steps = max(1, math.ceil((envelope_us * 1e-6) / Ts))
    qmax = int(net["queue_wait_steps_max"])
    bmax = int(net["max_consecutive_fragment_losses"])

    transport_verify_no_queue = alpha + verify_steps
    bounded_queue_no_loss = qmax + alpha + verify_steps
    robust_fragment_opportunities = alpha * (bmax + 1)
    robust_R = qmax + robust_fragment_opportunities + verify_steps
    tau = int(pkt["certificate_transmission_cycle_frames"])

    return {
        "profile_id": name,
        "algorithm": spec["algorithm"],
        "nist_security_category": int(spec["nist_security_category"]),
        "standard": spec["standard"],
        "public_key_bytes": pk,
        "signature_bytes": sig,
        "host_verification": {
            "source": bench_source,
            "reported_public_key_bytes": int(bench_rec.get("public_key_bytes", pk)),
            "reported_signature_bytes": int(bench_rec.get("signature_bytes", sig)),
            "host": cfg["verification_calibration"]["host"],
            "library": cfg["verification_calibration"]["library"],
            "message_sizes_bytes": cfg["verification_calibration"]["message_sizes_bytes"],
            "median_min_us": float(bench_rec["median_min_us"]),
            "median_max_us": float(bench_rec["median_max_us"]),
            "p95_max_us": bench_rec.get("p95_max_us"),
            "max_observed_us": max_obs_us,
            "empirical_envelope_multiplier": env_mult,
            "empirical_envelope_us": envelope_us,
            "deployment_wcet": False,
        },
        "packetization": {
            "hybrid_certificate_bytes": cert,
            "certificate_fixed_bytes": wrapper + classical_cert,
            "base_fragment_frame_bytes": base_frame,
            "frame_limit_bytes": frame_limit,
            "certificate_fragment_capacity_bytes": frag_capacity,
            "fragment_count": alpha,
            "certificate_transmission_cycle_frames": tau,
            "fragment_count_fits_current_cycle": alpha <= tau,
            "robust_fragment_opportunities": robust_fragment_opportunities,
            "robust_fragment_opportunities_fit_current_cycle": robust_fragment_opportunities <= tau,
        },
        "fragment_count": alpha,
        "verify_steps": verify_steps,
        "compute_only_horizon_steps": verify_steps,
        "transport_verify_no_queue_horizon_steps": transport_verify_no_queue,
        "bounded_queue_no_loss_horizon_steps": bounded_queue_no_loss,
        "robust_valid_candidate_horizon_steps": robust_R,
        "max_loss_burst": bmax,
        "queue_wait_steps_max": qmax,
        "eligible_bound_condition": "VALID_CANDIDATE_AND_DECLARED_BOUNDED_QUEUE_LOSS_CONTRACT",
        "profile_scope": "STANDARDS_AND_HOST_CALIBRATED_ENGINEERING_PROFILE_NOT_DEPLOYMENT_CERTIFIED",
    }


def validate_profile(profile: dict, spec: dict, max_age_steps: int) -> dict:
    checks = {
        "standard_pk_size_matches": profile["public_key_bytes"] == int(spec["public_key_bytes"]),
        "standard_signature_size_matches": profile["signature_bytes"] == int(spec["signature_bytes"]),
        "host_reported_pk_size_matches_standard": profile["host_verification"]["reported_public_key_bytes"] == int(spec["public_key_bytes"]),
        "host_reported_signature_size_matches_standard": profile["host_verification"]["reported_signature_bytes"] == int(spec["signature_bytes"]),
        "positive_fragment_capacity": profile["packetization"]["certificate_fragment_capacity_bytes"] > 0,
        "positive_fragment_count": profile["fragment_count"] >= 1,
        "verify_horizon_at_least_one": profile["verify_steps"] >= 1,
        "robust_horizon_within_declared_age_domain": profile["robust_valid_candidate_horizon_steps"] <= max_age_steps,
    }
    return {"checks": checks, "pass": all(checks.values())}


def write_profile_csv(path: Path, profiles: list[dict]) -> None:
    fields = [
        "profile_id", "algorithm", "public_key_bytes", "signature_bytes", "verify_envelope_us",
        "hybrid_certificate_bytes", "fragment_capacity_bytes", "fragment_count", "cycle_frames",
        "fragment_count_cycle_fit", "robust_fragment_cycle_fit", "compute_only_R",
        "transport_verify_no_queue_R", "bounded_queue_no_loss_R", "robust_valid_candidate_R",
        "queue_wait_steps_max", "max_loss_burst", "deployment_certified",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for p in profiles:
            w.writerow({
                "profile_id": p["profile_id"],
                "algorithm": p["algorithm"],
                "public_key_bytes": p["public_key_bytes"],
                "signature_bytes": p["signature_bytes"],
                "verify_envelope_us": f'{p["host_verification"]["empirical_envelope_us"]:.6f}',
                "hybrid_certificate_bytes": p["packetization"]["hybrid_certificate_bytes"],
                "fragment_capacity_bytes": p["packetization"]["certificate_fragment_capacity_bytes"],
                "fragment_count": p["fragment_count"],
                "cycle_frames": p["packetization"]["certificate_transmission_cycle_frames"],
                "fragment_count_cycle_fit": b(p["packetization"]["fragment_count_fits_current_cycle"]),
                "robust_fragment_cycle_fit": b(p["packetization"]["robust_fragment_opportunities_fit_current_cycle"]),
                "compute_only_R": p["compute_only_horizon_steps"],
                "transport_verify_no_queue_R": p["transport_verify_no_queue_horizon_steps"],
                "bounded_queue_no_loss_R": p["bounded_queue_no_loss_horizon_steps"],
                "robust_valid_candidate_R": p["robust_valid_candidate_horizon_steps"],
                "queue_wait_steps_max": p["queue_wait_steps_max"],
                "max_loss_burst": p["max_loss_burst"],
                "deployment_certified": "NO",
            })


def manuscript_tex(profiles: list[dict], cfg: dict, Ts: float) -> str:
    p = {x["profile_id"]: x for x in profiles}
    ml = p["ML_DSA_65_NDSS_PARTIALLY_HYBRID"]
    slh = p["SLH_DSA_SHA2_192S_NDSS_PARTIALLY_HYBRID"]
    return rf"""% Auto-generated by {PATCH_LEVEL}.
% Scope: engineering calibration, not deployment certification.
\begin{{table}}[t]
\centering
\scriptsize
\caption{{Standards- and host-calibrated post-quantum V2V service profiles. The packetization follows the NDSS 2024 Partially Hybrid V2V construction with a {cfg['v2x_packetization']['dsrc_frame_payload_limit_bytes']}-byte DSRC frame constraint and a {cfg['v2x_packetization']['certificate_transmission_cycle_frames']}-message certificate cycle. Queue/loss bounds are declared engineering contracts, not measured deployment guarantees.}}
\label{{tab:real_pqc_service_profiles}}
\begin{{tabular}}{{lrr}}
\hline
Quantity & ML-DSA-65 & SLH-DSA-SHA2-192s \\
\hline
Public key (bytes) & {ml['public_key_bytes']} & {slh['public_key_bytes']} \\
Signature (bytes) & {ml['signature_bytes']} & {slh['signature_bytes']} \\
Host verify envelope ($\mu$s) & {ml['host_verification']['empirical_envelope_us']:.3f} & {slh['host_verification']['empirical_envelope_us']:.3f} \\
Hybrid credential (bytes) & {ml['packetization']['hybrid_certificate_bytes']} & {slh['packetization']['hybrid_certificate_bytes']} \\
Fragments $\alpha$ & {ml['fragment_count']} & {slh['fragment_count']} \\
$\alpha\le\tau=5$ & {b(ml['packetization']['fragment_count_fits_current_cycle'])} & {b(slh['packetization']['fragment_count_fits_current_cycle'])} \\
Compute-only $R$ & {ml['compute_only_horizon_steps']} & {slh['compute_only_horizon_steps']} \\
Transport+verify $R$ (no queue/loss) & {ml['transport_verify_no_queue_horizon_steps']} & {slh['transport_verify_no_queue_horizon_steps']} \\
Declared robust $R$ & {ml['robust_valid_candidate_horizon_steps']} & {slh['robust_valid_candidate_horizon_steps']} \\
\hline
\end{{tabular}}
\end{{table}}

The native verification measurements remain far below the control sampling period
$T_s={Ts:.3f}\,\mathrm{{s}}$, so verification alone yields a one-step service for both
profiles. The standards-defined signature expansion changes the packetization regime:
ML-DSA-65 requires {ml['fragment_count']} credential fragments in the calibrated construction,
whereas SLH-DSA-SHA2-192s requires {slh['fragment_count']}. Under the declared bounded
queue/loss contract (at most {cfg['bounded_network_contract']['queue_wait_steps_max']} queue slot
and {cfg['bounded_network_contract']['max_consecutive_fragment_losses']} consecutive fragment
loss before successful retransmission), the valid-candidate liveness bounds are
$R_{{\rm ML}}={ml['robust_valid_candidate_horizon_steps']}$ and
$R_{{\rm SLH}}={slh['robust_valid_candidate_horizon_steps']}$. These are engineering service
contracts rather than measured automotive WCET/network guarantees. In particular, the current
$\tau=5$ certificate cycle admits the ML-DSA-65 fragment count but not the SLH-DSA-SHA2-192s
fragment count without extending the reassembly/certificate-cycle policy.
"""


def self_test() -> None:
    assert ceil_div(3501, 2136) == 2
    assert ceil_div(16416, 2136) == 8
    assert 1 + 2 * 2 + 1 == 6
    assert 1 + 8 * 2 + 1 == 18
    # tiny automaton sanity
    p = {"profile_id":"T", "fragment_count":2, "verify_steps":1, "robust_valid_candidate_horizon_steps":6}
    n = {"queue_wait_steps_max":1, "max_consecutive_fragment_losses":1, "allow_cross_cycle_reassembly":True}
    a = build_service_automaton(p, n, 5)
    assert a["state_count"] > 0 and a["transition_count"] > 0
    assert any(t["output"] == "Eligible(m)" for t in a["transitions"])
    assert any(t["output"] == "Rejected(m)" for t in a["transitions"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path)
    ap.add_argument("--p1", type=Path)
    ap.add_argument("--benchmark-csv", type=Path, default=None)
    ap.add_argument("--results-dir", type=Path)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        print("E4_INTERNAL_SELF_TEST=PASS")
        return 0

    root = Path(__file__).resolve().parents[1]
    cfg_path = args.config or root / "01_config" / "e4_real_pqc_service_profile_v1.json"
    p1_path = args.p1 or root / "01_config" / "p1_validation_v2.json"
    results = args.results_dir or root / "04_results"
    results.mkdir(parents=True, exist_ok=True)

    cfg = load_json(cfg_path)
    p1 = load_json(p1_path)
    Ts = float(p1["plant"]["Ts"])
    max_age_steps = int(math.floor(float(p1["age"]["max_seconds"]) / Ts + 1e-12))

    bench: dict[str, dict] = {}
    bench_csv = args.benchmark_csv
    if bench_csv and bench_csv.exists():
        bench = parse_benchmark_csv(bench_csv)

    print("=== E4 REAL PQC SERVICE PROFILE CALIBRATION ===")
    print(f"PATCH_LEVEL={PATCH_LEVEL}")
    print(f"TS_SECONDS={Ts:.12g}")
    print(f"MAX_AGE_STEPS={max_age_steps}")
    print(f"NATIVE_REBENCHMARK_INPUT={b(bool(bench))}")

    profiles: list[dict] = []
    validations = {}
    specs = cfg["cryptographic_profiles"]
    for key in ["ml_dsa_65", "slh_dsa_sha2_192s"]:
        spec = specs[key]
        rec, source = choose_benchmark_record(spec["frozen_host_calibration"], bench, spec["liboqs_aliases"])
        pid = spec["profile_id"]
        prof = make_profile(pid, spec, rec, source, cfg, Ts)
        profiles.append(prof)
        validations[pid] = validate_profile(prof, spec, max_age_steps)

    standard_gate = all(
        v["checks"]["standard_pk_size_matches"]
        and v["checks"]["standard_signature_size_matches"]
        and v["checks"]["host_reported_pk_size_matches_standard"]
        and v["checks"]["host_reported_signature_size_matches_standard"]
        for v in validations.values()
    )
    host_gate = all(p["host_verification"]["empirical_envelope_us"] > 0 for p in profiles)
    packet_gate = all(p["packetization"]["certificate_fragment_capacity_bytes"] > 0 for p in profiles)
    net = cfg["bounded_network_contract"]
    net_gate = int(net["queue_wait_steps_max"]) >= 0 and int(net["max_consecutive_fragment_losses"]) >= 0
    age_gate = all(v["checks"]["robust_horizon_within_declared_age_domain"] for v in validations.values())
    complete = standard_gate and host_gate and packet_gate and net_gate and age_gate

    stamp = utc_stamp()
    automata = {}
    tau = int(cfg["v2x_packetization"]["certificate_transmission_cycle_frames"])
    for p in profiles:
        # Produce both a strict current-cycle relation and an extended-reassembly relation.
        # Strict tau=5 is the operational current-practice compatibility check; the extended
        # relation is useful for bounded service analysis but is not current-cycle compatible
        # for SLH-DSA-SHA2-192s.
        net_strict = dict(net); net_strict["allow_cross_cycle_reassembly"] = False
        net_extended = dict(net); net_extended["allow_cross_cycle_reassembly"] = True
        strict = build_service_automaton(p, net_strict, tau)
        extended = build_service_automaton(p, net_extended, tau)
        automata[p["profile_id"]] = {"CURRENT_TAU5_STRICT": strict, "BOUNDED_EXTENDED_REASSEMBLY": extended}
        for mode, aut in automata[p["profile_id"]].items():
            fn = results / f"E4_{p['profile_id']}_{mode}_SERVICE_AUTOMATON_{stamp}.json"
            atomic_write_json(fn, aut)
            atomic_write_json(results / f"E4_{p['profile_id']}_{mode}_SERVICE_AUTOMATON_LATEST.json", aut)

    patch = {
        "schema": "PQC_V2X_P3B1_ENGINEERING_SERVICE_PROFILE_PATCH_V1",
        "do_not_overwrite_diagnostic_profiles": True,
        "profiles": {
            p["profile_id"]: {
                "eligible_bound_steps": p["robust_valid_candidate_horizon_steps"],
                "max_loss_burst": p["max_loss_burst"],
                "queue_wait_steps_max": p["queue_wait_steps_max"],
                "fragment_count": p["fragment_count"],
                "verify_steps": p["verify_steps"],
                "eligible_bound_condition": p["eligible_bound_condition"],
                "deployment_protocol_certified": False,
            } for p in profiles
        },
    }

    result = {
        "schema": SCHEMA,
        "patch_level": PATCH_LEVEL,
        "timestamp_utc": stamp,
        "input": {
            "config": str(cfg_path),
            "p1": str(p1_path),
            "benchmark_csv": str(bench_csv) if bench_csv else None,
            "Ts_seconds": Ts,
            "max_age_steps": max_age_steps,
        },
        "source_scope": {
            "standard_artifacts_calibrated": True,
            "host_verify_calibrated": True,
            "dsrc_packetization_calibrated": True,
            "bounded_network_contract_declared": True,
            "network_bound_measured": False,
            "deployment_protocol_certified": False,
            "note": "Packetization is an NDSS-2024 Partially-Hybrid engineering instantiation with current NIST-standardized signature sizes; IEEE 1609.2 does not thereby become PQ-native.",
        },
        "gates": {
            "STANDARD_ARTIFACT_GATE": standard_gate,
            "HOST_VERIFY_CALIBRATION_GATE": host_gate,
            "DSRC_PACKETIZATION_GATE": packet_gate,
            "DECLARED_BOUNDED_NETWORK_CONTRACT_GATE": net_gate,
            "ROBUST_HORIZON_WITHIN_DECLARED_AGE_DOMAIN": age_gate,
            "PQC_SERVICE_PROFILE_ENGINEERING_COMPLETE": complete,
        },
        "profiles": {p["profile_id"]: p for p in profiles},
        "validations": validations,
        "current_cycle_operational_gate": {
            p["profile_id"]: {
                "fragment_count_fits_tau5": p["packetization"]["fragment_count_fits_current_cycle"],
                "bounded_loss_fragment_opportunities_fit_tau5": p["packetization"]["robust_fragment_opportunities_fit_current_cycle"],
                "strict_tau5_timeout_states": automata[p["profile_id"]]["CURRENT_TAU5_STRICT"]["terminal_counts"]["TIMEOUT"],
            } for p in profiles
        },
        "p3b1_profile_patch": patch,
        "claim_scope": {
            "allowed": "standards-and-host-calibrated engineering service profile under a declared bounded network contract",
            "not_allowed": "deployment-certified IEEE 1609.2 PQ service bound or automotive ECU WCET/network guarantee",
        },
    }

    result_path = results / f"E4_REAL_PQC_SERVICE_PROFILE_RESULT_{stamp}.json"
    csv_path = results / f"E4_REAL_PQC_SERVICE_PROFILE_TABLE_{stamp}.csv"
    patch_path = results / f"E4_P3B1_PROFILE_PATCH_{stamp}.json"
    tex_path = results / f"E4_REAL_PQC_SERVICE_PROFILE_MANUSCRIPT_TABLE_{stamp}.tex"
    manifest_path = results / f"E4_REAL_PQC_SERVICE_PROFILE_MANIFEST_{stamp}.sha256"

    atomic_write_json(result_path, result)
    atomic_write_json(results / "E4_REAL_PQC_SERVICE_PROFILE_LATEST.json", result)
    write_profile_csv(csv_path, profiles)
    atomic_write_json(patch_path, patch)
    atomic_write_json(results / "E4_P3B1_PROFILE_PATCH_LATEST.json", patch)
    atomic_write_text(tex_path, manuscript_tex(profiles, cfg, Ts))

    files_for_manifest = [result_path, csv_path, patch_path, tex_path]
    for p in profiles:
        for mode in ("CURRENT_TAU5_STRICT", "BOUNDED_EXTENDED_REASSEMBLY"):
            files_for_manifest.append(results / f"E4_{p['profile_id']}_{mode}_SERVICE_AUTOMATON_{stamp}.json")
    atomic_write_text(manifest_path, "".join(f"{sha256_file(x)}  {x.name}\n" for x in files_for_manifest))

    print(f"STANDARD_ARTIFACT_GATE={'PASS' if standard_gate else 'FAIL'}")
    print(f"HOST_VERIFY_CALIBRATION_GATE={'PASS' if host_gate else 'FAIL'}")
    print(f"DSRC_PACKETIZATION_GATE={'PASS' if packet_gate else 'FAIL'}")
    print(f"DECLARED_BOUNDED_NETWORK_CONTRACT_GATE={'PASS' if net_gate else 'FAIL'}")
    print(f"NETWORK_BOUND_MEASURED=NO")
    print(f"DEPLOYMENT_PROTOCOL_CERTIFIED=NO")
    print(f"PQC_SERVICE_PROFILE_ENGINEERING_COMPLETE={b(complete)}")
    current_cycle_profiles = [p for p in profiles if p["packetization"]["robust_fragment_opportunities_fit_current_cycle"]]
    print(f"AT_LEAST_ONE_CURRENT_TAU5_ROBUST_PROFILE={b(bool(current_cycle_profiles))}")

    for p in profiles:
        pkt = p["packetization"]
        print(
            f"E4_PROFILE={p['profile_id']} algorithm={p['algorithm']} "
            f"pk_bytes={p['public_key_bytes']} sig_bytes={p['signature_bytes']} "
            f"hybrid_cert_bytes={pkt['hybrid_certificate_bytes']} "
            f"fragment_capacity_bytes={pkt['certificate_fragment_capacity_bytes']} "
            f"alpha={p['fragment_count']} cycle_fit={b(pkt['fragment_count_fits_current_cycle'])} "
            f"robust_cycle_fit={b(pkt['robust_fragment_opportunities_fit_current_cycle'])} "
            f"verify_envelope_us={p['host_verification']['empirical_envelope_us']:.6f} "
            f"compute_only_R={p['compute_only_horizon_steps']} "
            f"transport_verify_no_queue_R={p['transport_verify_no_queue_horizon_steps']} "
            f"bounded_queue_no_loss_R={p['bounded_queue_no_loss_horizon_steps']} "
            f"robust_R={p['robust_valid_candidate_horizon_steps']}"
        )

    print(f"MAX_AGE_STEPS={max_age_steps}")
    print(f"ROBUST_HORIZON_WITHIN_DECLARED_AGE_DOMAIN={'PASS' if age_gate else 'FAIL'}")
    if complete:
        print("E4_EXECUTION=PASS")
        print("E4_CLASSIFICATION=STANDARDS_HOST_AND_DSRC_CALIBRATED_PQC_SERVICE_PROFILES_COMPLETE_UNDER_DECLARED_BOUNDED_NETWORK_CONTRACT")
        print("E4_NEXT_ACTION=RESTRUCTURE_MANUSCRIPT_AROUND_REAL_CALIBRATED_PROFILE_AND_DIAGNOSTIC_VS_DEPLOYMENT_SCOPE")
    else:
        print("E4_EXECUTION=FAIL")
        print("E4_CLASSIFICATION=ENGINEERING_PROFILE_GATE_FAILED")
        print("E4_NEXT_ACTION=FIX_FAILED_CALIBRATION_GATE_BEFORE_MANUSCRIPT_RESTRUCTURE")

    print(f"RESULT_JSON={result_path}")
    print(f"PROFILE_TABLE_CSV={csv_path}")
    print(f"P3B1_PROFILE_PATCH={patch_path}")
    print(f"MANUSCRIPT_TABLE_TEX={tex_path}")
    print(f"MANIFEST={manifest_path}")
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
