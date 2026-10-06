#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import shlex
import shutil
import subprocess
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "PQC_V2X_E5_REAL_PROFILE_NUMERICAL_CLOSURE_V1"
TARGETED_BAR_A = [-2.4140625, -2.328125, -2.2421875, -2.15625, -2.0703125, -1.984375]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_json(path: Path, obj: Any) -> None:
    atomic_write(path, json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        atomic_write(path, "")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def run_checked(argv: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    cp = subprocess.run(
        argv,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if cp.returncode != 0:
        raise RuntimeError(
            "COMMAND_FAILED rc=%d\nARGV=%s\nOUTPUT=\n%s"
            % (cp.returncode, json.dumps(argv), cp.stdout)
        )
    return cp


def clean_default_pkg_env() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("PKG_CONFIG_PATH", None)
    env.pop("PKG_CONFIG_LIBDIR", None)
    env.pop("DYLD_LIBRARY_PATH", None)
    return env


def pkgconfig_paths(prefix: Path) -> str:
    candidates = [prefix / "lib" / "pkgconfig", prefix / "lib64" / "pkgconfig"]
    present = [str(p) for p in candidates if p.exists()]
    if not present:
        present = [str(candidates[0])]
    return os.pathsep.join(present)


def explicit_pkg_env(prefix: Path) -> dict[str, str]:
    env = clean_default_pkg_env()
    env["PKG_CONFIG_PATH"] = pkgconfig_paths(prefix)
    env["DYLD_LIBRARY_PATH"] = str(prefix / "lib")
    return env


def query_liboqs_version(prefix: Path | None = None) -> str:
    env = clean_default_pkg_env() if prefix is None else explicit_pkg_env(prefix)
    cp = run_checked(["pkg-config", "--modversion", "liboqs"], env=env)
    return cp.stdout.strip().splitlines()[-1].strip()


def build_native_bench(source: Path, out: Path, prefix: Path) -> dict[str, Any]:
    env = explicit_pkg_env(prefix)
    cflags = shlex.split(run_checked(["pkg-config", "--cflags", "liboqs"], env=env).stdout.strip())
    libs = shlex.split(run_checked(["pkg-config", "--libs", "liboqs"], env=env).stdout.strip())
    compiler = "/usr/bin/clang" if Path("/usr/bin/clang").exists() else (shutil.which("cc") or "cc")
    out.parent.mkdir(parents=True, exist_ok=True)
    argv = [compiler, "-O3", "-std=c11", *cflags, str(source), *libs]
    libdir = prefix / "lib"
    if libdir.exists():
        argv += [f"-Wl,-rpath,{libdir}"]
    argv += ["-o", str(out)]
    cp = run_checked(argv, env=env)
    return {"argv": argv, "output": cp.stdout, "sha256": sha256_file(out)}


def parse_kv_tokens(line: str) -> dict[str, str]:
    parts = line.strip().split()
    out: dict[str, str] = {}
    for part in parts[1:]:
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
    return out


def run_native_bench(binary: Path, prefix: Path, iterations: int, expected_version: str) -> dict[str, Any]:
    env = explicit_pkg_env(prefix)
    cp = run_checked([str(binary), "--iterations", str(iterations)], env=env)
    version = None
    rows: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    resolved: list[dict[str, Any]] = []
    for raw in cp.stdout.splitlines():
        line = raw.strip()
        if line.startswith("LIBOQS_VERSION="):
            version = line.split("=", 1)[1]
        elif line.startswith("BENCH_ROW "):
            kv = parse_kv_tokens(line)
            rows.append({
                "algorithm": kv["algorithm"],
                "liboqs_name": kv["liboqs_name"],
                "message_bytes": int(kv["msg"]),
                "iterations": int(kv["iterations"]),
                "median_us": float(kv["median_us"]),
                "p95_us": float(kv["p95_us"]),
                "max_us": float(kv["max_us"]),
            })
        elif line.startswith("ARTIFACT_ROW "):
            kv = parse_kv_tokens(line)
            artifacts.append({
                "algorithm": kv["algorithm"],
                "liboqs_name": kv["liboqs_name"],
                "public_key_bytes": int(kv["pk_bytes"]),
                "secret_key_bytes": int(kv["sk_bytes"]),
                "signature_bytes": int(kv["sig_bytes"]),
            })
        elif line.startswith("ALGORITHM_RESOLVED "):
            resolved.append(parse_kv_tokens(line))
    if version is None or not rows or not artifacts:
        raise RuntimeError("Native benchmark output is incomplete:\n" + cp.stdout)
    if version != expected_version:
        raise RuntimeError(f"Expected liboqs {expected_version}, native binary reported {version}")
    return {
        "version": version,
        "stdout": cp.stdout,
        "rows": rows,
        "artifacts": artifacts,
        "resolved": resolved,
    }


def summarize_bench(bench: dict[str, Any], ts_seconds: float) -> dict[str, Any]:
    by_alg: dict[str, list[dict[str, Any]]] = {}
    for row in bench["rows"]:
        by_alg.setdefault(row["algorithm"], []).append(row)
    summary: dict[str, Any] = {}
    for alg, rows in by_alg.items():
        max_obs = max(float(x["max_us"]) for x in rows)
        envelope = 1.25 * max_obs
        r_verify = max(1, math.ceil((envelope * 1e-6) / ts_seconds - 1e-15))
        summary[alg] = {
            "max_observed_us": max_obs,
            "empirical_envelope_us": envelope,
            "compute_only_R": r_verify,
            "median_min_us": min(float(x["median_us"]) for x in rows),
            "median_max_us": max(float(x["median_us"]) for x in rows),
            "p95_max_us": max(float(x["p95_us"]) for x in rows),
        }
    return summary


def provenance_audit(contract: dict[str, Any]) -> dict[str, Any]:
    p = contract["packetization"]
    profs = contract["pqc_profiles"]
    baseline = int(p["pure_ecdsa_first_frame_bytes"]) - int(p["mac_frame_overhead_bytes"]) - int(p["spdu_fixed_overhead_bytes"]) - int(p["ecdsa_certificate_bytes"])
    capacity = int(p["frame_limit_bytes"]) - int(p["mac_frame_overhead_bytes"]) - int(p["spdu_fixed_overhead_bytes"]) - baseline
    rows = []
    ok = baseline == int(p["derived_baseline_bsm_plus_signature_bytes"]) and capacity == int(p["derived_fragment_capacity_bytes"])
    for key, profile in profs.items():
        cert = int(p["hybrid_certificate_fixed_prefix_bytes"]) + int(p["hybrid_classical_credential_component_bytes"]) + int(profile["signature_bytes"])
        alpha = math.ceil(cert / capacity)
        rows.append({
            "profile_key": key,
            "algorithm": profile["display_name"],
            "hybrid_certificate_bytes": cert,
            "fragment_capacity_bytes": capacity,
            "fragment_count": alpha,
        })
        if cert <= 0 or alpha <= 0:
            ok = False
    return {
        "gate": ok,
        "derived_baseline_bsm_plus_signature_bytes": baseline,
        "derived_fragment_capacity_bytes": capacity,
        "profile_rows": rows,
        "direct_facts": {
            "frame_limit_bytes": p["frame_limit_bytes"],
            "mac_frame_overhead_bytes": p["mac_frame_overhead_bytes"],
            "spdu_fixed_overhead_bytes": p["spdu_fixed_overhead_bytes"],
            "ecdsa_certificate_bytes": p["ecdsa_certificate_bytes"],
            "pure_ecdsa_first_frame_bytes": p["pure_ecdsa_first_frame_bytes"],
            "tau_messages": p["tau_messages"],
        },
        "declared_assumptions": contract["network_contract"],
    }


def runtime_artifact_gate(contract: dict[str, Any], benches: dict[str, dict[str, Any]]) -> dict[str, Any]:
    expected = {
        v["display_name"]: (int(v["public_key_bytes"]), int(v["signature_bytes"]))
        for v in contract["pqc_profiles"].values()
    }
    issues = []
    for version, bench in benches.items():
        seen = {row["algorithm"]: row for row in bench["artifacts"]}
        for alg, (pk, sig) in expected.items():
            row = seen.get(alg)
            if row is None:
                issues.append(f"{version}:{alg}:missing")
            elif row["public_key_bytes"] != pk or row["signature_bytes"] != sig:
                issues.append(
                    f"{version}:{alg}:runtime(pk={row['public_key_bytes']},sig={row['signature_bytes']}) != standard(pk={pk},sig={sig})"
                )
    return {"gate": not issues, "issues": issues}


@dataclass(frozen=True)
class ServiceState:
    stage: str
    frag_remaining: int = 0
    loss_streak: int = 0
    tx_used: int = 0
    queue_remaining: int = 0
    verify_remaining: int = 0

    def label(self) -> str:
        if self.stage == "QUEUE":
            return f"QUEUE(q={self.queue_remaining})"
        if self.stage == "TX":
            return f"TX(f={self.frag_remaining},l={self.loss_streak},u={self.tx_used})"
        if self.stage == "VERIFY":
            return f"VERIFY(v={self.verify_remaining})"
        return self.stage


def entry_state(alpha: int, qmax: int, policy: str) -> ServiceState:
    if qmax > 0:
        return ServiceState("QUEUE", queue_remaining=qmax)
    if policy == "strict_tau5":
        return ServiceState("TX", frag_remaining=alpha, loss_streak=0, tx_used=0)
    return ServiceState("TX", frag_remaining=alpha, loss_streak=0)


def tx_start_state(alpha: int, policy: str) -> ServiceState:
    if policy == "strict_tau5":
        return ServiceState("TX", frag_remaining=alpha, loss_streak=0, tx_used=0)
    return ServiceState("TX", frag_remaining=alpha, loss_streak=0)


def service_successors(
    state: ServiceState,
    *,
    alpha: int,
    qmax: int,
    lmax: int,
    rverify: int,
    tau: int,
    policy: str,
    verification_mode: str,
) -> list[tuple[str, ServiceState]]:
    ent = entry_state(alpha, qmax, policy)
    if state.stage == "QUEUE":
        if state.queue_remaining > 1:
            return [("NoOutput", ServiceState("QUEUE", queue_remaining=state.queue_remaining - 1))]
        return [("NoOutput", tx_start_state(alpha, policy))]

    if state.stage == "TX":
        branches: list[tuple[str, ServiceState]] = []
        used_next = state.tx_used + 1 if policy == "strict_tau5" else 0
        if state.frag_remaining == 1:
            target = ServiceState("VERIFY", verify_remaining=max(1, rverify))
            branches.append(("NoOutput", target))
        else:
            if policy == "strict_tau5" and used_next >= tau:
                branches.append(("Timeout", ent))
            else:
                if policy == "strict_tau5":
                    target = ServiceState("TX", frag_remaining=state.frag_remaining - 1, loss_streak=0, tx_used=used_next)
                else:
                    target = ServiceState("TX", frag_remaining=state.frag_remaining - 1, loss_streak=0)
                branches.append(("NoOutput", target))

        if state.loss_streak < lmax:
            if policy == "strict_tau5" and used_next >= tau:
                branches.append(("Timeout", ent))
            else:
                if policy == "strict_tau5":
                    target = ServiceState("TX", frag_remaining=state.frag_remaining, loss_streak=state.loss_streak + 1, tx_used=used_next)
                else:
                    target = ServiceState("TX", frag_remaining=state.frag_remaining, loss_streak=state.loss_streak + 1)
                branches.append(("NoOutput", target))
        # De-duplicate branches that may coincide at the strict-cycle boundary.
        unique: list[tuple[str, ServiceState]] = []
        seen: set[tuple[str, ServiceState]] = set()
        for b in branches:
            if b not in seen:
                seen.add(b)
                unique.append(b)
        return unique

    if state.stage == "VERIFY":
        if state.verify_remaining > 1:
            return [("NoOutput", ServiceState("VERIFY", verify_remaining=state.verify_remaining - 1))]
        if verification_mode == "eligible_valid_message":
            return [("Eligible", ent)]
        if verification_mode == "general_verify_outcome":
            return [("Eligible", ent), ("Rejected", ent)]
        raise ValueError(f"unknown verification_mode={verification_mode}")

    raise ValueError(f"unknown service state {state}")


def build_service_automaton(
    *, alpha: int, qmax: int, lmax: int, rverify: int, tau: int,
    policy: str, verification_mode: str,
) -> dict[str, Any]:
    if policy not in {"strict_tau5", "extended_reassembly"}:
        raise ValueError(policy)
    start = entry_state(alpha, qmax, policy)
    noqueue = tx_start_state(alpha, policy)
    states: list[ServiceState] = []
    index: dict[ServiceState, int] = {}
    transitions: dict[int, list[tuple[str, int]]] = {}
    q = deque([start, noqueue])
    while q:
        s = q.popleft()
        if s in index:
            continue
        idx = len(states)
        index[s] = idx
        states.append(s)
        raw = service_successors(
            s, alpha=alpha, qmax=qmax, lmax=lmax, rverify=rverify,
            tau=tau, policy=policy, verification_mode=verification_mode,
        )
        for _, t in raw:
            if t not in index:
                q.append(t)
    for s, i in index.items():
        transitions[i] = []
        for out, t in service_successors(
            s, alpha=alpha, qmax=qmax, lmax=lmax, rverify=rverify,
            tau=tau, policy=policy, verification_mode=verification_mode,
        ):
            transitions[i].append((out, index[t]))
    return {
        "states": states,
        "labels": [s.label() for s in states],
        "transitions": transitions,
        "entry_index": index[start],
        "noqueue_index": index[noqueue],
        "parameters": {
            "alpha": alpha, "qmax": qmax, "lmax": lmax, "rverify": rverify,
            "tau": tau, "policy": policy, "verification_mode": verification_mode,
        },
    }


def worst_eligible_steps(automaton: dict[str, Any]) -> float:
    """Worst-case steps to an Eligible output; +inf if adversarial cycling prevents a finite bound."""
    n = len(automaton["states"])
    # Monotone dynamic programming. A finite acyclic-to-Eligible automaton settles quickly.
    v = np.zeros(n, dtype=float)
    finite = np.zeros(n, dtype=bool)
    for _ in range(max(10, n * 4)):
        changed = False
        new_v = v.copy()
        new_f = finite.copy()
        for i in range(n):
            branches = automaton["transitions"][i]
            vals = []
            all_finite = True
            for out, j in branches:
                if out == "Eligible":
                    vals.append(1.0)
                elif finite[j]:
                    vals.append(1.0 + v[j])
                else:
                    all_finite = False
                    break
            if all_finite and vals:
                nv = max(vals)
                if (not finite[i]) or abs(nv - v[i]) > 1e-12:
                    changed = True
                new_f[i] = True
                new_v[i] = nv
        v, finite = new_v, new_f
        if not changed:
            break
    idx = int(automaton["entry_index"])
    return float(v[idx]) if finite[idx] else math.inf


def cell_corner_max(nodal: np.ndarray) -> np.ndarray:
    out = nodal
    for axis in range(6):
        left = [slice(None)] * out.ndim
        right = [slice(None)] * out.ndim
        left[axis] = slice(0, -1)
        right[axis] = slice(1, None)
        out = np.maximum(out[tuple(left)], out[tuple(right)])
    return out


def future_lookup(cell_flat: list[np.ndarray], qidx: int, index: np.ndarray, valid: np.ndarray, n: int) -> np.ndarray:
    out = np.full(n, np.inf, dtype=float)
    out[valid] = cell_flat[qidx][index[valid]]
    return out


def solve_explicit_service_fixed_point(
    name: str,
    automaton: dict[str, Any],
    cfg: dict[str, Any],
    shape: tuple[int, ...],
    transition_data: dict[str, Any],
) -> dict[str, Any]:
    fallback_flat = np.asarray(transition_data["fallback_required"], dtype=float)
    fallback = fallback_flat.reshape(shape)
    nnode = fallback.size
    nsvc = len(automaton["states"])
    h = np.repeat(fallback[..., None], nsvc, axis=-1)
    tolerance = float(cfg["fixed_point"]["tolerance_m"])
    max_iterations = int(cfg["fixed_point"]["max_iterations"])
    history = []
    monotone_violation = 0.0

    for iteration in range(1, max_iterations + 1):
        cell = cell_corner_max(h)
        cell_flat = [cell[..., q].reshape(-1) for q in range(nsvc)]
        h_new = np.repeat(fallback[..., None], nsvc, axis=-1)

        for qi in range(nsvc):
            service_branches = automaton["transitions"][qi]
            best = np.full(nnode, np.inf, dtype=float)
            for trans in transition_data["transitions"]:
                branch_values = []
                for output, target_q in service_branches:
                    if output == "Eligible":
                        adopted = future_lookup(
                            cell_flat, target_q,
                            trans["completion_index"], trans["completion_valid"], nnode,
                        )
                        held = future_lookup(
                            cell_flat, target_q,
                            trans["defer_index"], trans["defer_valid"], nnode,
                        )
                        # Adoption is selected after the service output is observed.
                        future = np.minimum(adopted, held)
                    else:
                        future = future_lookup(
                            cell_flat, target_q,
                            trans["defer_index"], trans["defer_valid"], nnode,
                        )
                    branch_values.append(future)
                # Service uncertainty is adversarial after the physical action is selected.
                future_worst = np.maximum.reduce(branch_values)
                required = np.maximum(
                    trans["step_loss_upper"],
                    trans["closing_end"] + future_worst,
                )
                best = np.minimum(best, required)
            best = np.maximum(best, 0.0)
            candidate = np.minimum(fallback_flat, best)
            h_new[..., qi] = candidate.reshape(shape)

        h_new = np.minimum(h, h_new)
        violation = float(np.max(h_new - h))
        monotone_violation = max(monotone_violation, violation)
        delta = float(np.max(np.abs(h_new - h)))
        history.append({
            "iteration": iteration,
            "sup_change_m": delta,
            "mean_entry_required_gap_m": float(np.mean(h_new[..., int(automaton["entry_index"])])),
        })
        h = h_new
        if delta <= tolerance:
            break

    converged = history[-1]["sup_change_m"] <= tolerance
    q_span = np.ptp(h, axis=-1)
    q_tol = 1e-10
    return {
        "name": name,
        "h": h,
        "automaton": automaton,
        "history": history,
        "converged": bool(converged),
        "iterations": len(history),
        "final_change_m": float(history[-1]["sup_change_m"]),
        "monotone_violation_m": monotone_violation,
        "q_sensitive_nodes": int(np.count_nonzero(q_span > q_tol)),
        "max_q_span_m": float(np.max(q_span)),
        "entry_threshold": h[..., int(automaton["entry_index"])].copy(),
        "noqueue_threshold": h[..., int(automaton["noqueue_index"])].copy(),
    }


def compare_fields(a: np.ndarray, b: np.ndarray, tol: float = 1e-10) -> dict[str, Any]:
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    ad = np.abs(diff)
    return {
        "node_count": int(diff.size),
        "a_gt_b_count": int(np.count_nonzero(diff > tol)),
        "b_gt_a_count": int(np.count_nonzero(diff < -tol)),
        "equal_within_tol_count": int(np.count_nonzero(ad <= tol)),
        "max_abs_gap_m": float(np.max(ad)),
        "mean_abs_gap_m": float(np.mean(ad)),
        "p95_abs_gap_m": float(np.quantile(ad, 0.95)),
        "max_signed_a_minus_b_m": float(np.max(diff)),
        "min_signed_a_minus_b_m": float(np.min(diff)),
    }


def finite_result_summary(result: dict[str, Any]) -> dict[str, Any]:
    auto = result["automaton"]
    return {
        "name": result["name"],
        "service_state_count": len(auto["states"]),
        "service_states": auto["labels"],
        "entry_index": int(auto["entry_index"]),
        "noqueue_index": int(auto["noqueue_index"]),
        "worst_eligible_steps_from_worst_queue_entry": worst_eligible_steps(auto),
        "converged": result["converged"],
        "iterations": result["iterations"],
        "final_change_m": result["final_change_m"],
        "monotone_violation_m": result["monotone_violation_m"],
        "q_sensitive_nodes": result["q_sensitive_nodes"],
        "max_q_span_m": result["max_q_span_m"],
        "entry_threshold_mean_m": float(np.mean(result["entry_threshold"])),
        "entry_threshold_max_m": float(np.max(result["entry_threshold"])),
        "noqueue_threshold_mean_m": float(np.mean(result["noqueue_threshold"])),
    }


def load_upstream_solver(root: Path):
    source = root / "02_src" / "p3b1_augmented_fixed_point_v1_r3.py"
    if not source.exists():
        raise RuntimeError(f"Required upstream solver missing: {source}")
    spec = importlib.util.spec_from_file_location("e5_upstream_p3b1_r3", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load upstream solver")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(source.parent))
    spec.loader.exec_module(module)
    required = [
        "load_json", "static_interface_audit", "build_grid", "endpoint_semantics_audit",
        "build_transition_data", "solve_profile_fixed_point",
    ]
    missing = [x for x in required if not hasattr(module, x)]
    if missing:
        raise RuntimeError("Upstream solver API missing: " + ",".join(missing))
    return module, source


def targeted_cfg(base: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    vals = sorted(set(float(x) for x in cfg["grid"]["bar_a"]) | set(TARGETED_BAR_A))
    cfg["grid"]["bar_a"] = vals
    return cfg


def build_physical_engine(root: Path, geometry: str):
    up, source = load_upstream_solver(root)
    cfg = up.load_json(root / "01_config" / "p3b1_augmented_fixed_point_v1.json")
    p1 = up.load_json(root / "01_config" / "p1_validation_v2.json")
    p2b = up.load_json(root / "01_config" / "p2b_hybrid_fallback_v1.json")
    p2c = up.load_json(root / "01_config" / "p2c_switching_guard_v1.json")
    p3a = up.load_json(root / "01_config" / "p3a_information_contract_v1.json")
    if geometry == "targeted":
        cfg = targeted_cfg(cfg)
    elif geometry != "base":
        raise ValueError(geometry)
    up.static_interface_audit(cfg, p1, p2b, p2c, p3a)
    axes, flat, shape = up.build_grid(cfg)
    endpoint = up.endpoint_semantics_audit(cfg, p1, p2b, p3a, flat)
    if not endpoint["pass"]:
        raise RuntimeError("UPSTREAM_ENDPOINT_SEMANTICS_GATE=FAIL")
    transition = up.build_transition_data(cfg, p1, p2b, p2c, p3a, axes, flat)
    return {
        "upstream": up,
        "source": source,
        "cfg": cfg,
        "p1": p1,
        "p2b": p2b,
        "p2c": p2c,
        "p3a": p3a,
        "axes": axes,
        "flat": flat,
        "shape": shape,
        "transition": transition,
        "endpoint_audit": endpoint,
    }


def scalar_bridge(engine: dict[str, Any], r_ml: int, r_slh: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    up = engine["upstream"]
    cfg = engine["cfg"]
    shape = engine["shape"]
    td = engine["transition"]
    ml = up.solve_profile_fixed_point("E5_ML_SCALAR", r_ml, cfg, shape, td)
    slh = up.solve_profile_fixed_point("E5_SLH_SCALAR", r_slh, cfg, shape, td)
    tol = 1e-10
    ml_span = np.ptp(ml["h"], axis=-1)
    slh_span = np.ptp(slh["h"], axis=-1)
    cmp = compare_fields(ml["h"][..., -1], slh["h"][..., -1], tol)
    result = {
        "classification": "SCALAR_HORIZON_DIAGNOSTIC_ONLY",
        "not_full_service_automaton": True,
        "ML_DSA_65": {
            "R": r_ml, "converged": ml["converged"], "iterations": ml["iterations"],
            "q_sensitive_nodes": int(np.count_nonzero(ml_span > tol)),
            "max_q_span_m": float(np.max(ml_span)),
        },
        "SLH_DSA_SHA2_192S": {
            "R": r_slh, "converged": slh["converged"], "iterations": slh["iterations"],
            "q_sensitive_nodes": int(np.count_nonzero(slh_span > tol)),
            "max_q_span_m": float(np.max(slh_span)),
        },
        "initial_horizon_field_comparison": cmp,
    }
    rows = [
        {"mode": "scalar", "algorithm": "ML-DSA-65", **result["ML_DSA_65"]},
        {"mode": "scalar", "algorithm": "SLH-DSA-SHA2-192s", **result["SLH_DSA_SHA2_192S"]},
    ]
    return result, rows


def profile_parameters(contract: dict[str, Any], prov: dict[str, Any], bench_summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cap = int(prov["derived_fragment_capacity_bytes"])
    qmax = int(contract["network_contract"]["queue_bound_steps"])
    lmax = int(contract["network_contract"]["consecutive_loss_bound"])
    tau = int(contract["packetization"]["tau_messages"])
    prefix = int(contract["packetization"]["hybrid_certificate_fixed_prefix_bytes"])
    classical = int(contract["packetization"]["hybrid_classical_credential_component_bytes"])
    out = {}
    for key, p in contract["pqc_profiles"].items():
        alg = p["display_name"]
        cert = prefix + classical + int(p["signature_bytes"])
        alpha = math.ceil(cert / cap)
        rv = int(bench_summary[alg]["compute_only_R"])
        out[key] = {
            "algorithm": alg,
            "hybrid_certificate_bytes": cert,
            "fragment_capacity_bytes": cap,
            "alpha": alpha,
            "qmax": qmax,
            "lmax": lmax,
            "rverify": rv,
            "tau": tau,
            "robust_R_formula": qmax + alpha * (lmax + 1) + rv,
        }
    return out


def run_full_service_suite(engine: dict[str, Any], params: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cfg = engine["cfg"]
    shape = engine["shape"]
    td = engine["transition"]
    results: dict[tuple[str, str, str], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    policies = ["strict_tau5", "extended_reassembly"]
    modes = ["eligible_valid_message", "general_verify_outcome"]
    for mode in modes:
        for policy in policies:
            for key, p in params.items():
                auto = build_service_automaton(
                    alpha=int(p["alpha"]), qmax=int(p["qmax"]), lmax=int(p["lmax"]),
                    rverify=int(p["rverify"]), tau=int(p["tau"]), policy=policy,
                    verification_mode=mode,
                )
                name = f"{key}:{policy}:{mode}"
                solved = solve_explicit_service_fixed_point(name, auto, cfg, shape, td)
                results[(key, policy, mode)] = solved
                s = finite_result_summary(solved)
                rows.append({
                    "algorithm": p["algorithm"], "profile_key": key, "policy": policy,
                    "verification_mode": mode, **{k: v for k, v in s.items() if k not in {"service_states"}},
                    "robust_R_formula": p["robust_R_formula"],
                })

    comparisons: list[dict[str, Any]] = []
    for mode in modes:
        for policy in policies:
            ml = results[("ML_DSA_65", policy, mode)]
            slh = results[("SLH_DSA_SHA2_192S", policy, mode)]
            comp = compare_fields(ml["entry_threshold"], slh["entry_threshold"])
            comparisons.append({
                "comparison": "ML_minus_SLH_entry", "policy": policy,
                "verification_mode": mode, **comp,
            })
        for key in params:
            strict = results[(key, "strict_tau5", mode)]
            ext = results[(key, "extended_reassembly", mode)]
            comp = compare_fields(strict["entry_threshold"], ext["entry_threshold"])
            comparisons.append({
                "comparison": f"{key}_strict_minus_extended_entry",
                "policy": "strict_vs_extended", "verification_mode": mode, **comp,
            })

    summaries = {
        f"{key}|{policy}|{mode}": finite_result_summary(res)
        for (key, policy, mode), res in results.items()
    }
    all_converged = all(res["converged"] for res in results.values())
    all_monotone = all(res["monotone_violation_m"] <= 1e-12 for res in results.values())
    primary = next(
        x for x in comparisons
        if x["comparison"] == "ML_minus_SLH_entry"
        and x["policy"] == "extended_reassembly"
        and x["verification_mode"] == "eligible_valid_message"
    )
    difference = primary["max_abs_gap_m"] > 1e-10
    classification = (
        "REAL_PQC_SERVICE_AUTOMATA_INDUCE_FINITE_POINT_ACTION_VIABILITY_DIFFERENCE"
        if difference else
        "REAL_PQC_SERVICE_AUTOMATA_DIFFER_BUT_ARE_NONBINDING_FOR_FINITE_POINT_ACTION_VIABILITY_ON_TESTED_DOMAIN"
    )
    return {
        "gate": bool(all_converged and all_monotone),
        "classification": classification,
        "continuous_state_separation_certified": False,
        "continuous_action_interval_gfp_solved": False,
        "deployment_protocol_certified": False,
        "profile_summaries": summaries,
        "comparisons": comparisons,
        "primary_real_profile_comparison": primary,
    }, rows + comparisons


def self_tests(contract: dict[str, Any]) -> None:
    prov = provenance_audit(contract)
    assert prov["gate"]
    assert prov["derived_baseline_bsm_plus_signature_bytes"] == 104
    assert prov["derived_fragment_capacity_bytes"] == 2136
    by_alg = {x["algorithm"]: x for x in prov["profile_rows"]}
    assert by_alg["ML-DSA-65"]["hybrid_certificate_bytes"] == 3501
    assert by_alg["ML-DSA-65"]["fragment_count"] == 2
    assert by_alg["SLH-DSA-SHA2-192s"]["hybrid_certificate_bytes"] == 16416
    assert by_alg["SLH-DSA-SHA2-192s"]["fragment_count"] == 8

    ml_ext = build_service_automaton(alpha=2, qmax=1, lmax=1, rverify=1, tau=5,
                                     policy="extended_reassembly", verification_mode="eligible_valid_message")
    slh_ext = build_service_automaton(alpha=8, qmax=1, lmax=1, rverify=1, tau=5,
                                      policy="extended_reassembly", verification_mode="eligible_valid_message")
    ml_strict = build_service_automaton(alpha=2, qmax=1, lmax=1, rverify=1, tau=5,
                                        policy="strict_tau5", verification_mode="eligible_valid_message")
    slh_strict = build_service_automaton(alpha=8, qmax=1, lmax=1, rverify=1, tau=5,
                                         policy="strict_tau5", verification_mode="eligible_valid_message")
    assert worst_eligible_steps(ml_ext) == 6.0
    assert worst_eligible_steps(slh_ext) == 18.0
    assert worst_eligible_steps(ml_strict) == 6.0
    assert math.isinf(worst_eligible_steps(slh_strict))

    # Synthetic six-dimensional Bellman smoke test.
    shape = (2, 2, 2, 2, 2, 2)
    n = int(np.prod(shape))
    fallback = np.full(n, 10.0)
    cell_index = np.zeros(n, dtype=int)
    valid = np.ones(n, dtype=bool)
    transition_data = {
        "fallback_required": fallback,
        "transitions": [{
            "action": -1.0,
            "step_loss_upper": np.ones(n),
            "closing_end": np.zeros(n),
            "defer_index": cell_index,
            "defer_valid": valid,
            "completion_index": cell_index,
            "completion_valid": valid,
        }],
    }
    cfg = {"fixed_point": {"tolerance_m": 1e-8, "max_iterations": 20}}
    smoke = solve_explicit_service_fixed_point("smoke", ml_ext, cfg, shape, transition_data)
    assert smoke["converged"]
    assert smoke["monotone_violation_m"] <= 1e-12
    print("E5_INTERNAL_SELF_TEST=PASS")


def manuscript_handoff(contract: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "E5_POST_RESULT_MANUSCRIPT_HANDOFF_V1",
        "manuscript_mutation_performed": False,
        "theory_items_after_numerical_closeout": contract["post_result_manuscript_obligations"],
        "literature_bridge": contract["literature_bridge"],
        "numerical_classification": result.get("full_service", {}).get("classification"),
        "liboqs_version_robustness": result.get("verify_depth_robustness_gate"),
        "packetization_provenance_gate": result.get("packetization_provenance_gate"),
        "next_action": "Update theorem/Related Work/title/finite-gap interpretation only after reviewing the E5 numerical evidence ledger."
    }


def render_tex_evidence(result: dict[str, Any]) -> str:
    b15 = result["bench_summary"]["0.15.0"]
    b16 = result["bench_summary"]["0.16.0"]
    fs = result["full_service"]
    primary = fs["primary_real_profile_comparison"]
    return rf"""% E5 numerical evidence only; not an automatic manuscript patch.
\begin{{table}}[!t]
\centering
\caption{{Cross-version native verification and real-profile service-depth robustness.}}
\label{{tab:e5_cross_version_pqc}}
\begin{{tabular}}{{lrrrr}}
\toprule
Profile & liboqs & Envelope ($\mu$s) & $R_{{\rm ver}}$ & $R^{{\rm rob}}$ \\
\midrule
ML-DSA-65 & 0.15.0 & {b15['ML-DSA-65']['empirical_envelope_us']:.3f} & {b15['ML-DSA-65']['compute_only_R']} & {result['profile_parameters']['ML_DSA_65']['robust_R_formula']} \\
ML-DSA-65 & 0.16.0 & {b16['ML-DSA-65']['empirical_envelope_us']:.3f} & {b16['ML-DSA-65']['compute_only_R']} & {result['profile_parameters']['ML_DSA_65']['robust_R_formula']} \\
SLH-DSA-SHA2-192s & 0.15.0 & {b15['SLH-DSA-SHA2-192s']['empirical_envelope_us']:.3f} & {b15['SLH-DSA-SHA2-192s']['compute_only_R']} & {result['profile_parameters']['SLH_DSA_SHA2_192S']['robust_R_formula']} \\
SLH-DSA-SHA2-192s & 0.16.0 & {b16['SLH-DSA-SHA2-192s']['empirical_envelope_us']:.3f} & {b16['SLH-DSA-SHA2-192s']['compute_only_R']} & {result['profile_parameters']['SLH_DSA_SHA2_192S']['robust_R_formula']} \\
\bottomrule
\end{{tabular}}
\end{{table}}

% Primary full-service point-action comparison:
% policy = extended_reassembly; verification semantics = eligible_valid_message.
% max absolute ML-vs-SLH entry-threshold gap = {primary['max_abs_gap_m']:.12g} m.
% classification = {fs['classification']}.
% No continuous-domain maximal-kernel claim is implied.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", default=None)
    ap.add_argument("--root", default=os.environ.get(
        "E5_ROOT",
        str(Path.home() / "Desktop" / "paper set" / "PQC_V2X_Security_Conditioned" / "numerical_experiments"),
    ))
    ap.add_argument("--bench-iters", type=int, default=int(os.environ.get("E5_BENCH_ITERS", "10000")))
    ap.add_argument("--geometry", choices=["base", "targeted"], default=os.environ.get("E5_GEOMETRY", "targeted"))
    ap.add_argument("--self-test-only", action="store_true")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    contract_path = Path(args.contract).expanduser().resolve() if args.contract else here / "E5_PROVENANCE_CONTRACT.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    self_tests(contract)
    if args.self_test_only:
        return 0

    if args.bench_iters < 100:
        raise RuntimeError("E5_BENCH_ITERS must be >= 100 for a full run")

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise RuntimeError(f"Workspace not found: {root}")
    results_dir = root / "04_results"
    build_dir = root / "03_build"
    logs_dir = root / "07_logs"
    results_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp()

    print("=== E5 REAL PQC PROFILE NUMERICAL CLOSURE ===")
    print(f"E5_ROOT={root}")
    print(f"PYTHON={sys.executable}")
    print(f"PYTHON_VERSION={sys.version.split()[0]}")
    print(f"NUMPY_VERSION={np.__version__}")
    print(f"GEOMETRY={args.geometry}")
    print(f"BENCH_ITERS={args.bench_iters}")
    print("VIRTUAL_ENV_CREATED=NO")

    prov = provenance_audit(contract)
    if not prov["gate"]:
        raise RuntimeError("E5_PACKETIZATION_PROVENANCE_GATE=FAIL")
    print("E5_PACKETIZATION_PROVENANCE_GATE=PASS")
    print(f"E5_DERIVED_BASELINE_BSM_PLUS_SIGNATURE_BYTES={prov['derived_baseline_bsm_plus_signature_bytes']}")
    print(f"E5_FRAGMENT_CAPACITY_BYTES={prov['derived_fragment_capacity_bytes']}")

    oqs = contract["liboqs"]
    legacy_prefix = Path(os.environ.get("E5_LIBOQS_015_PREFIX", oqs["legacy_prefix_default"])).expanduser().resolve()
    research_prefix = Path(os.environ.get("E5_LIBOQS_016_PREFIX", oqs["research_prefix_default"])).expanduser().resolve()
    default_before = query_liboqs_version(None)
    v15 = query_liboqs_version(legacy_prefix)
    v16 = query_liboqs_version(research_prefix)
    if default_before != oqs["legacy_expected_version"]:
        raise RuntimeError(f"Default liboqs changed: expected {oqs['legacy_expected_version']}, got {default_before}")
    if v15 != oqs["legacy_expected_version"]:
        raise RuntimeError(f"Explicit 0.15 prefix mismatch: {v15}")
    if v16 != oqs["research_expected_version"]:
        raise RuntimeError(f"Explicit 0.16 prefix mismatch: {v16}")
    print("E5_PLATFORM_ISOLATION_GATE=PASS")

    source = root / "02_src" / "e5_pqc_verify_bench.c"
    if not source.exists():
        raise RuntimeError(f"Installed C benchmark missing: {source}")
    b15path = build_dir / "e5_pqc_verify_bench_015"
    b16path = build_dir / "e5_pqc_verify_bench_016"
    build15 = build_native_bench(source, b15path, legacy_prefix)
    print("E5_LIBOQS_015_BUILD_GATE=PASS")
    build16 = build_native_bench(source, b16path, research_prefix)
    print("E5_LIBOQS_016_COMPAT_BUILD_GATE=PASS")

    bench15 = run_native_bench(b15path, legacy_prefix, args.bench_iters, oqs["legacy_expected_version"])
    print("E5_LIBOQS_015_BENCH_GATE=PASS")
    bench16 = run_native_bench(b16path, research_prefix, args.bench_iters, oqs["research_expected_version"])
    print("E5_LIBOQS_016_COMPAT_GATE=PASS")

    ts = float(contract["sampling_period_seconds"])
    summary15 = summarize_bench(bench15, ts)
    summary16 = summarize_bench(bench16, ts)
    benches = {"0.15.0": bench15, "0.16.0": bench16}
    artifact_gate = runtime_artifact_gate(contract, benches)
    if not artifact_gate["gate"]:
        raise RuntimeError("E5_STANDARD_ARTIFACT_RUNTIME_GATE=FAIL: " + "; ".join(artifact_gate["issues"]))
    print("E5_STANDARD_ARTIFACT_RUNTIME_GATE=PASS")

    verify_depth_ok = all(
        x[alg]["compute_only_R"] == 1
        for x in (summary15, summary16)
        for alg in ("ML-DSA-65", "SLH-DSA-SHA2-192s")
    )
    if not verify_depth_ok:
        raise RuntimeError("E5_VERIFY_DEPTH_ROBUSTNESS_GATE=FAIL")
    print("E5_VERIFY_DEPTH_ROBUSTNESS_GATE=PASS")

    params15 = profile_parameters(contract, prov, summary15)
    params16 = profile_parameters(contract, prov, summary16)
    # If R_verify is invariant, the discrete E4 service automata are invariant across library versions.
    profile_invariant = all(
        params15[key]["robust_R_formula"] == params16[key]["robust_R_formula"]
        and params15[key]["rverify"] == params16[key]["rverify"]
        for key in params15
    )
    if not profile_invariant:
        raise RuntimeError("E5_LIBOQS_VERSION_INVARIANCE_GATE=FAIL")
    print("E5_LIBOQS_VERSION_INVARIANCE_GATE=PASS")

    engine = build_physical_engine(root, args.geometry)
    print("E5_UPSTREAM_PHYSICAL_ENGINE_GATE=PASS")
    print(f"E5_UPSTREAM_SOLVER_SHA256={sha256_file(engine['source'])}")
    print(f"E5_PHYSICAL_NODE_COUNT={int(np.prod(engine['shape']))}")

    scalar, scalar_rows = scalar_bridge(
        engine,
        int(params15["ML_DSA_65"]["robust_R_formula"]),
        int(params15["SLH_DSA_SHA2_192S"]["robust_R_formula"]),
    )
    if not scalar["ML_DSA_65"]["converged"] or not scalar["SLH_DSA_SHA2_192S"]["converged"]:
        raise RuntimeError("E5_SCALAR_BRIDGE_GATE=FAIL")
    print("E5_SCALAR_BRIDGE_GATE=PASS")
    print("SCALAR_HORIZON_DIAGNOSTIC_ONLY=YES")

    full, full_rows = run_full_service_suite(engine, params15)
    if not full["gate"]:
        raise RuntimeError("E5_FULL_SERVICE_SOLVE_GATE=FAIL")
    print("E5_FULL_SERVICE_SOLVE_GATE=PASS")
    print(f"E5_REAL_PROFILE_CLASSIFICATION={full['classification']}")
    print(f"E5_PRIMARY_REAL_PROFILE_MAX_GAP_M={full['primary_real_profile_comparison']['max_abs_gap_m']:.12g}")

    default_after = query_liboqs_version(None)
    default_unchanged = default_before == default_after == oqs["legacy_expected_version"]
    if not default_unchanged:
        raise RuntimeError(f"DEFAULT_LIBOQS_UNCHANGED_GATE=FAIL before={default_before} after={default_after}")
    print("DEFAULT_LIBOQS_UNCHANGED_GATE=PASS")

    platform_result = {
        "default_version_before": default_before,
        "default_version_after": default_after,
        "legacy_prefix": str(legacy_prefix),
        "legacy_explicit_version": v15,
        "research_prefix": str(research_prefix),
        "research_explicit_version": v16,
        "build_015": build15,
        "build_016": build16,
        "default_unchanged": default_unchanged,
        "platform": platform.platform(),
        "python": sys.version,
    }

    result = {
        "schema": SCHEMA,
        "timestamp_utc": stamp,
        "status": "PASS",
        "geometry": args.geometry,
        "packetization_provenance_gate": prov["gate"],
        "packetization_provenance": prov,
        "platform_isolation_gate": True,
        "platform": platform_result,
        "artifact_runtime_gate": artifact_gate,
        "bench_summary": {"0.15.0": summary15, "0.16.0": summary16},
        "verify_depth_robustness_gate": verify_depth_ok,
        "liboqs_service_profile_invariance_gate": profile_invariant,
        "profile_parameters": params15,
        "profile_parameters_016": params16,
        "scalar_bridge": scalar,
        "full_service": full,
        "continuous_state_separation_certified": False,
        "continuous_action_interval_gfp_solved": False,
        "network_bound_measured": False,
        "deployment_protocol_certified": False,
        "manuscript_mutation_performed": False,
    }

    # Output artifacts.
    platform_path = results_dir / f"E5_PLATFORM_AND_OQS_COMPAT_{stamp}.json"
    prov_path = results_dir / f"E5_PACKETIZATION_PROVENANCE_{stamp}.json"
    bench_csv = results_dir / f"E5_LIBOQS_DUAL_BENCH_{stamp}.csv"
    scalar_path = results_dir / f"E5_REAL_PROFILE_SCALAR_BRIDGE_{stamp}.json"
    full_path = results_dir / f"E5_REAL_PROFILE_FULL_SERVICE_GFP_{stamp}.json"
    comp_csv = results_dir / f"E5_REAL_PROFILE_COMPARISON_{stamp}.csv"
    handoff_path = results_dir / f"E5_POST_RESULT_MANUSCRIPT_HANDOFF_{stamp}.json"
    tex_path = results_dir / f"E5_NUMERICAL_EVIDENCE_FOR_MANUSCRIPT_{stamp}.tex"
    result_path = results_dir / f"E5_RESULT_{stamp}.json"
    latest_path = results_dir / "E5_LATEST.json"

    write_json(platform_path, platform_result)
    write_json(prov_path, prov)
    all_bench_rows = []
    for ver, bench in benches.items():
        for row in bench["rows"]:
            all_bench_rows.append({"liboqs_version": ver, **row})
    write_csv(bench_csv, all_bench_rows)
    write_json(scalar_path, scalar)
    write_json(full_path, full)
    write_csv(comp_csv, scalar_rows + full_rows)
    write_json(handoff_path, manuscript_handoff(contract, result))
    atomic_write(tex_path, render_tex_evidence(result))
    write_json(result_path, result)
    write_json(latest_path, result)

    manifest_path = results_dir / f"E5_MANIFEST_{stamp}.sha256"
    artifacts = [
        platform_path, prov_path, bench_csv, scalar_path, full_path, comp_csv,
        handoff_path, tex_path, result_path, latest_path,
        root / "01_config" / "E5_PROVENANCE_CONTRACT.json",
        root / "02_src" / "e5_real_profile_closure.py",
        root / "02_src" / "e5_pqc_verify_bench.c",
    ]
    manifest = "\n".join(f"{sha256_file(p)}  {p}" for p in artifacts) + "\n"
    atomic_write(manifest_path, manifest)

    print("CONTINUOUS_STATE_SEPARATION_CERTIFIED=NO")
    print("CONTINUOUS_ACTION_INTERVAL_GFP_SOLVED=NO")
    print("NETWORK_BOUND_MEASURED=NO")
    print("DEPLOYMENT_PROTOCOL_CERTIFIED=NO")
    print("MANUSCRIPT_MUTATION_PERFORMED=NO")
    print("E5_EXECUTION=PASS")
    print("E5_NEXT_ACTION=REVIEW_E5_RESULTS_THEN_UPDATE_THEORY_RELATED_WORK_TITLE_AND_NUMERICAL_TEXT")
    print(f"E5_RESULT_JSON={result_path}")
    print(f"E5_FULL_SERVICE_JSON={full_path}")
    print(f"E5_COMPARISON_CSV={comp_csv}")
    print(f"E5_MANUSCRIPT_HANDOFF={handoff_path}")
    print(f"E5_MANIFEST={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
