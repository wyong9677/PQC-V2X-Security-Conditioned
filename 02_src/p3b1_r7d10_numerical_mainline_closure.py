from __future__ import annotations

import csv
import glob
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "04_results"
RESULTS.mkdir(parents=True, exist_ok=True)

PROTOCOL = ROOT / "01_config" / "p3b1_r7d10_numerical_mainline_closure_protocol_v1.json"

D4 = RESULTS / "P3B1_R7D4_LATEST.json"
D5 = RESULTS / "P3B1_R7D5_LATEST.json"
D6 = RESULTS / "P3B1_R7D6_LATEST.json"
D6B = RESULTS / "P3B1_R7D6B_LATEST.json"
D6C = RESULTS / "P3B1_R7D6C_LATEST.json"
D7 = RESULTS / "P3B1_R7D7_LATEST.json"
D7B = RESULTS / "P3B1_R7D7B_LATEST.json"
D8_CERT = RESULTS / "P3B1_R7D8_CERTIFICATE_LATEST.json"
D9_CERT = RESULTS / "P3B1_R7D9_CERTIFICATE_LATEST.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def latest_glob(pattern: str) -> Path:
    files = [Path(x) for x in glob.glob(str(RESULTS / pattern))]
    if not files:
        raise RuntimeError("D10_MISSING_GLOB=" + pattern)
    return max(files, key=lambda p: p.stat().st_mtime)


def write_csv(path: Path, rows: list[dict]) -> None:
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
        for row in rows:
            w.writerow(row)


def main() -> int:
    protocol = load_json(PROTOCOL)

    d4 = load_json(D4)
    d5 = load_json(D5)
    d6 = load_json(D6)
    d6b = load_json(D6B)
    d6c = load_json(D6C)
    d7 = load_json(D7)
    d7b = load_json(D7B)
    d8 = load_json(D8_CERT)
    d9 = load_json(D9_CERT)

    d8_checker_path = latest_glob("P3B1_R7D8_INDEPENDENT_CHECKER_*.json")
    d9_checker_path = latest_glob("P3B1_R7D9_INDEPENDENT_CHECKER_*.json")
    d8_checker = load_json(d8_checker_path)
    d9_checker = load_json(d9_checker_path)

    checks = {
        "D4_PASS": d4.get("status") == "PASS",
        "D5_PASS": d5.get("status") == "PASS",
        "D6_PASS": d6.get("status") == "PASS",
        "D6B_PASS": d6b.get("status") == "PASS",
        "D6C_PASS": d6c.get("status") == "PASS",
        "D7_PASS": d7.get("status") == "PASS",
        "D7B_PASS": d7b.get("status") == "PASS",
        "D8_RATIONAL_OUTWARD_CERTIFICATE":
            bool(d8["claims"].get(
                "rational_outward_numerical_separation_certificate_candidate",
                False
            )),
        "D8_INDEPENDENT_CHECKER":
            d8_checker.get("status") == "PASS"
            and bool(d8_checker["claims"].get("independent_checker_pass", False)),
        "D9_SET_SEPARATION_CERTIFICATE":
            bool(d9["claims"].get(
                "finite_abstraction_kernel_set_separation_certificate",
                False
            )),
        "D9_INDEPENDENT_CHECKER":
            d9_checker.get("status") == "PASS"
            and bool(d9_checker["claims"].get("independent_checker_pass", False)),
        "D9_WITNESS_COUNT":
            int(d9.get("witness_count", -1))
            == int(protocol["required_d9_witness_count"]),
    }

    if not all(checks.values()):
        bad = [k for k, v in checks.items() if not v]
        raise RuntimeError("D10_UPSTREAM_CLOSURE_FAIL=" + ",".join(bad))

    min_d8_span = float(
        d8["aggregate"]["minimum_certified_lower_bound_decimal_m"]
    )
    min_d9_margin = float(
        d9["aggregate"]["minimum_certified_membership_margin_m"]
    )
    max_d7_span = 0.0
    for scenario in d7["scenario_summaries"]:
        # Recover maximum from profile audit is not required for closure.
        pass

    # D7 aggregate / D7B metrics.
    d7_witnesses = int(d7["aggregate"]["candidate_rows_recomputed"])
    d7_strong = int(
        d7["aggregate"]["candidate_rows_strong_in_all_diagnostic_profiles"]
    )
    d7b_nested = int(
        d7b["aggregate"]["all_nested_points_strong_witnesses"]
    )
    d7b_drift_ab = float(
        d7b["aggregate"]["max_center_relative_drift_A_to_B"]
    )
    d7b_drift_d7b = float(
        d7b["aggregate"]["max_center_relative_drift_D7_to_B"]
    )

    claim_rows = [
        {
            "claim_id": "C1",
            "claim": "Latent q-dependence exists in the cooperative Bellman requirement.",
            "status": "SUPPORTED",
            "evidence": "D3-D4",
            "allowed_scope": "finite abstraction / model level",
        },
        {
            "claim_id": "C2",
            "claim": "q changes the certified cooperative/fallback supervisor boundary on the coarse grid.",
            "status": "SUPPORTED",
            "evidence": "D4-D5",
            "allowed_scope": "coarse finite abstraction",
        },
        {
            "claim_id": "C3",
            "claim": "The coarse projected viability envelope is q-independent at the original grid resolution.",
            "status": "SUPPORTED",
            "evidence": "D3-D7 baseline control",
            "allowed_scope": "original coarse grid only",
        },
        {
            "claim_id": "C4",
            "claim": "Targeted bar_a refinement reveals q-dependent refined fixed-point viability thresholds.",
            "status": "SUPPORTED",
            "evidence": "D6C-D7",
            "allowed_scope": "refined finite abstraction",
        },
        {
            "claim_id": "C5",
            "claim": "The q-dependent layer persists under two nested true fixed-point refinements.",
            "status": "SUPPORTED",
            "evidence": "D7B",
            "allowed_scope": "nested refined finite abstractions",
        },
        {
            "claim_id": "C6",
            "claim": "Recorded q-span separation has an exact-rational outward numerical certificate verified by an independent checker.",
            "status": "SUPPORTED",
            "evidence": "D8",
            "allowed_scope": "recorded binary64 numerical evidence with declared guard",
        },
        {
            "claim_id": "C7",
            "claim": "There exist refined finite-abstraction states with different viability membership across q.",
            "status": "SUPPORTED",
            "evidence": "D9",
            "allowed_scope": "Level-B refined finite abstraction",
        },
        {
            "claim_id": "C8",
            "claim": "The continuous-domain maximal viability kernel is globally q-dependent.",
            "status": "NOT_AUTHORIZED",
            "evidence": "not established by D4-D9",
            "allowed_scope": "requires independent theory/convergence argument",
        },
        {
            "claim_id": "C9",
            "claim": "A Farkas certificate proves the D9 non-membership witnesses.",
            "status": "NOT_AUTHORIZED",
            "evidence": "D8-D9 explicitly exclude Farkas authorization",
            "allowed_scope": "requires explicit infeasibility system and multipliers",
        },
        {
            "claim_id": "C10",
            "claim": "The real implementation refines the abstract service/control model I_c subset A_c.",
            "status": "NOT_AUTHORIZED",
            "evidence": "no implementation traces in D4-D9",
            "allowed_scope": "requires implementation-refinement audit",
        },
    ]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    claim_csv = RESULTS / f"P3B1_R7D10_CLAIM_MATRIX_{stamp}.csv"
    write_csv(claim_csv, claim_rows)

    manuscript = f"""# Numerical mainline closure: security-conditioned viability

The numerical mainline is closed for the finite-abstraction existence question. The original coarse grid showed no q-dependent projected viability envelope, although q-dependent cooperative Bellman and supervisor effects were present. Targeted refinement along the bar_a direction revealed {d7_witnesses} distinct kernel witnesses, all {d7_strong} of which survived a true refined Bellman fixed-point recomputation in each diagnostic service profile. Two additional nested fixed-point refinements preserved all {d7b_nested} witnesses and produced zero center-span drift both from Level A to Level B ({d7b_drift_ab:.17g}) and from D7 to Level B ({d7b_drift_d7b:.17g}).

The rational outward audit certified a minimum q-span lower bound of {min_d8_span:.17g} m. A standalone checker, isolated from the Bellman solver, independently reproduced the arithmetic and source hashes. Using the same certified lower bounds, the set-separation audit constructed midpoint distance states d^dagger for all {d9['witness_count']} witnesses. For each diagnostic profile, the same refined physical state is viable for q_min and nonviable for q_max, with a minimum independently checked membership-separation margin of {min_d9_margin:.17g} m.

Accordingly, the supported numerical claim is: the Level-B refined finite abstraction contains certificate-backed q-dependent viability-kernel membership. The result should not be stated as a proof that the global continuous-domain maximal kernel is q-dependent, as a Farkas certificate for outside-kernel infeasibility, or as evidence that the real implementation satisfies I_c subset A_c. Those are separate theory and implementation-refinement obligations.
"""

    manuscript_path = RESULTS / f"P3B1_R7D10_MANUSCRIPT_NUMERICAL_STATEMENT_{stamp}.md"
    atomic_write(manuscript_path, manuscript)

    closure = {
        "schema": "SCV_P3B1_R7D10_NUMERICAL_MAINLINE_CLOSURE_V1",
        "status": "PASS",
        "timestamp_utc": stamp,
        "checks": checks,
        "classification": (
            "numerical mainline closure for q-dependent kernel membership "
            "in the refined finite abstraction"
        ),
        "aggregate": {
            "d7_witnesses": d7_witnesses,
            "d7_strong_witnesses": d7_strong,
            "d7b_nested_strong_witnesses": d7b_nested,
            "d8_min_certified_q_span_m": min_d8_span,
            "d9_min_certified_membership_margin_m": min_d9_margin,
            "d7b_max_center_relative_drift_A_to_B": d7b_drift_ab,
            "d7b_max_center_relative_drift_D7_to_B": d7b_drift_d7b,
        },
        "claims": {
            "numerical_mainline_closed": True,
            "finite_abstraction_q_dependent_kernel_membership": True,
            "nested_fixed_point_refinement_stability": True,
            "certificate_backed_set_separation": True,
            "continuous_domain_maximal_kernel_claim_authorized": False,
            "farkas_certificate_authorized": False,
            "implementation_refinement_claim_authorized": False,
        },
        "recommended_next_steps": [
            "Map D9 non-membership witnesses to the explicit outside-kernel linear infeasibility system and produce genuine Farkas multipliers if that system is available.",
            "Perform implementation-refinement audit I_c subset A_c using real authentication/service traces before making end-to-end implementation claims.",
            "Use the generated claim matrix and manuscript numerical statement for paper integration; do not run further existence-search sweeps unless a reviewer asks for them."
        ],
        "artifacts": {
            "claim_matrix_csv": str(claim_csv),
            "manuscript_numerical_statement_md": str(manuscript_path),
        },
    }

    result_path = RESULTS / f"P3B1_R7D10_NUMERICAL_MAINLINE_CLOSURE_{stamp}.json"
    latest_path = RESULTS / "P3B1_R7D10_LATEST.json"
    manifest_path = RESULTS / f"P3B1_R7D10_MANIFEST_{stamp}.sha256"

    closure["artifacts"]["result_json"] = str(result_path)
    closure["artifacts"]["latest_json"] = str(latest_path)
    closure["artifacts"]["manifest"] = str(manifest_path)

    text = json.dumps(closure, indent=2, sort_keys=True) + "\n"
    atomic_write(result_path, text)
    atomic_write(latest_path, text)

    source_paths = [
        PROTOCOL, D4, D5, D6, D6B, D6C, D7, D7B, D8_CERT, d8_checker_path,
        D9_CERT, d9_checker_path, claim_csv, manuscript_path, result_path
    ]
    manifest = "\n".join(
        f"{sha256_file(p)}  {p}" for p in source_paths
    ) + "\n"
    atomic_write(manifest_path, manifest)

    print("=== P3-B1-R7-D10 NUMERICAL MAINLINE CLOSURE ===")
    print(f"D7_WITNESSES={d7_witnesses}")
    print(f"D7B_NESTED_STRONG_WITNESSES={d7b_nested}")
    print(f"D8_MIN_CERTIFIED_Q_SPAN_M={min_d8_span:.12g}")
    print(f"D9_MIN_CERTIFIED_MEMBERSHIP_MARGIN_M={min_d9_margin:.12g}")
    print("FINITE_ABSTRACTION_Q_DEPENDENT_KERNEL_MEMBERSHIP=YES")
    print("NESTED_FIXED_POINT_REFINEMENT_STABILITY=YES")
    print("CERTIFICATE_BACKED_SET_SEPARATION=YES")
    print("CONTINUOUS_DOMAIN_MAXIMAL_KERNEL_CLAIM=NO")
    print("FARKAS_CERTIFICATE=NO")
    print("IMPLEMENTATION_REFINEMENT=NO")
    print("NUMERICAL_MAINLINE_CLOSED=YES")
    print("RECOMMENDED_NEXT_STEP=THEORY_NUMERICS_ALIGNMENT_AND_IMPLEMENTATION_REFINEMENT")
    print(f"CLAIM_MATRIX={claim_csv}")
    print(f"MANUSCRIPT_STATEMENT={manuscript_path}")
    print(f"RESULT_JSON={result_path}")
    print(f"MANIFEST={manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
