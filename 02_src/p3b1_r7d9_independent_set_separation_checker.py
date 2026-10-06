from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def frac(obj: dict) -> Fraction:
    return Fraction(int(obj["numerator"]), int(obj["denominator"]))


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: checker.py CERTIFICATE_JSON REPORT_JSON"
        )

    cert_path = Path(sys.argv[1]).resolve()
    report_path = Path(sys.argv[2]).resolve()
    cert = load_json(cert_path)

    if cert.get("schema") != (
        "SCV_P3B1_R7D9_KERNEL_SET_SEPARATION_CERTIFICATE_V1"
    ):
        raise RuntimeError("D9_CHECKER_SCHEMA_FAIL")

    source_hash_checks = {}
    for key, meta in cert["source_files"].items():
        p = Path(meta["path"])
        actual = sha256_file(p)
        ok = actual == meta["sha256"]
        source_hash_checks[key] = {
            "expected": meta["sha256"],
            "actual": actual,
            "pass": ok,
        }
        if not ok:
            raise RuntimeError(
                f"D9_CHECKER_SOURCE_HASH_FAIL={key}"
            )

    threshold_csv = Path(
        cert["source_files"]["threshold_witness_csv"]["path"]
    )
    rows = read_csv(threshold_csv)
    row_by_uid = {
        int(row["unique_witness_id"]): row
        for row in rows
    }

    d8_path = Path(
        cert["source_files"]["d8_certificate"]["path"]
    )
    d8 = load_json(d8_path)
    d8_by_uid = {
        int(w["unique_witness_id"]): w
        for w in d8["witness_certificates"]
    }

    checks = []
    min_margin = None

    for witness in cert["witnesses"]:
        uid = int(witness["unique_witness_id"])
        if uid not in row_by_uid:
            raise RuntimeError(
                f"D9_CHECKER_THRESHOLD_ROW_MISSING uid={uid}"
            )
        if uid not in d8_by_uid:
            raise RuntimeError(
                f"D9_CHECKER_D8_ROW_MISSING uid={uid}"
            )

        source = row_by_uid[uid]
        profile_checks = {}

        for profile, pc in witness["profiles"].items():
            h_low = frac(pc["h_viable_exact_rational"])
            h_high = frac(pc["h_nonviable_exact_rational"])
            d_mid = frac(pc["midpoint_d_exact_rational"])
            halfspan = frac(pc["raw_halfspan_exact_rational"])
            d8_lower = frac(
                pc[
                    "d8_certified_span_lower_bound_exact_rational"
                ]
            )
            cert_margin = frac(
                pc["certified_membership_margin_exact_rational"]
            )

            if not (h_low < h_high):
                raise RuntimeError(
                    f"D9_CHECKER_ORDER_FAIL uid={uid} profile={profile}"
                )
            if d_mid != (h_low + h_high) / 2:
                raise RuntimeError(
                    f"D9_CHECKER_MIDPOINT_FAIL uid={uid} profile={profile}"
                )
            if halfspan != (h_high - h_low) / 2:
                raise RuntimeError(
                    f"D9_CHECKER_HALFSPAN_FAIL uid={uid} profile={profile}"
                )
            if cert_margin != d8_lower / 2:
                raise RuntimeError(
                    f"D9_CHECKER_CERT_MARGIN_FAIL uid={uid} profile={profile}"
                )

            d8_source_lower = frac(
                d8_by_uid[uid]["profile_certificates"][profile][
                    "certified_lower_bound_exact_rational"
                ]
            )
            if d8_lower != d8_source_lower:
                raise RuntimeError(
                    f"D9_CHECKER_D8_LOWER_MISMATCH uid={uid} profile={profile}"
                )

            if not (
                d_mid - h_low >= cert_margin > 0
                and
                h_high - d_mid >= cert_margin
            ):
                raise RuntimeError(
                    f"D9_CHECKER_SEPARATION_MARGIN_FAIL uid={uid} profile={profile}"
                )

            # Cross-check the exported decimal threshold table.
            if int(source[f"{profile}_q_viable_index"]) != int(
                pc["q_viable_index"]
            ):
                raise RuntimeError(
                    f"D9_CHECKER_QLOW_INDEX_FAIL uid={uid} profile={profile}"
                )
            if int(source[f"{profile}_q_nonviable_index"]) != int(
                pc["q_nonviable_index"]
            ):
                raise RuntimeError(
                    f"D9_CHECKER_QHIGH_INDEX_FAIL uid={uid} profile={profile}"
                )

            if min_margin is None or cert_margin < min_margin:
                min_margin = cert_margin

            profile_checks[profile] = {
                "q_viable_index": int(pc["q_viable_index"]),
                "q_nonviable_index": int(pc["q_nonviable_index"]),
                "midpoint_membership_viable_q": True,
                "midpoint_membership_nonviable_q": False,
                "certified_margin_m":
                    format(float(cert_margin), ".17g"),
                "pass": True,
            }

        checks.append(
            {
                "unique_witness_id": uid,
                "profiles": profile_checks,
            }
        )

    assert min_margin is not None

    required_true = (
        "finite_abstraction_kernel_set_separation_certificate",
        "certificate_backed_q_dependent_kernel_membership",
    )
    for key in required_true:
        if cert["claims"].get(key) is not True:
            raise RuntimeError(
                "D9_CHECKER_REQUIRED_CLAIM_FAIL=" + key
            )

    forbidden_true = (
        "scientific_kernel_claim_authorized",
        "maximal_continuous_kernel_claim_authorized",
        "farkas_certificate_authorized",
        "implementation_refinement_claim_authorized",
    )
    for key in forbidden_true:
        if cert["claims"].get(key) is not False:
            raise RuntimeError(
                "D9_CHECKER_CLAIM_CALIBRATION_FAIL=" + key
            )

    report = {
        "schema":
            "SCV_P3B1_R7D9_INDEPENDENT_SET_SEPARATION_CHECKER_V1",
        "timestamp_utc":
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "checker_scope":
            (
                "Standard-library-only verification of source hashes, exact "
                "rational midpoint arithmetic, D8 certified lower-bound linkage, "
                "and finite-abstraction membership separation. No solver module "
                "or numpy/scipy import is used."
            ),
        "certificate_path":
            str(cert_path),
        "certificate_sha256":
            sha256_file(cert_path),
        "source_hash_checks":
            source_hash_checks,
        "witness_checks":
            checks,
        "aggregate": {
            "witness_count": len(checks),
            "profile_count": len(cert["profiles"]),
            "minimum_certified_membership_margin_m":
                format(float(min_margin), ".17g"),
            "all_set_separation_checks_pass": True,
        },
        "claims": {
            "independent_checker_pass": True,
            "finite_abstraction_kernel_set_separation_verified": True,
            "scientific_kernel_claim_authorized": False,
            "farkas_certificate_authorized": False,
            "implementation_refinement_claim_authorized": False,
        },
        "status": "PASS",
    }

    atomic_write(
        report_path,
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )

    print("=== P3-B1-R7-D9 INDEPENDENT SET-SEPARATION CHECKER ===")
    print(f"WITNESS_COUNT={len(checks)}")
    print(f"PROFILE_COUNT={len(cert['profiles'])}")
    print(
        "MIN_CERTIFIED_MEMBERSHIP_MARGIN_M="
        f"{float(min_margin):.12g}"
    )
    print("FINITE_ABSTRACTION_KERNEL_SET_SEPARATION_VERIFIED=YES")
    print("INDEPENDENT_CHECKER_PASS=YES")
    print("SCIENTIFIC_KERNEL_CLAIM_AUTHORIZED=NO")
    print("FARKAS_CERTIFICATE_AUTHORIZED=NO")
    print("IMPLEMENTATION_REFINEMENT_CLAIM=NO")
    print(f"REPORT_JSON={report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
