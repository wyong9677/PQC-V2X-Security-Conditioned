from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path


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


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def frac_from_obj(obj: dict) -> Fraction:
    return Fraction(int(obj["numerator"]), int(obj["denominator"]))


def fraction_from_decimal_float_text(text: str) -> Fraction:
    return Fraction.from_float(float(text))


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: independent_checker.py CERTIFICATE_JSON REPORT_JSON"
        )

    cert_path = Path(sys.argv[1]).resolve()
    report_path = Path(sys.argv[2]).resolve()
    cert = load_json(cert_path)

    if cert.get("schema") != "SCV_P3B1_R7D8_RATIONAL_OUTWARD_CERTIFICATE_V1":
        raise RuntimeError("D8_CHECKER_CERT_SCHEMA_FAIL")

    # No project solver imports are used in this checker.
    source_files = cert["source_files"]

    source_hash_checks = {}
    for key, meta in source_files.items():
        path = Path(meta["path"])
        actual = sha256_file(path)
        source_hash_checks[key] = {
            "expected": meta["sha256"],
            "actual": actual,
            "pass": actual == meta["sha256"],
        }

    if not all(x["pass"] for x in source_hash_checks.values()):
        raise RuntimeError("D8_CHECKER_SOURCE_HASH_FAIL")

    witness_csv = Path(source_files["witness_csv"]["path"])
    witness_rows = read_csv(witness_csv)
    row_by_uid = {
        int(row["unique_witness_id"]): row
        for row in witness_rows
    }

    arithmetic = cert["arithmetic_model"]
    tol = frac_from_obj(arithmetic["numeric_tolerance_exact_rational"])
    diff_guard = frac_from_obj(
        arithmetic["difference_guard_exact_rational"]
    )
    min_multiple = int(
        arithmetic[
            "minimum_certified_lower_bound_multiple_of_tolerance"
        ]
    )

    profiles = tuple(cert["profiles"])
    suffixes = tuple(cert["span_fields_per_profile"])

    witness_checks = []
    global_min_lower = None
    global_min_multiple = None

    for witness in cert["witness_certificates"]:
        uid = int(witness["unique_witness_id"])
        if uid not in row_by_uid:
            raise RuntimeError(f"D8_CHECKER_WITNESS_MISSING uid={uid}")

        source_row = row_by_uid[uid]
        profile_checks = {}

        for profile in profiles:
            pc = witness["profile_certificates"][profile]

            source_values = []
            field_checks = {}

            for suffix in suffixes:
                field = f"{profile}_{suffix}"
                source_text = source_row[field]
                exact = fraction_from_decimal_float_text(source_text)

                declared = frac_from_obj(
                    pc["fields"][suffix]["exact_binary64_rational"]
                )

                field_checks[suffix] = {
                    "pass": exact == declared,
                    "source_text": source_text,
                }
                if exact != declared:
                    raise RuntimeError(
                        f"D8_CHECKER_FIELD_RATIONAL_MISMATCH "
                        f"uid={uid} field={field}"
                    )

                source_values.append(exact)

            observed_min = min(source_values)
            declared_min = frac_from_obj(
                pc["observed_min_span_exact_rational"]
            )
            if observed_min != declared_min:
                raise RuntimeError(
                    f"D8_CHECKER_MIN_MISMATCH uid={uid} profile={profile}"
                )

            lower = max(Fraction(0, 1), observed_min - diff_guard)
            declared_lower = frac_from_obj(
                pc["certified_lower_bound_exact_rational"]
            )
            if lower != declared_lower:
                raise RuntimeError(
                    f"D8_CHECKER_LOWER_BOUND_MISMATCH "
                    f"uid={uid} profile={profile}"
                )

            positive = lower > 0
            multiple = lower / tol
            passes_multiple = multiple >= min_multiple

            if bool(pc["positive_after_guard"]) != positive:
                raise RuntimeError(
                    f"D8_CHECKER_POSITIVE_FLAG_MISMATCH "
                    f"uid={uid} profile={profile}"
                )
            if bool(pc["passes_minimum_multiple"]) != passes_multiple:
                raise RuntimeError(
                    f"D8_CHECKER_MULTIPLE_FLAG_MISMATCH "
                    f"uid={uid} profile={profile}"
                )

            if global_min_lower is None or lower < global_min_lower:
                global_min_lower = lower
            if (
                global_min_multiple is None
                or multiple < global_min_multiple
            ):
                global_min_multiple = multiple

            profile_checks[profile] = {
                "fields": field_checks,
                "certified_lower_bound_exact_rational": {
                    "numerator": str(lower.numerator),
                    "denominator": str(lower.denominator),
                },
                "positive_after_guard": positive,
                "passes_minimum_multiple": passes_multiple,
            }

        witness_checks.append(
            {
                "unique_witness_id": uid,
                "profiles": profile_checks,
            }
        )

    assert global_min_lower is not None
    assert global_min_multiple is not None

    aggregate = cert["aggregate"]

    if global_min_lower != frac_from_obj(
        aggregate["minimum_certified_lower_bound_exact_rational"]
    ):
        raise RuntimeError("D8_CHECKER_GLOBAL_MIN_LOWER_MISMATCH")

    all_positive = all(
        p["positive_after_guard"]
        for w in witness_checks
        for p in w["profiles"].values()
    )
    all_multiple = all(
        p["passes_minimum_multiple"]
        for w in witness_checks
        for p in w["profiles"].values()
    )

    candidate = bool(
        all_positive
        and all_multiple
        and len(witness_checks) == cert["witness_count"]
    )

    if bool(
        cert["claims"][
            "rational_outward_numerical_separation_certificate_candidate"
        ]
    ) != candidate:
        raise RuntimeError("D8_CHECKER_CANDIDATE_FLAG_MISMATCH")

    forbidden_true = (
        "scientific_kernel_claim_authorized",
        "maximal_continuous_kernel_claim_authorized",
        "farkas_certificate_authorized",
        "implementation_refinement_claim_authorized",
    )
    for key in forbidden_true:
        if cert["claims"].get(key) is not False:
            raise RuntimeError(
                "D8_CHECKER_CLAIM_CALIBRATION_FAIL=" + key
            )

    report = {
        "schema": "SCV_P3B1_R7D8_INDEPENDENT_CHECKER_REPORT_V1",
        "timestamp_utc": datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        ),
        "checker_scope": (
            "Standard-library-only independent arithmetic/source-integrity "
            "verification. No Bellman, fixed-point, numpy, scipy, or project "
            "solver module is imported."
        ),
        "certificate_path": str(cert_path),
        "certificate_sha256": sha256_file(cert_path),
        "source_hash_checks": source_hash_checks,
        "witness_checks": witness_checks,
        "aggregate": {
            "witness_count": len(witness_checks),
            "profile_count": len(profiles),
            "all_lower_bounds_positive": all_positive,
            "all_lower_bounds_pass_minimum_multiple": all_multiple,
            "minimum_certified_lower_bound_m":
                format(float(global_min_lower), ".17g"),
            "minimum_certified_lower_bound_multiple_of_tol":
                format(float(global_min_multiple), ".17g"),
        },
        "claims": {
            "independent_checker_pass": candidate,
            "rational_outward_numerical_separation_certificate_verified":
                candidate,
            "scientific_kernel_claim_authorized": False,
            "farkas_certificate_authorized": False,
            "implementation_refinement_claim_authorized": False,
        },
        "status": "PASS" if candidate else "FAIL",
    }

    atomic_write(
        report_path,
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )

    print("=== P3-B1-R7-D8 INDEPENDENT CHECKER ===")
    print(f"WITNESS_COUNT={len(witness_checks)}")
    print(f"PROFILE_COUNT={len(profiles)}")
    print(
        "MIN_CERTIFIED_LOWER_BOUND_M="
        f"{float(global_min_lower):.12g}"
    )
    print(
        "MIN_CERTIFIED_LOWER_BOUND_MULTIPLE_OF_TOL="
        f"{float(global_min_multiple):.12g}"
    )
    print(
        "INDEPENDENT_CHECKER_PASS="
        + ("YES" if candidate else "NO")
    )
    print("SCIENTIFIC_KERNEL_CLAIM_AUTHORIZED=NO")
    print("FARKAS_CERTIFICATE_AUTHORIZED=NO")
    print("IMPLEMENTATION_REFINEMENT_CLAIM=NO")
    print(f"REPORT_JSON={report_path}")

    return 0 if candidate else 2


if __name__ == "__main__":
    raise SystemExit(main())
