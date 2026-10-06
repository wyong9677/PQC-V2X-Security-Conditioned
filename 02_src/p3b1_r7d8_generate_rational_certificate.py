from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PROTOCOL_PATH = (
    ROOT / "01_config" / "p3b1_r7d8_rational_outward_protocol_v1.json"
)
D7_PATH = ROOT / "04_results" / "P3B1_R7D7_LATEST.json"
D7B_PATH = ROOT / "04_results" / "P3B1_R7D7B_LATEST.json"

RESULTS_DIR = ROOT / "04_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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


def fraction_from_decimal_float_text(text: str) -> Fraction:
    """
    The CSV was written from Python floats using round-trip decimal text.
    Parsing back to float and Fraction.from_float recovers the exact binary64
    value represented by that text.
    """
    return Fraction.from_float(float(text))


def frac_obj(x: Fraction) -> dict:
    x = Fraction(x)
    return {
        "numerator": str(x.numerator),
        "denominator": str(x.denominator),
    }


def frac_from_obj(obj: dict) -> Fraction:
    return Fraction(int(obj["numerator"]), int(obj["denominator"]))


def min_fraction(values):
    values = list(values)
    if not values:
        raise ValueError("empty fraction sequence")
    out = values[0]
    for x in values[1:]:
        if x < out:
            out = x
    return out


def main() -> int:
    protocol = load_json(PROTOCOL_PATH)
    d7 = load_json(D7_PATH)
    d7b = load_json(D7B_PATH)

    if d7.get("status") != "PASS":
        raise RuntimeError("P3B1_R7D8_D7_UPSTREAM_FAIL")
    if d7b.get("status") != "PASS":
        raise RuntimeError("P3B1_R7D8_D7B_UPSTREAM_FAIL")

    expected = (
        "Q_DEPENDENT_KERNEL_LAYER_IS_STABLE_UNDER_TWO_NESTED_TRUE_FIXED_POINT_REFINEMENTS_WITH_POSITIVE_OUTWARD_MARGIN"
    )
    if d7b.get("interpretation") != expected:
        raise RuntimeError(
            "P3B1_R7D8_D7B_INTERPRETATION_FAIL="
            + str(d7b.get("interpretation"))
        )

    if not d7b["claims"].get(
        "second_level_refined_fixed_point_stability_candidate", False
    ):
        raise RuntimeError("P3B1_R7D8_D7B_STABILITY_CANDIDATE_FAIL")
    if not d7b["claims"].get(
        "numerical_outward_precertificate_candidate", False
    ):
        raise RuntimeError("P3B1_R7D8_D7B_PRECERT_CANDIDATE_FAIL")

    witness_csv = Path(d7b["artifacts"]["witness_csv"])
    geometry_csv = Path(d7b["artifacts"]["geometry_csv"])
    witness_rows = read_csv(witness_csv)
    geometry_rows = read_csv(geometry_csv)

    required_count = int(protocol["required_witness_count"])
    profiles = tuple(protocol["required_profiles"])
    field_suffixes = tuple(protocol["span_fields_per_profile"])

    if len(witness_rows) != required_count:
        raise RuntimeError(
            f"P3B1_R7D8_WITNESS_COUNT_FAIL={len(witness_rows)}"
        )
    if len(geometry_rows) != required_count:
        raise RuntimeError(
            f"P3B1_R7D8_GEOMETRY_COUNT_FAIL={len(geometry_rows)}"
        )

    tol = fraction_from_decimal_float_text(
        str(protocol["numeric_tolerance_m"])
    )
    endpoint_guard = (
        int(protocol["endpoint_guard_multiple_of_tolerance"]) * tol
    )
    difference_guard = (
        int(protocol["difference_guard_factor"]) * endpoint_guard
    )
    min_multiple = int(
        protocol["minimum_certified_lower_bound_multiple_of_tolerance"]
    )

    source_files = {
        "protocol": PROTOCOL_PATH,
        "d7_latest": D7_PATH,
        "d7b_latest": D7B_PATH,
        "witness_csv": witness_csv,
        "geometry_csv": geometry_csv,
    }

    sources = {
        key: {
            "path": str(path),
            "sha256": sha256_file(path),
        }
        for key, path in source_files.items()
    }

    cert_rows = []
    min_lower = None
    min_lower_multiple = None
    max_observed = Fraction(0, 1)

    for row in witness_rows:
        uid = int(row["unique_witness_id"])

        if str(row["all_nested_refinement_points_strong"]).lower() not in (
            "true", "1"
        ):
            raise RuntimeError(
                f"P3B1_R7D8_NON_STRONG_WITNESS uid={uid}"
            )

        profile_certs = {}

        for profile in profiles:
            exact_values = []
            fields = {}

            for suffix in field_suffixes:
                field = f"{profile}_{suffix}"
                if field not in row:
                    raise RuntimeError(
                        f"P3B1_R7D8_MISSING_FIELD uid={uid} field={field}"
                    )

                x = fraction_from_decimal_float_text(row[field])
                exact_values.append(x)
                fields[suffix] = {
                    "source_text": row[field],
                    "exact_binary64_rational": frac_obj(x),
                    "float_ulp_m": repr(math.ulp(float(row[field]))),
                }
                if x > max_observed:
                    max_observed = x

            observed_min = min_fraction(exact_values)
            lower = observed_min - difference_guard
            if lower < 0:
                lower = Fraction(0, 1)

            lower_multiple = lower / tol

            if min_lower is None or lower < min_lower:
                min_lower = lower
            if (
                min_lower_multiple is None
                or lower_multiple < min_lower_multiple
            ):
                min_lower_multiple = lower_multiple

            profile_certs[profile] = {
                "fields": fields,
                "observed_min_span_exact_rational": frac_obj(observed_min),
                "difference_guard_exact_rational": frac_obj(difference_guard),
                "certified_lower_bound_exact_rational": frac_obj(lower),
                "certified_lower_bound_decimal_m": format(float(lower), ".17g"),
                "certified_lower_bound_multiple_of_tol_decimal":
                    format(float(lower_multiple), ".17g"),
                "positive_after_guard": bool(lower > 0),
                "passes_minimum_multiple": bool(
                    lower_multiple >= min_multiple
                ),
            }

        cert_rows.append(
            {
                "unique_witness_id": uid,
                "profile_certificates": profile_certs,
            }
        )

    assert min_lower is not None
    assert min_lower_multiple is not None

    all_positive = all(
        p["positive_after_guard"]
        for w in cert_rows
        for p in w["profile_certificates"].values()
    )
    all_min_multiple = all(
        p["passes_minimum_multiple"]
        for w in cert_rows
        for p in w["profile_certificates"].values()
    )

    certificate_candidate = bool(
        all_positive
        and all_min_multiple
        and len(cert_rows) == required_count
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cert_path = (
        RESULTS_DIR
        / f"P3B1_R7D8_RATIONAL_OUTWARD_CERTIFICATE_{stamp}.json"
    )
    latest_cert_path = RESULTS_DIR / "P3B1_R7D8_CERTIFICATE_LATEST.json"

    certificate = {
        "schema": "SCV_P3B1_R7D8_RATIONAL_OUTWARD_CERTIFICATE_V1",
        "timestamp_utc": stamp,
        "scope": (
            "Exact-rational verification of recorded binary64 q-span values "
            "under the pre-registered D7B two-sided numerical guard. This "
            "certificate does not independently prove the Bellman solver, "
            "continuous-domain maximality, Farkas infeasibility, or "
            "implementation refinement."
        ),
        "source_files": sources,
        "arithmetic_model": {
            "numeric_tolerance_exact_rational": frac_obj(tol),
            "endpoint_guard_multiple_of_tolerance":
                int(protocol["endpoint_guard_multiple_of_tolerance"]),
            "endpoint_guard_exact_rational": frac_obj(endpoint_guard),
            "difference_guard_factor":
                int(protocol["difference_guard_factor"]),
            "difference_guard_exact_rational": frac_obj(difference_guard),
            "minimum_certified_lower_bound_multiple_of_tolerance":
                min_multiple,
            "binary64_reconstruction":
                "Fraction.from_float(float(csv_decimal_text))",
        },
        "witness_count": len(cert_rows),
        "profiles": list(profiles),
        "span_fields_per_profile": list(field_suffixes),
        "witness_certificates": cert_rows,
        "aggregate": {
            "all_witness_profile_lower_bounds_positive": all_positive,
            "all_witness_profile_lower_bounds_pass_minimum_multiple":
                all_min_multiple,
            "minimum_certified_lower_bound_exact_rational":
                frac_obj(min_lower),
            "minimum_certified_lower_bound_decimal_m":
                format(float(min_lower), ".17g"),
            "minimum_certified_lower_bound_multiple_of_tol_decimal":
                format(float(min_lower_multiple), ".17g"),
            "maximum_observed_span_exact_rational":
                frac_obj(max_observed),
            "maximum_observed_span_decimal_m":
                format(float(max_observed), ".17g"),
        },
        "claims": {
            "rational_outward_numerical_separation_certificate_candidate":
                certificate_candidate,
            "certificate_backed_finite_abstraction_witness_candidate":
                certificate_candidate,
            "scientific_kernel_claim_authorized": False,
            "maximal_continuous_kernel_claim_authorized": False,
            "farkas_certificate_authorized": False,
            "implementation_refinement_claim_authorized": False,
        },
    }

    atomic_write(
        cert_path,
        json.dumps(certificate, indent=2, sort_keys=True) + "\n",
    )
    atomic_write(
        latest_cert_path,
        json.dumps(certificate, indent=2, sort_keys=True) + "\n",
    )

    print("=== P3-B1-R7-D8 CERTIFICATE GENERATION ===")
    print(f"WITNESS_COUNT={len(cert_rows)}")
    print(f"PROFILE_COUNT={len(profiles)}")
    print(
        "MIN_CERTIFIED_LOWER_BOUND_M="
        f"{float(min_lower):.12g}"
    )
    print(
        "MIN_CERTIFIED_LOWER_BOUND_MULTIPLE_OF_TOL="
        f"{float(min_lower_multiple):.12g}"
    )
    print(
        "RATIONAL_OUTWARD_CERTIFICATE_CANDIDATE="
        + ("YES" if certificate_candidate else "NO")
    )
    print("FARKAS_CERTIFICATE_AUTHORIZED=NO")
    print("SCIENTIFIC_KERNEL_CLAIM_AUTHORIZED=NO")
    print("IMPLEMENTATION_REFINEMENT_CLAIM=NO")
    print(f"CERTIFICATE_JSON={cert_path}")
    print(f"CERTIFICATE_LATEST={latest_cert_path}")

    return 0 if certificate_candidate else 2


if __name__ == "__main__":
    raise SystemExit(main())
