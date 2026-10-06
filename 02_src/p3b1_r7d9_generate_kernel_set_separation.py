from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

import p3b1_r7d1_node_level_audit as d1
import p3b1_r7d7_refined_fixed_point_recompute as d7

r3 = d1.r3
r7 = d1.r7
r6 = d1.r6

P1_CFG_PATH = ROOT / "01_config" / "p1_validation_v2.json"
P2B_CFG_PATH = ROOT / "01_config" / "p2b_hybrid_fallback_v1.json"
P2C_CFG_PATH = ROOT / "01_config" / "p2c_switching_guard_v1.json"
P3A_CFG_PATH = ROOT / "01_config" / "p3a_information_contract_v1.json"
D9_CFG_PATH = (
    ROOT / "01_config" / "p3b1_r7d9_kernel_set_separation_protocol_v1.json"
)

D7_PATH = ROOT / "04_results" / "P3B1_R7D7_LATEST.json"
D7B_PATH = ROOT / "04_results" / "P3B1_R7D7B_LATEST.json"
D8_CERT_PATH = ROOT / "04_results" / "P3B1_R7D8_CERTIFICATE_LATEST.json"

RESULTS_DIR = ROOT / "04_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

AXIS_NAMES = ("v_f", "v_p", "a_f", "bar_a", "bar_u", "age")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


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


def frac_from_float(x: float) -> Fraction:
    return Fraction.from_float(float(x))


def frac_obj(x: Fraction) -> dict:
    x = Fraction(x)
    return {
        "numerator": str(x.numerator),
        "denominator": str(x.denominator),
    }


def frac_from_obj(obj: dict) -> Fraction:
    return Fraction(int(obj["numerator"]), int(obj["denominator"]))


def point_index(axes, shape, row):
    coords = [
        float(row["v_f"]),
        float(row["v_p"]),
        float(row["a_f"]),
        float(row["bar_a"]),
        float(row["bar_u"]),
        float(row["age"]),
    ]
    idx = tuple(
        d7.exact_axis_index(axis, value)
        for axis, value in zip(axes, coords)
    )
    return int(np.ravel_multi_index(idx, shape))


def main() -> int:
    protocol = load_json(D9_CFG_PATH)
    d7_latest = load_json(D7_PATH)
    d7b_latest = load_json(D7B_PATH)
    d8_cert = load_json(D8_CERT_PATH)

    for label, obj in (
        ("D7", d7_latest),
        ("D7B", d7b_latest),
    ):
        if obj.get("status") != "PASS":
            raise RuntimeError(f"P3B1_R7D9_{label}_UPSTREAM_FAIL")

    expected = (
        "Q_DEPENDENT_KERNEL_LAYER_IS_STABLE_UNDER_TWO_NESTED_TRUE_FIXED_POINT_REFINEMENTS_WITH_POSITIVE_OUTWARD_MARGIN"
    )
    if d7b_latest.get("interpretation") != expected:
        raise RuntimeError(
            "P3B1_R7D9_D7B_INTERPRETATION_FAIL="
            + str(d7b_latest.get("interpretation"))
        )

    if not d8_cert["claims"].get(
        "rational_outward_numerical_separation_certificate_candidate",
        False,
    ):
        raise RuntimeError("P3B1_R7D9_D8_CERTIFICATE_CANDIDATE_FAIL")

    profiles = tuple(protocol["required_profiles"])
    required_count = int(protocol["required_witness_count"])

    d7_candidate_csv = Path(d7_latest["artifacts"]["candidate_csv"])
    d7b_witness_csv = Path(d7b_latest["artifacts"]["witness_csv"])

    center_rows = read_csv(d7_candidate_csv)
    d7b_rows = read_csv(d7b_witness_csv)

    if len(center_rows) != required_count:
        raise RuntimeError(
            f"P3B1_R7D9_CENTER_COUNT_FAIL={len(center_rows)}"
        )

    center_by_uid = {
        int(row["unique_witness_id"]): row
        for row in center_rows
    }
    d7b_by_uid = {
        int(row["unique_witness_id"]): row
        for row in d7b_rows
    }
    cert_by_uid = {
        int(row["unique_witness_id"]): row
        for row in d8_cert["witness_certificates"]
    }

    if not (
        set(center_by_uid)
        == set(d7b_by_uid)
        == set(cert_by_uid)
    ):
        raise RuntimeError("P3B1_R7D9_WITNESS_ID_SET_MISMATCH")

    data = d1.reconstruct()
    p1_cfg = load_json(P1_CFG_PATH)
    p2b_cfg = load_json(P2B_CFG_PATH)
    p2c_cfg = load_json(P2C_CFG_PATH)
    p3a_cfg = load_json(P3A_CFG_PATH)

    level_b_values = [
        float(x) for x in d7b_latest["level_b"]["inserted_bar_a_values"]
    ]

    refined_eval_axes = [
        np.asarray(axis, dtype=float).copy()
        for axis in data["eval_axes"]
    ]
    refined_eval_axes[3] = d7.union_axis(
        refined_eval_axes[3],
        level_b_values,
    )

    refined_lookup_axes = [
        np.asarray(axis, dtype=float).copy()
        for axis in data["lookup_axes"]
    ]
    refined_lookup_axes[3] = d7.union_axis(
        refined_lookup_axes[3],
        level_b_values,
    )

    refined_eval_cfg = d7.set_cfg_axes(
        data["eval_cfg"],
        refined_eval_axes,
    )

    eval_flat, eval_shape = r6.mesh_flat(refined_eval_axes)
    lookup_flat, lookup_shape = r6.mesh_flat(refined_lookup_axes)

    stop = r3.endpoint_semantics_audit(
        refined_eval_cfg,
        p1_cfg,
        p2b_cfg,
        p3a_cfg,
        eval_flat,
    )
    if not stop["pass"]:
        raise RuntimeError("P3B1_R7D9_STOP_SEMANTICS_FAIL")

    eval_node_indices = r7.exact_node_indices(
        refined_eval_axes,
        refined_lookup_axes,
    )

    lookup_fallback = r6.fallback_required_on_grid(
        refined_eval_cfg,
        p1_cfg,
        p2b_cfg,
        p2c_cfg,
        p3a_cfg,
        lookup_flat,
    )

    transition_data = r7.build_transition_data_to_lookup(
        refined_eval_cfg,
        p1_cfg,
        p2b_cfg,
        p2c_cfg,
        p3a_cfg,
        eval_flat,
        refined_lookup_axes,
    )

    completion_invalid = sum(
        int(np.count_nonzero(~tr["completion_valid"]))
        for tr in transition_data["transitions"]
    )
    defer_invalid = sum(
        int(np.count_nonzero(~tr["defer_valid"]))
        for tr in transition_data["transitions"]
    )
    if completion_invalid or defer_invalid:
        raise RuntimeError(
            "P3B1_R7D9_FROZEN_HALO_COVERAGE_FAIL"
        )

    eval_fallback = np.asarray(
        transition_data["fallback_required"],
        dtype=float,
    )

    service_profiles = p1_cfg["diagnostic_service_profiles"]
    results = {}

    for name in ("ideal",) + profiles:
        R = r7.service_horizon(name, service_profiles[name])
        result = r7.solve_frozen_halo_profile(
            name,
            R,
            refined_eval_cfg,
            eval_node_indices,
            len(eval_node_indices),
            lookup_shape,
            lookup_fallback,
            eval_fallback,
            transition_data,
        )
        if not result["converged"]:
            raise RuntimeError(
                f"P3B1_R7D9_FIXED_POINT_NOT_CONVERGED profile={name}"
            )
        results[name] = result

    witness_rows = []
    certificate_rows = []

    min_cert_margin = None
    min_raw_halfspan = None

    for uid in sorted(center_by_uid):
        center = center_by_uid[uid]
        eval_idx = point_index(
            refined_eval_axes,
            eval_shape,
            center,
        )
        lookup_idx = int(eval_node_indices[eval_idx])

        out = {
            "unique_witness_id": uid,
            "v_f": float(center["v_f"]),
            "v_p": float(center["v_p"]),
            "a_f": float(center["a_f"]),
            "bar_a": float(center["bar_a"]),
            "bar_u": float(center["bar_u"]),
            "age": float(center["age"]),
            "refined_eval_index": eval_idx,
            "refined_lookup_index": lookup_idx,
        }

        cert_profiles = {}

        for name in profiles:
            hq = np.asarray(
                results[name]["h_flat"][lookup_idx, :],
                dtype=float,
            )

            q_low = int(np.argmin(hq))
            q_high = int(np.argmax(hq))
            h_low = float(hq[q_low])
            h_high = float(hq[q_high])
            span = float(h_high - h_low)

            if not (h_high > h_low):
                raise RuntimeError(
                    f"P3B1_R7D9_NONPOSITIVE_CENTER_SPAN uid={uid} profile={name}"
                )

            # Exact rational arithmetic on the stored binary64 thresholds.
            h_low_q = frac_from_float(h_low)
            h_high_q = frac_from_float(h_high)
            midpoint_q = (h_low_q + h_high_q) / 2
            raw_halfspan_q = (h_high_q - h_low_q) / 2

            d8_pc = cert_by_uid[uid]["profile_certificates"][name]
            d8_lower_q = frac_from_obj(
                d8_pc["certified_lower_bound_exact_rational"]
            )
            cert_margin_q = d8_lower_q / 2

            if cert_margin_q <= 0:
                raise RuntimeError(
                    f"P3B1_R7D9_NONPOSITIVE_CERT_MARGIN uid={uid} profile={name}"
                )
            if raw_halfspan_q < cert_margin_q:
                raise RuntimeError(
                    f"P3B1_R7D9_CERT_MARGIN_EXCEEDS_RAW uid={uid} profile={name}"
                )

            d7b_center_span = float(
                d7b_by_uid[uid][f"{name}_level_b_center_span_m"]
            )
            if span != d7b_center_span:
                raise RuntimeError(
                    f"P3B1_R7D9_LEVEL_B_CENTER_SPAN_MISMATCH "
                    f"uid={uid} profile={name} rerun={span!r} source={d7b_center_span!r}"
                )

            if min_cert_margin is None or cert_margin_q < min_cert_margin:
                min_cert_margin = cert_margin_q
            if min_raw_halfspan is None or raw_halfspan_q < min_raw_halfspan:
                min_raw_halfspan = raw_halfspan_q

            out[f"{name}_q_viable_index"] = q_low
            out[f"{name}_q_nonviable_index"] = q_high
            out[f"{name}_h_viable_m"] = h_low
            out[f"{name}_h_nonviable_m"] = h_high
            out[f"{name}_q_span_m"] = span
            out[f"{name}_midpoint_d_m"] = float(midpoint_q)
            out[f"{name}_raw_halfspan_m"] = float(raw_halfspan_q)
            out[f"{name}_certified_membership_margin_m"] = float(cert_margin_q)

            cert_profiles[name] = {
                "q_viable_index": q_low,
                "q_nonviable_index": q_high,
                "h_viable_exact_rational": frac_obj(h_low_q),
                "h_nonviable_exact_rational": frac_obj(h_high_q),
                "midpoint_d_exact_rational": frac_obj(midpoint_q),
                "raw_halfspan_exact_rational": frac_obj(raw_halfspan_q),
                "d8_certified_span_lower_bound_exact_rational":
                    d8_pc["certified_lower_bound_exact_rational"],
                "certified_membership_margin_exact_rational":
                    frac_obj(cert_margin_q),
                "membership_statements": {
                    "viable_q":
                        "d_mid >= h_star(q_viable) with certified positive margin",
                    "nonviable_q":
                        "d_mid < h_star(q_nonviable) with certified positive margin",
                },
            }

        witness_rows.append(out)
        certificate_rows.append(
            {
                "unique_witness_id": uid,
                "state_without_d": {
                    "v_f": str(center["v_f"]),
                    "v_p": str(center["v_p"]),
                    "a_f": str(center["a_f"]),
                    "bar_a": str(center["bar_a"]),
                    "bar_u": str(center["bar_u"]),
                    "age": str(center["age"]),
                },
                "profiles": cert_profiles,
            }
        )

    assert min_cert_margin is not None
    assert min_raw_halfspan is not None

    stamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    witness_csv = (
        RESULTS_DIR
        / f"P3B1_R7D9_KERNEL_SET_SEPARATION_WITNESSES_{stamp}.csv"
    )
    write_csv(witness_csv, witness_rows)

    source_files = {
        "protocol": D9_CFG_PATH,
        "d7_latest": D7_PATH,
        "d7b_latest": D7B_PATH,
        "d8_certificate": D8_CERT_PATH,
        "d7_candidate_csv": d7_candidate_csv,
        "d7b_witness_csv": d7b_witness_csv,
        "threshold_witness_csv": witness_csv,
    }

    certificate = {
        "schema":
            "SCV_P3B1_R7D9_KERNEL_SET_SEPARATION_CERTIFICATE_V1",
        "timestamp_utc":
            stamp,
        "scope":
            (
                "Certificate-backed set-separation witnesses for the Level-B "
                "refined finite abstraction under membership semantics "
                "d >= h_star(y,q). This certifies existence of states with "
                "different viability membership across q in the recorded "
                "finite abstraction; it is not a maximal continuous-domain, "
                "Farkas, or implementation-refinement certificate."
            ),
        "membership_semantics":
            protocol["membership_semantics"],
        "source_files": {
            key: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for key, path in source_files.items()
        },
        "witness_count":
            len(certificate_rows),
        "profiles":
            list(profiles),
        "witnesses":
            certificate_rows,
        "aggregate": {
            "minimum_certified_membership_margin_exact_rational":
                frac_obj(min_cert_margin),
            "minimum_certified_membership_margin_m":
                format(float(min_cert_margin), ".17g"),
            "minimum_raw_halfspan_exact_rational":
                frac_obj(min_raw_halfspan),
            "minimum_raw_halfspan_m":
                format(float(min_raw_halfspan), ".17g"),
        },
        "claims": {
            "finite_abstraction_kernel_set_separation_certificate":
                True,
            "certificate_backed_q_dependent_kernel_membership":
                True,
            "scientific_kernel_claim_authorized":
                False,
            "maximal_continuous_kernel_claim_authorized":
                False,
            "farkas_certificate_authorized":
                False,
            "implementation_refinement_claim_authorized":
                False,
        },
    }

    cert_path = (
        RESULTS_DIR
        / f"P3B1_R7D9_KERNEL_SET_SEPARATION_CERTIFICATE_{stamp}.json"
    )
    latest_path = RESULTS_DIR / "P3B1_R7D9_CERTIFICATE_LATEST.json"

    atomic_write(
        cert_path,
        json.dumps(certificate, indent=2, sort_keys=True) + "\n",
    )
    atomic_write(
        latest_path,
        json.dumps(certificate, indent=2, sort_keys=True) + "\n",
    )

    print("=== P3-B1-R7-D9 KERNEL SET-SEPARATION GENERATION ===")
    print(f"WITNESS_COUNT={len(certificate_rows)}")
    print(f"PROFILE_COUNT={len(profiles)}")
    print(
        "MIN_CERTIFIED_MEMBERSHIP_MARGIN_M="
        f"{float(min_cert_margin):.12g}"
    )
    print(
        "FINITE_ABSTRACTION_KERNEL_SET_SEPARATION_CERTIFICATE=YES"
    )
    print("SCIENTIFIC_KERNEL_CLAIM_AUTHORIZED=NO")
    print("FARKAS_CERTIFICATE_AUTHORIZED=NO")
    print("IMPLEMENTATION_REFINEMENT_CLAIM=NO")
    print(f"CERTIFICATE_JSON={cert_path}")
    print(f"CERTIFICATE_LATEST={latest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
