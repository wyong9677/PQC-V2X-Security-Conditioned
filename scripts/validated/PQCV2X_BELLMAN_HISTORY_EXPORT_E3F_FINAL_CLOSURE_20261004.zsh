#!/bin/zsh -f
set -euo pipefail
umask 077
setopt null_glob

###############################################################################
# PQC-V2X numerical supplement closure
# Stage E3F: FINAL CLAIM-BOUNDARY AUDIT AND FREEZE
#
# No E5 rerun. No source modification.
#
# Canonical supported claim:
#   Across four matched ML-DSA-65 / SLH-DSA-SHA2-192s real-profile solves,
#   the terminal common projected `required` vector is elementwise exactly
#   equal on all 94,325 physical nodes (max absolute gap 0.0).
#
# Explicitly NOT established:
#   - full Bellman-trajectory equality;
#   - equality of profile-specific internal `h` states;
#   - continuous maximal-kernel equality;
#   - witness service equivalence;
#   - implementation supply / deployment qualification.
###############################################################################

TARGET_ROOT="$HOME/Desktop/paper set/PQC_V2X_Security_Conditioned/numerical_experiments"
E3E_RUN="$TARGET_ROOT/BELLMAN_HISTORY_EXPORT_E3E_COMMON_PROJECTION_20261004T105118Z"
SUMMARY="$E3E_RUN/common_projection_summary.json"
E3E_STATUS="$E3E_RUN/E3E_STATUS.txt"
E3E_ARCHIVE="$TARGET_ROOT/BELLMAN_HISTORY_EXPORT_E3E_COMMON_PROJECTION_20261004T105118Z.tar.gz"
EXPECTED_E3E_ARCHIVE_SHA256="5fe4aef47736a5809a75f772d49ed659f4b310252cf5bfe29bdc5187f02473ca"

CANONICAL_SOURCE="$TARGET_ROOT/02_src/e5_real_profile_closure.py"
EXPECTED_CANONICAL_SHA256="33a1218eb777030be8b1b856d10600e739df5c4615d7ceff0f446ac5f0b2a321"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="BELLMAN_HISTORY_EXPORT_E3F_FINAL_CLOSURE_${STAMP}"
RUN_ROOT="$TARGET_ROOT/$RUN_NAME"
VALIDATED_ROOT="$TARGET_ROOT/scripts/validated"
SELF_PATH="${0:A}"
RESEARCH_PYTHON="$HOME/envs/research/bin/python"

mkdir -p "$RUN_ROOT"
LOG="$RUN_ROOT/master.log"
exec > >(tee -a "$LOG") 2>&1

CURRENT_STEP="INITIALIZATION"
on_error() {
    local rc=$?
    trap - ERR
    {
        echo "E3F_STATUS=FAIL"
        echo "FAILED_STEP=$CURRENT_STEP"
        echo "EXIT_CODE=$rc"
        echo "RUN_ROOT=$RUN_ROOT"
        echo "E5_RERUN_PERFORMED=NO"
        echo "SOURCE_FILES_MODIFIED=NO"
        echo "FAILED_EVIDENCE_RETAINED=YES"
    } | tee "$RUN_ROOT/FAIL_STATUS.txt"
    exit "$rc"
}
trap on_error ERR

fail() {
    echo "ERROR=$1"
    return 1
}

echo "======================================================================"
echo "PQC-V2X BELLMAN HISTORY E3F — FINAL CLAIM-BOUNDARY CLOSURE"
echo "UTC_STAMP=$STAMP"
echo "E3E_RUN=$E3E_RUN"
echo "RUN_ROOT=$RUN_ROOT"
echo "======================================================================"
echo

###############################################################################
# STEP 0 — preflight / identity
###############################################################################

CURRENT_STEP="STEP_0_PREFLIGHT"
echo "======================================================================"
echo "STEP 0 — PREFLIGHT"
echo "======================================================================"

[[ -d "$E3E_RUN" ]] || fail "E3E_RUN_NOT_FOUND:$E3E_RUN"
[[ -f "$SUMMARY" ]] || fail "COMMON_PROJECTION_SUMMARY_NOT_FOUND:$SUMMARY"
[[ -f "$E3E_STATUS" ]] || fail "E3E_STATUS_NOT_FOUND:$E3E_STATUS"
[[ -f "$E3E_ARCHIVE" ]] || fail "E3E_ARCHIVE_NOT_FOUND:$E3E_ARCHIVE"
[[ -f "$CANONICAL_SOURCE" ]] || fail "CANONICAL_SOURCE_NOT_FOUND:$CANONICAL_SOURCE"
[[ -x "$RESEARCH_PYTHON" ]] || fail "RESEARCH_PYTHON_NOT_FOUND:$RESEARCH_PYTHON"

CANONICAL_SHA="$(shasum -a 256 "$CANONICAL_SOURCE" | awk '{print $1}')"
E3E_ARCHIVE_SHA="$(shasum -a 256 "$E3E_ARCHIVE" | awk '{print $1}')"

echo "EXPECTED_CANONICAL_SHA256=$EXPECTED_CANONICAL_SHA256"
echo "ACTUAL_CANONICAL_SHA256=$CANONICAL_SHA"
echo "EXPECTED_E3E_ARCHIVE_SHA256=$EXPECTED_E3E_ARCHIVE_SHA256"
echo "ACTUAL_E3E_ARCHIVE_SHA256=$E3E_ARCHIVE_SHA"

[[ "$CANONICAL_SHA" == "$EXPECTED_CANONICAL_SHA256" ]] || \
    fail "CANONICAL_SOURCE_SHA256_MISMATCH"
[[ "$E3E_ARCHIVE_SHA" == "$EXPECTED_E3E_ARCHIVE_SHA256" ]] || \
    fail "E3E_ARCHIVE_SHA256_MISMATCH"

grep -Fxq "E3E_STATUS=PASS" "$E3E_STATUS" || fail "E3E_STATUS_NOT_PASS"
grep -Fxq "PROJECT_IMMUTABILITY=PASS" "$E3E_STATUS" || fail "E3E_IMMUTABILITY_NOT_PASS"
grep -Fxq "PROFILE_PAIRING_STATUS=PASS" "$E3E_STATUS" || fail "E3E_PAIRING_NOT_PASS"
grep -Fxq "COMMON_PROJECTION_STATUS=PASS" "$E3E_STATUS" || fail "E3E_COMMON_PROJECTION_NOT_PASS"

echo "PREFLIGHT=PASS"
echo "E5_RERUN_PERFORMED=NO"
echo

###############################################################################
# STEP 1 — strict evidence audit
###############################################################################

CURRENT_STEP="STEP_1_STRICT_EVIDENCE_AUDIT"
echo "======================================================================"
echo "STEP 1 — STRICT EVIDENCE AUDIT"
echo "======================================================================"

"$RESEARCH_PYTHON" - "$SUMMARY" "$RUN_ROOT" <<'PY'
from __future__ import annotations
import json
import math
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
run_root = Path(sys.argv[2])
data = json.loads(summary_path.read_text(encoding="utf-8"))

pairs = data.get("pair_summaries", [])
structures = data.get("structure_rows", [])

if len(pairs) != 4:
    raise SystemExit(f"EXPECTED_4_PAIRS_GOT_{len(pairs)}")
if len(structures) != 8:
    raise SystemExit(f"EXPECTED_8_STRUCTURE_ROWS_GOT_{len(structures)}")

pair_report = []
for p in pairs:
    idx = int(p["pair_index"])
    arrays = p.get("comparable_arrays", {})
    if set(arrays) != {"required"}:
        raise SystemExit(
            f"PAIR_{idx}_COMPARABLE_ARRAYS_NOT_EXACTLY_REQUIRED:"
            + ",".join(sorted(arrays))
        )
    req = arrays["required"]

    if req.get("shape") not in ("94325", [94325]):
        raise SystemExit(f"PAIR_{idx}_REQUIRED_SHAPE_UNEXPECTED:{req.get('shape')}")
    if req.get("terminal_exact_equal") is not True:
        raise SystemExit(f"PAIR_{idx}_TERMINAL_REQUIRED_NOT_EXACT_EQUAL")
    if req.get("different_capture_count") != 0:
        raise SystemExit(f"PAIR_{idx}_REQUIRED_DIFFERENT_CAPTURE_COUNT_NONZERO")
    gap = float(req.get("max_abs_gap"))
    if gap != 0.0:
        raise SystemExit(f"PAIR_{idx}_REQUIRED_MAX_GAP_NONZERO:{gap}")

    pair_report.append({
        "pair_index": idx,
        "ml_call_id": p["ml_call_id"],
        "slh_call_id": p["slh_call_id"],
        "required_shape": req["shape"],
        "terminal_exact_equal": True,
        "max_abs_gap": 0.0,
    })

# Verify that no intermediate common projected vector exists in the captured
# phases, and that `h` is explicitly shape-incompatible.
by_pair = {}
for row in structures:
    idx = int(row["pair_index"])
    by_pair.setdefault(idx, []).append(row)

for idx in range(1, 5):
    rows = sorted(by_pair[idx], key=lambda x: int(x["capture_index"]))
    if [r["phase"] for r in rows] != ["loop_entry", "terminal"]:
        raise SystemExit(f"PAIR_{idx}_UNEXPECTED_PHASE_SEQUENCE")

    loop_row, terminal_row = rows

    if int(loop_row["same_shape_dtype_numeric_count"]) != 0:
        raise SystemExit(f"PAIR_{idx}_LOOP_ENTRY_HAS_COMPARABLE_ARRAY")
    if "h" not in str(loop_row.get("shape_mismatch_arrays", "")).split(";"):
        raise SystemExit(f"PAIR_{idx}_LOOP_ENTRY_H_NOT_SHAPE_MISMATCH")

    if int(terminal_row["same_shape_dtype_numeric_count"]) != 1:
        raise SystemExit(f"PAIR_{idx}_TERMINAL_COMPARABLE_COUNT_NOT_ONE")
    if "h" not in str(terminal_row.get("shape_mismatch_arrays", "")).split(";"):
        raise SystemExit(f"PAIR_{idx}_TERMINAL_H_NOT_SHAPE_MISMATCH")

audit = {
    "matched_pair_count": 4,
    "terminal_common_projected_array": "required",
    "terminal_required_vector_length": 94325,
    "all_four_pairs_terminal_required_exact_equal": True,
    "all_four_pairs_terminal_required_max_abs_gap": 0.0,
    "profile_specific_internal_h_shape_compatible": False,
    "comparable_intermediate_common_projection_available": False,
    "full_bellman_trajectory_equality_established": False,
    "pair_report": pair_report,
}
(run_root / "FINAL_EVIDENCE_AUDIT.json").write_text(
    json.dumps(audit, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("MATCHED_PAIR_COUNT=4")
print("TERMINAL_COMMON_PROJECTED_ARRAY=required")
print("TERMINAL_REQUIRED_VECTOR_LENGTH=94325")
print("ALL_FOUR_PAIRS_TERMINAL_REQUIRED_EXACT_EQUAL=YES")
print("ALL_FOUR_PAIRS_TERMINAL_REQUIRED_MAX_ABS_GAP=0.0")
print("PROFILE_SPECIFIC_INTERNAL_H_SHAPE_COMPATIBLE=NO")
print("COMPARABLE_INTERMEDIATE_COMMON_PROJECTION_AVAILABLE=NO")
print("FULL_BELLMAN_TRAJECTORY_EQUALITY_ESTABLISHED=NO")
print("STRICT_EVIDENCE_AUDIT=PASS")
PY

echo

###############################################################################
# STEP 2 — canonical manuscript-safe interpretation
###############################################################################

CURRENT_STEP="STEP_2_CANONICAL_INTERPRETATION"
echo "======================================================================"
echo "STEP 2 — CANONICAL MANUSCRIPT-SAFE INTERPRETATION"
echo "======================================================================"

cat > "$RUN_ROOT/FINAL_MANUSCRIPT_SAFE_INTERPRETATION.md" <<'EOF'
# Final manuscript-safe interpretation

Across the four matched ML-DSA-65 / SLH-DSA-SHA2-192s real-profile
fixed-point call pairs, the terminal common projected `required` vector has
length 94,325 and is elementwise exactly equal in every pair. The maximum
absolute difference is 0.0 in all four comparisons.

This strengthens the previously reported terminal paired-gap result by showing
that the zero gap is not produced by aggregation alone: the complete terminal
projected required-spacing vector agrees nodewise on the tested common
physical geometry.

The result does **not** establish equality of the complete Bellman
trajectories. At the captured `loop_entry` phase there is no structurally
compatible common numeric array, and the profile-specific internal `h` arrays
have different shapes for ML-DSA-65 and SLH-DSA-SHA2-192s. The two service
automata therefore retain different internal finite-state representations even
though their terminal projected required-spacing vectors coincide.

The result is restricted to the declared finite abstraction and tested common
geometry. It does not establish continuous maximal-kernel equality, mutual
witness service dominance, formal implementation supply, or deployment-level
qualification.
EOF

cat "$RUN_ROOT/FINAL_MANUSCRIPT_SAFE_INTERPRETATION.md"

echo
echo "CANONICAL_CLAIM_CLASS=TERMINAL_COMMON_PROJECTED_VECTOR_EXACT_EQUALITY"
echo "FULL_TRAJECTORY_EQUALITY_CLAIM=PROHIBITED"
echo "MANUSCRIPT_SAFE_INTERPRETATION=PASS"
echo

###############################################################################
# STEP 3 — publication decision
###############################################################################

CURRENT_STEP="STEP_3_PUBLICATION_DECISION"
echo "======================================================================"
echo "STEP 3 — PUBLICATION DECISION"
echo "======================================================================"

cat > "$RUN_ROOT/PUBLICATION_DECISION.md" <<'EOF'
# Publication decision

## Recommended use

Retain the existing headline result
\[
\Delta_{\mathrm{ML,SLH},h}^{\mathrm{pair}}=0
\]
as the canonical cross-profile finite comparison.

Add at most one supporting sentence in the numerical section or appendix:

> Across four matched real-profile fixed-point solves, the terminal common
> projected required-spacing vectors were elementwise identical on all
> 94,325 physical nodes (maximum absolute difference \(0\)); the
> profile-specific internal Bellman states remained structurally different.

## Do not claim

- equality of complete Bellman trajectories;
- equality of profile-specific internal service states;
- mutual witness service equivalence;
- continuous maximal-kernel equality;
- implementation-level service equivalence;
- formal demand--supply qualification.

## Further numerical expansion

STOP. No additional numerical experiment is required for this question.
The remaining manuscript gaps are structural: finite-to-continuous transfer
and formal implementation supply cannot be closed by adding more experiments
of the same class.
EOF

cat "$RUN_ROOT/PUBLICATION_DECISION.md"

echo "FURTHER_NUMERICAL_EXPANSION=STOP"
echo "PUBLICATION_DECISION=PASS"
echo

###############################################################################
# STEP 4 — freeze / archive
###############################################################################

CURRENT_STEP="STEP_4_FREEZE"
echo "======================================================================"
echo "STEP 4 — FREEZE"
echo "======================================================================"

STATUS="$RUN_ROOT/E3F_STATUS.txt"
RUNNER_SHA="$(shasum -a 256 "$SELF_PATH" | awk '{print $1}')"

{
    echo "E3F_STATUS=PASS"
    echo "CANONICAL_CLAIM_CLASS=TERMINAL_COMMON_PROJECTED_VECTOR_EXACT_EQUALITY"
    echo "MATCHED_PAIR_COUNT=4"
    echo "TERMINAL_REQUIRED_VECTOR_LENGTH=94325"
    echo "ALL_FOUR_PAIRS_TERMINAL_REQUIRED_EXACT_EQUAL=YES"
    echo "MAX_ABS_GAP=0.0"
    echo "FULL_BELLMAN_TRAJECTORY_EQUALITY_ESTABLISHED=NO"
    echo "FURTHER_NUMERICAL_EXPANSION=STOP"
    echo "E5_RERUN_PERFORMED=NO"
    echo "SOURCE_FILES_MODIFIED=NO"
    echo "RUNNER_SHA256=$RUNNER_SHA"
    echo "NEXT_ACTION=OPTIONALLY_ADD_ONE_SUPPORTING_SENTENCE_TO_MANUSCRIPT_THEN_RUN_FINAL_STATIC_AUDIT"
} | tee "$STATUS"

MANIFEST="$RUN_ROOT/SHA256_MANIFEST.txt"
(
    cd "$RUN_ROOT"
    find . -type f ! -name SHA256_MANIFEST.txt -print0 \
    | sort -z \
    | xargs -0 shasum -a 256
) > "$MANIFEST"

ARCHIVE="$TARGET_ROOT/${RUN_NAME}.tar.gz"
tar -C "$TARGET_ROOT" -czf "$ARCHIVE" "$RUN_NAME"
tar -tzf "$ARCHIVE" >/dev/null
ARCHIVE_SHA="$(shasum -a 256 "$ARCHIVE" | awk '{print $1}')"

mkdir -p "$VALIDATED_ROOT"
VALIDATED_RUNNER="$VALIDATED_ROOT/$(basename "$SELF_PATH")"
cp -p "$SELF_PATH" "$VALIDATED_RUNNER"
VALIDATED_RUNNER_SHA="$(shasum -a 256 "$VALIDATED_RUNNER" | awk '{print $1}')"

echo "MANIFEST=$MANIFEST"
echo "ARCHIVE=$ARCHIVE"
echo "ARCHIVE_SHA256=$ARCHIVE_SHA"
echo "ARCHIVE_INTEGRITY=PASS"
echo "VALIDATED_RUNNER=$VALIDATED_RUNNER"
echo "VALIDATED_RUNNER_SHA256=$VALIDATED_RUNNER_SHA"

echo
echo "======================================================================"
cat "$STATUS"
echo "RUN_ROOT=$RUN_ROOT"
echo "ENGINEERING_HYGIENE=PASS"
echo "======================================================================"
