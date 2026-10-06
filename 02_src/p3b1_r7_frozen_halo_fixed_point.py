from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import qmc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

import p3b1_augmented_fixed_point_v1_r3 as r3
import p3b1_r5_refinement_attribution as r5
import p3b1_r6_continuation_halo as r6

CFG_PATH = ROOT / "01_config" / "p3b1_augmented_fixed_point_v1.json"
P1_CFG_PATH = ROOT / "01_config" / "p1_validation_v2.json"
P2B_CFG_PATH = ROOT / "01_config" / "p2b_hybrid_fallback_v1.json"
P2C_CFG_PATH = ROOT / "01_config" / "p2c_switching_guard_v1.json"
P3A_CFG_PATH = ROOT / "01_config" / "p3a_information_contract_v1.json"
R6_PATH = ROOT / "04_results" / "P3B1_R6_LATEST.json"

RESULTS_DIR = ROOT / "04_results"
FIGURES_DIR = ROOT / "05_figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

AXIS_NAMES = ("v_f", "v_p", "a_f", "bar_a", "bar_u", "age")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def mesh_flat(axes):
    mesh = np.meshgrid(*axes, indexing="ij")
    flat = [x.reshape(-1) for x in mesh]
    shape = tuple(len(axis) for axis in axes)
    return flat, shape


def exact_node_indices(eval_axes, lookup_axes):
    per_axis = []
    for eval_axis, lookup_axis in zip(eval_axes, lookup_axes):
        idx = []
        for x in eval_axis:
            matches = np.flatnonzero(
                np.isclose(
                    lookup_axis,
                    x,
                    rtol=0.0,
                    atol=1e-12,
                )
            )
            if len(matches) != 1:
                raise RuntimeError(
                    "EVAL_NODE_NOT_UNIQUELY_EMBEDDED_IN_HALO"
                )
            idx.append(int(matches[0]))
        per_axis.append(np.asarray(idx, dtype=int))

    grid = np.meshgrid(*per_axis, indexing="ij")
    lookup_shape = tuple(len(axis) for axis in lookup_axes)

    flat_indices = np.ravel_multi_index(
        tuple(g.reshape(-1) for g in grid),
        lookup_shape,
    )

    return flat_indices


def build_transition_data_to_lookup(
    cfg,
    p1_cfg,
    p2b_cfg,
    p2c_cfg,
    p3a_cfg,
    eval_flat,
    lookup_axes,
):
    # r3.build_transition_data only uses `axes` to locate successor cells;
    # the supplied state vectors can remain the evaluation nodes.
    return r3.build_transition_data(
        cfg,
        p1_cfg,
        p2b_cfg,
        p2c_cfg,
        p3a_cfg,
        lookup_axes,
        eval_flat,
    )


def service_horizon(profile_name: str, profile: dict) -> int:
    bound = int(profile["eligible_bound_steps"])
    if profile_name == "ideal":
        return 1
    return max(1, bound)


def solve_frozen_halo_profile(
    name: str,
    R: int,
    cfg: dict,
    eval_node_indices: np.ndarray,
    eval_node_count: int,
    lookup_shape,
    lookup_fallback: np.ndarray,
    eval_fallback: np.ndarray,
    transition_data: dict,
):
    lookup_nodes = int(np.prod(lookup_shape))

    if len(lookup_fallback) != lookup_nodes:
        raise RuntimeError("LOOKUP_FALLBACK_SIZE_MISMATCH")

    h_flat = np.repeat(
        lookup_fallback[:, None],
        R,
        axis=1,
    )

    tol = float(cfg["fixed_point"]["tolerance_m"])
    max_iterations = int(cfg["fixed_point"]["max_iterations"])

    history = []
    monotone_violation = 0.0
    outside_change_max = 0.0

    eval_mask = np.zeros(lookup_nodes, dtype=bool)
    eval_mask[eval_node_indices] = True
    outside = ~eval_mask

    for iteration in range(1, max_iterations + 1):
        h = h_flat.reshape(lookup_shape + (R,))
        cell_max = r3.cell_corner_max(h)

        cell_flat = [
            cell_max[..., q].reshape(-1)
            for q in range(R)
        ]

        old_eval = h_flat[eval_node_indices, :].copy()
        new_eval = old_eval.copy()

        for r in range(1, R + 1):
            q_index = r - 1

            best = np.full(
                eval_node_count,
                np.inf,
                dtype=float,
            )

            for trans in transition_data["transitions"]:
                completion_future = np.full(
                    eval_node_count,
                    np.inf,
                    dtype=float,
                )

                cmask = trans["completion_valid"]
                completion_future[cmask] = (
                    cell_flat[R - 1][
                        trans["completion_index"][cmask]
                    ]
                )

                future = completion_future

                if r > 1:
                    defer_future = np.full(
                        eval_node_count,
                        np.inf,
                        dtype=float,
                    )

                    dmask = trans["defer_valid"]
                    defer_future[dmask] = (
                        cell_flat[r - 2][
                            trans["defer_index"][dmask]
                        ]
                    )

                    future = np.maximum(
                        completion_future,
                        defer_future,
                    )

                required = np.maximum(
                    trans["step_loss_upper"],
                    trans["closing_end"] + future,
                )

                best = np.minimum(
                    best,
                    required,
                )

            best = np.maximum(best, 0.0)

            candidate = np.minimum(
                eval_fallback,
                best,
            )

            new_eval[:, q_index] = np.minimum(
                old_eval[:, q_index],
                candidate,
            )

        violation = float(
            np.max(new_eval - old_eval)
        )
        monotone_violation = max(
            monotone_violation,
            violation,
        )

        delta = float(
            np.max(
                np.abs(
                    new_eval - old_eval
                )
            )
        )

        h_flat[eval_node_indices, :] = new_eval

        # Halo nodes outside D_eval must remain exactly frozen at fallback.
        expected_outside = np.repeat(
            lookup_fallback[outside, None],
            R,
            axis=1,
        )

        outside_change = float(
            np.max(
                np.abs(
                    h_flat[outside, :]
                    -
                    expected_outside
                )
            )
        ) if np.any(outside) else 0.0

        outside_change_max = max(
            outside_change_max,
            outside_change,
        )

        history.append(
            {
                "iteration": iteration,
                "sup_change_m": delta,
                "mean_eval_required_gap_m": float(
                    np.mean(
                        new_eval[:, R - 1]
                    )
                ),
                "min_eval_required_gap_m": float(
                    np.min(
                        new_eval[:, R - 1]
                    )
                ),
                "max_eval_required_gap_m": float(
                    np.max(
                        new_eval[:, R - 1]
                    )
                ),
            }
        )

        if delta <= tol:
            return {
                "name": name,
                "R": R,
                "h_flat": h_flat,
                "history": history,
                "converged": True,
                "iterations": iteration,
                "final_change": delta,
                "monotone_violation": monotone_violation,
                "outside_halo_change_max": outside_change_max,
            }

    return {
        "name": name,
        "R": R,
        "h_flat": h_flat,
        "history": history,
        "converged": False,
        "iterations": max_iterations,
        "final_change": history[-1]["sup_change_m"],
        "monotone_violation": monotone_violation,
        "outside_halo_change_max": outside_change_max,
    }


def qmc_augmented_states(power, seed, p2b_cfg, p3a_cfg):
    sampler = qmc.Sobol(
        d=7,
        scramble=True,
        seed=seed,
    )
    U = sampler.random_base2(int(power))

    sd = p2b_cfg["state_domain"]
    pc = p3a_cfg["predecessor"]

    lower = np.array(
        [
            sd["d_min"],
            sd["v_f_min"],
            sd["v_p_min"],
            sd["a_f_min"],
            pc["command_min"],
            pc["command_min"],
            0.0,
        ],
        dtype=float,
    )

    upper = np.array(
        [
            sd["d_max"],
            sd["v_f_max"],
            sd["v_p_max"],
            sd["a_f_max"],
            pc["command_max"],
            pc["command_max"],
            2.0,
        ],
        dtype=float,
    )

    return lower + U * (upper - lower)


def lookup_required_gap(
    h_flat,
    lookup_shape,
    lookup_axes,
    values,
    q_index,
):
    R = h_flat.shape[1]
    h = h_flat.reshape(lookup_shape + (R,))
    cell = r3.cell_corner_max(h)
    flat = cell[..., q_index].reshape(-1)

    idx, valid = r3.locate_cells(
        lookup_axes,
        values,
    )

    result = np.full(
        len(idx),
        np.inf,
        dtype=float,
    )

    result[valid] = flat[idx[valid]]
    return result, valid


def evaluate_common_domain(
    cfg,
    p2b_cfg,
    p3a_cfg,
    lookup_axes,
    lookup_shape,
    lookup_fallback,
    profile_results,
):
    X = qmc_augmented_states(
        int(cfg["evaluation"]["qmc_power"]),
        int(cfg["random_seed"]) + 17000,
        p2b_cfg,
        p3a_cfg,
    )

    d = X[:, 0]
    values = [
        X[:, 1],
        X[:, 2],
        X[:, 3],
        X[:, 4],
        X[:, 5],
        X[:, 6],
    ]

    d_min = float(
        p2b_cfg["state_domain"]["d_min"]
    )

    fallback_h = lookup_fallback[:, None]
    fallback_req, fallback_valid = lookup_required_gap(
        fallback_h,
        lookup_shape,
        lookup_axes,
        values,
        0,
    )

    verdicts = {
        "fallback":
            d >= d_min + fallback_req
    }

    required = {
        "fallback":
            fallback_req
    }

    all_valid = fallback_valid.copy()

    for name, result in profile_results.items():
        req, valid = lookup_required_gap(
            result["h_flat"],
            lookup_shape,
            lookup_axes,
            values,
            result["R"] - 1,
        )

        required[name] = req
        verdicts[name] = (
            d >= d_min + req
        )
        all_valid &= valid

    fractions = {
        name: float(np.mean(verdict))
        for name, verdict in verdicts.items()
    }

    return {
        "X": X,
        "required": required,
        "verdicts": verdicts,
        "fractions": fractions,
        "all_lookup_valid": bool(all_valid.all()),
        "invalid_lookup_count": int(
            np.count_nonzero(~all_valid)
        ),
    }


def make_convergence_figure(path, profile_results):
    fig, ax = plt.subplots(figsize=(7.6, 4.8))

    any_positive = False

    for name, result in profile_results.items():
        x = [
            row["iteration"]
            for row in result["history"]
        ]
        y = [
            row["sup_change_m"]
            for row in result["history"]
        ]

        if any(value > 0.0 for value in y):
            any_positive = True

        ax.plot(
            x,
            y,
            marker="o",
            markersize=3,
            label=name,
        )

    if any_positive:
        ax.set_yscale("log")

    ax.set_xlabel("Frozen-halo predecessor iteration")
    ax.set_ylabel("Sup-norm eval-domain gap change (m)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def make_fraction_figure(path, fractions):
    names = list(fractions.keys())
    values = [fractions[name] for name in names]

    fig, ax = plt.subplots(figsize=(7.5, 4.7))
    ax.bar(np.arange(len(names)), values)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=20)
    ax.set_ylabel("Common-domain candidate viable fraction")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def main():
    cfg = load_json(CFG_PATH)
    p1_cfg = load_json(P1_CFG_PATH)
    p2b_cfg = load_json(P2B_CFG_PATH)
    p2c_cfg = load_json(P2C_CFG_PATH)
    p3a_cfg = load_json(P3A_CFG_PATH)
    r6_result = load_json(R6_PATH)

    if r6_result.get("status") != "PASS":
        raise RuntimeError("R6_NOT_PASS")

    selected_axes = list(
        r6_result["selected_refinement_axes"]
    )

    eval_cfg = r5.refine_cfg(
        cfg,
        tuple(selected_axes),
    )

    eval_axes, eval_flat, eval_shape = r3.build_grid(
        eval_cfg
    )

    halo_spec = load_json(
        Path(
            r6_result["halo_spec"]
        )
    )

    lookup_axes = [
        np.asarray(
            halo_spec["lookup_halo_grid"][name],
            dtype=float,
        )
        for name in AXIS_NAMES
    ]

    lookup_flat, lookup_shape = r6.mesh_flat(
        lookup_axes
    )

    stop = r3.endpoint_semantics_audit(
        eval_cfg,
        p1_cfg,
        p2b_cfg,
        p3a_cfg,
        eval_flat,
    )

    if not stop["pass"]:
        raise RuntimeError("P3B1_R7_STOP_SEMANTICS_GATE=FAIL")

    eval_node_indices = exact_node_indices(
        eval_axes,
        lookup_axes,
    )

    lookup_fallback = r6.fallback_required_on_grid(
        eval_cfg,
        p1_cfg,
        p2b_cfg,
        p2c_cfg,
        p3a_cfg,
        lookup_flat,
    )

    transition_data = build_transition_data_to_lookup(
        eval_cfg,
        p1_cfg,
        p2b_cfg,
        p2c_cfg,
        p3a_cfg,
        eval_flat,
        lookup_axes,
    )

    completion_invalid = sum(
        int(
            np.count_nonzero(
                ~trans["completion_valid"]
            )
        )
        for trans in transition_data["transitions"]
    )

    defer_invalid = sum(
        int(
            np.count_nonzero(
                ~trans["defer_valid"]
            )
        )
        for trans in transition_data["transitions"]
    )

    if completion_invalid != 0 or defer_invalid != 0:
        raise RuntimeError(
            "P3B1_R7_FROZEN_HALO_COVERAGE=FAIL"
        )

    eval_fallback = np.asarray(
        transition_data["fallback_required"],
        dtype=float,
    )

    profiles = p1_cfg[
        "diagnostic_service_profiles"
    ]

    profile_order = [
        "ideal",
        "diagnostic_fast",
        "diagnostic_nominal",
        "diagnostic_stressed",
    ]

    profile_results = {}

    print("=== P3-B1-R7 FROZEN-HALO REFINED FIXED POINT ===")
    print("STOP_SEMANTICS_GATE=PASS")
    print("FROZEN_HALO_COVERAGE=PASS")
    print(
        "SELECTED_REFINEMENT_AXES="
        + ",".join(selected_axes)
    )
    print(
        "EVAL_GRID_NODES="
        f"{int(np.prod(eval_shape))}"
    )
    print(
        "LOOKUP_GRID_NODES="
        f"{int(np.prod(lookup_shape))}"
    )

    for name in profile_order:
        R = service_horizon(
            name,
            profiles[name],
        )

        result = solve_frozen_halo_profile(
            name,
            R,
            eval_cfg,
            eval_node_indices,
            len(eval_node_indices),
            lookup_shape,
            lookup_fallback,
            eval_fallback,
            transition_data,
        )

        profile_results[name] = result

        print(
            f"{name.upper()} "
            f"R={R} "
            f"ITER={result['iterations']} "
            "CONVERGED="
            f"{'YES' if result['converged'] else 'NO'} "
            "FINAL_CHANGE_M="
            f"{result['final_change']:.12g} "
            "HALO_CHANGE_MAX="
            f"{result['outside_halo_change_max']:.12g}"
        )

    evaluation = evaluate_common_domain(
        eval_cfg,
        p2b_cfg,
        p3a_cfg,
        lookup_axes,
        lookup_shape,
        lookup_fallback,
        profile_results,
    )

    verdicts = evaluation["verdicts"]
    fractions = evaluation["fractions"]

    fallback = verdicts["fallback"]
    ideal = verdicts["ideal"]
    fast = verdicts["diagnostic_fast"]
    nominal = verdicts["diagnostic_nominal"]
    stressed = verdicts["diagnostic_stressed"]

    fallback_inclusion_violations = {
        name: int(
            np.count_nonzero(
                fallback
                &
                ~verdicts[name]
            )
        )
        for name in profile_order
    }

    service_order_violations = {
        "fast_not_subset_ideal":
            int(np.count_nonzero(fast & ~ideal)),
        "nominal_not_subset_fast":
            int(np.count_nonzero(nominal & ~fast)),
        "stressed_not_subset_nominal":
            int(np.count_nonzero(stressed & ~nominal)),
    }

    strict_gain = {
        "ideal_over_fallback":
            int(np.count_nonzero(ideal & ~fallback)),
        "fast_over_fallback":
            int(np.count_nonzero(fast & ~fallback)),
        "nominal_over_fallback":
            int(np.count_nonzero(nominal & ~fallback)),
        "stressed_over_fallback":
            int(np.count_nonzero(stressed & ~fallback)),
        "ideal_over_stressed":
            int(np.count_nonzero(ideal & ~stressed)),
        "fast_over_stressed":
            int(np.count_nonzero(fast & ~stressed)),
    }

    all_converged = all(
        result["converged"]
        for result in profile_results.values()
    )

    max_monotone_violation = max(
        result["monotone_violation"]
        for result in profile_results.values()
    )

    max_halo_change = max(
        result["outside_halo_change_max"]
        for result in profile_results.values()
    )

    checks = {
        "STOP_SEMANTICS_GATE":
            bool(stop["pass"]),
        "FROZEN_HALO_COVERAGE":
            completion_invalid == 0
            and defer_invalid == 0,
        "EVAL_LOOKUP_EMBEDDING_COMPLETE":
            len(eval_node_indices)
            ==
            int(np.prod(eval_shape)),
        "FIXED_POINT_CONVERGED_ALL":
            all_converged,
        "MONOTONE_PREDECESSOR_ITERATION":
            max_monotone_violation <= 1e-12,
        "HALO_OUTSIDE_EVAL_FROZEN":
            max_halo_change <= 1e-12,
        "COMMON_DOMAIN_LOOKUP_VALID":
            bool(evaluation["all_lookup_valid"]),
        "FALLBACK_INCLUDED_ALL_PROFILES":
            sum(
                fallback_inclusion_violations.values()
            ) == 0,
        "SERVICE_ORDER_NESTED":
            sum(
                service_order_violations.values()
            ) == 0,
        "STRICT_COOPERATIVE_GAIN_EXISTS":
            strict_gain["ideal_over_fallback"] > 0,
        "STRICT_SERVICE_EFFECT_EXISTS":
            strict_gain["fast_over_stressed"] > 0,
    }

    status = "PASS" if all(checks.values()) else "FAIL"

    print("=== COMMON-DOMAIN FRACTIONS ===")
    for name, value in fractions.items():
        print(
            f"{name.upper()}="
            f"{value:.12g}"
        )

    print("=== STRICT GAINS ===")
    for name, value in strict_gain.items():
        print(
            f"{name.upper()}="
            f"{value}"
        )

    stamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    convergence_rows = []
    for name, result in profile_results.items():
        for row in result["history"]:
            convergence_rows.append(
                {
                    "profile": name,
                    **row,
                }
            )

    convergence_csv = (
        RESULTS_DIR
        /
        f"P3B1_R7_FIXED_POINT_CONVERGENCE_{stamp}.csv"
    )
    write_csv(
        convergence_csv,
        convergence_rows,
    )

    witness_mask = fast & ~stressed
    witness_indices = np.flatnonzero(witness_mask)

    X = evaluation["X"]
    witness_rows = []

    for i in witness_indices[
        :int(
            cfg["evaluation"]["witness_max_rows"]
        )
    ]:
        witness_rows.append(
            {
                "sample_index": int(i),
                "d": float(X[i, 0]),
                "v_f": float(X[i, 1]),
                "v_p": float(X[i, 2]),
                "a_f": float(X[i, 3]),
                "bar_a": float(X[i, 4]),
                "bar_u": float(X[i, 5]),
                "age": float(X[i, 6]),
                "required_gap_fast_m":
                    float(
                        evaluation["required"][
                            "diagnostic_fast"
                        ][i]
                    ),
                "required_gap_stressed_m":
                    float(
                        evaluation["required"][
                            "diagnostic_stressed"
                        ][i]
                    ),
            }
        )

    witness_csv = (
        RESULTS_DIR
        /
        f"P3B1_R7_SERVICE_WITNESSES_{stamp}.csv"
    )
    write_csv(
        witness_csv,
        witness_rows,
    )

    convergence_fig = (
        FIGURES_DIR
        /
        f"P3B1_R7_FIXED_POINT_CONVERGENCE_{stamp}.png"
    )

    fraction_fig = (
        FIGURES_DIR
        /
        f"P3B1_R7_COMMON_DOMAIN_FRACTIONS_{stamp}.png"
    )

    make_convergence_figure(
        convergence_fig,
        profile_results,
    )

    make_fraction_figure(
        fraction_fig,
        fractions,
    )

    metrics = {
        "evaluation_grid_nodes":
            int(np.prod(eval_shape)),
        "lookup_grid_nodes":
            int(np.prod(lookup_shape)),
        "common_qmc_samples":
            int(len(X)),
        "common_domain_invalid_lookup_count":
            int(evaluation["invalid_lookup_count"]),
        "max_monotone_iteration_violation":
            float(max_monotone_violation),
        "max_outside_halo_change":
            float(max_halo_change),
        **{
            f"fraction_{name}":
                float(value)
            for name, value in fractions.items()
        },
        **{
            f"strict_gain_{name}":
                int(value)
            for name, value in strict_gain.items()
        },
    }

    for name, result in profile_results.items():
        metrics[f"{name}_horizon_steps"] = int(
            result["R"]
        )
        metrics[f"{name}_iterations"] = int(
            result["iterations"]
        )
        metrics[f"{name}_final_change_m"] = float(
            result["final_change"]
        )

    output = {
        "schema":
            "SCV_P3B1_R7_FROZEN_HALO_REFINED_FIXED_POINT_V1",
        "status":
            status,
        "timestamp_utc":
            stamp,
        "classification":
            (
                "candidate frozen-halo finite-abstraction "
                "augmented fixed point; evaluation domain fixed; "
                "not P6 certified and not a maximal continuous kernel"
            ),
        "checks":
            {k: bool(v) for k, v in checks.items()},
        "selected_refinement_axes":
            selected_axes,
        "metrics":
            metrics,
        "fractions":
            fractions,
        "strict_gain_counts":
            strict_gain,
        "fallback_inclusion_violations":
            fallback_inclusion_violations,
        "service_order_violations":
            service_order_violations,
        "profile_summary": {
            name: {
                "R": int(result["R"]),
                "converged": bool(result["converged"]),
                "iterations": int(result["iterations"]),
                "final_change_m": float(
                    result["final_change"]
                ),
            }
            for name, result in profile_results.items()
        },
        "p6_certified":
            False,
        "maximal_continuous_kernel_claim":
            False,
        "final_pqc_service_profiles":
            False,
        "artifacts": {
            "convergence_csv":
                str(convergence_csv),
            "service_witness_csv":
                str(witness_csv),
            "convergence_figure":
                str(convergence_fig),
            "fraction_figure":
                str(fraction_fig),
        },
        "platform": {
            "python": sys.version,
            "executable": sys.executable,
            "system": platform.platform(),
            "numpy": np.__version__,
        },
    }

    result_path = (
        RESULTS_DIR
        /
        f"P3B1_R7_FIXED_POINT_{stamp}.json"
    )
    latest_path = (
        RESULTS_DIR
        /
        "P3B1_R7_LATEST.json"
    )

    text = json.dumps(
        output,
        indent=2,
        sort_keys=True,
    )
    atomic_write(result_path, text)
    atomic_write(latest_path, text)

    manifest_path = (
        RESULTS_DIR
        /
        f"P3B1_R7_MANIFEST_{stamp}.sha256"
    )

    manifest_files = [
        CFG_PATH,
        P1_CFG_PATH,
        P2B_CFG_PATH,
        P2C_CFG_PATH,
        P3A_CFG_PATH,
        R6_PATH,
        Path(r6_result["halo_spec"]),
        Path(__file__),
        result_path,
        convergence_csv,
        witness_csv,
        convergence_fig,
        fraction_fig,
    ]

    manifest = "\n".join(
        f"{sha256_file(path)}  {path}"
        for path in manifest_files
    ) + "\n"

    atomic_write(manifest_path, manifest)

    print(
        f"P3B1_R7_FIXED_POINT={status}"
    )
    print(
        "MAXIMAL_CONTINUOUS_KERNEL_CLAIM=NO"
    )
    print(
        "P6_CERTIFIED=NO"
    )
    print(
        "FINAL_PQC_SERVICE_PROFILES=NO"
    )
    print(
        f"RESULT_JSON={result_path}"
    )
    print(
        f"MANIFEST={manifest_path}"
    )

    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
