"""Records-only tests of absolute cross-query relational response predictions.

Each observed three-margin vector is patched minus native recipient output.
Directions are natural target endpoints minus that same native output. Scientific
rows have different donor and recipient queries; same-query measurements and
technical self-shams must be reported separately by the runner. This module does
not fit gains, run a model, or turn synthetic demonstrations into model evidence.

The error is RMS across the three pairwise margins, averaged across scientific
cells within a family. A .25-nat average RMS bound is NOT a maximum-margin bound.
Diagnostic cells additionally need their own RMS errors <= .25 nat.
"""

import copy
import math
from collections.abc import Mapping

import numpy as np

from query_route_analysis import InvalidControlsError, clopper_pearson
from role_geometry import closest_line_fit, rms_distance, validate_logits


PRECISIONS = ("float32", "float64")
QUERIES = ("giver", "receiver", "observer")
RIVALS = ("name", "position", "switch", "no_op")
DEFAULT_WORDINGS = ("fit", "heldout")
ERROR_TOLERANCE = .25
DIAGNOSTIC_GAP = .52
COVERAGE = .80
ADVANTAGE_BOUNDARY = .10
NUMERICAL_BUDGET = .01
MC_ENDPOINT_BUDGET = .01
MARGIN_IDENTITY_TOLERANCE = 1e-8
COVERAGE_ALPHA = .025
ADVANTAGE_ALPHA = .025
BOOTSTRAP_DRAWS = 20000
PRIMARY_SEED = 320260930
SECONDARY_SEED = 320260931


def _integer(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    try:
        value = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _vector(raw, label):
    try:
        vector = validate_logits(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must contain three finite pairwise margins") from exc
    if abs(vector[0] - vector[1] + vector[2]) > MARGIN_IDENTITY_TOLERANCE:
        raise ValueError(f"{label} violates (A-B)-(A-C)+(B-C)=0")
    return vector


def _controls(raw, label):
    if not isinstance(raw, Mapping) or raw.get("passed") is not True:
        raise InvalidControlsError(f"{label}.passed must be true")


def _ids(values, label, minimum=1):
    if not isinstance(values, (list, tuple)) or len(values) < minimum:
        raise ValueError(f"{label} must be a list/tuple with >= {minimum} entries")
    if any(not isinstance(v, str) or not v for v in values) or len(set(values)) != len(values):
        raise ValueError(f"{label} requires unique nonempty strings")
    return tuple(values)


def _design(expected_cell_ids, expected_wordings, expected_diagnostics, expected_structure):
    ids = _ids(expected_cell_ids, "expected_cell_ids", 2)
    wordings = _ids(expected_wordings, "expected_wordings", 2)
    if len(wordings) != 2:
        raise ValueError("the frozen contract requires exactly two wordings")
    if not isinstance(expected_diagnostics, Mapping) or set(expected_diagnostics) != set(RIVALS):
        raise ValueError("expected_diagnostics must contain name, position, switch, no_op")
    diagnostics = {r: _ids(expected_diagnostics[r], f"expected_diagnostics.{r}") for r in RIVALS}
    if any(not set(cells).issubset(ids) for cells in diagnostics.values()):
        raise ValueError("diagnostics must refer to expected cell IDs")
    if len(diagnostics["switch"]) != 2:
        raise ValueError("switch diagnostics must be the two rows of one query group")
    labels = {cell: sorted(r for r in RIVALS if cell in diagnostics[r]) for cell in ids}
    if not isinstance(expected_structure, Mapping) or set(expected_structure) != set(ids):
        raise ValueError("expected_structure must bind every expected cell ID exactly")
    structure, groups = {}, {}
    for cell_id in ids:
        entry = expected_structure[cell_id]
        if not isinstance(entry, Mapping) or set(entry) != {"group", "recipient_query", "donor_query"}:
            raise ValueError("expected_structure entries require exactly group, recipient_query, donor_query")
        group, qr, qd = entry["group"], entry["recipient_query"], entry["donor_query"]
        if (not isinstance(group, str) or not group or qr not in QUERIES
                or qd not in QUERIES or qr == qd):
            raise ValueError("expected_structure must describe named cross-query groups")
        structure[cell_id] = dict(entry)
        groups.setdefault(group, []).append((qr, qd))
    if any(len(rows) != 2 or len({qr for qr, qd in rows}) != 1
           or len({qd for qr, qd in rows}) != 2 for rows in groups.values()):
        raise ValueError("expected_structure requires two alternative donor queries per group")
    if len({structure[cell]["group"] for cell in diagnostics["switch"]}) != 1:
        raise ValueError("frozen switch diagnostics must belong to the same expected group")
    return ids, wordings, diagnostics, labels, structure


def _analyze_cells(raw, label, ids, expected_labels, gains, expected_structure):
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} must be a mapping")
    _controls(raw.get("controls"), f"{label}.controls")
    cells = raw.get("cells")
    if not isinstance(cells, (list, tuple)) or len(cells) != len(ids):
        raise ValueError(f"{label} must contain exactly {len(ids)} scientific cells")
    parsed, groups = {}, {}
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise ValueError(f"{label}.cell must be a mapping")
        cell_id = cell.get("cell_id")
        if not isinstance(cell_id, str) or cell_id not in ids or cell_id in parsed:
            raise ValueError(f"{label}: missing, unexpected, or repeated cell_id")
        prefix = f"{label}.{cell_id}"
        group = cell.get("group")
        if not isinstance(group, str) or not group:
            raise ValueError(f"{prefix}.group must be a nonempty string")
        recipient_query, donor_query = cell.get("recipient_query"), cell.get("donor_query")
        if recipient_query not in QUERIES or donor_query not in QUERIES:
            raise ValueError(f"{prefix}: unknown queried role")
        if donor_query == recipient_query:
            raise ValueError(f"{prefix}: same-query rows cannot dilute scientific losses")
        actual_structure = {"group": group, "recipient_query": recipient_query, "donor_query": donor_query}
        if actual_structure != expected_structure[cell_id]:
            raise ValueError(f"{prefix}: cell/group/query differs from frozen expected_structure")
        diagnostics = cell.get("diagnostics")
        if (not isinstance(diagnostics, (list, tuple))
                or any(not isinstance(x, str) for x in diagnostics)
                or len(set(diagnostics)) != len(diagnostics)
                or sorted(diagnostics) != expected_labels[cell_id]):
            raise ValueError(f"{prefix}: diagnostic labels differ from the frozen design")
        vectors = {field: _vector(cell.get(field), f"{prefix}.{field}")
                   for field in ("observed", "role_direction", "name_direction", "position_direction")}
        prediction = tuple(_finite(gains[recipient_query] * v, f"{prefix}.prediction")
                           for v in vectors["role_direction"])
        oracle = {r: closest_line_fit(vectors["observed"], vectors[f"{r}_direction"])
                  for r in ("name", "position")}
        gaps = {r: closest_line_fit(prediction, vectors[f"{r}_direction"])["distance"]
                for r in ("name", "position")}
        oracle["no_op"] = {"projection": (0., 0., 0.),
                            "distance": rms_distance(vectors["observed"], (0., 0., 0.))}
        gaps["no_op"] = rms_distance(prediction, (0., 0., 0.))
        normalized = copy.deepcopy(dict(cell))
        normalized.update({k: list(v) for k, v in vectors.items()})
        normalized.update({"diagnostics": list(diagnostics), "role_prediction": list(prediction),
                           "role_error": rms_distance(vectors["observed"], prediction),
                           "oracle_fits": oracle, "diagnostic_distances": gaps})
        parsed[cell_id] = normalized
        groups.setdefault(group, []).append(cell_id)
    if set(parsed) != set(ids):
        raise ValueError(f"{label}: scientific cell set is incomplete")
    for group, group_ids in groups.items():
        rows = [parsed[cell_id] for cell_id in group_ids]
        if (len(rows) != 2 or len({r["recipient_query"] for r in rows}) != 1
                or len({r["donor_query"] for r in rows}) != 2):
            raise ValueError(f"{label}.{group}: require exactly the two alternative donor queries")
        # For two rows, their mean is a minimizer of the sum of Euclidean/RMS
        # distances among all shared responses. It does not use a role endpoint.
        observed_center = [rows[0]["observed"][j]/2 + rows[1]["observed"][j]/2 for j in range(3)]
        role_center = [rows[0]["role_prediction"][j]/2 + rows[1]["role_prediction"][j]/2 for j in range(3)]
        for row in rows:
            row["oracle_fits"]["switch"] = {
                "projection": observed_center,
                "distance": rms_distance(row["observed"], observed_center)}
            row["diagnostic_distances"]["switch"] = rms_distance(row["role_prediction"], role_center)
    switch_groups = {parsed[cell_id]["group"] for cell_id in ids if "switch" in expected_labels[cell_id]}
    if len(switch_groups) != 1:
        raise ValueError("the two frozen switch diagnostics must belong to the same group")
    ordered = [parsed[cell_id] for cell_id in ids]
    losses = {"role": math.fsum(c["role_error"] / len(ids) for c in ordered)}
    losses.update({r: math.fsum(c["oracle_fits"][r]["distance"] / len(ids) for c in ordered)
                   for r in RIVALS})
    if not all(math.isfinite(v) for v in losses.values()):
        raise ValueError("family losses must be finite")
    marked = [c for c in ordered if c["diagnostics"]]
    adequate = losses["role"] <= ERROR_TOLERANCE and all(
        c["role_error"] <= ERROR_TOLERANCE for c in marked)
    informative_by_rival = {r: all(c["diagnostic_distances"][r] > DIAGNOSTIC_GAP
                                    for c in ordered if r in c["diagnostics"])
                            for r in RIVALS}
    result = copy.deepcopy(dict(raw))
    result.update({"cells": ordered, "losses": losses,
                   "paired_advantages": {r: _finite(losses[r] - losses["role"], f"{label}.advantage.{r}")
                                          for r in RIVALS},
                   "adequate": adequate, "informative_by_rival": informative_by_rival,
                   "informative": all(informative_by_rival.values())})
    return result


def _bootstrap(values, draws, seed):
    """Resample whole units with common draws across wording/precision/rival axes."""
    rng = np.random.default_rng(seed)
    output = np.empty((draws,) + values.shape[1:], dtype=np.float64)
    batch = max(1, min(128, 1_000_000 // values.size))
    for start in range(0, draws, batch):
        end = min(draws, start + batch)
        indices = rng.integers(0, len(values), size=(end-start, len(values)))
        output[start:end] = values[indices].mean(axis=1)
    if not np.isfinite(output).all():
        raise ValueError("bootstrap means must remain finite")
    family_size = int(np.prod(values.shape[1:]))
    tail = ADVANTAGE_ALPHA / (2 * family_size)
    return np.quantile(output, [tail, 1-tail], axis=0)


def _advantage_decision(interval):
    if interval[0] > ADVANTAGE_BOUNDARY:
        return "role_advantage"
    if interval[1] < -ADVANTAGE_BOUNDARY:
        return "rival_advantage"
    return "unresolved"


def analyze_records(records, gains, *, expected_cell_ids, expected_diagnostics, expected_structure,
                    expected_wordings=DEFAULT_WORDINGS, bootstrap_draws=BOOTSTRAP_DRAWS,
                    seed=PRIMARY_SEED, secondary_seed=SECONDARY_SEED, evidence_kind="unspecified"):
    """Analyze complete paired families without fitting gains or dropping rows.

    Each family has case_id and precisions[p].wordings[w], where each wording
    contains controls.passed and cells. Precision-level controls.passed is also
    required. Cells contain cell_id, group, recipient_query, donor_query,
    diagnostics, observed and role/name/position_direction three-margin vectors.
    Expected IDs, group/query structure and diagnostic labels are supplied from
    the frozen design, never inferred from the first observed family.

    'fit' denotes the original wording, NOT a fit/confirmation data split. All
    families here belong to the caller's declared evaluation split. Re-running
    this deterministic analysis is valid; repeated model execution must be
    prevented by the runner's frozen manifest, outside this records-only module.
    """
    ids, wordings, diagnostics, labels, structure = _design(
        expected_cell_ids, expected_wordings, expected_diagnostics, expected_structure)
    if not isinstance(gains, Mapping) or set(gains) != set(QUERIES):
        raise ValueError("gains must contain exactly giver, receiver, observer")
    gains = {q: _finite(gains[q], f"gain.{q}") for q in QUERIES}
    if any(not 0 <= g <= 1 for g in gains.values()):
        raise ValueError("frozen partial-transfer gains must lie in [0,1]")
    _integer(bootstrap_draws, "bootstrap_draws", 2)
    _integer(seed, "seed")
    _integer(secondary_seed, "secondary_seed")
    if seed == secondary_seed:
        raise ValueError("bootstrap seeds must differ")
    if evidence_kind not in ("synthetic", "model_measurements", "unspecified"):
        raise ValueError("unknown evidence_kind")
    if isinstance(records, (str, bytes, Mapping)):
        raise ValueError("records must be an iterable of family mappings")
    try:
        records = list(records)
    except TypeError as exc:
        raise ValueError("records must be iterable") from exc
    if not records:
        raise ValueError("at least one independent family is required")
    seen, output = set(), []
    for raw in records:
        if not isinstance(raw, Mapping):
            raise ValueError("each family must be a mapping")
        case_id = raw.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError("case_id must be a unique nonempty draw ID")
        seen.add(case_id)
        raw_precisions = raw.get("precisions")
        if not isinstance(raw_precisions, Mapping) or set(raw_precisions) != set(PRECISIONS):
            raise ValueError("require exactly float32 and float64 evaluations")
        parsed = {}
        for precision in PRECISIONS:
            raw_p = raw_precisions[precision]
            if not isinstance(raw_p, Mapping):
                raise ValueError("each precision must be a mapping")
            _controls(raw_p.get("controls"), f"{case_id}.{precision}.controls")
            raw_wordings = raw_p.get("wordings")
            if not isinstance(raw_wordings, Mapping) or set(raw_wordings) != set(wordings):
                raise ValueError("both frozen wordings are required exactly")
            parsed[precision] = copy.deepcopy(dict(raw_p))
            parsed[precision]["wordings"] = {}
            for wording in wordings:
                analyzed = _analyze_cells(raw_wordings[wording], f"{case_id}.{precision}.{wording}",
                                          ids, labels, gains, structure)
                parsed[precision]["wordings"][wording] = analyzed
        discrepancies = {}
        for wording in wordings:
            a, b = (parsed[p]["wordings"][wording] for p in PRECISIONS)
            for ca, cb in zip(a["cells"], b["cells"]):
                prefix = f"{wording}.{ca['cell_id']}"
                for field in ("observed", "role_prediction"):
                    discrepancies[f"{prefix}.{field}"] = max(abs(x-y) for x, y in zip(ca[field], cb[field]))
                discrepancies[f"{prefix}.role_error"] = abs(ca["role_error"]-cb["role_error"])
                for rival in RIVALS:
                    discrepancies[f"{prefix}.{rival}.error"] = abs(
                        ca["oracle_fits"][rival]["distance"]-cb["oracle_fits"][rival]["distance"])
                    discrepancies[f"{prefix}.{rival}.projection"] = max(abs(x-y) for x, y in zip(
                        ca["oracle_fits"][rival]["projection"], cb["oracle_fits"][rival]["projection"]))
                    discrepancies[f"{prefix}.{rival}.gap"] = abs(
                        ca["diagnostic_distances"][rival]-cb["diagnostic_distances"][rival])
            for field in ("losses", "paired_advantages"):
                for key in a[field]:
                    discrepancies[f"{wording}.{field}.{key}"] = abs(a[field][key]-b[field][key])
        discrepancies = {k: _finite(v, f"{case_id}.numerical_discrepancy") for k, v in discrepancies.items()}
        resolved = max(discrepancies.values()) <= NUMERICAL_BUDGET
        adequate = all(parsed[p]["wordings"][w]["adequate"] for p in PRECISIONS for w in wordings)
        informative = all(parsed[p]["wordings"][w]["informative"] for p in PRECISIONS for w in wordings)
        result = copy.deepcopy(dict(raw))
        result.update({"precisions": parsed, "resolved": resolved, "adequate": adequate,
                       "informative": informative, "joint_success": resolved and adequate and informative,
                       "resolution": {"passed": resolved, "discrepancies": discrepancies,
                                      "maximum_discrepancy": max(discrepancies.values())}})
        output.append(result)
    n = len(output)
    count = sum(r["joint_success"] for r in output)
    lo, hi = clopper_pearson(count, n, alpha=COVERAGE_ALPHA, family_size=1)
    joint_status = "adequate" if lo > COVERAGE else "excluded" if hi < COVERAGE else "unresolved"
    values = np.array([[[[r["precisions"][p]["wordings"][w]["paired_advantages"][rival]
                          for rival in RIVALS] for w in wordings] for p in PRECISIONS] for r in output])
    primary, secondary = _bootstrap(values, bootstrap_draws, seed), _bootstrap(values, bootstrap_draws, secondary_seed)
    means = values.mean(axis=0)
    if not np.isfinite(means).all():
        raise ValueError("mean paired advantages must be finite")
    unresolved_ids = [r["case_id"] for r in output if not r["resolved"]]
    advantages = {}
    for wi, wording in enumerate(wordings):
        advantages[wording] = {}
        for ri, rival in enumerate(RIVALS):
            by_precision = {}
            for pi, precision in enumerate(PRECISIONS):
                interval, repeat = primary[:, pi, wi, ri], secondary[:, pi, wi, ri]
                decision, repeated = _advantage_decision(interval), _advantage_decision(repeat)
                shift = float(np.max(np.abs(interval-repeat)))
                by_precision[precision] = {"mean": float(means[pi, wi, ri]), "interval": interval.tolist(),
                    "secondary_interval": repeat.tolist(), "decision": decision,
                    "secondary_decision": repeated, "mc_endpoint_shift": shift,
                    "mc_stable": shift <= MC_ENDPOINT_BUDGET and decision == repeated}
            interval_difference = max(float(np.max(np.abs(c[:, 0, wi, ri]-c[:, 1, wi, ri])))
                                      for c in (primary, secondary))
            reasons = []
            if n < 2:
                reasons.append("fewer_than_two_independent_families")
            if unresolved_ids or interval_difference > NUMERICAL_BUDGET:
                reasons.append("cross_precision_disagreement")
            if not all(v["mc_stable"] for v in by_precision.values()):
                reasons.append("bootstrap_monte_carlo_instability")
            decisions = {v["decision"] for v in by_precision.values()}
            if len(decisions) != 1:
                reasons.append("precision_decisions_disagree")
            advantages[wording][rival] = {"status": next(iter(decisions)) if not reasons else "unresolved",
                "precisions": by_precision, "blocked_reasons": reasons,
                "maximum_interval_precision_discrepancy": interval_difference}
    all_advantages = all(advantages[w][r]["status"] == "role_advantage" for w in wordings for r in RIVALS)
    status = ("supports_relational_response_within_declared_menu" if joint_status == "adequate" and all_advantages
              else "relative_winner_without_joint_adequacy" if all_advantages else "unresolved")
    protocol_matches = (bootstrap_draws == BOOTSTRAP_DRAWS and seed == PRIMARY_SEED
                        and secondary_seed == SECONDARY_SEED)
    return {"schema_version": "role-response-analysis-v1", "status": status,
            "evidence_kind": evidence_kind,
            "model_evidence_supported": False,
            "statistical_support_given_external_gates": (
                status == "supports_relational_response_within_declared_menu"
                and evidence_kind == "model_measurements" and protocol_matches),
            "external_gates_unchecked": [
                "Public pre-inference freeze, source/model/code hashes and single-run authorization.",
                "Independent family sampling, declared exclusions and absence of outcome-based selection.",
                "Native competence and endpoint geometry on all required roles, forms and wordings.",
                "Gain fitting on separate development families with heldout wording patch outcomes unused.",
                "Observed responses and endpoint directions bound to their raw logits and fixed name ordering.",
                "Mandatory same-query raw results, exact self-shams and independent intervention-fidelity controls."],
            "protocol_matches_frozen_statistics": protocol_matches,
            "n_units": n, "n_resolved_units": n-len(unresolved_ids), "n_unresolved_units": len(unresolved_ids),
            "unresolved_case_ids": unresolved_ids,
            "joint_coverage": {"successes": count, "n": n, "fraction": count/n,
                               "interval": [lo, hi], "status": joint_status},
            "descriptive_counts": {"adequate": sum(r["adequate"] for r in output),
                                   "informative": sum(r["informative"] for r in output)},
            "paired_advantages": advantages, "records": output,
            "contract": {"frozen_gains": gains, "gain_range": [0, 1],
                "expected_cell_ids": list(ids), "wordings": list(wordings),
                "expected_structure": structure,
                "expected_diagnostics": {k: list(v) for k, v in diagnostics.items()},
                "error_metric": "RMS of three pairwise logit-margin errors, then mean over scientific cells",
                "absolute_error_tolerance_nat": ERROR_TOLERANCE, "diagnostic_gap_nat": DIAGNOSTIC_GAP,
                "coverage": COVERAGE, "coverage_alpha": COVERAGE_ALPHA,
                "advantage_boundary_nat": ADVANTAGE_BOUNDARY, "advantage_family_alpha": ADVANTAGE_ALPHA,
                "advantage_family_size": len(RIVALS)*len(wordings)*len(PRECISIONS),
                "bootstrap_draws_per_seed": bootstrap_draws, "seed": seed, "secondary_seed": secondary_seed,
                "mc_endpoint_budget_nat": MC_ENDPOINT_BUDGET, "numerical_budget_nat": NUMERICAL_BUDGET},
            "assumptions_and_scope": [
                "Synthetic records test this analysis contract and provide no empirical evidence about a language model.",
                "Fit and heldout name wording strata, not family-level data splits; gains must be frozen on separate development families.",
                "Primary predictions concern absolute patched-minus-native responses on cross-query cells only.",
                "Same-query measurements and exact self-shams require separate mandatory runner reporting; this analyzer does not establish their validity.",
                "Each independent family includes all cells, wordings and precisions; none are additional independent observations.",
                "Name and position rivals receive unrestricted signed per-cell gains on absolute endpoint-minus-native directions.",
                "The switch rival receives a shared arbitrary response for the two alternative donor queries in each recipient group.",
                "The .25-nat family average RMS criterion is not a maximum error bound; marked diagnostic cells also require RMS error <= .25.",
                "Joint CP coverage includes adequacy and separation from every declared rival, with numerical failures retained as failures.",
                "Paired bootstrap intervals have nominal, not finite-sample guaranteed, simultaneous coverage.",
                "Descriptive adequacy/informativeness counts are not additional simultaneous population claims.",
                "A supported result identifies this response within the declared endpoint model menu, not a unique semantic representation or native circuit."]}
