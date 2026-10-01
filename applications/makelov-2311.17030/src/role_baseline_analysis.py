"""Stage A: native competence and necessary geometry, with no patch outcomes.

The 32 independent families each contain 48 logical native rows: two wordings,
eight contexts and three questions. Duplicate prompts in distinct declared
contexts remain rows in the fixed design, never extra independent families.
Accuracy intervals resample whole families and are descriptive, not gate tests.
Geometry is evaluated at the largest allowed role gain (one); passing establishes
an opportunity for a later test, not a learned gain or relational transfer.
"""

import copy
import math
from collections.abc import Mapping

import numpy as np

from query_route_analysis import InvalidControlsError, clopper_pearson
from role_design import DIAGNOSTICS, DONORS, RECIPIENTS, ROLES, WORDINGS, directions
from role_geometry import closest_line_fit, pairwise_margins, rms_distance, validate_logits


PRECISIONS = ("float32", "float64")
CONTEXTS = {**DONORS, **RECIPIENTS}
FORMS = ("gave", "received", "watched")
N_FAMILIES = 32
N_ROWS = 48
COMPETENCE_THRESHOLD = .90
POTENTIAL_GEOMETRY_THRESHOLD = .90
DIAGNOSTIC_GAP = .52
NUMERICAL_BUDGET = .01
BOOTSTRAP_DRAWS = 20000
BOOTSTRAP_SEED = 320260940
METADATA_FIELDS = ("wording", "context_id", "form", "query", "correct_name_index")
GROUPS = tuple((query, wording, form) for query in ROLES for wording in WORDINGS for form in FORMS)


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


def _integer(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _list(values, label):
    if isinstance(values, (str, bytes, Mapping)):
        raise ValueError(f"{label} must be an iterable of mappings")
    try:
        return list(values)
    except TypeError as exc:
        raise ValueError(f"{label} must be iterable") from exc


def _context_matches(value, context_id):
    """The sampler stores the exact context ID; explicit context maps also bind."""
    if isinstance(value, str):
        return value == context_id
    context = CONTEXTS[context_id]
    return (isinstance(value, Mapping) and value.get("form") == context["form"]
            and isinstance(value.get("binding"), (list, tuple))
            and list(value["binding"]) == list(context["binding"]))


def _expected_cases(cases):
    cases = _list(cases, "cases")
    if len(cases) != N_FAMILIES:
        raise ValueError("Stage A requires exactly 32 frozen development families")
    parsed = {}
    required_combinations = {(w, c, q) for w in WORDINGS for c in CONTEXTS for q in ROLES}
    for case in cases:
        if not isinstance(case, Mapping):
            raise ValueError("each frozen case must be a mapping")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in parsed:
            raise ValueError("frozen case_id values must be unique nonempty strings")
        names, tokens = case.get("names"), case.get("answer_token_ids")
        if (not isinstance(names, (list, tuple)) or len(names) != 3
                or any(not isinstance(n, str) or not n for n in names) or len(set(names)) != 3):
            raise ValueError("each case needs three distinct names in fixed A/B/C order")
        if not isinstance(tokens, (list, tuple)) or len(tokens) != 3:
            raise ValueError("each case needs three answer_token_ids")
        for token in tokens:
            _integer(token, "answer_token_id")
        if len(set(tokens)) != 3:
            raise ValueError("answer_token_ids must be distinct")
        rows = case.get("rows")
        if not isinstance(rows, (list, tuple)) or len(rows) != N_ROWS:
            raise ValueError("each frozen case requires all 48 native rows")
        expected, combinations = {}, set()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("each frozen row must be a mapping")
            row_id = row.get("row_id")
            if not isinstance(row_id, str) or not row_id or row_id in expected:
                raise ValueError("frozen row_id values must be unique nonempty strings")
            wording, context_id, query = (row.get(k) for k in ("wording", "context_id", "query"))
            if any(not isinstance(v, str) for v in (wording, context_id, query)):
                raise ValueError("frozen wording/context/query labels must be strings")
            combination = (wording, context_id, query)
            if combination not in required_combinations or combination in combinations:
                raise ValueError("frozen native rows must cover each wording/context/query exactly once")
            context = CONTEXTS[context_id]
            correct = _integer(row.get("correct_name_index"), "correct_name_index")
            if row.get("form") != context["form"] or correct != context["binding"][ROLES.index(query)]:
                raise ValueError("frozen row metadata contradicts role_design context/role bindings")
            if "context" in row and not _context_matches(row["context"], context_id):
                raise ValueError("frozen context metadata contradicts role_design")
            if "correct_answer_token_id" in row and _integer(row["correct_answer_token_id"], "correct_answer_token_id") != tokens[correct]:
                raise ValueError("frozen correct answer token does not match its name index")
            expected[row_id] = row
            combinations.add(combination)
        if combinations != required_combinations:
            raise ValueError("frozen native row grid is incomplete")
        parsed[case_id] = {"case": case, "rows": expected, "answer_token_ids": tuple(tokens)}
    return parsed


def _native_geometry(rows, wording):
    endpoints = {context: {} for context in RECIPIENTS}
    for row in rows:
        if row["wording"] == wording and row["context_id"] in RECIPIENTS:
            endpoints[row["context_id"]][row["query"]] = row["name_logits"]
    vectors = {row["cell_id"]: row for row in directions(endpoints)}
    diagnostic_rows = {}
    switch_predictions = [vectors[cell]["role_direction"] for cell in DIAGNOSTICS["switch"]]
    switch_center = [switch_predictions[0][j]/2 + switch_predictions[1][j]/2 for j in range(3)]
    for rival, cell_ids in DIAGNOSTICS.items():
        diagnostic_rows[rival] = []
        for cell_id in cell_ids:
            cell = vectors[cell_id]
            prediction = cell["role_direction"]  # Maximum allowed gain is one.
            if rival in ("name", "position"):
                fitted = closest_line_fit(prediction, cell[rival + "_direction"])
                projection, gap = list(fitted["projection"]), fitted["distance"]
            elif rival == "switch":
                projection, gap = switch_center, rms_distance(prediction, switch_center)
            else:
                projection, gap = [0., 0., 0.], rms_distance(prediction, (0., 0., 0.))
            diagnostic_rows[rival].append({"cell_id": cell_id,
                "maximum_gain_role_prediction": list(prediction), "oracle_projection": projection,
                "gap": gap, "passes": gap > DIAGNOSTIC_GAP})
    minimum = min(row["gap"] for rows in diagnostic_rows.values() for row in rows)
    return {"maximum_gain": 1., "diagnostics": diagnostic_rows,
            "minimum_gap": minimum, "potential_separation": minimum > DIAGNOSTIC_GAP}


def _precision(raw, expected, label):
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} must be a mapping")
    controls = raw.get("controls")
    if not isinstance(controls, Mapping) or controls.get("passed") is not True:
        raise InvalidControlsError(f"{label}.controls.passed must be true")
    rows = raw.get("rows")
    if not isinstance(rows, (list, tuple)) or len(rows) != N_ROWS:
        raise ValueError(f"{label} requires all 48 native rows")
    parsed = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"{label}.row must be a mapping")
        row_id = row.get("row_id")
        if not isinstance(row_id, str) or row_id not in expected["rows"] or row_id in parsed:
            raise ValueError(f"{label}: missing, unexpected or duplicate row_id")
        fixed = expected["rows"][row_id]
        if any(row.get(key) != fixed[key] for key in METADATA_FIELDS):
            raise ValueError(f"{label}.{row_id}: metadata differs from frozen case")
        if "context" in row and not _context_matches(row["context"], fixed["context_id"]):
            raise ValueError(f"{label}.{row_id}: context metadata differs from frozen case")
        for field in ("prompt", "token_ids", "position", "name_token_positions", "first_mention_name_indices"):
            if field in row and field in fixed and row[field] != fixed[field]:
                raise ValueError(f"{label}.{row_id}: {field} differs from frozen case")
        _integer(row["correct_name_index"], "correct_name_index")
        if ("correct_answer_token_id" in row
                and _integer(row["correct_answer_token_id"], "correct_answer_token_id")
                    != expected["answer_token_ids"][row["correct_name_index"]]):
            raise ValueError(f"{label}.{row_id}: correct answer token differs from frozen case")
        logits = validate_logits(row.get("name_logits"))
        margins = pairwise_margins(logits)
        mass = _finite(row.get("name_probability_mass"), f"{label}.{row_id}.name_probability_mass")
        if not 0 <= mass <= 1:
            raise ValueError("name_probability_mass must lie in [0,1]")
        argmax = _integer(row.get("full_vocab_argmax_id"), "full_vocab_argmax_id")
        if "full_vocab_argmax_token" in row and not isinstance(row["full_vocab_argmax_token"], str):
            raise ValueError("full_vocab_argmax_token must be a string when supplied")
        if "vocab_size" in row and argmax >= _integer(row["vocab_size"], "vocab_size", 1):
            raise ValueError("full_vocab_argmax_id is outside the recorded vocabulary")
        top = max(logits)
        winners = [j for j, value in enumerate(logits) if value == top]
        correct = len(winners) == 1 and winners[0] == row["correct_name_index"]
        value = copy.deepcopy(dict(row))
        value.update({"name_logits": list(logits), "pairwise_margins": list(margins),
                      "name_probability_mass": mass, "three_name_correct": correct,
                      "three_name_top_tie": len(winners) > 1,
                      "three_name_prediction": winners[0] if len(winners) == 1 else None,
                      "full_vocab_argmax_is_correct_name": argmax == expected["answer_token_ids"][row["correct_name_index"]]})
        parsed[row_id] = value
    if set(parsed) != set(expected["rows"]):
        raise ValueError(f"{label}: native row grid is incomplete")
    # Minimal controls remain useful for explicit constructed-logit fixtures.
    # A recorded producer replay is an indivisible control: check its evidence,
    # not just the producer's omnibus boolean.
    replay_fields = {"batch_vs_single_max_name_logit_error", "batch_vs_single_tolerance",
                     "token_replay_passed", "no_padding", "replay_row_id", "replay_name_logits"}
    if replay_fields.intersection(controls):
        if not replay_fields.issubset(controls):
            raise InvalidControlsError(f"{label}: incomplete replay controls")
        if controls["token_replay_passed"] is not True or controls["no_padding"] is not True:
            raise InvalidControlsError(f"{label}: token replay/no-padding controls failed")
        replay_id = controls["replay_row_id"]
        if replay_id != next(iter(expected["rows"])):
            raise InvalidControlsError(f"{label}: replay row must be the first frozen row")
        replay_logits = validate_logits(controls["replay_name_logits"])
        error = _finite(controls["batch_vs_single_max_name_logit_error"], "replay error")
        tolerance = _finite(controls["batch_vs_single_tolerance"], "replay tolerance")
        observed_error = max(abs(a-b) for a, b in zip(replay_logits, parsed[replay_id]["name_logits"]))
        if error < 0 or tolerance < 0 or error > tolerance or observed_error != error:
            raise InvalidControlsError(f"{label}: replay error is inconsistent or exceeds tolerance")
    ordered = [parsed[row_id] for row_id in expected["rows"]]
    scores = {}
    for query, wording, form in GROUPS:
        selected = [row for row in ordered if (row["query"], row["wording"], row["form"]) == (query, wording, form)]
        key = "|".join((query, wording, form))
        scores[key] = {"correct": sum(row["three_name_correct"] for row in selected),
                       "rows": len(selected), "family_accuracy": sum(row["three_name_correct"] for row in selected)/len(selected)}
    result = copy.deepcopy(dict(raw))
    result.update({"rows": ordered, "group_scores": scores,
                   "potential_geometry": {wording: _native_geometry(ordered, wording) for wording in WORDINGS}})
    return result


def _descriptive_intervals(values, draws, seed):
    rng = np.random.default_rng(seed)
    boot = np.empty((draws,) + values.shape[1:], dtype=np.float64)
    batch = min(128, draws)
    for start in range(0, draws, batch):
        end = min(start+batch, draws)
        indices = rng.integers(0, len(values), size=(end-start, len(values)))
        boot[start:end] = values[indices].mean(axis=1)
    return np.quantile(boot, [.025, .975], axis=0)


def analyze_baselines(cases, records, *, bootstrap_draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    """Freeze-bound native feasibility checks. No gains or patch outputs are fit.

    Malformed/incomplete records or failed producer controls raise before a
    scientific result. Finite cross-precision disagreement returns INVALIDNUMERICS
    and preserves all families. PASS authorizes consideration of development,
    never confirmation or a positive relational response claim.
    """
    _integer(bootstrap_draws, "bootstrap_draws", 2)
    _integer(seed, "seed")
    expected = _expected_cases(cases)
    records = _list(records, "records")
    if len(records) != N_FAMILIES:
        raise ValueError("Stage A requires exactly 32 complete measurement records")
    seen, results = set(), {}
    for raw in records:
        if not isinstance(raw, Mapping):
            raise ValueError("each measurement record must be a mapping")
        case_id = raw.get("case_id")
        if not isinstance(case_id, str) or case_id not in expected or case_id in seen:
            raise ValueError("measurement case_id must match one unique frozen family")
        seen.add(case_id)
        precisions = raw.get("precisions")
        if not isinstance(precisions, Mapping) or set(precisions) != set(PRECISIONS):
            raise ValueError("each family requires exactly float32 and float64")
        parsed = {p: _precision(precisions[p], expected[case_id], f"{case_id}.{p}") for p in PRECISIONS}
        discrepancies, probability_discrepancies, argmax_changes = {}, [], 0
        for left, right in zip(parsed["float32"]["rows"], parsed["float64"]["rows"]):
            discrepancies[f"row.{left['row_id']}.margins"] = max(abs(a-b) for a, b in zip(
                left["pairwise_margins"], right["pairwise_margins"]))
            probability_discrepancies.append(abs(left["name_probability_mass"]-right["name_probability_mass"]))
            argmax_changes += left["full_vocab_argmax_id"] != right["full_vocab_argmax_id"]
        for wording in WORDINGS:
            left, right = (parsed[p]["potential_geometry"][wording]["diagnostics"] for p in PRECISIONS)
            for rival in DIAGNOSTICS:
                for a, b in zip(left[rival], right[rival]):
                    prefix = f"{wording}.{rival}.{a['cell_id']}"
                    discrepancies[prefix+".gap"] = abs(a["gap"]-b["gap"])
                    for field in ("maximum_gain_role_prediction", "oracle_projection"):
                        discrepancies[prefix+"."+field] = max(abs(x-y) for x, y in zip(a[field], b[field]))
        discrepancies = {k: _finite(v, "numerical discrepancy") for k, v in discrepancies.items()}
        maximum = max(discrepancies.values())
        resolved = maximum <= NUMERICAL_BUDGET
        potential = all(parsed[p]["potential_geometry"][w]["potential_separation"] for p in PRECISIONS for w in WORDINGS)
        result = copy.deepcopy(dict(raw))
        result.update({"precisions": parsed, "resolved": resolved,
                       "potential_geometry_success": resolved and potential,
                       "numerical_resolution": {"passed": resolved, "maximum_discrepancy_nat": maximum,
                            "discrepancies_nat": discrepancies,
                            "maximum_name_mass_discrepancy_descriptive": max(probability_discrepancies),
                            "full_vocab_argmax_disagreements_descriptive": argmax_changes}})
        results[case_id] = result
    if seen != set(expected):
        raise ValueError("not all frozen families were measured")
    ordered = [results[case_id] for case_id in expected]
    keys = ["|".join(group) for group in GROUPS]
    values = np.array([[[r["precisions"][p]["group_scores"][key]["family_accuracy"] for key in keys]
                        for p in PRECISIONS] for r in ordered], dtype=np.float64)
    intervals = _descriptive_intervals(values, bootstrap_draws, seed)
    groups = {}
    failed_groups = []
    for gi, (key, (query, wording, form)) in enumerate(zip(keys, GROUPS)):
        by_precision = {}
        for pi, precision in enumerate(PRECISIONS):
            score = float(values[:, pi, gi].mean())
            rows_per_family = ordered[0]["precisions"][precision]["group_scores"][key]["rows"]
            by_precision[precision] = {"mean_family_accuracy": score,
                "correct_rows": sum(r["precisions"][precision]["group_scores"][key]["correct"] for r in ordered),
                "rows_per_family": rows_per_family, "n_families": N_FAMILIES,
                "family_scores": values[:, pi, gi].tolist(),
                "descriptive_95_percent_interval": intervals[:, pi, gi].tolist(),
                "passes": score >= COMPETENCE_THRESHOLD}
        passed = all(item["passes"] for item in by_precision.values())
        groups[key] = {"query": query, "wording": wording, "form": form,
                       "precisions": by_precision, "passes": passed}
        if not passed:
            failed_groups.append(key)
    count = sum(r["potential_geometry_success"] for r in ordered)
    geometry_passed = count/N_FAMILIES >= POTENTIAL_GEOMETRY_THRESHOLD
    geometry_interval = clopper_pearson(count, N_FAMILIES, alpha=.05, family_size=1)
    unresolved = [r["case_id"] for r in ordered if not r["resolved"]]
    reasons = []
    if unresolved:
        reasons.append("INVALIDNUMERICS")
    if failed_groups:
        reasons.append("STOP_COMPETENCE")
    if not geometry_passed:
        reasons.append("STOP_GEOMETRY")
    return {"schema_version": "role-baseline-analysis-v1", "stage": "A_native_only",
            "status": reasons[0] if reasons else "PASS", "failure_reasons": reasons,
            "n_units": N_FAMILIES, "n_rows_per_precision_per_family": N_ROWS,
            "competence_groups": groups, "failed_competence_groups": failed_groups,
            "potential_geometry": {"successes": count, "n": N_FAMILIES, "fraction": count/N_FAMILIES,
                "minimum_required_families": math.ceil(POTENTIAL_GEOMETRY_THRESHOLD*N_FAMILIES),
                "descriptive_95_percent_CP_interval": list(geometry_interval), "passes": geometry_passed,
                "maximum_gain_only": True},
            "numerical_resolution": {"passed": not unresolved, "unresolved_case_ids": unresolved,
                "maximum_discrepancy_nat": max(r["numerical_resolution"]["maximum_discrepancy_nat"] for r in ordered)},
            "records": ordered, "relational_response_supported": False, "confirmation_authorized": False,
            "contract": {"n_families": N_FAMILIES, "rows_per_precision_per_family": N_ROWS,
                "competence_threshold": COMPETENCE_THRESHOLD, "potential_geometry_fraction_threshold": POTENTIAL_GEOMETRY_THRESHOLD,
                "geometry_diagnostic_gap_nat": DIAGNOSTIC_GAP, "maximum_allowed_role_gain": 1.,
                "numerical_budget_nat": NUMERICAL_BUDGET, "bootstrap_draws": bootstrap_draws, "bootstrap_seed": seed,
                "CI_role": "descriptive only, point-estimate feasibility thresholds govern; no multiplicity-adjusted competence claim"},
            "assumptions_and_scope": [
                "Three-name conditional correctness requires a unique maximum at the declared answer; ties fail.",
                "Full-vocabulary argmax and name probability mass are retained, but do not replace the declared three-name competence gate.",
                "Accuracy is averaged within each family/query/wording/form group before whole-family resampling.",
                "Logical duplicate native prompts stay in their declared contexts; context rows are not independent samples.",
                "Descriptive intervals are unadjusted and cannot rescue a failed feasibility point threshold.",
                "Maximum-gain endpoint separation is necessary opportunity, not a fitted gain, predicted success rate, or observed patch response.",
                "Gain-one geometry failures cannot be repaired by a smaller allowed gain, and families are never removed or replaced.",
                "Numerical failures invalidate interpretation; raw measurements and all failed families remain visible.",
                "Native probability-mass discrepancies are dimensionless descriptive diagnostics, not subjected to a nat budget.",
                "Sampling, token eligibility, source hashes and absence of patch inference require the external frozen runner/manifest."]}
