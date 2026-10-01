"""Round 1 development splits A and B (protocol v2): generation order and decisions.

Standard library plus the analyzer and the pilot gate table. DEVELOPMENT DATA. Split A
selects the cell; split B, in the selected cell only, supplies every value that would be
frozen. Nothing here freezes anything or touches the confirmation seed block.

Generation (declared before the runs, SPLIT_A_B_PROTOCOL.md):

- Split A: family i = 4j + k uses candidate cell k (c1..c4) and seed 2,000,000 + i;
  cell k stops at its 50th qualifying family; at most 100 families per cell.
- Split B: family i uses the selected cell and seed 3,000,000 + i; generation stops at
  the 200th qualifying family; at most 400 families.

Decisions:

- Both splits: gates 1-5 and 7 (the generic gate table of ``mixing_pilot_summary``) and
  the quota must hold, else STOP.
- Split A: selection by ``select_on_split_a`` (largest d, ties c1, c2, c3, c4, d >= d_min,
  else NOT_DECIDABLE, S1). Gates 6a, 6b and 8 are binding on split B; if the selected
  cell fails any of them on split A, the run STOPs before split B (declared,
  conservative).
- Split B: ``freeze_from_split_b`` (s_min, resolution, anchors, agreement gates, d, the
  final N with its adequacy label, delta), all from split B.
"""
import json
import math
from pathlib import Path

import mixing_pilot_summary
import mixing_round1_analysis as ra

GENERIC_GATES = ("1_model_hashes", "2_native_and_yield", "3_tokens", "4_hooks", "5_identity", "7_dtype_device")


def generate_families(run_family, cells, *, seed_base, quota, cap, per_cell, prefix, on_record):
    """Run families in the declared order until the quota is met or the cap reached.

    ``run_family(case_id, draw_index, seed, cell_key, cell)`` returns (record, extra);
    ``on_record(record, extra)`` stores it. With ``per_cell`` the families interleave
    over the cells (i = len(cells) * j + k) and each cell stops at its quota; the cap is
    per cell. Otherwise one cell, i = 0, 1, ..., and the cap is on the total. Returns
    per-cell counts and whether every quota was met."""
    counts = {key: {"generated": 0, "qualifying": 0} for key, _ in cells}

    def one(i, key, cell):
        record, extra = run_family(case_id=f"{prefix}-{i:04d}", draw_index=i, seed=seed_base + i,
                                   cell_key=key, cell=cell)
        counts[key]["generated"] += 1
        counts[key]["qualifying"] += bool(record["qualifies"])
        on_record(record, extra)

    if per_cell:
        for j in range(cap):
            open_cells = [(k, key, cell) for k, (key, cell) in enumerate(cells) if counts[key]["qualifying"] < quota]
            if not open_cells:
                break
            for k, key, cell in open_cells:
                one(len(cells) * j + k, key, cell)
    else:
        if len(cells) != 1:
            raise ValueError("a single-cell split needs exactly one cell")
        key, cell = cells[0]
        i = 0
        while counts[key]["qualifying"] < quota and i < cap:
            one(i, key, cell)
            i += 1
    return counts, all(c["qualifying"] >= quota for c in counts.values())


def _finite_or_none(values):
    """Undefined anchors (d = -inf) are reported as None, with the analyzer's reason."""
    return {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in values.items()}


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _cell_data(records, key):
    group = [r for r in records if r.get("cell_key") == key and r.get("qualifies")]
    return {"conflict": group,
            "agreement": [(k, a["j"], a) for k, r in enumerate(group) for a in r["agreement"]],
            "nopatch": [r["nopatch"] for r in group]}


def _generic(output, spec, label):
    table = mixing_pilot_summary.summarize(output, spec)
    table["label"] = label
    for key in ("planning_N", "planning_N_note", "unresolved_rate_for_planning_N"):
        table.pop(key, None)
    table["runtime"].pop("confirmation_at_planning_N", None)
    return table


def _quota(records, spec, per_cell):
    counts = {}
    for r in records:
        row = counts.setdefault(r["cell_key"], {"generated": 0, "qualifying": 0})
        row["generated"] += 1
        row["qualifying"] += bool(r["qualifies"])
    quota = spec["qualifying_per_cell"] if per_cell else spec["qualifying"]
    met = all(counts.get(key, {"qualifying": 0})["qualifying"] == quota for key, _ in spec["cells"])
    return counts, quota, met


def split_a_decision(output, spec):
    output = Path(output)
    records = _jsonl(output / "records.jsonl")
    table = _generic(output, spec, "SPLIT A, development data (protocol v2): cell selection only; "
                                   "no value from split A is frozen")
    stops = [f"gate {g} failed" for g in GENERIC_GATES if not table["gates"][g]["passed"]]
    counts, quota, met = _quota(records, spec, per_cell=True)
    if not met:
        stops.append(f"quota of {quota} qualifying families per cell not met within the cap")
    selection = ra.select_on_split_a({key: _cell_data(records, key) for key, _ in spec["cells"]},
                                     [(key, cell) for key, cell in spec["cells"]], n=spec["n"],
                                     s_min_quantile=spec["s_min_quantile"], s_min_floor=spec["s_min_floor"],
                                     d_min=spec["d_min"])
    estimates = selection["selection"]["estimates"]
    chosen = selection["selection"].get("selected")
    ordered = sorted((d for _, d in estimates), reverse=True)
    margin = ordered[0] - ordered[1] if len(ordered) > 1 else None
    if selection["status"] != "PROCEED":
        stops.append(selection["reason"])
    selected_checks = None
    if chosen is not None:
        cell_row = table["per_cell"][chosen]
        agreement = cell_row["agreement"]
        resolution = 1 - cell_row["unresolved"]["rate"]
        selected_checks = {
            "6a_agreement_transfer": {"value": agreement["agreement_transfer_rate"],
                                      "floor": spec["agreement_transfer_floor"],
                                      "passed": agreement["agreement_transfer_rate"] >= spec["agreement_transfer_floor"]},
            "6b_agreement_resolution": {"value": agreement["agreement_resolution_rate"],
                                        "floor": spec["agreement_resolution_floor"],
                                        "passed": agreement["agreement_resolution_rate"] >= spec["agreement_resolution_floor"]},
            "8_support": {"value": resolution, "floor": spec["resolution_rate_floor"],
                          "passed": resolution >= spec["resolution_rate_floor"]}}
        stops += [f"selected cell fails {name} on split A (declared conservative STOP)"
                  for name, check in selected_checks.items() if not check["passed"]]
        likely = table["gates"]["9_separation_descriptive"]["likely_selection"]
        if likely is None or likely.get("selected") != chosen:
            stops.append("internal inconsistency: gate table and analyzer select different cells")
    return {
        "label": "SPLIT A DECISION, development data (protocol v2)",
        "status": "STOP" if stops else "PROCEED", "stops": stops,
        "level": "S1" if stops else None,
        "selected": chosen if not stops else None,
        "selection": selection["selection"],
        "d_by_cell": dict(estimates), "margin_over_second": margin,
        "s_min_by_cell_split_A": {k: v["s_min"] for k, v in selection["candidates"].items()},
        "anchors_by_cell_split_A": {k: _finite_or_none({x: v["split_A"].get(x) for x in
                                                        ("T_W", "T_A", "d", "q_bar", "T_A_by_target", "error")})
                                    for k, v in selection["candidates"].items()},
        "selected_cell_checks_on_A": selected_checks,
        "quota": {"per_cell": quota, "counts": counts, "met": met},
        "gate_table": table,
        "note": "split A values serve the selection only; every value to freeze comes from split B",
    }


def split_b_decision(output, spec):
    output = Path(output)
    records = _jsonl(output / "records.jsonl")
    table = _generic(output, spec, "SPLIT B, development data (protocol v2): source of every value "
                                   "that would be frozen; not frozen")
    stops = [f"gate {g} failed" for g in GENERIC_GATES if not table["gates"][g]["passed"]]
    counts, quota, met = _quota(records, spec, per_cell=False)
    if not met:
        stops.append(f"quota of {quota} qualifying families not met within the cap")
    (key, cell), = spec["cells"]
    freeze = ra.freeze_from_split_b(_cell_data(records, key), cell, n=spec["n"],
                                    s_min_quantile=spec["s_min_quantile"], s_min_floor=spec["s_min_floor"],
                                    d_min=spec["d_min"], agreement_transfer_floor=spec["agreement_transfer_floor"],
                                    false_invalid_rate=spec["delta"]["false_invalid_rate"],
                                    resamples=spec["delta"]["resamples"], seed=spec["delta"]["seed"])
    if freeze["status"] != "PROCEED":
        stops.append(freeze["reason"])
    return {
        "label": "SPLIT B DECISION, development data (protocol v2); values for review, NOT FROZEN",
        "status": "STOP" if stops else "PROCEED", "stops": stops, "level": "S1" if stops else None,
        "cell_key": key, "quota": {"total": quota, "counts": counts, "met": met},
        "split_B": freeze, "gate_table": table,
    }
