"""Exact attention countermodels: identical replacements, opposite zero-reference effects.

These are constructed explanatory witnesses, not fitted Transformer mechanisms.
Fractions avoid treating floating-point rounding as an identification result.
"""
from fractions import Fraction as F
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
N = 1000
NATIVE_A = (F(3, 5), F(3, 10), F(1, 10))
NATIVE_B = (F(1, 10), F(3, 10), F(3, 5))
UNIFORM = (F(1, 3),) * 3
MEAN = tuple((a + b) / 2 for a, b in zip(NATIVE_A, NATIVE_B))


def solve(matrix, rhs):
    """Small exact Gaussian elimination, independent of statistical fitting."""
    a = [list(map(F, row)) + [F(y)] for row, y in zip(matrix, rhs)]
    for col in range(len(a)):
        pivot = next(i for i in range(col, len(a)) if a[i][col])
        a[col], a[pivot] = a[pivot], a[col]
        scale = a[col][col]
        a[col] = [x / scale for x in a[col]]
        for i in range(len(a)):
            if i != col:
                factor = a[i][col]
                a[i] = [x - factor * y for x, y in zip(a[i], a[col])]
    return tuple(row[-1] for row in a)


def world(counts, zero_count):
    q = tuple(F(c, N) for c in counts)
    b = F(zero_count, N)
    values = [solve([native, UNIFORM, MEAN], [target - b for target in q])
              for native in (NATIVE_A, NATIVE_B)]
    predictions = {name: [] for name in ("native", "uniform", "mean", "zero")}
    # Exactly 500 of each native pattern: MEAN is their actual dataset mean.
    for i in range(N):
        native = (NATIVE_A, NATIVE_B)[i % 2]
        v = values[i % 2]
        threshold = F(2 * i + 1, 2 * N)
        for name, weights in (("native", native), ("uniform", UNIFORM),
                              ("mean", MEAN), ("zero", (F(0),) * 3)):
            rejection_margin = b + sum(x * y for x, y in zip(weights, v)) - threshold
            predictions[name].append(int(rejection_margin > 0))
    actual = {name: sum(row) for name, row in predictions.items()}
    assert [actual[k] for k in ("native", "uniform", "mean")] == list(counts)
    assert actual["zero"] == zero_count
    return {"values": values, "rest": b, "predictions": predictions, "counts": actual}


def witnesses(counts, step=100):
    """Build two witnesses matching all three supplied finite-set accuracies."""
    if len(counts) != 3 or any(type(x) is not int or not 0 <= x <= N for x in counts):
        raise ValueError("Three accuracy counts in [0, 1000] required.")
    if not 0 < step <= min(counts[0], N - counts[0]):
        raise ValueError("Both countermodels need room inside the accuracy range.")
    removal = world(counts, counts[0] + step)
    replacement = world(counts, counts[0] - step)
    for name in ("native", "uniform", "mean"):
        assert removal["predictions"][name] == replacement["predictions"][name]
    # Stronger than three matches: a common value shift cancels for ANY
    # row-normalized attention, including these exact native and mean patterns.
    shift = replacement["values"][0][0] - removal["values"][0][0]
    for va, vb in zip(removal["values"], replacement["values"]):
        assert all(b - a == shift for a, b in zip(va, vb))
    assert replacement["rest"] - removal["rest"] == -shift
    return removal, replacement


def report(counts):
    removal, replacement = witnesses(counts, step=min(100, counts[0], N-counts[0]))
    config = {
        "cells": [{"id": k} for k in ("native", "uniform", "mean", "zero")],
        "prediction_source": "Constructed exact attention witnesses; NOT empirical forecasts for the neural model.",
        "predictions": {
            label: [w["counts"][k] / N for k in ("native", "uniform", "mean", "zero")]
            for label, w in (("removal_helps", removal), ("replacement_rescues", replacement))
        },
    }
    path = ROOT.parents[1] / "examples" / "causal_preflight.py"
    spec = importlib.util.spec_from_file_location("existing_preflight", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def serialize(w):
        return {"correct_counts": w["counts"], "rest": str(w["rest"]),
                "value_vectors": [[str(v) for v in row] for row in w["values"]]}
    return {
        "status": "constructive_nonidentification_witness_not_neural_validation",
        "n_synthetic_cases": N,
        "native_patterns": [[str(v) for v in row] for row in (NATIVE_A, NATIVE_B)],
        "dataset_mean_pattern": [str(v) for v in MEAN],
        "witnesses": {"removal_helps": serialize(removal),
                      "replacement_rescues": serialize(replacement)},
        "identical_per_case_predictions_in_all_three_existing_conditions": True,
        "existing_design": module.check(config, ["native", "uniform", "mean"], structure_only=True),
        "design_with_zero_reference": module.check(config, structure_only=True),
        "preflight_input": config,
        "limit": "Zero is a specified reference, not a unique semantic absence; no real-model prediction is certified.",
    }
