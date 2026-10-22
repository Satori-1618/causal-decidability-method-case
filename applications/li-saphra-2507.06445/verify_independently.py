"""Read-only second calculation; imports neither the analyzer nor its countermodels.

Independence here means a separate arithmetic implementation, not an external
researcher, blind analysis, or independent data collection.
"""
import csv
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def correct(s):
    value = F(s) * 1000
    n = round(value)
    assert abs(value - n) < F(1, 1000000)
    assert 0 <= n <= 1000
    return n


def main():
    lock = json.loads((ROOT/"SOURCE_LOCK.json").read_text())
    for path, rec in lock["files"].items():
        data = (ROOT/"upstream"/path).read_bytes()
        assert len(data) == rec["bytes"]
        assert hashlib.sha256(data).hexdigest() == rec["sha256"]
    manifest = json.loads((ROOT/"results/manifest.json").read_text())
    for path, expected in manifest["input_and_producer_sha256"].items():
        assert hashlib.sha256((ROOT.parents[1]/path).read_bytes()).hexdigest() == expected
    for path, expected in manifest["output_sha256"].items():
        assert hashlib.sha256((ROOT/"results"/path).read_bytes()).hexdigest() == expected
    hp = load_csv(ROOT/"upstream/data/transformer_head_properties.csv")
    singles = load_csv(ROOT/"upstream/heldout/results/mean_ablation_dyck_single.csv")
    upstream = {(r["id"],int(r["layer"]),int(r["head"])):r for r in singles}
    stored = load_csv(ROOT/"results/heads.csv")
    delivered = {(r["model_id"],int(r["layer"]),int(r["head"])):r for r in stored}
    expected_keys = {(r["id"],l,h) for r in hp for l in range(1,int(r["n_layer"])+1)
                     for h in range(1,int(r["n_head"])+1)}
    assert len(upstream) == len(singles) == len(delivered) == len(stored) == 1620
    assert expected_keys == set(upstream) == set(delivered)
    target = []
    for model in hp:
        for l in range(1,int(model["n_layer"])+1):
            for h in range(1,int(model["n_head"])+1):
                key = (model["id"],l,h)
                u, out = upstream[key], delivered[key]
                n = correct(model["cp5_ood_acc"])
                a = correct(model[f"cp5_l{l}_h{h}_ood"])
                b = correct(u["ood_acc_single_mean"])
                assert n == correct(u["ood_acc"]) == correct(u["csv_ood_acc"])
                assert n == int(out["native_correct"])
                assert a == int(out["uniform_correct"])
                assert b == int(out["mean_correct"])
                assert a-n == int(out["uniform_change_count"])
                assert b-n == int(out["mean_change_count"])
                sign = F(model[f"cp5_sign_head_l{l}_h{h}_ood"]) >= F(4,5)
                viol = F(model[f"cp5_neg_head_l{l}_h{h}_ood"]) >= F(4,5)
                if int(model["n_layer"]) >= 2 and sign and not viol:
                    target.append((key,n,a,b))
    assert len(hp) == 270
    assert sum(int(r["n_layer"])>=2 for r in hp) == 180
    assert len(target) == 42 and len({r[0][0] for r in target}) == 41
    assert sum(a-n>10 and b-n>10 for _,n,a,b in target) == 40
    assert sum(a>n for _,n,a,b in target) == sum(b>n for _,n,a,b in target) == 41
    assert not any(a<n or b<n for _,n,a,b in target)
    summary = json.loads((ROOT/"results/summary.json").read_text())
    for idx,op in ((2,"uniform"),(3,"mean")):
        measured = sum(r[idx]-r[1] for r in target)/len(target)/10
        assert abs(measured-summary["single_head_by_type"]["sign-matching"][op]["head_weighted_mean_pp"])<1e-12
    example = summary["example"]
    assert (example["model_id"],example["layer"],example["head"]) == ("1aez5d6p",2,2)
    assert [example[k] for k in ("native_correct","uniform_correct","mean_correct")] == [779,823,827]

    # Re-evaluate serialized value vectors, not outputs from the producer function.
    proof = json.loads((ROOT/"results/countermodels.json").read_text())
    patterns = [tuple(map(F,row)) for row in proof["native_patterns"]]
    mean_pattern = tuple(sum(row[j] for row in patterns)/2 for j in range(3))
    assert mean_pattern == tuple(map(F,proof["dataset_mean_pattern"]))
    worlds, margins = {}, {}
    for label, w in proof["witnesses"].items():
        v = [tuple(map(F,row)) for row in w["value_vectors"]]
        rest = F(w["rest"])
        worlds[label] = {k:[] for k in ("native","uniform","mean","zero")}
        margins[label] = {k:[] for k in worlds[label]}
        for i in range(1000):
            for name,weights in (("native",patterns[i%2]),("uniform",(F(1,3),)*3),
                                 ("mean",mean_pattern),("zero",(F(0),)*3)):
                value = rest+sum(a*b for a,b in zip(weights,v[i%2]))-F(2*i+1,2000)
                worlds[label][name].append(int(value>0))
                margins[label][name].append(value)
        assert {k:sum(x) for k,x in worlds[label].items()} == w["correct_counts"]
    for condition in ("native","uniform","mean"):
        assert worlds["removal_helps"][condition] == worlds["replacement_rescues"][condition]
        assert margins["removal_helps"][condition] == margins["replacement_rescues"][condition]
    assert sum(worlds["removal_helps"]["zero"]) == 879
    assert sum(worlds["replacement_rescues"]["zero"]) == 679
    # Check the general normalized-mixture invariance on a rational simplex grid.
    ws = proof["witnesses"]
    for branch in (0,1):
        for i in range(21):
            for j in range(21-i):
                weights = (F(i,20),F(j,20),F(20-i-j,20))
                values = []
                for label in ("removal_helps","replacement_rescues"):
                    values.append(F(ws[label]["rest"])+sum(a*F(b) for a,b in zip(weights,ws[label]["value_vectors"][branch])))
                assert values[0] == values[1]
    assert len(proof["existing_design"]["identical_mean_groups"]) == 1
    assert len(proof["design_with_zero_reference"]["identical_mean_groups"]) == 2
    print("PASS: separate raw-data calculation, complete head coverage, all 42 target heads, source/output hashes, exact witness outputs and normalized-mixture invariance.")


if __name__ == "__main__":
    main()
