"""
Milestone 1 quick plot: Danna vs SWAN vs AdaptiveWaterfiller
on Uninett2010 / gravity / scale factors {1, 16, 64}.

Run from inside traffic_engineering/scripts/, after danna_practical.py,
swan.py, and approx.py (APPROACH_LIST=[constants.APPROX_BET]) have finished.
"""
import os
import sys
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../")))

from scripts import benchmark_plot_utils as bpu
from utilities import utils
from ncflow.lib.problem import Problem

# ----------------------------------------------------------------------
# 1. FILL THESE IN with your own fid folder prefixes from ../outputs/
#    (run `ls ../outputs/` and copy the exact names, minus the .txt)
# ----------------------------------------------------------------------
DANNA_PREFIX = "danna_practical"
SWAN_PREFIX = "swan"
AW_PREFIX = "heuristic"   # check your actual folder name — adjust if different

OUTPUTS_DIR = "../outputs"
THETA_FAIRNESS = 0.1
SCALE_FACTORS = [1.0, 16.0, 64.0]


def find_txt(prefix):
    matches = glob.glob(os.path.join(OUTPUTS_DIR, prefix + "*.txt"))
    if not matches:
        raise FileNotFoundError(f"No log file found matching {prefix}*.txt in {OUTPUTS_DIR}")
    return matches[0]


def parse_summary_txt(txt_path):
    """Returns {scale_factor: dict(topo_path=, tm_path=, total_flow=, duration=, per_flow_name=)}"""
    result = {}
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tuple_end = line.index("),") + 1
            tuple_str = line[:tuple_end]
            rest = line[tuple_end + 1:].split(",")
            tuple_parts = [p.strip().strip("'\"()") for p in tuple_str.split(",")]
            scale_factor = float(tuple_parts[2])
            topo_path = tuple_parts[3]
            tm_path = tuple_parts[4].rstrip(")")
            total_flow = float(rest[-3])
            duration = float(rest[-2])
            per_flow_name = rest[-1].strip()
            result[scale_factor] = dict(topo_path=topo_path, tm_path=tm_path,
                                         total_flow=total_flow, duration=duration,
                                         per_flow_name=per_flow_name)
    return result


def load_per_path_rates(fid_folder, per_flow_name):
    """Raw per-(src,dst) -> per-path-rate structure, exactly as the repo saved it."""
    path = os.path.join(OUTPUTS_DIR, fid_folder, per_flow_name + "_per_path_flow.pickle")
    with open(path, "rb") as fp:
        return pickle.load(fp)


danna_txt = find_txt(DANNA_PREFIX)
swan_txt = find_txt(SWAN_PREFIX)
aw_txt = find_txt(AW_PREFIX)

danna_data = parse_summary_txt(danna_txt)
swan_data = parse_summary_txt(swan_txt)
aw_data = parse_summary_txt(aw_txt)

danna_folder = os.path.basename(danna_txt)[:-4]
swan_folder = os.path.basename(swan_txt)[:-4]
aw_folder = os.path.basename(aw_txt)[:-4]

results = {name: {"sf": [], "fairness": [], "speedup_vs_swan": [], "eff_vs_danna": []}
           for name in ["Danna", "SWAN", "AdaptiveWaterfiller"]}

for sf in SCALE_FACTORS:
    d = danna_data[sf]
    s = swan_data[sf]
    a = aw_data[sf]

    # Rebuild the Problem for this instance to get its commodity list,
    # exactly as the original benchmark scripts did before computing.
    problem = Problem.from_file(d["topo_path"], d["tm_path"])
    utils.revise_list_commodities(problem)
    commodity_list = problem.sparse_commodity_list   # [(fid, (src, dst, demand)), ...]

    danna_rates = load_per_path_rates(danna_folder, d["per_flow_name"])
    swan_rates = load_per_path_rates(swan_folder, s["per_flow_name"])
    aw_rates = load_per_path_rates(aw_folder, a["per_flow_name"])

    # Build Danna's baseline vector in fid order (this IS the baseline, so
    # its own "fairness vs itself" will trivially come out to 1.0 — that's
    # expected and a good sanity check).
    baseline_vec = np.zeros(len(commodity_list))
    for fid, (src, dst, _) in commodity_list:
        baseline_vec[fid] = np.sum(danna_rates[(src, dst)])

    for name, rates, dur, flow in [
        ("Danna", danna_rates, d["duration"], d["total_flow"]),
        ("SWAN", swan_rates, s["duration"], s["total_flow"]),
        ("AdaptiveWaterfiller", aw_rates, a["duration"], a["total_flow"]),
    ]:
        fairness = bpu.compute_fairness_no_vectorized_baseline(
            baseline_vec, rates, commodity_list, THETA_FAIRNESS)
        results[name]["sf"].append(sf)
        results[name]["fairness"].append(fairness)
        results[name]["speedup_vs_swan"].append(s["duration"] / dur)
        results[name]["eff_vs_danna"].append(flow / d["total_flow"])

# ----------------------------------------------------------------------
# Plot: Fig 8 style (fairness vs speedup-vs-SWAN) + Fig 9 style (efficiency)
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for name, marker in [("Danna", "o"), ("SWAN", "s"), ("AdaptiveWaterfiller", "^")]:
    axes[0].scatter(results[name]["speedup_vs_swan"], results[name]["fairness"],
                     label=name, marker=marker, s=80)
axes[0].set_xscale("log")
axes[0].set_xlabel("Speedup w.r.t. SWAN")
axes[0].set_ylabel("Fairness (q_theta vs Danna)")
axes[0].set_title("Fairness vs Speed (Milestone 1, Uninett2010/gravity)")
axes[0].legend()

x = np.arange(len(SCALE_FACTORS))
width = 0.25
for i, name in enumerate(["Danna", "SWAN", "AdaptiveWaterfiller"]):
    axes[1].bar(x + i * width, results[name]["eff_vs_danna"], width, label=name)
axes[1].set_xticks(x + width)
axes[1].set_xticklabels([str(int(s)) for s in SCALE_FACTORS])
axes[1].set_xlabel("Scale factor")
axes[1].set_ylabel("Total flow relative to Danna")
axes[1].set_title("Efficiency (Milestone 1)")
axes[1].legend()

plt.tight_layout()
plt.savefig("../outputs/milestone1_fairness_efficiency.png", dpi=150)
print("Saved to ../outputs/milestone1_fairness_efficiency.png")
plt.show()