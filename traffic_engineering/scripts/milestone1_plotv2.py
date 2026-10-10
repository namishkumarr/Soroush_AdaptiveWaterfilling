"""
Milestone 1 plots (v2): Danna vs SWAN vs AdaptiveWaterfiller
on Uninett2010, 2 distributions (gravity, poisson-high-inter),
6 scale factors each: {1, 4, 16, 32, 64, 128}.

Produces 2 combined plots (NOT split by distribution):
  Plot 1: Runtime vs Load Factor (line plot, log-log)
          one line per algorithm, solid=gravity, dashed=poisson
  Plot 2: Fairness vs Speedup (scatter, Fig.8-style)
          color=algorithm, filled marker=gravity, hollow marker=poisson

Run from inside traffic_engineering/scripts/, after all 6 runs
(danna/swan/approx x gravity/poisson) have finished and their
output folders are on disk under ../outputs/.
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
# 1. FILL THESE IN — the fid-folder PREFIXES for each of your 6 runs.
#    Run `ls ../outputs/` and copy exact names (minus the trailing
#    timestamp/hash is fine, glob matches on prefix).
#    If you deleted-and-reran as discussed, each (algo, distribution)
#    pair should have exactly ONE matching folder now.
# ----------------------------------------------------------------------
RUNS = {
    ("Danna", "gravity"):              "danna_practical",
    ("SWAN", "gravity"):                "swan",
    ("AdaptiveWaterfiller", "gravity"): "heuristic",          # check your real prefix
    ("Danna", "poisson-high-inter"):              "danna_practical",
    ("SWAN", "poisson-high-inter"):                "swan",
    ("AdaptiveWaterfiller", "poisson-high-inter"): "heuristic",
}

OUTPUTS_DIR = "../outputs"
THETA_FAIRNESS = 0.1
SCALE_FACTORS = [1.0, 4.0, 16.0, 32.0, 64.0, 128.0]
ALGOS = ["Danna", "SWAN", "AdaptiveWaterfiller"]
DISTS = ["gravity", "poisson-high-inter"]

COLORS = {"Danna": "tab:blue", "SWAN": "tab:orange", "AdaptiveWaterfiller": "tab:green"}
LINESTYLE = {"gravity": "-", "poisson-high-inter": "--"}
MARKER = {"Danna": "o", "SWAN": "s", "AdaptiveWaterfiller": "^"}


# ----------------------------------------------------------------------
# NOTE: since each (algo, distribution) has its own run and own folder,
# but TM_MODEL_LIST only had ONE distribution active per run, each
# folder's .txt contains all 6 scale factors for THAT ONE distribution.
# We disambiguate folders below by checking which distribution's .pkl
# paths actually appear inside each log file's tm_path column.
# ----------------------------------------------------------------------

def find_all_txts(prefix):
    return sorted(glob.glob(os.path.join(OUTPUTS_DIR, prefix + "*.txt")),
                  key=os.path.getmtime)


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


def pick_txt_for_distribution(prefix, distribution):
    candidates = find_all_txts(prefix)
    for txt in candidates:
        data = parse_summary_txt(txt)
        sample = next(iter(data.values()))
        tm = sample["tm_path"].replace("\\", "/")
        if f"/{distribution}/" in tm:
            return txt, data
    raise FileNotFoundError(
        f"No {prefix}*.txt file found whose traffic-matrix path matches "
        f"distribution '{distribution}'. Candidates checked: {candidates}")


def load_per_path_rates(fid_folder, per_flow_name):
    path = os.path.join(OUTPUTS_DIR, fid_folder, per_flow_name + "_per_path_flow.pickle")
    with open(path, "rb") as fp:
        return pickle.load(fp)


# ----------------------------------------------------------------------
# 2. Load everything
# ----------------------------------------------------------------------
# data[algo][dist] = parsed summary dict {scale_factor: {...}}
# folder[algo][dist] = fid folder name (for loading pickles)
data = {}
folder = {}
for algo in ALGOS:
    data[algo] = {}
    folder[algo] = {}
    for dist in DISTS:
        prefix = RUNS[(algo, dist)]
        txt_path, parsed = pick_txt_for_distribution(prefix, dist)
        data[algo][dist] = parsed
        folder[algo][dist] = os.path.basename(txt_path)[:-4]
        print(f"[ok] {algo} / {dist} -> {folder[algo][dist]} "
              f"({len(parsed)} scale factors)")

# ----------------------------------------------------------------------
# 3. Compute fairness (vs Danna, same distribution) per (algo, dist, sf)
# ----------------------------------------------------------------------
fairness = {algo: {dist: {} for dist in DISTS} for algo in ALGOS}
speedup_vs_swan = {algo: {dist: {} for dist in DISTS} for algo in ALGOS}
eff_vs_danna = {algo: {dist: {} for dist in DISTS} for algo in ALGOS}
runtime = {algo: {dist: {} for dist in DISTS} for algo in ALGOS}

for dist in DISTS:
    for sf in SCALE_FACTORS:
        d = data["Danna"][dist][sf]
        s = data["SWAN"][dist][sf]

        problem = Problem.from_file(d["topo_path"], d["tm_path"])
        utils.revise_list_commodities(problem)
        commodity_list = problem.sparse_commodity_list

        danna_rates = load_per_path_rates(folder["Danna"][dist], d["per_flow_name"])
        baseline_vec = np.zeros(len(commodity_list))
        for fid, (src, dst, _) in commodity_list:
            baseline_vec[fid] = np.sum(danna_rates[(src, dst)])

        for algo in ALGOS:
            entry = data[algo][dist][sf]
            rates = load_per_path_rates(folder[algo][dist], entry["per_flow_name"])
            f_no = bpu.compute_fairness_no_vectorized_baseline(
                baseline_vec, rates, commodity_list, THETA_FAIRNESS)
            fairness[algo][dist][sf] = f_no
            speedup_vs_swan[algo][dist][sf] = s["duration"] / entry["duration"]
            eff_vs_danna[algo][dist][sf] = entry["total_flow"] / d["total_flow"]
            runtime[algo][dist][sf] = entry["duration"]

# ----------------------------------------------------------------------
# PLOT 1: Runtime vs Load Factor (log-log line plot)
# ----------------------------------------------------------------------
plt.figure(figsize=(7, 5))
for algo in ALGOS:
    for dist in DISTS:
        ys = [runtime[algo][dist][sf] for sf in SCALE_FACTORS]
        plt.plot(SCALE_FACTORS, ys,
                  color=COLORS[algo], linestyle=LINESTYLE[dist],
                  marker=MARKER[algo], label=f"{algo} ({dist})")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Scale factor (load)")
plt.ylabel("Runtime (s)")
plt.title("Runtime vs Load Factor — Uninett2010")
plt.legend(fontsize=8)
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("../outputs/milestone1_runtime_vs_load.png", dpi=150)
print("Saved ../outputs/milestone1_runtime_vs_load.png")

# ----------------------------------------------------------------------
# PLOT 2: Fairness vs Speedup (Fig. 8 style scatter)
# ----------------------------------------------------------------------
plt.figure(figsize=(7, 5))
for algo in ALGOS:
    for dist in DISTS:
        xs = [speedup_vs_swan[algo][dist][sf] for sf in SCALE_FACTORS]
        ys = [fairness[algo][dist][sf] for sf in SCALE_FACTORS]
        facecolor = COLORS[algo] if dist == "gravity" else "none"
        plt.scatter(xs, ys, s=90, marker=MARKER[algo],
                    facecolors=facecolor, edgecolors=COLORS[algo], linewidths=1.5,
                    label=f"{algo} ({dist})")
plt.xscale("log")
plt.xlabel("Speedup w.r.t. SWAN")
plt.ylabel("Fairness (q_theta vs Danna)")
plt.title("Fairness vs Speed — Uninett2010")
plt.legend(fontsize=8)
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("../outputs/milestone1_fairness_vs_speed.png", dpi=150)
print("Saved ../outputs/milestone1_fairness_vs_speed.png")

plt.show()