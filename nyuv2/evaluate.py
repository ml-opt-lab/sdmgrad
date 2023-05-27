import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import itertools

methods = [
    "sdmgrad-1e-1", "sdmgrad-2e-1", "sdmgrad-3e-1", "sdmgrad-4e-1", "sdmgrad-5e-1", "sdmgrad-6e-1", "sdmgrad-7e-1",
    "sdmgrad-8e-1", "sdmgrad-9e-1", "sdmgrad-1e0"
]

colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "tab:green", "tab:cyan", "tab:blue", "tab:red"]
stats = [
    "semantic loss", "mean iou", "pix acc", "depth loss", "abs err", "rel err", "normal loss", "mean", "median",
    "<11.25", "<22.5", "<30"
]

delta_stats = ["mean iou", "pix acc", "abs err", "rel err", "mean", "median", "<11.25", "<22.5", "<30"]

stats_idx_map = [4, 5, 6, 8, 9, 10, 12, 13, 14, 15, 16, 17]

time_idx = 34

# change random seeds used in the experiments here
seeds = [0, 1, 2]

logs = {}
min_epoch = 100000

for m in methods:
    logs[m] = {"train": [None for _ in range(3)], "test": [None for _ in range(3)]}

    for seed in seeds:
        logs[m]["train"][seed] = {}
        logs[m]["test"][seed] = {}

    for stat in stats:
        for seed in seeds:
            logs[m]["train"][seed][stat] = []
            logs[m]["test"][seed][stat] = []

    for seed in seeds:
        logs[m]["train"][seed]["time"] = []

    for seed in seeds:
        fname = f"logs/{m}-sd{seed}.log"
        with open(fname, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("Epoch"):
                    ws = line.split(" ")
                    for i, stat in enumerate(stats):
                        logs[m]["train"][seed][stat].append(float(ws[stats_idx_map[i]]))
                        logs[m]["test"][seed][stat].append(float(ws[stats_idx_map[i] + 15]))
                    logs[m]["train"][seed]["time"].append(float(ws[time_idx]))
            min_epoch = min(min(min_epoch, len(logs[m]["train"][seed]["semantic loss"])),
                            len(logs[m]["test"][seed]["semantic loss"]))

test_stats = {}
train_stats = {}
learning_time = {}

print(" " * 25 + " | ".join([f"{s:5s}" for s in stats]))

for mi, mode in enumerate(["train", "test"]):
    if mi == 1:
        print(mode)
    for mmi, m in enumerate(methods):
        if m not in test_stats:
            test_stats[m] = {}
            train_stats[m] = {}

        string = f"{m:30s} "
        for stat in stats:
            x = []
            for seed in seeds:
                x.append(np.array(logs[m][mode][seed][stat][min_epoch - 10:min_epoch]).mean())
            x = np.array(x)
            if mode == "test":
                test_stats[m][stat] = x.copy()
            else:
                train_stats[m][stat] = x.copy()
            mu = x.mean()
            std = x.std() / np.sqrt(3)
            string += f" | {mu:5.4f}"
        if mode == "test":
            print(string)

for m in methods:
    learning_time[m] = np.array([np.array(logs[m]["train"][sd]["time"]).mean() for sd in seeds])

### print average training loss
for method in methods:
    average_loss = np.mean([
        train_stats[method]["semantic loss"].mean(), train_stats[method]["depth loss"].mean(),
        train_stats[method]["normal loss"].mean()
    ])
    print(f"{method} average training loss {average_loss}")

### print delta M

base = np.array([0.3830, 0.6376, 0.6754, 0.2780, 25.01, 19.21, 0.3014, 0.5720, 0.6915])
sign = np.array([1, 1, 0, 0, 0, 0, 1, 1, 1])
kk = np.ones(9) * -1


def delta_fn(a):
    return (kk**sign * (a - base) / base).mean() * 100.  # *100 for percentage


deltas = {}
for method in methods:
    tmp = np.zeros(9)
    for i, stat in enumerate(delta_stats):
        tmp[i] = test_stats[method][stat].mean()
    deltas[method] = delta_fn(tmp)
    print(f"{method:30s} delta: {deltas[method]:4.3f}")
