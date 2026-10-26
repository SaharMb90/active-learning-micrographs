"""End-to-end demonstrator.

1. Generate synthetic micrographs from a source instrument (A) and a shifted
   target instrument (B: noisier, blurrier, lower contrast).
2. Build three representations (raw-pixel PCA, multi-scale texture,
   texture + unsupervised instrument alignment).
3. Label images from A one batch at a time, choosing them either at random or
   by Gaussian-process predictive uncertainty, and evaluate accuracy and
   uncertainty calibration on held-out images from A and from B.
"""
import json, sys, time, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
from data import make_dataset
from representations import PixelPCA, Texture, TextureAligned
from active_learning import run

t0 = time.time()
XA, yA, sA = make_dataset(600, "A", seed=1)      # labelled pool (source)
XA_t, yA_t, sA_t = make_dataset(300, "A", seed=4)  # source test set
XB_t, yB_t, sB_t = make_dataset(300, "B", seed=2)  # target test set (never labelled)
XU, _, sU = make_dataset(600, "B", seed=3)         # unlabelled target images

SEEDS, BUDGET = [0, 1, 2, 3, 4], 80
results = {}
for rep in [PixelPCA(), Texture(), TextureAligned()]:
    rep.fit(XA, sA, XU, sU)
    Zp = rep.transform(XA, sA, "source")
    tests = {"A": (rep.transform(XA_t, sA_t, "source"), yA_t),
             "B": (rep.transform(XB_t, sB_t, "target"), yB_t)}
    for strat in ("random", "uncertainty"):
        for dom, (Zt, yt) in tests.items():
            curves = np.array([run(Zp, yA, Zt, yt, strat, budget=BUDGET, seed=s) for s in SEEDS])
            results[f"{rep.name}|{strat}|{dom}"] = {"mean": curves.mean(0).tolist(), "std": curves.std(0).tolist()}
    print(f"{rep.name}: done ({time.time()-t0:.0f}s)", flush=True)

json.dump(results, open("results/results.json", "w"), indent=1)

# ---- figure: RMSE vs labels on A and B, and calibration on B
colors = {"pixel_pca": "tab:gray", "texture": "tab:blue", "texture_aligned": "tab:orange"}
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for key, v in results.items():
    name, strat, dom = key.split("|")
    c = np.array(v["mean"]); ls = "-" if strat == "uncertainty" else "--"
    lab = f"{name} ({strat})"
    if dom == "A":
        ax[0].plot(c[:, 0], c[:, 1], ls, color=colors[name], label=lab)
    else:
        ax[1].plot(c[:, 0], c[:, 1], ls, color=colors[name])
        ax[2].plot(c[:, 0], c[:, 2], ls, color=colors[name])
for a, d in zip(ax[:2], ("A (same instrument)", "B (shifted instrument)")):
    a.axhline(yB_t.std() if "B" in d else yA_t.std(), color="k", lw=0.8, ls=":")
    a.set(xlabel="labelled images from A", ylabel="RMSE (hardness proxy)", title=f"Test on {d}")
ax[2].axhline(0.9, color="k", lw=0.8, ls=":")
ax[2].set(xlabel="labelled images from A", ylabel="empirical coverage of 90% interval", title="Calibration on B", ylim=(0, 1.05))
ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig("results/learning_curves.png", dpi=130)

# ---- example images
fig, ax = plt.subplots(2, 6, figsize=(12, 4.4))
for i in range(6):
    ax[0, i].imshow(XA_t[i], cmap="gray"); ax[1, i].imshow(XB_t[i], cmap="gray")
    ax[0, i].set_title(f"A, {sA_t[i]} um/px", fontsize=8); ax[1, i].set_title(f"B, {sB_t[i]} um/px", fontsize=8)
for a in ax.ravel(): a.axis("off")
plt.tight_layout(); plt.savefig("results/example_micrographs.png", dpi=110)

lines = [f"Final results at {BUDGET} labels, mean +/- std over {len(SEEDS)} seeds",
         f"(predict-the-mean RMSE: A {yA_t.std():.3f}, B {yB_t.std():.3f})", "",
         f"{'representation':18s}{'strategy':13s}{'RMSE A':>14s}{'RMSE B':>14s}{'cov90 B':>9s}"]
for rep in ("pixel_pca", "texture", "texture_aligned"):
    for strat in ("random", "uncertainty"):
        a = results[f"{rep}|{strat}|A"]; b = results[f"{rep}|{strat}|B"]
        lines.append(f"{rep:18s}{strat:13s}{a['mean'][-1][1]:8.3f}±{a['std'][-1][1]:.3f}"
                     f"{b['mean'][-1][1]:8.3f}±{b['std'][-1][1]:.3f}{b['mean'][-1][2]:9.2f}")
open("results/summary.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
