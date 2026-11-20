"""
OBIA Flood Detection – Improvement Experiments (multi-location)
===============================================================

Re-uses the object features saved by flood_obia_multisite.py
(all_sites_object_features.joblib) – NO re-segmentation needed, so it is fast.

Runs leave-one-location-out (LOLO) validation for every combination of:

  * Model ............... Random Forest / Extra Trees / Gradient Boosting / Logistic Reg.
  * Feature set ......... NDFI + SD  |  NDFI only  |  SD only        (ablation)
  * Normalisation ....... None  |  Per-site robust  ((x - median) / IQR within each location)
  * Balanced sampling ... equal number of objects per location (all flood objects kept,
                          up to half the cap) – stops one big location dominating
  * Cut-off tuning ...... flood-probability cut-off chosen to maximise F1, using ONLY the
                          training locations (inner leave-one-location-out) – no leakage

Each configuration is reported at cut-off 0.5 AND at the tuned cut-off.

Outputs (output folder):
  experiments_summary.csv      one row per configuration / cut-off
  experiments_per_site.csv     every metric for every test location
  Selected configuration  ->   final model (.joblib, usable in flood_obia_ml_gui.py
                               'Apply saved model') and LOLO flood maps per location.

Keep this file in the same folder as flood_obia_ml_gui.py and flood_obia_multisite.py.
"""

import csv
import itertools
import os
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import joblib
import numpy as np
import rasterio
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from flood_obia_ml_gui import FEATURES, make_model, site_normalise, write_tif
from flood_obia_multisite import metrics_from_counts, pixel_metrics

FEATURE_SETS = {
    "NDFI + SD": FEATURES,
    "NDFI only": [f for f in FEATURES if f.startswith("NDFI_")] + ["Area_px"],
    "SD only": [f for f in FEATURES if f.startswith("SD_")] + ["Area_px"],
}
MODEL_CHOICES = ["Random Forest", "Extra Trees", "Gradient Boosting", "Logistic Regression"]
NORMS = ["None", "Per-site robust"]
THR_GRID = np.round(np.arange(0.05, 0.951, 0.025), 3)
INNER_PRED_CAP = 200_000          # objects per location used when tuning the cut-off


# ===================================================================== core
def training_indices(d, balanced, cap, pure, rng):
    ok = np.ones(len(d["y"]), bool)
    if pure:
        ok &= (d["frac"] <= 0.2) | (d["frac"] >= 0.8)
    idx = np.where(ok)[0]
    if balanced and len(idx) > cap:
        fl = idx[d["y"][idx] == 1]
        nf = idx[d["y"][idx] == 0]
        nfl = min(len(fl), cap // 2)
        if len(fl) > nfl:
            fl = rng.choice(fl, nfl, replace=False)
        nnf = min(len(nf), cap - len(fl))
        nf = rng.choice(nf, nnf, replace=False)
        idx = np.concatenate([fl, nf])
    return idx


def best_threshold(preds):
    """preds: list of (prob, area, flood_px) per location -> cut-off maximising mean F1."""
    scores = []
    for t in THR_GRID:
        f1 = [pixel_metrics(p >= t, a, f)["F1"] for p, a, f in preds]
        scores.append(np.mean(f1))
    return float(THR_GRID[int(np.argmax(scores))])


def run_config(data, model_name, fs, norm, balanced, cap, pure, tune, status):
    rng = np.random.default_rng(0)
    cols = [FEATURES.index(f) for f in FEATURE_SETS[fs]]
    Xs = [(site_normalise(d["X"]) if norm != "None" else d["X"])[:, cols] for d in data]
    tr = [training_indices(d, balanced, cap, pure, rng) for d in data]
    sub = []                                   # subsample for inner cut-off tuning
    for d in data:
        n = len(d["y"])
        sub.append(np.arange(n) if n <= INNER_PRED_CAP else
                   np.sort(rng.choice(n, INNER_PRED_CAP, replace=False)))

    def fit(sites):
        X = np.vstack([Xs[i][tr[i]] for i in sites])
        y = np.concatenate([data[i]["y"][tr[i]] for i in sites])
        if len(np.unique(y)) < 2:
            raise ValueError("A training set contains only one class.")
        m = make_model(model_name)
        m.fit(X, y)
        return m

    n = len(data)
    res = {}
    for i in range(n):
        d = data[i]
        others = [j for j in range(n) if j != i]
        status(f"test={d['name']}: training…")
        p = fit(others).predict_proba(Xs[i])[:, 1].astype("float32")
        t = 0.5
        if tune and len(others) >= 2:
            inner = []
            for j in others:
                status(f"test={d['name']}: tuning cut-off (inner {data[j]['name']})…")
                m = fit([k for k in others if k != j])
                s = sub[j]
                inner.append((m.predict_proba(Xs[j][s])[:, 1], data[j]["area"][s], data[j]["flood_px"][s]))
            t = best_threshold(inner)
        res[d["name"]] = dict(
            prob=p, thr=t,
            m05=pixel_metrics(p >= 0.5, d["area"], d["flood_px"]),
            mt=pixel_metrics(p >= t, d["area"], d["flood_px"]))
    return res


def summarise(res, key):
    ms = [r[key] for r in res.values()]
    f1 = np.array([m["F1"] for m in ms])
    tot = {k: sum(m[k] for m in ms) for k in ["TP", "FP", "FN", "TN"]}
    pooled = metrics_from_counts(tot["TP"], tot["FP"], tot["FN"], tot["TN"])
    return dict(mean_F1=f1.mean(), sd_F1=f1.std(),
                mean_IoU=np.mean([m["IoU"] for m in ms]),
                mean_Kappa=np.mean([m["Kappa"] for m in ms]),
                mean_Prec=np.mean([m["Precision"] for m in ms]),
                mean_Rec=np.mean([m["Recall"] for m in ms]),
                pooled_F1=pooled["F1"], site_F1={k: r[key]["F1"] for k, r in res.items()})


# ====================================================================== GUI
class ExperimentsApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("OBIA Flood – Improvement Experiments (LOLO)")
        self.geometry("1500x950")
        self.feat_path = tk.StringVar()
        self.outdir = tk.StringVar()
        self.models = {m: tk.BooleanVar(value=(m == "Extra Trees")) for m in MODEL_CHOICES}
        self.fsets = {f: tk.BooleanVar(value=True) for f in FEATURE_SETS}
        self.norms = {n: tk.BooleanVar(value=True) for n in NORMS}
        self.balanced = tk.BooleanVar(value=True)
        self.cap = tk.IntVar(value=30000)
        self.tune = tk.BooleanVar(value=True)
        self.pure = tk.BooleanVar(value=False)
        self.obj_size = tk.IntVar(value=100)
        self.compact = tk.DoubleVar(value=1.0)
        self.data = None
        self.runs = []            # list of dict(cfg, res)
        self._build()

    def _build(self):
        f = ttk.LabelFrame(self, text="Inputs", padding=6); f.pack(fill="x", padx=8, pady=4)
        ttk.Label(f, text="Object features (all_sites_object_features.joblib):").grid(row=0, column=0, sticky="w")
        ttk.Entry(f, textvariable=self.feat_path, width=95).grid(row=0, column=1, padx=4)
        ttk.Button(f, text="Browse", command=self._pick_feat).grid(row=0, column=2)
        ttk.Label(f, text="Output folder:").grid(row=1, column=0, sticky="w")
        ttk.Entry(f, textvariable=self.outdir, width=95).grid(row=1, column=1, padx=4)
        ttk.Button(f, text="Browse", command=lambda: self.outdir.set(filedialog.askdirectory() or self.outdir.get())
                   ).grid(row=1, column=2)

        g = ttk.LabelFrame(self, text="Experiments (every ticked combination is run)", padding=6)
        g.pack(fill="x", padx=8, pady=4)

        def group(r, label, d):
            ttk.Label(g, text=label, width=16).grid(row=r, column=0, sticky="w")
            for k, (name, var) in enumerate(d.items()):
                ttk.Checkbutton(g, text=name, variable=var, command=self._count).grid(row=r, column=k + 1, sticky="w", padx=4)

        group(0, "Models:", self.models)
        group(1, "Feature sets:", self.fsets)
        group(2, "Normalisation:", self.norms)
        o = ttk.Frame(g); o.grid(row=3, column=0, columnspan=6, sticky="w", pady=4)
        ttk.Checkbutton(o, text="Balanced sampling, objects per location:", variable=self.balanced).pack(side="left")
        ttk.Spinbox(o, from_=5000, to=500000, increment=5000, textvariable=self.cap, width=8).pack(side="left", padx=4)
        ttk.Checkbutton(o, text="Tune cut-off (inner LOLO)", variable=self.tune).pack(side="left", padx=12)
        ttk.Checkbutton(o, text="Pure objects only", variable=self.pure).pack(side="left", padx=12)
        ttk.Label(o, text="   Segmentation used (for saved model) – object size:").pack(side="left")
        ttk.Spinbox(o, from_=10, to=5000, increment=10, textvariable=self.obj_size, width=6).pack(side="left", padx=2)
        ttk.Label(o, text="compactness:").pack(side="left")
        ttk.Spinbox(o, from_=0.1, to=20, increment=0.5, textvariable=self.compact, width=5).pack(side="left", padx=2)

        b = ttk.Frame(self); b.pack(fill="x", padx=8)
        ttk.Button(b, text="Run experiments", command=self.run).pack(side="left")
        ttk.Button(b, text="Save final model for selected row", command=self.save_final).pack(side="left", padx=8)
        ttk.Button(b, text="Write LOLO maps for selected row", command=self.write_maps).pack(side="left")
        self.count_lbl = ttk.Label(b, text="", foreground="gray"); self.count_lbl.pack(side="left", padx=12)
        self.status = ttk.Label(self, text="Ready.", foreground="gray"); self.status.pack(fill="x", padx=10)

        body = ttk.PanedWindow(self, orient="vertical"); body.pack(fill="both", expand=True, padx=8, pady=4)
        tf = ttk.Frame(body)
        self.cols = ["#", "Model", "Features", "Norm", "Cut-off", "Mean F1", "± SD", "Pooled F1",
                     "Mean IoU", "Mean Kappa", "Mean Prec", "Mean Rec"]
        self.tree = ttk.Treeview(tf, columns=self.cols, show="headings", height=10)
        for c in self.cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=150 if c in ("Model", "Features", "Norm") else 80, anchor="center")
        sb = ttk.Scrollbar(tf, command=self.tree.yview); self.tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y"); self.tree.pack(fill="both", expand=True)
        body.add(tf, weight=1)
        ff = ttk.Frame(body)
        self.fig = Figure(figsize=(12, 4.5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=ff)
        NavigationToolbar2Tk(self.canvas, ff)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        body.add(ff, weight=2)
        self._count()

    def _pick_feat(self):
        p = filedialog.askopenfilename(filetypes=[("joblib", "*.joblib")])
        if p:
            self.feat_path.set(p)
            if not self.outdir.get():
                self.outdir.set(os.path.join(os.path.dirname(p), "experiments"))

    def _sel(self, d):
        return [k for k, v in d.items() if v.get()]

    def _count(self):
        n = len(self._sel(self.models)) * len(self._sel(self.fsets)) * len(self._sel(self.norms))
        self.count_lbl.config(text=f"{n} configuration(s)")

    def _st(self, m):
        self.status.config(text=m); self.update_idletasks()

    # -------------------------------------------------------------- run
    def run(self):
        try:
            if not self.feat_path.get():
                raise ValueError("Select all_sites_object_features.joblib (made by flood_obia_multisite.py).")
            out = self.outdir.get() or os.path.dirname(self.feat_path.get())
            os.makedirs(out, exist_ok=True)
            self._st("Loading object features…")
            self.data = joblib.load(self.feat_path.get())
            if len(self.data) < 3:
                raise ValueError("Need at least 3 locations (cut-off tuning needs 2 training locations).")
            combos = list(itertools.product(self._sel(self.models), self._sel(self.fsets), self._sel(self.norms)))
            if not combos:
                raise ValueError("Tick at least one model, feature set and normalisation.")
            self.runs = []
            self.tree.delete(*self.tree.get_children())
            t0 = time.time()
            for k, (m, fs, nm) in enumerate(combos, 1):
                pre = f"[{k}/{len(combos)}] {m} | {fs} | {nm}: "
                res = run_config(self.data, m, fs, nm, self.balanced.get(), self.cap.get(),
                                 self.pure.get(), self.tune.get(), lambda s: self._st(pre + s))
                cfg = dict(model=m, fs=fs, norm=nm, balanced=self.balanced.get(),
                           cap=self.cap.get(), pure=self.pure.get(), tune=self.tune.get())
                self.runs.append(dict(cfg=cfg, res=res))
                self._add_rows(len(self.runs) - 1)
                self._export(out)
                self._plot()
            self._st(f"Done in {(time.time() - t0) / 60:.1f} min. Results in {out}. "
                     "Select a row to save its final model or write its maps.")
        except Exception as e:
            self._st("Error.")
            messagebox.showerror("Error", str(e))
            raise

    def _rows_for(self, i):
        r = self.runs[i]
        keys = [("0.5", "m05")] + ([("tuned", "mt")] if r["cfg"]["tune"] else [])
        return [(lab, summarise(r["res"], key)) for lab, key in keys]

    def _add_rows(self, i):
        c = self.runs[i]["cfg"]
        for lab, s in self._rows_for(i):
            if lab == "tuned":
                ts = [v["thr"] for v in self.runs[i]["res"].values()]
                lab = f"tuned ({min(ts):.2f}–{max(ts):.2f})"
            self.tree.insert("", "end", iid=f"{i}|{lab[:5]}", values=(
                i + 1, c["model"], c["fs"], c["norm"], lab,
                f"{s['mean_F1']:.3f}", f"{s['sd_F1']:.3f}", f"{s['pooled_F1']:.3f}",
                f"{s['mean_IoU']:.3f}", f"{s['mean_Kappa']:.3f}",
                f"{s['mean_Prec']:.3f}", f"{s['mean_Rec']:.3f}"))

    def _export(self, out):
        sites = [d["name"] for d in self.data]
        with open(os.path.join(out, "experiments_summary.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["Run", "Model", "Features", "Normalisation", "Balanced", "Cap", "Pure_only", "Cut-off",
                        "Mean_F1", "SD_F1", "Pooled_F1", "Mean_IoU", "Mean_Kappa", "Mean_Precision",
                        "Mean_Recall"] + [f"F1_{s}" for s in sites])
            for i, r in enumerate(self.runs):
                c = r["cfg"]
                for lab, s in self._rows_for(i):
                    w.writerow([i + 1, c["model"], c["fs"], c["norm"], c["balanced"], c["cap"], c["pure"], lab,
                                *(f"{s[k]:.4f}" for k in ["mean_F1", "sd_F1", "pooled_F1", "mean_IoU",
                                                           "mean_Kappa", "mean_Prec", "mean_Rec"]),
                                *(f"{s['site_F1'][x]:.4f}" for x in sites)])
        with open(os.path.join(out, "experiments_per_site.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            keys = ["OA", "Precision", "Recall", "F1", "IoU", "Kappa", "TP", "FP", "FN", "TN"]
            w.writerow(["Run", "Model", "Features", "Normalisation", "Test_site", "Cut-off_type",
                        "Cut-off"] + keys)
            for i, r in enumerate(self.runs):
                c = r["cfg"]
                for site, v in r["res"].items():
                    for lab, key, t in [("0.5", "m05", 0.5), ("tuned", "mt", v["thr"])]:
                        if lab == "tuned" and not c["tune"]:
                            continue
                        w.writerow([i + 1, c["model"], c["fs"], c["norm"], site, lab, t] +
                                   [f"{v[key][k]:.4f}" for k in keys])

    def _plot(self):
        self.fig.clear()
        rows = []
        for i in range(len(self.runs)):
            c = self.runs[i]["cfg"]
            for lab, s in self._rows_for(i):
                rows.append((f"{i + 1}. {c['model'].split()[0]} | {c['fs']} | "
                             f"{'norm' if c['norm'] != 'None' else 'raw'} | {lab}", s))
        ax = self.fig.add_subplot(1, 2, 1)
        names = [r[0] for r in rows][::-1]
        ax.barh(names, [r[1]["mean_F1"] for r in rows][::-1],
                xerr=[r[1]["sd_F1"] for r in rows][::-1], color="#4a7fb5", capsize=2)
        ax.set_xlim(0, 1); ax.set_xlabel("Mean LOLO F1 (± SD across locations)")
        ax.tick_params(axis="y", labelsize=7)
        ax2 = self.fig.add_subplot(1, 2, 2)
        sites = [d["name"] for d in self.data]
        M = np.array([[r[1]["site_F1"][s] for s in sites] for r in rows])
        im = ax2.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax2.set_xticks(range(len(sites))); ax2.set_xticklabels(sites)
        ax2.set_yticks(range(len(rows))); ax2.set_yticklabels([r[0].split(".")[0] + ". " + r[0].split("|")[-1]
                                                               for r in rows], fontsize=7)
        for a in range(M.shape[0]):
            for b in range(M.shape[1]):
                ax2.text(b, a, f"{M[a, b]:.2f}", ha="center", va="center", fontsize=7)
        ax2.set_title("F1 per test location (row = run | cut-off)")
        self.fig.colorbar(im, ax=ax2, fraction=0.03)
        self.fig.tight_layout()
        self.canvas.draw()

    # ------------------------------------------------- selected row actions
    def _selected(self):
        sel = self.tree.selection()
        if not sel:
            raise ValueError("Select a row in the results table first.")
        i, lab = sel[0].split("|")
        return int(i), lab.startswith("tuned")

    def save_final(self):
        try:
            i, tuned = self._selected()
            r = self.runs[i]; c = r["cfg"]
            out = self.outdir.get()
            self._st("Training final model on all locations…")
            rng = np.random.default_rng(0)
            cols = [FEATURES.index(f) for f in FEATURE_SETS[c["fs"]]]
            Xs = [(site_normalise(d["X"]) if c["norm"] != "None" else d["X"])[:, cols] for d in self.data]
            tr = [training_indices(d, c["balanced"], c["cap"], c["pure"], rng) for d in self.data]
            model = make_model(c["model"])
            model.fit(np.vstack([Xs[k][tr[k]] for k in range(len(Xs))]),
                      np.concatenate([d["y"][tr[k]] for k, d in enumerate(self.data)]))
            # cut-off for the final model: best on the out-of-location (LOLO) predictions
            thr = best_threshold([(v["prob"], d["area"], d["flood_px"])
                                  for d, v in zip(self.data, r["res"].values())]) if tuned else 0.5
            tag = f"{c['model'].split()[0]}_{c['fs'].replace(' ', '').replace('+', '_')}_" \
                  f"{'norm' if c['norm'] != 'None' else 'raw'}"
            path = os.path.join(out, f"flood_model_FINAL_{tag}.joblib")
            joblib.dump(dict(model=model, features=FEATURE_SETS[c["fs"]], feat_idx=cols,
                             site_norm=c["norm"] != "None", threshold=thr,
                             nlims=None, slims=None, obj_size=self.obj_size.get(),
                             compact=self.compact.get(),
                             model_name=f"{c['model']} | {c['fs']} | {c['norm']} | all locations"), path)
            self._st(f"Saved {path} (cut-off {thr:.3f}).")
            messagebox.showinfo("Saved", f"{path}\n\nCut-off: {thr:.3f}\n"
                                         "Apply it with flood_obia_ml_gui.py → 'Apply saved model'.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def write_maps(self):
        try:
            i, tuned = self._selected()
            r = self.runs[i]; c = r["cfg"]
            out = self.outdir.get()
            tag = f"{c['model'].split()[0]}_{'tuned' if tuned else '05'}"
            for d in self.data:
                v = r["res"][d["name"]]
                if not os.path.exists(d["lab_path"]):
                    raise ValueError(f"Object map not found: {d['lab_path']}")
                self._st(f"Writing map for {d['name']}…")
                with rasterio.open(d["lab_path"]) as src:
                    labels = src.read(1); prof = src.profile.copy()
                lut = np.full(len(v["prob"]) + 1, -1, "float32"); lut[1:] = v["prob"]
                prob = lut[labels]
                t = v["thr"] if tuned else 0.5
                write_tif(os.path.join(out, f"{d['name']}_floodprob_run{i + 1}_{tag}.tif"), prob, prof, "float32", -1)
                write_tif(os.path.join(out, f"{d['name']}_flood_run{i + 1}_{tag}.tif"),
                          np.where(prob < 0, 255, prob >= t).astype("uint8"), prof, "uint8", 255)
                del labels, prob
            self._st(f"Maps written to {out}")
        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    ExperimentsApp().mainloop()
