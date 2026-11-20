"""
Multi-Location OBIA Flood Detection (NDFI + Temporal Std-Dev)
=============================================================

Uses several locations (each with NDFI, StdDev and a flood reference layer) to
train and validate flood-detection models on image objects.

Validation: LEAVE-ONE-LOCATION-OUT (LOLO)
    For each location L:  train on all other locations -> test on L.
    This measures how well the model works at a location it has NEVER seen.
    Each location's flood map is produced by the model that did NOT see it.

Final model: trained on ALL locations, saved as .joblib.
    It can be applied to new scenes with the "Apply saved model" tab of
    flood_obia_ml_gui.py (keep both .py files in the same folder).

Runs on CPU only (no GPU needed). Uses all CPU cores for tree models.

Requirements (conda env "flood"):
    numpy rasterio scikit-image matplotlib scipy scikit-learn joblib
"""

import csv
import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import joblib
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

# reuse processing functions from the single-scene tool (same folder)
from flood_obia_ml_gui import (FEATURES, MODELS, SVM_MAX_TRAIN, align_to_reference,
                               crop_to, make_model, normalise, object_features,
                               object_flood_fraction, read_raster, segment_objects,
                               write_tif)
from sklearn.inspection import permutation_importance

METRIC_KEYS = ["OA", "Precision", "Recall", "F1", "IoU", "Kappa"]


def metrics_from_counts(tp, fp, fn, tn):
    n = tp + fp + fn + tn
    oa = (tp + tn) / n if n else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else np.nan
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2 if n else np.nan
    kap = (oa - pe) / (1 - pe) if n and pe != 1 else np.nan
    return dict(OA=oa, Precision=prec, Recall=rec, F1=f1, IoU=iou, Kappa=kap,
                TP=tp, FP=fp, FN=fn, TN=tn)


def pixel_metrics(pred, area, flood_px):
    """Pixel-level metrics computed from objects (area and flood pixels per object)."""
    tp = flood_px[pred].sum(); fp = (area - flood_px)[pred].sum()
    fn = flood_px[~pred].sum(); tn = (area - flood_px)[~pred].sum()
    return metrics_from_counts(float(tp), float(fp), float(fn), float(tn))


def object_metrics(pred, y):
    y = y.astype(bool)
    return metrics_from_counts(float((pred & y).sum()), float((pred & ~y).sum()),
                               float((~pred & y).sum()), float((~pred & ~y).sum()))


class MultiSiteApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Multi-Location OBIA Flood Detection – Leave-One-Location-Out")
        self.geometry("1450x930")
        self.sites = []                     # dicts: name, ndfi, sd, flood
        self.outdir = tk.StringVar()
        self.model_name = tk.StringVar(value="Random Forest")
        self.compare_all = tk.BooleanVar(value=False)
        self.obj_size = tk.IntVar(value=100)
        self.compact = tk.DoubleVar(value=1.0)
        self.pure_only = tk.BooleanVar(value=False)
        self.results = None
        self._build()

    # ------------------------------------------------------------------ layout
    def _build(self):
        top = ttk.LabelFrame(self, text="Locations (each needs NDFI, StdDev and flood layer)", padding=8)
        top.pack(fill="x", padx=8, pady=6)
        cols = ("Name", "NDFI", "StdDev", "Flood")
        self.tree = ttk.Treeview(top, columns=cols, show="headings", height=6)
        for c, w in zip(cols, (120, 380, 380, 380)):
            self.tree.heading(c, text=c); self.tree.column(c, width=w)
        self.tree.grid(row=0, column=0, rowspan=4, sticky="nsew")
        ttk.Button(top, text="Add location", command=self.add_site).grid(row=0, column=1, padx=6, sticky="ew")
        ttk.Button(top, text="Remove selected", command=self.remove_site).grid(row=1, column=1, padx=6, sticky="ew")
        ttk.Button(top, text="Save list", command=self.save_list).grid(row=2, column=1, padx=6, sticky="ew")
        ttk.Button(top, text="Load list", command=self.load_list).grid(row=3, column=1, padx=6, sticky="ew")

        opt = ttk.Frame(self, padding=(8, 0)); opt.pack(fill="x")
        ttk.Label(opt, text="Output folder:").grid(row=0, column=0, sticky="w")
        ttk.Entry(opt, textvariable=self.outdir, width=90).grid(row=0, column=1, columnspan=6, sticky="w", padx=4)
        ttk.Button(opt, text="Browse", command=lambda: self.outdir.set(filedialog.askdirectory() or self.outdir.get())
                   ).grid(row=0, column=7)
        r = ttk.Frame(opt); r.grid(row=1, column=0, columnspan=8, sticky="w", pady=6)
        ttk.Label(r, text="Model:").pack(side="left")
        ttk.Combobox(r, textvariable=self.model_name, values=MODELS, width=18,
                     state="readonly").pack(side="left", padx=4)
        ttk.Checkbutton(r, text="Compare all models", variable=self.compare_all).pack(side="left", padx=8)
        ttk.Label(r, text="Object size (px):").pack(side="left", padx=(10, 0))
        ttk.Spinbox(r, from_=10, to=5000, increment=10, textvariable=self.obj_size, width=6).pack(side="left", padx=4)
        ttk.Label(r, text="Compactness:").pack(side="left", padx=(10, 0))
        ttk.Spinbox(r, from_=0.1, to=20, increment=0.5, textvariable=self.compact, width=5).pack(side="left", padx=4)
        ttk.Checkbutton(r, text="Train on pure objects only", variable=self.pure_only).pack(side="left", padx=10)
        ttk.Button(r, text="Run (LOLO validation + final model)", command=self.run).pack(side="left", padx=10)

        self.status = ttk.Label(self, text="Add at least 2 locations.", foreground="gray")
        self.status.pack(fill="x", padx=10)

        body = ttk.PanedWindow(self, orient="horizontal")
        body.pack(fill="both", expand=True, padx=8, pady=4)
        ff = ttk.Frame(body)
        self.fig = Figure(figsize=(8.5, 6.5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=ff)
        NavigationToolbar2Tk(self.canvas, ff)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        body.add(ff, weight=3)
        tf = ttk.Frame(body)
        self.text = tk.Text(tf, wrap="none", font=("Courier", 9))
        sb = ttk.Scrollbar(tf, command=self.text.yview)
        self.text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y"); self.text.pack(fill="both", expand=True)
        body.add(tf, weight=2)

    # ---------------------------------------------------------- site list
    def _refresh(self):
        self.tree.delete(*self.tree.get_children())
        for s in self.sites:
            self.tree.insert("", "end", values=(s["name"], s["ndfi"], s["sd"], s["flood"]))
        self.status.config(text=f"{len(self.sites)} location(s).")

    def add_site(self):
        ft = [("GeoTIFF", "*.tif *.tiff"), ("All", "*.*")]
        ndfi = filedialog.askopenfilename(title="1/3  Select NDFI", filetypes=ft)
        if not ndfi: return
        sd = filedialog.askopenfilename(title="2/3  Select temporal StdDev", filetypes=ft,
                                        initialdir=os.path.dirname(ndfi))
        if not sd: return
        fl = filedialog.askopenfilename(title="3/3  Select flood reference", filetypes=ft,
                                        initialdir=os.path.dirname(ndfi))
        if not fl: return
        name = simpledialog.askstring("Location name", "Name for this location:",
                                      initialvalue=f"Site{len(self.sites) + 1}")
        if not name: return
        self.sites.append(dict(name=name, ndfi=ndfi, sd=sd, flood=fl))
        if not self.outdir.get():
            self.outdir.set(os.path.join(os.path.dirname(ndfi), "obia_multisite_output"))
        self._refresh()

    def remove_site(self):
        sel = [self.tree.index(i) for i in self.tree.selection()]
        self.sites = [s for i, s in enumerate(self.sites) if i not in sel]
        self._refresh()

    def save_list(self):
        p = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not p: return
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["name", "ndfi", "sd", "flood"])
            w.writeheader(); w.writerows(self.sites)

    def load_list(self):
        p = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not p: return
        with open(p, newline="") as fh:
            self.sites = [dict(r) for r in csv.DictReader(fh)]
        self._refresh()

    # ---------------------------------------------------------------- helpers
    def _st(self, msg):
        self.status.config(text=msg); self.update_idletasks()

    def _log(self, s=""):
        self.text.insert("end", s + "\n"); self.text.see("end"); self.update_idletasks()

    def _extract(self, site, out):
        """Align, segment, compute object features for one location."""
        nm = site["name"]
        self._st(f"[{nm}] reading & aligning…")
        ndfi, vn, prof = read_raster(site["ndfi"])
        sd, vs = align_to_reference(site["sd"], prof)
        fl, vf = align_to_reference(site["flood"], prof)
        (ndfi, sd, fl, valid), prof = crop_to(vn & vs & vf, [ndfi, sd, fl, vn & vs & vf], prof)
        nn, _ = normalise(ndfi, valid)          # per-scene normalisation (segmentation only)
        sn, _ = normalise(sd, valid)
        labels = segment_objects(nn, sn, valid, self.obj_size.get(), self.compact.get(),
                                 lambda m: self._st(f"[{nm}] {m}"))
        del nn, sn
        X = object_features(labels, ndfi, sd, valid, lambda m: self._st(f"[{nm}] {m}"))
        frac = object_flood_fraction(labels, fl)
        area = X[:, FEATURES.index("Area_px")].astype("float64")
        lab_path = os.path.join(out, f"{nm}_objects.tif")
        write_tif(lab_path, labels, prof, "int32", 0)
        return dict(name=nm, X=X, frac=frac, y=(frac >= 0.5).astype(int), area=area,
                    flood_px=frac * area, lab_path=lab_path, n_px=int(valid.sum()))

    def _fit(self, name, X, y, rng):
        model = make_model(name)
        if name.startswith("SVM") and len(y) > SVM_MAX_TRAIN:
            sel = rng.choice(len(y), SVM_MAX_TRAIN, replace=False)
            X, y = X[sel], y[sel]
        model.fit(X, y)
        return model

    def _train_mask(self, d):
        if self.pure_only.get():
            return (d["frac"] <= 0.2) | (d["frac"] >= 0.8)
        return np.ones(len(d["y"]), bool)

    # -------------------------------------------------------------------- run
    def run(self):
        try:
            if len(self.sites) < 2:
                raise ValueError("Add at least 2 locations.")
            out = self.outdir.get()
            if not out:
                raise ValueError("Choose an output folder.")
            os.makedirs(out, exist_ok=True)
            self.text.delete("1.0", "end")
            rng = np.random.default_rng(0)

            # 1. features for every site
            data = []
            for s in self.sites:
                d = self._extract(s, out)
                data.append(d)
                self._log(f"{d['name']:12s} objects={len(d['y']):8d}  flood objects={int(d['y'].sum()):7d} "
                          f"({100 * d['flood_px'].sum() / d['area'].sum():.1f}% flood pixels)")
            joblib.dump(data, os.path.join(out, "all_sites_object_features.joblib"))

            models = MODELS if self.compare_all.get() else [self.model_name.get()]
            fold_rows = []
            summary = {}
            best_probs = {}

            # 2. leave-one-location-out
            for mname in models:
                self._log(f"\n=== LOLO: {mname} ===")
                self._log(f"{'Test site':12s}{'F1':>7}{'IoU':>7}{'Kappa':>7}{'Prec':>7}{'Rec':>7}{'OA':>7}  (pixel level)")
                probs = {}
                agg = np.zeros(4)
                for i, test in enumerate(data):
                    self._st(f"LOLO {mname}: testing on {test['name']} ({i + 1}/{len(data)})…")
                    trs = [d for j, d in enumerate(data) if j != i]
                    Xtr = np.vstack([d["X"][self._train_mask(d)] for d in trs])
                    ytr = np.concatenate([d["y"][self._train_mask(d)] for d in trs])
                    if len(np.unique(ytr)) < 2:
                        raise ValueError("Training locations contain only one class.")
                    model = self._fit(mname, Xtr, ytr, rng)
                    p = model.predict_proba(test["X"])[:, 1]
                    probs[test["name"]] = p
                    pred = p >= 0.5
                    mp = pixel_metrics(pred, test["area"], test["flood_px"])
                    mo = object_metrics(pred, test["y"])
                    agg += [mp["TP"], mp["FP"], mp["FN"], mp["TN"]]
                    self._log(f"{test['name']:12s}" + "".join(f"{mp[k]:7.3f}" for k in
                                                         ["F1", "IoU", "Kappa", "Precision", "Recall", "OA"]))
                    fold_rows.append(dict(Model=mname, Test_site=test["name"], Level="pixel", **mp))
                    fold_rows.append(dict(Model=mname, Test_site=test["name"], Level="object", **mo))
                f1s = [r["F1"] for r in fold_rows if r["Model"] == mname and r["Level"] == "pixel"]
                pooled = metrics_from_counts(*agg)
                summary[mname] = dict(mean_F1=float(np.mean(f1s)), std_F1=float(np.std(f1s)),
                                      pooled=pooled, f1s=f1s)
                self._log(f"{'MEAN ± SD':12s}{np.mean(f1s):7.3f} ± {np.std(f1s):.3f}   "
                          f"pooled: F1={pooled['F1']:.3f} IoU={pooled['IoU']:.3f} Kappa={pooled['Kappa']:.3f}")
                best_probs[mname] = probs

            best = max(summary, key=lambda m: summary[m]["mean_F1"])
            self._log(f"\nBest model (mean LOLO F1): {best}")

            # 3. LOLO flood maps (each site mapped by the model that never saw it)
            import rasterio
            for d in data:
                self._st(f"Writing LOLO map for {d['name']}…")
                with rasterio.open(d["lab_path"]) as src:
                    labels = src.read(1); prof = src.profile.copy()
                p = best_probs[best][d["name"]]
                lut = np.full(len(p) + 1, -1, "float32"); lut[1:] = p
                prob = lut[labels]
                write_tif(os.path.join(out, f"{d['name']}_floodprob_LOLO_{best.split()[0]}.tif"),
                          prob, prof, "float32", -1)
                write_tif(os.path.join(out, f"{d['name']}_flood_LOLO_{best.split()[0]}.tif"),
                          np.where(prob < 0, 255, prob >= 0.5).astype("uint8"), prof, "uint8", 255)
                del labels, prob

            # 4. final model on all sites
            self._st(f"Training final {best} on all locations…")
            Xall = np.vstack([d["X"][self._train_mask(d)] for d in data])
            yall = np.concatenate([d["y"][self._train_mask(d)] for d in data])
            final = self._fit(best, Xall, yall, rng)
            mpath = os.path.join(out, f"flood_model_ALLSITES_{best.split()[0]}.joblib")
            joblib.dump(dict(model=final, features=FEATURES, nlims=None, slims=None,
                             obj_size=self.obj_size.get(), compact=self.compact.get(),
                             model_name=f"{best} (all {len(data)} sites)"), mpath)

            # feature importance of final model (on a sample)
            self._st("Feature importance…")
            sel = rng.choice(len(yall), min(10000, len(yall)), replace=False)
            pi = permutation_importance(final, Xall[sel], yall[sel], scoring="f1",
                                        n_repeats=5, random_state=0, n_jobs=-1)
            imp = sorted(zip(FEATURES, pi.importances_mean, pi.importances_std), key=lambda t: -t[1])
            self._log("\nFeature importance, final model (drop in F1 when shuffled):")
            for f, m, s in imp:
                self._log(f"  {f:20s} {m:8.4f} ± {s:.4f}")

            # 5. CSV report
            with open(os.path.join(out, "LOLO_results.csv"), "w", newline="") as fh:
                keys = ["Model", "Test_site", "Level"] + METRIC_KEYS + ["TP", "FP", "FN", "TN"]
                w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
                w.writeheader(); w.writerows(fold_rows)
                fh.write("\nModel,Mean_F1,SD_F1,Pooled_F1,Pooled_IoU,Pooled_Kappa\n")
                for m, s in summary.items():
                    fh.write(f"{m},{s['mean_F1']:.4f},{s['std_F1']:.4f},{s['pooled']['F1']:.4f},"
                             f"{s['pooled']['IoU']:.4f},{s['pooled']['Kappa']:.4f}\n")
                fh.write("\nFeature,Importance,Std\n")
                for f, m, s in imp:
                    fh.write(f"{f},{m:.5f},{s:.5f}\n")

            self._log(f"\nFinal model saved: {mpath}")
            self._log("Apply it to new scenes with flood_obia_ml_gui.py → 'Apply saved model'.")
            self._log(f"Outputs: LOLO flood maps per site, LOLO_results.csv, object maps → {out}")
            self._plot(data, summary, imp)
            self._st("Done.")
        except Exception as e:
            self._st("Error.")
            messagebox.showerror("Error", str(e))
            raise

    def _plot(self, data, summary, imp):
        self.fig.clear()
        ax1 = self.fig.add_subplot(1, 2, 1)
        names = [d["name"] for d in data]
        models = list(summary)
        wbar = 0.8 / len(models)
        for k, m in enumerate(models):
            x = np.arange(len(names)) + k * wbar - 0.4 + wbar / 2
            ax1.bar(x, summary[m]["f1s"], width=wbar, label=m)
        ax1.set_xticks(range(len(names))); ax1.set_xticklabels(names, rotation=30, ha="right")
        ax1.set_ylim(0, 1); ax1.set_ylabel("F1 (pixel level)")
        ax1.set_title("Leave-one-location-out: F1 per test location")
        ax1.legend(fontsize=7)
        ax2 = self.fig.add_subplot(1, 2, 2)
        ax2.barh([t[0] for t in imp][::-1], [t[1] for t in imp][::-1], color="#4a7fb5")
        ax2.set_title("Feature importance (final model)")
        ax2.tick_params(axis="y", labelsize=7)
        self.fig.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    MultiSiteApp().mainloop()
