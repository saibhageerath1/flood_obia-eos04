"""
Pixel-Based Flood Classification – Leave-One-Location-Out (baseline for OBIA)
==============================================================================

Same locations, same validation, same models and same metrics as
flood_obia_multisite.py – but every PIXEL is classified on its own instead of
image objects. Use it to compare pixel-based vs object-based classification.

Per pixel features
  * NDFI, StdDev, NDFI x StdDev                      (pure pixel-based)
  * optional: 3x3 mean of NDFI and StdDev            ("pixel + 3x3" variant)

Sampling (keeps memory and time manageable on a CPU)
  * Training: random sample of N pixels per location (default 300,000),
    class proportions kept; models use balanced class weights (as in OBIA).
  * Testing:  random sample of up to M pixels per location (default 2,000,000).
    Accuracy from a 2-million-pixel random sample differs from the full-image
    value only in the third decimal place.

Cut-off: 0.5 (same as the main OBIA results) and, optionally, a tuned cut-off
chosen by inner leave-one-location-out on the training locations only.

Optional: full flood maps for each location (slow for very large scenes).
Optional: object-based LOLO_results.csv -> side-by-side comparison table + chart.

Keep this file in the same folder as flood_obia_ml_gui.py and flood_obia_multisite.py.
"""

import csv
import os
import time
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import joblib
import numpy as np
from scipy import ndimage
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from flood_obia_ml_gui import align_to_reference, crop_to, make_model, read_raster, write_tif
from flood_obia_multisite import metrics_from_counts

MODELS = ["Extra Trees", "Random Forest", "Gradient Boosting", "Logistic Regression", "SVM (RBF)"]
SVM_MAX = 5000
THR_GRID = np.round(np.arange(0.05, 0.951, 0.025), 3)
CHUNK = 1_000_000


def pixel_features(ndfi, sd, idx, ctx3, ndfi3=None, sd3=None):
    n = ndfi.ravel()[idx]; s = sd.ravel()[idx]
    cols = [n, s, n * s]
    if ctx3:
        cols += [ndfi3.ravel()[idx], sd3.ravel()[idx]]
    return np.column_stack(cols).astype("float32")


def mean3(a, valid):
    w = ndimage.uniform_filter(valid.astype("float32"), 3) + 1e-6
    return (ndimage.uniform_filter(np.where(valid, a, 0).astype("float32"), 3) / w).astype("float32")


def counts(pred, ref):
    return (float((pred & ref).sum()), float((pred & ~ref).sum()),
            float((~pred & ref).sum()), float((~pred & ~ref).sum()))


def best_threshold(preds):
    best, bt = -1, 0.5
    for t in THR_GRID:
        f = np.mean([metrics_from_counts(*counts(p >= t, y))["F1"] for p, y in preds])
        if f > best:
            best, bt = f, float(t)
    return bt


class PixelApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Pixel-Based Flood Classification – LOLO baseline")
        self.geometry("1450x940")
        self.sites = []
        self.outdir = tk.StringVar()
        self.obj_csv = tk.StringVar()
        self.models = {m: tk.BooleanVar(value=(m == "Extra Trees")) for m in MODELS}
        self.ctx3 = tk.BooleanVar(value=False)
        self.n_train = tk.IntVar(value=300000)
        self.n_test = tk.IntVar(value=2000000)
        self.tune = tk.BooleanVar(value=True)
        self.maps = tk.BooleanVar(value=False)
        self.rows = []
        self._build()

    # --------------------------------------------------------------- layout
    def _build(self):
        top = ttk.LabelFrame(self, text="Locations (same as the OBIA run)", padding=6)
        top.pack(fill="x", padx=8, pady=4)
        cols = ("Name", "NDFI", "StdDev", "Flood")
        self.tree = ttk.Treeview(top, columns=cols, show="headings", height=5)
        for c, w in zip(cols, (110, 390, 390, 390)):
            self.tree.heading(c, text=c); self.tree.column(c, width=w)
        self.tree.grid(row=0, column=0, rowspan=3, sticky="nsew")
        ttk.Button(top, text="Load list (CSV)", command=self.load_list).grid(row=0, column=1, padx=6, sticky="ew")
        ttk.Button(top, text="Add location", command=self.add_site).grid(row=1, column=1, padx=6, sticky="ew")
        ttk.Button(top, text="Remove selected", command=self.remove_site).grid(row=2, column=1, padx=6, sticky="ew")

        o = ttk.Frame(self, padding=(8, 2)); o.pack(fill="x")
        ttk.Label(o, text="Output folder:").grid(row=0, column=0, sticky="w")
        ttk.Entry(o, textvariable=self.outdir, width=90).grid(row=0, column=1, padx=4)
        ttk.Button(o, text="Browse", command=lambda: self.outdir.set(filedialog.askdirectory() or self.outdir.get())).grid(row=0, column=2)
        ttk.Label(o, text="Object-based LOLO_results.csv (optional, for comparison):").grid(row=1, column=0, sticky="w")
        ttk.Entry(o, textvariable=self.obj_csv, width=90).grid(row=1, column=1, padx=4)
        ttk.Button(o, text="Browse", command=lambda: self.obj_csv.set(
            filedialog.askopenfilename(filetypes=[("CSV", "*.csv")]) or self.obj_csv.get())).grid(row=1, column=2)

        m = ttk.Frame(self, padding=(8, 2)); m.pack(fill="x")
        ttk.Label(m, text="Models:").pack(side="left")
        for k, v in self.models.items():
            ttk.Checkbutton(m, text=k, variable=v).pack(side="left", padx=4)
        r = ttk.Frame(self, padding=(8, 2)); r.pack(fill="x")
        ttk.Checkbutton(r, text="Add 3×3 neighbourhood mean (pixel + 3×3 variant)", variable=self.ctx3).pack(side="left")
        ttk.Label(r, text="   Training pixels / location:").pack(side="left")
        ttk.Spinbox(r, from_=20000, to=2000000, increment=50000, textvariable=self.n_train, width=9).pack(side="left", padx=3)
        ttk.Label(r, text="Test pixels / location:").pack(side="left")
        ttk.Spinbox(r, from_=100000, to=20000000, increment=500000, textvariable=self.n_test, width=10).pack(side="left", padx=3)
        ttk.Checkbutton(r, text="Tune cut-off", variable=self.tune).pack(side="left", padx=8)
        ttk.Checkbutton(r, text="Write full flood maps (slow for big scenes)", variable=self.maps).pack(side="left", padx=8)
        ttk.Button(r, text="Run", command=self.run).pack(side="left", padx=10)

        self.status = ttk.Label(self, text="Load the same location list used for the OBIA run.", foreground="gray")
        self.status.pack(fill="x", padx=10)
        body = ttk.PanedWindow(self, orient="horizontal"); body.pack(fill="both", expand=True, padx=8, pady=4)
        ff = ttk.Frame(body)
        self.fig = Figure(figsize=(8.5, 6.5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=ff)
        NavigationToolbar2Tk(self.canvas, ff)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        body.add(ff, weight=3)
        tf = ttk.Frame(body)
        self.text = tk.Text(tf, wrap="none", font=("Courier", 9))
        sb = ttk.Scrollbar(tf, command=self.text.yview); self.text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y"); self.text.pack(fill="both", expand=True)
        body.add(tf, weight=2)

    def _refresh(self):
        self.tree.delete(*self.tree.get_children())
        for s in self.sites:
            self.tree.insert("", "end", values=(s["name"], s["ndfi"], s["sd"], s["flood"]))

    def load_list(self):
        p = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if p:
            with open(p, newline="") as fh:
                self.sites = [dict(r) for r in csv.DictReader(fh)]
            if self.sites and not self.outdir.get():
                self.outdir.set(os.path.join(os.path.dirname(self.sites[0]["ndfi"]), "pixel_lolo_output"))
            self._refresh()

    def add_site(self):
        ft = [("GeoTIFF", "*.tif *.tiff")]
        a = filedialog.askopenfilename(title="1/3 NDFI", filetypes=ft)
        if not a: return
        b = filedialog.askopenfilename(title="2/3 StdDev", filetypes=ft, initialdir=os.path.dirname(a))
        if not b: return
        c = filedialog.askopenfilename(title="3/3 Flood reference", filetypes=ft, initialdir=os.path.dirname(a))
        if not c: return
        n = simpledialog.askstring("Name", "Location name:", initialvalue=f"Site{len(self.sites) + 1}")
        if n:
            self.sites.append(dict(name=n, ndfi=a, sd=b, flood=c)); self._refresh()

    def remove_site(self):
        sel = [self.tree.index(i) for i in self.tree.selection()]
        self.sites = [s for i, s in enumerate(self.sites) if i not in sel]; self._refresh()

    def _st(self, m):
        self.status.config(text=m); self.update_idletasks()

    def _log(self, s=""):
        self.text.insert("end", s + "\n"); self.text.see("end"); self.update_idletasks()

    # ---------------------------------------------------------------- data
    def _load_site(self, s, rng):
        self._st(f"[{s['name']}] reading & aligning…")
        ndfi, vn, prof = read_raster(s["ndfi"])
        sd, vs = align_to_reference(s["sd"], prof)
        fl, vf = align_to_reference(s["flood"], prof)
        (ndfi, sd, fl, valid), prof = crop_to(vn & vs & vf, [ndfi, sd, fl, vn & vs & vf], prof)
        ctx = self.ctx3.get()
        n3 = s3 = None
        if ctx:
            self._st(f"[{s['name']}] 3×3 means…")
            n3, s3 = mean3(ndfi, valid), mean3(sd, valid)
        vidx = np.flatnonzero(valid.ravel())
        y_all = fl.ravel()[vidx] > 0
        # stratified training sample (class proportions kept)
        ntr = min(self.n_train.get(), len(vidx))
        fl_i, nf_i = vidx[y_all], vidx[~y_all]
        k = int(round(ntr * len(fl_i) / len(vidx)))
        tr = np.concatenate([rng.choice(fl_i, min(k, len(fl_i)), replace=False),
                             rng.choice(nf_i, min(ntr - k, len(nf_i)), replace=False)])
        te = vidx if len(vidx) <= self.n_test.get() else rng.choice(vidx, self.n_test.get(), replace=False)
        d = dict(name=s["name"], n_valid=len(vidx), flood_pct=100 * y_all.mean(),
                 Xtr=pixel_features(ndfi, sd, tr, ctx, n3, s3), ytr=(fl.ravel()[tr] > 0).astype(int),
                 Xte=pixel_features(ndfi, sd, te, ctx, n3, s3), yte=fl.ravel()[te] > 0)
        if self.maps.get():
            d["full"] = (ndfi, sd, n3, s3, valid, prof)
        return d

    def _fit(self, name, X, y, rng):
        m = make_model(name)
        if name.startswith("SVM") and len(y) > SVM_MAX:
            sel = rng.choice(len(y), SVM_MAX, replace=False); X, y = X[sel], y[sel]
        m.fit(X, y)
        return m

    def _predict(self, model, X):
        out = np.empty(len(X), "float32")
        for i in range(0, len(X), CHUNK):
            out[i:i + CHUNK] = model.predict_proba(X[i:i + CHUNK])[:, 1]
        return out

    # ----------------------------------------------------------------- run
    def run(self):
        try:
            if len(self.sites) < 3 and self.tune.get():
                raise ValueError("Need at least 3 locations (2 when cut-off tuning is off).")
            if len(self.sites) < 2:
                raise ValueError("Need at least 2 locations.")
            out = self.outdir.get()
            if not out:
                raise ValueError("Choose an output folder.")
            os.makedirs(out, exist_ok=True)
            models = [m for m, v in self.models.items() if v.get()]
            if not models:
                raise ValueError("Tick at least one model.")
            self.text.delete("1.0", "end")
            rng = np.random.default_rng(0)
            t0 = time.time()
            data = [self._load_site(s, rng) for s in self.sites]
            variant = "pixel + 3x3" if self.ctx3.get() else "pixel"
            for d in data:
                self._log(f"{d['name']:10s} valid px={d['n_valid']:>12,d}  flood={d['flood_pct']:.1f}%  "
                          f"train={len(d['ytr']):,}  test={len(d['yte']):,}")
            self.rows = []
            n = len(data)
            for mname in models:
                self._log(f"\n=== {variant} | {mname} ===")
                self._log(f"{'Test site':10s}{'F1@0.5':>8}{'Kappa':>8}{'OA':>8}{'Prec':>8}{'Rec':>8}"
                          + (f"{'cut':>7}{'F1@tuned':>10}" if self.tune.get() else ""))
                for i, d in enumerate(data):
                    others = [j for j in range(n) if j != i]
                    self._st(f"{mname}: test={d['name']} – training…")
                    model = self._fit(mname, np.vstack([data[j]["Xtr"] for j in others]),
                                      np.concatenate([data[j]["ytr"] for j in others]), rng)
                    self._st(f"{mname}: test={d['name']} – predicting {len(d['yte']):,} pixels…")
                    p = self._predict(model, d["Xte"])
                    m05 = metrics_from_counts(*counts(p >= 0.5, d["yte"]))
                    row = dict(Approach=variant, Model=mname, Test_site=d["name"], Cutoff_type="0.5", Cutoff=0.5, **m05)
                    self.rows.append(row)
                    line = f"{d['name']:10s}" + "".join(f"{m05[k]:8.3f}" for k in ["F1", "Kappa", "OA", "Precision", "Recall"])
                    t = 0.5
                    if self.tune.get():
                        inner = []
                        for j in others:
                            self._st(f"{mname}: test={d['name']} – tuning cut-off (inner {data[j]['name']})…")
                            mm = self._fit(mname, np.vstack([data[k]["Xtr"] for k in others if k != j]),
                                           np.concatenate([data[k]["ytr"] for k in others if k != j]), rng)
                            inner.append((self._predict(mm, data[j]["Xtr"]), data[j]["ytr"].astype(bool)))
                        t = best_threshold(inner)
                        mt = metrics_from_counts(*counts(p >= t, d["yte"]))
                        self.rows.append(dict(Approach=variant, Model=mname, Test_site=d["name"], Cutoff_type="tuned", Cutoff=t, **mt))
                        line += f"{t:7.3f}{mt['F1']:10.3f}"
                    self._log(line)
                    if self.maps.get():
                        self._write_map(model, d, mname, t, out)
                for ct in (["0.5", "tuned"] if self.tune.get() else ["0.5"]):
                    f1 = [r["F1"] for r in self.rows if r["Model"] == mname and r["Cutoff_type"] == ct]
                    self._log(f"{'MEAN ' + ct:10s}{np.mean(f1):8.3f} ± {np.std(f1):.3f}")
            self._export(out, variant)
            self._compare(out, variant)
            self._st(f"Done in {(time.time() - t0) / 60:.1f} min. Results in {out}")
        except Exception as e:
            self._st("Error.")
            messagebox.showerror("Error", str(e))
            raise

    def _write_map(self, model, d, mname, t, out):
        ndfi, sd, n3, s3, valid, prof = d["full"]
        vidx = np.flatnonzero(valid.ravel())
        prob = np.full(valid.size, -1, "float32")
        for i in range(0, len(vidx), CHUNK):
            self._st(f"Map {d['name']}: {100 * i / len(vidx):.0f}%")
            ii = vidx[i:i + CHUNK]
            prob[ii] = model.predict_proba(pixel_features(ndfi, sd, ii, self.ctx3.get(), n3, s3))[:, 1]
        prob = prob.reshape(valid.shape)
        tag = f"{'pix3x3' if self.ctx3.get() else 'pix'}_{mname.split()[0]}"
        write_tif(os.path.join(out, f"{d['name']}_floodprob_LOLO_{tag}.tif"), prob, prof, "float32", -1)
        write_tif(os.path.join(out, f"{d['name']}_flood_LOLO_{tag}.tif"),
                  np.where(prob < 0, 255, prob >= t).astype("uint8"), prof, "uint8", 255)

    def _export(self, out, variant):
        keys = ["Approach", "Model", "Test_site", "Cutoff_type", "Cutoff", "OA", "Precision", "Recall", "F1", "IoU", "Kappa",
                "TP", "FP", "FN", "TN"]
        name = "pixel3x3_LOLO_results.csv" if self.ctx3.get() else "pixel_LOLO_results.csv"
        with open(os.path.join(out, name), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(self.rows)
        self._log(f"\nSaved {name}")

    # -------------------------------------------------------- comparison
    def _compare(self, out, variant):
        sites = [s["name"] for s in self.sites]
        obj = {}
        if self.obj_csv.get() and os.path.exists(self.obj_csv.get()):
            with open(self.obj_csv.get(), newline="") as fh:
                for r in csv.reader(fh):
                    if len(r) >= 9 and r[2] == "pixel" and r[0] in MODELS:
                        obj[(r[0], r[1])] = dict(OA=float(r[3]), Precision=float(r[4]), Recall=float(r[5]),
                                                 F1=float(r[6]), IoU=float(r[7]), Kappa=float(r[8]))
        models = sorted({r["Model"] for r in self.rows}, key=MODELS.index)
        lines = [["Model", "Approach", "Cut-off"] + [f"F1_{s}" for s in sites] +
                 ["Mean_F1", "SD_F1", "Mean_Kappa", "Mean_OA", "Mean_Precision", "Mean_Recall"]]

        def summ(get):
            vals = [get(s) for s in sites]
            if any(v is None for v in vals):
                return None
            return vals

        series = []
        for m in models:
            for lab, get in [("Object-based", lambda s: obj.get((m, s))),
                             (f"{variant} (0.5)", lambda s: next((r for r in self.rows if r["Model"] == m and r["Test_site"] == s and r["Cutoff_type"] == "0.5"), None)),
                             (f"{variant} (tuned)", lambda s: next((r for r in self.rows if r["Model"] == m and r["Test_site"] == s and r["Cutoff_type"] == "tuned"), None))]:
                v = summ(get)
                if v is None:
                    continue
                f1 = [x["F1"] for x in v]
                lines.append([m, lab.split(" (")[0], "0.5" if "tuned" not in lab else "tuned"] + [f"{x:.4f}" for x in f1] +
                             [f"{np.mean(f1):.4f}", f"{np.std(f1):.4f}"] +
                             [f"{np.mean([x[k] for x in v]):.4f}" for k in ["Kappa", "OA", "Precision", "Recall"]])
                series.append((f"{m} – {lab}", f1))
        with open(os.path.join(out, "object_vs_pixel_comparison.csv"), "w", newline="") as fh:
            csv.writer(fh).writerows(lines)
        self._log("Saved object_vs_pixel_comparison.csv")
        if obj:
            self._log("\n=== Object-based vs pixel-based (mean F1 across locations, cut-off 0.5) ===")
            for m in models:
                o = summ(lambda s: obj.get((m, s)))
                p = summ(lambda s: next((r for r in self.rows if r["Model"] == m and r["Test_site"] == s and r["Cutoff_type"] == "0.5"), None))
                if o and p:
                    self._log(f"{m:20s} object={np.mean([x['F1'] for x in o]):.3f}   {variant}={np.mean([x['F1'] for x in p]):.3f}")
        # chart
        self.fig.clear()
        ax = self.fig.add_subplot(1, 1, 1)
        k = len(series); w = 0.8 / max(k, 1)
        for i, (lab, f1) in enumerate(series):
            ax.bar(np.arange(len(sites)) + (i - (k - 1) / 2) * w, f1, w * 0.95, label=lab)
        ax.set_xticks(range(len(sites))); ax.set_xticklabels(sites)
        ax.set_ylim(0, 1); ax.set_ylabel("F1 (pixel level)")
        ax.set_title("Object-based vs pixel-based: F1 per held-out location")
        ax.legend(fontsize=7)
        self.fig.tight_layout(); self.canvas.draw()
        self.fig.savefig(os.path.join(out, "object_vs_pixel_F1.png"), dpi=200)


if __name__ == "__main__":
    PixelApp().mainloop()
