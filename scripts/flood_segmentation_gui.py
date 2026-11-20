"""
Flood Multi-Class Segmentation GUI (NDFI / Standard Deviation)
--------------------------------------------------------------
- Loads an index raster (NDFI or Std-Dev, .tif) and a binary flood reference raster (.tif)
- The flood layer is automatically reprojected / resampled / clipped to the index grid,
  and both are trimmed to their common (overlapping) extent
- Automated multi-class segmentation of the index into N classes (default 10):
      * K-means (1-D)      - natural groupings of values (default)
      * Multi-Otsu         - automatic multi-level Otsu (max 5 classes, slow above that)
      * Quantile           - equal number of pixels per class
      * Equal interval     - equal value ranges (2nd-98th percentile)
  Classes are numbered 1..N in increasing index value (N = highest NDFI / StdDev).
- For every class: pixel count, mean, std, min, median, max of the index,
  share of the class that is flood, share of total flood it captures,
  mean / std inside flood and non-flood reference areas,
  and a similarity score to the flood vs non-flood distributions
  (Bhattacharyya distance, using each class's mean and std).
- Flags flood-related classes and evaluates "classes k..N = flood" cut-offs
  (precision, recall, F1, IoU, Kappa) to suggest the best class boundary.
- Saves the class map as GeoTIFF and exports statistics to CSV.

Requirements (conda):
    conda create -n flood -c conda-forge python=3.11 numpy rasterio scikit-image matplotlib
"""

import csv
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure
import matplotlib

NODATA = 255
FLOOD_CLASS_FRACTION = 0.5     # class is "flood-related" if >= 50 % of its pixels are flood


# ================================================================ raster I/O
def read_raster(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float64")
        profile = src.profile.copy()
        nodata = src.nodata
    valid = np.isfinite(arr)
    if nodata is not None:
        valid &= arr != nodata
    return arr, valid, profile


def align_to_reference(path, ref_profile):
    """Reproject/resample/clip a raster onto the reference grid (nearest neighbour)."""
    h, w = ref_profile["height"], ref_profile["width"]
    dst = np.full((h, w), np.nan, dtype="float32")
    with rasterio.open(path) as src:
        reproject(source=rasterio.band(src, 1), destination=dst,
                  src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
                  dst_transform=ref_profile["transform"],
                  dst_crs=ref_profile["crs"] or src.crs, dst_nodata=np.nan,
                  resampling=Resampling.nearest)
        nodata = src.nodata
    valid = np.isfinite(dst)
    if nodata is not None:
        valid &= dst != nodata
    return dst.astype("float64"), valid


def crop_to_common(idx, v_idx, flood, v_fl, profile):
    """Trim all arrays to the bounding box where both layers have data."""
    both = v_idx & v_fl
    if not both.any():
        raise ValueError("The index and flood layers do not overlap.")
    rows = np.where(both.any(axis=1))[0]
    cols = np.where(both.any(axis=0))[0]
    r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    win = Window(c0, r0, c1 - c0, r1 - r0)
    prof = profile.copy()
    prof.update(height=r1 - r0, width=c1 - c0,
                transform=rasterio.windows.transform(win, profile["transform"]))
    sl = (slice(r0, r1), slice(c0, c1))
    return idx[sl], v_idx[sl], flood[sl], v_fl[sl], prof


# ============================================================ segmentation
def kmeans_1d_thresholds(vals, k, max_iter=100, sample=2_000_000, seed=0):
    rng = np.random.default_rng(seed)
    x = vals if vals.size <= sample else rng.choice(vals, sample, replace=False)
    c = np.quantile(x, (np.arange(k) + 0.5) / k)          # robust initialisation
    for _ in range(max_iter):
        edges = (c[:-1] + c[1:]) / 2
        lab = np.digitize(x, edges)
        new = np.array([x[lab == i].mean() if np.any(lab == i) else c[i] for i in range(k)])
        if np.allclose(new, c):
            break
        c = np.sort(new)
    return (c[:-1] + c[1:]) / 2


def multiotsu_thresholds(vals, k, sample=2_000_000):
    from skimage.filters import threshold_multiotsu
    if k > 5:
        raise ValueError("Multi-Otsu is limited to 5 classes here (it becomes extremely slow). "
                         "Use K-means for more classes.")
    rng = np.random.default_rng(0)
    x = vals if vals.size <= sample else rng.choice(vals, sample, replace=False)
    return threshold_multiotsu(x, classes=k)


def quantile_thresholds(vals, k):
    return np.quantile(vals, np.arange(1, k) / k)


def equal_interval_thresholds(vals, k):
    lo, hi = np.percentile(vals, [2, 98])
    return np.linspace(lo, hi, k + 1)[1:-1]


METHODS = {
    "K-means": kmeans_1d_thresholds,
    "Multi-Otsu": multiotsu_thresholds,
    "Quantile": quantile_thresholds,
    "Equal interval": equal_interval_thresholds,
}


def segment_multiclass(index_arr, valid, method, k):
    vals = index_arr[valid]
    if vals.size == 0:
        raise ValueError("Index image has no valid pixels.")
    thr = np.unique(np.asarray(METHODS[method](vals, k), dtype="float64"))
    seg = np.full(index_arr.shape, NODATA, dtype=np.uint8)
    seg[valid] = (np.digitize(index_arr[valid], thr) + 1).astype(np.uint8)   # 1..N
    return seg, thr


# ============================================================== statistics
def bhattacharyya(m1, s1, m2, s2):
    """Bhattacharyya distance between two normal distributions (0 = identical)."""
    s1, s2 = max(s1, 1e-12), max(s2, 1e-12)
    v = (s1 ** 2 + s2 ** 2) / 2
    return 0.125 * (m1 - m2) ** 2 / v + 0.5 * np.log(v / (s1 * s2))


def _ms(v):
    return (float(v.mean()), float(v.std())) if v.size else (np.nan, np.nan)


def kappa_metrics(pred, ref):
    tp = int((pred & ref).sum()); fp = int((pred & ~ref).sum())
    fn = int((~pred & ref).sum()); tn = int((~pred & ~ref).sum())
    n = tp + fp + fn + tn
    oa = (tp + tn) / n if n else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * prec * rec / (prec + rec) if (tp and prec + rec) else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else np.nan
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2 if n else np.nan
    kap = (oa - pe) / (1 - pe) if n and pe != 1 else np.nan
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, OA=oa, Precision=prec,
                Recall=rec, F1=f1, IoU=iou, Kappa=kap)


def compute_statistics(index_arr, seg, flood, mask, n_classes):
    s = seg[mask]
    f = flood[mask] > 0
    x = index_arr[mask]
    total_flood = int(f.sum())

    fm, fs = _ms(x[f])            # reference flood distribution
    nm, ns = _ms(x[~f])           # reference non-flood distribution

    rows = []
    for c in range(1, n_classes + 1):
        sel = s == c
        n = int(sel.sum())
        if n == 0:
            continue
        v = x[sel]
        nf = int((sel & f).sum())
        cm, cs = float(v.mean()), float(v.std())
        mf, sf = _ms(x[sel & f])
        mn, sn = _ms(x[sel & ~f])
        bf = bhattacharyya(cm, cs, fm, fs) if total_flood else np.nan
        bn = bhattacharyya(cm, cs, nm, ns) if (~f).any() else np.nan
        frac = nf / n
        rows.append(dict(
            Class=c, Pixels=n, Pct_of_image=100 * n / s.size,
            Mean=cm, Std=cs, Min=float(v.min()), Median=float(np.median(v)), Max=float(v.max()),
            Flood_pixels=nf, NoFlood_pixels=n - nf,
            Flood_pct_of_class=100 * frac,
            Pct_of_total_flood=100 * nf / total_flood if total_flood else np.nan,
            Mean_in_flood=mf, Std_in_flood=sf, Mean_in_noflood=mn, Std_in_noflood=sn,
            Bhatt_to_flood=bf, Bhatt_to_noflood=bn,
            Closer_to="Flood" if bf < bn else "No Flood",
            Flood_related="YES" if frac >= FLOOD_CLASS_FRACTION else "no",
        ))

    # cumulative cut-offs: classes >= k labelled flood
    cuts = []
    for k in range(2, n_classes + 1):
        m = kappa_metrics(s >= k, f)
        cuts.append(dict(Flood_if_class_ge=k, **m))
    best = max(cuts, key=lambda d: (d["F1"] if np.isfinite(d["F1"]) else -1)) if cuts else None

    ref = dict(Flood_mean=fm, Flood_std=fs, NoFlood_mean=nm, NoFlood_std=ns,
               Flood_pixels=total_flood, Total_pixels=int(s.size))
    return rows, cuts, best, ref


# ======================================================================= GUI
class FloodApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Flood Multi-Class Segmentation")
        self.geometry("1450x900")
        self.index_path = tk.StringVar()
        self.flood_path = tk.StringVar()
        self.out_path = tk.StringVar()
        self.method = tk.StringVar(value="K-means")
        self.layer_type = tk.StringVar(value="NDFI")
        self.n_classes = tk.IntVar(value=10)
        self.result = None
        self._build()

    # ------------------------------------------------------------- layout
    def _build(self):
        top = ttk.LabelFrame(self, text="Inputs / Output", padding=8)
        top.pack(fill="x", padx=8, pady=6)

        def row(r, label, var, cmd):
            ttk.Label(top, text=label).grid(row=r, column=0, sticky="w")
            ttk.Entry(top, textvariable=var, width=100).grid(row=r, column=1, padx=4, pady=2)
            ttk.Button(top, text="Browse", command=cmd).grid(row=r, column=2)

        row(0, "Index layer (NDFI / StdDev):", self.index_path, self._pick_index)
        row(1, "Flood reference (binary):", self.flood_path,
            lambda: self._pick(self.flood_path))
        row(2, "Class map output (.tif):", self.out_path, self._pick_out)

        opt = ttk.Frame(top)
        opt.grid(row=3, column=0, columnspan=3, sticky="w", pady=4)
        ttk.Label(opt, text="Layer type:").pack(side="left")
        ttk.Combobox(opt, textvariable=self.layer_type, width=18, state="readonly",
                     values=["NDFI", "Standard Deviation"]).pack(side="left", padx=4)
        ttk.Label(opt, text="Method:").pack(side="left", padx=(15, 0))
        ttk.Combobox(opt, textvariable=self.method, width=14, state="readonly",
                     values=list(METHODS)).pack(side="left", padx=4)
        ttk.Label(opt, text="No. of classes:").pack(side="left", padx=(15, 0))
        ttk.Spinbox(opt, from_=2, to=30, textvariable=self.n_classes,
                    width=5).pack(side="left", padx=4)
        ttk.Button(opt, text="Run", command=self.run).pack(side="left", padx=15)
        ttk.Button(opt, text="Export statistics CSV",
                   command=self.export_csv).pack(side="left")
        self.status = ttk.Label(opt, text="", foreground="gray")
        self.status.pack(side="left", padx=15)

        body = ttk.PanedWindow(self, orient="horizontal")
        body.pack(fill="both", expand=True, padx=8, pady=4)

        fig_frame = ttk.Frame(body)
        self.fig = Figure(figsize=(9, 7.5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=fig_frame)
        NavigationToolbar2Tk(self.canvas, fig_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        body.add(fig_frame, weight=3)

        right = ttk.Notebook(body)
        # per-class table
        tab1 = ttk.Frame(right)
        cols = ("Class", "Pixels", "Mean", "Std", "Flood %", "% of flood",
                "B→Flood", "B→NoFlood", "Closer", "Flood?")
        self.tree = ttk.Treeview(tab1, columns=cols, show="headings", height=20)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=78, anchor="center")
        self.tree.tag_configure("flood", background="#cfe5ff")
        ysb = ttk.Scrollbar(tab1, command=self.tree.yview)
        xsb = ttk.Scrollbar(tab1, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)
        ysb.pack(side="right", fill="y"); xsb.pack(side="bottom", fill="x")
        self.tree.pack(fill="both", expand=True)
        right.add(tab1, text="Class table")
        # text report
        tab2 = ttk.Frame(right)
        self.text = tk.Text(tab2, wrap="none", font=("Courier", 9))
        sb = ttk.Scrollbar(tab2, command=self.text.yview)
        self.text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.text.pack(fill="both", expand=True)
        right.add(tab2, text="Full report")
        body.add(right, weight=2)

    # ------------------------------------------------------------ pickers
    def _pick(self, var):
        p = filedialog.askopenfilename(filetypes=[("GeoTIFF", "*.tif *.tiff"), ("All", "*.*")])
        if p:
            var.set(p)

    def _pick_index(self):
        self._pick(self.index_path)
        p = self.index_path.get()
        if p and not self.out_path.get():
            self.out_path.set(os.path.splitext(p)[0] + "_classes.tif")

    def _pick_out(self):
        p = filedialog.asksaveasfilename(defaultextension=".tif",
                                         filetypes=[("GeoTIFF", "*.tif")])
        if p:
            self.out_path.set(p)

    def _set_status(self, msg):
        self.status.config(text=msg)
        self.update_idletasks()

    # --------------------------------------------------------------- run
    def run(self):
        try:
            if not (self.index_path.get() and self.flood_path.get() and self.out_path.get()):
                raise ValueError("Please select index layer, flood layer and output path.")
            k = int(self.n_classes.get())
            if k < 2:
                raise ValueError("Number of classes must be at least 2.")
            method = self.method.get()

            self._set_status("Reading index layer…")
            idx, v_idx, profile = read_raster(self.index_path.get())
            self._set_status("Aligning / clipping flood layer…")
            flood, v_fl = align_to_reference(self.flood_path.get(), profile)
            idx, v_idx, flood, v_fl, profile = crop_to_common(idx, v_idx, flood, v_fl, profile)
            v_idx = v_idx & v_fl

            self._set_status(f"Segmenting into {k} classes ({method})…")
            seg, thr = segment_multiclass(idx, v_idx, method, k)
            n_cls = len(thr) + 1

            out = profile.copy()
            out.update(dtype="uint8", count=1, nodata=NODATA, compress="lzw")
            with rasterio.open(self.out_path.get(), "w", **out) as dst:
                dst.write(seg, 1)

            self._set_status("Computing statistics…")
            rows, cuts, best, ref = compute_statistics(idx, seg, flood, v_idx, n_cls)
            self.result = dict(rows=rows, cuts=cuts, best=best, ref=ref, thr=thr,
                               method=method, n=n_cls)
            self._fill_table()
            self._fill_report()
            self._plot(idx, v_idx, seg, flood, v_fl, n_cls)
            self._set_status("Done.")
        except Exception as e:
            self._set_status("Error.")
            messagebox.showerror("Error", str(e))

    # ------------------------------------------------------------ outputs
    def _fill_table(self):
        self.tree.delete(*self.tree.get_children())
        for r in self.result["rows"]:
            self.tree.insert("", "end", tags=("flood",) if r["Flood_related"] == "YES" else (),
                             values=(r["Class"], r["Pixels"], f"{r['Mean']:.4f}",
                                     f"{r['Std']:.4f}", f"{r['Flood_pct_of_class']:.1f}",
                                     f"{r['Pct_of_total_flood']:.1f}",
                                     f"{r['Bhatt_to_flood']:.3f}", f"{r['Bhatt_to_noflood']:.3f}",
                                     r["Closer_to"], r["Flood_related"]))

    def _fill_report(self):
        R = self.result
        t = self.text
        t.delete("1.0", "end")
        ref = R["ref"]
        t.insert("end",
                 f"Layer    : {self.layer_type.get()}\nMethod   : {R['method']}\n"
                 f"Classes  : {R['n']} (1 = lowest values, {R['n']} = highest)\n"
                 f"Output   : {self.out_path.get()}\n"
                 f"Thresholds: {', '.join(f'{v:.4f}' for v in R['thr'])}\n\n"
                 "=== Reference distributions ===\n"
                 f"Flood     : mean={ref['Flood_mean']:.4f}  std={ref['Flood_std']:.4f}  "
                 f"pixels={ref['Flood_pixels']}\n"
                 f"No Flood  : mean={ref['NoFlood_mean']:.4f}  std={ref['NoFlood_std']:.4f}  "
                 f"pixels={ref['Total_pixels'] - ref['Flood_pixels']}\n\n"
                 "=== Per-class statistics ===\n")
        for r in R["rows"]:
            t.insert("end",
                     f"\nClass {r['Class']}  [{'FLOOD-RELATED' if r['Flood_related'] == 'YES' else 'non-flood'}]"
                     f"  closer to: {r['Closer_to']}\n"
                     f"  pixels={r['Pixels']} ({r['Pct_of_image']:.2f}% of area)\n"
                     f"  mean={r['Mean']:.4f} std={r['Std']:.4f} "
                     f"min={r['Min']:.4f} median={r['Median']:.4f} max={r['Max']:.4f}\n"
                     f"  flood: {r['Flood_pct_of_class']:.1f}% of class, "
                     f"{r['Pct_of_total_flood']:.1f}% of all flood\n"
                     f"  in flood   : mean={r['Mean_in_flood']:.4f} std={r['Std_in_flood']:.4f}\n"
                     f"  in no-flood: mean={r['Mean_in_noflood']:.4f} std={r['Std_in_noflood']:.4f}\n"
                     f"  Bhattacharyya distance → flood={r['Bhatt_to_flood']:.3f}, "
                     f"no-flood={r['Bhatt_to_noflood']:.3f}\n")
        t.insert("end", "\n=== Cut-off evaluation: classes ≥ k = flood ===\n"
                        f"{'k':>3}{'Prec':>8}{'Recall':>8}{'F1':>8}{'IoU':>8}{'Kappa':>8}{'OA':>8}\n")
        for c in R["cuts"]:
            t.insert("end", f"{c['Flood_if_class_ge']:>3}{c['Precision']:>8.3f}{c['Recall']:>8.3f}"
                            f"{c['F1']:>8.3f}{c['IoU']:>8.3f}{c['Kappa']:>8.3f}{c['OA']:>8.3f}\n")
        b = R["best"]
        if b:
            k = b["Flood_if_class_ge"]
            cut_val = R["thr"][k - 2]
            t.insert("end", f"\nBest cut-off (max F1): classes {k}–{R['n']} = flood  "
                            f"(index > {cut_val:.4f})\n  F1={b['F1']:.3f}  IoU={b['IoU']:.3f}  "
                            f"Kappa={b['Kappa']:.3f}\n")
        t.insert("end", "\nNotes: 'Flood %' = share of the class lying in reference flood.\n"
                        "Bhattacharyya distance compares the class mean/std with the flood and\n"
                        "no-flood reference distributions (smaller = more similar).\n")

    def _plot(self, idx, v_idx, seg, flood, v_fl, n):
        R = self.result
        self.fig.clear()
        ax1, ax2, ax3, ax4 = [self.fig.add_subplot(2, 2, i) for i in range(1, 5)]

        disp = np.where(v_idx, idx, np.nan)
        lo, hi = np.nanpercentile(disp, [2, 98])
        im = ax1.imshow(disp, cmap="viridis", vmin=lo, vmax=hi)
        ax1.set_title(self.layer_type.get())
        self.fig.colorbar(im, ax=ax1, fraction=0.046)

        cmap = ListedColormap(matplotlib.colormaps["Spectral_r"](np.linspace(0, 1, n)))
        norm = BoundaryNorm(np.arange(0.5, n + 1.5), n)
        im2 = ax2.imshow(np.ma.masked_equal(seg, NODATA), cmap=cmap, norm=norm,
                         interpolation="nearest")
        ax2.set_title(f"{n} classes ({R['method']})")
        self.fig.colorbar(im2, ax=ax2, fraction=0.046, ticks=range(1, n + 1))

        ax3.imshow(np.ma.masked_where(~v_fl, flood > 0),
                   cmap=ListedColormap(["#d9d9d9", "#1f5fbf"]), vmin=0, vmax=1)
        ax3.set_title("Flood reference")

        cls = [r["Class"] for r in R["rows"]]
        fp = [r["Flood_pct_of_class"] for r in R["rows"]]
        colors = [cmap((c - 1) / max(n - 1, 1)) for c in cls]
        ax4.bar(cls, fp, color=colors, edgecolor="k", linewidth=0.5)
        ax4.axhline(100 * FLOOD_CLASS_FRACTION, color="k", ls="--", lw=0.8)
        ax4.set_xlabel("Class (low → high index)")
        ax4.set_ylabel("% of class that is flood")
        ax4.set_ylim(0, 100)
        ax4.set_xticks(cls)
        ax4.set_title("Flood share per class")
        ax5 = ax4.twinx()
        ax5.errorbar(cls, [r["Mean"] for r in R["rows"]], yerr=[r["Std"] for r in R["rows"]],
                     fmt="o-", color="k", ms=3, lw=0.8, capsize=2)
        ax5.set_ylabel("Class mean ± std")
        if R["best"]:
            ax4.axvline(R["best"]["Flood_if_class_ge"] - 0.5, color="red", lw=1.2)

        for a in (ax1, ax2, ax3):
            a.axis("off")
        self.fig.tight_layout()
        self.canvas.draw()

    def export_csv(self):
        if not self.result:
            messagebox.showinfo("Info", "Run the segmentation first.")
            return
        p = filedialog.asksaveasfilename(defaultextension=".csv",
                                         filetypes=[("CSV", "*.csv")])
        if not p:
            return
        R = self.result
        with open(p, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["Layer", self.layer_type.get(), "Method", R["method"],
                        "Classes", R["n"]])
            w.writerow(["Thresholds"] + [f"{v:.6f}" for v in R["thr"]])
            for k, v in R["ref"].items():
                w.writerow([k, v])
            w.writerow([])
            keys = list(R["rows"][0].keys())
            w.writerow(keys)
            for r in R["rows"]:
                w.writerow([r[k] for k in keys])
            w.writerow([])
            if R["cuts"]:
                keys = list(R["cuts"][0].keys())
                w.writerow(keys)
                for c in R["cuts"]:
                    w.writerow([c[k] for k in keys])
        messagebox.showinfo("Saved", f"Statistics saved to\n{p}")


if __name__ == "__main__":
    FloodApp().mainloop()
