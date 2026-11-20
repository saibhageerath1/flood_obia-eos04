"""
OBIA Flood Detection with Machine Learning (NDFI + Temporal Std-Dev)
====================================================================

Workflow
--------
1. ALIGN      Std-Dev and flood layers are reprojected/clipped onto the NDFI grid,
              then trimmed to the common extent.
2. SEGMENT    SLIC superpixels on the 2-band stack (NDFI, StdDev) -> image objects.
              Processed tile by tile, so large scenes fit in memory.
3. FEATURES   Per object: mean, std, min, max of NDFI and StdDev, NDFI x StdDev,
              area, and context (mean of each band in a window around the object).
4. LABEL      Each object gets the flood fraction from the reference layer.
              Objects with >= 50 % flood = flood (1), else non-flood (0).
              Optionally train only on "pure" objects (< 20 % or > 80 % flood).
5. SPLIT      Spatial block split (the image is cut into blocks; ~30 % of blocks are
              held out for testing) - avoids over-optimistic accuracy from
              neighbouring objects ending up in both train and test.
6. TRAIN      Random Forest / Extra Trees / Gradient Boosting / SVM / Logistic Reg.
7. EVALUATE   Object-level and pixel-level accuracy on the test blocks,
              confusion matrix, F1, IoU, Kappa, feature importance.
8. MAP        Flood map + flood-probability map for the whole scene (GeoTIFF).
9. SAVE       Model saved (.joblib) so it can be APPLIED to another date / area
              that has only NDFI + StdDev (no flood reference needed).

Requirements (conda):
    conda install -n flood -c conda-forge scikit-learn joblib
    (on top of: numpy rasterio scikit-image matplotlib scipy)
"""

import csv
import warnings
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import joblib
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
from scipy import ndimage
from skimage.segmentation import slic, find_boundaries
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

warnings.filterwarnings("ignore", category=FutureWarning)

TILE = 2048
CONTEXT_WIN = 31          # pixels, window for neighbourhood context features


# =================================================================== raster I/O
def read_raster(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        profile = src.profile.copy()
        nodata = src.nodata
    valid = np.isfinite(arr)
    if nodata is not None:
        valid &= arr != nodata
    return arr, valid, profile


def align_to_reference(path, ref_profile):
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
    return dst, valid


def crop_to(mask, arrays, profile):
    if not mask.any():
        raise ValueError("The input layers do not overlap.")
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    prof = profile.copy()
    prof.update(height=r1 - r0, width=c1 - c0,
                transform=rasterio.windows.transform(Window(c0, r0, c1 - c0, r1 - r0),
                                                     profile["transform"]))
    sl = (slice(r0, r1), slice(c0, c1))
    return [a[sl] for a in arrays], prof


def write_tif(path, arr, profile, dtype, nodata):
    p = profile.copy()
    p.update(dtype=dtype, count=1, nodata=nodata, compress="lzw")
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr.astype(dtype), 1)


# ================================================================ segmentation
def normalise(a, valid, lims=None):
    if lims is None:
        lims = tuple(np.percentile(a[valid], [2, 98]))
    lo, hi = lims
    out = np.clip((a - lo) / max(hi - lo, 1e-12), 0, 1).astype("float32")
    out[~valid] = 0
    return out, lims


def segment_objects(ndfi_n, sd_n, valid, obj_size, compactness, status=None):
    """Tiled SLIC on the 2-band stack. Returns int32 label image (0 = no object)."""
    H, W = valid.shape
    labels = np.zeros((H, W), dtype=np.int32)
    offset = 0
    tiles = [(r, c) for r in range(0, H, TILE) for c in range(0, W, TILE)]
    for i, (r, c) in enumerate(tiles, 1):
        if status:
            status(f"Segmenting tile {i}/{len(tiles)}…")
        sl = (slice(r, min(r + TILE, H)), slice(c, min(c + TILE, W)))
        m = valid[sl]
        if m.sum() < 10:
            continue
        stack = np.dstack([ndfi_n[sl], sd_n[sl]])
        n_seg = max(1, int(m.sum() / obj_size))
        # unmasked SLIC is ~50x faster than masked SLIC; invalid pixels are dropped after
        lab = slic(stack, n_segments=n_seg, compactness=compactness, sigma=1,
                   channel_axis=-1, start_label=1, enforce_connectivity=True,
                   max_num_iter=10)
        lab = lab.astype(np.int32)
        lab[~m] = 0
        lab[lab > 0] += offset
        labels[sl] = lab
        offset = labels.max()
    # relabel consecutively
    uniq, inv = np.unique(labels, return_inverse=True)
    if uniq[0] != 0:
        inv += 1
    return inv.reshape(H, W).astype(np.int32)


# ==================================================================== features
FEATURES = ["NDFI_mean", "NDFI_std", "NDFI_min", "NDFI_max",
            "SD_mean", "SD_std", "SD_min", "SD_max",
            "NDFIxSD_mean", "Area_px",
            "NDFI_context", "SD_context",
            "NDFI_minus_context", "SD_minus_context"]


def object_features(labels, ndfi, sd, valid, status=None):
    n = int(labels.max())
    lab = labels.ravel()
    keep = lab > 0
    l = lab[keep]
    idx = np.arange(1, n + 1)

    def mstats(a):
        v = a.ravel()[keep].astype("float64")
        cnt = np.bincount(l, minlength=n + 1)[1:]
        s = np.bincount(l, v, minlength=n + 1)[1:]
        s2 = np.bincount(l, v * v, minlength=n + 1)[1:]
        mean = s / np.maximum(cnt, 1)
        std = np.sqrt(np.maximum(s2 / np.maximum(cnt, 1) - mean ** 2, 0))
        mn = ndimage.minimum(a, labels, idx)
        mx = ndimage.maximum(a, labels, idx)
        return cnt, mean, std, np.asarray(mn), np.asarray(mx)

    if status: status("Computing object features (NDFI)…")
    cnt, nm, ns, nmin, nmax = mstats(ndfi)
    if status: status("Computing object features (StdDev)…")
    _, sm, ss, smin, smax = mstats(sd)
    _, xm, _, _, _ = mstats(ndfi * sd)

    if status: status("Computing context features…")
    w = valid.astype("float32")
    den = ndimage.uniform_filter(w, CONTEXT_WIN) + 1e-6
    nctx_px = ndimage.uniform_filter(np.where(valid, ndfi, 0), CONTEXT_WIN) / den
    sctx_px = ndimage.uniform_filter(np.where(valid, sd, 0), CONTEXT_WIN) / den
    _, nctx, _, _, _ = mstats(nctx_px)
    _, sctx, _, _, _ = mstats(sctx_px)

    X = np.column_stack([nm, ns, nmin, nmax, sm, ss, smin, smax, xm, cnt,
                         nctx, sctx, nm - nctx, sm - sctx]).astype("float32")
    return X


def object_flood_fraction(labels, flood):
    n = int(labels.max())
    lab = labels.ravel()
    keep = lab > 0
    cnt = np.bincount(lab[keep], minlength=n + 1)[1:]
    fl = np.bincount(lab[keep], (flood.ravel()[keep] > 0).astype("float64"),
                     minlength=n + 1)[1:]
    return fl / np.maximum(cnt, 1)


def object_blocks(labels, n_blocks):
    """Assign each object to a spatial block (by centroid)."""
    H, W = labels.shape
    n = int(labels.max())
    rr, cc = np.indices((H, W))
    lab = labels.ravel(); keep = lab > 0
    cnt = np.bincount(lab[keep], minlength=n + 1)[1:]
    r = np.bincount(lab[keep], rr.ravel()[keep], minlength=n + 1)[1:] / np.maximum(cnt, 1)
    c = np.bincount(lab[keep], cc.ravel()[keep], minlength=n + 1)[1:] / np.maximum(cnt, 1)
    br = np.minimum((r / H * n_blocks).astype(int), n_blocks - 1)
    bc = np.minimum((c / W * n_blocks).astype(int), n_blocks - 1)
    return br * n_blocks + bc


# ======================================================================= models
def make_model(name):
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1,
                                      class_weight="balanced", random_state=0)
    if name == "Extra Trees":
        return ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1,
                                    class_weight="balanced", random_state=0)
    if name == "Gradient Boosting":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                              class_weight="balanced", random_state=0)
    if name == "SVM (RBF)":
        return make_pipeline(StandardScaler(),
                             SVC(C=10, gamma="scale", probability=True,
                                 class_weight="balanced", random_state=0))
    if name == "Logistic Regression":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, class_weight="balanced"))
    raise ValueError(name)


MODELS = ["Random Forest", "Extra Trees", "Gradient Boosting", "SVM (RBF)",
          "Logistic Regression"]
SVM_MAX_TRAIN = 5000      # SVM training time grows ~quadratically


def site_normalise(X):
    """Per-scene robust standardisation of object features: (x - median) / IQR.
    Area_px is left unchanged. Makes features comparable between locations."""
    X = X.astype("float32").copy()
    med = np.median(X, axis=0)
    q25, q75 = np.percentile(X, [25, 75], axis=0)
    iqr = q75 - q25
    iqr[iqr < 1e-9] = 1.0
    keep = FEATURES.index("Area_px")
    Z = (X - med) / iqr
    Z[:, keep] = X[:, keep]
    return Z


def metrics(pred, ref, weight=None):
    pred = pred.astype(bool); ref = ref.astype(bool)
    w = np.ones(pred.shape) if weight is None else weight.astype("float64")
    tp = w[pred & ref].sum(); fp = w[pred & ~ref].sum()
    fn = w[~pred & ref].sum(); tn = w[~pred & ~ref].sum()
    n = tp + fp + fn + tn
    oa = (tp + tn) / n if n else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else np.nan
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2 if n else np.nan
    kap = (oa - pe) / (1 - pe) if n and pe != 1 else np.nan
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, OA=oa, Precision=prec, Recall=rec,
                F1=f1, IoU=iou, Kappa=kap)


# ========================================================================= GUI
class OBIAApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("OBIA Flood Detection – Machine Learning")
        self.geometry("1450x920")
        self.v = {k: tk.StringVar() for k in
                  ["ndfi", "sd", "flood", "outdir", "model_in"]}
        self.model_name = tk.StringVar(value="Random Forest")
        self.obj_size = tk.IntVar(value=100)
        self.compact = tk.DoubleVar(value=1.0)
        self.test_frac = tk.DoubleVar(value=0.3)
        self.pure_only = tk.BooleanVar(value=False)
        self.report = None
        self._build()

    # ------------------------------------------------------------------ layout
    def _build(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="x", padx=8, pady=6)
        tr = ttk.Frame(nb, padding=8); ap = ttk.Frame(nb, padding=8)
        nb.add(tr, text="1. Train model"); nb.add(ap, text="2. Apply saved model")

        def frow(parent, r, label, key, save_dir=False, ext=None):
            ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w")
            ttk.Entry(parent, textvariable=self.v[key], width=95).grid(row=r, column=1, padx=4, pady=2)
            cmd = (lambda: self._pick_dir(key)) if save_dir else (lambda: self._pick(key, ext))
            ttk.Button(parent, text="Browse", command=cmd).grid(row=r, column=2)

        frow(tr, 0, "NDFI (.tif):", "ndfi")
        frow(tr, 1, "Temporal StdDev (.tif):", "sd")
        frow(tr, 2, "Flood reference (binary .tif):", "flood")
        frow(tr, 3, "Output folder:", "outdir", save_dir=True)

        opt = ttk.Frame(tr); opt.grid(row=4, column=0, columnspan=3, sticky="w", pady=6)
        ttk.Label(opt, text="Model:").pack(side="left")
        ttk.Combobox(opt, textvariable=self.model_name, values=MODELS, width=18,
                     state="readonly").pack(side="left", padx=4)
        ttk.Label(opt, text="Object size (px):").pack(side="left", padx=(12, 0))
        ttk.Spinbox(opt, from_=10, to=5000, increment=10, textvariable=self.obj_size,
                    width=6).pack(side="left", padx=4)
        ttk.Label(opt, text="Compactness:").pack(side="left", padx=(12, 0))
        ttk.Spinbox(opt, from_=0.1, to=20, increment=0.5, textvariable=self.compact,
                    width=6).pack(side="left", padx=4)
        ttk.Label(opt, text="Test fraction:").pack(side="left", padx=(12, 0))
        ttk.Spinbox(opt, from_=0.1, to=0.5, increment=0.05, textvariable=self.test_frac,
                    width=5).pack(side="left", padx=4)
        ttk.Checkbutton(opt, text="Train on pure objects only",
                        variable=self.pure_only).pack(side="left", padx=12)
        ttk.Button(opt, text="Train + Evaluate + Map", command=self.train).pack(side="left", padx=8)
        ttk.Button(opt, text="Export report CSV", command=self.export_csv).pack(side="left")

        frow(ap, 0, "NDFI (.tif):", "ndfi")
        frow(ap, 1, "Temporal StdDev (.tif):", "sd")
        frow(ap, 2, "Saved model (.joblib):", "model_in", ext=[("Model", "*.joblib")])
        frow(ap, 3, "Output folder:", "outdir", save_dir=True)
        ttk.Label(ap, text="(Flood reference optional – if given in tab 1, it is used for accuracy.)"
                  ).grid(row=4, column=0, columnspan=2, sticky="w")
        ttk.Button(ap, text="Apply model", command=self.apply).grid(row=4, column=2, pady=6)

        self.status = ttk.Label(self, text="Ready.", foreground="gray")
        self.status.pack(fill="x", padx=10)

        body = ttk.PanedWindow(self, orient="horizontal")
        body.pack(fill="both", expand=True, padx=8, pady=4)
        ff = ttk.Frame(body)
        self.fig = Figure(figsize=(9, 7.5))
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

    def _pick(self, key, ext=None):
        p = filedialog.askopenfilename(filetypes=ext or [("GeoTIFF", "*.tif *.tiff"), ("All", "*.*")])
        if p:
            self.v[key].set(p)
            if key == "ndfi" and not self.v["outdir"].get():
                self.v["outdir"].set(os.path.join(os.path.dirname(p), "obia_output"))

    def _pick_dir(self, key):
        p = filedialog.askdirectory()
        if p:
            self.v[key].set(p)

    def _st(self, msg):
        self.status.config(text=msg); self.update_idletasks()

    def _log(self, s=""):
        self.text.insert("end", s + "\n"); self.text.see("end"); self.update_idletasks()

    # ---------------------------------------------------------- common steps
    def _load(self, need_flood):
        for k in ["ndfi", "sd", "outdir"] + (["flood"] if need_flood else []):
            if not self.v[k].get():
                raise ValueError(f"Please select: {k}")
        os.makedirs(self.v["outdir"].get(), exist_ok=True)
        self._st("Reading NDFI…")
        ndfi, vn, prof = read_raster(self.v["ndfi"].get())
        self._st("Aligning StdDev to NDFI grid…")
        sd, vs = align_to_reference(self.v["sd"].get(), prof)
        arrays = [ndfi, sd, vn, vs]
        mask = vn & vs
        flood = vf = None
        if need_flood or self.v["flood"].get():
            self._st("Aligning flood layer…")
            flood, vf = align_to_reference(self.v["flood"].get(), prof)
            arrays += [flood, vf]
            mask &= vf
        arrays, prof = crop_to(mask, arrays, prof)
        ndfi, sd, vn, vs = arrays[:4]
        valid = vn & vs
        if flood is not None:
            flood, vf = arrays[4:6]
            valid &= vf
        return ndfi, sd, flood, valid, prof

    def _objects(self, ndfi, sd, valid, nlims=None, slims=None, obj_size=None, compact=None):
        nn, nlims = normalise(ndfi, valid, nlims)
        sn, slims = normalise(sd, valid, slims)
        labels = segment_objects(nn, sn, valid, obj_size or self.obj_size.get(),
                                 compact or self.compact.get(), self._st)
        X = object_features(labels, ndfi, sd, valid, self._st)
        return labels, X, nlims, slims

    # ----------------------------------------------------------------- train
    def train(self):
        try:
            self.text.delete("1.0", "end")
            ndfi, sd, flood, valid, prof = self._load(need_flood=True)
            labels, X, nlims, slims = self._objects(ndfi, sd, valid)
            n_obj = X.shape[0]
            self._log(f"Scene: {valid.shape[1]} x {valid.shape[0]} px, {int(valid.sum())} valid")
            self._log(f"Objects created: {n_obj}  (mean size {valid.sum() / n_obj:.0f} px)")

            frac = object_flood_fraction(labels, flood)
            y = (frac >= 0.5).astype(int)
            self._log(f"Flood objects: {int(y.sum())}   Non-flood objects: {int((1 - y).sum())}")

            # spatial block split
            blocks = object_blocks(labels, 8)
            rng = np.random.default_rng(0)
            ub = np.unique(blocks)
            test_b = rng.choice(ub, max(1, int(round(len(ub) * self.test_frac.get()))), replace=False)
            test = np.isin(blocks, test_b)
            train = ~test
            if self.pure_only.get():
                train &= (frac <= 0.2) | (frac >= 0.8)
            if len(np.unique(y[train])) < 2:
                raise ValueError("Training set has only one class – check the flood layer / extent.")
            self._log(f"Train objects: {int(train.sum())}   Test objects: {int(test.sum())} "
                      f"({len(test_b)}/{len(ub)} spatial blocks held out)")

            name = self.model_name.get()
            model = make_model(name)
            Xtr, ytr = X[train], y[train]
            if name.startswith("SVM") and len(ytr) > SVM_MAX_TRAIN:
                sel = rng.choice(len(ytr), SVM_MAX_TRAIN, replace=False)
                Xtr, ytr = Xtr[sel], ytr[sel]
                self._log(f"SVM: training on a random subset of {SVM_MAX_TRAIN} objects")
            self._st(f"Training {name}…")
            model.fit(Xtr, ytr)

            self._st("Predicting…")
            prob = model.predict_proba(X)[:, 1]
            pred = prob >= 0.5

            # ---- evaluation on held-out blocks
            area = X[:, FEATURES.index("Area_px")]
            m_obj = metrics(pred[test], y[test])
            obj_test_mask = np.zeros(n_obj + 1, bool); obj_test_mask[1:] = test
            px_test = obj_test_mask[labels] & valid
            pred_px_lut = np.zeros(n_obj + 1, bool); pred_px_lut[1:] = pred
            m_px = metrics(pred_px_lut[labels][px_test], flood[px_test] > 0)
            m_train = metrics(pred[train], y[train])

            self._st("Feature importance…")
            sub = np.where(test)[0]
            if len(sub) > 5000:
                sub = rng.choice(sub, 5000, replace=False)
            pi = permutation_importance(model, X[sub], y[sub], scoring="f1",
                                        n_repeats=5, random_state=0, n_jobs=-1)
            imp = sorted(zip(FEATURES, pi.importances_mean, pi.importances_std),
                         key=lambda t: -t[1])

            # ---- outputs
            out = self.v["outdir"].get()
            base = os.path.splitext(os.path.basename(self.v["ndfi"].get()))[0]
            lut_p = np.zeros(n_obj + 1, "float32"); lut_p[1:] = prob
            prob_px = np.where(valid, lut_p[labels], -1)
            flood_px = np.where(valid, (prob_px >= 0.5).astype("uint8"), 255)
            write_tif(os.path.join(out, f"{base}_flood_{name.split()[0]}.tif"), flood_px, prof, "uint8", 255)
            write_tif(os.path.join(out, f"{base}_floodprob_{name.split()[0]}.tif"), prob_px, prof, "float32", -1)
            write_tif(os.path.join(out, f"{base}_objects.tif"), labels, prof, "int32", 0)
            mpath = os.path.join(out, f"flood_model_{name.split()[0]}.joblib")
            joblib.dump(dict(model=model, features=FEATURES, nlims=nlims, slims=slims,
                             obj_size=self.obj_size.get(), compact=self.compact.get(),
                             model_name=name), mpath)
            # object table
            with open(os.path.join(out, f"{base}_objects_features.csv"), "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["object_id"] + FEATURES + ["flood_fraction", "label", "set",
                                                       "flood_prob", "pred"])
                for i in range(n_obj):
                    w.writerow([i + 1] + [f"{v:.5g}" for v in X[i]] +
                               [f"{frac[i]:.3f}", y[i], "test" if test[i] else "train",
                                f"{prob[i]:.3f}", int(pred[i])])

            self.report = dict(model=name, n_obj=n_obj, m_obj=m_obj, m_px=m_px,
                               m_train=m_train, imp=imp)
            self._print_report(mpath)
            self._plot(ndfi, labels, flood, valid, prob_px, flood_px, px_test, imp)
            self._st(f"Done. Outputs in {out}")
        except Exception as e:
            self._st("Error.")
            messagebox.showerror("Error", str(e))
            raise

    def _print_report(self, mpath):
        R = self.report
        self._log("\n=== Model: " + R["model"] + " ===")

        def block(title, m):
            self._log(f"\n{title}")
            self._log(f"  OA={m['OA']:.4f}  Precision={m['Precision']:.4f}  Recall={m['Recall']:.4f}")
            self._log(f"  F1={m['F1']:.4f}  IoU={m['IoU']:.4f}  Kappa={m['Kappa']:.4f}")
            self._log(f"  Confusion  TP={m['TP']:.0f} FP={m['FP']:.0f} FN={m['FN']:.0f} TN={m['TN']:.0f}")

        block("Training objects (optimistic – for reference only):", R["m_train"])
        block("TEST objects (held-out spatial blocks):", R["m_obj"])
        block("TEST pixels (held-out spatial blocks):", R["m_px"])
        self._log("\nFeature importance (drop in F1 when feature is shuffled):")
        for f, m, s in R["imp"]:
            self._log(f"  {f:20s} {m:8.4f} ± {s:.4f}")
        self._log(f"\nModel saved: {mpath}")
        self._log("Also saved: flood map, flood-probability map, object map, object feature CSV.")

    # ----------------------------------------------------------------- apply
    def apply(self):
        try:
            self.text.delete("1.0", "end")
            if not self.v["model_in"].get():
                raise ValueError("Please select a saved model (.joblib).")
            bundle = joblib.load(self.v["model_in"].get())
            ndfi, sd, flood, valid, prof = self._load(need_flood=False)
            labels, X, _, _ = self._objects(ndfi, sd, valid, bundle["nlims"], bundle["slims"],
                                            bundle["obj_size"], bundle["compact"])
            self._st("Predicting…")
            if bundle.get("site_norm"):
                X = site_normalise(X)
            if bundle.get("feat_idx") is not None:
                X = X[:, bundle["feat_idx"]]
            thr = float(bundle.get("threshold", 0.5))
            prob = bundle["model"].predict_proba(X)[:, 1]
            lut = np.zeros(len(prob) + 1, "float32"); lut[1:] = prob
            prob_px = np.where(valid, lut[labels], -1)
            flood_px = np.where(valid, (prob_px >= thr).astype("uint8"), 255)
            out = self.v["outdir"].get()
            base = os.path.splitext(os.path.basename(self.v["ndfi"].get()))[0]
            write_tif(os.path.join(out, f"{base}_flood_applied.tif"), flood_px, prof, "uint8", 255)
            write_tif(os.path.join(out, f"{base}_floodprob_applied.tif"), prob_px, prof, "float32", -1)
            self._log(f"Model: {bundle['model_name']}   objects: {len(prob)}   cut-off: {thr:.3f}")
            self._log(f"Flood area: {int((flood_px == 1).sum())} px "
                      f"({100 * (flood_px == 1).sum() / valid.sum():.2f}% of valid area)")
            px_test = None
            if flood is not None:
                m = metrics(flood_px[valid] == 1, flood[valid] > 0)
                self._log(f"Accuracy vs reference: OA={m['OA']:.4f} F1={m['F1']:.4f} "
                          f"IoU={m['IoU']:.4f} Kappa={m['Kappa']:.4f}")
            self._plot(ndfi, labels, flood, valid, prob_px, flood_px, px_test, None)
            self._st(f"Done. Outputs in {out}")
        except Exception as e:
            self._st("Error.")
            messagebox.showerror("Error", str(e))

    # ------------------------------------------------------------------ plot
    def _plot(self, ndfi, labels, flood, valid, prob_px, flood_px, px_test, imp):
        self.fig.clear()
        axs = [self.fig.add_subplot(2, 2, i) for i in range(1, 5)]
        H, W = labels.shape
        step = max(1, int(max(H, W) / 1500))          # downsample for display
        s = (slice(None, None, step), slice(None, None, step))

        d = np.where(valid, ndfi, np.nan)[s]
        lo, hi = np.nanpercentile(d, [2, 98])
        axs[0].imshow(d, cmap="gray", vmin=lo, vmax=hi)
        b = find_boundaries(labels[s], mode="inner")
        axs[0].imshow(np.ma.masked_where(~b, b), cmap=ListedColormap(["yellow"]), alpha=0.6)
        axs[0].set_title("NDFI + object boundaries")

        im = axs[1].imshow(np.ma.masked_less(prob_px[s], 0), cmap="RdYlBu", vmin=0, vmax=1)
        axs[1].set_title("Flood probability")
        self.fig.colorbar(im, ax=axs[1], fraction=0.046)

        if flood is not None:
            # 0 TN, 1 FP, 2 FN, 3 TP
            comp = np.where(valid, (flood_px == 1).astype(int) + 2 * (flood > 0), -1)[s]
            axs[2].imshow(np.ma.masked_less(comp, 0),
                          cmap=ListedColormap(["#eeeeee", "#e6550d", "#9e9ac8", "#1f5fbf"]),
                          vmin=-0.5, vmax=3.5)
            if px_test is not None:
                axs[2].contour(px_test[s], levels=[0.5], colors="k", linewidths=0.6)
            axs[2].set_title("Blue=TP  Orange=FP  Purple=FN  (black = test blocks)", fontsize=8)
        else:
            axs[2].imshow(np.ma.masked_equal(flood_px[s], 255),
                          cmap=ListedColormap(["#eeeeee", "#1f5fbf"]), vmin=0, vmax=1)
            axs[2].set_title("Predicted flood")

        if imp:
            names = [t[0] for t in imp][::-1]
            vals = [t[1] for t in imp][::-1]
            axs[3].barh(names, vals, color="#4a7fb5")
            axs[3].set_title("Feature importance (Δ F1)")
            axs[3].tick_params(axis="y", labelsize=7)
        else:
            axs[3].axis("off")
        for a in axs[:3]:
            a.axis("off")
        self.fig.tight_layout()
        self.canvas.draw()

    def export_csv(self):
        if not self.report:
            messagebox.showinfo("Info", "Train a model first."); return
        p = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not p:
            return
        R = self.report
        with open(p, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["Model", R["model"], "Objects", R["n_obj"]])
            w.writerow([])
            w.writerow(["Set", "OA", "Precision", "Recall", "F1", "IoU", "Kappa", "TP", "FP", "FN", "TN"])
            for nm, m in [("Train objects", R["m_train"]), ("Test objects", R["m_obj"]),
                          ("Test pixels", R["m_px"])]:
                w.writerow([nm] + [m[k] for k in ["OA", "Precision", "Recall", "F1", "IoU",
                                                  "Kappa", "TP", "FP", "FN", "TN"]])
            w.writerow([])
            w.writerow(["Feature", "Importance", "Std"])
            for f, m, s in R["imp"]:
                w.writerow([f, m, s])
        messagebox.showinfo("Saved", p)


if __name__ == "__main__":
    OBIAApp().mainloop()
