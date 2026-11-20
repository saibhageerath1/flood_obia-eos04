"""
Make all map figures for the flood OBIA report, with cartographic elements
(coordinate grid, north arrow, scale bar, district & state boundaries, legends).

For every location in the site list it creates:
  <name>_inputs.png        NDFI | temporal std-dev | reference flood          (report: input layers)
  <name>_segmentation.png  SLIC object boundaries over NDFI, zoomed subset    (report: segmentation)
  <name>_floodmaps.png     reference | Extra Trees flood map | probability    (report: flood maps)
  <name>_errors.png        TP / FP / FN map with locator inset                (report: error / false alarms)
and one overview:
  all_sites_errors.png     error maps of all locations side by side

Inputs
  sites.csv         the location list saved by flood_obia_multisite.py ("Save list"):
                    columns name, ndfi, sd, flood
  results folder    the flood_obia_multisite.py output folder containing
                    <name>_objects.tif, <name>_flood_LOLO_Extra.tif, <name>_floodprob_LOLO_Extra.tif

Usage (Anaconda Prompt, env "flood"):
  python make_report_maps.py sites.csv "path\\to\\obia_multisite_output" "path\\to\\report_maps"
  optional: --model Extra   --zoom_km 6   --width 2200
            --zoom NAME:LAT,LON  (centre of the segmentation zoom for one site; repeatable)

india_districts.geojson and india_states.geojson must be in the same folder as this script.
Maps are drawn from a resampled copy (~2200 px wide) for display; accuracy numbers in the
report come from the full-resolution results.
"""
import argparse
import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import from_bounds as win_from_bounds
from shapely.geometry import box
from skimage.segmentation import find_boundaries
from mpl_toolkits.axes_grid1 import make_axes_locatable


def bottom_strip(ax):
    cax = make_axes_locatable(ax).append_axes("bottom", size="4.5%", pad=0.42)
    return cax


HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("sites"); ap.add_argument("results"); ap.add_argument("out")
ap.add_argument("--model", default="Extra")
ap.add_argument("--zoom_km", type=float, default=6.0)
ap.add_argument("--width", type=int, default=2200)
ap.add_argument("--zoom", action="append", default=[])
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True)

DIST = gpd.read_file(os.path.join(HERE, "india_districts.geojson")).to_crs("EPSG:4326")
STAT = gpd.read_file(os.path.join(HERE, "india_states.geojson")).to_crs("EPSG:4326")
ZOOM = {}
for z in A.zoom:
    n, c = z.split(":"); la, lo = map(float, c.split(",")); ZOOM[n] = (la, lo)

ERR_CMAP = ListedColormap(["#f2f2f2", "#1f5fbf", "#e6550d", "#9e9ac8"])
BIN_CMAP = ListedColormap(["#f2f2f2", "#1f5fbf"])
SRC_NOTE = ("Boundaries: districts (datta07/INDIAN-SHAPEFILES), states/UTs (Datameet, Survey of India "
            "depiction). WGS 84.")


# =================================================================== raster helpers
def display_grid(path, width):
    with rasterio.open(path) as s:
        x0, y0, x1, y1 = transform_bounds(s.crs, "EPSG:4326", *s.bounds)
    latc = (y0 + y1) / 2
    h = int(round(width * (y1 - y0) / ((x1 - x0) * math.cos(math.radians(latc)))))
    return dict(ext=(x0, x1, y0, y1), w=width, h=h, latc=latc,
                tr=from_bounds(x0, y0, x1, y1, width, h))


def warp(path, g, how=Resampling.average):
    dst = np.full((g["h"], g["w"]), np.nan, "float32")
    with rasterio.open(path) as s:
        nd = s.nodata
        reproject(rasterio.band(s, 1), dst, src_transform=s.transform, src_crs=s.crs, src_nodata=nd,
                  dst_transform=g["tr"], dst_crs="EPSG:4326", dst_nodata=np.nan, resampling=how)
    return dst


def stretch(a):
    v = a[np.isfinite(a)]
    return np.percentile(v, [2, 98]) if v.size else (0, 1)


# ================================================================== map decoration
def nice(span):
    for s in [0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1, 2]:
        if span / s <= 5:
            return s
    return 5


def decorate(ax, ext, latc, arrow=True, scalebar=True, dist_labels=True, ylabels=True, fs=8):
    x0, x1, y0, y1 = ext
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect(1 / math.cos(math.radians(latc)))
    frame = box(x0, y0, x1, y1)
    d = DIST[DIST.intersects(frame)]
    if len(d):
        d.boundary.plot(ax=ax, color="#333", lw=0.6, linestyle="--", zorder=3)
        if dist_labels:
            for _, r in d.iterrows():
                g = r.geometry.intersection(frame)
                if g.is_empty or g.area < 0.02 * frame.area or not r["DISTRICT"]:
                    continue
                p = g.representative_point()
                ax.text(p.x, p.y, r["DISTRICT"], fontsize=fs - 0.5, ha="center", va="center", zorder=5,
                        color="#111", path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
    s = STAT[STAT.intersects(frame)]
    if len(s):
        s.boundary.plot(ax=ax, color="#000", lw=1.3, zorder=4)
    sx, sy = nice(x1 - x0), nice(y1 - y0)
    dec = 2 if min(sx, sy) < 0.1 else 1
    xt = np.arange(math.ceil(x0 / sx) * sx, x1, sx); yt = np.arange(math.ceil(y0 / sy) * sy, y1, sy)
    ax.set_xticks(xt); ax.set_xticklabels([f"{v:.{dec}f}°E" for v in xt], fontsize=fs)
    ax.set_yticks(yt); ax.set_yticklabels([f"{v:.{dec}f}°N" for v in yt] if ylabels else [], fontsize=fs)
    ax.tick_params(direction="in", top=True, right=True)
    ax.grid(color="#555", lw=0.35, ls=":", zorder=2)
    ax.set_xlabel(""); ax.set_ylabel("")
    if arrow:
        ax.annotate("", xy=(0.93, 0.96), xytext=(0.93, 0.85), xycoords="axes fraction", zorder=8,
                    arrowprops=dict(arrowstyle="-|>,head_width=0.45,head_length=0.8", color="#111", lw=1.5))
        ax.text(0.93, 0.965, "N", transform=ax.transAxes, ha="center", va="bottom", fontsize=fs + 4,
                fontweight="bold", zorder=8, path_effects=[pe.withStroke(linewidth=3, foreground="white")])
    if scalebar:
        kmd = 111.32 * math.cos(math.radians(latc))
        L = min([0.5, 1, 2, 5, 10, 20, 25, 50, 100], key=lambda v: abs(v - 0.22 * (x1 - x0) * kmd))
        seg = L / kmd / 4; bx = x0 + 0.05 * (x1 - x0); by = y0 + 0.05 * (y1 - y0); hh = 0.013 * (y1 - y0)
        ax.add_patch(Rectangle((bx - 0.015 * (x1 - x0), by - 0.025 * (y1 - y0)), 4 * seg + 0.07 * (x1 - x0),
                               0.09 * (y1 - y0), fc="white", ec="none", alpha=0.85, zorder=7))
        for i in range(4):
            ax.add_patch(Rectangle((bx + i * seg, by), seg, hh, fc="#111" if i % 2 == 0 else "white",
                                   ec="#111", lw=0.7, zorder=8))
        for i, t in [(0, "0"), (2, f"{L / 2:g}"), (4, f"{L:g} km")]:
            ax.text(bx + i * seg, by + hh + 0.01 * (y1 - y0), t, ha="center", va="bottom", fontsize=fs - 0.5, zorder=8)


def note(fig, extra=""):
    fig.text(0.01, 0.005, (extra + " " if extra else "") + SRC_NOTE, fontsize=7, color="#555")


def state_of(lon, lat):
    from shapely.geometry import Point
    h = DIST[DIST.contains(Point(lon, lat))]
    return h["ST_NM"].iloc[0] if len(h) else ""


def inset(fig, rect, ext, state):
    s = STAT[STAT["ST_NM"].str.lower().str[:6] == state.lower()[:6]]
    if not len(s):
        return
    x0, x1, y0, y1 = ext
    ia = fig.add_axes(rect)
    s.plot(ax=ia, color="#e9e9e9", edgecolor="#333", lw=0.7)
    ia.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="#e6550d", ec="#e6550d", alpha=0.55))
    ia.set_xticks([]); ia.set_yticks([]); ia.set_xlabel(""); ia.set_ylabel("")
    ia.set_aspect(1 / math.cos(math.radians((y0 + y1) / 2)))
    ia.set_title(f"Location in {state}", fontsize=9)


def save(fig, name):
    p = os.path.join(A.out, name)
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("  saved", p)


# ======================================================================= per site
def find(name, kind):
    p = os.path.join(A.results, f"{name}_{kind}.tif")
    return p if os.path.exists(p) else None


with open(A.sites, newline="") as fh:
    SITES = [dict(r) for r in csv.DictReader(fh)]
overview = []

for S in SITES:
    name = S["name"]; print(f"[{name}]")
    g = display_grid(S["ndfi"], A.width); ext, latc = g["ext"], g["latc"]
    ndfi = warp(S["ndfi"], g); sd = warp(S["sd"], g)
    ref = warp(S["flood"], g)
    valid = np.isfinite(ndfi) & np.isfinite(sd) & np.isfinite(ref)
    refb = np.where(valid, (ref >= 0.5).astype("float32"), np.nan)
    st = state_of((ext[0] + ext[1]) / 2, (ext[2] + ext[3]) / 2)
    title_loc = f"{name}" + (f" ({st})" if st and st.lower() not in name.lower() else "")
    asp = g["h"] / g["w"]

    # ---- 1. inputs
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.4 * asp + 1.6))
    for k, (arr, cm, lab) in enumerate([(ndfi, "YlGnBu", "NDFI"), (sd, "magma", "Temporal std-dev")]):
        lo, hi = stretch(arr)
        im = axs[k].imshow(np.where(valid, arr, np.nan), cmap=cm, vmin=lo, vmax=hi, extent=ext, zorder=1,
                           interpolation="nearest")
        cb = fig.colorbar(im, cax=bottom_strip(axs[k]), orientation="horizontal")
        cb.set_label(f"{lab} (2–98% stretch)", fontsize=9); cb.ax.tick_params(labelsize=8)
        axs[k].set_title(lab, fontsize=12, fontweight="bold", loc="left")
    axs[2].imshow(np.ma.masked_invalid(refb), cmap=BIN_CMAP, vmin=0, vmax=1, extent=ext, zorder=1, interpolation="nearest")
    axs[2].set_title("Reference flood layer", fontsize=12, fontweight="bold", loc="left")
    c2 = bottom_strip(axs[2]); c2.set_axis_off()
    c2.legend(handles=[Patch(fc="#1f5fbf", label="Flood"), Patch(fc="#f2f2f2", ec="#999", label="No flood")],
              loc="center", ncol=2, fontsize=9, frameon=False)
    for k, ax in enumerate(axs):
        decorate(ax, ext, latc, arrow=(k == 0), scalebar=(k == 0), dist_labels=(k == 2), ylabels=(k == 0))
    fig.suptitle(f"Input layers – {title_loc}", x=0.01, ha="left", fontsize=14, fontweight="bold")
    note(fig); save(fig, f"{name}_inputs.png")

    # ---- 2. segmentation zoom (full resolution window)
    op = find(name, "objects")
    if op:
        if name in ZOOM:
            zla, zlo = ZOOM[name]
        else:   # default: zoom on the most flooded part of the scene
            fr = np.nan_to_num(refb)
            from scipy import ndimage
            sm = ndimage.uniform_filter(fr, 61, mode="constant")
            iy, ix = np.unravel_index(np.argmax(sm), sm.shape)
            zlo = ext[0] + (ix + 0.5) * (ext[1] - ext[0]) / g["w"]
            zla = ext[3] - (iy + 0.5) * (ext[3] - ext[2]) / g["h"]
        half_lat = A.zoom_km / 2 / 110.574; half_lon = A.zoom_km / 2 / (111.32 * math.cos(math.radians(zla)))
        zlo = min(max(zlo, ext[0] + half_lon), ext[1] - half_lon)
        zla = min(max(zla, ext[2] + half_lat), ext[3] - half_lat)
        zb = (zlo - half_lon, zla - half_lat, zlo + half_lon, zla + half_lat)
        with rasterio.open(op) as so:
            b = transform_bounds("EPSG:4326", so.crs, *zb)
            win = win_from_bounds(*b, so.transform).round_offsets().round_lengths()
            lab = so.read(1, window=win, boundless=True, fill_value=0)
            wtr = so.window_transform(win); crs = so.crs
        zh, zw = lab.shape
        zn = np.full((zh, zw), np.nan, "float32")
        with rasterio.open(S["ndfi"]) as sn:
            reproject(rasterio.band(sn, 1), zn, src_transform=sn.transform, src_crs=sn.crs, src_nodata=sn.nodata,
                      dst_transform=wtr, dst_crs=crs, dst_nodata=np.nan, resampling=Resampling.nearest)
        zr = np.full((zh, zw), np.nan, "float32")
        with rasterio.open(S["flood"]) as sf:
            reproject(rasterio.band(sf, 1), zr, src_transform=sf.transform, src_crs=sf.crs, src_nodata=sf.nodata,
                      dst_transform=wtr, dst_crs=crs, dst_nodata=np.nan, resampling=Resampling.nearest)
        zext_native = rasterio.transform.array_bounds(zh, zw, wtr)
        zx0, zy0, zx1, zy1 = transform_bounds(crs, "EPSG:4326", *zext_native)
        zext = (zx0, zx1, zy0, zy1)
        bnd = find_boundaries(lab, mode="inner") & (lab > 0)
        fig, axs = plt.subplots(1, 2, figsize=(15, 7.6))
        lo, hi = stretch(zn)
        for k, ax in enumerate(axs):
            ax.imshow(zn, cmap="gray", vmin=lo, vmax=hi, extent=zext, zorder=1, interpolation="nearest")
            if k == 1:
                ax.imshow(np.ma.masked_where(~(zr > 0), zr), cmap=ListedColormap(["#1f5fbf"]), alpha=0.45,
                          extent=zext, zorder=1.5, interpolation="nearest")
            ax.imshow(np.ma.masked_where(~bnd, bnd), cmap=ListedColormap(["#ffd400"]), extent=zext, zorder=2.5,
                      interpolation="nearest")
            decorate(ax, zext, zla, arrow=(k == 0), scalebar=(k == 0), dist_labels=False, ylabels=(k == 0))
        axs[0].set_title("SLIC objects over NDFI", fontsize=12, fontweight="bold", loc="left")
        axs[1].set_title("SLIC objects with reference flood", fontsize=12, fontweight="bold", loc="left")
        axs[1].legend(handles=[Patch(fc="none", ec="#ffd400", lw=2, label="Object boundary"),
                               Patch(fc="#1f5fbf", alpha=0.45, label="Reference flood")],
                      loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2, fontsize=9, frameon=False)
        fig.suptitle(f"Image objects (zoom {A.zoom_km:g} km, centre {zla:.3f}°N {zlo:.3f}°E) – {title_loc}",
                     x=0.01, ha="left", fontsize=13, fontweight="bold")
        note(fig, "Full-resolution subset."); save(fig, f"{name}_segmentation.png")
    else:
        print("  objects file not found – segmentation figure skipped")

    # ---- 3. flood maps + 4. error map
    fp = find(name, f"flood_LOLO_{A.model}"); pp = find(name, f"floodprob_LOLO_{A.model}")
    if not fp:
        print(f"  {name}_flood_LOLO_{A.model}.tif not found – flood and error maps skipped"); continue
    pred = warp(fp, g); predb = np.where(valid & np.isfinite(pred), (pred >= 0.5).astype("float32"), np.nan)
    prob = warp(pp, g) if pp else None
    if prob is not None:
        prob[prob < 0] = np.nan
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.4 * asp + 1.6))
    for k, (arr, t) in enumerate([(refb, "Reference flood layer"), (predb, "Extra Trees flood map (unseen location)")]):
        axs[k].imshow(np.ma.masked_invalid(arr), cmap=BIN_CMAP, vmin=0, vmax=1, extent=ext, zorder=1, interpolation="nearest")
        axs[k].set_title(t, fontsize=12, fontweight="bold", loc="left")
        ck = bottom_strip(axs[k]); ck.set_axis_off()
        ck.legend(handles=[Patch(fc="#1f5fbf", label="Flood"), Patch(fc="#f2f2f2", ec="#999", label="No flood")],
                  loc="center", ncol=2, fontsize=9, frameon=False)
    if prob is not None:
        im = axs[2].imshow(np.where(valid, prob, np.nan), cmap="Blues", vmin=0, vmax=1, extent=ext, zorder=1,
                           interpolation="nearest")
        cb = fig.colorbar(im, cax=bottom_strip(axs[2]), orientation="horizontal")
        cb.set_label("Flood probability", fontsize=9); cb.ax.tick_params(labelsize=8)
    axs[2].set_title("Flood probability", fontsize=12, fontweight="bold", loc="left")
    for k, ax in enumerate(axs):
        decorate(ax, ext, latc, arrow=(k == 0), scalebar=(k == 0), dist_labels=(k == 1), ylabels=(k == 0))
    fig.suptitle(f"Flood maps – {title_loc} (leave-one-location-out)", x=0.01, ha="left", fontsize=14, fontweight="bold")
    note(fig); save(fig, f"{name}_floodmaps.png")

    err = np.full(predb.shape, np.nan, "float32")
    ok = np.isfinite(predb) & np.isfinite(refb)
    P, R = predb == 1, refb == 1
    err[ok & ~P & ~R] = 0; err[ok & P & R] = 1; err[ok & P & ~R] = 2; err[ok & ~P & R] = 3
    overview.append((title_loc, err, ext, latc))
    fig = plt.figure(figsize=(13, max(7.5, 9.0 * asp + 1.6)))
    ax = fig.add_axes([0.06, 0.08, 0.66, 0.84])
    ax.imshow(np.ma.masked_invalid(err), cmap=ERR_CMAP, vmin=-0.5, vmax=3.5, extent=ext, zorder=1, interpolation="nearest")
    decorate(ax, ext, latc, fs=9)
    ax.legend(handles=[Patch(fc="#1f5fbf", label="Correct flood (TP)"), Patch(fc="#e6550d", label="False alarm (FP)"),
                       Patch(fc="#9e9ac8", label="Missed flood (FN)"), Patch(fc="#f2f2f2", ec="#999", label="Correct no-flood (TN)"),
                       plt.Line2D([], [], color="#333", lw=0.8, ls="--", label="District boundary"),
                       plt.Line2D([], [], color="#000", lw=1.3, label="State boundary")],
              loc="upper left", bbox_to_anchor=(1.10, 0.62), borderaxespad=0, fontsize=9.5, framealpha=0.95)
    if st:
        inset(fig, [0.76, 0.64, 0.21, 0.26], ext, st)
    fig.suptitle(f"Flood prediction errors – {title_loc}", x=0.06, y=0.975, ha="left", fontsize=14, fontweight="bold")
    note(fig, "Extra Trees, leave-one-location-out. Display resampled; exact accuracy from error_map.py.")
    save(fig, f"{name}_errors.png")

# ======================================================================== overview
if overview:
    n = len(overview); cols = 3 if n > 4 else 2; rows = math.ceil(n / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(6.2 * cols, 6.0 * rows))
    axs = np.atleast_1d(axs).ravel()
    for k, (t, err, ext, latc) in enumerate(overview):
        axs[k].imshow(np.ma.masked_invalid(err), cmap=ERR_CMAP, vmin=-0.5, vmax=3.5, extent=ext, zorder=1,
                      interpolation="nearest")
        decorate(axs[k], ext, latc, dist_labels=False, fs=7)
        axs[k].set_title(t, fontsize=11, fontweight="bold", loc="left")
    for ax in axs[n:]:
        ax.set_axis_off()
    fig.legend(handles=[Patch(fc="#1f5fbf", label="Correct flood (TP)"), Patch(fc="#e6550d", label="False alarm (FP)"),
                        Patch(fc="#9e9ac8", label="Missed flood (FN)"), Patch(fc="#f2f2f2", ec="#999", label="Correct no-flood (TN)"),
                        plt.Line2D([], [], color="#333", lw=0.8, ls="--", label="District boundary"),
                        plt.Line2D([], [], color="#000", lw=1.3, label="State boundary")],
               loc="lower center", ncol=6, fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Extra Trees flood prediction errors at each unseen location", x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.text(0.01, -0.012, SRC_NOTE, fontsize=7, color="#555")
    save(fig, "all_sites_errors.png")
print("Done.")
