"""
Error map with cartographic elements.

Compares a predicted flood map with the reference flood layer and draws a map with
coordinate grid, north arrow, scale bar, district and state boundaries, district
names, a locator inset and legend with pixel statistics.

Classes:  1 = correct flood (TP)   2 = false alarm (FP)   3 = missed flood (FN)
          0 = correct no-flood (TN)   255 = no data

Usage (Anaconda Prompt, env "flood"):
    python error_map.py <predicted.tif> <reference.tif> <output_prefix>
           [--districts india_districts.geojson] [--states india_states.geojson]
           [--title "Uttar Pradesh"]

By default the two GeoJSON files are looked for next to this script.
Writes <output_prefix>.tif (GeoTIFF) and <output_prefix>.png (300 dpi map).
Needs only: numpy, rasterio, matplotlib, geopandas (conda install -c conda-forge geopandas).
"""
import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
import matplotlib.patheffects as pe
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely.geometry import box

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("pred"); ap.add_argument("ref"); ap.add_argument("out")
ap.add_argument("--districts", default=os.path.join(HERE, "india_districts.geojson"))
ap.add_argument("--states", default=os.path.join(HERE, "india_states.geojson"))
ap.add_argument("--title", default="")
a = ap.parse_args()


# ------------------------------------------------------------ error raster
def align(path, prof):
    dst = np.full((prof["height"], prof["width"]), np.nan, "float32")
    with rasterio.open(path) as src:
        reproject(rasterio.band(src, 1), dst, src_transform=src.transform, src_crs=src.crs,
                  src_nodata=src.nodata, dst_transform=prof["transform"],
                  dst_crs=prof["crs"] or src.crs, dst_nodata=np.nan, resampling=Resampling.nearest)
        nd = src.nodata
    ok = np.isfinite(dst)
    if nd is not None:
        ok &= dst != nd
    return dst, ok

with rasterio.open(a.pred) as src:
    pred = src.read(1); prof = src.profile.copy()
ref, vref = align(a.ref, prof)
valid = (pred != 255) & vref
p, r = pred == 1, ref > 0
err = np.full(pred.shape, 255, np.uint8)
err[valid & ~p & ~r] = 0; err[valid & p & r] = 1; err[valid & p & ~r] = 2; err[valid & ~p & r] = 3
op = prof.copy(); op.update(dtype="uint8", count=1, nodata=255, compress="lzw")
with rasterio.open(a.out + ".tif", "w", **op) as dst:
    dst.write(err, 1)
n = {k: int((err == k).sum()) for k in range(4)}
tp, fp, fn, tn = n[1], n[2], n[3], n[0]
prec = tp / (tp + fp) if tp + fp else float("nan")
rec = tp / (tp + fn) if tp + fn else float("nan")
f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
print(f"TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}")
print(f"Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}  "
      f"False alarms = {100 * fp / max(tp + fp, 1):.1f}% of predicted flood")

# ----------------------------------------- reproject for display (lat/lon)
H, W = err.shape
left, bottom, right, top = rasterio.transform.array_bounds(H, W, prof["transform"])
tr, w2, h2 = calculate_default_transform(prof["crs"], "EPSG:4326", W, H, left, bottom, right, top)
scale = max(1.0, max(w2, h2) / 3000.0)                       # max ~3000 px on display
w2, h2 = int(w2 / scale), int(h2 / scale)
tr, w2, h2 = calculate_default_transform(prof["crs"], "EPSG:4326", W, H, left, bottom, right, top,
                                         dst_width=w2, dst_height=h2)
disp = np.full((h2, w2), 255, np.uint8)
reproject(err, disp, src_transform=prof["transform"], src_crs=prof["crs"], src_nodata=255,
          dst_transform=tr, dst_crs="EPSG:4326", dst_nodata=255, resampling=Resampling.mode)
x0, y1 = tr.c, tr.f
x1, y0 = x0 + tr.a * w2, y1 + tr.e * h2
lat_c = (y0 + y1) / 2

# --------------------------------------------------------------- map frame
asp = (y1 - y0) / ((x1 - x0) * math.cos(math.radians(lat_c)))
fig = plt.figure(figsize=(13, max(7.5, 9.0 * asp + 1.6)))
ax = fig.add_axes([0.06, 0.08, 0.66, 0.84])
cmap = ListedColormap(["#f2f2f2", "#1f5fbf", "#e6550d", "#9e9ac8"])
ax.imshow(np.ma.masked_equal(disp, 255), cmap=cmap, vmin=-0.5, vmax=3.5, interpolation="nearest",
          extent=(x0, x1, y0, y1), zorder=1)
ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
ax.set_aspect(1 / math.cos(math.radians(lat_c)))
frame = box(x0, y0, x1, y1)

# boundaries
state_name = None
if os.path.exists(a.districts):
    d = gpd.read_file(a.districts).to_crs("EPSG:4326")
    dc = d[d.intersects(frame)].copy()
    dc.boundary.plot(ax=ax, color="#444", lw=0.6, linestyle="--", zorder=3)
    for _, row in dc.iterrows():
        g = row.geometry.intersection(frame)
        if g.is_empty or g.area < 0.004 * frame.area or not row["DISTRICT"]:
            continue
        pt = g.representative_point()
        ax.text(pt.x, pt.y, row["DISTRICT"], fontsize=8, ha="center", va="center", color="#222", zorder=5,
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
    cen = gpd.GeoSeries([box(x0, y0, x1, y1).centroid], crs="EPSG:4326")
    hit = d[d.contains(cen.iloc[0])]
    state_name = hit["ST_NM"].iloc[0] if len(hit) else None
else:
    print("District file not found – boundaries skipped:", a.districts)
st = None
if os.path.exists(a.states):
    st = gpd.read_file(a.states).to_crs("EPSG:4326")
    st[st.intersects(frame)].boundary.plot(ax=ax, color="#111", lw=1.4, zorder=4)

# coordinate grid
def nice_step(span):
    for s in [0.05, 0.1, 0.2, 0.25, 0.5, 1, 2]:
        if span / s <= 6:
            return s
    return 5
sx, sy = nice_step(x1 - x0), nice_step(y1 - y0)
xt = np.arange(math.ceil(x0 / sx) * sx, x1, sx); yt = np.arange(math.ceil(y0 / sy) * sy, y1, sy)
fmt = lambda v, h: f"{v:.2f}°{h}" if (sx < 0.1 or sy < 0.1) else f"{v:.1f}°{h}"
ax.set_xticks(xt); ax.set_xticklabels([fmt(v, "E") for v in xt], fontsize=9)
ax.set_yticks(yt); ax.set_yticklabels([fmt(v, "N") for v in yt], fontsize=9)
ax.tick_params(top=True, right=True, labeltop=True, labelright=True, direction="in")
ax.grid(color="#666", lw=0.4, ls=":", zorder=2)
ax.set_xlabel(""); ax.set_ylabel("")

# north arrow
ax.annotate("", xy=(0.95, 0.96), xytext=(0.95, 0.86), xycoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>,head_width=0.5,head_length=0.9", color="#111", lw=1.6), zorder=8)
ax.text(0.95, 0.965, "N", transform=ax.transAxes, ha="center", va="bottom", fontsize=14, fontweight="bold", zorder=8,
        path_effects=[pe.withStroke(linewidth=3, foreground="white")])

# scale bar
km_deg = 111.32 * math.cos(math.radians(lat_c))
width_km = (x1 - x0) * km_deg
target = width_km * 0.2
L = min([1, 2, 5, 10, 20, 25, 50, 100, 200], key=lambda v: abs(v - target))
bx, by = x0 + 0.04 * (x1 - x0), y0 + 0.05 * (y1 - y0)
seg = L / km_deg / 4; hh = 0.012 * (y1 - y0)
ax.add_patch(Rectangle((bx - 0.01 * (x1 - x0), by - 0.02 * (y1 - y0)), 4 * seg + 0.05 * (x1 - x0),
                       hh + 0.075 * (y1 - y0), fc="white", ec="none", alpha=0.85, zorder=7))
for i in range(4):
    ax.add_patch(Rectangle((bx + i * seg, by), seg, hh, fc="#111" if i % 2 == 0 else "white", ec="#111", lw=0.8, zorder=8))
for i, lab in [(0, "0"), (2, f"{L / 2:g}"), (4, f"{L:g} km")]:
    ax.text(bx + i * seg, by + hh + 0.008 * (y1 - y0), lab, ha="center", va="bottom", fontsize=8.5, zorder=8)

# legend
tot = tp + fp + fn + tn
ax.legend(handles=[Patch(fc="#1f5fbf", label=f"Correct flood (TP)  {100 * tp / tot:.2f}%"),
                   Patch(fc="#e6550d", label=f"False alarm (FP)  {100 * fp / tot:.2f}%"),
                   Patch(fc="#9e9ac8", label=f"Missed flood (FN)  {100 * fn / tot:.2f}%"),
                   Patch(fc="#f2f2f2", ec="#999", label=f"Correct no-flood (TN)  {100 * tn / tot:.1f}%"),
                   plt.Line2D([], [], color="#444", lw=0.8, ls=(0, (4, 2)), label="District boundary"),
                   plt.Line2D([], [], color="#111", lw=1.4, label="State boundary")],
          loc="upper left", bbox_to_anchor=(1.10, 0.62), borderaxespad=0, fontsize=9.5, title=f"Precision {prec:.2f} · Recall {rec:.2f} · F1 {f1:.2f}",
          title_fontsize=9.5, framealpha=0.95).set_zorder(9)

# locator inset
if st is not None and state_name:
    s_geom = st[st["ST_NM"].str.replace("&", "and").str.lower().str[:6] ==
                state_name.replace("&", "and").lower()[:6]]
    if len(s_geom):
        ins = fig.add_axes([0.76, 0.64, 0.21, 0.26])
        s_geom.plot(ax=ins, color="#e9e9e9", edgecolor="#333", lw=0.7)
        ins.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="#e6550d", ec="#e6550d", alpha=0.5, lw=1.2))
        ins.set_xticks([]); ins.set_yticks([]); ins.set_xlabel(''); ins.set_ylabel('')
        ins.set_aspect(1 / math.cos(math.radians(lat_c)))
        ins.set_title(f"Location in {state_name}", fontsize=10)
        for sp in ins.spines.values():
            sp.set_edgecolor("#333")

title = a.title or (state_name or "")
fig.suptitle(f"Flood prediction errors – {title}" if title else "Flood prediction errors",
             x=0.06, y=0.975, ha="left", fontsize=14, fontweight="bold")
fig.text(0.06, 0.03, "Prediction: Extra Trees (leave-one-location-out). Boundaries: current districts (datta07/INDIAN-SHAPEFILES) "
         "and state/UT boundaries (Datameet). WGS 84.", fontsize=7.5, color="#555")
fig.savefig(a.out + ".png", dpi=300, bbox_inches="tight", facecolor="white")
print("Saved", a.out + ".tif", "and", a.out + ".png")
