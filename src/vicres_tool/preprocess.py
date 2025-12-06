# -*- coding: utf-8 -*-
"""
Orchestrator cho VIC-Res ingest:
- build_grid(): DEM → lưới VIC 0.05° + template NC (+ quicklook grid PNG)
- precip(): CHIRPS/IMERG → daily pr (grid & ASCII) + (tuỳ chọn) QC PNG
- temperature(): ERA5/ERA5-Land → Tmin/Tmax → VIC (grid/cell + ASCII)
- wind(): ERA5/ERA5-Land → WIND → VIC (cell + ASCII)

Chạy từ notebook: from vicres_tool.preprocess import *  → gọi các hàm
Chạy CLI:  python -m src.vicres_tool.preprocess <subcmd> [options]
"""

from __future__ import annotations
from pathlib import Path
import json, argparse, yaml
from . import dem_utils, precip_utils, temperature_utils as tair_utils, wind_utils

REPO_ROOT = Path(__file__).resolve().parents[2]

def _abs(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p)

def _load_params(p: Path | None = None) -> dict:
    p = p or (REPO_ROOT / "params.yaml")
    if not p.exists():
        raise FileNotFoundError(f"Không thấy {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8"))

# ---- helper: vẽ quicklook lưới VIC chồng ranh lưu vực ----
def _quicklook_grid(csv_path, basin_gpkg, out_png, res_deg=0.05, title_prefix="DaRiver", max_cells=50000):
    from pathlib import Path
    import pandas as pd, geopandas as gpd
    from shapely.geometry import box
    import matplotlib.pyplot as plt

    csv_path  = Path(csv_path)
    basin_gpkg = Path(basin_gpkg)
    out_png    = Path(out_png)

    df = pd.read_csv(csv_path)
    lon_candidates = ("lon","longitude","x","lon_b","center_lon","grid_lon")
    lat_candidates = ("lat","latitude","y","lat_b","center_lat","grid_lat")
    lon_col = next(c for c in df.columns if c.lower() in lon_candidates)
    lat_col = next(c for c in df.columns if c.lower() in lat_candidates)
    if "frac" in df.columns:
        df = df[df["frac"] > 0]
    if len(df) > max_cells:
        df = df.sample(max_cells, random_state=0)

    dx = dy = float(res_deg)
    geoms = [box(lon - dx/2, lat - dy/2, lon + dx/2, lat + dy/2)
             for lon, lat in zip(df[lon_col], df[lat_col])]
    g = gpd.GeoDataFrame(df, geometry=geoms, crs=4326)
    basin = gpd.read_file(basin_gpkg).to_crs(4326)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    ax = basin.boundary.plot(figsize=(7,7), linewidth=1.0)
    g.plot(ax=ax, linewidth=0.2, facecolor="none", alpha=0.6)
    ax.set_title(f"VIC grid ({res_deg}°) – {title_prefix}")
    ax.set_axis_off()
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()
    return out_png

# ---------- BUILD GRID ----------
def build_grid(params: dict | None = None) -> dict:
    from shutil import copy2
    P = params or _load_params()
    root = _abs(P["project"]["root"]); root.mkdir(parents=True, exist_ok=True)

    zip_shp = _abs(P["basin"]["vector_zip"])
    gpkg    = _abs(P["basin"].get("vector_gpkg", ""))
    dem_tif = _abs(P["dem"]["tif"])
    res     = float(P["grid"]["res_deg"])
    minfrac = float(P["grid"].get("min_fraction", 0.01))
    prefix  = P["grid"].get("prefix", "DaRiver")

    grid_df, target_grid, csv_out, nc_out = dem_utils.run_all(
        zip_shp=zip_shp, root=root, dem_tif=dem_tif, res_deg=res,
        min_fraction=minfrac, prefix=prefix, do_plot=True
    )

    cfg_csv = _abs(P["grid"]["csv"])
    cfg_nc  = _abs(P["grid"]["template_nc"])
    cfg_csv.parent.mkdir(parents=True, exist_ok=True)

    if cfg_csv.resolve() != csv_out.resolve():
        copy2(csv_out, cfg_csv)
    else:
        cfg_csv = csv_out
    if cfg_nc.resolve() != nc_out.resolve():
        copy2(nc_out, cfg_nc)
    else:
        cfg_nc = nc_out

    fig_png = REPO_ROOT / "outputs" / "figs" / f"quicklook_grid_{prefix}.png"
    try:
        basin_gpkg = gpkg if gpkg and gpkg.exists() else _abs(P["basin"]["vector_gpkg"])
        _quicklook_grid(cfg_csv, basin_gpkg, fig_png, res_deg=res, title_prefix=prefix)
    except Exception as e:
        print("[warn] quicklook grid error:", e)

    return {"csv": str(cfg_csv), "template_nc": str(cfg_nc), "quicklook": str(fig_png)}

# ---------- PRECIP ----------
def precip(params: dict | None = None) -> dict:
    P = params or _load_params()
    root = _abs(P["project"]["root"])
    precip_utils.init_paths(root)
    precip_utils.set_target_grid(_abs(P["grid"]["template_nc"]))

    src      = P["forcing"]["precip"]["source"].strip().upper()
    dl       = P["forcing"]["precip"].get("download", "auto")
    cleanup  = bool(P["forcing"]["precip"].get("cleanup_raw", False))
    write_nc = bool(P["forcing"]["precip"].get("write_netcdf", True))
    want_qc  = bool(P["forcing"]["precip"].get("qc_figs_cli", False))

    qc_png = None

    if src == "CHIRPS":
        ds = precip_utils.run_chirps_pipeline(
            P["forcing"]["precip"]["t0"], P["forcing"]["precip"]["t1"],
            res        = P["forcing"]["precip"].get("res", "p05"),
            method     = P["forcing"]["precip"].get("method", "linear"),
            download   = dl,
            cleanup_raw= cleanup,
            write_netcdf = write_nc,
            save_ascii = True
        )
        if want_qc:
            t0 = P["forcing"]["precip"]["t0"].replace("-", "")
            t1 = P["forcing"]["precip"]["t1"].replace("-", "")
            qc_png = REPO_ROOT / "outputs" / "figs" / "qc" / f"precip_CHIRPS_{t0}_{t1}.png"
            precip_utils.quick_qc_basin(
                ds, "CHIRPS→VIC 0.05°",
                basin_vector=_abs(P["basin"]["vector_gpkg"]),
                clim_fixed=(0, 30),
                save_png=qc_png
            )

    elif src == "IMERG":
        ds = precip_utils.run_imerg_pipeline(
            P["forcing"]["precip"]["t0"], P["forcing"]["precip"]["t1"],
            method     = P["forcing"]["precip"].get("method", "linear"),
            download   = dl,
            cleanup_raw= cleanup,
            write_netcdf = write_nc,
            save_ascii = True
        )
        if want_qc:
            t0 = P["forcing"]["precip"]["t0"].replace("-", "")
            t1 = P["forcing"]["precip"]["t1"].replace("-", "")
            qc_png = REPO_ROOT / "outputs" / "figs" / "qc" / f"precip_IMERG_{t0}_{t1}.png"
            precip_utils.quick_qc_basin(
                ds, "IMERG→VIC 0.05°",
                basin_vector=_abs(P["basin"]["vector_gpkg"]),
                clim_fixed=(0, 30),
                save_png=qc_png
            )
    else:
        raise ValueError("forcing.precip.source phải là CHIRPS hoặc IMERG")

    out = {"n_days": int(ds.sizes["time"])}
    if qc_png is not None:
        out["qc_png"] = str(qc_png)
    return out

# ---------- TEMPERATURE ----------
def temperature(params: dict | None = None) -> dict:
    P = params or _load_params()
    out = tair_utils.run_era5_temperature_to_vicres(
        root=_abs(P["project"]["root"]),
        dataset=P["forcing"]["temperature"]["dataset"],
        years=P["forcing"]["temperature"]["years"],
        months=P["forcing"]["temperature"]["months"],
        tz_offset_hours=int(P["forcing"]["temperature"].get("tz", 7)),
        vic_grid_csv=_abs(P["grid"]["csv"]),
        vic_grid_template_nc=_abs(P["grid"]["template_nc"]),
        basin_vector=_abs(P["basin"]["vector_gpkg"]),
        export_ascii=bool(P["forcing"]["temperature"].get("export_ascii", True)),
        backend=P["forcing"]["temperature"].get("backend", "auto"),
        show_plots=bool(P["forcing"]["temperature"].get("show_plots", True)),
    )
    return {k: str(v) for k, v in out.items()}

# ---------- WIND (tương thích mọi bản wind_utils) ----------
# -------------------- WIND (ERA5/ERA5-Land) --------------------
# ---------- WIND (ERA5/ERA5-Land) ----------
def wind(params: dict | None = None) -> dict:
    """
    Build VIC-Res wind forcing from ERA5/ERA5-Land.
    Đọc cấu hình từ params.yaml (hoặc dict truyền vào).
    """
    from . import wind_utils

    P = params or _load_params()
    root = _abs(P["project"]["root"])

    fw = P.get("forcing", {}).get("wind", {})
    dataset = (fw.get("dataset", "ERA5-Land") or "ERA5-Land")
    years   = fw.get("years")
    months  = fw.get("months")
    tz      = int(fw.get("tz", 0))

    backend = (fw.get("backend", "auto") or "auto").lower()
    if backend in ("auto", "ecmwf"):
        backend = "cds"
    elif backend not in ("cds", "local"):
        backend = "cds"

    basin_gpkg = fw.get("basin_vector") or _abs(P["basin"]["vector_gpkg"])
    tpl_nc     = fw.get("vic_grid_template_nc") or _abs(P["grid"]["template_nc"])

    export_ascii = bool(fw.get("export_ascii", True))
    show_plots   = bool(fw.get("show_plots", True))

    out = wind_utils.run_era5_wind_to_vicres(
        root=str(root),
        dataset=dataset,
        years=years,
        months=months,
        tz_offset_hours=tz,
        basin_vector=str(basin_gpkg),
        vic_grid_template_nc=str(tpl_nc),
        backend=backend,
        export_ascii=export_ascii,
        show_plots=show_plots,
    )
    # ép về str cho đẹp khi json.dumps ở CLI
    return {k: str(v) for k, v in out.items()}

# ---------- CLI ----------
def _cli():
    p = argparse.ArgumentParser(prog="vicres-ingest")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("build-grid")

    sp1 = sub.add_parser("precip")
    sp1.add_argument("--t0")
    sp1.add_argument("--t1")
    sp1.add_argument("--source", choices=["CHIRPS", "IMERG"])
    sp1.add_argument("--download", choices=["auto", "true", "false"])
    sp1.add_argument("--cleanup-raw", action="store_true")
    sp1.add_argument("--no-netcdf", action="store_true")
    sp1.add_argument("--qc-figs", action="store_true")   # NEW

    sp2 = sub.add_parser("temperature")
    sp2.add_argument("--years",  nargs="+", type=int)
    sp2.add_argument("--months", nargs="+", type=int)
    sp2.add_argument("--tz", type=int)

    # ------ WIND CLI ------
    sp3 = sub.add_parser("wind")
    sp3.add_argument("--years",  nargs="+", type=int)
    sp3.add_argument("--months", nargs="+", type=int)
    sp3.add_argument("--tz", type=int)
    sp3.add_argument("--dataset", choices=["ERA5-Land", "ERA5"])
    sp3.add_argument("--backend", choices=["auto", "cds", "ecmwf", "local"])
    sp3.add_argument("--force-redownload", action="store_true")
    sp3.add_argument("--clean-raw", action="store_true")
    sp3.add_argument("--no-ascii", dest="export_ascii", action="store_false")
    sp3.add_argument("--plots", dest="show_plots", action="store_true")

    p.add_argument("-p", "--params", type=Path, default=REPO_ROOT / "params.yaml")
    a = p.parse_args()

    P = _load_params(a.params)

    if a.cmd == "precip":
        if a.t0:       P["forcing"]["precip"]["t0"] = a.t0
        if a.t1:       P["forcing"]["precip"]["t1"] = a.t1
        if a.source:   P["forcing"]["precip"]["source"] = a.source
        if a.download: P["forcing"]["precip"]["download"] = a.download
        if a.cleanup_raw: P["forcing"]["precip"]["cleanup_raw"] = True
        if a.no_netcdf:   P["forcing"]["precip"]["write_netcdf"] = False
        if a.qc_figs:     P["forcing"]["precip"]["qc_figs_cli"] = True
        res = precip(P)

    elif a.cmd == "temperature":
        if a.years:  P["forcing"]["temperature"]["years"]  = a.years
        if a.months: P["forcing"]["temperature"]["months"] = a.months
        if a.tz is not None: P["forcing"]["temperature"]["tz"] = a.tz
        res = temperature(P)

    elif a.cmd == "wind":
        if a.years:  P["forcing"]["wind"]["years"]  = a.years
        if a.months: P["forcing"]["wind"]["months"] = a.months
        if a.tz is not None: P["forcing"]["wind"]["tz"] = a.tz
        if a.dataset: P["forcing"]["wind"]["dataset"] = a.dataset
        if a.backend: P["forcing"]["wind"]["backend"] = a.backend
        if a.force_redownload: P["forcing"]["wind"]["force_redownload"] = True
        if a.clean_raw:        P["forcing"]["wind"]["clean_raw"] = True
        if a.export_ascii is not None: P["forcing"]["wind"]["export_ascii"] = a.export_ascii
        if a.show_plots  is not None:  P["forcing"]["wind"]["show_plots"]  = a.show_plots
        res = wind(P)

    else:  # build-grid
        res = build_grid(P)

    print(json.dumps(res, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    _cli()
