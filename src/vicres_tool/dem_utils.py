#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DEM.py — Tạo lưới VIC-Res 0.05° từ DEM (elev_m = trung bình ô), bền với nodata.

- Đọc basin từ .zip → EPSG:4326, union 1 polygon, lưu .gpkg (nếu có thể).
- Đọc DEM (không reproject), mask nodata→NaN.
- Sinh lưới đều (mặc định 0.05°), lat duyệt Bắc→Nam; tính area km², frac_in_basin.
- Lấy elev_m = mean DEM trong mỗi ô [lon±½res, lat±½res]; nếu toàn NaN → vá bằng KNN (k=1)
  theo tâm ô (lon,lat) dựa trên các ô đã có giá trị.
- Xuất CSV + NetCDF template (mask, frac, lat_b, lon_b), vẽ quicklook nếu cần.
"""

from __future__ import annotations

from pathlib import Path
import zipfile
import warnings
from typing import Optional, Dict

warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import box
from pyproj import Geod, CRS
from pyproj import datadir as _pyproj_datadir
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

# Đặt PROJ_LIB để tránh lỗi proj.db trên vài máy Windows
os.environ.setdefault("PROJ_LIB", _pyproj_datadir.get_data_dir())

__all__ = [
    "run_all", "make_vic_grid", "load_basin_from_zip", "load_dem",
    "plot_basin_and_dem_quick", "plot_grid_over_dem_quick", "build_dirs"
]

# ======================== THƯ MỤC DỰ ÁN ========================= #

def build_dirs(root: Path) -> Dict[str, Path]:
    d: Dict[str, Path] = {
        "basin": root/"data/basin",
        "dem": root/"data/dem",
        "raw_imerg": root/"data/raw/IMERG",
        "interim": root/"data/interim",
        "proc": root/"data/proc",
        "grids": root/"outputs/grids",
        "forc_nc": root/"outputs/forcings/netcdf",
        "forc_ascii": root/"outputs/forcings/ascii",
        "figs": root/"outputs/figs",
        "qc": root/"outputs/qc",
    }
    for v in d.values():
        v.mkdir(parents=True, exist_ok=True)
    return d

# ======================= ĐỌC BASIN & DEM ======================== #

def load_basin_from_zip(zip_path: Path, out_dir: Path) -> gpd.GeoDataFrame:
    """Đọc shapefile lưu vực từ .zip, chuẩn EPSG:4326, union 1 polygon; cố gắng lưu .gpkg."""
    try:
        gdf = gpd.read_file(f"zip://{zip_path}")
    except Exception:
        tmp = out_dir / "_unzipped"
        tmp.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp)
        shp = list(tmp.rglob("*.shp"))
        assert shp, "Không tìm thấy .shp trong zip"
        gdf = gpd.read_file(shp[0])

    gdf = gdf.set_crs(4326) if gdf.crs is None else gdf.to_crs(4326)
    geom = unary_union(gdf.geometry.values)
    basin = gpd.GeoDataFrame({"name": [zip_path.stem], "geometry": [geom]}, crs="EPSG:4326")

    out_gpkg = out_dir / f"{zip_path.stem}.gpkg"
    try:
        basin.to_file(out_gpkg, driver="GPKG")
    except Exception:
        try:
            import fiona
            basin.to_file(out_gpkg, driver="GPKG", engine="fiona")
        except Exception:
            print("[WARN] Không ghi được GPKG (pyogrio/fiona). Bỏ qua lưu file.")
    return basin


def load_dem(dem_path: Path) -> xr.DataArray:
    """Đọc DEM; KHÔNG reproject; gán EPSG:4326 nếu thiếu; mask nodata → NaN."""
    dem = rxr.open_rasterio(dem_path, masked=True).squeeze(drop=True)  # dims: y,x
    crs = dem.rio.crs
    if crs is None:
        dem = dem.rio.write_crs("EPSG:4326", inplace=False)
        print("[INFO] DEM không có CRS → gán EPSG:4326 theo tên file.")
    else:
        try:
            epsg = CRS.from_user_input(crs).to_epsg()
        except Exception:
            epsg = None
        if epsg and epsg != 4326:
            print(f"[WARN] DEM CRS={crs}; tạm thời không reproject (thiếu proj.db).")

    # Mask mọi giá trị nodata (ngăn số âm sentinel lọt vào)
    nodata = dem.rio.nodata
    if nodata is not None:
        dem = dem.where(dem != nodata)

    dem.name = "elevation"
    dem.attrs.update(units="m", long_name="Elevation")
    return dem

# ======================== HỖ TRỢ & VẼ ========================= #

def _safe_to_csv(df: pd.DataFrame, path: Path) -> Path:
    """Ghi CSV an toàn trên Windows: dùng .tmp rồi os.replace(); nếu bị khoá → thêm timestamp."""
    import time, os
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
        return path
    except PermissionError:
        ts = time.strftime("%Y%m%d_%H%M%S")
        alt = path.with_name(path.stem + f"_{ts}" + path.suffix)
        df.to_csv(alt, index=False)
        print(f"[WARN] File đang bị khoá (Excel?): {path}. Đã ghi sang: {alt}")
        return alt
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _decimate_for_plot(da: xr.DataArray, max_pixels: int = 1_200_000):
    nx, ny = da.sizes["x"], da.sizes["y"]
    k = int(np.ceil(np.sqrt((nx*ny)/max_pixels)))
    k = max(1, k)
    return da.isel(y=slice(None, None, k), x=slice(None, None, k)), k


def plot_basin_and_dem_quick(basin_gdf: gpd.GeoDataFrame, dem_da: xr.DataArray,
                             buffer_deg: float = 0.2, cmap: str = "terrain",
                             save: bool = True, figs_dir: Optional[Path] = None):
    minx, miny, maxx, maxy = basin_gdf.total_bounds
    dem_cut = dem_da.rio.clip_box(minx=minx-buffer_deg, miny=miny-buffer_deg,
                                  maxx=maxx+buffer_deg, maxy=maxy+buffer_deg)
    dem_view, k = _decimate_for_plot(dem_cut)
    fig, ax = plt.subplots(figsize=(7,6), constrained_layout=True)
    dem_view.plot(ax=ax, cmap=cmap, robust=True)
    basin_gdf.boundary.plot(ax=ax, color="red", lw=1.2)
    ax.set_title(f"DEM & ranh giới (decimate x{k})"); ax.grid(True, alpha=.3)
    if save and figs_dir is not None:
        figs_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(figs_dir/"quicklook_dem_basin.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.show()


def plot_grid_over_dem_quick(grid_df: pd.DataFrame, basin_gdf: gpd.GeoDataFrame,
                             dem_da: xr.DataArray, res_deg: float = 0.05,
                             buffer_deg: float = 0.2, save: bool = True,
                             figs_dir: Optional[Path] = None):
    minx, miny, maxx, maxy = basin_gdf.total_bounds
    dem_cut = dem_da.rio.clip_box(minx=minx-buffer_deg, miny=miny-buffer_deg,
                                  maxx=maxx+buffer_deg, maxy=maxy+buffer_deg)
    dem_view, k = _decimate_for_plot(dem_cut)
    cells = gpd.GeoDataFrame(
        grid_df[["cell_id","lon","lat"]],
        geometry=[box(x-res_deg/2, y-res_deg/2, x+res_deg/2, y+res_deg/2)
                  for x,y in zip(grid_df.lon, grid_df.lat)],
        crs="EPSG:4326"
    )
    fig, ax = plt.subplots(figsize=(7,6), constrained_layout=True)
    dem_view.plot(ax=ax, cmap="terrain", robust=True)
    cells.boundary.plot(ax=ax, color="black", lw=0.3, alpha=0.6)
    basin_gdf.boundary.plot(ax=ax, color="red", lw=1.2)
    ax.set_title(f"Lưới VIC 0.05° chồng DEM (decimate x{k})"); ax.grid(True, alpha=.3)
    if save and figs_dir is not None:
        figs_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(figs_dir/"quicklook_grid_over_dem.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.show()

# ======================== LƯỚI & CAO ĐỘ ========================= #

def edges_from_centers(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c)
    mid = (c[:-1] + c[1:]) / 2
    first = c[0] - (mid[0] - c[0])
    last  = c[-1] + (c[-1] - mid[-1])
    return np.concatenate([[first], mid, [last]])


def cell_mean_elev(dem: xr.DataArray, lon: float, lat: float, half: float) -> float:
    """Trung bình độ cao trong ô [lon±half, lat±half]; nếu toàn NaN → nearest."""
    sub = dem.rio.clip_box(minx=lon-half, miny=lat-half, maxx=lon+half, maxy=lat+half)
    m = np.nanmean(sub.values)
    if np.isnan(m):
        try:
            m = float(dem.sel(x=lon, y=lat, method="nearest").values)
        except Exception:
            m = np.nan
    return float(m) if m is not None else np.nan


def make_vic_grid(basin_gdf: gpd.GeoDataFrame, dem_da: xr.DataArray,
                  res: float = 0.05, min_fraction: float = 0.01,
                  out_csv: Optional[Path] = None, out_nc: Optional[Path] = None,
                  prefix: str = "DaRiver"):
    """Sinh lưới VIC-Res, lấy elev_m = mean DEM trong ô, vá NaN bằng KNN(k=1)."""
    HALF = res/2
    geom = basin_gdf.geometry.values[0]
    minx, miny, maxx, maxy = basin_gdf.total_bounds
    # căn lưới theo chuẩn
    minx = np.floor((minx - HALF)/res)*res + HALF
    miny = np.floor((miny - HALF)/res)*res + HALF
    maxx = np.ceil ((maxx + HALF)/res)*res - HALF
    maxy = np.ceil ((maxy + HALF)/res)*res - HALF
    lons = np.arange(minx, maxx+1e-12, res)
    lats = np.arange(miny, maxy+1e-12, res)

    geod = Geod(ellps="WGS84")
    rows = []; cid = 1

    # Duyệt lat giảm dần (Bắc→Nam)
    for lat in lats[::-1]:
        for lon in lons:
            cell = box(lon-HALF, lat-HALF, lon+HALF, lat+HALF)
            inter = cell.intersection(geom)
            if inter.is_empty:
                continue
            a_cell,_ = geod.geometry_area_perimeter(cell)
            a_int,_  = geod.geometry_area_perimeter(inter)
            frac = abs(a_int)/abs(a_cell)
            if frac >= min_fraction:
                z = cell_mean_elev(dem_da, lon, lat, HALF)
                rows.append((cid, lon, lat, frac, abs(a_cell)/1e6, z)); cid += 1

    df = pd.DataFrame(rows, columns=["cell_id","lon","lat","frac_in_basin","cell_area_km2","elev_m"])

    # Vá các ô còn NaN bằng KNN (k=1) theo toạ độ tâm ô
    na = df["elev_m"].isna()
    if na.any():
        valid_xy = df.loc[~na, ["lon","lat"]].to_numpy()
        valid_z  = df.loc[~na, "elev_m"].to_numpy()
        miss_xy  = df.loc[na,  ["lon","lat"]].to_numpy()
        idx = cKDTree(valid_xy).query(miss_xy, k=1)[1]
        df.loc[na, "elev_m"] = valid_z[idx]

    # Template NetCDF
    latu = np.sort(df.lat.unique())[::-1]
    lonu = np.sort(df.lon.unique())
    mask = np.zeros((latu.size, lonu.size), bool)
    frac_arr = np.zeros_like(mask, float)
    ilat, ilon = {v:i for i,v in enumerate(latu)}, {v:i for i,v in enumerate(lonu)}
    for _, r in df.iterrows():
        mask[ilat[r.lat], ilon[r.lon]] = True
        frac_arr[ilat[r.lat], ilon[r.lon]] = r.frac_in_basin

    ds = xr.Dataset(
        {"mask": (("lat","lon"), mask), "frac": (("lat","lon"), frac_arr)},
        coords={"lat": ("lat", latu), "lon": ("lon", lonu),
                "lat_b": ("lat_b", edges_from_centers(latu)),
                "lon_b": ("lon_b", edges_from_centers(lonu))},
        attrs={"title": f"VIC-Res grid {res:.2f}° – {prefix} (elev=cell mean)",
               "conventions": "CF-1.8",
               "grid_res_deg": float(res),
               "mask_min_fraction": float(min_fraction)}
    )

    # Ghi file
    csv_path = None
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        csv_path = Path(str(out_csv)).with_name(out_csv.name.replace(".00",".0"))
        csv_path = _safe_to_csv(df, csv_path)

    nc_path = None
    if out_nc is not None:
        out_nc.parent.mkdir(parents=True, exist_ok=True)
        nc_path = Path(str(out_nc)).with_name(out_nc.name.replace(".00",".0"))
        comp = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
        ds.to_netcdf(nc_path, encoding=comp)

    print(f"[OK] Lưới VIC {res:.2f}°: {len(df)} ô. CSV + template NC đã lưu.")
    return df, ds, csv_path, nc_path

# ============================ WRAPPER ============================ #

def run_all(root: Path, zip_shp: Path, dem_tif: Path,
            res_deg: float = 0.05, min_fraction: float = 0.01,
            prefix: str = "DaRiver", do_plot: bool = True):
    """Pipeline gói gọn như notebook gốc. Trả về (grid_df, target_grid_ds, csv_path, nc_path)."""
    DIR = build_dirs(root)
    basin_gdf = load_basin_from_zip(zip_shp, DIR["basin"])
    dem_da = load_dem(dem_tif)

    csv_path = DIR["grids"]/f"vic_grid_{res_deg:.2f}_{prefix}.csv"
    nc_path  = DIR["grids"]/f"vic_grid_{res_deg:.2f}_{prefix}_template.nc"

    grid_df, target_grid, csv_out, nc_out = make_vic_grid(
        basin_gdf, dem_da, res=res_deg, min_fraction=min_fraction,
        out_csv=csv_path, out_nc=nc_path, prefix=prefix
    )

    if do_plot:
        plot_basin_and_dem_quick(basin_gdf, dem_da, save=True, figs_dir=DIR["figs"])
        plot_grid_over_dem_quick(grid_df, basin_gdf, dem_da, res_deg=res_deg,
                                 save=True, figs_dir=DIR["figs"])
    return grid_df, target_grid, csv_out, nc_out

# ============================== CLI ============================== #

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Tạo lưới VIC-Res từ DEM (elev_m = trung bình ô)")
    p.add_argument('--root', type=Path, required=True, help='Thư mục gốc dự án (VD D:/Python2_DaRiver)')
    p.add_argument('--zip-shp', type=Path, required=True, help='Đường dẫn .zip chứa shapefile lưu vực')
    p.add_argument('--dem-tif', type=Path, required=True, help='Đường dẫn DEM GeoTIFF (WGS84 hoặc gán EPSG:4326)')
    p.add_argument('--res', type=float, default=0.05, help='Độ phân giải lưới (độ). Mặc định 0.05')
    p.add_argument('--min-fraction', type=float, default=0.01, help='Ngưỡng diện tích chồng lấn để giữ ô (mặc định 0.01)')
    p.add_argument('--prefix', type=str, default='DaRiver', help='Tiền tố tên file đầu ra')
    p.add_argument('--no-plot', action='store_true', help='Tắt vẽ quicklook')

    a = p.parse_args()
    grid_df, target_grid, csv_out, nc_out = run_all(
        root=a.root, zip_shp=a.zip_shp, dem_tif=a.dem_tif,
        res_deg=a.res, min_fraction=a.min_fraction, prefix=a.prefix,
        do_plot=(not a.no_plot)
    )
    print("[OK] Lưới VIC đã tạo:")
    print("  CSV:", csv_out)
    print("  NC :", nc_out)
