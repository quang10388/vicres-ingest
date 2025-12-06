# -*- coding: utf-8 -*-
"""
vicres_utils_3.py — ERA5(-Land) 2m Temperature (Tmin/Tmax) → VIC/VIC-Res (Đà River)

- Tải ERA5-Land / ERA5 Reanalysis (2m_temperature) theo bbox lưu vực
- Gộp giờ → ngày (Tmin/Tmax) theo múi giờ địa phương (UTC+7 mặc định)
- Regrid sang lưới VIC 0.05° (conservative_normed nếu có xESMF/ESMF; dự phòng linear)
- Cắt theo lưu vực, vẽ QC (series + map) với tên file có prefix nguồn + kỳ thời gian
- Xuất NetCDF + ASCII tách riêng cho TAIR để không lẫn với PRCP
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr  # noqa
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
try:
    import geopandas as gpd
except Exception:
    gpd = None

from cdsapi import Client
xr.set_options(keep_attrs=True)

# ===== Hiển thị & giới hạn màu mặc định =====
CMAP_T = "RdYlBu_r"
TMAX_CLIM_DEFAULT = (10.0, 40.0)   # °C
TMIN_CLIM_DEFAULT = (-5.0, 30.0)   # °C

# ===== Thư mục chuẩn (tách riêng TAIR) =====
def default_dirs(root: Path) -> Dict[str, Path]:
    d = {
        "basin_data":   root / "data" / "basin",
        "basin_inputs": root / "inputs" / "basin",
        "grids":        root / "outputs" / "grids",
        "raw_era5l":    root / "data" / "raw" / "ERA5L" / "hourly",
        "raw_era5":     root / "data" / "raw" / "ERA5" / "hourly",
        "interim":      root / "data" / "interim",
        # --- tách riêng nhánh nhiệt độ:
        "proc_tair":    root / "data" / "proc" / "tair",
        "weights":      root / "data" / "weights",
        "forc_ascii_tair": root / "outputs" / "forcings" / "ascii" / "tair",
        "figs_tair":    root / "outputs" / "figs" / "qc" / "tair",
    }
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d

# ===== Hỗ trợ xESMF =====
def have_xesmf() -> tuple[bool, str]:
    try:
        import xesmf  # noqa
        try:
            import ESMF  # noqa
            return True, ""
        except Exception as e:
            return False, f"ESMF missing: {e}"
    except Exception as e:
        return False, f"xESMF import failed: {e}"

def pick_backend(prefer: str = "auto") -> str:
    has_x, why = have_xesmf()
    if prefer == "xesmf":
        if not has_x: raise RuntimeError("backend='xesmf' nhưng thiếu xesmf/ESMF: "+why)
        return "xesmf"
    if prefer == "xarray": return "xarray"
    return "xesmf" if has_x else "xarray"

# ===== Tìm kiếm dữ liệu đầu vào =====
def _scan_vectors(folder: Path) -> List[Path]:
    out = []
    for pat in ("*.gpkg","*.shp","*.geojson"):
        out += list(Path(folder).glob(pat))
    return out

def find_basin_vector_multi(*folders: Path) -> Path:
    cands = []
    for f in folders:
        if f and Path(f).exists():
            cands += _scan_vectors(f)
    if not cands:
        tried = " | ".join(str(f) for f in folders if f)
        raise FileNotFoundError("Không thấy ranh lưu vực (.gpkg/.shp/.geojson). Đã tìm trong: "+tried)
    return sorted(cands, key=lambda p: (p.suffix.lower()!=".gpkg", len(str(p))))[0]

def read_basin(path_vector: Path):
    if gpd is None: raise RuntimeError("Cần geopandas để đọc lưu vực.")
    gdf = gpd.read_file(path_vector)
    if gdf.empty: raise ValueError("Layer lưu vực rỗng.")
    if gdf.crs is None: raise ValueError("Layer không có CRS (cần EPSG:4326).")
    return gdf.to_crs(4326)

def basin_bbox_area(gdf) -> List[float]:
    minx, miny, maxx, maxy = gdf.total_bounds
    return [float(maxy), float(minx), float(miny), float(maxx)]  # [N,W,S,E] cho CDS

def find_vic_grid_csv(grids_dir: Path) -> Path:
    cands = list(Path(grids_dir).glob("vic_grid*.csv")) or list(Path(grids_dir).glob("*.csv"))
    for p in cands:
        try:
            head = pd.read_csv(p, nrows=3)
            if {"lon","lat"}.issubset(head.columns): return p
        except Exception: pass
    raise FileNotFoundError(f"Không thấy CSV lưới VIC trong {grids_dir}")

def find_vic_template_nc(grids_dir: Path) -> Optional[Path]:
    cands = list(Path(grids_dir).glob("*template*.nc")) + list(Path(grids_dir).glob("vic_grid*.nc"))
    return cands[0] if cands else None

# ===== Lưới VIC =====
def build_vic_grid(csv_path: Path, template_nc: Optional[Path] = None,
                   default_res_deg: float = 0.05):
    df = pd.read_csv(csv_path)
    assert {"lon","lat"}.issubset(df.columns)
    if "cell_id" not in df.columns:
        df["cell_id"] = np.arange(1, len(df)+1)
    df["lon_r"] = df["lon"].round(6); df["lat_r"] = df["lat"].round(6)
    lons = np.sort(df["lon_r"].unique()); lats = np.sort(df["lat_r"].unique())
    is_rect = (len(df) == len(lons)*len(lats))
    lon_b = lat_b = None
    if template_nc and Path(template_nc).exists():
        try:
            ds = xr.open_dataset(template_nc)
            xname = "lon" if "lon" in ds.coords else ("x" if "x" in ds.coords else None)
            yname = "lat" if "lat" in ds.coords else ("y" if "y" in ds.coords else None)
            if xname and yname:
                lons_tpl = np.sort(np.asarray(ds[xname].values).astype(float).ravel())
                lats_tpl = np.sort(np.asarray(ds[yname].values).astype(float).ravel())
                if np.allclose(lons, lons_tpl, atol=1e-4) and np.allclose(lats, lats_tpl, atol=1e-4):
                    lons, lats = lons_tpl, lats_tpl
        except Exception: pass
    if is_rect:
        resx = np.median(np.diff(lons)) if len(lons)>1 else default_res_deg
        resy = np.median(np.diff(lats)) if len(lats)>1 else default_res_deg
        lon_b = np.concatenate([[lons[0]-resx/2], (lons[:-1]+lons[1:])/2, [lons[-1]+resx/2]])
        lat_b = np.concatenate([[lats[0]-resy/2], (lats[:-1]+lats[1:])/2, [lats[-1]+resy/2]])
    out_grid = {"lon": lons, "lat": lats}
    if is_rect: out_grid.update({"lon_b": lon_b, "lat_b": lat_b})
    key = df[["cell_id","lat_r","lon_r"]].sort_values(["lat_r","lon_r"]).reset_index(drop=True)
    return out_grid, key, is_rect

# ===== Chuẩn hóa =====
def _as_list(x): return list(x) if isinstance(x, (list, tuple, range, np.ndarray)) else [int(x)]
def normalize_years_months(years, months):
    ys = [int(v) for v in _as_list(years)]
    ms = [int(v) for v in _as_list(months)]
    if min(ys)<1900: raise ValueError("years phải >=1900")
    if any((m<1 or m>12) for m in ms): raise ValueError("months trong 1..12")
    return ys, ms
def normalize_dataset_name(name: str) -> str:
    s = name.strip().lower().replace("_","-").replace(" ","-")
    if "land" in s and "era" in s: return "era5-land"
    if "era" in s: return "era5"
    raise ValueError("dataset phải là 'ERA5-Land' hoặc 'ERA5'.")

# ===== Download từ CDS =====
def download_era5l_t2m(gdf_basin, years, months, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    area = basin_bbox_area(gdf_basin); c = Client()
    months = [f"{int(m):02d}" for m in months]
    for y in years:
        ydir = out_dir / f"{int(y)}"; ydir.mkdir(exist_ok=True, parents=True)
        for m in months:
            tgt = ydir / f"era5land_t2m_hourly_{y}{m}.nc"
            if tgt.exists() and tgt.stat().st_size>1024: print("Đã có:", tgt.name); continue
            req = {
                "variable": ["2m_temperature"], "year": str(y), "month": m,
                "day": [f"{d:02d}" for d in range(1,32)],
                "time": [f"{h:02d}:00" for h in range(24)],
                "area": area, "format": "netcdf",
            }
            print(f"Tải ERA5-Land T2m {y}-{m} → {tgt.name}"); c.retrieve("reanalysis-era5-land", req, str(tgt))

def download_era5_t2m(gdf_basin, years, months, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    area = basin_bbox_area(gdf_basin); c = Client()
    months = [f"{int(m):02d}" for m in months]
    for y in years:
        ydir = out_dir / f"{int(y)}"; ydir.mkdir(exist_ok=True, parents=True)
        for m in months:
            tgt = ydir / f"era5_t2m_hourly_{y}{m}.nc"
            if tgt.exists() and tgt.stat().st_size>1024: print("Đã có:", tgt.name); continue
            req = {
                "product_type": "reanalysis",
                "variable": ["2m_temperature"], "year": str(y), "month": m,
                "day": [f"{d:02d}" for d in range(1,32)],
                "time": [f"{h:02d}:00" for h in range(24)],
                "area": area, "format": "netcdf",
            }
            print(f"Tải ERA5 T2m {y}-{m} → {tgt.name}"); c.retrieve("reanalysis-era5-single-levels", req, str(tgt))

# ===== Safe open =====
def _peek_header(path: Path, nbytes: int = 64) -> bytes:
    try:
        with open(path, "rb") as f: return f.read(nbytes)
    except Exception: return b""
def _is_likely_netcdf(header: bytes) -> bool:
    return (b"CDF" in header[:4]) or (b"\x89HDF" in header[:8])
def ensure_netcdf_unpacked(path: Path) -> Path:
    hdr = _peek_header(path, 8)
    if hdr.startswith(b"PK\x03\x04"):
        import zipfile
        zpath = path if path.suffix.lower()==".zip" else path.with_suffix(path.suffix+".zip")
        if path != zpath:
            try: path.rename(zpath)
            except Exception: zpath = path
        with zipfile.ZipFile(zpath,"r") as zf:
            nc_members = [zi for zi in zf.infolist() if zi.filename.lower().endswith(".nc")]
            if not nc_members: raise RuntimeError(f"ZIP không chứa .nc: {zpath}")
            member = max(nc_members, key=lambda z: z.file_size)
            out_nc = path.parent / Path(member.filename).name
            zf.extract(member, path.parent)
            target_nc = path.with_suffix(".nc")
            try:
                if out_nc != target_nc:
                    if target_nc.exists(): target_nc.unlink()
                    out_nc.replace(target_nc); out_nc = target_nc
            except Exception: pass
        return out_nc
    return path
def open_mfdataset_safe_nc(paths: List[str], chunks: dict) -> xr.Dataset:
    try: return xr.open_mfdataset(paths, combine="by_coords", chunks=chunks, engine="netcdf4")
    except Exception as e1:
        try: return xr.open_mfdataset(paths, combine="by_coords", chunks=chunks, engine="h5netcdf")
        except Exception as e2:
            p0 = Path(paths[0]); hdr = _peek_header(p0,64); size = p0.stat().st_size if p0.exists() else -1
            raise RuntimeError(
                f"Không mở được NetCDF: {p0}\n  Size: {size}\n  Header: {hdr!r}\n  IsNetCDF? {_is_likely_netcdf(hdr)}\n"
                f"netcdf4 err: {e1}\n h5netcdf err: {e2}"
            )

# ===== Đọc & tiền xử lý =====
def ensure_lonlat_names(ds: xr.Dataset) -> xr.Dataset:
    ren = {}
    if "longitude" in ds.coords: ren["longitude"] = "lon"
    if "latitude"  in ds.coords: ren["latitude"]  = "lat"
    return ds.rename(ren)
def sort_by_latlon(obj: xr.Dataset | xr.DataArray):
    obj = obj.sortby("lat")
    if "lon" in obj.coords:
        lon2 = ((obj["lon"]+180)%360)-180
        if not np.allclose(lon2, obj["lon"].values):
            obj = obj.assign_coords(lon=lon2).sortby("lon")
    return obj
def open_hourly_t2m_nc(raw_dir: Path, years: Iterable[int]) -> xr.DataArray:
    files = []
    for y in years: files += list((Path(raw_dir)/f"{int(y)}").glob("*.nc"))
    if not files: raise FileNotFoundError(f"Không thấy *.nc trong {raw_dir}/YYYY")
    files = [ensure_netcdf_unpacked(Path(p)) for p in sorted(files)]
    ds = open_mfdataset_safe_nc([str(p) for p in files], chunks={"time":24*31})
    ds = ensure_lonlat_names(ds)
    var = "t2m" if "t2m" in ds.data_vars else ("2m_temperature" if "2m_temperature" in ds.data_vars else None)
    if var is None: raise KeyError("Thiếu biến t2m/2m_temperature.")
    da = sort_by_latlon(ds[var]); da.attrs.update({"units":"K","long_name":"2 m air temperature hourly"})
    return da
def _find_time_coord_name(da: xr.DataArray) -> str:
    for nm in ["time","valid_time","date","datetime"]:
        if (nm in da.coords) or (nm in da.dims): return nm
    for nm,c in da.coords.items():
        if getattr(c,"ndim",0)==1 and np.issubdtype(c.dtype, np.datetime64): return nm
    for nm in da.dims:
        try:
            if np.issubdtype(da.coords[nm].dtype, np.datetime64): return nm
        except Exception: pass
    raise KeyError("Không tìm thấy trục thời gian.")
def clip_to_basin(da: xr.DataArray, basin_gdf) -> xr.DataArray:
    da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False).rio.write_crs(4326)
    return da.rio.clip(basin_gdf.geometry, basin_gdf.crs, drop=True)
def hourly_to_daily_tmin_tmax(da_hourly_K: xr.DataArray, tz_offset_hours: int = 0) -> xr.Dataset:
    tname = _find_time_coord_name(da_hourly_K)
    t = pd.to_datetime(da_hourly_K[tname].values) + pd.Timedelta(hours=tz_offset_hours)
    da2 = da_hourly_K.assign_coords({tname:(tname,t)}); 
    if tname!="time": da2 = da2.rename({tname:"time"})
    daC = (da2 - 273.15).astype("float32")
    tmax = daC.resample(time="1D", label="left", closed="left").max()
    tmin = daC.resample(time="1D", label="left", closed="left").min()
    tmax.name="tmax"; tmin.name="tmin"
    for v in (tmax,tmin): v.attrs.update({"units":"degC"})
    return xr.Dataset({"tmax":tmax,"tmin":tmin})

# ===== Regrid =====
def make_src_grid_from_da(da: xr.DataArray) -> Dict[str,np.ndarray]:
    def _edges(vals): vals=np.asarray(vals); d=np.median(np.diff(vals)); return np.concatenate([[vals[0]-d/2],(vals[:-1]+vals[1:])/2,[vals[-1]+d/2]])
    return {"lat":da["lat"].values,"lon":da["lon"].values,
            "lat_b":_edges(da["lat"].values),"lon_b":_edges(da["lon"].values)}
def regrid_conservative_to_rect_vic(da: xr.DataArray, out_grid: Dict[str,np.ndarray], weights_path: Path) -> xr.DataArray:
    import xesmf as xe
    src_grid = make_src_grid_from_da(da)
    rg = xe.Regridder(src_grid, out_grid, method="conservative_normed",
                      filename=str(weights_path), reuse_weights=True, periodic=False)
    out = rg(da).rename({"lat":"y","lon":"x"}); out.attrs.update(da.attrs); return out
def regrid_bilinear_locstream_to_cells(da: xr.DataArray, key_table: pd.DataFrame,
                                       reuse_weights_path: Optional[Path]=None) -> xr.DataArray:
    import xesmf as xe
    tgt = {"lon": key_table["lon_r"].values, "lat": key_table["lat_r"].values}
    rg = xe.Regridder(da, tgt, method="bilinear", locstream_out=True,
                      filename=str(reuse_weights_path) if reuse_weights_path else None,
                      reuse_weights=bool(reuse_weights_path))
    out = rg(da).rename({"points":"cell"})
    out = out.assign_coords(cell=("cell", key_table["cell_id"].values)); out.attrs.update(da.attrs)
    return out
def interp_grid_xarray(da: xr.DataArray, out_grid: Dict[str,np.ndarray]) -> xr.DataArray:
    out = da.interp(lon=("x", out_grid["lon"]), lat=("y", out_grid["lat"]), method="linear").rename({"lat":"y","lon":"x"})
    out.attrs.update(da.attrs); return out
def interp_cells_xarray(da: xr.DataArray, key_table: pd.DataFrame) -> xr.DataArray:
    out = da.interp(lon=("cell", key_table["lon_r"].values),
                    lat=("cell", key_table["lat_r"].values), method="linear").transpose("time","cell")
    out = out.assign_coords(cell=("cell", key_table["cell_id"].values)); out.attrs.update(da.attrs); return out
def grid_to_cell_1d(da_grid: xr.DataArray, key_table: pd.DataFrame) -> xr.DataArray:
    xs = da_grid["x"].values.round(6); ys = da_grid["y"].values.round(6)
    XX,YY = np.meshgrid(xs, ys)
    idx = pd.DataFrame({"lat_r":YY.ravel(),"lon_r":XX.ravel(),
                        "i":np.repeat(np.arange(len(ys)),len(xs)),
                        "j":np.tile(np.arange(len(xs)),len(ys))})
    merged = key_table.merge(idx, on=["lat_r","lon_r"], how="left")
    if merged["i"].isna().any(): raise ValueError("Một số cell VIC không khớp lưới regrid.")
    merged = merged.sort_values("cell_id").reset_index(drop=True)
    sel = da_grid.isel(y=xr.DataArray(merged["i"].values,dims="cell"),
                       x=xr.DataArray(merged["j"].values,dims="cell")).transpose("time","cell")
    sel = sel.assign_coords(cell=("cell", merged["cell_id"].values)); sel.attrs.update(da_grid.attrs); return sel

# ===== Ghi file & QC =====
def to_netcdf(ds: xr.Dataset | xr.DataArray, out_path: Path):
    if isinstance(ds, xr.DataArray): ds = ds.to_dataset(name=ds.name)
    enc = {v: {"zlib":True, "complevel":4, "_FillValue": np.float32(-9999)} for v in ds.data_vars}
    ds.to_netcdf(out_path, encoding=enc); print("Đã lưu:", out_path)

def to_ascii_vicres_tair_cell(ds_cell: xr.Dataset, out_dir: Path, float_fmt="%.2f"):
    out_dir.mkdir(parents=True, exist_ok=True)
    t = pd.to_datetime(ds_cell["time"].values); ymd = np.c_[t.year, t.month, t.day]
    for var in ("tmin","tmax"):
        A = ds_cell[var].fillna(-9999).values
        for k, cid in enumerate(ds_cell["cell"].values):
            df = pd.DataFrame(np.c_[ymd, A[:,k]], columns=["YEAR","MONTH","DAY", var.upper()])
            fp = out_dir / f"{var.upper()}_cell_{int(cid)}.txt"
            df.to_csv(fp, sep="\t", index=False, float_format=float_fmt)
    print("Đã lưu ASCII Tmin/Tmax tại:", out_dir)

def _common_color_limits_temp(da: xr.DataArray, fixed: Optional[Tuple[float,float]]):
    if fixed is not None: return float(fixed[0]), float(fixed[1])
    v = np.nanpercentile(da.values, [2,98]); return float(v[0]), float(v[1])

def _imshow_like(da2d: xr.DataArray, ax, cmap, vmin, vmax, basin_gdf, label, title):
    def _edges(vals):
        vals = np.asarray(vals); d = np.median(np.diff(vals))
        return np.concatenate([[vals[0]-d/2], (vals[:-1]+vals[1:])/2, [vals[-1]+d/2]])
    xname = "lon" if "lon" in da2d.coords else ("x" if "x" in da2d.coords else list(da2d.dims)[-1])
    yname = "lat" if "lat" in da2d.coords else ("y" if "y" in da2d.coords else list(da2d.dims)[-2])
    xe = _edges(da2d[xname].values); ye = _edges(da2d[yname].values)
    im = ax.imshow(da2d.transpose(yname,xname).values, origin="lower",
                   extent=[xe[0],xe[-1],ye[0],ye[-1]],
                   interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax); cbar.set_label(label)
    try: basin_gdf.boundary.plot(ax=ax, edgecolor="crimson", linewidth=1.0)
    except Exception: pass
    ax.set_title(title); ax.set_xlabel("lon"); ax.set_ylabel("lat")

def qc_time_series_tmin_tmax(ds_daily: xr.Dataset, basin_name: str, figs_dir: Path,
                             ds_tag: str, show_inline=True) -> Path:
    figs_dir.mkdir(parents=True, exist_ok=True)
    dims = set(ds_daily["tmax"].dims) - {"time"}
    ts_max = ds_daily["tmax"].mean(dim=list(dims)).to_pandas()
    ts_min = ds_daily["tmin"].mean(dim=list(dims)).to_pandas()
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(ts_max.index, ts_max.values, color="#d62728", lw=1.6, label="Tmax")
    ax.plot(ts_min.index, ts_min.values, color="#1f77b4", lw=1.6, label="Tmin")
    ax.set_title(f"{basin_name} – Basin mean Tmin/Tmax [°C] – {ds_tag}")
    ax.set_ylabel("°C"); ax.grid(True, alpha=0.3); ax.legend()
    loc = mdates.AutoDateLocator(minticks=3, maxticks=7)
    ax.xaxis.set_major_locator(loc); ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    fig.tight_layout()
    out = figs_dir / f"qc_tair_{ds_tag}_ts.png"
    fig.savefig(out, dpi=150)
    if show_inline: plt.show()
    else: plt.close(fig)
    print("Đã lưu:", out); return out

def qc_maps_temperature(ds_daily: xr.Dataset, basin_gdf, basin_name: str,
                        figs_dir: Path, ds_tag: str, show_inline=True,
                        tmax_clim=TMAX_CLIM_DEFAULT, tmin_clim=TMIN_CLIM_DEFAULT,
                        on_vic_grid=False) -> Path:
    figs_dir.mkdir(parents=True, exist_ok=True)
    mean_tmax = ds_daily["tmax"].mean(dim="time")
    mean_tmin = ds_daily["tmin"].mean(dim="time")
    vminX, vmaxX = _common_color_limits_temp(mean_tmax, tmax_clim)
    vminN, vmaxN = _common_color_limits_temp(mean_tmin, tmin_clim)
    fig, ax = plt.subplots(1,2, figsize=(12,4.6), constrained_layout=True)
    grid_name = "VIC 0.05°" if on_vic_grid else "Source"
    _imshow_like(mean_tmax, ax[0], CMAP_T, vminX, vmaxX, basin_gdf, "Tmax [°C]",
                 f"{basin_name} – Tmax mean ({grid_name}) – {ds_tag}")
    _imshow_like(mean_tmin, ax[1], CMAP_T, vminN, vmaxN, basin_gdf, "Tmin [°C]",
                 f"{basin_name} – Tmin mean ({grid_name}) – {ds_tag}")
    out = figs_dir / f"qc_tair_{ds_tag}_{'map_vic05' if on_vic_grid else 'map_src'}.png"
    fig.savefig(out, dpi=150)
    if show_inline: plt.show()
    else: plt.close(fig)
    print("Đã lưu:", out); return out

# ===== PIPELINE CHÍNH =====
def run_era5_temperature_to_vicres(
    root: str | Path = r"D:\Python4_DaRiver",
    dataset: str = "ERA5-Land",
    years: Union[Iterable[int], int] = (2020,),
    months: Union[Iterable[int], int] = (1,),
    tz_offset_hours: int = 7,
    vic_grid_csv: Optional[str | Path] = None,
    vic_grid_template_nc: Optional[str | Path] = None,
    basin_vector: Optional[str | Path] = None,
    export_ascii: bool = True,
    backend: str = "auto",
    show_plots: bool = True
) -> Dict[str, Path]:
    ROOT = Path(root); DIR = default_dirs(ROOT)
    ds_name = normalize_dataset_name(dataset); Y,M = normalize_years_months(years, months)

    basin_fp = Path(basin_vector) if basin_vector else \
        find_basin_vector_multi(DIR["basin_data"], DIR["basin_inputs"])
    basin = read_basin(basin_fp); basin_name = Path(basin_fp).stem

    vic_csv = Path(vic_grid_csv) if vic_grid_csv else find_vic_grid_csv(DIR["grids"])
    tpl_nc = Path(vic_grid_template_nc) if vic_grid_template_nc else find_vic_template_nc(DIR["grids"])
    out_grid, key_table, is_rect = build_vic_grid(vic_csv, tpl_nc, default_res_deg=0.05)

    print(f"[INFO] Basin: {basin_fp}")
    print(f"[INFO] VIC CSV: {vic_csv}")
    print(f"[INFO] VIC template: {tpl_nc if tpl_nc else 'None'}")
    print(f"[INFO] Lưới VIC đều: {is_rect}")
    print(f"[INFO] Dataset: {ds_name}  Years: {Y}  Months: {M}")

    # Download + open hourly T2m
    if ds_name=="era5-land":
        download_era5l_t2m(basin, Y, M, DIR["raw_era5l"])
        da_hrK = open_hourly_t2m_nc(DIR["raw_era5l"], Y)
    else:
        download_era5_t2m(basin, Y, M, DIR["raw_era5"])
        da_hrK = open_hourly_t2m_nc(DIR["raw_era5"], Y)

    # Clip + aggregate theo múi giờ
    da_hr_clip = clip_to_basin(da_hrK, basin)
    ds_day = hourly_to_daily_tmin_tmax(da_hr_clip, tz_offset_hours=tz_offset_hours)

    tag = f"{ds_name}_{min(Y)}{min(M):02d}_{max(Y)}{max(M):02d}"

    # QC trước regrid
    fig_ts = qc_time_series_tmin_tmax(ds_day, basin_name, DIR["figs_tair"], ds_tag=tag, show_inline=show_plots)
    map_src = qc_maps_temperature(ds_day, basin, basin_name, DIR["figs_tair"],
                                  ds_tag=tag, show_inline=show_plots, on_vic_grid=False)

    # Regrid → VIC
    bk = pick_backend(backend); print(f"[INFO] Backend regrid: {bk}")
    if is_rect:
        if bk=="xesmf" and all(k in out_grid for k in ("lon_b","lat_b")):
            wfile = DIR["weights"] / f"wgt_{tag}_to_vic05.nc"
            tmax_grid = regrid_conservative_to_rect_vic(ds_day["tmax"], out_grid, wfile)
            tmin_grid = regrid_conservative_to_rect_vic(ds_day["tmin"], out_grid, wfile)
        else:
            tmax_grid = interp_grid_xarray(ds_day["tmax"], out_grid)
            tmin_grid = interp_grid_xarray(ds_day["tmin"], out_grid)
        tmax_cell = grid_to_cell_1d(tmax_grid, key_table)
        tmin_cell = grid_to_cell_1d(tmin_grid, key_table)
    else:
        if bk=="xesmf":
            wfile = DIR["weights"] / f"wgt_{tag}_locstream.nc"
            tmax_cell = regrid_bilinear_locstream_to_cells(ds_day["tmax"], key_table, reuse_weights_path=wfile)
            tmin_cell = regrid_bilinear_locstream_to_cells(ds_day["tmin"], key_table, reuse_weights_path=wfile)
        else:
            tmax_cell = interp_cells_xarray(ds_day["tmax"], key_table)
            tmin_cell = interp_cells_xarray(ds_day["tmin"], key_table)
        tmax_grid = tmin_grid = None

    # QC sau regrid (nếu có grid)
    out: Dict[str, Path] = {"fig_ts": Path(fig_ts), "map_src": Path(map_src)}
    if tmax_grid is not None:
        ds_grid = xr.Dataset({"tmax":tmax_grid, "tmin":tmin_grid})
        map_vic = qc_maps_temperature(ds_grid, basin, basin_name, DIR["figs_tair"],
                                      ds_tag=tag, show_inline=show_plots, on_vic_grid=True)
        out["map_vic05"] = Path(map_vic)

    # Lưu NetCDF
    if tmax_grid is not None:
        ds_grid = xr.Dataset({"tmax":tmax_grid, "tmin":tmin_grid})
        ds_grid.attrs.update({"dataset":ds_name,"units":"degC","aggregation":"daily Tmin/Tmax",
                              "regrid_backend":bk,"vic_grid_csv":Path(vic_csv).name,
                              "vic_template_nc": Path(tpl_nc).name if tpl_nc else "None",
                              "basin_vector": Path(basin_fp).name})
        nc_grid = DIR["proc_tair"] / f"tair_{tag}_vic05_grid.nc"
        to_netcdf(ds_grid, nc_grid); out["nc_grid"] = nc_grid

    ds_cell = xr.Dataset({"tmax":tmax_cell,"tmin":tmin_cell})
    ds_cell.attrs.update({"dataset":ds_name,"units":"degC","aggregation":"daily Tmin/Tmax",
                          "regrid_backend":bk,"vic_grid_csv":Path(vic_csv).name,
                          "vic_template_nc": Path(tpl_nc).name if tpl_nc else "None",
                          "basin_vector": Path(basin_fp).name})
    nc_cell = DIR["proc_tair"] / f"tair_{tag}_vic05_cell.nc"
    to_netcdf(ds_cell, nc_cell); out["nc_cell"] = nc_cell

    # ASCII (tách theo nguồn)
    if export_ascii:
        ascii_dir = DIR["forc_ascii_tair"] / ds_name.replace("-","")
        to_ascii_vicres_tair_cell(ds_cell, ascii_dir); out["ascii_dir"] = ascii_dir

    print("[DONE] ERA5 Temperature → VIC-Res (Tmin/Tmax).")
    return out

if __name__ == "__main__":
    run_era5_temperature_to_vicres(
        root=r"D:\Python4_DaRiver",
        dataset="ERA5-Land", years=[2020], months=[1],
        tz_offset_hours=7, backend="auto", export_ascii=True, show_plots=False
    )
