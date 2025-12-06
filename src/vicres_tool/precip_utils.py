# -*- coding: utf-8 -*-
"""
VIC-Res forcing helpers (CHIRPS & IMERG only)

- CHIRPS daily (đọc file năm local: data/raw/CHIRPS/chirps-v2.0.<YYYY>.days_p05.nc)
- IMERG V07 Final Daily (nc4; tải tự động nếu thiếu bằng Earthdata token)

Thư mục (init_paths(ROOT) tạo):
  ROOT/
    data/raw/{CHIRPS,IMERG}/
    outputs/forcings/{netcdf,ascii}/
    outputs/grids/
    env/earthdata_token.txt  # 1 dòng token (tuỳ chọn)

Các điểm nhấn:
- Hàm quick_qc_basin() vẽ 2 panel (time-series + spatial mean), dùng imshow(nearest)
  nên các ô lưới hiển thị rõ. Trục thời gian dùng ConciseDateFormatter để tránh đè chữ.
- Thang màu thống nhất cho tất cả nguồn: Blues, vmin/vmax mặc định (0, 30) mm/day.
  Có thể truyền clim_fixed=None để tự động theo p98 của dữ liệu.
- IMERG: hỗ trợ download="auto" (chỉ tải thiếu), và cleanup_raw=True để xoá các file raw
  sau khi xuất NetCDF/ASCII, tiết kiệm dung lượng ổ.
- Mở NetCDF robust: thử chuẩn, sau đó engine="netcdf4", rồi "h5netcdf". Lọc bỏ file
  hỏng/truncated bằng kiểm tra header HDF/CDF & kích thước tối thiểu.
"""

# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re
from pathlib import Path
from datetime import timedelta
from typing import Iterable, Sequence, List

import numpy as np
import xarray as xr
import requests
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

try:
    import geopandas as gpd  # type: ignore
except Exception:
    gpd = None  # type: ignore

xr.set_options(keep_attrs=True)

ROOT: Path | None = None
DIR: dict = {}
TARGET_GRID: dict = {}
CMAP = plt.get_cmap("Blues")
DEFAULT_CLIM = (0.0, 30.0)

import datetime as _dt
import numpy as _np

def _to_date(x) -> _dt.date:
    if isinstance(x, _dt.date):
        return x if not isinstance(x, _dt.datetime) else x.date()
    if isinstance(x, _np.datetime64):
        s = _np.datetime_as_string(x, unit="D")
        return _dt.date.fromisoformat(s)
    s = str(x).strip().replace("/", "-")
    try: return _dt.date.fromisoformat(s)
    except Exception: pass
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", s)
    if m: return _dt.date(*map(int, m.groups()))
    m = re.match(r"^(\d{4})(\d{2})(\d{2})$", s)
    if m: return _dt.date(*map(int, m.groups()))
    raise ValueError(f"Không nhận dạng được ngày: {x!r}")

def init_paths(root: str | Path) -> None:
    global ROOT, DIR
    ROOT = Path(root).resolve()
    DIR.clear()
    DIR["raw_chirps"] = ROOT / "data" / "raw" / "CHIRPS"
    DIR["raw_imerg"]  = ROOT / "data" / "raw" / "IMERG"
    DIR["forc_nc"]    = ROOT / "outputs" / "forcings" / "netcdf"
    DIR["forc_ascii"] = ROOT / "outputs" / "forcings" / "ascii"
    DIR["grids"]      = ROOT / "outputs" / "grids"
    for k in ("raw_chirps","raw_imerg","forc_nc","forc_ascii","grids"):
        DIR[k].mkdir(parents=True, exist_ok=True)
    print(f"[OK] Paths → {DIR['forc_nc']} | {DIR['forc_ascii']}")

def set_target_grid(grid_nc: str | Path):
    ds = _open_dataset_fallback(grid_nc)
    ds = _normalize_coords_hard(ds)
    TARGET_GRID.clear()
    TARGET_GRID["lat"] = ds["lat"].values
    TARGET_GRID["lon"] = ds["lon"].values
    if "mask" in ds:
        TARGET_GRID["mask"] = np.asarray(ds["mask"].values, bool)
    elif "frac" in ds:
        TARGET_GRID["mask"] = np.asarray(ds["frac"].values > 0, bool)
        TARGET_GRID["frac"] = np.asarray(ds["frac"].values)
    else:
        TARGET_GRID["mask"] = np.ones((ds.sizes["lat"], ds.sizes["lon"]), bool)
    return ds

# ---- NetCDF helpers (giữ nguyên như bản của bạn) ----
def _open_dataset_fallback(path: str | Path, **kw) -> xr.Dataset:
    try: return xr.open_dataset(path, **kw)
    except Exception:
        try: return xr.open_dataset(path, **{**kw, "engine": "netcdf4"})
        except Exception: return xr.open_dataset(path, **{**kw, "engine": "h5netcdf"})

def _open_mfdataset_fallback(paths: Iterable[str | Path], **kw) -> xr.Dataset:
    paths = [str(p) for p in paths]; kw.setdefault("combine", "by_coords")
    try: return xr.open_mfdataset(paths, **kw)
    except Exception:
        try: return xr.open_mfdataset(paths, **{**kw, "engine":"netcdf4"})
        except Exception: return xr.open_mfdataset(paths, **{**kw, "engine":"h5netcdf"})

def _is_valid_netcdf(p: Path) -> bool:
    try:
        if not p.exists() or p.stat().st_size < 1024: return False
        with open(p, "rb") as f: head = f.read(8)
        return (b"CDF" in head) or (b"HDF" in head)
    except Exception: return False

def _filter_valid_nc(paths: Iterable[Path]) -> list[Path]:
    return [p for p in map(Path, paths) if _is_valid_netcdf(p)]

def _rename_if_present(ds: xr.Dataset, mapping: dict) -> xr.Dataset:
    found = {k: v for k, v in mapping.items() if (k in ds.dims or k in ds.coords)}
    return ds.rename(found) if found else ds

def _maybe_swap_lat_lon(ds: xr.Dataset) -> xr.Dataset:
    if "lat" in ds and "lon" in ds:
        latv = np.asarray(ds["lat"].values); lonv = np.asarray(ds["lon"].values)
        in_lat = (np.nanmin(latv) >= -90) and (np.nanmax(latv) <=  90)
        in_lat2= (np.nanmin(lonv) >= -90) and (np.nanmax(lonv) <=  90)
        if (not in_lat) and in_lat2:
            ds = ds.rename({"lat":"__lon__", "lon":"__lat__"}).rename({"__lat__":"lat", "__lon__":"lon"})
    return ds

def _force_dims_order_lat_lon(ds: xr.Dataset) -> xr.Dataset:
    for v in list(ds.data_vars):
        dims = ds[v].dims
        if ("lat" in dims) and ("lon" in dims):
            others = tuple(d for d in dims if d not in ("lat","lon"))
            ds[v] = ds[v].transpose(*others, "lat", "lon")
    return ds

def _normalize_coords_hard(ds: xr.Dataset) -> xr.Dataset:
    ds = _rename_if_present(ds, {"latitude":"lat","Latitude":"lat","y":"lat"})
    ds = _rename_if_present(ds, {"longitude":"lon","Longitude":"lon","x":"lon"})
    ds = _maybe_swap_lat_lon(ds)
    if "lat" in ds:
        if ds["lat"].ndim > 1: ds = ds.assign_coords(lat=ds["lat"].isel({ds["lat"].dims[-1]:0}, drop=True))
        ds = ds.sortby("lat")
    if "lon" in ds:
        if ds["lon"].ndim > 1: ds = ds.assign_coords(lon=ds["lon"].isel({ds["lon"].dims[0]:0}, drop=True))
        if float(ds["lon"].max()) > 180: ds = ds.assign_coords(lon=((ds["lon"]+180.0) % 360.0) - 180.0)
        ds = ds.sortby("lon")
    return _force_dims_order_lat_lon(ds)

def _standardize_pr_daily(ds: xr.Dataset) -> xr.Dataset:
    ds = _normalize_coords_hard(ds)
    pick = None
    for v in sorted(list(ds.data_vars), key=str.lower):
        if v.lower() in ("pr","precip","precipitation","precipitationcal"): pick = v; break
    if pick is None:
        for v in ds.data_vars:
            if ds[v].ndim >= 2: pick = v; break
    if pick is None: raise KeyError("Không tìm thấy biến mưa trong dataset!")
    pr = ds[pick].rename("pr").assign_attrs(units="mm/day", long_name="daily precipitation")
    return pr.to_dataset()

def _bbox_from_target(pad=0.25):
    lat, lon = TARGET_GRID["lat"], TARGET_GRID["lon"]
    return float(lat.min()-pad), float(lat.max()+pad), float(lon.min()-pad), float(lon.max()+pad)

def _crop_to_bbox_safe(ds: xr.Dataset, pad=0.25) -> xr.Dataset:
    ds = _normalize_coords_hard(ds)
    lat0, lat1, lon0, lon1 = _bbox_from_target(pad)
    return ds.sel(lat=slice(lat0, lat1), lon=slice(lon0, lon1))

def _regrid_to_vic(ds: xr.Dataset, method="linear") -> xr.Dataset:
    ds = _normalize_coords_hard(ds)
    return ds.interp(lat=("lat", TARGET_GRID["lat"]), lon=("lon", TARGET_GRID["lon"]), method=method)

def attach_mask_frac(ds: xr.Dataset) -> xr.Dataset:
    ds = _force_dims_order_lat_lon(ds)
    ds["mask"] = xr.DataArray(
        TARGET_GRID.get("mask", np.ones((ds.sizes["lat"], ds.sizes["lon"]), bool)),
        coords={"lat": ds["lat"], "lon": ds["lon"]}, dims=("lat","lon")
    )
    if "frac" in TARGET_GRID:
        ds["frac"] = xr.DataArray(TARGET_GRID["frac"], coords={"lat": ds["lat"], "lon": ds["lon"]}, dims=("lat","lon"))
    return ds

def export_netcdf(ds: xr.Dataset, out_dir: str | Path, prefix: str) -> Path:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    t0 = str(np.datetime_as_string(ds.time.values[0], unit="D")) if "time" in ds else "NA"
    t1 = str(np.datetime_as_string(ds.time.values[-1], unit="D")) if "time" in ds else "NA"
    fn = out_dir / f"{prefix}_{t0.replace('-','')}_{t1.replace('-','')}.nc"
    comp = dict(zlib=True, complevel=4, dtype="float32")
    enc = {v: comp for v in ds.data_vars}
    ds.to_netcdf(fn, encoding=enc)
    print("[OK] NetCDF:", fn)
    return fn

def export_ascii_for_vicres(ds: xr.Dataset, out_dir: str | Path, basin_code="DaRiver") -> int:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pr = ds["pr"]
    mask = TARGET_GRID.get("mask", np.ones((pr.sizes["lat"], pr.sizes["lon"]), bool))
    nfile, cid = 0, 1
    for j in range(pr.sizes["lat"]):
        for i in range(pr.sizes["lon"]):
            if not mask[j, i]: continue
            arr = pr.isel(lat=j, lon=i).values.astype("float32")
            np.savetxt(out_dir / f"{basin_code}_cell_{cid:05d}.txt", arr, fmt="%.4f")
            cid += 1; nfile += 1
    print(f"[OK] ASCII : {nfile} files -> {out_dir}")
    return nfile

# ---------------------- QC nhanh (giữ nguyên) ----------------------
# ... (đoạn quick_qc_basin của bạn giữ nguyên ở đây) ...

# ---------------------- CHIRPS -------------------------
def _chirps_base_url(res="p05") -> str:
    # CHIRPS daily global NetCDF — đường dẫn công khai ổn định
    return f"https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/{res}/"

def _list_chirps_year_files(t0: str, t1: str, res="p05") -> List[Path]:
    y0 = _to_date(t0).year; y1 = _to_date(t1).year
    out = []
    for y in range(y0, y1 + 1):
        f = DIR["raw_chirps"] / f"chirps-v2.0.{y}.days_{res}.nc"
        if f.exists() and f.stat().st_size > 0:
            out.append(f)
    return out

def _download_chirps_years(t0: str, t1: str, res="p05") -> int:
    y0 = _to_date(t0).year; y1 = _to_date(t1).year
    raw = DIR["raw_chirps"]; raw.mkdir(parents=True, exist_ok=True)
    base = _chirps_base_url(res)
    got = 0
    for y in range(y0, y1 + 1):
        fn = raw / f"chirps-v2.0.{y}.days_{res}.nc"
        if fn.exists() and fn.stat().st_size > 1024: continue
        url = base + f"chirps-v2.0.{y}.days_{res}.nc"
        r = requests.get(url, stream=True, timeout=180)
        if r.status_code == 200:
            with open(fn, "wb") as f:
                for chunk in r.iter_content(1<<16):
                    if chunk: f.write(chunk)
            got += 1
        else:
            print(f"[WARN] CHIRPS {y} {res}: HTTP {r.status_code}")
    return got

def _open_chirps_daily(files: Sequence[Path]) -> xr.Dataset:
    ds = _open_mfdataset_fallback(files, chunks={"time": 60})
    ds = _standardize_pr_daily(ds)
    return ds[["pr"]]

def run_chirps_pipeline(t0: str, t1: str, res="p05", method="linear",
                        save_ascii=True, download="auto", cleanup_raw=False,
                        write_netcdf=True) -> xr.Dataset:
    # 1) tải nếu thiếu
    years_needed = list(range(_to_date(t0).year, _to_date(t1).year + 1))
    files = _list_chirps_year_files(t0, t1, res=res)
    if download == "auto":
        if len(files) < len(years_needed):
            print("[INFO] CHIRPS thiếu file năm → tải bổ sung…")
            _download_chirps_years(t0, t1, res=res)
    elif download is True or str(download).lower() == "true":
        _download_chirps_years(t0, t1, res=res)

    files = _list_chirps_year_files(t0, t1, res=res)
    assert files, f"Không thấy CHIRPS ({res}) cho {t0}..{t1} ở {DIR['raw_chirps']}"

    # 2) xử lý
    ds = _open_chirps_daily(files).sel(time=slice(t0, t1))
    ds = _crop_to_bbox_safe(ds, pad=0.25)
    ds = _regrid_to_vic(ds, method=method)
    ds = attach_mask_frac(ds)

    if write_netcdf:
        export_netcdf(ds, DIR["forc_nc"], "precip_CHIRPS_daily_0.05deg_DaRiver")
    if save_ascii:
        export_ascii_for_vicres(ds, DIR["forc_ascii"], "DaRiver")

    # 3) dọn raw nếu muốn
    if cleanup_raw:
        y0 = _to_date(t0).year; y1 = _to_date(t1).year
        n_del = 0
        for y in range(y0, y1+1):
            f = DIR["raw_chirps"] / f"chirps-v2.0.{y}.days_{res}.nc"
            try:
                if f.exists(): f.unlink(); n_del += 1
            except Exception:
                pass
        if n_del: print(f"[CLEAN] Đã xoá {n_del} file CHIRPS raw ({y0}..{y1})")
    return ds

# ---------------------- IMERG (giữ nguyên + write_netcdf) -------------------------
def _earthdata_headers() -> dict:
    tok = os.getenv("EARTHDATA_TOKEN", "").strip()
    if not tok and ROOT is not None:
        f = ROOT / "env" / "earthdata_token.txt"
        if f.exists(): tok = f.read_text(encoding="utf-8").strip()
    h = {"User-Agent": "vicres-utils/1.0"}
    if tok: h["Authorization"] = f"Bearer {tok}"
    return h

def check_earthdata(host="gpm1") -> int:
    base = f"https://{host}.gesdisc.eosdis.nasa.gov"
    url = base + "/data/GPM_L3/GPM_3IMERGDF.07/2020/01/3B-DAY.MS.MRG.3IMERG.20200101-S000000-E235959.V07B.nc4"
    try:
        r = requests.get(url, headers=_earthdata_headers(), timeout=30, stream=True, allow_redirects=True)
        print(f"[CHECK] {host}: {r.status_code} {r.reason}")
        return r.status_code
    except Exception as e:
        print("[CHECK]", host, "EXC ->", e)
        return -1

def _imerg_url(y: int, m: int, d: int, host: str) -> str:
    base = f"https://{host}.gesdisc.eosdis.nasa.gov"
    return f"{base}/data/GPM_L3/GPM_3IMERGDF.07/{y:04d}/{m:02d}/3B-DAY.MS.MRG.3IMERG.{y:04d}{m:02d}{d:02d}-S000000-E235959.V07B.nc4"

def _download_imerg_one(url: str, out_file: Path) -> str:
    try:
        r = requests.get(url, headers=_earthdata_headers(), timeout=180, stream=True)
        if r.status_code == 200:
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "wb") as f:
                for chunk in r.iter_content(1<<16):
                    if chunk: f.write(chunk)
            return "ok"
        return "skip" if r.status_code == 404 else "fail"
    except Exception:
        return "fail"

def _download_imerg_range(t0: str, t1: str, out_dir: Path, hosts=("gpm2","gpm1")) -> int:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    d0 = _to_date(t0); d1 = _to_date(t1)
    got = 0; cur = d0
    while cur <= d1:
        fn = out_dir / f"3B-DAY.MS.MRG.3IMERG.{cur:%Y%m%d}-S000000-E235959.V07B.nc4"
        if fn.exists() and fn.stat().st_size > 0:
            cur += timedelta(days=1); continue
        ok = False
        for h in hosts:
            stat = _download_imerg_one(_imerg_url(cur.year, cur.month, cur.day, h), fn)
            if stat in ("ok","skip"):
                ok = True; break
        if ok and fn.exists() and fn.stat().st_size > 0:
            got += 1
        cur += timedelta(days=1)
    return got

def _list_imerg_files_range(t0: str, t1: str, raw: Path) -> list[Path]:
    d0 = _to_date(t0); d1 = _to_date(t1)
    patt = re.compile(r"3B-DAY.*?(\d{8}).*?\.nc4$", re.I)
    out: list[Path] = []
    for f in sorted(raw.rglob("*.nc4")):
        m = patt.search(f.name)
        if not m: continue
        dt = _to_date(m.group(1))
        if d0 <= dt <= d1 and _is_valid_netcdf(f):
            out.append(f)
    return out

def run_imerg_pipeline(t0: str, t1: str, method: str = "linear",
                       save_ascii: bool = True, download: str | bool = "auto",
                       cleanup_raw: bool = False, write_netcdf: bool = True) -> xr.Dataset:
    raw = DIR["raw_imerg"]; raw.mkdir(parents=True, exist_ok=True)

    if download == "auto":
        d0 = _to_date(t0); d1 = _to_date(t1)
        need = (d1 - d0).days + 1
        have = len([p for p in _list_imerg_files_range(t0, t1, raw)])
        if have < need:
            print(f"[INFO] IMERG thiếu ~{need-have} ngày → tải bổ sung…")
            _download_imerg_range(t0, t1, raw, hosts=("gpm2","gpm1"))
    elif download is True:
        _download_imerg_range(t0, t1, raw, hosts=("gpm2","gpm1"))

    files = _list_imerg_files_range(t0, t1, raw)
    if not files:
        raise FileNotFoundError(f"Không thấy file IMERG hợp lệ trong {raw} cho {t0}..{t1}")

    ds = _open_mfdataset_fallback(files, chunks={"time": 30})
    ds = _standardize_pr_daily(ds).sel(time=slice(t0, t1))
    ds = _crop_to_bbox_safe(ds, pad=0.25)
    ds = _regrid_to_vic(ds, method=method)
    ds = attach_mask_frac(ds)

    if write_netcdf:
        export_netcdf(ds, DIR["forc_nc"], "precip_IMERG_daily_0.05deg_DaRiver")
    if save_ascii:
        export_ascii_for_vicres(ds, DIR["forc_ascii"], "DaRiver")

    if cleanup_raw:
        from pandas import date_range   # import nhẹ tại chỗ
        dates = set(date_range(t0, t1, freq="D").strftime("%Y%m%d").tolist())
        n_del = 0
        for f in raw.rglob("*.nc4"):
            m = re.search(r"(\d{8})", f.name)
            if m and m.group(1) in dates:
                try: f.unlink(); n_del += 1
                except Exception: pass
        print(f"[CLEAN] Đã xoá {n_del} file IMERG raw trong khoảng {t0}→{t1}")
    return ds
def quick_qc_basin(ds: xr.Dataset, title: str,
                   basin_vector: str | Path | None = None,
                   clim_fixed=(0, 30),
                   save_png: str | Path | None = None) -> Path | None:
    """Vẽ 2 panel: (trái) Basin mean time series; (phải) Spatial mean lat-lon."""
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    gdf = None
    if basin_vector and gpd is not None:
        try:
            gdf = gpd.read_file(basin_vector).to_crs(4326)
        except Exception:
            gdf = None

    pr = ds["pr"]  # (time, lat, lon)
    # mask/frac (nếu có) để lấy basin mean hợp lý
    w = None
    if "frac" in ds: w = np.asarray(ds["frac"].values, float)
    elif "mask" in ds: w = np.asarray(ds["mask"].values, float)
    if w is None: w = np.ones((pr.sizes["lat"], pr.sizes["lon"]), float)

    # Basin mean theo thời gian
    w2 = np.broadcast_to(w, pr.shape)  # (time, lat, lon) broadcast
    ts = (pr * w2).sum(dim=("lat","lon")) / np.maximum(w.sum(), 1.0)

    # Spatial mean (trung bình theo time)
    m2 = pr.mean(dim="time")
    # mask ngoài lưu vực về NaN (ưu tiên 'frac', nếu không có thì dùng 'mask')
    mask = None
    if "frac" in ds:
        mask = (ds["frac"] > 0)
    elif "mask" in ds:
        mask = (ds["mask"] > 0)
    if mask is not None:
        m2 = m2.where(mask)

    ax2 = plt.subplot(1,2,2)
    vmin, vmax = clim_fixed
    im = ax2.pcolormesh(m2["lon"].values, m2["lat"].values, m2.values,
                    shading="auto", vmin=vmin, vmax=vmax)

    # vẽ ranh lưu vực
    if gdf is not None:
        gdf.boundary.plot(ax=ax2, color="r", linewidth=0.8)
        # phóng to vừa khít lưu vực (có buffer nhỏ cho đẹp)
        xmin, ymin, xmax, ymax = gdf.total_bounds
        ax2.set_xlim(xmin-0.1, xmax+0.1)
        ax2.set_ylim(ymin-0.1, ymax+0.1)

    cb = plt.colorbar(im, ax=ax2); cb.set_label("pr [mm/day]")
    ax2.set_xlabel("lon"); ax2.set_ylabel("lat")
    ax2.set_title("Spatial mean [mm/day]")

    # Vẽ
    fig = plt.figure(figsize=(11,4.8))
    ax1 = plt.subplot(1,2,1)
    ax1.plot(ts["time"].values, ts.values, lw=1.2)
    ax1.set_title(f"{title} – Basin mean [mm/day]")
    ax1.set_ylabel("mm/day")
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(1,2,2)
    vmin, vmax = clim_fixed
    im = ax2.pcolormesh(m2["lon"].values, m2["lat"].values, m2.values,
                        shading="auto", vmin=vmin, vmax=vmax)
    if gdf is not None:
        gdf.boundary.plot(ax=ax2, color="r", linewidth=0.8)
    cb = plt.colorbar(im, ax=ax2); cb.set_label("pr [mm/day]")
    ax2.set_xlabel("lon"); ax2.set_ylabel("lat")
    ax2.set_title("Spatial mean [mm/day]")
    plt.tight_layout()

    if save_png:
        out = Path(save_png); out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=160)
        plt.close(fig)
        print("[OK] QC PNG:", out)
        return out
    return None
