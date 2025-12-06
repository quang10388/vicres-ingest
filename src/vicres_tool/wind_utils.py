# -*- coding: utf-8 -*-
"""
wind_utils.py — ERA5/ERA5-Land wind → VIC-Res forcing (ASCII)

- Hỗ trợ backend 'local' (đọc NetCDF tháng có sẵn) và 'cds' (tải từ ADS).
- Tải an toàn: ghi .part, thử mở bằng xarray; nếu không phải NetCDF thì
  thử coi như file ZIP, giải nén .nc bên trong; hỏng thì xoá và retry (backoff).
- Có thể giới hạn vùng tải CDS theo bbox lưu vực / lưới VIC (area).
- Giữ format ASCII cũ: mỗi cell một file, cột: year month day value (m/s).

Yêu cầu: xarray, numpy, pandas, và ít nhất một engine NetCDF:
- h5netcdf (khuyên dùng) hoặc netCDF4 hoặc scipy
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import time
import shutil
import zipfile

import numpy as np
import pandas as pd
import xarray as xr


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _open_ds_any(path: Path) -> xr.Dataset:
    """Mở NetCDF lần lượt với các engine an toàn."""
    errs = []
    for eng in ("h5netcdf", "netcdf4", "scipy"):
        try:
            return xr.open_dataset(path, engine=eng, chunks={})
        except Exception as e:  # pragma: no cover - chỉ log cho người dùng
            errs.append(f"{eng}: {e}")
    raise RuntimeError(f"Không mở được {path} bằng các engine: " + " | ".join(errs))


def _standardize_lonlat(ds: xr.Dataset) -> xr.Dataset:
    """Đưa tên toạ độ về lon/lat & đảm bảo tăng dần."""
    ren = {}
    if "longitude" in ds.coords:
        ren["longitude"] = "lon"
    if "latitude" in ds.coords:
        ren["latitude"] = "lat"
    ds = ds.rename(ren)

    if "lon" not in ds.coords or "lat" not in ds.coords:
        raise RuntimeError("Dataset không có toạ độ lon/lat.")

    # lon 0..360 → -180..180
    lon = ds["lon"].values
    if np.any(lon > 180.0):
        lon = ((lon + 180.0) % 360.0) - 180.0
        ds = ds.assign_coords(lon=lon).sortby("lon")

    # đảo lat nếu đang giảm dần
    if ds["lat"].values[0] > ds["lat"].values[-1]:
        ds = ds.reindex(lat=list(reversed(ds["lat"].values)))

    return ds


def _normalize_time_and_dims(ds: xr.Dataset) -> xr.Dataset:
    """
    Chuẩn hoá chiều thời gian + bỏ các chiều phụ (number, expver,...):
    - Mọi kiểu time/valid_time/forecast_time → chỉ còn 'time'
    - Bỏ các chiều phụ bằng cách lấy index 0 (ensemble member đầu).
    """
    # chuẩn hoá tên lon/lat trước cho chắc
    ds = _standardize_lonlat(ds)

    time_like = [d for d in ds.dims if d in ("time", "valid_time", "forecast_time")]
    if not time_like:
        raise RuntimeError(f"Dataset không có chiều thời gian; dims = {tuple(ds.dims)}")

    main = "time" if "time" in time_like else time_like[0]

    # Giữ lại 1 chiều thời gian, các chiều time-like khác lấy index 0 rồi drop
    for d in time_like:
        if d != main:
            ds = ds.isel({d: 0}, drop=True)

    if main != "time":
        ds = ds.rename({main: "time"})

    # Bỏ các chiều ensemble/phụ
    for extra in ("number", "expver", "realization", "ensemble", "member"):
        if extra in ds.dims:
            ds = ds.isel({extra: 0}, drop=True)

    return ds


def _detect_wind_vars(ds: xr.Dataset) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Tự tìm tên biến gió; trả (u_name, v_name, si_name).
    Hỗ trợ cả ERA5 và ERA5-Land.
    """
    names = {k.lower(): k for k in ds.data_vars}
    cand_u = ["u10", "10u", "u10n", "u_10", "u_10m", "10m_u_component_of_wind"]
    cand_v = ["v10", "10v", "v10n", "v_10", "v_10m", "10m_v_component_of_wind"]
    cand_s = ["si10", "wind10", "ff10", "wind"]

    u = next((names[c] for c in cand_u if c in names), None)
    v = next((names[c] for c in cand_v if c in names), None)
    s = next((names[c] for c in cand_s if c in names), None)
    return u, v, s


def _to_daily_local_time(da: xr.DataArray, tz_hours: int) -> xr.DataArray:
    """Shift theo múi giờ rồi lấy trung bình ngày."""
    if "time" not in da.dims:
        raise RuntimeError("Thiếu trục thời gian 'time'.")

    out = da
    if tz_hours:
        out = out.assign_coords(time=out["time"] + np.timedelta64(int(tz_hours), "h"))

    # Sau resample, time là datetime64[ns]
    out = out.resample(time="1D").mean(skipna=True)
    return out


def _interp_to_vic_grid(da: xr.DataArray, vic_tmpl_nc: Path) -> xr.DataArray:
    """Nội suy bilinear sang lưới VIC (template phải có lon/lat 1D)."""
    vt = _open_ds_any(vic_tmpl_nc)
    lon_name = "lon" if "lon" in vt.coords else ("longitude" if "longitude" in vt.coords else None)
    lat_name = "lat" if "lat" in vt.coords else ("latitude" if "latitude" in vt.coords else None)
    if lon_name is None or lat_name is None:
        raise RuntimeError("Template VIC thiếu lon/lat.")

    grid_lon = vt[lon_name].values
    grid_lat = vt[lat_name].values

    # Đưa DataArray → Dataset để tái dùng normalize_dims
    ds = da.to_dataset(name="wind")
    ds = _normalize_time_and_dims(ds)
    src = ds["wind"]  # dims giờ chỉ còn (time, lat, lon)

    out = src.interp(lon=grid_lon, lat=grid_lat, method="linear").transpose("time", "lat", "lon")
    return out


def _export_ascii_per_cell(outdir: Path, da_daily: xr.DataArray, var: str = "wind"):
    """Xuất ASCII: mỗi ô một file, 4 cột year month day value (m/s)."""
    _ensure_dir(outdir)

    # time phải là datetime64; nếu là timedelta64 thì chuyển sang mốc 1970-01-01
    tvals = da_daily.time.values
    if np.issubdtype(tvals.dtype, np.timedelta64):
        base = np.datetime64("1970-01-01")
        t = pd.to_datetime(base + tvals)
    else:
        t = pd.to_datetime(tvals)

    ymd = np.c_[t.year, t.month, t.day]
    nlat = da_daily.sizes["lat"]
    nlon = da_daily.sizes["lon"]

    for j in range(nlat):
        for i in range(nlon):
            vals = np.asarray(da_daily.isel(lat=j, lon=i).values, dtype=float).reshape(-1)
            arr = np.c_[ymd, vals]
            fn = outdir / f"WIND_cell_{j:03d}_{i:03d}.txt"
            np.savetxt(fn, arr, fmt=["%d", "%d", "%d", "%.4f"], delimiter=" ")


def _plot_vic_mean_map(
    da_daily_vic: xr.DataArray,
    basin_vector: Optional[str | Path],
    save_png: Path,
):
    """Vẽ bản đồ mean WIND trên lưới VIC, overlay biên lưu vực (nếu có)."""
    try:
        import matplotlib.pyplot as plt
        import geopandas as gpd
    except Exception as e:  # pragma: no cover - chỉ để không làm hỏng pipeline
        print("[WARN] Không vẽ được map WIND (thiếu matplotlib/geopandas):", e)
        return

    mean_da = da_daily_vic.mean(dim="time")
    lon = mean_da["lon"].values
    lat = mean_da["lat"].values
    X, Y = np.meshgrid(lon, lat)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.pcolormesh(X, Y, mean_da.values, shading="auto")
    if basin_vector is not None:
        try:
            g = gpd.read_file(basin_vector).to_crs(4326)
            g.boundary.plot(ax=ax, edgecolor="red", linewidth=0.8)
        except Exception as e:  # pragma: no cover
            print("[WARN] Không overlay được lưu vực lên map WIND:", e)

    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.set_title("Basin mean WIND [m/s]")
    fig.colorbar(im, ax=ax, label="m/s")
    fig.tight_layout()
    _ensure_dir(save_png.parent)
    fig.savefig(save_png, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------
# Downloader (CDS monthly → NetCDF, có kiểm tra/đổi tên + retry)
# ------------------------------------------------------------
def _cds_monthly_download_era5land_u10_v10(
    year: int,
    month: int,
    out_nc: Path,
    area: Optional[List[float]] = None,
    max_retry: int = 6,
    backoff0: int = 30,
) -> Path:
    """
    Tải ERA5-Land giờ trong tháng (u10/v10) thành 1 NetCDF.
    - Tải vào .part, mở thử bằng xarray; lỗi thì thử coi như ZIP, giải nén .nc.
    - area: [N, W, S, E] (toạ độ độ), nếu None sẽ tải toàn cầu.
    """
    try:
        import cdsapi  # type: ignore
    except Exception as e:
        raise RuntimeError("Thiếu thư viện cdsapi. Cài: pip install cdsapi") from e

    if out_nc.exists() and out_nc.stat().st_size > 1_000_000:
        return out_nc

    ym = f"{year:04d}-{month:02d}"
    req = dict(
        product_type="reanalysis",
        variable=["10m_u_component_of_wind", "10m_v_component_of_wind"],
        year=f"{year:04d}",
        month=f"{month:02d}",
        day=[f"{d:02d}" for d in range(1, 32)],
        time=[f"{h:02d}:00" for h in range(24)],
        data_format="netcdf",
    )
    if area is not None:
        # ERA5 yêu cầu [N, W, S, E]
        req["area"] = [float(area[0]), float(area[1]), float(area[2]), float(area[3])]

    tmp = out_nc.with_suffix(".part")
    c = cdsapi.Client()

    for k in range(max_retry):
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass

        print(f"[CDS] ERA5-Land hourly u10/v10 {ym} -> {out_nc.name} (attempt {k+1}/{max_retry})")
        c.retrieve("reanalysis-era5-land", req, tmp.as_posix())

        # 1) thử mở trực tiếp như NetCDF
        try:
            _open_ds_any(tmp).close()
            tmp.replace(out_nc)
            return out_nc
        except Exception as e1:
            # 2) nếu fail, thử coi như file ZIP và giải nén .nc bên trong
            try:
                with zipfile.ZipFile(tmp, "r") as zf:
                    members = [n for n in zf.namelist() if n.lower().endswith(".nc")]
                    if not members:
                        raise RuntimeError("ZIP không chứa file .nc")  # sẽ catch bên dưới
                    name = members[0]
                    with zf.open(name) as src, open(out_nc, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                _open_ds_any(out_nc).close()
                return out_nc
            except Exception as e2:
                wait = backoff0 * (2 ** k)
                print(
                    "[WARN] File .part hỏng hoặc không phải NetCDF/ZIP hợp lệ "
                    f"({e1}; unzip_err={e2}). Retry sau {wait}s ..."
                )
                time.sleep(wait)

    raise RuntimeError(f"Tải ERA5-Land {ym} thất bại sau {max_retry} lần.")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def run_era5_wind_to_vicres(
    *,
    basin_vector: Optional[str | Path] = None,
    vic_grid_template_nc: str | Path,
    years: List[int],
    months: List[int],
    tz_offset_hours: int = 0,
    dataset: str = "ERA5-Land",
    backend: str = "local",
    export_ascii: bool = True,
    show_plots: bool = False,
    root: str | Path = ".",
) -> Dict[str, str]:
    """
    Pipeline:
      (1) Lấy NetCDF tháng (local hoặc cds) → kiểm tra biến gió
      (2) Ghép tháng → tính tốc độ gió (sqrt(u^2+v^2)) nếu cần
      (3) Đổi múi giờ + daily mean
      (4) Nội suy sang lưới VIC
      (5) Lưu NetCDF QC + ASCII VIC-Res (+ QC plots nếu cần)
    """

    # Chuẩn hoá backend
    bk = str(backend or "local").strip().lower()
    if bk == "auto":
        bk = "local"
    elif bk == "ecmwf":
        bk = "cds"

    root = Path(root).resolve()
    raw_dir = root / "data" / "raw" / "ERA5L" / "hourly"
    fig_dir = root / "outputs" / "figs" / "qc" / "wind"
    nc_dir = root / "outputs" / "forcings" / "netcdf"
    asc_dir = root / "outputs" / "forcings" / "ascii" / "wind"
    _ensure_dir(fig_dir)
    _ensure_dir(nc_dir)
    _ensure_dir(asc_dir)

    # --- nếu backend cds: tính bbox cho area để tải nhanh hơn ---
    cds_area: Optional[List[float]] = None
    if bk == "cds":
        try:
            if basin_vector is not None:
                import geopandas as gpd

                g = gpd.read_file(basin_vector).to_crs(4326)
                minx, miny, maxx, maxy = g.total_bounds
                pad = 0.25
                W, S, E, N = minx - pad, miny - pad, maxx + pad, maxy + pad
                cds_area = [float(N), float(W), float(S), float(E)]
            else:
                vt = _open_ds_any(Path(vic_grid_template_nc))
                lon = vt["lon"] if "lon" in vt.coords else vt["longitude"]
                lat = vt["lat"] if "lat" in vt.coords else vt["latitude"]
                W, E = float(lon.min()) - 0.25, float(lon.max()) + 0.25
                S, N = float(lat.min()) - 0.25, float(lat.max()) + 0.25
                cds_area = [float(N), float(W), float(S), float(E)]
            print(f"[INFO] CDS area bbox (N,W,S,E) = {tuple(cds_area)}")
        except Exception as e:  # pragma: no cover
            print("[WARN] Không tính được bbox lưu vực cho CDS, sẽ tải toàn cầu:", e)
            cds_area = None

    # 1) Lấy danh sách file tháng
    monthly_files: List[Path] = []
    for y in years:
        for m in months:
            ydir = _ensure_dir(raw_dir / f"{y}")
            fn = ydir / f"era5land_wind10_hourly_{y:04d}{m:02d}.nc"
            if bk == "local":
                if not fn.exists():
                    raise FileNotFoundError(
                        f"Không thấy {fn}.\n"
                        "→ Hãy tải NetCDF u10/v10 theo tháng vào đúng tên/thư mục trên "
                        "(hoặc dùng --backend cds)."
                    )
                monthly_files.append(fn)
            elif bk == "cds":
                monthly_files.append(
                    _cds_monthly_download_era5land_u10_v10(y, m, fn, area=cds_area)
                )
            else:
                raise ValueError("backend phải là 'local' hoặc 'cds'.")

    # 2) Ghép tháng → tính tốc độ gió (wind)
    pieces = []
    for fp in monthly_files:
        ds = _open_ds_any(fp)
        ds = _normalize_time_and_dims(ds)

        u_name, v_name, si_name = _detect_wind_vars(ds)
        if u_name and v_name:
            u = ds[u_name]
            v = ds[v_name]
            wind = np.sqrt(u ** 2 + v ** 2)
        elif si_name:
            wind = ds[si_name]
        else:
            raise KeyError(f"{fp.name}: Không tìm thấy biến u10/v10 hoặc si10.")
        wind = wind.rename("wind")
        wind.attrs.setdefault("units", "m s-1")
        pieces.append(wind.to_dataset())

    ds_all = xr.concat(pieces, dim="time").sortby("time")

    # 3) Daily theo múi giờ
    wind_daily = _to_daily_local_time(ds_all["wind"], tz_offset_hours)

    # 4) Nội suy sang lưới VIC
    wind_daily_vic = _interp_to_vic_grid(wind_daily, Path(vic_grid_template_nc))

    # 5) Lưu NetCDF QC
    y0, m0 = years[0], months[0]
    y1, m1 = years[-1], months[-1]
    out_nc = nc_dir / f"wind_daily_vic_{y0:04d}{m0:02d}_{y1:04d}{m1:02d}.nc"
    wind_daily_vic.to_dataset(name="wind").to_netcdf(out_nc)

    # 6) Xuất ASCII VIC-Res
    if export_ascii:
        _export_ascii_per_cell(asc_dir, wind_daily_vic, var="wind")

    # 7) QC figure (tuỳ chọn)
    if show_plots:
        try:
            import matplotlib.pyplot as plt

            ts = wind_daily_vic.mean(dim=("lat", "lon")).to_pandas()
            plt.figure(figsize=(10, 4))
            ts.plot()
            plt.title("Basin mean WIND [m/s]")
            plt.ylabel("m/s")
            plt.tight_layout()
            plt.savefig(fig_dir / f"qc_wind_basinmean_{y0:04d}{m0:02d}.png", dpi=150)
            plt.close()

            # Map mean WIND trên lưới VIC
            if basin_vector is not None:
                map_png = fig_dir / f"qc_wind_meanmap_{y0:04d}{m0:02d}_{y1:04d}{m1:02d}.png"
                _plot_vic_mean_map(wind_daily_vic, basin_vector, map_png)
        except Exception as e:  # pragma: no cover
            print("[WARN] Vẽ QC WIND không thành công:", e)

    return {"nc": str(out_nc), "ascii_dir": str(asc_dir)}
