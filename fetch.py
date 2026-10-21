import os, io, sys, time, datetime as dt
import numpy as np, requests, pandas as pd, geopandas as gpd

KEY  = os.environ["FIRMS_KEY"]
BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
BBOX = "95,-11,141,6"
STEP = 5
SP, NRT = "VIIRS_SNPP_SP", "VIIRS_SNPP_NRT"
PROV = gpd.read_file("data/provinces.geojson")[["shapeName", "geometry"]]
F_DAILY, F_RECENT, F_KAB = "data/daily_province.csv", "data/recent.csv", "data/kab_monthly.csv"
F_GEO, ADM2_LOCAL = "data/kab_simple.geojson", "data/adm2.geojson"

def load_kab():
    """Batas kab/kota. Pakai data/adm2.geojson kalau ada, kalau tidak unduh dari geoBoundaries."""
    if os.path.exists(ADM2_LOCAL):
        src = ADM2_LOCAL
    else:
        meta = requests.get("https://www.geoboundaries.org/api/current/gbOpen/IDN/ADM2/", timeout=60).json()
        meta = meta[0] if isinstance(meta, list) else meta
        r = requests.get(meta["simplifiedGeometryGeoJSON"], timeout=300); r.raise_for_status()
        src = "adm2_download.geojson"
        with open(src, "wb") as f: f.write(r.content)
    k = gpd.read_file(src)[["shapeName", "geometry"]].rename(columns={"shapeName": "kab"})
    k["geometry"] = k.geometry.make_valid()
    pt = k.copy(); pt["geometry"] = k.geometry.representative_point()
    pt = gpd.sjoin(pt, PROV, how="left", predicate="within")
    pt = pt[~pt.index.duplicated()]
    k["provinsi"] = pt["shapeName"].reindex(k.index).fillna("Tidak diketahui").values
    k["luas"] = (k.to_crs(6933).area / 1e6).round(0)
    print(f"Kab/kota dimuat: {len(k)} wilayah", flush=True)
    return k

KAB = load_kab()

def save_geo():
    if os.path.exists(F_GEO): return
    s = KAB.copy(); s["geometry"] = s.geometry.simplify(0.01, preserve_topology=True)
    s.to_file(F_GEO, driver="GeoJSON", COORDINATE_PRECISION=3)

def fetch(source, start, tries=6):
    url = f"{BASE}/{KEY}/{source}/{BBOX}/{STEP}/{start:%Y-%m-%d}"
    for i in range(tries):
        try:
            r = requests.get(url, timeout=180)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            if "latitude" not in df.columns:
                raise RuntimeError(f"Respon FIRMS aneh: {r.text[:200]}")
            print(f"OK {source} {start}", flush=True)
            return df
        except (requests.RequestException, pd.errors.EmptyDataError) as e:
            wait = 15 * (2 ** i)
            print(f"Gagal {source} {start}: {e}. Coba lagi {wait}s ({i+1}/{tries})", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"Menyerah di {source} {start}")

def pull(source, d0, d1):
    out, d = [], d0
    while d <= d1:
        out.append(fetch(source, d)); d += dt.timedelta(days=STEP); time.sleep(0.5)
    df = pd.concat(out).drop_duplicates(["latitude", "longitude", "acq_date", "acq_time"])
    return df[(df.acq_date >= str(d0)) & (df.acq_date <= str(d1))].reset_index(drop=True)

def tag(df):
    df = df.reset_index(drop=True)
    g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
    g = gpd.sjoin(g, KAB[["kab", "provinsi", "geometry"]], how="left", predicate="within")
    g = g[~g.index.duplicated()]
    g["provinsi"] = g["provinsi"].fillna("Di luar daratan")
    g["kab"] = g["kab"].fillna("-")
    return pd.DataFrame(g.drop(columns=["geometry", "index_right"]))

def daily(df):
    df = df.assign(high=(df.confidence == "h").astype(int))
    return (df.groupby(["acq_date", "provinsi"])
              .agg(titik=("frp", "size"), high=("high", "sum"), frp=("frp", "sum"))
              .reset_index().round(1))

def grid(df):
    df = df.assign(ym=df.acq_date.astype(str).str[:7],
                   lat=(np.floor(df.latitude * 10) / 10 + 0.05).round(2),
                   lon=(np.floor(df.longitude * 10) / 10 + 0.05).round(2),
                   high=(df.confidence == "h").astype(int))
    g = (df.groupby(["ym", "lat", "lon"])
           .agg(titik=("frp", "size"), high=("high", "sum"), frp=("frp", "sum")).reset_index())
    g["frp"] = g.frp.round(1)
    return g

def kab_month(df):
    df = df.assign(ym=df.acq_date.astype(str).str[:7], high=(df.confidence == "h").astype(int))
    return (df.groupby(["ym", "provinsi", "kab"])
              .agg(titik=("frp", "size"), high=("high", "sum"), frp=("frp", "sum")).reset_index())

def save_grid(g):
    for y, part in g.groupby(g.ym.str[:4]):
        f = f"data/grid_{y}.csv"
        if os.path.exists(f):
            old = pd.read_csv(f)
            part = pd.concat([old[~old.ym.isin(part.ym.unique())], part])
        part.sort_values(["ym", "lat", "lon"]).to_csv(f, index=False)

def save_kab(g):
    if os.path.exists(F_KAB):
        old = pd.read_csv(F_KAB)
        g = pd.concat([old[~old.ym.isin(g.ym.unique())], g])
    g = g.groupby(["ym", "provinsi", "kab"], as_index=False)[["titik", "high", "frp"]].sum().round(1)
    g.sort_values(["ym", "provinsi", "kab"]).to_csv(F_KAB, index=False)

def save_recent(df):
    cut = str(dt.date.today() - dt.timedelta(days=7))
    cols = ["latitude", "longitude", "acq_date", "acq_time", "frp", "confidence", "provinsi", "kab"]
    df[df.acq_date >= cut][cols].to_csv(F_RECENT, index=False)

def month_end(d):
    return (d.replace(day=28) + dt.timedelta(days=4)).replace(day=1) - dt.timedelta(days=1)

if __name__ == "__main__":
    mode, today = sys.argv[1], dt.date.today()
    if mode == "update" and not (os.path.exists(F_DAILY) and os.path.exists(F_KAB)):
        mode = "backfill"
    save_geo()
    if mode == "backfill":
        sp_end = today - dt.timedelta(days=100)
        cut = str(today - dt.timedelta(days=7))
        d, days, grids, kabs, recents = dt.date(2018, 1, 1), [], [], [], []
        while d <= today:
            e = min(month_end(d), today)
            if d <= sp_end < e: e = sp_end
            df = tag(pull(SP if e <= sp_end else NRT, d, e))
            days.append(daily(df)); grids.append(grid(df)); kabs.append(kab_month(df))
            recents.append(df[df.acq_date >= cut])
            d = e + dt.timedelta(days=1)
        pd.concat(days).to_csv(F_DAILY, index=False)
        save_grid(pd.concat(grids)); save_kab(pd.concat(kabs)); save_recent(pd.concat(recents))
    else:
        start = (today - dt.timedelta(days=10)).replace(day=1)
        new = tag(pull(NRT, start, today))
        old = pd.read_csv(F_DAILY); old = old[old.acq_date < str(start)]
        pd.concat([old, daily(new)]).sort_values(["acq_date", "provinsi"]).to_csv(F_DAILY, index=False)
        save_grid(grid(new)); save_kab(kab_month(new)); save_recent(new)
