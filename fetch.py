import os, io, sys, time, datetime as dt
import numpy as np, requests, pandas as pd, geopandas as gpd

KEY  = os.environ["FIRMS_KEY"]
BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
BBOX = "95,-11,141,6"
STEP = 5
SP, NRT = "VIIRS_SNPP_SP", "VIIRS_SNPP_NRT"
PROV = gpd.read_file("data/provinces.geojson")[["shapeName", "geometry"]]
F_DAILY, F_RECENT = "data/daily_province.csv", "data/recent.csv"

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
    return df[(df.acq_date >= str(d0)) & (df.acq_date <= str(d1))]

def tag(df):
    g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
    g = gpd.sjoin(g, PROV, how="left", predicate="within").rename(columns={"shapeName": "provinsi"})
    g["provinsi"] = g["provinsi"].fillna("Di luar daratan")
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

def save_grid(g):
    for y, part in g.groupby(g.ym.str[:4]):
        f = f"data/grid_{y}.csv"
        if os.path.exists(f):
            old = pd.read_csv(f)
            part = pd.concat([old[~old.ym.isin(part.ym.unique())], part])
        part.sort_values(["ym", "lat", "lon"]).to_csv(f, index=False)

def save_recent(df):
    cut = str(dt.date.today() - dt.timedelta(days=7))
    cols = ["latitude", "longitude", "acq_date", "acq_time", "frp", "confidence", "provinsi"]
    df[df.acq_date >= cut][cols].to_csv(F_RECENT, index=False)

def month_end(d):
    return (d.replace(day=28) + dt.timedelta(days=4)).replace(day=1) - dt.timedelta(days=1)

if __name__ == "__main__":
    mode, today = sys.argv[1], dt.date.today()
    if mode == "update" and not os.path.exists(F_DAILY):
        mode = "backfill"
    if mode == "backfill":
        sp_end = today - dt.timedelta(days=100)
        cut = str(today - dt.timedelta(days=7))
        d, days, grids, recents = dt.date(2018, 1, 1), [], [], []
        while d <= today:
            e = min(month_end(d), today)
            if d <= sp_end < e: e = sp_end
            df = tag(pull(SP if e <= sp_end else NRT, d, e))
            days.append(daily(df)); grids.append(grid(df)); recents.append(df[df.acq_date >= cut])
            d = e + dt.timedelta(days=1)
        pd.concat(days).to_csv(F_DAILY, index=False)
        save_grid(pd.concat(grids)); save_recent(pd.concat(recents))
    else:
        start = (today - dt.timedelta(days=10)).replace(day=1)
        new = tag(pull(NRT, start, today))
        old = pd.read_csv(F_DAILY); old = old[old.acq_date < str(start)]
        pd.concat([old, daily(new)]).sort_values(["acq_date", "provinsi"]).to_csv(F_DAILY, index=False)
        save_grid(grid(new)); save_recent(new)
