import os, io, sys, time, datetime as dt
import requests, pandas as pd, geopandas as gpd

KEY  = os.environ["FIRMS_KEY"]
BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
BBOX = "95,-11,141,6"          # seluruh Indonesia (W,S,E,N)
STEP = 5                       # hari per request (cek batas terbaru di docs FIRMS)
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
    raise RuntimeError(f"Menyerah di {source} {start} setelah {tries} percobaan")

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

def save_recent(df):
    cut = str(dt.date.today() - dt.timedelta(days=7))
    cols = ["latitude", "longitude", "acq_date", "acq_time", "frp", "confidence", "provinsi"]
    df[df.acq_date >= cut][cols].to_csv(F_RECENT, index=False)

if __name__ == "__main__":
    mode, today = sys.argv[1], dt.date.today()
    if mode == "backfill":
        # data standar (SP) tertunda ±2-3 bulan, sisanya pakai NRT
        sp_end = today - dt.timedelta(days=100)
        a = tag(pull("VIIRS_SNPP_SP", dt.date(2018, 1, 1), sp_end))
        b = tag(pull("VIIRS_SNPP_NRT", sp_end + dt.timedelta(days=1), today))
        daily(pd.concat([a, b])).to_csv(F_DAILY, index=False)
        save_recent(b)
    else:  # update: tarik ulang 10 hari terakhir, timpa
        start = today - dt.timedelta(days=10)
        new = tag(pull("VIIRS_SNPP_NRT", start, today))
        old = pd.read_csv(F_DAILY)
        old = old[old.acq_date < str(start)]
        pd.concat([old, daily(new)]).sort_values(["acq_date", "provinsi"]).to_csv(F_DAILY, index=False)
        save_recent(new)
