"""s05c — Covariáveis aerogeofísicas (v3).

Fontes (data/raw/geofisica): projeto 1039 (SP-RJ 1978, formato antigo)
e projeto 1105 (SJC-Resende 2010-14, Geosoft). Posições de canal
fixadas pela inspeção de 11/08/2026 (s05c_headers.txt).

Método:
  1. Leitura em chunks dos XYZ, filtro na AOI+buffer por LONG/LAT em
     graus decimais (decisão: escapa dos dois meridianos centrais do
     1039; shift Hayford->SIRGAS ~50 m é irrelevante a 250 m);
  2. Gridagem por BLOCO-MEDIANA a 250 m (EPSG 31983) por fonte;
  3. Harmonização DECLARADA: z-score por fonte (cps 1978 e
     concentrações 2013 não se misturam cruas); mosaico preferindo o
     1105 onde cobre; preenchimento de vazios por vizinho mais próximo
     limitado a 2 km;
  4. Derivadas do grid magnético mosaicado: 1DV (FFT), gradiente
     horizontal total (GHT) e tilt;
  5. Reamostragem ao grid de 60 m, gravação em covariaveis/ (prefixo
     geof_) e extração nos poços -> covariaveis_pocos.parquet.

Baseline congelada p/ o teste v3 (regra pré-registrada no config):
AUC espacial 0,650 | AUC externo 0,563 — a geofísica só é adotada se
melhorar OS DOIS.

Execução:  docker compose run --rm pipeline python stages/s05c_geofisica.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _comum import config, RAW, DERIVED

OUT = Path("/data/outputs")
COV = DERIVED / "covariaveis"
GEOF = RAW / "geofisica"
RELATORIO = OUT / "s05c_relatorio.txt"

RES_GEOF = 250.0          # m — grid de gridagem (>= espaçamento 1105)
BUFFER_GRAUS = 0.05
DECIMACAO_MAG_1105 = 10   # amostragem 0,1 s ~ 8 m; 10x -> ~80 m

# Posições de coluna (s05c_headers.txt, 11/08/2026)
FONTES = {
    "1039": {
        "arquivos": ["1039/spaulo_rjaneiro_sp.xyz"],
        "n_cols": 20, "lon": 2, "lat": 3, "decimacao": 1,
        "canais": {"mag": 4, "eth": 5, "eu": 6, "k": 7, "ct": 8},
    },
    "1105_mag": {
        "arquivos": ["1105/XYZ/1105_MagLine.XYZ", "1105/XYZ/1105_MagTie.XYZ"],
        "n_cols": 20, "lon": 10, "lat": 8,
        "decimacao": DECIMACAO_MAG_1105,
        "canais": {"mag": 17},
        "xy_alt": (3, 4),            # X/Y UTM 23S — fallback p/ dummies
    },
    "1105_gama": {
        "arquivos": ["1105/XYZ/1105_GamaLine.XYZ", "1105/XYZ/1105_GamaTie.XYZ"],
        "n_cols": 30, "lon": 29, "lat": 26, "decimacao": 1,
        "canais": {"k": 9, "eu": 10, "eth": 11, "ct": 12},
        "xy_alt": (14, 15),          # X/Y UTM 23S — fallback p/ dummies
    },
}


class Tee:
    def __init__(self, caminho):
        self.terminal = sys.stdout
        self.arquivo = open(caminho, "w", encoding="utf-8")

    def write(self, msg):
        self.terminal.write(msg); self.arquivo.write(msg)

    def flush(self):
        self.terminal.flush(); self.arquivo.flush()


def ler_xyz(caminho, spec, bbox):
    """Parser manual (robusto a marcadores 'Line'/'Tie' e cabeçalhos):
    converte lon/lat primeiro e só parseia os canais das linhas que
    caem na AOI — rápido mesmo em arquivos de GB."""
    i_lon, i_lat = spec["lon"], spec["lat"]
    canais = spec["canais"]
    xy_alt = spec.get("xy_alt")
    max_col = max(i_lon, i_lat, *canais.values(),
                  *(xy_alt if xy_alt else (0,)))
    dec = spec["decimacao"]
    cols = {"lon": [], "lat": [], **{k: [] for k in canais}}
    total = 0
    d = 0
    resgatados = 0
    transformador = None
    if xy_alt:
        from pyproj import Transformer
        transformador = Transformer.from_crs("EPSG:31983", "EPSG:4674",
                                             always_xy=True)
    with open(caminho, "r", encoding="latin-1", errors="replace") as f:
        for linha in f:
            if linha.startswith("/"):
                continue
            partes = linha.split()
            if len(partes) <= max_col:          # marcadores Line/Tie
                continue
            total += 1
            d += 1
            if dec > 1 and d % dec:
                continue
            lon = lat = None
            try:
                lon = float(partes[i_lon]); lat = float(partes[i_lat])
                if not (-180 <= lon <= 180 and -90 <= lat <= 90):
                    lon = None
            except ValueError:
                pass
            if lon is None and transformador is not None:
                # fallback: X/Y UTM do próprio levantamento
                try:
                    x = float(partes[xy_alt[0]])
                    y = float(partes[xy_alt[1]])
                    lon, lat = transformador.transform(x, y)
                    resgatados += 1
                except ValueError:
                    continue
            if lon is None:
                continue
            if not (bbox[0] <= lon <= bbox[2]
                    and bbox[1] <= lat <= bbox[3]):
                continue
            cols["lon"].append(lon); cols["lat"].append(lat)
            for nome, pos in canais.items():
                try:
                    cols[nome].append(float(partes[pos]))
                except ValueError:
                    cols[nome].append(np.nan)
    df = pd.DataFrame({k: np.asarray(v, dtype="float32")
                       for k, v in cols.items()})
    if resgatados:
        print(f"   coordenadas resgatadas via X/Y UTM: {resgatados:,}")
    return df, total


def gridar(df, canal, transform_g, shape_g):
    """Bloco-mediana no grid de RES_GEOF."""
    sub = df.dropna(subset=[canal])
    if not len(sub):
        return None
    inv = ~transform_g
    cc, ll = inv * (sub["x"].values, sub["y"].values)
    cc = np.floor(cc).astype(int); ll = np.floor(ll).astype(int)
    ok = (cc >= 0) & (cc < shape_g[1]) & (ll >= 0) & (ll < shape_g[0])
    cel = ll[ok] * shape_g[1] + cc[ok]
    med = pd.Series(sub[canal].values[ok]).groupby(cel).median()
    grade = np.full(shape_g[0] * shape_g[1], np.nan, dtype="float32")
    grade[med.index.values] = med.values
    return grade.reshape(shape_g)


def zscore(a):
    m, s = np.nanmean(a), np.nanstd(a)
    return (a - m) / s if s > 0 else a * 0.0


def preencher_vizinho(a, res, max_dist_m=2000.0):
    from scipy import ndimage
    mask = np.isnan(a)
    if not mask.any():
        return a
    dist, (iy, ix) = ndimage.distance_transform_edt(
        mask, return_indices=True)
    out = a[iy, ix]
    out[dist * res > max_dist_m] = np.nan
    return out.astype("float32")


def derivada_vertical_fft(a, res):
    """1ª derivada vertical de campo potencial via FFT."""
    pre = np.nan_to_num(a, nan=float(np.nanmedian(a)))
    ny, nx = pre.shape
    kx = np.fft.fftfreq(nx, d=res) * 2 * np.pi
    ky = np.fft.fftfreq(ny, d=res) * 2 * np.pi
    kxx, kyy = np.meshgrid(kx, ky)
    kmod = np.hypot(kxx, kyy)
    return np.real(np.fft.ifft2(np.fft.fft2(pre) * kmod)).astype("float32")


def main():
    import os
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import reproject, Resampling
    import geopandas as gpd
    from sqlalchemy import text
    from _comum import engine
    from s05_covariaveis import salvar

    OUT.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(RELATORIO)
    print("===== s05c — AEROGEOFÍSICA (v3) =====")

    bbox = [float(x) for x in os.environ["AOI_BBOX"].split(",")]
    bbox_buf = [bbox[0] - BUFFER_GRAUS, bbox[1] - BUFFER_GRAUS,
                bbox[2] + BUFFER_GRAUS, bbox[3] + BUFFER_GRAUS]
    epsg = int(os.environ.get("EPSG_TRABALHO", "31983"))

    with rasterio.open(COV / "mde.tif") as src:
        shape60 = (src.height, src.width)
        transform60 = src.transform

    # grid de gridagem (250 m) cobrindo o grid de 60 m
    x0, y1 = transform60.c, transform60.f
    x1 = x0 + transform60.a * shape60[1]
    y0 = y1 + transform60.e * shape60[0]
    ncol = int(np.ceil((x1 - x0) / RES_GEOF))
    nlin = int(np.ceil((y1 - y0) / RES_GEOF))
    transform_g = from_origin(x0, y1, RES_GEOF, RES_GEOF)
    shape_g = (nlin, ncol)

    # ---------- leitura e gridagem por fonte ----------
    grades = {}      # (fonte_base, canal) -> grade z-score
    for fonte, spec in FONTES.items():
        dfs = []
        for arq in spec["arquivos"]:
            caminho = GEOF / arq
            if not caminho.exists():
                print(f"[{fonte}] AUSENTE: {arq}")
                continue
            print(f"[{fonte}] lendo {arq} ...", flush=True)
            df, total = ler_xyz(caminho, spec, bbox_buf)
            print(f"   {total:,} linhas no arquivo -> "
                  f"{len(df):,} pontos na AOI")
            dfs.append(df)
        if not dfs:
            continue
        df = pd.concat(dfs, ignore_index=True)
        g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(
            df.lon, df.lat), crs="EPSG:4674").to_crs(epsg=epsg)
        df["x"], df["y"] = g.geometry.x, g.geometry.y
        base = fonte.split("_")[0]
        for canal in spec["canais"]:
            grade = gridar(df, canal, transform_g, shape_g)
            if grade is None:
                continue
            n_val = int(np.isfinite(grade).sum())
            print(f"   canal {canal}: {n_val:,} células com dado")
            grades[(base, canal)] = zscore(grade)

    # ---------- mosaico 1105-sobre-1039 ----------
    canais_finais = {}
    for canal in ("mag", "k", "eth", "eu", "ct"):
        g39 = grades.get(("1039", canal))
        g05 = grades.get(("1105", canal))
        if g39 is None and g05 is None:
            continue
        if g39 is None:
            mos = g05
        elif g05 is None:
            mos = g39
        else:
            mos = np.where(np.isfinite(g05), g05, g39)
        mos = preencher_vizinho(mos, RES_GEOF)
        canais_finais[f"geof_{canal}"] = mos
        cob = 100 * np.isfinite(mos).mean()
        pref = (100 * np.isfinite(g05).mean()
                if g05 is not None else 0.0)
        print(f"[mosaico] geof_{canal}: cobertura {cob:.0f}% "
              f"(1105 em {pref:.0f}%)")

    # ---------- derivadas magnéticas ----------
    if "geof_mag" in canais_finais:
        mag = canais_finais["geof_mag"]
        dv1 = derivada_vertical_fft(mag, RES_GEOF)
        gy, gx = np.gradient(np.nan_to_num(mag, nan=0.0), RES_GEOF)
        ght = np.hypot(gx, gy).astype("float32")
        tilt = np.arctan2(dv1, np.maximum(ght, 1e-9)).astype("float32")
        canais_finais["geof_mag_1dv"] = dv1
        canais_finais["geof_mag_ght"] = ght
        canais_finais["geof_mag_tilt"] = tilt

    # ---------- reamostrar a 60 m, salvar e extrair nos poços ----------
    finais60 = {}
    for nome, arr in canais_finais.items():
        destino = np.full(shape60, np.nan, dtype="float32")
        reproject(arr, destino, src_transform=transform_g,
                  src_crs=f"EPSG:{epsg}", dst_transform=transform60,
                  dst_crs=f"EPSG:{epsg}",
                  resampling=Resampling.bilinear,
                  src_nodata=np.nan, dst_nodata=np.nan)
        finais60[nome] = destino
        salvar(nome, destino, transform60, epsg)

    eng = engine()
    with eng.connect() as conn:
        pocos = pd.read_sql(text("""
            SELECT codigo_siagas, ST_X(geom) lon, ST_Y(geom) lat
            FROM pocos WHERE qaqc_aprovado"""), conn)
    g = gpd.GeoDataFrame(pocos, geometry=gpd.points_from_xy(
        pocos.lon, pocos.lat), crs="EPSG:4674").to_crs(epsg=epsg)
    inv60 = ~transform60
    cc, ll = inv60 * (g.geometry.x.values, g.geometry.y.values)
    cc = np.clip(cc.astype(int), 0, shape60[1] - 1)
    ll = np.clip(ll.astype(int), 0, shape60[0] - 1)

    cov = pd.read_parquet(DERIVED / "covariaveis_pocos.parquet")
    cov = cov.drop(columns=[c for c in cov.columns
                            if c.startswith("geof_")], errors="ignore")
    ext = pd.DataFrame({"codigo_siagas": pocos["codigo_siagas"]})
    for nome, arr in finais60.items():
        ext[nome] = arr[ll, cc]
    cov = cov.merge(ext, on="codigo_siagas", how="left")
    cov.to_parquet(DERIVED / "covariaveis_pocos.parquet", index=False)

    n_geof = int(ext.drop(columns="codigo_siagas")
                 .notna().all(axis=1).sum())
    print(f"\nCovariáveis geofísicas: {len(finais60)} "
          f"({', '.join(finais60)})")
    print(f"Poços com todas as geofísicas: {n_geof} de {len(pocos)}")

    # quicklook
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        im0 = axs[0].imshow(np.clip(finais60.get("geof_mag_1dv"),
                                    -3, 3), cmap="RdBu_r")
        axs[0].set_title("Anomalia magnética — 1ª derivada vertical (z)")
        fig.colorbar(im0, ax=axs[0], shrink=.8)
        im1 = axs[1].imshow(np.clip(finais60.get("geof_k"), -3, 3),
                            cmap="viridis")
        axs[1].set_title("Potássio (z-score por fonte)")
        fig.colorbar(im1, ax=axs[1], shrink=.8)
        for ax in axs:
            ax.axis("off")
        fig.savefig(OUT / "quicklook_geofisica.png", dpi=120,
                    bbox_inches="tight")
        print("Quicklook: data/outputs/quicklook_geofisica.png")
    except Exception as e:
        print(f"Quicklook não gerado: {e}")

    print("\nPróximo: s06 -> s07 -> s07_externa; regra v3 pré-registrada:")
    print("adotar geofísica SOMENTE se AUC espacial > 0,650 E "
          "AUC externo > 0,563.")
    print(f"Relatório salvo em: {RELATORIO}")
    sys.stdout.arquivo.close()
    sys.stdout = sys.stdout.terminal


if __name__ == "__main__":
    main()
