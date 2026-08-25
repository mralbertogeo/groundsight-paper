"""Etapa 5 — Covariáveis espaciais (M5, v1).

Blocos:
  A. MDE: download do Copernicus GLO-30 (bucket público AWS, cache em
     /data/raw/mde), mosaico, recorte AOI+buffer e reprojeção para o
     grid de trabalho (EPSG_TRABALHO, resolução GRID_RES);
  B. Geomorfometria local: declividade, orientação (northness),
     curvatura, rugosidade (TRI);
  C. Hidrologia derivada (D8 próprio, sem dependências externas):
     preenchimento de depressões, direção de fluxo, acumulação, TWI,
     rede de drenagem (limiar de área), HAND, distância à drenagem e
     densidade de drenagem;
  D. Superfícies do M4 reamostradas ao grid e convertidas de
     profundidade para COTA (cota_contato = MDE − profundidade);
  E. Cotas dos poços a partir do MDE (regra 2.8 adaptada: o SIAGAS não
     traz cota nesta AOI — o MDE é a fonte, cota_fonte='mde') e carga
     hidráulica h = cota − NE gravada em hidraulica_derivada;
  F. Extração de todas as covariáveis nas posições dos poços
     (covariaveis_pocos.parquet) + tabela de proveniência.

DECISÕES v1 declaradas: lineamentos, precipitação e MapBiomas ficam
para v2 (débito técnico) — o conjunto v1 (elevação, declividade, HAND,
TWI + espessuras do manto) cobre os preditores dominantes da
literatura de GPM.

Execução:  docker compose run --rm pipeline python stages/s05_covariaveis.py
"""
import heapq
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sqlalchemy import text

from _comum import engine, config, RAW, DERIVED

OUT = Path("/data/outputs")
COV = DERIVED / "covariaveis"
GEO = DERIVED / "geologia_3d"
MDE_DIR = RAW / "mde"

S3 = "https://copernicus-dem-30m.s3.amazonaws.com"


# ======================================================================
# A. MDE
# ======================================================================
def tiles_necessarios(bbox, buffer_graus=0.05):
    x0, y0, x1, y1 = (bbox[0] - buffer_graus, bbox[1] - buffer_graus,
                      bbox[2] + buffer_graus, bbox[3] + buffer_graus)
    tiles = set()
    for lon in range(math.floor(x0), math.floor(x1) + 1):
        for lat in range(math.floor(y0), math.floor(y1) + 1):
            # Tiles Copernicus são nomeados pelo canto SUDOESTE:
            # lat -23 (cobre -23..-22) -> "S23"
            ns = f"S{abs(lat):02d}" if lat < 0 else f"N{lat:02d}"
            ew = f"W{abs(lon):03d}" if lon < 0 else f"E{lon:03d}"
            tiles.add(f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM")
    return sorted(tiles), (x0, y0, x1, y1)


def baixar_mde(bbox, epsg, res):
    import rasterio
    from rasterio.merge import merge
    from rasterio.warp import calculate_default_transform, reproject, Resampling

    MDE_DIR.mkdir(parents=True, exist_ok=True)
    nomes, bbox_buf = tiles_necessarios(bbox)
    arquivos = []
    for nome in nomes:
        destino = MDE_DIR / f"{nome}.tif"
        if not destino.exists():
            url = f"{S3}/{nome}/{nome}.tif"
            print(f"[MDE] baixando {nome} ...")
            r = requests.get(url, timeout=300, stream=True)
            if r.status_code != 200:
                print(f"   tile inexistente ({r.status_code}) — ignorado")
                continue
            with open(destino, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        arquivos.append(destino)
    if not arquivos:
        raise RuntimeError("Nenhum tile de MDE obtido.")

    fontes = [rasterio.open(a) for a in arquivos]
    mosaico, transf = merge(fontes, bounds=bbox_buf)
    perfil = fontes[0].profile
    for f in fontes:
        f.close()

    # Reprojeção para o grid de trabalho
    from rasterio.io import MemoryFile
    perfil.update(height=mosaico.shape[1], width=mosaico.shape[2],
                  transform=transf, count=1)
    with MemoryFile() as mem:
        with mem.open(**perfil) as tmp:
            tmp.write(mosaico[0], 1)
            dst_crs = f"EPSG:{epsg}"
            transform, w, h = calculate_default_transform(
                tmp.crs, dst_crs, tmp.width, tmp.height, *tmp.bounds,
                resolution=res)
            destino = np.full((h, w), np.nan, dtype="float32")
            reproject(source=rasterio.band(tmp, 1), destination=destino,
                      src_transform=tmp.transform, src_crs=tmp.crs,
                      dst_transform=transform, dst_crs=dst_crs,
                      resampling=Resampling.bilinear, dst_nodata=np.nan)
    return destino, transform


def salvar(nome, arr, transform, epsg):
    import rasterio
    COV.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
            COV / f"{nome}.tif", "w", driver="GTiff",
            height=arr.shape[0], width=arr.shape[1], count=1,
            dtype="float32", crs=f"EPSG:{epsg}", transform=transform,
            nodata=-9999) as dst:
        dst.write(np.nan_to_num(arr, nan=-9999).astype("float32"), 1)


# ======================================================================
# C. Hidrologia D8
# ======================================================================
VIZ = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def preencher_depressoes(dem):
    """Priority-flood (Barnes et al., 2014, simplificado)."""
    h, w = dem.shape
    preenchido = dem.copy()
    fechado = np.zeros((h, w), dtype=bool)
    fila = []
    for i in range(h):
        for j in (0, w - 1):
            heapq.heappush(fila, (preenchido[i, j], i, j)); fechado[i, j] = True
    for j in range(w):
        for i in (0, h - 1):
            if not fechado[i, j]:
                heapq.heappush(fila, (preenchido[i, j], i, j)); fechado[i, j] = True
    while fila:
        z, i, j = heapq.heappop(fila)
        for di, dj in VIZ:
            a, b = i + di, j + dj
            if 0 <= a < h and 0 <= b < w and not fechado[a, b]:
                fechado[a, b] = True
                preenchido[a, b] = max(preenchido[a, b], z + 1e-4)
                heapq.heappush(fila, (preenchido[a, b], a, b))
    return preenchido


def d8_e_acumulacao(preenchido, res):
    h, w = preenchido.shape
    alvo = np.full((h, w), -1, dtype=np.int64)
    for i in range(h):
        for j in range(w):
            z0, melhor, alvo_ij = preenchido[i, j], 0.0, -1
            for di, dj in VIZ:
                a, b = i + di, j + dj
                if 0 <= a < h and 0 <= b < w:
                    dist = res * (1.41421356 if di and dj else 1.0)
                    decl = (z0 - preenchido[a, b]) / dist
                    if decl > melhor:
                        melhor, alvo_ij = decl, a * w + b
            alvo[i, j] = alvo_ij
    ordem = np.argsort(preenchido, axis=None)[::-1]  # do alto para o baixo
    acc = np.ones(h * w, dtype=np.float64)
    alvo_flat = alvo.ravel()
    for c in ordem:
        t = alvo_flat[c]
        if t >= 0:
            acc[t] += acc[c]
    return alvo_flat, acc.reshape(h, w), ordem


def hand_calc(dem, alvo_flat, ordem, riacho):
    """Height Above Nearest Drainage seguindo o caminho de fluxo."""
    h, w = dem.shape
    ref = np.full(h * w, np.nan)
    dem_f = dem.ravel()
    riacho_f = riacho.ravel()
    for c in ordem[::-1]:            # do baixo para o alto
        if riacho_f[c]:
            ref[c] = dem_f[c]
        else:
            t = alvo_flat[c]
            ref[c] = ref[t] if t >= 0 and not np.isnan(ref[t]) else dem_f[c]
    return (dem_f - ref).reshape(h, w)


# ======================================================================
def main():
    import rasterio
    from rasterio.warp import reproject, Resampling
    from scipy import ndimage

    cfg = config().get("covariaveis", {})
    bbox = [float(x) for x in os.environ["AOI_BBOX"].split(",")]
    epsg = int(os.environ.get("EPSG_TRABALHO", "31983"))
    res = float(os.environ.get("GRID_RES", "60"))

    # ---------- A. MDE ----------
    mde, transform = baixar_mde(bbox, epsg, res)
    if np.nanmax(mde) - np.nanmin(mde) < 1.0:
        raise RuntimeError(
            "MDE degenerado (relevo < 1 m) — tiles errados ou fora da AOI. "
            "Verifique AOI_BBOX e os arquivos em data/raw/mde.")
    salvar("mde", mde, transform, epsg)
    print(f"[MDE] grid {mde.shape}, cotas {np.nanmin(mde):.0f}–"
          f"{np.nanmax(mde):.0f} m")

    dem = np.nan_to_num(mde, nan=float(np.nanmedian(mde)))

    # ---------- B. Geomorfometria ----------
    gy, gx = np.gradient(dem, res)
    decliv = np.degrees(np.arctan(np.hypot(gx, gy)))
    aspecto = np.arctan2(-gx, gy)
    northness = np.cos(aspecto)
    curvatura = ndimage.laplace(dem) / (res ** 2)
    tri = ndimage.generic_filter(dem, np.std, size=3)
    for nome, arr in [("declividade", decliv), ("northness", northness),
                      ("curvatura", curvatura), ("tri", tri)]:
        salvar(nome, arr, transform, epsg)

    # ---------- C. Hidrologia ----------
    print("[Hidro] preenchimento de depressões ...")
    dem_p = preencher_depressoes(dem)
    print("[Hidro] D8 + acumulação ...")
    alvo, acc, ordem = d8_e_acumulacao(dem_p, res)
    area_km2 = float(cfg.get("limiar_drenagem_km2", 1.0))
    riacho = acc * (res ** 2) / 1e6 >= area_km2
    twi = np.log((acc * res) / np.tan(np.radians(np.maximum(decliv, 0.1))))
    print("[Hidro] HAND ...")
    # v1.1: referência no MDE PREENCHIDO (fluxo é calculado nele) e
    # truncagem em zero — corrige HAND negativo em depressões preenchidas
    hand = np.maximum(hand_calc(dem_p, alvo, ordem, riacho), 0.0)
    dist_dren = ndimage.distance_transform_edt(~riacho) * res
    dens = ndimage.uniform_filter(riacho.astype(float), size=int(2000 / res))
    for nome, arr in [("twi", twi), ("hand", hand),
                      ("dist_drenagem", dist_dren),
                      ("dens_drenagem", dens)]:
        salvar(nome, arr, transform, epsg)

    # v1.1: estatísticas FOCAIS (média em raios) — robustez à incerteza
    # posicional das coordenadas SIAGAS (até ~km)
    focais = {}
    for nome, arr in [("declividade", decliv), ("hand", hand),
                      ("twi", twi), ("tri", tri)]:
        for r_m in (250, 500, 1000):
            k = max(3, int(2 * r_m / res) + 1)
            f = ndimage.uniform_filter(arr, size=k)
            focais[f"{nome}_f{r_m}"] = f
            salvar(f"{nome}_f{r_m}", f, transform, epsg)

    # ---------- D. Superfícies do M4 -> grid e cotas ----------
    superficies = {}
    for f in ["prof_base_solo", "prof_topo_zona_rocha",
              "prof_topo_rocha_sa", "esp_solo", "esp_saprolito",
              "esp_rocha_alterada"]:
        caminho = GEO / f"{f}.tif"
        if not caminho.exists():
            continue
        with rasterio.open(caminho) as src:
            destino = np.full(dem.shape, np.nan, dtype="float32")
            reproject(rasterio.band(src, 1), destino,
                      dst_transform=transform, dst_crs=f"EPSG:{epsg}",
                      resampling=Resampling.bilinear, dst_nodata=np.nan)
        superficies[f] = destino
        salvar(f, destino, transform, epsg)
        if f.startswith("prof_"):
            salvar(f.replace("prof_", "cota_"), mde - destino,
                   transform, epsg)

    # ---------- E. Cotas e carga hidráulica ----------
    eng = engine()
    with eng.connect() as conn:
        pocos = pd.read_sql(text("""
            SELECT codigo_siagas, ST_X(geom) lon, ST_Y(geom) lat
            FROM pocos WHERE qaqc_aprovado"""), conn)
    import geopandas as gpd
    g = gpd.GeoDataFrame(pocos, geometry=gpd.points_from_xy(
        pocos.lon, pocos.lat), crs="EPSG:4674").to_crs(epsg=epsg)
    inv = ~transform
    cols, lins = [], []
    for pt in g.geometry:
        c, l = inv * (pt.x, pt.y)
        cols.append(int(c)); lins.append(int(l))
    pocos["col"] = np.clip(cols, 0, dem.shape[1] - 1)
    pocos["lin"] = np.clip(lins, 0, dem.shape[0] - 1)
    pocos["cota_mde"] = mde[pocos["lin"], pocos["col"]]

    with eng.begin() as conn:
        for _, r in pocos.iterrows():
            if not np.isnan(r["cota_mde"]):
                conn.execute(text("""
                    UPDATE pocos SET cota_terreno_m=:z, cota_fonte='mde'
                    WHERE codigo_siagas=:c"""),
                    {"z": float(r["cota_mde"]), "c": r["codigo_siagas"]})
        n_h = conn.execute(text("""
            UPDATE hidraulica_derivada d SET carga_h_m = p.cota_terreno_m - d.ne_m
            FROM pocos p WHERE p.codigo_siagas=d.codigo_siagas
              AND p.cota_terreno_m IS NOT NULL AND d.ne_m IS NOT NULL
        """)).rowcount

    # ---------- F. Extração nos poços + proveniência ----------
    rasters = {"mde": mde, "declividade": decliv, "northness": northness,
               "curvatura": curvatura, "tri": tri, "twi": twi,
               "hand": hand, "dist_drenagem": dist_dren,
               "dens_drenagem": dens, **focais, **superficies}
    for nome, arr in rasters.items():
        pocos[nome] = arr[pocos["lin"], pocos["col"]]
    saida = pocos.drop(columns=["col", "lin", "cota_mde"])
    saida.to_parquet(DERIVED / "covariaveis_pocos.parquet", index=False)

    prov = pd.DataFrame([
        {"camada": "mde", "fonte": "Copernicus GLO-30 (AWS)",
         "resolucao_original": "30 m", "data_acesso": pd.Timestamp.now().date()},
        {"camada": "derivadas geomorfométricas/hidrológicas",
         "fonte": "calculadas do MDE (D8 próprio)",
         "resolucao_original": f"{res:.0f} m", "data_acesso": pd.Timestamp.now().date()},
        {"camada": "superfícies de intemperismo",
         "fonte": "GroundSight s04 (krigagem de perfis SIAGAS)",
         "resolucao_original": "300 m", "data_acesso": pd.Timestamp.now().date()},
    ])
    prov.to_csv(COV / "proveniencia.csv", index=False)

    # Quicklook
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        im0 = axs[0].imshow(mde, cmap="terrain"); axs[0].set_title("MDE (m)")
        fig.colorbar(im0, ax=axs[0], shrink=.8)
        im1 = axs[1].imshow(np.clip(hand, 0, 150), cmap="magma")
        axs[1].set_title("HAND (m)")
        fig.colorbar(im1, ax=axs[1], shrink=.8)
        OUT.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT / "quicklook_mde_hand.png", dpi=120,
                    bbox_inches="tight")
    except Exception as e:
        print(f"Quicklook não gerado: {e}")

    # ---------- Relatório ----------
    print("\n===== RELATÓRIO s05 (covariáveis) =====")
    print(f"Grid: {dem.shape[1]} x {dem.shape[0]} células a {res:.0f} m "
          f"(EPSG:{epsg})")
    print(f"Cotas MDE na AOI: {np.nanmin(mde):.0f} – {np.nanmax(mde):.0f} m "
          f"(mediana {np.nanmedian(mde):.0f} m)")
    print(f"Rasters gerados em covariaveis/: {len(rasters) + 3}")
    print(f"Poços com cota atribuída (2.8): "
          f"{int(pocos['cota_mde'].notna().sum())} de {len(pocos)}")
    print(f"Poços com carga hidráulica h:   {n_h}")
    print(f"Células de drenagem (>= {area_km2} km²): {int(riacho.sum())}")
    print("Covariáveis nos poços: data/derived/covariaveis_pocos.parquet")
    print("Quicklook: data/outputs/quicklook_mde_hand.png")
    print("v1 sem lineamentos, precipitação e MapBiomas (débito técnico).")


if __name__ == "__main__":
    main()
