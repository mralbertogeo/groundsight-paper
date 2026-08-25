"""Etapa 5b — Lineamentos e covariáveis estruturais (v2).

Rotas (convergem no mesmo formato de covariável):
  AUTO — extração do MDE: hillshades multi-azimute -> Canny -> Hough
         probabilístico -> segmentos filtrados por comprimento mínimo,
         classificados por FAMÍLIA de azimute (config);
  SGB  — opcional: qualquer shapefile de estruturas colocado em
         /data/raw/estrutural/*.shp é ingerido, reprojetado e
         processado nas mesmas famílias (sufixo _sgb).

Covariáveis geradas (por família + total, cada rota):
  dens_lin_<fam>  — densidade de lineamentos (km/km², raio config)
  dist_lin_<fam>  — distância ao lineamento mais próximo (m)

Justificativa das famílias (config): em terreno cristalino do SE, a
família ~E-W subparalela ao sigma1 atual tende a ser a transmissiva
(Fernandes, 2008), ainda que o fabric NE-SW domine a paisagem.

Saídas: rasters em covariaveis/; covariaveis_pocos.parquet ATUALIZADO;
quicklook com segmentos + diagrama de roseta (validação visual do
hidrogeólogo — bloqueante); relatório em tela e em txt.

Execução:  docker compose run --rm pipeline python stages/s05b_lineamentos.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from _comum import engine, config, RAW, DERIVED

OUT = Path("/data/outputs")
COV = DERIVED / "covariaveis"
ESTR = RAW / "estrutural"
RELATORIO = OUT / "s05b_relatorio.txt"


class Tee:
    def __init__(self, caminho):
        self.terminal = sys.stdout
        self.arquivo = open(caminho, "w", encoding="utf-8")

    def write(self, msg):
        self.terminal.write(msg); self.arquivo.write(msg)

    def flush(self):
        self.terminal.flush(); self.arquivo.flush()


def hillshade(dem, res, azimute_graus, altitude_graus=45.0):
    gy, gx = np.gradient(dem, res)
    decl = np.arctan(np.hypot(gx, gy))
    asp = np.arctan2(-gx, gy)
    az = np.radians(360.0 - azimute_graus + 90.0)
    alt = np.radians(altitude_graus)
    hs = (np.sin(alt) * np.cos(decl)
          + np.cos(alt) * np.sin(decl) * np.cos(az - asp))
    return np.clip(hs, 0, 1)


def azimute_segmento(p0, p1):
    """Azimute geográfico 0-180 de um segmento em coords de imagem
    (linha cresce para baixo)."""
    (r0, c0), (r1, c1) = p0, p1
    dx, dy = (c1 - c0), -(r1 - r0)          # dy para norte
    az = np.degrees(np.arctan2(dx, dy)) % 180.0
    return az


def familia_do_azimute(az, familias):
    for nome, (a0, a1) in familias.items():
        if a0 <= az < a1 or (a0 > a1 and (az >= a0 or az < a1)):
            return nome
    return None


def extrair_segmentos_mde(dem, res, cfg):
    from skimage.feature import canny
    from skimage.transform import probabilistic_hough_line

    segmentos = []
    # v2.2: extração MULTIESCALA — a escala longa captura feixes
    # regionais que a curta fragmenta
    passes = [(az, esc) for az in cfg["azimutes_iluminacao"]
              for esc in cfg["escalas"]]
    for az_sol, esc in passes:
        comp_min_px = int(esc["comprimento_min_m"] / res)
        hs = hillshade(dem, res, az_sol)
        bordas = canny(hs, sigma=cfg["canny_sigma"])
        linhas = probabilistic_hough_line(
            bordas, threshold=10, line_length=comp_min_px,
            line_gap=int(esc["gap_px"]))
        h, w = dem.shape
        margem = 10  # px — descarta artefatos da moldura do recorte
        for (c0, r0), (c1, r1) in linhas:
            na_borda = (
                (r0 < margem and r1 < margem) or
                (r0 >= h - margem and r1 >= h - margem) or
                (c0 < margem and c1 < margem) or
                (c0 >= w - margem and c1 >= w - margem))
            if not na_borda:
                segmentos.append(((r0, c0), (r1, c1)))
    return segmentos


def segmentos_de_shapefile(transform, shape, res):
    import geopandas as gpd
    shps = sorted(ESTR.glob("*.shp"))
    if not shps:
        return None
    segs = []
    inv = ~transform
    for shp in shps:
        g = gpd.read_file(shp)
        if g.crs is None:
            print(f"   {shp.name}: sem CRS — assumindo EPSG:4674")
            g = g.set_crs("EPSG:4674")
        g = g.to_crs(transform_crs_epsg)
        for geom in g.geometry:
            if geom is None:
                continue
            linhas = (geom.geoms if geom.geom_type.startswith("Multi")
                      else [geom])
            for ln in linhas:
                xs, ys = ln.xy
                for i in range(len(xs) - 1):
                    c0, r0 = inv * (xs[i], ys[i])
                    c1, r1 = inv * (xs[i + 1], ys[i + 1])
                    if (0 <= r0 < shape[0] and 0 <= c0 < shape[1]) or \
                       (0 <= r1 < shape[0] and 0 <= c1 < shape[1]):
                        segs.append(((int(r0), int(c0)),
                                     (int(r1), int(c1))))
    print(f"[SGB] {len(shps)} shapefile(s), {len(segs)} segmentos")
    return segs


def rasterizar_e_derivar(segmentos, familias, shape, res, raio_m, sufixo,
                         transform, epsg, salvar):
    from skimage.draw import line as sk_line
    from scipy import ndimage

    mascaras = {f: np.zeros(shape, bool) for f in familias}
    mascaras["total"] = np.zeros(shape, bool)
    azimutes = []
    for p0, p1 in segmentos:
        az = azimute_segmento(p0, p1)
        azimutes.append(az)
        fam = familia_do_azimute(az, familias)
        rr, cc = sk_line(*p0, *p1)
        ok = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
        rr, cc = rr[ok], cc[ok]
        mascaras["total"][rr, cc] = True
        if fam:
            mascaras[fam][rr, cc] = True

    # v2.1: INTERSEÇÕES entre famílias — alvo clássico de locação em
    # cristalino (máxima conectividade de fraturas)
    fams = [f for f in familias if mascaras[f].any()]
    if len(fams) >= 2:
        dil = {f: ndimage.binary_dilation(mascaras[f], iterations=1)
               for f in fams}
        inter = np.zeros(mascaras["total"].shape, bool)
        for i in range(len(fams)):
            for j in range(i + 1, len(fams)):
                inter |= dil[fams[i]] & dil[fams[j]]
        mascaras["intersec"] = inter

    k = max(3, int(2 * raio_m / res) + 1)
    saidas = {}
    for fam, m in mascaras.items():
        dens = ndimage.uniform_filter(m.astype(float), size=k) * 1000.0 / res
        dist = ndimage.distance_transform_edt(~m) * res
        saidas[f"dens_lin_{fam}{sufixo}"] = dens
        saidas[f"dist_lin_{fam}{sufixo}"] = dist
        salvar(f"dens_lin_{fam}{sufixo}", dens, transform, epsg)
        salvar(f"dist_lin_{fam}{sufixo}", dist, transform, epsg)
    return saidas, np.array(azimutes)


def main():
    import os
    import rasterio
    from s05_covariaveis import salvar

    global transform_crs_epsg
    OUT.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(RELATORIO)

    cfg = config().get("lineamentos", {})
    familias = {k: tuple(v) for k, v in cfg.get(
        "familias", {"ne_sw": [20, 70], "ew": [70, 110]}).items()}
    raio_m = float(cfg.get("raio_densidade_m", 1000))

    with rasterio.open(COV / "mde.tif") as src:
        dem = src.read(1).astype(float)
        dem[dem == src.nodata] = np.nan
        transform, epsg = src.transform, src.crs.to_epsg()
        res = transform.a
    transform_crs_epsg = f"EPSG:{epsg}"
    dem = np.nan_to_num(dem, nan=float(np.nanmedian(dem)))

    print("===== s05b — LINEAMENTOS =====")
    print(f"Famílias de azimute: {familias}")

    # ---- Rota AUTO ----
    segs_auto = extrair_segmentos_mde(dem, res, {
        "canny_sigma": float(cfg.get("canny_sigma", 2.0)),
        "azimutes_iluminacao": cfg.get("azimutes_iluminacao",
                                       [135, 0, 90, 45]),
        "escalas": cfg.get("escalas", [
            {"comprimento_min_m": 1500, "gap_px": 3},
            {"comprimento_min_m": 4000, "gap_px": 8}]),
    })
    print(f"[AUTO] segmentos extraídos do MDE: {len(segs_auto)}")
    saidas_auto, az_auto = rasterizar_e_derivar(
        segs_auto, familias, dem.shape, res, raio_m, "",
        transform, epsg, salvar)

    # ---- Rota SGB (opcional) ----
    segs_sgb = segmentos_de_shapefile(transform, dem.shape, res)
    saidas_sgb = {}
    if segs_sgb:
        saidas_sgb, _ = rasterizar_e_derivar(
            segs_sgb, familias, dem.shape, res, raio_m, "_sgb",
            transform, epsg, salvar)
    else:
        print("[SGB] nenhum shapefile em data/raw/estrutural — rota pulada "
              "(coloque os .shp lá e reexecute para ativar)")

    # ---- Atualiza covariáveis nos poços ----
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
    lin = np.clip(lins, 0, dem.shape[0] - 1)
    col = np.clip(cols, 0, dem.shape[1] - 1)

    cov = pd.read_parquet(DERIVED / "covariaveis_pocos.parquet")
    novas = {**saidas_auto, **saidas_sgb}
    ext = pd.DataFrame({"codigo_siagas": pocos["codigo_siagas"]})
    for nome, arr in novas.items():
        ext[nome] = arr[lin, col]
    cov = cov.drop(columns=[c for c in cov.columns
                            if c.startswith(("dens_lin", "dist_lin"))],
                   errors="ignore")
    cov = cov.merge(ext, on="codigo_siagas", how="left")
    cov.to_parquet(DERIVED / "covariaveis_pocos.parquet", index=False)

    # ---- Quicklook: segmentos + roseta ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(15, 7))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(hillshade(dem, res, 315), cmap="gray")
        for p0, p1 in segs_auto:
            ax1.plot([p0[1], p1[1]], [p0[0], p1[0]], "r-", lw=0.4)
        ax1.set_title(f"Lineamentos AUTO (n={len(segs_auto)})")
        ax1.axis("off")
        ax2 = fig.add_subplot(1, 2, 2, projection="polar")
        ax2.set_theta_zero_location("N"); ax2.set_theta_direction(-1)
        bins = np.radians(np.arange(0, 181, 10))
        h, _ = np.histogram(np.radians(az_auto), bins=bins)
        centros = (bins[:-1] + bins[1:]) / 2
        ax2.bar(centros, h, width=np.radians(9), color="firebrick")
        ax2.bar(centros + np.pi, h, width=np.radians(9), color="firebrick")
        ax2.set_title("Roseta de azimutes (AUTO)")
        fig.savefig(OUT / "quicklook_lineamentos.png", dpi=130,
                    bbox_inches="tight")
        print("Quicklook: data/outputs/quicklook_lineamentos.png")
    except Exception as e:
        print(f"Quicklook não gerado: {e}")

    # ---- Relatório ----
    print("\n===== RELATÓRIO s05b =====")
    for fam in list(familias) + ["total"]:
        n = int((np.array([familia_do_azimute(a, familias)
                           for a in az_auto]) == fam).sum()) \
            if fam != "total" else len(az_auto)
        print(f"Segmentos AUTO família {fam:8s}: {n}")
    print(f"Covariáveis novas nos poços: {len(novas)}")
    print(f"Relatório salvo em: {RELATORIO}")
    sys.stdout.arquivo.close()
    sys.stdout = sys.stdout.terminal


if __name__ == "__main__":
    main()
