"""fig_export.py — Figuras cartográficas do artigo, direto da instância.

Gera em /data/outputs/artigo/ :
  fig5_<area>.png/.pdf : mapa de favorabilidade (probabilidade) +
                         incerteza, rótulos em INGLÊS, com grade de
                         coordenadas, barra de escala e norte;
  fig7_<area>.png/.pdf : roseta de azimutes dos lineamentos (EN);
  fig1_insumos_<area>.gpkg : poços aprovados + AOI (para a Figura 1
                             de localização, composta depois).

Rode NAS DUAS instâncias:
  docker compose run --rm pipeline python stages/fig_export.py area1
  docker compose run --rm pipeline python stages/fig_export.py area2
(o argumento é só o sufixo dos nomes de arquivo)
"""
import sys
from pathlib import Path

import numpy as np

from _comum import DERIVED

OUT = Path("/data/outputs/artigo")
ROTULO_AREA = {"area1": "Area 1 (Paraná Basin border)",
               "area2": "Area 2 (Taubaté rift)"}


def barra_escala(ax, transform, shape):
    """Barra de 20 km no canto inferior esquerdo."""
    res = transform.a
    km20 = 20_000 / res
    x0, y0 = shape[1] * 0.05, shape[0] * 0.95
    ax.plot([x0, x0 + km20], [y0, y0], "k-", lw=2.5)
    ax.plot([x0, x0], [y0 - shape[0]*0.008, y0 + shape[0]*0.008], "k-", lw=2.5)
    ax.plot([x0 + km20]*2, [y0 - shape[0]*0.008, y0 + shape[0]*0.008],
            "k-", lw=2.5)
    ax.text(x0 + km20/2, y0 - shape[0]*0.015, "20 km", ha="center",
            fontsize=8)


def norte(ax, shape):
    x, y = shape[1]*0.95, shape[0]*0.10
    ax.annotate("N", xy=(x, y - shape[0]*0.05), xytext=(x, y),
                ha="center", fontsize=11, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.5))


def main():
    import rasterio
    import geopandas as gpd
    import pandas as pd
    from sqlalchemy import text
    from _comum import engine
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    area = sys.argv[1] if len(sys.argv) > 1 else "areaX"
    rotulo = ROTULO_AREA.get(area, area)
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "figure.dpi": 300})

    # ---------------- FIG 5: probabilidade + incerteza ----------------
    with rasterio.open(DERIVED / "covariaveis" / "favorabilidade_prob.tif") as s:
        prob = s.read(1).astype(float)
        transform, shape = s.transform, s.shape
    # incerteza: procurar o raster de divergência
    inc = None
    p = DERIVED / "covariaveis" / "incerteza_modelos.tif"
    if p.exists():
        with rasterio.open(p) as s:
            inc = s.read(1).astype(float)

    n_paineis = 2 if inc is not None else 1
    fig, axs = plt.subplots(1, n_paineis,
                            figsize=(7.1 if n_paineis == 2 else 3.6, 3.6))
    axs = np.atleast_1d(axs)
    im0 = axs[0].imshow(np.ma.masked_invalid(prob), cmap="RdYlGn",
                        vmin=0, vmax=np.nanpercentile(prob, 99))
    axs[0].set_title(f"(a) P(high favorability) — {rotulo}", fontsize=9)
    fig.colorbar(im0, ax=axs[0], shrink=0.75, label="probability")
    if inc is not None:
        im1 = axs[1].imshow(np.ma.masked_invalid(inc), cmap="magma",
                            vmin=0, vmax=np.nanpercentile(inc, 99))
        axs[1].set_title("(b) Inter-model disagreement", fontsize=9)
        fig.colorbar(im1, ax=axs[1], shrink=0.75, label="|RF − XGB|")
    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([])
        barra_escala(ax, transform, shape)
        norte(ax, shape)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig5_{area}.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"fig5_{area} ok" + ("" if inc is not None
                               else " (SEM painel de incerteza — raster "
                                    "não encontrado; me informe o nome)"))

    # ---------------- FIG 7: roseta ----------------
    try:
        candidatos = []
        for base in (DERIVED, Path("/data/raw/estrutural")):
            for pad in ("**/*linea*.gpkg", "**/*linea*.shp",
                        "**/*linea*.parquet", "**/*linea*.geojson",
                        "**/*segmento*.gpkg", "**/*segmento*.parquet"):
                candidatos += sorted(base.glob(pad))
        if not candidatos:
            # fallback: TODAS as estruturas SGB de raw/estrutural
            for pad in ("*.shp", "*.gpkg"):
                candidatos += sorted(Path("/data/raw/estrutural").glob(pad))
        if not candidatos:
            raise FileNotFoundError("nenhum vetor estrutural encontrado")
        if len(candidatos) > 1:
            import pandas as pd
            partes = []
            for arq in candidatos:
                try:
                    partes.append(gpd.read_file(arq).to_crs(epsg=31983))
                except Exception:
                    pass
            lin = gpd.GeoDataFrame(pd.concat(partes, ignore_index=True))
            print(f"lineamentos: {len(candidatos)} fontes SGB combinadas")
            candidatos = [None]
        if candidatos[0] is not None:
            print(f"lineamentos: {candidatos[0]}")
            arq = candidatos[0]
            lin = (gpd.read_parquet(arq) if arq.suffix == ".parquet"
                   else gpd.read_file(arq))
        lin = lin[lin.geometry.type.isin(["LineString",
                                          "MultiLineString"])]
        lin = lin.explode(index_parts=False).reset_index(drop=True)

        # RECORTE PELA AOI — sem isto a roseta descreve o estado, não a
        # área de estudo (correção v4).
        import os as _os
        from shapely.geometry import box as _box
        bb = [float(v) for v in _os.environ["AOI_BBOX"].split(",")]
        aoi_g = gpd.GeoDataFrame(geometry=[_box(*bb)], crs="EPSG:4674")
        if lin.crs is None:
            lin = lin.set_crs("EPSG:4674")
        aoi_g = aoi_g.to_crs(lin.crs)
        antes = len(lin)
        lin = gpd.clip(lin, aoi_g)
        lin = lin[~lin.geometry.is_empty & lin.geometry.notna()]
        lin = lin.explode(index_parts=False).reset_index(drop=True)
        print(f"recorte pela AOI: {antes} -> {len(lin)} segmentos")
        if len(lin) == 0:
            raise RuntimeError("nenhuma estrutura dentro da AOI após o "
                               "recorte — verifique o CRS das fontes")
        az = None
        for c in ("azimute", "azimuth", "az"):
            if c in lin.columns:
                az = lin[c].values.astype(float) % 180
                break
        if az is None:
            g = lin.geometry
            p0 = g.apply(lambda l: l.coords[0]); p1 = g.apply(lambda l: l.coords[-1])
            dx = np.array([b[0]-a[0] for a, b in zip(p0, p1)])
            dy = np.array([b[1]-a[1] for a, b in zip(p0, p1)])
            az = (np.degrees(np.arctan2(dx, dy))) % 180
        comp = lin.to_crs(epsg=31983).length.values
        tet = np.radians(np.concatenate([az, az + 180]))
        pesos = np.concatenate([comp, comp])
        fig = plt.figure(figsize=(3.6, 3.6))
        ax = fig.add_subplot(111, projection="polar")
        ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
        ax.hist(tet, bins=36, weights=pesos, color="#8B0000")
        ax.set_title(f"Structural azimuths (SGB sources) — {rotulo}\n(n = {len(az):,})",
                     fontsize=9)
        ax.set_yticklabels([])
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"fig7_{area}.{ext}", dpi=300,
                        bbox_inches="tight")
        plt.close(fig)
        print(f"fig7_{area} ok")
    except Exception as e:
        print(f"fig7 não gerada ({e}) — me informe o caminho/formato dos "
              f"lineamentos desta instância")

    # ---------------- FIG 1: insumos ----------------
    import os
    eng = engine()
    with eng.connect() as conn:
        pocos = gpd.read_postgis(text(
            "SELECT codigo_siagas, qaqc_aprovado, geom FROM pocos"),
            conn, geom_col="geom")
    bbox = [float(x) for x in os.environ["AOI_BBOX"].split(",")]
    from shapely.geometry import box
    aoi = gpd.GeoDataFrame({"area": [rotulo]},
                           geometry=[box(*bbox)], crs="EPSG:4674")
    destino = OUT / f"fig1_insumos_{area}.gpkg"
    pocos.to_file(destino, layer="pocos", driver="GPKG")
    aoi.to_file(destino, layer="aoi", driver="GPKG")
    print(f"fig1_insumos_{area}.gpkg ok ({len(pocos)} poços + AOI)")
    print(f"\nTudo em: {OUT}")


if __name__ == "__main__":
    main()
