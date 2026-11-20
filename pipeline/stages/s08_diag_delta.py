"""s08_diag_delta.py — O delta produtivo é regionalmente variável?

Para cada poço de alta produtividade (log Q/s >= q75), calcula o
aprofundamento na rocha sã (delta = profundidade do poço − topo da
rocha sã interpolado no local) e estratifica:

  A) Global (p10/p25/mediana/p75/p90);
  B) Por domínio hidrogeológico (fissural / poroso / outro);
  C) Por unidade captada;
  D) Por blocos espaciais de 20 km (mediana por bloco, com n mínimo)
     — a resposta direta à pergunta: as medianas de bloco variam
     o suficiente para justificar um delta espacialmente variável?

Gera mapa dos poços 'alta' coloridos pelo delta (quicklook) e
relatório em tela + txt.

Execução:  docker compose run --rm pipeline python stages/s08_diag_delta.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from _comum import engine, DERIVED

OUT = Path("/data/outputs")
COV = DERIVED / "covariaveis"
GEO = DERIVED / "geologia_3d"
RELATORIO = OUT / "s08_diag_delta_relatorio.txt"


class Tee:
    def __init__(self, caminho):
        self.terminal = sys.stdout
        self.arquivo = open(caminho, "w", encoding="utf-8")

    def write(self, msg):
        self.terminal.write(msg); self.arquivo.write(msg)

    def flush(self):
        self.terminal.flush(); self.arquivo.flush()


def resumo(s: pd.Series) -> str:
    q = s.quantile([.1, .25, .5, .75, .9])
    return (f"n={len(s):4d}  p10={q[.1]:5.0f}  p25={q[.25]:5.0f}  "
            f"med={q[.5]:5.0f}  p75={q[.75]:5.0f}  p90={q[.9]:5.0f}")


def main():
    import rasterio
    from rasterio.warp import reproject, Resampling
    import geopandas as gpd

    OUT.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(RELATORIO)

    with rasterio.open(COV / "mde.tif") as src:
        shape = (src.height, src.width)
        transform, epsg = src.transform, src.crs.to_epsg()
    with rasterio.open(GEO / "prof_topo_rocha_sa.tif") as src2:
        prof_rocha = np.full(shape, np.nan, dtype="float32")
        reproject(rasterio.band(src2, 1), prof_rocha,
                  dst_transform=transform, dst_crs=f"EPSG:{epsg}",
                  resampling=Resampling.bilinear, dst_nodata=np.nan)

    hd = pd.read_parquet(DERIVED / "hidraulica_derivada.parquet")
    eng = engine()
    with eng.connect() as conn:
        pocos = pd.read_sql(text("""
            SELECT codigo_siagas, profundidade_m,
                   ST_X(geom) lon, ST_Y(geom) lat
            FROM pocos WHERE qaqc_aprovado"""), conn)
    with eng.connect() as conn:
        uni = pd.read_sql(text(
            "SELECT codigo_siagas, unidade_captada "
            "FROM hidraulica_derivada"), conn)
    df = (hd[["codigo_siagas", "log_qs", "dominio"]]
          .merge(uni, on="codigo_siagas", how="left")
          .merge(pocos, on="codigo_siagas")
          .dropna(subset=["log_qs", "profundidade_m"]))

    q75 = df["log_qs"].quantile(0.75)
    alta = df[df["log_qs"] >= q75].copy()

    g = gpd.GeoDataFrame(alta, geometry=gpd.points_from_xy(
        alta.lon, alta.lat), crs="EPSG:4674").to_crs(epsg=epsg)
    inv = ~transform
    deltas, xs, ys = [], [], []
    for pt, prof in zip(g.geometry, g["profundidade_m"]):
        c, l = inv * (pt.x, pt.y)
        c = int(np.clip(c, 0, shape[1] - 1))
        l = int(np.clip(l, 0, shape[0] - 1))
        pr = prof_rocha[l, c]
        deltas.append(float(prof) - float(pr) if not np.isnan(pr) else np.nan)
        xs.append(pt.x); ys.append(pt.y)
    alta["delta"] = deltas
    alta["x"], alta["y"] = xs, ys
    alta = alta.dropna(subset=["delta"])
    alta = alta[(alta["delta"] > -20) & (alta["delta"] < 300)]

    print("===== DIAGNÓSTICO — DELTA PRODUTIVO (m na rocha sã) =====")
    print(f"Poços 'alta' analisados: {len(alta)} "
          f"(limiar log Q/s >= {q75:.2f})\n")
    print(f"A) GLOBAL:      {resumo(alta['delta'])}\n")

    print("B) POR DOMÍNIO:")
    for dom, sub in alta.groupby("dominio"):
        if len(sub) >= 15:
            print(f"   {dom:15s} {resumo(sub['delta'])}")

    print("\nC) POR UNIDADE CAPTADA:")
    for uni, sub in alta.groupby("unidade_captada"):
        if len(sub) >= 15:
            print(f"   {str(uni):15s} {resumo(sub['delta'])}")

    print("\nD) POR BLOCO ESPACIAL DE 20 km (n >= 10):")
    bx = (alta["x"] // 20000).astype(int)
    by = (alta["y"] // 20000).astype(int)
    alta["bloco"] = bx.astype(str) + "_" + by.astype(str)
    medianas = []
    for bloco, sub in alta.groupby("bloco"):
        if len(sub) >= 10:
            medianas.append({"bloco": bloco, "n": len(sub),
                             "mediana": sub["delta"].median()})
    dfb = pd.DataFrame(medianas).sort_values("mediana")
    print(dfb.to_string(index=False, float_format=lambda v: f"{v:.0f}"))
    if len(dfb):
        amp = dfb["mediana"].max() - dfb["mediana"].min()
        print(f"\nBlocos válidos: {len(dfb)} | medianas de "
              f"{dfb['mediana'].min():.0f} a {dfb['mediana'].max():.0f} m "
              f"(amplitude {amp:.0f} m)")
        print("""
Leitura:
  amplitude < ~30 m  -> delta escalar único é adequado (decidir teto);
  amplitude >= ~30 m -> delta espacialmente variável se justifica
                        (medianas de bloco interpoladas, com limites).""")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 8))
        sc = ax.scatter(alta["x"], alta["y"], c=alta["delta"], s=14,
                        cmap="viridis", vmin=0, vmax=200)
        fig.colorbar(sc, label="Delta produtivo (m)")
        ax.set_title("Poços 'alta' — aprofundamento na rocha sã")
        ax.set_aspect("equal")
        fig.savefig(OUT / "quicklook_delta_produtivo.png", dpi=130,
                    bbox_inches="tight")
        print("\nQuicklook: data/outputs/quicklook_delta_produtivo.png")
    except Exception as e:
        print(f"Quicklook não gerado: {e}")

    print(f"Relatório salvo em: {RELATORIO}")
    sys.stdout.arquivo.close()
    sys.stdout = sys.stdout.terminal


if __name__ == "__main__":
    main()
