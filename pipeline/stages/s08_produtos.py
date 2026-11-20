"""Etapa 8 — Produtos de fechamento GroundSight v1 (pacote validado).

Conforme desfecho pré-registrado da v2.3, esta etapa NÃO gera mapa de
favorabilidade (reprovado na validação espacial). Produtos:

  1. Base auditada: GeoPackage com poços + QA/QC + hidráulica derivada;
  2. Modelo 3D em cota (reamostrado ao grid) + PROFUNDIDADE RECOMENDADA
     DE PERFURAÇÃO: prof_topo_rocha_sa + delta produtivo, onde o delta
     é DERIVADO DO ACERVO — mediana do aprofundamento na rocha sã dos
     poços de alta produtividade (q75), limitado a [min, max] do config;
  3. Superfície potenciométrica: krigagem da carga h (cota do NA), com
     verificação de sanidade (cargas acima do terreno truncadas e
     reportadas);
  4. Quicklooks e relatório em tela + txt.

Execução:  docker compose run --rm pipeline python stages/s08_produtos.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from _comum import engine, config, DERIVED

OUT = Path("/data/outputs")
COV = DERIVED / "covariaveis"
GEO = DERIVED / "geologia_3d"
PROD = Path("/data/outputs/produtos")
RELATORIO = OUT / "s08_relatorio.txt"


class Tee:
    def __init__(self, caminho):
        self.terminal = sys.stdout
        self.arquivo = open(caminho, "w", encoding="utf-8")

    def write(self, msg):
        self.terminal.write(msg); self.arquivo.write(msg)

    def flush(self):
        self.terminal.flush(); self.arquivo.flush()


def reprojetar_para_grid(caminho, mde_shape, transform, epsg):
    import rasterio
    from rasterio.warp import reproject, Resampling
    with rasterio.open(caminho) as src:
        destino = np.full(mde_shape, np.nan, dtype="float32")
        reproject(rasterio.band(src, 1), destino,
                  dst_transform=transform, dst_crs=f"EPSG:{epsg}",
                  resampling=Resampling.bilinear, dst_nodata=np.nan)
    return destino


def main():
    import rasterio
    import geopandas as gpd
    from scipy import ndimage
    from s05_covariaveis import salvar
    from s04_geologia3d import interpolar

    OUT.mkdir(parents=True, exist_ok=True)
    PROD.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(RELATORIO)
    cfg = config().get("produtos", {})
    eng = engine()

    with rasterio.open(COV / "mde.tif") as src:
        mde = src.read(1).astype(float)
        mde[mde == src.nodata] = np.nan
        transform, epsg = src.transform, src.crs.to_epsg()
        res = transform.a
    dem = np.nan_to_num(mde, nan=float(np.nanmedian(mde)))

    print("===== s08 — PRODUTOS DE FECHAMENTO (v1) =====")

    # ---------- 1. Base auditada ----------
    with eng.connect() as conn:
        pocos = gpd.read_postgis(
            "SELECT p.*, d.qs_m3h_m, d.log_qs, d.fonte_qs, d.ne_m, d.nd_m,"
            "       d.tipo_intervalo, d.topo_capt_m, d.base_capt_m,"
            "       d.lf_m, d.carga_h_m, d.unidade_captada, d.frac_captada,"
            "       d.conf_geo3d "
            "FROM pocos p LEFT JOIN hidraulica_derivada d USING (codigo_siagas)",
            conn, geom_col="geom")
    pocos["qaqc_flags"] = pocos["qaqc_flags"].apply(
        lambda v: ";".join(v) if isinstance(v, (list, tuple)) else (v or ""))
    pocos.to_file(PROD / "groundsight_base_auditada.gpkg",
                  layer="pocos", driver="GPKG")
    resumo = pocos.drop(columns="geom")
    resumo.to_excel(PROD / "groundsight_base_auditada.xlsx", index=False)
    print(f"[1] Base auditada: {len(pocos)} poços "
          f"(gpkg + xlsx em outputs/produtos)")

    # ---------- 2. Modelo 3D em cota + profundidade recomendada ----------
    prof_rocha = reprojetar_para_grid(GEO / "prof_topo_rocha_sa.tif",
                                      dem.shape, transform, epsg)
    cota_rocha = mde - prof_rocha
    salvar_prod = lambda nome, arr: salvar(nome, arr, transform, epsg) or \
        (COV / f"{nome}.tif").rename(PROD / f"{nome}.tif")
    salvar("cota_topo_rocha_sa", cota_rocha, transform, epsg)

    # Delta produtivo ESPACIALMENTE VARIÁVEL (diagnóstico de 10/08/2026:
    # amplitude de 141 m entre medianas de bloco — escalar único
    # inadequado). Método: medianas por bloco de bloco_delta_km,
    # interpoladas por IDW, com limites de segurança.
    # Exclusões declaradas: poços completados em solo/aluvião (não
    # atingem a rocha — recomendação não se aplica) e deltas negativos
    # (artefato da superfície suavizada).
    aprov = pocos.dropna(subset=["log_qs", "profundidade_m"])
    q75 = aprov["log_qs"].quantile(0.75)
    alta = aprov[(aprov["log_qs"] >= q75)
                 & (~aprov["unidade_captada"].isin(
                     ["solo_coluviao", "aluviao"]))].to_crs(epsg=epsg)
    inv = ~transform
    regs = []
    for _, r in alta.iterrows():
        c, l = inv * (r["geom"].x, r["geom"].y)
        c = int(np.clip(c, 0, dem.shape[1] - 1))
        l = int(np.clip(l, 0, dem.shape[0] - 1))
        pr = prof_rocha[l, c]
        if not np.isnan(pr):
            d = float(r["profundidade_m"]) - float(pr)
            if 0 < d < 300:
                regs.append({"x": r["geom"].x, "y": r["geom"].y, "d": d})
    dfa = pd.DataFrame(regs)

    bloco_m = float(cfg.get("bloco_delta_km", 20)) * 1000
    n_min = int(cfg.get("n_min_bloco", 8))
    dfa["bx"] = (dfa["x"] // bloco_m).astype(int)
    dfa["by"] = (dfa["y"] // bloco_m).astype(int)
    blocos = (dfa.groupby(["bx", "by"])
              .agg(n=("d", "size"), med=("d", "median"))
              .reset_index())
    blocos = blocos[blocos["n"] >= n_min]
    cx = (blocos["bx"] + 0.5) * bloco_m
    cy = (blocos["by"] + 0.5) * bloco_m

    # IDW das medianas de bloco para o grid
    from scipy.spatial import cKDTree
    tree = cKDTree(np.c_[cx, cy])
    xs = transform.c + (np.arange(dem.shape[1]) + 0.5) * res
    ys = transform.f - (np.arange(dem.shape[0]) + 0.5) * res
    gxx, gyy = np.meshgrid(xs, ys)
    dist, idx = tree.query(np.c_[gxx.ravel(), gyy.ravel()],
                           k=min(4, len(blocos)))
    if dist.ndim == 1:
        dist, idx = dist[:, None], idx[:, None]
    w = 1.0 / np.maximum(dist, 1.0) ** 2
    vals = blocos["med"].values[idx]
    delta_r = ((w * vals).sum(axis=1) / w.sum(axis=1)).reshape(dem.shape)

    d_min = float(cfg.get("aprofundamento_min_m", 20))
    d_max = float(cfg.get("aprofundamento_max_m", 160))
    delta_r = np.clip(delta_r, d_min, d_max)
    salvar("delta_produtivo", delta_r, transform, epsg)
    prof_rec = prof_rocha + delta_r
    salvar("prof_recomendada_perfuracao", prof_rec, transform, epsg)
    print(f"[2] Delta produtivo variável: {len(dfa)} poços alta válidos, "
          f"{len(blocos)} blocos de {bloco_m/1000:.0f} km (n>={n_min}); "
          f"medianas de bloco {blocos['med'].min():.0f}–"
          f"{blocos['med'].max():.0f} m; raster limitado a "
          f"[{d_min:.0f},{d_max:.0f}] m")
    print(f"[2] Rasters delta_produtivo.tif e "
          f"prof_recomendada_perfuracao.tif gerados "
          f"(= topo da rocha sã + delta local)")

    # ---------- 3. Potenciometria ----------
    # Método: interpola-se a PROFUNDIDADE do NA (NE — variável quase-
    # estacionária) e reconstrói-se a cota como MDE - profundidade.
    # Krigar a cota diretamente gera NA acima do terreno nos
    # interflúvios (poços concentrados nos vales) — 25% de truncamento
    # na primeira tentativa, contra <1% por este método.
    na = pocos.dropna(subset=["ne_m"]).to_crs(epsg=epsg)
    na = na[(na["ne_m"] >= 0) & (na["ne_m"] <= 120)]
    res_pot = float(cfg.get("res_potenciometria_m", 300))
    gx = np.arange(transform.c, transform.c + res * dem.shape[1], res_pot)
    gy = np.arange(transform.f - res * dem.shape[0], transform.f, res_pot)
    grade, metodo = interpolar(na.geometry.x.values, na.geometry.y.values,
                               na["ne_m"].values.astype(float), gx, gy)
    zoom = (dem.shape[0] / grade.shape[0], dem.shape[1] / grade.shape[1])
    prof_na = ndimage.zoom(grade[::-1, :], zoom, order=1)
    negativos = int((prof_na < 0).sum())
    prof_na = np.clip(prof_na, 0, None)
    cota_na = mde - prof_na
    salvar("cota_na", cota_na, transform, epsg)
    salvar("prof_na", prof_na, transform, epsg)
    pct = 100 * negativos / np.isfinite(mde).sum()
    print(f"[3] Potenciometria (profundidade do NA, {metodo}): "
          f"{len(na)} poços; células corrigidas (prof < 0): "
          f"{negativos} ({pct:.2f}%)")

    # ---------- 4. Quicklook ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        im0 = axs[0].imshow(np.clip(prof_rec, 0, 200), cmap="viridis")
        axs[0].set_title(f"Profundidade recomendada (m) = topo rocha sã "
                         f"+ delta local")
        fig.colorbar(im0, ax=axs[0], shrink=.8)
        im1 = axs[1].imshow(np.clip(prof_na, 0, 60), cmap="Blues_r")
        axs[1].set_title("Profundidade do NA (m)")
        fig.colorbar(im1, ax=axs[1], shrink=.8)
        fig.savefig(OUT / "quicklook_produtos.png", dpi=120,
                    bbox_inches="tight")
        print("[4] Quicklook: data/outputs/quicklook_produtos.png")
    except Exception as e:
        print(f"[4] Quicklook não gerado: {e}")

    print("\nRasters de produto no grid (data/derived/covariaveis/): "
          "cota_topo_rocha_sa, prof_recomendada_perfuracao, cota_na, prof_na")
    print(f"Relatório salvo em: {RELATORIO}")
    sys.stdout.arquivo.close()
    sys.stdout = sys.stdout.terminal


if __name__ == "__main__":
    main()
