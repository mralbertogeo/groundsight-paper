"""s04b — Superfícies de intemperismo v4: variografia + krigagem com
deriva de terreno (regression-kriging), com adoção por CV espacial.

Motivação (diagnóstico de 10/08/2026): a krigagem ordinária colapsou
as superfícies para a média regional (quase-pepita dos contatos de
perfurador), achatando as espessuras (esp_solo 10–20 m etc.) e
anulando seu poder preditivo.

Método:
  A. Reconstrói os contatos por poço (hidrolito do banco +
     contatos_do_perfil do s04) e junta covariáveis de terreno do
     parquet (config: superficies_rk.covariaveis_deriva);
  B. DIAGNÓSTICO por superfície: semivariograma empírico (fração
     pepita), correlação entre vizinhos (500 m) e R² da regressão de
     deriva — quanto do contato o terreno explica;
  C. REGRESSION-KRIGING: regressão linear na deriva de terreno +
     krigagem ordinária dos resíduos (fallback IDW), no MESMO grid do
     s04;
  D. CV ESPACIAL (blocos de 10 km, 5 folds): RMSE de predição nos
     poços de teste — krigagem ordinária (método atual) vs RK;
  E. REGRA PRÉ-REGISTRADA (config: reducao_rmse_min = 0.05): RK é
     adotada POR SUPERFÍCIE somente se reduzir o RMSE de CV em >= 5%.
     Superfícies adotadas substituem os prof_*.tif canônicos (backup
     prof_*_v1ok.tif) e as esp_* são recalculadas com empilhamento
     coerente.

Duas camadas de decisão (declaradas):
  - PRODUTOS (modelo 3D, profundidade recomendada): decididos AQUI,
    pela CV de predição de contato — validação direta do que a
    superfície afirma;
  - MODELO de favorabilidade: as esp_* novas seguem para s05→s07 e
    valem as regras de sempre (AUC espacial e externo vs baseline
    0,660/0,593), reportadas separadamente.

Execução:  docker compose run --rm pipeline python stages/s04b_superficies_rk.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from _comum import engine, config, DERIVED
from s04_geologia3d import contatos_do_perfil, salvar_geotiff

OUT = Path("/data/outputs")
GEO = DERIVED / "geologia_3d"
COV = DERIVED / "covariaveis"
RELATORIO = OUT / "s04b_relatorio.txt"

SUPERFICIES = ["base_solo", "topo_zona_rocha", "topo_rocha_sa"]


class Tee:
    def __init__(self, caminho):
        self.terminal = sys.stdout
        self.arquivo = open(caminho, "w", encoding="utf-8")

    def write(self, msg):
        self.terminal.write(msg); self.arquivo.write(msg)

    def flush(self):
        self.terminal.flush(); self.arquivo.flush()


# ---------------------------------------------------------------------
def krige_pontos(xt, yt, zt, xq, yq):
    """Prediz em pontos: krigagem ordinária com fallback IDW."""
    try:
        from pykrige.ok import OrdinaryKriging
        ok = OrdinaryKriging(xt, yt, zt, variogram_model="spherical",
                             nlags=12, enable_plotting=False)
        pred, _ = ok.execute("points", xq, yq)
        return np.asarray(pred)
    except Exception:
        from scipy.spatial import cKDTree
        tree = cKDTree(np.c_[xt, yt])
        dist, idx = tree.query(np.c_[xq, yq], k=min(12, len(xt)))
        w = 1.0 / np.maximum(dist, 1.0) ** 2
        return (w * zt[idx]).sum(axis=1) / w.sum(axis=1)


def krige_grade(xt, yt, zt, gx, gy):
    from s04_geologia3d import interpolar
    grade, metodo = interpolar(xt, yt, zt, gx, gy)
    return np.asarray(grade), metodo


def blocos(x, y, bloco_km, n_folds, seed=42):
    bx = (x // (bloco_km * 1000)).astype(int)
    by = (y // (bloco_km * 1000)).astype(int)
    ids = pd.Series(bx.astype(str)) + "_" + pd.Series(by.astype(str))
    rng = np.random.default_rng(seed)
    unicos = np.array(sorted(ids.unique()))
    rng.shuffle(unicos)
    mapa = {b: i % n_folds for i, b in enumerate(unicos)}
    return ids.map(mapa).values


def semivario(x, y, z, bins=(0, 300, 600, 1200, 2500, 5000, 10000, 20000)):
    rng = np.random.default_rng(42)
    n = len(z)
    m = min(300_000, n * (n - 1) // 2)
    ii = rng.integers(0, n, m); jj = rng.integers(0, n, m)
    ok = ii != jj
    ii, jj = ii[ok], jj[ok]
    h = np.hypot(x[ii] - x[jj], y[ii] - y[jj])
    g = 0.5 * (z[ii] - z[jj]) ** 2
    linhas = []
    for a, b in zip(bins[:-1], bins[1:]):
        sel = (h > a) & (h <= b)
        if sel.sum() >= 80:
            linhas.append((a, b, int(sel.sum()), float(g[sel].mean())))
    return linhas


# ---------------------------------------------------------------------
def main():
    import rasterio
    from rasterio.warp import reproject, Resampling
    import geopandas as gpd
    from sklearn.linear_model import LinearRegression

    OUT.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(RELATORIO)
    cfg = config().get("superficies_rk", {})
    covs_deriva = cfg.get("covariaveis_deriva",
                          ["mde", "declividade", "tri", "twi", "hand",
                           "dist_drenagem"])
    red_min = float(cfg.get("reducao_rmse_min", 0.05))
    bloco_km = float(cfg.get("bloco_km", 10))
    n_folds = int(cfg.get("n_folds", 5))

    print("===== s04b — SUPERFÍCIES v4 (regression-kriging) =====")
    print(f"Deriva de terreno: {covs_deriva}")
    print(f"Regra pré-registrada: adotar RK por superfície se o RMSE "
          f"de CV espacial cair >= {red_min:.0%}\n")

    # ---------- A. Contatos + covariáveis nos poços ----------
    eng = engine()
    with eng.connect() as conn:
        lito = pd.read_sql(text("""
            SELECT l.codigo_siagas, l.de_m, l.ate_m, l.hidrolito
            FROM litologia l JOIN pocos p USING (codigo_siagas)
            WHERE p.qaqc_aprovado AND l.hidrolito IS NOT NULL"""), conn)
        coords = pd.read_sql(text("""
            SELECT codigo_siagas, ST_X(geom) lon, ST_Y(geom) lat,
                   profundidade_m
            FROM pocos WHERE qaqc_aprovado"""), conn)

    contatos = []
    prof_idx = coords.set_index("codigo_siagas")["profundidade_m"]
    for cod, perfil in lito.groupby("codigo_siagas"):
        c = contatos_do_perfil(perfil, prof_idx.get(cod))
        c["codigo_siagas"] = cod
        contatos.append(c)
    dfc = pd.DataFrame(contatos).merge(coords, on="codigo_siagas")
    g = gpd.GeoDataFrame(dfc, geometry=gpd.points_from_xy(dfc.lon, dfc.lat),
                         crs="EPSG:4674").to_crs(epsg=31983)
    dfc["x"], dfc["y"] = g.geometry.x, g.geometry.y

    covp = pd.read_parquet(DERIVED / "covariaveis_pocos.parquet")
    faltam = [c for c in covs_deriva if c not in covp.columns]
    if faltam:
        raise RuntimeError(f"Covariáveis ausentes no parquet: {faltam}")
    dfc = dfc.merge(covp[["codigo_siagas"] + covs_deriva],
                    on="codigo_siagas", how="left")
    for c in covs_deriva:
        dfc[c] = pd.to_numeric(dfc[c], errors="coerce")
        dfc[c] = dfc[c].fillna(dfc[c].median())

    # ---------- grid idêntico ao do s04 ----------
    with rasterio.open(GEO / "prof_base_solo.tif") as src:
        t0, epsg = src.transform, src.crs.to_epsg()
        h0, w0 = src.shape
        res = t0.a
    gx = t0.c + res / 2 + res * np.arange(w0)
    gy = (t0.f - res / 2 - res * np.arange(h0))[::-1]      # ascendente

    # deriva no grid: covariáveis 60 m -> grid das superfícies
    deriva_grid = {}
    for c in covs_deriva:
        with rasterio.open(COV / f"{c}.tif") as src:
            arr = np.full((h0, w0), np.nan, dtype="float32")
            reproject(rasterio.band(src, 1), arr, dst_transform=t0,
                      dst_crs=f"EPSG:{epsg}",
                      resampling=Resampling.bilinear, dst_nodata=np.nan)
        arr = np.nan_to_num(arr, nan=float(np.nanmedian(arr)))
        deriva_grid[c] = np.flipud(arr)                    # gy ascendente

    resultados = {}
    superficies_novas = {}
    for nome in SUPERFICIES:
        sub = dfc.dropna(subset=[nome]).reset_index(drop=True)
        x, y = sub["x"].values, sub["y"].values
        z = sub[nome].values.astype(float)
        X = sub[covs_deriva].values
        print(f"\n===== {nome} (n={len(sub)}) =====")

        # ---------- B. diagnóstico ----------
        sv = semivario(x, y, z)
        var = z.var()
        if sv:
            print(f"{'lag':>14s} {'n':>8s} {'semivar':>9s} {'%patamar':>9s}")
            for a, b, n, gm in sv:
                print(f"{a:6.0f}–{b:6.0f} {n:8d} {gm:9.1f} "
                      f"{100 * gm / var:8.0f}%")
            pepita = min(1.0, sv[0][3] / var)
            print(f"Fração pepita: {pepita:.0%}")
        reg_diag = LinearRegression().fit(X, z)
        r2_der = reg_diag.score(X, z)
        print(f"R² da deriva de terreno (in-sample): {r2_der:.3f}")

        # ---------- D. CV espacial: OK vs RK ----------
        folds = blocos(x, y, bloco_km, n_folds)
        e_ok, e_rk = [], []
        for f in range(n_folds):
            tr, te = folds != f, folds == f
            if te.sum() < 10:
                continue
            p_ok = krige_pontos(x[tr], y[tr], z[tr], x[te], y[te])
            e_ok.extend((p_ok - z[te]).tolist())
            reg = LinearRegression().fit(X[tr], z[tr])
            resid = z[tr] - reg.predict(X[tr])
            p_res = krige_pontos(x[tr], y[tr], resid, x[te], y[te])
            p_rk = reg.predict(X[te]) + p_res
            e_rk.extend((p_rk - z[te]).tolist())
        rmse_ok = float(np.sqrt(np.mean(np.square(e_ok))))
        rmse_rk = float(np.sqrt(np.mean(np.square(e_rk))))
        reducao = (rmse_ok - rmse_rk) / rmse_ok
        adotar = reducao >= red_min
        print(f"RMSE CV espacial — OK: {rmse_ok:.2f} m | RK: {rmse_rk:.2f} m"
              f" | redução: {reducao:+.1%} -> "
              f"{'RK ADOTADA' if adotar else 'mantém OK'}")
        resultados[nome] = {"rmse_ok": rmse_ok, "rmse_rk": rmse_rk,
                            "reducao": reducao, "adotar": adotar,
                            "r2_deriva": r2_der}

        # ---------- C/E. superfície final ----------
        if adotar:
            reg = LinearRegression().fit(X, z)
            resid = z - reg.predict(X)
            grade_res, metodo = krige_grade(x, y, resid, gx, gy)
            deriva = sum(coef * deriva_grid[c] for coef, c
                         in zip(reg.coef_, covs_deriva)) + reg.intercept_
            grade = np.maximum(deriva + grade_res, 0.0)
            superficies_novas[nome] = grade
            amp_v = np.percentile(z, [5, 95])
            amp_g = np.nanpercentile(grade, [5, 95])
            print(f"Amplitude p5–p95 — poços: {amp_v[0]:.0f}–{amp_v[1]:.0f} m"
                  f" | superfície RK: {amp_g[0]:.0f}–{amp_g[1]:.0f} m"
                  f" ({metodo} nos resíduos)")

    # ---------- E. gravação com empilhamento coerente ----------
    if superficies_novas:
        atuais = {}
        for nome in SUPERFICIES:
            if nome in superficies_novas:
                atuais[nome] = superficies_novas[nome]
            else:
                with rasterio.open(GEO / f"prof_{nome}.tif") as src:
                    atuais[nome] = np.flipud(src.read(1).astype(float))
        s1 = atuais["base_solo"]
        s2 = np.maximum(atuais["topo_zona_rocha"], s1)
        s3 = np.maximum(atuais["topo_rocha_sa"], s2)
        for nome in SUPERFICIES:
            origem = GEO / f"prof_{nome}.tif"
            backup = GEO / f"prof_{nome}_v1ok.tif"
            if not backup.exists():
                origem.rename(backup)
        salvar_geotiff(GEO / "prof_base_solo.tif", s1, gx, gy, epsg)
        salvar_geotiff(GEO / "prof_topo_zona_rocha.tif", s2, gx, gy, epsg)
        salvar_geotiff(GEO / "prof_topo_rocha_sa.tif", s3, gx, gy, epsg)
        salvar_geotiff(GEO / "esp_solo.tif", s1, gx, gy, epsg)
        salvar_geotiff(GEO / "esp_saprolito.tif", s2 - s1, gx, gy, epsg)
        salvar_geotiff(GEO / "esp_rocha_alterada.tif", s3 - s2, gx, gy, epsg)
        print("\nSuperfícies canônicas atualizadas (backup *_v1ok.tif); "
              "espessuras recalculadas com empilhamento coerente.")
    else:
        print("\nNenhuma superfície atingiu a redução mínima — "
              "método atual mantido integralmente (resultado registrado).")

    # ---------- quicklook comparativo ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        with rasterio.open(GEO / ("prof_topo_rocha_sa_v1ok.tif"
                                  if superficies_novas else
                                  "prof_topo_rocha_sa.tif")) as src:
            antigo = np.flipud(src.read(1).astype(float))
        novo = (np.maximum(superficies_novas.get("topo_rocha_sa", antigo),
                           0))
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        vmax = np.nanpercentile(novo, 98)
        for ax, arr, tit in [(axs[0], antigo, "OK (v1)"),
                             (axs[1], novo, "RK (v4)")]:
            im = ax.imshow(arr, origin="lower", cmap="viridis",
                           vmin=0, vmax=vmax)
            ax.set_title(f"prof. topo rocha sã — {tit}")
            fig.colorbar(im, ax=ax, shrink=.8)
            ax.axis("off")
        fig.savefig(OUT / "quicklook_superficies_v4.png", dpi=120,
                    bbox_inches="tight")
        print("Quicklook: data/outputs/quicklook_superficies_v4.png")
    except Exception as e:
        print(f"Quicklook não gerado: {e}")

    print(f"\nRelatório salvo em: {RELATORIO}")
    sys.stdout.arquivo.close()
    sys.stdout = sys.stdout.terminal


if __name__ == "__main__":
    main()
