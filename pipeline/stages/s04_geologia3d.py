"""Etapa 4 — Geologia 3D por superfícies de intemperismo (M4, v1).

Blocos:
  A. Classificação: aplica o dicionário de-para às descrições
     litológicas (litologia.hidrolito) e exporta amostra estratificada
     de 50 descrições para revisão do hidrogeólogo (bloqueante);
  B. Contatos por poço: base do solo/colúvio, topo da zona de rocha
     (base do saprolito) e topo da rocha sã, derivados do perfil
     classificado — apenas de poços que atingiram cada contato
     (censura declarada);
  C. Superfícies: interpolação (krigagem ordinária; fallback IDW) das
     PROFUNDIDADES dos contatos em grid regular. DECISÃO v1 declarada:
     interpola-se profundidade (não cota) — hipótese de perfil de
     intemperismo subparalelo ao terreno; conversão a cotas ocorrerá
     no s05 com o MDE;
  D. Cruzamento: fração do intervalo captado em cada classe, com
     hierarquia de método: perfil do próprio poço (cobertura >=
     cobertura_min_log) > superfícies interpoladas no ponto.
     conf_geo3d = true somente para perfil próprio + intervalo não
     aproximado. Atualiza hidraulica_derivada.

Saídas: litologia.hidrolito; /data/outputs/revisao_dicionario.csv;
rasters GeoTIFF em /data/derived/geologia_3d/; quicklook PNG;
hidraulica_derivada.{unidade_captada, frac_captada, conf_geo3d}.

Execução:  docker compose run --rm pipeline python stages/s04_geologia3d.py
"""
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sqlalchemy import text

from _comum import engine, config, DERIVED

OUT = Path("/data/outputs")
GEO = DERIVED / "geologia_3d"

CLASSES = ["solo_coluviao", "aluviao", "saprolito", "rocha_fraturada",
           "rocha_sa", "rocha_sa_sed"]
COBERTURA = ("solo_coluviao", "aluviao")
ZONA_ROCHA = ("rocha_fraturada", "rocha_sa", "rocha_sa_sed")


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s).lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def carregar_dicionario() -> list[tuple[str, list[str]]]:
    cfg = config()["geologia_3d"]
    with open(cfg["dicionario_litologico"], encoding="utf-8") as f:
        d = yaml.safe_load(f)
    return [(classe, [norm(t) for t in termos]) for classe, termos in d.items()]


def classificar(descricao: str, dicionario) -> str | None:
    t = norm(descricao)
    for classe, termos in dicionario:
        if any(term in t for term in termos):
            return classe
    return None


# ----------------------------------------------------------------------
# B. Contatos por poço
# ----------------------------------------------------------------------
def contatos_do_perfil(perfil: pd.DataFrame, prof_poco) -> dict:
    """Deriva profundidades de contato de um perfil classificado.
    Retorna somente contatos efetivamente atingidos (sem extrapolação)."""
    p = perfil.sort_values("de_m")
    out = {"base_solo": None, "topo_zona_rocha": None, "topo_rocha_sa": None}

    # base do solo: fim da sequência contígua de solo a partir do topo
    base = None
    for _, r in p.iterrows():
        if r["hidrolito"] in COBERTURA and (base is None or r["de_m"] <= base + 0.5):
            base = r["ate_m"]
        else:
            break
    if base is not None:
        out["base_solo"] = float(base)
    elif len(p) and p.iloc[0]["hidrolito"] not in COBERTURA:
        out["base_solo"] = float(p.iloc[0]["de_m"])  # sem solo: contato no topo

    # topo da zona de rocha (base do saprolito): primeiro intervalo
    # rocha_alterada ou rocha_sa
    zr = p[p["hidrolito"].isin(ZONA_ROCHA)]
    if len(zr):
        out["topo_zona_rocha"] = float(zr.iloc[0]["de_m"])

    # topo da rocha sã
    rs = p[p["hidrolito"].isin(["rocha_sa", "rocha_sa_sed"])]
    if len(rs):
        out["topo_rocha_sa"] = float(rs.iloc[0]["de_m"])

    return out


# ----------------------------------------------------------------------
# C. Interpolação (krigagem com fallback IDW)
# ----------------------------------------------------------------------
def interpolar(x, y, z, gx, gy):
    try:
        from pykrige.ok import OrdinaryKriging
        ok = OrdinaryKriging(x, y, z, variogram_model="spherical",
                             nlags=12, enable_plotting=False)
        grade, _ = ok.execute("grid", gx, gy)
        return np.asarray(grade), "krigagem_ordinaria_esferico"
    except Exception as e:
        print(f"   Krigagem falhou ({e}); usando IDW.")
        from scipy.spatial import cKDTree
        tree = cKDTree(np.c_[x, y])
        gxx, gyy = np.meshgrid(gx, gy)
        pts = np.c_[gxx.ravel(), gyy.ravel()]
        dist, idx = tree.query(pts, k=min(12, len(x)))
        w = 1.0 / np.maximum(dist, 1.0) ** 2
        vals = (w * z[idx]).sum(axis=1) / w.sum(axis=1)
        return vals.reshape(gyy.shape), "idw_k12_p2"


def salvar_geotiff(caminho, arr, gx, gy, epsg):
    import rasterio
    from rasterio.transform import from_origin
    res = gx[1] - gx[0]
    transform = from_origin(gx[0] - res / 2, gy[-1] + res / 2, res, res)
    with rasterio.open(
            caminho, "w", driver="GTiff", height=arr.shape[0],
            width=arr.shape[1], count=1, dtype="float32",
            crs=f"EPSG:{epsg}", transform=transform, nodata=-9999) as dst:
        dst.write(np.flipud(arr).astype("float32"), 1)


def main():
    import os
    cfg = config()["geologia_3d"]
    epsg = int(os.environ.get("EPSG_TRABALHO", "31983"))
    res = float(cfg.get("res_superficies_m", 300))
    eng = engine()

    # ================= A. Classificação =================
    dicionario = carregar_dicionario()
    with eng.connect() as conn:
        lito = pd.read_sql(text("""
            SELECT l.ctid::text AS rid, l.codigo_siagas, l.de_m, l.ate_m,
                   l.descricao_raw
            FROM litologia l JOIN pocos p USING (codigo_siagas)
            WHERE p.qaqc_aprovado"""), conn)

    lito["hidrolito"] = lito["descricao_raw"].apply(
        lambda d: classificar(d, dicionario) if pd.notna(d) else None)

    with eng.begin() as conn:
        conn.execute(text("UPDATE litologia SET hidrolito=NULL"))
        for _, r in lito.dropna(subset=["hidrolito"]).iterrows():
            conn.execute(text("""
                UPDATE litologia SET hidrolito=:h
                WHERE codigo_siagas=:c AND de_m=:de AND ate_m=:ate"""),
                {"h": r["hidrolito"], "c": r["codigo_siagas"],
                 "de": r["de_m"], "ate": r["ate_m"]})

    n_tot = len(lito)
    n_cls = int(lito["hidrolito"].notna().sum())
    cobertura = n_cls / n_tot if n_tot else 0

    # Amostra de revisão: até 12 por classe + 12 não classificadas
    OUT.mkdir(parents=True, exist_ok=True)
    partes = []
    for classe in CLASSES:
        sub = lito[lito["hidrolito"] == classe]
        partes.append(sub.sample(min(12, len(sub)), random_state=42))
    nc = lito[lito["hidrolito"].isna()]
    partes.append(nc.sample(min(12, len(nc)), random_state=42))
    amostra = pd.concat(partes)[
        ["codigo_siagas", "de_m", "ate_m", "descricao_raw", "hidrolito"]]
    amostra["classificacao_correta_SIM_NAO"] = ""
    amostra["classe_correta_se_NAO"] = ""
    amostra.to_csv(OUT / "revisao_dicionario.csv", index=False,
                   sep=";", encoding="utf-8-sig")

    nao_cls = (lito.loc[lito["hidrolito"].isna(), "descricao_raw"]
               .value_counts().head(15))

    # ================= B. Contatos =================
    with eng.connect() as conn:
        coords = pd.read_sql(text("""
            SELECT codigo_siagas, ST_X(geom) lon, ST_Y(geom) lat,
                   profundidade_m
            FROM pocos WHERE qaqc_aprovado"""), conn)

    contatos = []
    for cod, perfil in lito.dropna(subset=["hidrolito"]).groupby("codigo_siagas"):
        prof = coords.set_index("codigo_siagas")["profundidade_m"].get(cod)
        c = contatos_do_perfil(perfil, prof)
        c["codigo_siagas"] = cod
        contatos.append(c)
    # Todos os poços aprovados entram (fallback de superfícies vale
    # também para poços sem perfil litológico próprio)
    dfc = coords.merge(pd.DataFrame(contatos), on="codigo_siagas",
                       how="left")

    # Projeta para o EPSG de trabalho
    import geopandas as gpd
    g = gpd.GeoDataFrame(dfc, geometry=gpd.points_from_xy(dfc.lon, dfc.lat),
                         crs="EPSG:4674").to_crs(epsg=epsg)
    dfc["x"], dfc["y"] = g.geometry.x, g.geometry.y

    # ================= C. Superfícies =================
    GEO.mkdir(parents=True, exist_ok=True)
    x0, x1 = dfc.x.min() - res, dfc.x.max() + res
    y0, y1 = dfc.y.min() - res, dfc.y.max() + res
    gx = np.arange(x0, x1, res)
    gy = np.arange(y0, y1, res)

    superficies = {}
    metodos = {}
    for nome in ["base_solo", "topo_zona_rocha", "topo_rocha_sa"]:
        sub = dfc.dropna(subset=[nome])
        print(f"[{nome}] poços condicionantes: {len(sub)}")
        if len(sub) < 30:
            print(f"   Insuficiente (<30) — superfície não gerada.")
            continue
        grade, metodo = interpolar(sub.x.values, sub.y.values,
                                   sub[nome].values.astype(float), gx, gy)
        grade = np.maximum(grade, 0.0)
        superficies[nome] = grade
        metodos[nome] = metodo
        salvar_geotiff(GEO / f"prof_{nome}.tif", grade, gx, gy, epsg)

    # Coerência de empilhamento + espessuras
    if all(k in superficies for k in
           ["base_solo", "topo_zona_rocha", "topo_rocha_sa"]):
        s1 = superficies["base_solo"]
        s2 = np.maximum(superficies["topo_zona_rocha"], s1)
        s3 = np.maximum(superficies["topo_rocha_sa"], s2)
        superficies["topo_zona_rocha"], superficies["topo_rocha_sa"] = s2, s3
        salvar_geotiff(GEO / "esp_solo.tif", s1, gx, gy, epsg)
        salvar_geotiff(GEO / "esp_saprolito.tif", s2 - s1, gx, gy, epsg)
        salvar_geotiff(GEO / "esp_rocha_alterada.tif", s3 - s2, gx, gy, epsg)

        # Quicklook
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(9, 7))
            im = ax.imshow(s2 - s1, origin="lower",
                           extent=[x0, x1, y0, y1], cmap="viridis")
            ax.scatter(dfc.x, dfc.y, s=2, c="white", alpha=.5)
            fig.colorbar(im, label="Espessura de saprolito (m)")
            ax.set_title("Espessura de saprolito — GroundSight v1")
            fig.savefig(OUT / "quicklook_esp_saprolito.png", dpi=130,
                        bbox_inches="tight")
        except Exception as e:
            print(f"Quicklook não gerado: {e}")

    # ================= D. Cruzamento =================
    with eng.connect() as conn:
        deriv = pd.read_sql(text("""
            SELECT codigo_siagas, topo_capt_m, base_capt_m, tipo_intervalo
            FROM hidraulica_derivada
            WHERE topo_capt_m IS NOT NULL AND base_capt_m IS NOT NULL"""),
            conn)

    lito_ok = lito.dropna(subset=["hidrolito"])
    lito_g = lito_ok.groupby("codigo_siagas")
    dfc_i = dfc.set_index("codigo_siagas")
    cob_min = float(cfg.get("cobertura_min_log", 0.6))

    def fracoes_do_log(cod, topo, base):
        if cod not in lito_g.groups:
            return None
        p = lito_g.get_group(cod)
        fr = {c: 0.0 for c in CLASSES}
        cobertos = 0.0
        for _, r in p.iterrows():
            ov = min(base, r["ate_m"]) - max(topo, r["de_m"])
            if ov > 0:
                fr[r["hidrolito"]] += ov
                cobertos += ov
        if (base - topo) <= 0 or cobertos / (base - topo) < cob_min:
            return None
        return {c: v / cobertos for c, v in fr.items()}

    def fracoes_das_superficies(cod, topo, base):
        if cod not in dfc_i.index or not superficies:
            return None
        row = dfc_i.loc[cod]
        j = int(np.clip((row.x - gx[0]) / res, 0, len(gx) - 1))
        i = int(np.clip((row.y - gy[0]) / res, 0, len(gy) - 1))
        b1 = superficies["base_solo"][i, j]
        b2 = superficies["topo_zona_rocha"][i, j]
        b3 = superficies["topo_rocha_sa"][i, j]
        limites = [0.0, b1, b2, b3, 1e5]
        bins = ["solo_coluviao", "saprolito", "rocha_fraturada", "rocha_sa"]
        fr = {c: 0.0 for c in CLASSES}
        for k, classe in enumerate(bins):
            ov = min(base, limites[k + 1]) - max(topo, limites[k])
            fr[classe] += max(ov, 0.0)
        total = sum(fr.values())
        return {c: v / total for c, v in fr.items()} if total > 0 else None

    n_log, n_sup, n_sem = 0, 0, 0
    resultados = []
    for _, r in deriv.iterrows():
        cod, topo, base = r["codigo_siagas"], r["topo_capt_m"], r["base_capt_m"]
        fr = fracoes_do_log(cod, float(topo), float(base))
        metodo = "log_proprio"
        if fr is None:
            fr = fracoes_das_superficies(cod, float(topo), float(base))
            metodo = "superficies"
        if fr is None:
            n_sem += 1
            continue
        n_log += metodo == "log_proprio"
        n_sup += metodo == "superficies"
        dominante = max(fr, key=fr.get)
        conf = (metodo == "log_proprio"
                and r["tipo_intervalo"] in ("filtros", "trecho_aberto"))
        resultados.append({"c": cod, "u": dominante,
                           "f": round(fr[dominante], 3), "cf": bool(conf)})

    with eng.begin() as conn:
        for res_ in resultados:
            conn.execute(text("""
                UPDATE hidraulica_derivada
                SET unidade_captada=:u, frac_captada=:f, conf_geo3d=:cf
                WHERE codigo_siagas=:c"""), res_)

    # ================= Relatório =================
    print("\n===== RELATÓRIO s04 (geologia 3D) =====")
    print(f"Intervalos litológicos:            {n_tot}")
    print(f"Classificados pelo dicionário:     {n_cls} ({cobertura:.1%})")
    print("Não classificados mais frequentes:")
    for desc, n in nao_cls.items():
        print(f"   {n:4d}  {str(desc)[:70]}")
    for nome, met in metodos.items():
        print(f"Superfície {nome}: {met}")
    print(f"Cruzamento — perfil próprio:       {n_log}")
    print(f"Cruzamento — superfícies:          {n_sup}")
    print(f"Cruzamento — sem atribuição:       {n_sem}")

    with eng.connect() as conn:
        med = pd.read_sql(text("""
            SELECT unidade_captada, count(*) n,
                   round(percentile_cont(0.5) WITHIN GROUP
                         (ORDER BY log_qs)::numeric, 2) med_log_qs
            FROM hidraulica_derivada
            WHERE log_qs IS NOT NULL AND unidade_captada IS NOT NULL
            GROUP BY 1 ORDER BY n DESC"""), conn)
    print("log10(Q/s) mediano por unidade captada:")
    print(med.to_string(index=False))
    print(f"\nRevisão do dicionário (SUA TAREFA): data/outputs/revisao_dicionario.csv")
    print("Quicklook: data/outputs/quicklook_esp_saprolito.png")


if __name__ == "__main__":
    main()
