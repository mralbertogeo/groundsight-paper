"""Etapa 1 — Extração SIAGAS para a AOI (M2).

Duas fases, reusando as rotas do SIAGAS Explorer:
  Fase A — CADASTRO em massa: camada ArcGIS REST do SGB, com filtro
           espacial pelo envelope AOI_BBOX (.env). Grava tabela `pocos`
           e parquet imutável em /data/raw.
  Fase B — FICHA por poço: HTML da ficha técnica (com cache em
           /data/raw/cache_siagas). Grava `litologia`, `filtros` e
           `hidraulica_raw`.

Idempotente: reexecutar atualiza cadastro (upsert) e só baixa fichas
ausentes do cache. Interrupções não perdem progresso (commit por lote).

Execução:  docker compose run --rm pipeline python stages/s01_extract_siagas.py
"""
import json
import os
import time

import pandas as pd
import requests
from sqlalchemy import text
from tqdm import tqdm

from _comum import engine, config, RAW
from siagas_ficha import (CACHE_DIR, buscar_html, parsear_html,
                          extrair_teste, limpar)

LAYER_URL = ("https://geoportal.sgb.gov.br/server/rest/services/"
             "Siagas_WebMap_MIL1/MapServer/0")


# ----------------------------------------------------------------------
# Fase A — cadastro em massa via ArcGIS REST com filtro espacial
# ----------------------------------------------------------------------
def _arcgis_get(session, url, params, retries=5, sleep=2.0):
    for tentativa in range(1, retries + 1):
        try:
            r = session.get(url, params={k: v for k, v in params.items()
                                         if v is not None}, timeout=120)
            r.raise_for_status()
            data = r.json()
            if "error" in data:
                raise RuntimeError(json.dumps(data["error"], ensure_ascii=False))
            return data
        except Exception:
            if tentativa == retries:
                raise
            time.sleep(sleep * tentativa)


def baixar_cadastro_aoi(bbox: list[float]) -> pd.DataFrame:
    """Baixa o cadastro dos poços dentro do envelope AOI (lon/lat WGS84)."""
    sess = requests.Session()
    sess.headers.update({"User-Agent": "GroundSight-s01/1.0"})
    query_url = LAYER_URL + "/query"

    geometria = json.dumps({
        "xmin": bbox[0], "ymin": bbox[1],
        "xmax": bbox[2], "ymax": bbox[3],
        "spatialReference": {"wkid": 4326},
    })
    filtro_espacial = {
        "geometry": geometria,
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
    }

    ids = _arcgis_get(sess, query_url, {
        "where": "1=1", "returnIdsOnly": "true", "f": "json",
        **filtro_espacial,
    }).get("objectIds") or []
    print(f"[Fase A] Poços na AOI segundo o servidor: {len(ids)}")
    if not ids:
        raise RuntimeError("Nenhum poço na AOI — confira AOI_BBOX no .env "
                           "(formato lon_min,lat_min,lon_max,lat_max).")

    registros = []
    lote = 500
    for i in tqdm(range(0, len(ids), lote), desc="Cadastro (lotes)"):
        feats = _arcgis_get(sess, query_url, {
            "objectIds": ",".join(map(str, ids[i:i + lote])),
            "outFields": "*", "returnGeometry": "true",
            "outSR": "4326", "f": "json",
        }).get("features", [])
        for f in feats:
            attrs = (f.get("attributes") or {}).copy()
            geom = f.get("geometry") or {}
            attrs["longitude"] = geom.get("x")
            attrs["latitude"] = geom.get("y")
            registros.append(attrs)
        time.sleep(0.1)

    df = pd.DataFrame(registros)
    df.columns = [c.lower().strip() for c in df.columns]
    return df


def _col(df, *candidatos):
    """Primeira coluna existente dentre os nomes candidatos, ou None."""
    for c in candidatos:
        if c in df.columns:
            return c
    return None



def _data_perfuracao(v):
    """Converte a data de perfuração da camada ArcGIS em date.

    O servidor do SGB entrega epoch (milissegundos ou segundos desde
    1970); fichas antigas podem trazer texto DD/MM/YYYY. Valores fora
    da faixa plausível (1900–hoje) retornam None.
    """
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return None
    n = pd.to_numeric(s, errors="coerce")
    if not pd.isna(n):
        try:
            if abs(n) > 1e11:
                dt = pd.to_datetime(n, unit="ms")
            else:
                dt = pd.to_datetime(n, unit="s")
        except Exception:
            return None
    else:
        dt = None
        for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
            try:
                dt = pd.to_datetime(s, format=fmt)
                break
            except Exception:
                continue
        if dt is None:
            return None
    if dt.year < 1900 or dt > pd.Timestamp.now():
        return None
    return dt.date()


def carregar_pocos(df: pd.DataFrame) -> int:
    """Upsert do cadastro na tabela pocos. Retorna nº de poços carregados."""
    c_cod = _col(df, "idt_ponto", "codigo", "cd_ponto")
    if c_cod is None:
        raise RuntimeError(f"Coluna de código não encontrada. "
                           f"Colunas: {sorted(df.columns)[:40]}")
    c_prof = _col(df, "num_profundidade", "profundidade")
    c_nat = _col(df, "str_natureza_ponto", "natureza")
    c_uso = _col(df, "str_uso_agua", "uso_agua")
    c_cota = _col(df, "num_cota", "cota", "num_cota_terreno")
    c_data = _col(df, "data_perfuracao", "dat_perfuracao")

    sql = text("""
        INSERT INTO pocos (codigo_siagas, geom, cota_terreno_m, cota_fonte,
                           profundidade_m, natureza, uso, data_perfuracao)
        VALUES (:cod, ST_SetSRID(ST_MakePoint(:lon, :lat), 4674),
                :cota, :cota_fonte, :prof, :nat, :uso, :dtp)
        ON CONFLICT (codigo_siagas) DO UPDATE SET
            geom = EXCLUDED.geom,
            cota_terreno_m = COALESCE(EXCLUDED.cota_terreno_m, pocos.cota_terreno_m),
            profundidade_m = COALESCE(EXCLUDED.profundidade_m, pocos.profundidade_m),
            natureza = COALESCE(EXCLUDED.natureza, pocos.natureza),
            uso = COALESCE(EXCLUDED.uso, pocos.uso)
    """)

    n = 0
    with engine().begin() as conn:
        for _, r in df.iterrows():
            lon, lat = r.get("longitude"), r.get("latitude")
            if pd.isna(lon) or pd.isna(lat):
                continue
            cota = pd.to_numeric(r.get(c_cota), errors="coerce") if c_cota else None
            conn.execute(sql, {
                "cod": str(r[c_cod]).strip(),
                "lon": float(lon), "lat": float(lat),
                "cota": None if pd.isna(cota) else float(cota),
                "cota_fonte": "siagas" if (c_cota and not pd.isna(cota)) else None,
                "prof": _f(r.get(c_prof)) if c_prof else None,
                "nat": limpar(str(r.get(c_nat, "") or "")) or None,
                "uso": limpar(str(r.get(c_uso, "") or "")) or None,
                "dtp": _data_perfuracao(r.get(c_data)) if c_data else None,
            })
            n += 1
    return n


def _f(v):
    v = pd.to_numeric(v, errors="coerce")
    return None if pd.isna(v) else float(v)



def carregar_hidraulica_cadastro(df: pd.DataFrame) -> int:
    """Insere NE/ND/Q-s vindos da camada de cadastro (metodo='cadastro_layer').
    Fonte independente da ficha; o s03 reconcilia as duas."""
    c_ne = _col(df, "ne")
    c_nd = _col(df, "nd")
    c_qs = _col(df, "num_vazao_especifica", "vazao_especifica")
    c_cod = _col(df, "idt_ponto", "codigo", "cd_ponto")
    if not any([c_ne, c_nd, c_qs]):
        return 0
    n = 0
    with engine().begin() as conn:
        conn.execute(text(
            "DELETE FROM hidraulica_raw WHERE metodo='cadastro_layer'"))
        for _, r in df.iterrows():
            ne, nd = _f(r.get(c_ne)) if c_ne else None, _f(r.get(c_nd)) if c_nd else None
            qs = _f(r.get(c_qs)) if c_qs else None
            if ne is None and nd is None and qs is None:
                continue
            lon, lat = r.get("longitude"), r.get("latitude")
            if pd.isna(lon) or pd.isna(lat):
                continue
            conn.execute(text("""INSERT INTO hidraulica_raw
                (codigo_siagas, ne_m, nd_m, qs_cadastro, metodo)
                VALUES (:c,:ne,:nd,:qs,'cadastro_layer')"""),
                {"c": str(r[c_cod]).strip(), "ne": ne, "nd": nd, "qs": qs})
            n += 1
    return n


# ----------------------------------------------------------------------
# Fase B — ficha técnica por poço
# ----------------------------------------------------------------------
def carregar_ficha(conn, codigo: str, dados: dict) -> dict:
    """Grava litologia, filtros e teste de bombeamento de um poço.
    Retorna contagens para o relatório."""
    conn.execute(text("DELETE FROM litologia WHERE codigo_siagas=:c"), {"c": codigo})
    conn.execute(text("DELETE FROM filtros WHERE codigo_siagas=:c"), {"c": codigo})
    conn.execute(text("DELETE FROM hidraulica_raw WHERE codigo_siagas=:c "
                      "AND metodo='ficha_siagas'"), {"c": codigo})
    conn.execute(text("DELETE FROM revestimento WHERE codigo_siagas=:c"), {"c": codigo})
    conn.execute(text("DELETE FROM entradas_dagua WHERE codigo_siagas=:c"), {"c": codigo})

    for li in dados.get("litologia", []):
        conn.execute(text("""INSERT INTO litologia
            (codigo_siagas, de_m, ate_m, descricao_raw)
            VALUES (:c,:de,:ate,:desc)"""),
            {"c": codigo, "de": li["topo"], "ate": li["base"],
             "desc": limpar(li.get("descricao", ""))})

    for ft in dados.get("filtros", []):
        conn.execute(text("""INSERT INTO filtros (codigo_siagas, de_m, ate_m)
            VALUES (:c,:de,:ate)"""),
            {"c": codigo, "de": ft["topo"], "ate": ft["base"]})

    for rv in dados.get("revestimento", []):
        conn.execute(text("""INSERT INTO revestimento (codigo_siagas, de_m, ate_m)
            VALUES (:c,:de,:ate)"""),
            {"c": codigo, "de": rv["topo"], "ate": rv["base"]})

    for prof in dados.get("entradas_dagua", []):
        conn.execute(text("""INSERT INTO entradas_dagua (codigo_siagas, profundidade_m)
            VALUES (:c,:p)"""), {"c": codigo, "p": prof})

    teste = extrair_teste(dados)
    tem_teste = any(v is not None for v in teste.values())
    if tem_teste:
        conn.execute(text("""INSERT INTO hidraulica_raw
            (codigo_siagas, vazao_m3h, ne_m, nd_m, tempo_h, qs_ficha, metodo)
            VALUES (:c,:q,:ne,:nd,:t,:qs,'ficha_siagas')"""),
            {"c": codigo, "q": teste["vazao_m3h"], "ne": teste["ne_m"],
             "nd": teste["nd_m"], "t": teste["tempo_h"],
             "qs": teste["qs_ficha"]})

    return {"lito": len(dados.get("litologia", [])),
            "filt": len(dados.get("filtros", [])),
            "rev": len(dados.get("revestimento", [])),
            "ent": len(dados.get("entradas_dagua", [])),
            "teste": int(tem_teste)}


def main():
    cfg = config().get("extracao", {})
    bbox = [float(x) for x in os.environ["AOI_BBOX"].split(",")]
    assert len(bbox) == 4, "AOI_BBOX deve ter 4 valores: lon_min,lat_min,lon_max,lat_max"

    # ---------- Fase A ----------
    df = baixar_cadastro_aoi(bbox)
    RAW.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RAW / "cadastro_aoi.parquet", index=False)  # bruto imutável
    n_pocos = carregar_pocos(df)
    print(f"[Fase A] Poços carregados/atualizados na tabela pocos: {n_pocos}")
    n_hidro_cad = carregar_hidraulica_cadastro(df)
    print(f"[Fase A] Poços com hidráulica no cadastro (NE/ND/Q-s): {n_hidro_cad}")

    # ---------- Fase B ----------
    with engine().connect() as conn:
        codigos = [r[0] for r in conn.execute(
            text("SELECT codigo_siagas FROM pocos ORDER BY codigo_siagas"))]

    max_pocos = cfg.get("max_pocos_ficha")  # limite p/ testes; None = todos
    if max_pocos:
        codigos = codigos[:int(max_pocos)]

    sleep_s = float(cfg.get("sleep_fichas_s", 0.5))
    stats = {"ok": 0, "falha": 0, "lito": 0, "filt": 0, "rev": 0,
             "ent": 0, "teste": 0}
    lote, buffer = 50, []

    eng = engine()
    for codigo in tqdm(codigos, desc="Fichas (Fase B)"):
        em_cache = (CACHE_DIR / f"{codigo}.html").exists()
        html = buscar_html(codigo)
        if html is None:
            stats["falha"] += 1
            continue
        buffer.append((codigo, parsear_html(html)))
        if len(buffer) >= lote:
            with eng.begin() as conn:
                for c, d in buffer:
                    r = carregar_ficha(conn, c, d)
                    stats["ok"] += 1
                    for k in ("lito", "filt", "rev", "ent", "teste"):
                        stats[k] += r[k]
            buffer = []
        if not em_cache:
            time.sleep(sleep_s)

    if buffer:
        with eng.begin() as conn:
            for c, d in buffer:
                r = carregar_ficha(conn, c, d)
                stats["ok"] += 1
                for k in ("lito", "filt", "rev", "ent", "teste"):
                    stats[k] += r[k]

    # ---------- Relatório ----------
    print("\n===== RELATÓRIO s01 =====")
    print(f"Poços no cadastro (AOI):        {n_pocos}")
    print(f"Fichas processadas:             {stats['ok']}")
    print(f"Fichas com falha de download:   {stats['falha']}")
    print(f"Intervalos litológicos:         {stats['lito']}")
    print(f"Seções filtrantes:              {stats['filt']}")
    print(f"Intervalos de revestimento:     {stats['rev']}")
    print(f"Entradas d'água (fraturas):     {stats['ent']}")
    print(f"Poços com teste de bombeamento: {stats['teste']}")
    print(f"Poços com hidráulica do cadastro: {n_hidro_cad}")
    print("=========================")
    if stats["falha"] > 0:
        print("Fichas com falha serão retentadas na próxima execução "
              "(não estão no cache).")


if __name__ == "__main__":
    main()
