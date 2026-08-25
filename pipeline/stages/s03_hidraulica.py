"""Etapa 3 — Derivação hidráulica com reconciliação de fontes (M3).

Para cada poço aprovado no QA/QC:
  1. Reconcilia NE/ND entre ficha e cadastro (prioridade: ficha; flag
     'nivel_divergente' se diferirem além de tolerancia_nivel_m);
  2. Determina Q/s em cascata de prioridade, registrando a proveniência
     em fonte_qs:
        calculado_ficha  = vazao/(ND-NE) do teste da ficha
        ficha            = Q/s pré-calculado da ficha
        cadastro         = Q/s da camada de cadastro
     Verificação cruzada: calculado × ficha divergentes além de
     tolerancia_qs_rel -> flag 'qs_inconsistente' (mantém o calculado);
  3. Regras 2.3 (ND>NE, vazão>0) e 2.4 (tempo >= mínimo) como flags;
  4. Intervalo captado, em hierarquia de confiança:
        filtros > trecho_aberto (base do revestimento ao fundo) >
        coluna_saturada (NE ao fundo — aproximação p/ cristalino sem
        construtivo, flag 'intervalo_aproximado') > indefinido (flag);
  5. T estimada por relação T=a·(Q/s)^b APENAS onde houver relação
     configurada para o domínio (v1: fissural sem relação — decisão
     declarada no config.yaml; K e carga h ficam para s04/s05);
  6. Regra 2.7: outliers de log10(Q/s) por domínio (> outlier_mad
     desvios absolutos medianos) -> flag 'qs_outlier'.

DECISÃO v1 (declarada): sem normalização de Jacob do Q/s — a duração
do teste é majoritariamente ausente nas fichas da AOI e a correção
exigiria parâmetros (r, S) não disponíveis. tempo_h é armazenado e
testes curtos são flagados; normalização fica como débito técnico.

Saída: tabela hidraulica_derivada + /data/derived/hidraulica_derivada.parquet

Execução:  docker compose run --rm pipeline python stages/s03_hidraulica.py
"""
import math

import numpy as np
import pandas as pd
from sqlalchemy import text

from _comum import engine, config, DERIVED


def dominio_do_texto(aquifero: str | None, mapa: dict) -> str:
    if not aquifero:
        return "nao_informado"
    t = aquifero.lower()
    for dom, termos in mapa.items():
        if any(term in t for term in termos):
            return dom
    return "outro"


def main():
    cfg = config()["hidraulica"]
    cfg_q = config()["qaqc"]
    eng = engine()

    with eng.connect() as conn:
        pocos = pd.read_sql(text("""
            SELECT codigo_siagas, profundidade_m, aquifero_cadastro
            FROM pocos WHERE qaqc_aprovado"""), conn)
        ficha = pd.read_sql(text("""
            SELECT codigo_siagas, vazao_m3h, ne_m, nd_m, tempo_h, qs_ficha
            FROM hidraulica_raw WHERE metodo='ficha_siagas'"""), conn)
        cad = pd.read_sql(text("""
            SELECT codigo_siagas, ne_m AS ne_cad, nd_m AS nd_cad,
                   qs_cadastro
            FROM hidraulica_raw WHERE metodo='cadastro_layer'"""), conn)
        filtros = pd.read_sql(text(
            "SELECT codigo_siagas, de_m, ate_m FROM filtros"), conn)
        rev = pd.read_sql(text(
            "SELECT codigo_siagas, ate_m FROM revestimento"), conn)

    df = (pocos.merge(ficha, on="codigo_siagas", how="left")
               .merge(cad, on="codigo_siagas", how="left"))

    mapa_dom = cfg["dominio_por_texto"]
    df["dominio"] = df["aquifero_cadastro"].apply(
        lambda a: dominio_do_texto(a, mapa_dom))

    registros = []
    flags_por_poco = {}

    def flag(cod, f):
        flags_por_poco.setdefault(cod, set()).add(f)

    filt_g = filtros.groupby("codigo_siagas")
    rev_g = rev.groupby("codigo_siagas")["ate_m"].max()

    for _, r in df.iterrows():
        cod = r["codigo_siagas"]

        # ---- 1. Reconciliação NE/ND ----
        ne = r["ne_m"] if pd.notna(r["ne_m"]) else r.get("ne_cad")
        nd = r["nd_m"] if pd.notna(r["nd_m"]) else r.get("nd_cad")
        for a, b, nome in [(r["ne_m"], r.get("ne_cad"), "ne"),
                           (r["nd_m"], r.get("nd_cad"), "nd")]:
            if pd.notna(a) and pd.notna(b) and abs(a - b) > cfg["tolerancia_nivel_m"]:
                flag(cod, "nivel_divergente")

        # ---- 3. Regras 2.3 / 2.4 ----
        if pd.notna(ne) and pd.notna(nd) and nd <= ne:
            flag(cod, "nd_menor_igual_ne")
        if pd.notna(r["vazao_m3h"]) and r["vazao_m3h"] <= 0:
            flag(cod, "vazao_invalida")
        if pd.notna(r["tempo_h"]) and r["tempo_h"] < cfg_q["tempo_bombeamento_min_h"]:
            flag(cod, "teste_curto")

        # ---- 2. Cascata do Q/s ----
        qs, fonte = None, None
        if (pd.notna(r["vazao_m3h"]) and r["vazao_m3h"] > 0
                and pd.notna(ne) and pd.notna(nd) and nd > ne):
            qs = float(r["vazao_m3h"]) / (float(nd) - float(ne))
            fonte = "calculado_ficha"
            if pd.notna(r["qs_ficha"]) and r["qs_ficha"] > 0:
                if abs(qs - r["qs_ficha"]) / r["qs_ficha"] > cfg["tolerancia_qs_rel"]:
                    flag(cod, "qs_inconsistente")
        elif pd.notna(r["qs_ficha"]) and r["qs_ficha"] > 0:
            qs, fonte = float(r["qs_ficha"]), "ficha"
        elif pd.notna(r["qs_cadastro"]) and r["qs_cadastro"] > 0:
            qs, fonte = float(r["qs_cadastro"]), "cadastro"

        # ---- 4. Intervalo captado ----
        prof = r["profundidade_m"]
        if cod in filt_g.groups:
            fs = filt_g.get_group(cod)
            tipo = "filtros"
            topo, base = float(fs["de_m"].min()), float(fs["ate_m"].max())
            lf = float((fs["ate_m"] - fs["de_m"]).sum())
        elif cod in rev_g.index and pd.notna(prof) and prof > rev_g[cod]:
            tipo = "trecho_aberto"
            topo, base = float(rev_g[cod]), float(prof)
            lf = base - topo
        elif pd.notna(ne) and pd.notna(prof) and float(prof) > float(ne):
            # Aproximação p/ cristalino sem construtivo: coluna saturada
            tipo = "coluna_saturada"
            topo, base = float(ne), float(prof)
            lf = base - topo
            flag(cod, "intervalo_aproximado")
        else:
            tipo, topo, base, lf = "indefinido", None, None, None
            flag(cod, "intervalo_indefinido")

        bsat = None
        if base is not None and pd.notna(ne):
            bsat = max(0.0, base - max(float(ne), topo))

        # ---- 5. T por relação configurada ----
        t_m2d, rel_nome = None, None
        rel = cfg["relacao_t_qs"].get(r["dominio"])
        if qs is not None and rel:
            qs_m2d = qs * 24.0  # m³/h/m -> m²/d
            t_m2d = rel["a"] * (qs_m2d ** rel["b"])
            rel_nome = rel["nome"]

        registros.append({
            "codigo_siagas": cod,
            "qs_m3h_m": qs,
            "log_qs": math.log10(qs) if qs else None,
            "fonte_qs": fonte,
            "ne_m": None if pd.isna(ne) else float(ne),
            "nd_m": None if pd.isna(nd) else float(nd),
            "tempo_h": None if pd.isna(r["tempo_h"]) else float(r["tempo_h"]),
            "t_m2_d": t_m2d, "relacao_t_qs": rel_nome,
            "lf_m": lf, "bsat_m": bsat,
            "tipo_intervalo": tipo,
            "topo_capt_m": topo, "base_capt_m": base,
            "dominio": r["dominio"],
        })

    dfd = pd.DataFrame(registros)

    # ---- 6. Regra 2.7 — outliers por domínio ----
    n_out = 0
    for dom, sub in dfd.dropna(subset=["log_qs"]).groupby("dominio"):
        if len(sub) < 30:
            continue
        med = sub["log_qs"].median()
        mad = (sub["log_qs"] - med).abs().median()
        if mad == 0:
            continue
        lim = cfg_q["outlier_mad"]
        for cod in sub.loc[(sub["log_qs"] - med).abs() / mad > lim,
                           "codigo_siagas"]:
            flag(cod, "qs_outlier")
            n_out += 1

    # ---- Gravação ----
    with eng.begin() as conn:
        conn.execute(text("DELETE FROM hidraulica_derivada"))
        for reg in registros:
            conn.execute(text("""
                INSERT INTO hidraulica_derivada
                  (codigo_siagas, qs_m3h_m, log_qs, fonte_qs, ne_m, nd_m,
                   tempo_h, t_m2_d, relacao_t_qs, lf_m, bsat_m,
                   tipo_intervalo, topo_capt_m, base_capt_m)
                VALUES (:codigo_siagas,:qs_m3h_m,:log_qs,:fonte_qs,:ne_m,
                        :nd_m,:tempo_h,:t_m2_d,:relacao_t_qs,:lf_m,:bsat_m,
                        :tipo_intervalo,:topo_capt_m,:base_capt_m)
            """), {k: v for k, v in reg.items() if k != "dominio"})
        for cod, fls in flags_por_poco.items():
            for fl in sorted(fls):
                conn.execute(text("""
                    UPDATE pocos SET qaqc_flags = array_append(qaqc_flags,:f)
                    WHERE codigo_siagas=:c
                      AND NOT (:f = ANY(qaqc_flags))"""),
                    {"c": cod, "f": fl})

    DERIVED.mkdir(parents=True, exist_ok=True)
    dfd.to_parquet(DERIVED / "hidraulica_derivada.parquet", index=False)

    # ---- Relatório ----
    com_qs = dfd.dropna(subset=["qs_m3h_m"])
    print("\n===== RELATÓRIO s03 (derivação hidráulica) =====")
    print(f"Poços aprovados processados:     {len(dfd)}")
    print(f"Poços com Q/s (variável-alvo):   {len(com_qs)}")
    print("Por fonte:")
    for fonte, n in com_qs["fonte_qs"].value_counts().items():
        print(f"   {fonte:18s} {n}")
    print("Intervalo captado:")
    for tipo, n in dfd["tipo_intervalo"].value_counts().items():
        print(f"   {tipo:18s} {n}")
    print("log10(Q/s) por domínio (mediana [p10; p90]):")
    for dom, sub in com_qs.groupby(dfd["dominio"]):
        q = sub["log_qs"]
        print(f"   {dom:15s} n={len(q):4d}  {q.median():6.2f} "
              f"[{q.quantile(.1):6.2f}; {q.quantile(.9):6.2f}]")
    print(f"Flags 2.7 qs_outlier:            {n_out}")
    print(f"Poços com T estimada:            {dfd['t_m2_d'].notna().sum()}"
          f"  (fissural sem relação — decisão v1)")
    print("Parquet: data/derived/hidraulica_derivada.parquet")


if __name__ == "__main__":
    main()
