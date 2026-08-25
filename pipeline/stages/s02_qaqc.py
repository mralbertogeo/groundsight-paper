"""Etapa 2 — QA/QC (M3).

Aplica as regras de qualidade sobre os dados extraídos no s01:
  2.1  Coordenadas dentro da AOI (bloqueante);
  2.2  Duplicatas espaciais < dist_duplicata_m — mantém o registro mais
       completo, reprova os demais (bloqueante para o descartado);
  2.5  Consistência construtiva: filtro abaixo do fundo do poço,
       filtros sobrepostos (flags);
  2.6  Continuidade litológica: lacunas > lacuna_litologia_m (flag);
  —    Flags manuais de conferência (config: qaqc.flags_manuais);
  —    Enriquecimento: aquifero_cadastro a partir do parquet da Fase A.

Regras que dependem de dados posteriores: 2.3/2.4/2.7 (hidráulica) são
aplicadas no s03; 2.8 (cota × MDE) no s05.

Semântica: qaqc_aprovado=false significa poço EXCLUÍDO da modelagem
(posição/cadastro inválidos). Flags não bloqueantes ficam em qaqc_flags.

Saída: tabela pocos atualizada + /data/derived/qaqc_relatorio.csv

Execução:  docker compose run --rm pipeline python stages/s02_qaqc.py
"""
import os
from collections import Counter

import pandas as pd
from sqlalchemy import text

from _comum import engine, config, RAW, DERIVED


def enriquecer_aquifero(conn) -> int:
    pq = RAW / "cadastro_aoi.parquet"
    if not pq.exists():
        print("Aviso: cadastro_aoi.parquet ausente — enriquecimento pulado.")
        return 0
    df = pd.read_parquet(pq)
    df.columns = [c.lower() for c in df.columns]
    c_cod = "idt_ponto" if "idt_ponto" in df.columns else "codigo"
    if "str_aquifero" not in df.columns:
        return 0
    n = 0
    for _, r in df[[c_cod, "str_aquifero"]].dropna().iterrows():
        conn.execute(text(
            "UPDATE pocos SET aquifero_cadastro=:a WHERE codigo_siagas=:c"),
            {"a": str(r["str_aquifero"]).strip(), "c": str(r[c_cod]).strip()})
        n += 1
    return n


def completude(conn) -> dict:
    """Escore de completude por poço (para desempate de duplicatas)."""
    sql = text("""
        SELECT p.codigo_siagas,
               (p.profundidade_m IS NOT NULL)::int
             + LEAST((SELECT count(*) FROM litologia l
                       WHERE l.codigo_siagas=p.codigo_siagas), 5)
             + LEAST((SELECT count(*) FROM filtros f
                       WHERE f.codigo_siagas=p.codigo_siagas), 3)
             + LEAST((SELECT count(*) FROM revestimento r
                       WHERE r.codigo_siagas=p.codigo_siagas), 3)
             + 3*(SELECT count(*) FROM hidraulica_raw h
                   WHERE h.codigo_siagas=p.codigo_siagas
                     AND h.metodo='ficha_siagas')
             + (SELECT count(*) FROM hidraulica_raw h
                 WHERE h.codigo_siagas=p.codigo_siagas
                   AND h.metodo='cadastro_layer') AS escore
        FROM pocos p
    """)
    return {r[0]: r[1] for r in conn.execute(sql)}


def main():
    cfg = config()["qaqc"]
    bbox = [float(x) for x in os.environ["AOI_BBOX"].split(",")]
    contagem = Counter()
    eng = engine()

    with eng.begin() as conn:
        # Reset (idempotência)
        conn.execute(text(
            "UPDATE pocos SET qaqc_aprovado=true, qaqc_flags='{}'"))
        n_total = conn.execute(text("SELECT count(*) FROM pocos")).scalar()

        # Enriquecimento
        n_aq = enriquecer_aquifero(conn)
        print(f"Poços com aquífero do cadastro: {n_aq}")

        # ---- 2.1 Coordenadas dentro da AOI ----
        r = conn.execute(text("""
            UPDATE pocos SET qaqc_aprovado=false,
                   qaqc_flags = array_append(qaqc_flags, 'fora_da_aoi')
            WHERE NOT (ST_X(geom) BETWEEN :x1 AND :x2
                   AND ST_Y(geom) BETWEEN :y1 AND :y2)
        """), {"x1": bbox[0], "x2": bbox[2], "y1": bbox[1], "y2": bbox[3]})
        contagem["2.1 fora_da_aoi (bloqueante)"] = r.rowcount

        # ---- 2.2 Duplicatas espaciais ----
        pares = conn.execute(text("""
            SELECT a.codigo_siagas, b.codigo_siagas
            FROM pocos a
            JOIN pocos b ON a.codigo_siagas < b.codigo_siagas
             AND ST_DWithin(a.geom::geography, b.geom::geography, :d)
            WHERE a.qaqc_aprovado AND b.qaqc_aprovado
        """), {"d": cfg["dist_duplicata_m"]}).all()
        if pares:
            escores = completude(conn)
            # agrupa em componentes simples via união de pares
            grupos = {}
            for a, b in pares:
                g = grupos.get(a) or grupos.get(b) or {a, b}
                g |= {a, b}
                for m in g:
                    grupos[m] = g
            vistos = set()
            for g in grupos.values():
                chave = tuple(sorted(g))
                if chave in vistos:
                    continue
                vistos.add(chave)
                vencedor = max(g, key=lambda c: (escores.get(c, 0), c))
                for perdedor in g - {vencedor}:
                    conn.execute(text("""
                        UPDATE pocos SET qaqc_aprovado=false,
                          qaqc_flags = array_append(qaqc_flags, :f)
                        WHERE codigo_siagas=:c
                    """), {"c": perdedor, "f": f"duplicata_de:{vencedor}"})
                    contagem["2.2 duplicata (bloqueante)"] += 1

        # ---- 2.5 Consistência construtiva (flags) ----
        r = conn.execute(text("""
            UPDATE pocos p SET
              qaqc_flags = array_append(qaqc_flags, 'filtro_abaixo_fundo')
            WHERE p.profundidade_m IS NOT NULL AND EXISTS (
              SELECT 1 FROM filtros f WHERE f.codigo_siagas=p.codigo_siagas
               AND f.ate_m > p.profundidade_m + :tol)
        """), {"tol": cfg["tolerancia_filtro_fundo_m"]})
        contagem["2.5 filtro_abaixo_fundo (flag)"] = r.rowcount

        r = conn.execute(text("""
            UPDATE pocos p SET
              qaqc_flags = array_append(qaqc_flags, 'filtros_sobrepostos')
            WHERE EXISTS (
              SELECT 1 FROM filtros f1
              JOIN filtros f2 ON f1.codigo_siagas=f2.codigo_siagas
               AND f1.ctid < f2.ctid
               AND f1.de_m < f2.ate_m AND f2.de_m < f1.ate_m
              WHERE f1.codigo_siagas=p.codigo_siagas)
        """))
        contagem["2.5 filtros_sobrepostos (flag)"] = r.rowcount

        # ---- 2.6 Lacunas litológicas (flag) ----
        lacunas = conn.execute(text("""
            SELECT DISTINCT codigo_siagas FROM (
              SELECT codigo_siagas,
                     de_m - lag(ate_m) OVER
                       (PARTITION BY codigo_siagas ORDER BY de_m) AS gap
              FROM litologia) s
            WHERE gap > :g
        """), {"g": cfg["lacuna_litologia_m"]}).all()
        for (cod,) in lacunas:
            conn.execute(text("""
                UPDATE pocos SET
                  qaqc_flags = array_append(qaqc_flags, 'lacuna_litologia')
                WHERE codigo_siagas=:c"""), {"c": cod})
        contagem["2.6 lacuna_litologia (flag)"] = len(lacunas)

        # ---- Flags manuais de conferência ----
        for cod, flags in (cfg.get("flags_manuais") or {}).items():
            for fl in flags:
                conn.execute(text("""
                    UPDATE pocos SET
                      qaqc_flags = array_append(qaqc_flags, :f)
                    WHERE codigo_siagas=:c"""), {"c": str(cod), "f": fl})
                contagem[f"manual {fl}"] += 1

        n_aprov = conn.execute(text(
            "SELECT count(*) FROM pocos WHERE qaqc_aprovado")).scalar()

    # ---- Relatório ----
    DERIVED.mkdir(parents=True, exist_ok=True)
    linhas = [{"regra": k, "pocos": v} for k, v in sorted(contagem.items())]
    linhas.append({"regra": "TOTAL pocos", "pocos": n_total})
    linhas.append({"regra": "TOTAL aprovados", "pocos": n_aprov})
    pd.DataFrame(linhas).to_csv(DERIVED / "qaqc_relatorio.csv", index=False)

    print("\n===== RELATÓRIO s02 (QA/QC) =====")
    for li in linhas:
        print(f"{li['pocos']:6d}  {li['regra']}")
    print("Relatório salvo em data/derived/qaqc_relatorio.csv")
    print("Pendentes de dados posteriores: 2.3/2.4/2.7 no s03; 2.8 no s05.")


if __name__ == "__main__":
    main()
