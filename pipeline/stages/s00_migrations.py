"""s00_migrations.py — Garante que o schema do banco está atualizado.

Execução OBRIGATÓRIA antes do s01 em qualquer instância nova ou após
atualização do pipeline. Idempotente: pode ser rodado quantas vezes
for necessário sem efeito colateral.

Execução:  docker compose run --rm pipeline python stages/s00_migrations.py
"""
import sys
from sqlalchemy import text
from _comum import engine

MIGRACOES = [
    # v1 → v2: coluna qs_cadastro na hidraulica_raw
    """ALTER TABLE hidraulica_raw
       ADD COLUMN IF NOT EXISTS qs_cadastro FLOAT""",

    # v2: tabelas auxiliares que podem estar ausentes em instâncias antigas
    """CREATE TABLE IF NOT EXISTS revestimento (
        id SERIAL PRIMARY KEY,
        codigo_siagas TEXT,
        de_m FLOAT,
        ate_m FLOAT,
        material TEXT,
        diametro_mm FLOAT
    )""",
    """CREATE TABLE IF NOT EXISTS entradas_dagua (
        id SERIAL PRIMARY KEY,
        codigo_siagas TEXT,
        de_m FLOAT,
        ate_m FLOAT,
        tipo TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS bomba (
        id SERIAL PRIMARY KEY,
        codigo_siagas TEXT,
        profundidade_m FLOAT,
        vazao_nominal_m3h FLOAT,
        potencia_cv FLOAT
    )""",
    """CREATE TABLE IF NOT EXISTS poco_uso (
        id SERIAL PRIMARY KEY,
        codigo_siagas TEXT,
        uso TEXT,
        situacao TEXT
    )""",
]


def main():
    print("===== s00 — MIGRAÇÕES DE SCHEMA =====")
    eng = engine()
    with eng.begin() as conn:
        for sql in MIGRACOES:
            try:
                conn.execute(text(sql))
                nome = sql.strip().split("\n")[0][:60]
                print(f"[OK] {nome}")
            except Exception as e:
                print(f"[SKIP] {str(e)[:80]}")
    print("Schema atualizado — pode executar o s01.")


if __name__ == "__main__":
    main()
