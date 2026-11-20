"""s01_diag3.py — Duas verificações decisivas:

  A) Imprime o HTML CRU da tabela de teste de bombeamento de 3 fichas
     (estrutura real das células, sem interpretação);
  B) Inventaria o cadastro_aoi.parquet (Fase A): quais colunas
     hidráulicas existem e quantos poços têm valor preenchido —
     é o plano B para a variável-alvo Q/s.

Execução:  docker compose run --rm pipeline python stages/s01_diag3.py
"""
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

CACHE = Path("/data/raw/cache_siagas")
PARQUET = Path("/data/raw/cadastro_aoi.parquet")

# ---------- A) HTML cru da seção de teste ----------
print("################ A) HTML CRU — SEÇÃO DE TESTE ################")
exibidos = 0
for arq in sorted(CACHE.glob("*.html")):
    html = arq.read_text(encoding="iso-8859-1")
    if "bombeamento" not in html.lower():
        continue
    soup = BeautifulSoup(html, "html.parser")
    for tabela in soup.find_all("table"):
        header = tabela.find("td", class_="subsection-header")
        if header and "bombeamento" in header.get_text(strip=True).lower():
            bruto = tabela.prettify()
            print(f"\n========== POÇO {arq.stem} ==========")
            print(bruto[:3500])
            if len(bruto) > 3500:
                print(f"... [truncado — tabela tem {len(bruto)} caracteres]")
            exibidos += 1
            break
    if exibidos >= 3:
        break
if exibidos == 0:
    print("Nenhuma ficha com a palavra 'bombeamento' encontrada.")

# ---------- B) Inventário hidráulico do cadastro ----------
print("\n################ B) CADASTRO (Fase A) — CAMPOS HIDRÁULICOS ################")
if not PARQUET.exists():
    print(f"Arquivo não encontrado: {PARQUET}")
else:
    df = pd.read_parquet(PARQUET)
    print(f"Poços no parquet: {len(df)}")
    print(f"Total de colunas: {len(df.columns)}\n")
    padrao = re.compile(
        r"vaz|ne$|nd$|nivel|estatic|dinamic|espec|cota|profun|bombe|teste|aquif",
        re.IGNORECASE)
    achou = False
    for c in sorted(df.columns):
        if padrao.search(c):
            serie = pd.to_numeric(df[c], errors="coerce")
            n_num = int(serie.notna().sum())
            n_txt = int(df[c].notna().sum())
            exemplo = df[c].dropna().head(3).tolist()
            print(f"{c:35s} preenchidos: {n_txt:5d} | numéricos: {n_num:5d} | ex.: {exemplo}")
            achou = True
    if not achou:
        print("Nenhuma coluna com padrão hidráulico. Colunas disponíveis:")
        print(sorted(df.columns))
