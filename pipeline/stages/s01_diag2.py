"""s01_diag2.py — Imprime o conteúdo BRUTO das seções de teste de
bombeamento e aquífero de 10 fichas que possuem a seção, para
corrigirmos a extração vendo o dado exato.

Execução:  docker compose run --rm pipeline python stages/s01_diag2.py
"""
from pathlib import Path

from siagas_ficha import parsear_html

CACHE = Path("/data/raw/cache_siagas")

exibidos = 0
for arq in sorted(CACHE.glob("*.html")):
    html = arq.read_text(encoding="iso-8859-1")
    d = parsear_html(html)
    tb = d.get("teste_bombeamento") or {}
    if not tb:
        continue
    exibidos += 1
    print(f"\n========== POÇO {arq.stem} ==========")
    print("[teste_bombeamento] (chave -> valor):")
    for k, v in tb.items():
        print(f"   {k!r:45s} -> {v!r}")
    aq = d.get("aquifero") or {}
    if aq:
        print("[aquifero] (chave -> valor):")
        for k, v in aq.items():
            print(f"   {k!r:45s} -> {v!r}")
    if exibidos >= 10:
        break

print(f"\nFichas exibidas: {exibidos}")
