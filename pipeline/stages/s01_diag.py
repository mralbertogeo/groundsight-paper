"""s01_diag.py — Diagnóstico das seções da ficha SIAGAS (usa o cache).

Varre uma amostra das fichas em /data/raw/cache_siagas e reporta:
  1. Todos os nomes de seção encontrados e sua frequência;
  2. Para as seções não aproveitadas (outras_secoes), os rótulos das
     colunas — para identificarmos onde está o teste de bombeamento;
  3. Quantos poços têm revestimento registrado (para o intervalo aberto).

Execução:  docker compose run --rm pipeline python stages/s01_diag.py
"""
import random
from collections import Counter, defaultdict
from pathlib import Path

from siagas_ficha import parsear_html

CACHE = Path("/data/raw/cache_siagas")
AMOSTRA = 300  # fichas analisadas (aleatórias)

arquivos = sorted(CACHE.glob("*.html"))
random.seed(42)
amostra = random.sample(arquivos, min(AMOSTRA, len(arquivos)))

secoes = Counter()
rotulos_por_secao = defaultdict(Counter)
com_revestimento = 0
com_teste_dict = 0

for arq in amostra:
    html = arq.read_text(encoding="iso-8859-1")
    d = parsear_html(html)
    if d.get("revestimento"):
        com_revestimento += 1
    if d.get("teste_bombeamento"):
        com_teste_dict += 1
        secoes["<teste_bombeamento (aproveitada)>"] += 1
    for nome, registros in (d.get("outras_secoes") or {}).items():
        secoes[nome] += 1
        for reg in registros[:2]:
            for rotulo in reg.keys():
                rotulos_por_secao[nome][rotulo] += 1
    for chave in ("filtros", "litologia", "formacoes", "aquifero"):
        if d.get(chave):
            secoes[f"<{chave} (aproveitada)>"] += 1

print(f"Fichas analisadas: {len(amostra)} de {len(arquivos)} em cache\n")
print(f"Poços com revestimento na ficha: {com_revestimento}")
print(f"Poços com teste_bombeamento capturado pelo parser atual: {com_teste_dict}\n")

print("===== SEÇÕES ENCONTRADAS (frequência na amostra) =====")
for nome, n in secoes.most_common():
    print(f"{n:5d}  {nome}")

print("\n===== RÓTULOS DAS SEÇÕES NÃO APROVEITADAS =====")
for nome, rot in rotulos_por_secao.items():
    print(f"\n[{nome}]")
    for r, n in rot.most_common(12):
        print(f"   {n:4d}  {r}")
