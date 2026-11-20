"""s05c_headers.py — Cabeçalhos COMPLETOS dos XYZ (sem truncar).

Imprime as primeiras 8 linhas de cada arquivo na íntegra, com as
colunas numeradas — insumo final para fixar as posições de canal no
s05c definitivo.

Execução:  docker compose run --rm pipeline python stages/s05c_headers.py
"""
from pathlib import Path

RAIZ = Path("/data/raw/geofisica")
SAIDA = Path("/data/outputs/s05c_headers.txt")

with open(SAIDA, "w", encoding="utf-8") as out:
    def p(msg=""):
        print(msg)
        out.write(msg + "\n")

    for xyz in sorted(RAIZ.rglob("*.xyz")) + sorted(RAIZ.rglob("*.XYZ")):
        p(f"\n########## {xyz.relative_to(RAIZ)} ##########")
        with open(xyz, "r", encoding="latin-1", errors="replace") as f:
            for i, linha in enumerate(f):
                if i >= 14:
                    break
                p(f"{i:3d}| {linha.rstrip()}")
                # numera os tokens das linhas de cabeçalho de canais
                if linha.startswith("/") and len(linha.split()) > 8:
                    tokens = linha.replace("/", " ").split()
                    p("     " + " | ".join(
                        f"[{k}]{t}" for k, t in enumerate(tokens)))
    p(f"\nSalvo em: {SAIDA}")
