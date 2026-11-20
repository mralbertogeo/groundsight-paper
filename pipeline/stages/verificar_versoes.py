"""verificar_versoes.py — Confirma quais versões dos estágios estão
instaladas ANTES de executar (elimina ambiguidade de cópia).

Execução:  docker compose run --rm pipeline python stages/verificar_versoes.py
"""
from pathlib import Path

CHECAGENS = [
    ("s05c_geofisica.py", "resgatadas via X/Y",
     "COM FALLBACK X/Y (atual)", "SEM fallback (ANTIGO)"),
    ("s06_treino.py", "geof_",
     "v3 com geofísica (atual)", "sem geofísica (ANTIGO)"),
    ("s07_validacao.py", "Barra bloqueante",
     "BARRA RECALIBRADA (atual)", "barra fixa 0,70 (ANTIGO)"),
    ("s08b_mapa_exploratorio.py", "VALIDADO DENTRO DO TETO",
     "mapa PROMOVIDO (atual)", "mapa exploratório (ANTIGO)"),
]
CONFIG = ("../config/config.yaml", "teto_vizinho_auc",
          "com pré-registro do teto (atual)", "sem teto (ANTIGO)")

base = Path(__file__).parent
tudo_ok = True
for arquivo, marca, ok, ruim in CHECAGENS:
    p = base / arquivo
    if not p.exists():
        print(f"[AUSENTE ] {arquivo}")
        tudo_ok = False
        continue
    atual = marca in p.read_text(encoding="utf-8", errors="replace")
    print(f"[{'OK' if atual else 'TROCAR'}{' ' if atual else ''}] "
          f"{arquivo}: {ok if atual else ruim}")
    tudo_ok &= atual

p = base / CONFIG[0]
atual = p.exists() and CONFIG[1] in p.read_text(encoding="utf-8",
                                                errors="replace")
print(f"[{'OK' if atual else 'TROCAR'}{' ' if atual else ''}] "
      f"config.yaml: {CONFIG[2] if atual else CONFIG[3]}")
tudo_ok &= atual

print("\n" + ("TUDO ATUAL — pode executar a sequência."
              if tudo_ok else
              "HÁ ARQUIVOS ANTIGOS — extraia o zip consolidado NA RAIZ "
              "do projeto (C:\\HydroAI\\groundsight), aceitando "
              "substituir, e rode esta verificação de novo."))
